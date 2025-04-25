#!/usr/bin/env python3
"""
Verification Script for Halo Weave System (Brightness, Compactor, KV Patcher)

This script runs a controlled test of the Halo Weave token generation and compaction system.
It loads a large initial context, generates a fixed number of tokens, and logs detailed
information about the process, including brightness scores, compactor actions, and token changes.
"""

import os
import sys
import time
import argparse
import asyncio
import json
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# Import positronic_brain modules
from positronic_brain import config
from positronic_brain.kv_mirror import KVMirror
from positronic_brain.model_io import load_model, execute_forward_pass
from positronic_brain.kv_patcher import KVCachePatcher
from positronic_brain.brightness_engine import update_brightness_scores
from positronic_brain.sampler import SamplerState, select_next_token
from positronic_brain.compactor import compactor_task, compute_diff
from positronic_brain.diffuser_runner import DiffuserModel

# Event types for logging
@dataclass
class TokenGenerationEvent:
    step: int
    token_id: int
    token_text: str
    position: int
    token_brightness: Optional[float] = None

@dataclass
class TokenRepairEvent:
    step: int
    position: int
    old_token_id: int
    new_token_id: int
    old_token_text: str
    new_token_text: str
    brightness_before: Optional[float] = None
    brightness_after: Optional[float] = None

@dataclass
class TokenCullingEvent:
    step: int
    position: int
    token_id: int
    token_text: str
    token_brightness: float

class CompactionVerifier:
    def __init__(self, args):
        self.args = args
        self.events = []
        self.log_file = args.log_file
        self.model_name = args.model_name or config.MODEL_NAME
        self.trust_remote_code = config.TRUST_REMOTE_CODE
        self.device = config.GPU_DEVICE
        
        # Components that will be initialized
        self.model = None
        self.processor = None
        self.kv_mirror_manager = None
        self.kv_patcher = None
        self.diffuser = None
        self.pending_diffs_queue = None
        self.shutdown_event = None
        
        # State variables
        self.input_ids = None
        self.attention_mask = None
        self.past_key_values = None
        self.initial_context_text = ""
        self.initial_context_length = 0
        self.generated_tokens = []
        
        # Create log file directory if it doesn't exist and if there is a directory component
        if self.log_file and os.path.dirname(self.log_file):
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
        # Set up logging
        self.log(f"=== Compaction Verification Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        self.log(f"Model: {self.model_name}")
        self.log(f"Context file: {args.context_file}")
        self.log(f"Num steps: {args.num_steps}")
        
    def log(self, message):
        """Log a message to both console and log file."""
        print(message)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{message}\n")
                
    async def initialize(self):
        """Initialize all components needed for the test."""
        self.log("Initializing components...")
        
        # Load model and processor/tokenizer
        self.log(f"Loading model: {self.model_name}...")
        self.model, self.processor = load_model(self.model_name, self.trust_remote_code)
        self.model.to(self.device)
        self.model.eval()
        
        # For this test, we need to re-enable output_attentions
        # Patch execute_forward_pass to use output_attentions=True
        from positronic_brain.model_io import execute_forward_pass as original_execute_forward_pass
        
        # Create a wrapper with output_attentions=True
        def patched_execute_forward_pass(*args, **kwargs):
            # Force output_attentions=True for this verification script
            kwargs['output_attentions'] = True
            return original_execute_forward_pass(*args, **kwargs)
        
        # Replace the imported function with our patched version
        import positronic_brain.model_io
        positronic_brain.model_io.execute_forward_pass = patched_execute_forward_pass
        
        # Initialize KV Mirror
        self.log("Initializing KV Mirror...")
        self.kv_mirror_manager = KVMirror()
        
        # Initialize Patcher
        self.log("Initializing KV Patcher...")
        self.kv_patcher = KVCachePatcher(self.model)
        
        # Initialize queues
        self.pending_diffs_queue = asyncio.Queue(maxsize=config.COMPACTOR_BUFFER_SIZE)
        self.shutdown_event = asyncio.Event()
        
        # Load and initialize the initial context
        await self.load_initial_context()
        
        self.log("Initialization complete.")
        
    async def load_initial_context(self):
        """Load the initial context from file and initialize the system state."""
        context_file = self.args.context_file
        if not os.path.exists(context_file):
            self.log(f"Error: Context file {context_file} does not exist.")
            raise FileNotFoundError(f"Context file {context_file} not found")
            
        # Read the file
        with open(context_file, "r", encoding="utf-8") as f:
            self.initial_context_text = f.read()
            
        self.log(f"Loaded initial context: {len(self.initial_context_text)} characters")
        
        # Tokenize the context
        encoded = self.processor(self.initial_context_text, 
                               return_tensors="pt", 
                               padding=False, 
                               truncation=False)
        
        # Make sure to use the model's device consistently
        # Get the model's actual device (which might be different from self.device due to accelerate hooks)
        actual_device = next(self.model.parameters()).device
        self.log(f"Using model device: {actual_device}")
        
        self.input_ids = encoded["input_ids"].to(actual_device)
        self.attention_mask = encoded["attention_mask"].to(actual_device)
        self.initial_context_length = self.input_ids.shape[1]
        
        self.log(f"Initial context tokenized: {self.initial_context_length} tokens")
        
        # Register all initial tokens with the KV Mirror
        self.log("Registering initial tokens with KV Mirror...")
        self.kv_mirror_manager.clear()
        for i in range(self.initial_context_length):
            token_id = self.input_ids[0, i].item()
            self.kv_mirror_manager.add(
                position=i,
                token_id=token_id,
                source="system_init",
                brightness=config.INITIAL_TOKEN_BRIGHTNESS
            )
            
        # Prime the KV cache with the initial context
        self.log("Priming KV cache with initial context...")
        with torch.no_grad():
            _, initial_past_key_values, _ = execute_forward_pass(
                model=self.model,
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                past_key_values=None,  # Initial priming pass
                position_ids=None  # Let model calculate
            )
            
        self.past_key_values = initial_past_key_values
        self.log(f"KV cache initialized with {self.initial_context_length} tokens")
        
        # Log mirror stats
        stats = self.kv_mirror_manager.get_stats()
        self.log(f"KV Mirror stats: {stats}")
    
    async def run_test(self):
        """Run the compaction verification test."""
        self.log(f"Starting compaction test for {self.args.num_steps} steps...")
        
        # Load diffuser model directly for verification
        self.log("Loading diffuser model...")
        self.diffuser = DiffuserModel(model_name=config.DIFFUSER_MODEL_NAME, device=self.device)
        self.log("Diffuser model loaded for testing")
        
        # Create a request queue for the compactor
        compactor_request_queue = asyncio.Queue()
        
        # Start the compactor task with the correct parameters
        compactor = asyncio.create_task(
            compactor_task(
                kv_mirror_manager=self.kv_mirror_manager,
                diffuser_model=self.diffuser,
                pending_diffs_queue=self.pending_diffs_queue,
                compactor_request_queue=compactor_request_queue,
                shutdown_event=self.shutdown_event
            )
        )
        
        self.log("Compactor task started.")
        
        # Prepare sampler state - use default values from the SamplerState class in config.py
        from positronic_brain.config import SamplerState
        sampler_state = SamplerState(
            temperature=0.6,  # Default from SamplerState class
            top_k=50,         # Default from SamplerState class
            top_p=1.0,        # Default from SamplerState class
            repetition_penalty=1.1,  # Default from SamplerState class
            # Additional parameters not needed
        )
        
        # Create shared state dict for sampler
        shared_state = {"token_frequency": {}}
        
        # Define stop tokens (usually just EOS)
        stop_tokens = {self.processor.eos_token_id} if hasattr(self.processor, 'eos_token_id') else set()
        
        # Start generating tokens
        step = 0
        
        try:
            # Main generation loop
            for step in range(self.args.num_steps):
                # 1. Check and apply any pending diffs from Compactor
                diffs_to_apply = []
                try:
                    while not self.pending_diffs_queue.empty():
                        diff_item = await self.pending_diffs_queue.get()
                        diffs_to_apply.append(diff_item)
                        self.pending_diffs_queue.task_done()
                        
                    # Apply diffs if any
                    if diffs_to_apply:
                        self.log(f"[Step {step}] Applying {len(diffs_to_apply)} diffs from Compactor")
                        
                        # Apply to KV Mirror
                        update_summary = self.kv_mirror_manager.apply_diff(diffs_to_apply)
                        self.log(f"[Step {step}] KV Mirror diff application: {update_summary}")
                        
                        # Apply to KV Cache
                        self.past_key_values = self.kv_patcher.patch(self.past_key_values, diffs_to_apply)
                        self.log(f"[Step {step}] KV Cache patched successfully")
                        
                        # Record repair events
                        for diff in diffs_to_apply:
                            position = diff.get('pos')
                            old_token_id = diff.get('old_id')
                            new_token_id = diff.get('new_id')
                            
                            # Get token texts
                            old_token_text = self.processor.decode([old_token_id])
                            new_token_text = self.processor.decode([new_token_id])
                            
                            # Get brightness if available
                            brightness_before = self.kv_mirror_manager.get_token_brightness(position)
                            
                            # Record the repair event
                            repair_event = TokenRepairEvent(
                                step=step,
                                position=position,
                                old_token_id=old_token_id,
                                new_token_id=new_token_id,
                                old_token_text=old_token_text,
                                new_token_text=new_token_text,
                                brightness_before=brightness_before,
                                brightness_after=None  # Will be updated after next forward pass
                            )
                            
                            self.events.append(repair_event)
                            
                except Exception as e:
                    self.log(f"[Step {step}] Error applying diffs: {e}")
                
                # 2. Prepare for next token generation
                model_input_ids = self.input_ids[:, -1:]  # Only the last token with KV cache
                
                # Calculate position_ids (critical for coherence with TinyLlama)
                cache_seq_len = self.past_key_values[0][0].shape[2]
                position_ids = torch.full(
                    (1, 1),
                    cache_seq_len,
                    device=model_input_ids.device,
                    dtype=torch.long
                )
                
                # Set attention_mask to None (O1's fix)
                current_attention_mask_for_call = None
                
                # 3. Forward pass to generate next token
                try:
                    with torch.no_grad():
                        logits, llm_output_past_key_values, outputs_attentions = execute_forward_pass(
                            model=self.model,
                            input_ids=model_input_ids.to(self.model.device),
                            attention_mask=current_attention_mask_for_call,
                            position_ids=position_ids,
                            past_key_values=self.past_key_values
                        )
                        
                    # Update past_key_values
                    self.past_key_values = llm_output_past_key_values
                    
                    # Prepare outputs object for brightness update
                    outputs = type('ModelOutputs', (), {})()
                    outputs.attentions = outputs_attentions
                    
                    # 4. Update brightness scores
                    brightness_scores = None
                    if not self.args.disable_brightness and outputs_attentions is not None:
                        brightness_scores = update_brightness_scores(
                            kv_mirror_manager=self.kv_mirror_manager,
                            outputs=outputs,
                            alpha=config.BRIGHTNESS_ALPHA,
                            beta=config.BRIGHTNESS_BETA
                        )
                        self.log(f"[Step {step}] Updated brightness scores")
                        
                        # Update brightness_after for any repair events from this step
                        for event in self.events:
                            if isinstance(event, TokenRepairEvent) and event.step == step:
                                event.brightness_after = self.kv_mirror_manager.get_token_brightness(event.position)
                    
                    # 5. Sample next token
                    selected_token_id, final_probs, _ = select_next_token(
                        logits=logits,
                        input_ids=self.input_ids,
                        sampler_state=sampler_state
                    )
                    
                    # Skip generation if it's a stop token
                    if selected_token_id in stop_tokens:
                        self.log(f"[Step {step}] Generated stop token {selected_token_id}, stopping generation")
                        break
                        
                    # 6. Update state
                    # Add the new token to input_ids
                    new_token_tensor = torch.tensor([[selected_token_id]], device=self.input_ids.device)
                    self.input_ids = torch.cat([self.input_ids, new_token_tensor], dim=1)
                    
                    # Extend attention_mask for tracking (even if we don't use it in forward pass)
                    new_mask = torch.ones((1, 1), device=self.attention_mask.device, dtype=self.attention_mask.dtype)
                    self.attention_mask = torch.cat([self.attention_mask, new_mask], dim=1)
                    
                    # Register new token with KV Mirror
                    new_token_position = self.input_ids.shape[1] - 1
                    self.kv_mirror_manager.add(
                        position=new_token_position,
                        token_id=selected_token_id,
                        source="generation",
                        brightness=config.INITIAL_TOKEN_BRIGHTNESS
                    )
                    
                    # Store generated token
                    token_text = self.processor.decode([selected_token_id])
                    self.generated_tokens.append(token_text)
                    
                    # 7. Log generation event
                    # Always try to retrieve the brightness directly from KVMirror, regardless of brightness calculation
                    token_brightness = self.kv_mirror_manager.get_token_brightness(new_token_position)
                    
                    gen_event = TokenGenerationEvent(
                        step=step,
                        token_id=selected_token_id,
                        token_text=token_text,
                        position=new_token_position,
                        token_brightness=token_brightness
                    )
                    
                    self.events.append(gen_event)
                    
                    # Progress update (every 10 steps)
                    if step % 10 == 0:
                        self.log(f"[Progress] {step}/{self.args.num_steps} tokens generated")
                    
                except Exception as e:
                    self.log(f"[Step {step}] Error in token generation: {e}")
                    raise
                    
        finally:
            # Stop the compactor
            self.log("Setting shutdown event...")
            self.shutdown_event.set()
            
            # Wait for compactor to finish
            self.log("Waiting for compactor to finish...")
            await compactor
            
            # Generate final report
            self.generate_report(steps_completed=step+1)
    
    def generate_report(self, steps_completed):
        """Generate and print a detailed report of the test results."""
        self.log("\n\n=== VERIFICATION TEST REPORT ===\n")
        
        # Basic stats
        initial_length = self.initial_context_length
        final_length = self.input_ids.shape[1]
        generated_length = final_length - initial_length
        
        self.log(f"Initial context length: {initial_length} tokens")
        self.log(f"Final context length: {final_length} tokens")
        self.log(f"Total tokens generated: {generated_length} tokens")
        self.log(f"Steps completed: {steps_completed}/{self.args.num_steps}")
        
        # KV Mirror stats
        try:
            stats = self.kv_mirror_manager.get_stats()
            self.log(f"Final KV Mirror stats: {stats}")
            
            # Get brightness stats for active tokens
            brightness_values = []
            for pos in range(final_length):
                try:
                    brightness = self.kv_mirror_manager.get_token_brightness(pos)
                    if brightness is not None:
                        brightness_values.append(brightness)
                except Exception:
                    pass
                
            if brightness_values:
                min_brightness = min(brightness_values)
                max_brightness = max(brightness_values)
                avg_brightness = sum(brightness_values) / len(brightness_values)
                median_brightness = sorted(brightness_values)[len(brightness_values) // 2]
                
                self.log(f"Brightness stats for {len(brightness_values)} active tokens:")
                self.log(f"  Min: {min_brightness:.2f}")
                self.log(f"  Max: {max_brightness:.2f}")
                self.log(f"  Avg: {avg_brightness:.2f}")
                self.log(f"  Median: {median_brightness:.2f}")
        except Exception as e:
            self.log(f"Error getting KV Mirror stats: {e}")
        
        # Count event types
        gen_events = [e for e in self.events if isinstance(e, TokenGenerationEvent)]
        repair_events = [e for e in self.events if isinstance(e, TokenRepairEvent)]
        cull_events = [e for e in self.events if isinstance(e, TokenCullingEvent)]
        
        self.log(f"Event counts:")
        self.log(f"  Token generations: {len(gen_events)}")
        self.log(f"  Token repairs: {len(repair_events)}")
        self.log(f"  Token cullings: {len(cull_events)}")
        
        # Final generated text - improve readability with proper spacing
        # Instead of just concatenating tokens, decode the entire sequence to get proper spacing
        generated_token_ids = [self.input_ids[0, i].item() for i in range(self.initial_context_length, self.input_ids.shape[1])]
        generated_text = self.processor.decode(generated_token_ids, skip_special_tokens=True)
        self.log(f"\nGenerated text ({len(generated_text)} chars):")
        self.log("```")
        self.log(generated_text)
        self.log("```")
        
        # Full initial + generated text (for coherence assessment)
        full_text = self.initial_context_text + generated_text
        self.log(f"\nFull text (last 1000 chars):")
        self.log("```")
        self.log(full_text[-1000:])
        self.log("```")
        
        # Chronological events log
        self.log("\nChronological event log (first 50 events):")
        events_sorted = sorted(self.events, key=lambda e: (e.step, 0 if isinstance(e, TokenGenerationEvent) else 1))
        
        for i, event in enumerate(events_sorted[:50]):
            if isinstance(event, TokenGenerationEvent):
                self.log(f"Step {event.step}: Added token '{event.token_text}' (ID: {event.token_id}) at position {event.position}")
                if event.token_brightness is not None:
                    self.log(f"  Brightness: {event.token_brightness:.2f}")
            
            elif isinstance(event, TokenRepairEvent):
                self.log(f"Step {event.step}: Repaired token at position {event.position}")
                self.log(f"  {event.old_token_text} (ID: {event.old_token_id}) -> {event.new_token_text} (ID: {event.new_token_id})")
                if event.brightness_before is not None:
                    self.log(f"  Brightness: {event.brightness_before:.2f} -> {event.brightness_after:.2f}")
            
            elif isinstance(event, TokenCullingEvent):
                self.log(f"Step {event.step}: Culled token '{event.token_text}' (ID: {event.token_id}) at position {event.position}")
                self.log(f"  Brightness: {event.token_brightness:.2f}")
                
        # Write detailed event log to file if requested
        if self.args.json_log:
            json_events = []
            for event in events_sorted:
                if isinstance(event, TokenGenerationEvent):
                    json_events.append({
                        "type": "generation",
                        **asdict(event)
                    })
                elif isinstance(event, TokenRepairEvent):
                    json_events.append({
                        "type": "repair",
                        **asdict(event)
                    })
                elif isinstance(event, TokenCullingEvent):
                    json_events.append({
                        "type": "culling",
                        **asdict(event)
                    })
                    
            with open(self.args.json_log, "w") as f:
                json.dump(json_events, f, indent=2)
                
            self.log(f"Detailed event log written to {self.args.json_log}")
        
        self.log("\n=== END OF REPORT ===")

async def main():
    parser = argparse.ArgumentParser(description="Verification script for Halo Weave compaction system")
    parser.add_argument("--context_file", default="resume_context.txt", help="Path to initial context file")
    parser.add_argument("--num_steps", type=int, default=300, help="Number of tokens to generate")
    parser.add_argument("--log_file", default="compaction_verify.log", help="Path to log file")
    parser.add_argument("--json_log", default="compaction_events.json", help="Path to detailed JSON event log")
    parser.add_argument("--model_name", default=None, help="Override model name from config")
    parser.add_argument("--disable_brightness", action="store_true", help="Disable brightness updates")
    
    args = parser.parse_args()
    
    verifier = CompactionVerifier(args)
    await verifier.initialize()
    await verifier.run_test()

if __name__ == "__main__":
    asyncio.run(main())
