"""
Integration test for the Halo Weave v0 brightness culling system.

This test is adapted from the verify_compaction.py script but focuses on testing
the deterministic culling mechanism with brightness-based token selection.
"""

import os
import json
import torch
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from positronic_brain import config
from positronic_brain.kv_mirror import KVMirror
from positronic_brain.model_io import load_model
from positronic_brain.brightness_engine import update_brightness_scores
from positronic_brain.culler import select_tokens_for_cull, culling_task

# --- Test Event Classes ---

@dataclass
class TokenGenerationEvent:
    """Event representing a token being generated."""
    step: int
    token_id: int
    token_text: str
    position: int
    token_brightness: Optional[float] = None
    source: str = "llm"

@dataclass
class TokenCullingEvent:
    """Event representing a token being culled."""
    step: int
    token_id: int
    token_text: str
    position: int
    token_brightness: float
    reason: str = "lowest_brightness"

@dataclass
class BrightnessUpdateEvent:
    """Event representing a token's brightness being updated."""
    step: int
    position: int
    token_id: int
    token_text: str
    brightness_before: float
    brightness_after: float
    attention_score: float

# --- Test Runner ---

class CullingIntegrationTest:
    """Integration test for the brightness culling system."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the test with the provided arguments."""
        self.args = args
        self.log_file = args.log_file
        self.json_log = args.json_log
        
        # Setup logging
        self.setup_logging()
        
        # State tracking
        self.events = []
        self.generated_tokens = []
        self.initial_context_text = ""
        self.initial_context_length = 0
        
        # Tracked token positions for brightness monitoring
        self.tracked_positions = set()  # Will be populated in setup()
        
        # Test components (initialized in setup)
        self.model = None
        self.processor = None
        self.kv_mirror_manager = None
        self.input_ids = None
        self.attention_mask = None
        
    def setup_logging(self):
        """Setup logging to both console and file."""
        # Create logger
        self.logger = logging.getLogger("culling_test")
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
    
    def log(self, message: str):
        """Log a message to both console and file."""
        self.logger.info(message)
    
    async def setup(self):
        """Initialize model, processor, and KV Mirror."""
        self.log(f"Setting up culling integration test...")
        
        # Initialize KV Mirror
        self.kv_mirror_manager = KVMirror()
        
        # Load model and processor
        self.log(f"Loading model {config.MODEL_NAME}...")
        self.model, self.processor = load_model(config.MODEL_NAME, config.TRUST_REMOTE_CODE)
        
        # Load initial context
        if self.args.context_file and os.path.exists(self.args.context_file):
            self.log(f"Loading initial context from {self.args.context_file}")
            with open(self.args.context_file, 'r', encoding='utf-8') as f:
                self.initial_context_text = f.read()
        else:
            self.log(f"Using default context (short text)")
            self.initial_context_text = "This is a test of the Halo Weave brightness culling system."
        
        # Tokenize the initial context
        tokenized = self.processor(self.initial_context_text, return_tensors="pt")
        self.input_ids = tokenized["input_ids"].to(self.model.device)
        self.attention_mask = tokenized["attention_mask"].to(self.model.device)
        
        self.initial_context_length = self.input_ids.shape[1]
        
        # Register initial tokens in KV Mirror
        initial_brightness_values = []
        for i in range(self.initial_context_length):
            token_id = self.input_ids[0, i].item()
            token_text = self.processor.decode([token_id])
            self.kv_mirror_manager.add(
                token_id=token_id,
                position=i,
                source="system_init",
                brightness=None  # Will use source-based seeding from config
            )
            # Get the actual brightness assigned
            brightness = self.kv_mirror_manager.get_token_brightness(i)
            initial_brightness_values.append(brightness)
            
        # Set up tracked positions for brightness monitoring
        # Track the first token, a middle token, the last initial token, and later new tokens
        self.tracked_positions = {0}  # Start with position 0
        if self.initial_context_length > 10:  # Add a middle position if we have enough tokens
            middle_pos = self.initial_context_length // 2
            self.tracked_positions.add(middle_pos)
        if self.initial_context_length > 1:  # Add the last initial token if we have more than one
            self.tracked_positions.add(self.initial_context_length - 1)
            
        # Log initial brightness statistics
        if initial_brightness_values:
            min_brightness = min(initial_brightness_values)
            max_brightness = max(initial_brightness_values)
            avg_brightness = sum(initial_brightness_values) / len(initial_brightness_values)
            self.log(f"Initialized with {self.initial_context_length} tokens of context")
            self.log(f"Initial brightness range: Min={min_brightness:.2f}, Max={max_brightness:.2f}, Avg={avg_brightness:.2f}")
    
    async def execute_forward_pass(self):
        """Execute a forward pass of the model."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                output_attentions=True,
                return_dict=True
            )
            return outputs
    
    def get_tokens_with_brightness(self, positions):
        """Get token info including brightness for the specified positions."""
        result = []
        for position in positions:
            token_id = self.input_ids[0, position].item() if position < self.input_ids.shape[1] else None
            token_text = self.processor.decode([token_id]) if token_id is not None else None
            brightness = self.kv_mirror_manager.get_token_brightness(position)
            result.append({
                'position': position,
                'token_id': token_id,
                'token_text': token_text,
                'brightness': brightness
            })
        return result
    
    async def update_brightness_from_attentions(self, outputs, generation_step):
        """Update brightness scores based on attentions and track changes for monitored positions."""
        # Capture brightness before update for tracked positions
        pre_update_brightness = {}
        for position in self.tracked_positions:
            brightness = self.kv_mirror_manager.get_token_brightness(position)
            pre_update_brightness[position] = brightness
            
        # Get attention scores for tracked positions (if available)
        attention_scores = {}
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions[-1]  # Get last layer's attention
            latest_token_attention = attentions.mean(dim=1)[:, -1, :]
            for position in self.tracked_positions:
                if position < latest_token_attention.shape[1]:
                    attention_scores[position] = latest_token_attention[0, position].item()
        
        # Update brightness scores
        update_result = update_brightness_scores(
            kv_mirror_manager=self.kv_mirror_manager,
            outputs=outputs,
            generation_step=generation_step,
            decay_per_tick=config.BRIGHTNESS_DECAY_PER_TICK,
            gain_coefficient=config.BRIGHTNESS_GAIN_COEFFICIENT
        )
        
        # Capture brightness after update for tracked positions
        for position in self.tracked_positions:
            # Skip positions that may have been culled
            if position not in pre_update_brightness:
                continue
                
            post_brightness = self.kv_mirror_manager.get_token_brightness(position)
            if post_brightness is not None and pre_update_brightness[position] is not None:
                token_id = self.input_ids[0, position].item() if position < self.input_ids.shape[1] else None
                if token_id is not None:
                    token_text = self.processor.decode([token_id])
                    # Create brightness update event
                    update_event = BrightnessUpdateEvent(
                        step=generation_step,
                        position=position,
                        token_id=token_id,
                        token_text=token_text,
                        brightness_before=pre_update_brightness[position],
                        brightness_after=post_brightness,
                        attention_score=attention_scores.get(position, 0.0)
                    )
                    self.events.append(update_event)
                    
                    # Log significant changes
                    delta = post_brightness - pre_update_brightness[position]
                    if abs(delta) > 5.0:  # Only log significant changes to reduce noise
                        self.log(f"[Step {generation_step}] Brightness for pos {position} changed: "
                                f"{pre_update_brightness[position]:.2f} → {post_brightness:.2f} (Δ{delta:+.2f})")
        
        return update_result
    
    async def run_test(self):
        """Run the integration test."""
        self.log(f"Starting culling integration test with {self.args.num_steps} steps")
        
        for step in range(1, self.args.num_steps + 1):
            # 1. Execute forward pass to get next token prediction
            outputs = await self.execute_forward_pass()
            
            # 2. Get the predicted token distribution
            # Shape: [batch_size, sequence_length, vocab_size]
            logits = outputs.logits
            
            # Get the last token's prediction (shape: [batch_size, vocab_size])
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature and get probabilities
            next_token_logits = next_token_logits / 0.8
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # 3. Select next token (simple greedy decoding for test)
            selected_token_id = torch.argmax(next_token_probs, dim=-1).item()
            
            # 4. Update the input sequence with the new token
            new_token_tensor = torch.tensor([[selected_token_id]], device=self.input_ids.device)
            self.input_ids = torch.cat([self.input_ids, new_token_tensor], dim=1)
            
            # 5. Extend attention mask
            new_mask = torch.ones((1, 1), device=self.attention_mask.device, dtype=self.attention_mask.dtype)
            self.attention_mask = torch.cat([self.attention_mask, new_mask], dim=1)
            
            # 6. Add new token to KV Mirror
            new_token_position = self.input_ids.shape[1] - 1
            self.kv_mirror_manager.add(
                token_id=selected_token_id,
                position=new_token_position,
                source="llm"  # Generated by model
            )
            
            # 7. Store generated token
            token_text = self.processor.decode([selected_token_id])
            self.generated_tokens.append(token_text)
            
            # Add this position to tracked positions (replace oldest if we're tracking too many)
            self.tracked_positions.add(new_token_position)
            if len(self.tracked_positions) > 5:  # Limit tracked positions to avoid too much logging
                # Remove oldest non-initial position
                positions_list = sorted(list(self.tracked_positions))
                for pos in positions_list:
                    if pos >= self.initial_context_length and pos != new_token_position:
                        self.tracked_positions.remove(pos)
                        break
            
            # 8. Update brightness scores based on attention
            brightness_result = await self.update_brightness_from_attentions(outputs, step)
            
            # 9. Record token generation event
            token_brightness = self.kv_mirror_manager.get_token_brightness(new_token_position)
            gen_event = TokenGenerationEvent(
                step=step,
                token_id=selected_token_id,
                token_text=token_text,
                position=new_token_position,
                token_brightness=token_brightness,
                source="llm"
            )
            self.events.append(gen_event)
            
            # Log the initial brightness of the generated token
            self.log(f"[Step {step}] Generated token '{token_text}' at pos {new_token_position} with initial brightness {token_brightness:.2f}")
            
            # 10. Run culling cycle on every generation step
            # Only log detailed info every 10 steps to reduce output
            verbose_logging = (step % 10 == 0)
            if True:  # Run on every step
                if verbose_logging:
                    self.log(f"[Step {step}] Running culling cycle...")
                
                # Get current mirror state
                snapshot = self.kv_mirror_manager.snapshot()
                kv_mirror = snapshot['kv_mirror']   # Dictionary {position: instance_id}
                active_tokens = snapshot['tokens']  # Dictionary {instance_id: ContextToken}
                current_size = len(kv_mirror)
                
                if verbose_logging:
                    self.log(f"[Step {step}] Current context size: {current_size}, Target: {config.CONTEXT_WINDOW_TARGET}")
                
                # Get positions to cull (manual culling based on brightness)
                positions_to_cull = select_tokens_for_cull(
                    self.kv_mirror_manager, 
                    config.CONTEXT_WINDOW_TARGET
                )
                
                if positions_to_cull:
                    # Create brightness map to analyze culling decisions
                    brightness_map = {}
                    for position, instance_id in kv_mirror.items():
                        if instance_id in active_tokens:
                            token_info = active_tokens[instance_id]
                            brightness_map[position] = token_info.brightness
                    
                    # Get the positions and brightnesses sorted by brightness
                    sorted_positions = sorted(brightness_map.keys(), key=lambda pos: brightness_map[pos])
                    
                    # Detailed logging only on every 10th step
                    if verbose_logging:
                        # Log culling decision verification
                        culled_brightnesses = [brightness_map[pos] for pos in positions_to_cull]
                        min_culled = min(culled_brightnesses) if culled_brightnesses else None
                        max_culled = max(culled_brightnesses) if culled_brightnesses else None
                        
                        self.log(f"[Step {step}] Positions selected for culling: {positions_to_cull}")
                        self.log(f"[Step {step}] Culled token brightness range: {min_culled:.2f} - {max_culled:.2f}")
                        
                        # Get a few non-culled tokens with the next lowest brightness values for verification
                        potential_next_tokens = [pos for pos in sorted_positions if pos not in positions_to_cull][:3]
                        if potential_next_tokens:
                            next_brightnesses = [brightness_map[pos] for pos in potential_next_tokens]
                            min_next = min(next_brightnesses) if next_brightnesses else None
                            tokens_info = self.get_tokens_with_brightness(potential_next_tokens)
                            self.log(f"[Step {step}] Next lowest tokens NOT culled:")
                            for token in tokens_info:
                                self.log(f"  Pos {token['position']}: '{token['token_text']}' - Brightness: {token['brightness']:.2f}")
                            
                            # Verify that all culled tokens had lower brightness than non-culled
                            if min_next is not None and max_culled is not None:
                                if max_culled > min_next:
                                    self.log(f"WARNING: Culling decision inconsistency detected! Max culled brightness ({max_culled:.2f}) > min non-culled brightness ({min_next:.2f})")
                    
                    # Record culling events before removing tokens
                    for position in positions_to_cull:
                        # Get token info before culling
                        token_id = self.input_ids[0, position].item()
                        token_text = self.processor.decode([token_id])
                        token_brightness = self.kv_mirror_manager.get_token_brightness(position)
                        
                        # Record culling event
                        cull_event = TokenCullingEvent(
                            step=step,
                            token_id=token_id,
                            token_text=token_text,
                            position=position,
                            token_brightness=token_brightness,
                            reason="lowest_brightness"
                        )
                        self.events.append(cull_event)
                        
                        # Remove culled position from tracked positions
                        if position in self.tracked_positions:
                            self.tracked_positions.remove(position)
                    
                    # Cull the tokens
                    self.kv_mirror_manager.prune(positions_to_cull)
                    if verbose_logging:
                        self.log(f"[Step {step}] Culled {len(positions_to_cull)} tokens")
                    elif len(positions_to_cull) > 0:
                        # Minimal logging for non-verbose steps that actually culled something
                        self.log(f"[Step {step}] Culled {len(positions_to_cull)} tokens")
                    
                    # Verify positions were actually culled
                    new_snapshot = self.kv_mirror_manager.snapshot()
                    new_kv_mirror = new_snapshot['kv_mirror']
                    for pos in positions_to_cull:
                        if pos in new_kv_mirror:
                            self.log(f"WARNING: Position {pos} still present in KV mirror after culling!")
            
            # Progress update
            if step % 10 == 0:
                self.log(f"[Progress] {step}/{self.args.num_steps} tokens generated")
                # Report on current context size
                current_size = len(self.kv_mirror_manager.snapshot()['kv_mirror'])
                self.log(f"[Context Size] {current_size} active tokens")
        
        # Generate final report
        await self.generate_report()
    
    async def generate_report(self):
        """Generate a report of the test results."""
        self.log("\n--- Culling Integration Test Report ---")
        
        # Context size stats
        snapshot = self.kv_mirror_manager.snapshot()
        current_size = len(snapshot['kv_mirror'])
        active_tokens = snapshot['tokens']
        
        self.log(f"Final context size: {current_size} tokens (target: {config.CONTEXT_WINDOW_TARGET})")
        
        # Verify convergence to target size if we ran enough steps
        if self.initial_context_length >= config.CONTEXT_WINDOW_TARGET:
            if current_size > config.CONTEXT_WINDOW_TARGET:
                self.log(f"WARNING: Final context size {current_size} exceeds target {config.CONTEXT_WINDOW_TARGET}")
            elif self.args.num_steps >= 20:  # Only assert for longer runs that had time to converge
                try:
                    assert current_size <= config.CONTEXT_WINDOW_TARGET, f"Final size {current_size} > target {config.CONTEXT_WINDOW_TARGET}"
                    self.log(f"✓ Size convergence test PASSED: {current_size} <= {config.CONTEXT_WINDOW_TARGET}")
                except AssertionError as e:
                    self.log(f"❌ Size convergence test FAILED: {str(e)}")
        
        # Brightness stats
        try:
            brightness_values = []
            for position, instance_id in snapshot['kv_mirror'].items():
                token = active_tokens.get(instance_id)
                if token and hasattr(token, 'brightness'):
                    brightness_values.append(token.brightness)
            
            if brightness_values:
                brightness_values.sort()
                min_brightness = min(brightness_values)
                max_brightness = max(brightness_values)
                avg_brightness = sum(brightness_values) / len(brightness_values)
                
                # Median calculation
                if len(brightness_values) % 2 == 0:
                    median_brightness = (brightness_values[len(brightness_values)//2 - 1] + 
                                         brightness_values[len(brightness_values)//2]) / 2
                else:
                    median_brightness = brightness_values[len(brightness_values)//2]
                
                self.log(f"Brightness stats for {len(brightness_values)} active tokens:")
                self.log(f"  Min: {min_brightness:.2f}")
                self.log(f"  Max: {max_brightness:.2f}")
                self.log(f"  Avg: {avg_brightness:.2f}")
                self.log(f"  Median: {median_brightness:.2f}")
        except Exception as e:
            self.log(f"Error getting KV Mirror stats: {e}")
        
        # Count event types
        gen_events = [e for e in self.events if isinstance(e, TokenGenerationEvent)]
        cull_events = [e for e in self.events if isinstance(e, TokenCullingEvent)]
        
        self.log(f"Event counts:")
        self.log(f"  Token generations: {len(gen_events)}")
        self.log(f"  Token cullings: {len(cull_events)}")
        
        # Generated text
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
        
        # Save events to JSON log
        if self.json_log:
            self.save_events_json()
    
    def save_events_json(self):
        """Save events to a JSON file."""
        try:
            events_json = []
            for event in self.events:
                if isinstance(event, TokenGenerationEvent):
                    events_json.append({
                        "type": "generation",
                        **asdict(event)
                    })
                elif isinstance(event, TokenCullingEvent):
                    events_json.append({
                        "type": "culling",
                        **asdict(event)
                    })
                elif isinstance(event, BrightnessUpdateEvent):
                    events_json.append({
                        "type": "brightness_update",
                        **asdict(event)
                    })
            
            with open(self.json_log, 'w', encoding='utf-8') as f:
                json.dump(events_json, f, indent=2)
            
            self.log(f"Saved {len(events_json)} events to {self.json_log}")
        except Exception as e:
            self.log(f"Error saving events to JSON: {e}")


# --- Command Line Interface ---

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a culling integration test")
    parser.add_argument("--context_file", type=str, help="Path to initial context file")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of generation steps to run")
    parser.add_argument("--log_file", type=str, default="culling_test.log", help="Log file path")
    parser.add_argument("--json_log", type=str, default="culling_events.json", help="JSON log file path")
    return parser.parse_args()


# --- Main Test Runner ---

async def main():
    """Main entry point."""
    args = parse_args()
    
    test = CullingIntegrationTest(args)
    await test.setup()
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())
