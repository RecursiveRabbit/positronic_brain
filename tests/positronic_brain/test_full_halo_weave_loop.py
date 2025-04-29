"""
Integration test for the complete Halo Weave v1 system.

This test simulates a complete run of the Halo Weave v1 system with brightness updates,
culling, and repair mechanisms activated, providing a repeatable verification of the
entire system without requiring manual log inspection.
"""

import pytest
import pytest_asyncio
import asyncio
import torch
import random
import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Set

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from positronic_brain import config
from positronic_brain.kv_mirror import KVMirror, ContextToken
from positronic_brain.context_maintenance import ContextMaintenance
from positronic_brain.brightness_engine import update_brightness_scores
from positronic_brain.culler import select_tokens_for_cull
from positronic_brain.kv_patcher import KVCachePatcher
from positronic_brain.model_io import execute_forward_pass, load_model
from positronic_brain.sampler import select_next_token
from positronic_brain.sampler_types import SamplerState
from positronic_brain.diffuser_runner import DiffuserModel, compute_diff, load_diffuser_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# We'll use real models instead of mocks
# Configuration constants for testing
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model suitable for testing
DIFFUSER_MODEL_NAME = "distilbert-base-uncased"  # Small diffuser model for testing

# Global variable to store the original config value to restore after tests
_original_context_window_target = config.CONTEXT_WINDOW_TARGET
_original_brightness_repair_threshold = config.BRIGHTNESS_REPAIR_THRESHOLD
_original_brightness_decay_per_tick = config.BRIGHTNESS_DECAY_PER_TICK

# Mock metrics functions to avoid side effects in tests
@pytest.fixture(autouse=True)
def mock_metrics():
    """Mock metrics-related functions to avoid side effects."""
    # Use a try/except block since we're no longer using unittest.mock
    # This is a temporary stand-in - in a real production environment,
    # we would use a proper mocking framework or dependency injection
    original_timed_histogram = None
    original_set_gauge = None
    original_inc_counter = None
    original_inc_histogram = None
    
    # Save original metrics functions
    try:
        from positronic_brain.metrics import timed_histogram, set_gauge, inc_counter, inc_histogram
        original_timed_histogram = timed_histogram
        original_set_gauge = set_gauge
        original_inc_counter = inc_counter
        original_inc_histogram = inc_histogram
        
        # Replace with no-op functions
        def noop_decorator(name):
            def decorator(func):
                return func
            return decorator
            
        def noop(*args, **kwargs):
            pass
            
        # Apply the no-op replacements
        from positronic_brain import metrics
        metrics.timed_histogram = noop_decorator
        metrics.set_gauge = noop
        metrics.inc_counter = noop
        metrics.inc_histogram = noop
        
        # Also disable attention saving
        from positronic_brain.brightness_engine import _async_save_attention
        original_save_attention = _async_save_attention
        
        async def noop_save_attention(*args, **kwargs):
            return None
            
        import positronic_brain.brightness_engine
        positronic_brain.brightness_engine._async_save_attention = noop_save_attention
        
    except ImportError as e:
        print(f"Warning: Could not mock metrics functions: {e}")
    
    yield
    
    # Restore original functions
    try:
        if original_timed_histogram:
            from positronic_brain import metrics
            metrics.timed_histogram = original_timed_histogram
            metrics.set_gauge = original_set_gauge
            metrics.inc_counter = original_inc_counter
            metrics.inc_histogram = original_inc_histogram
            
        if original_save_attention:
            import positronic_brain.brightness_engine
            positronic_brain.brightness_engine._async_save_attention = original_save_attention
    except Exception as e:
        print(f"Warning: Could not restore original metrics functions: {e}")

# Helper function to prepare initial context for the test
def prepare_initial_context(tokenizer, model, num_tokens=500):
    """
    Create an initial context with sequential token IDs for testing.
    
    Args:
        tokenizer: The tokenizer to use
        model: The model, used to determine the target device
        num_tokens: The number of tokens to generate
        
    Returns:
        Tuple of (input_ids, attention_mask)
    """
    # Use a random prompt as the seed text
    target_device = model.device
    print(f"Preparing initial context on device: {target_device}")
    
    # Create a basic prompt with enough tokens
    prompt = "The quick brown fox jumps over the lazy dog. "
    # Repeat prompt until we have at least num_tokens tokens
    while len(tokenizer.encode(prompt)) < num_tokens:
        prompt += prompt
    
    # Tokenize and truncate to exact num_tokens size
    encoded = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = encoded[:, :num_tokens].to(target_device)
    
    # Create attention mask that exactly matches input_ids
    attention_mask = torch.ones_like(input_ids)
    
    return input_ids, attention_mask

# Event tracking class for the test
class EventTracker:
    """Track events during the Halo Weave run."""
    
    def __init__(self):
        self.events = []
        
    def add_token(self, token_id, position):
        """Record a token addition."""
        self.events.append({
            'type': 'add',
            'token_id': token_id,
            'position': position,
            'step': len(self.events)
        })
        
    def cull_token(self, position, token_id, brightness):
        """Record a token culling."""
        self.events.append({
            'type': 'cull',
            'position': position,
            'token_id': token_id,
            'brightness': brightness,
            'step': len(self.events)
        })
        
    def repair_token(self, position, old_token_id, new_token_id, old_brightness, new_brightness):
        """Record a token repair."""
        self.events.append({
            'type': 'repair',
            'position': position,
            'old_token_id': old_token_id,
            'new_token_id': new_token_id,
            'old_brightness': old_brightness,
            'new_brightness': new_brightness,
            'step': len(self.events)
        })
        
    def summary(self, first_n=10, last_n=10):
        """Generate a summary of the first_n and last_n events."""
        total_events = len(self.events)
        
        first_events = self.events[:first_n]
        last_events = self.events[-last_n:] if total_events > last_n else []
        
        # Count event types
        event_counts = {
            'add': sum(1 for e in self.events if e['type'] == 'add'),
            'cull': sum(1 for e in self.events if e['type'] == 'cull'),
            'repair': sum(1 for e in self.events if e['type'] == 'repair'),
        }
        
        # Format event descriptions
        def format_event(event):
            if event['type'] == 'add':
                return f"Step {event['step']}: Added token {event['token_id']} at position {event['position']}"
            elif event['type'] == 'cull':
                return f"Step {event['step']}: Culled token {event['token_id']} at position {event['position']} (brightness: {event['brightness']:.2f})"
            elif event['type'] == 'repair':
                return f"Step {event['step']}: Repaired token at position {event['position']}: {event['old_token_id']} -> {event['new_token_id']} (brightness: {event['old_brightness']:.2f} -> {event['new_brightness']:.2f})"
            return f"Unknown event: {event}"
        
        first_descriptions = [format_event(e) for e in first_events]
        last_descriptions = [format_event(e) for e in last_events]
        
        return {
            'total_events': total_events,
            'event_counts': event_counts,
            'first_events': first_descriptions,
            'last_events': last_descriptions
        }

# Fixture to setup the Halo Weave system with real components
@pytest_asyncio.fixture
async def initialized_halo_weave_system():
    """
    Initialize a complete Halo Weave system with real components for integration testing.
    
    This fixture:
    1. Loads the real TinyLlama model and tokenizer
    2. Loads the real Diffuser model
    3. Sets up the KV Patcher and Context Maintenance systems
    4. Initializes the context with some starting tokens
    5. Configures the system for testing (smaller context window, faster decay)
    
    Returns:
        Dict containing all components needed for the test
    """
    original_config = {
        'CONTEXT_WINDOW_TARGET': config.CONTEXT_WINDOW_TARGET,
        'BRIGHTNESS_REPAIR_THRESHOLD': config.BRIGHTNESS_REPAIR_THRESHOLD,
        'BRIGHTNESS_DECAY_PER_TICK': config.BRIGHTNESS_DECAY_PER_TICK
    }
    
    # Temporarily modify config for testing
    config.CONTEXT_WINDOW_TARGET = 480  # Smaller window to prevent diffuser issues and OOM
    config.BRIGHTNESS_REPAIR_THRESHOLD = 200.0  # Higher threshold for more aggressive repairs
    config.BRIGHTNESS_DECAY_PER_TICK = 5.0  # Faster decay for testing
    
    print("\n=== Setting up Halo Weave integration test ===\n")
    
    # Load the real model and tokenizer
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model(MODEL_NAME, trust_remote_code=False)
    model.eval()  # Ensure evaluation mode
    
    # Load the real diffuser model
    print(f"Loading diffuser model: {DIFFUSER_MODEL_NAME}")
    diffuser_model = DiffuserModel(DIFFUSER_MODEL_NAME)
        
    # Initialize KV Mirror
    kv_mirror = KVMirror()
        
    # Initialize KV Cache Patcher
    kv_patcher = KVCachePatcher(
        model=model
    )
        
    # Initialize Context Maintenance handler
    context_maintenance = ContextMaintenance(
        kv_mirror_manager=kv_mirror,
        diffuser=diffuser_model,
        kv_patcher=kv_patcher,
        main_model=model,
        processor=tokenizer
    )
    
    # Create initial context with 500 tokens
    input_ids, attention_mask = prepare_initial_context(tokenizer, model, num_tokens=500)
    
    # Register initial tokens in KV Mirror
    for pos, token_id in enumerate(input_ids[0]):
        kv_mirror.add(
            token_id=token_id.item(),
            position=pos,
            source='system_init',
            removal_bias=0.0
        )
    
    # Create initial KV cache with a mock forward pass
    past_key_values = None  # Will be populated during the first full forward pass
    
    # Create event tracker for monitoring system behavior
    event_tracker = EventTracker()
    
    # Create sampler state for token selection
    sampler_state = SamplerState(
        temperature=0.8,  # Moderate sampling temperature
        top_k=40,         # Standard top-k value
        top_p=0.95        # Standard top-p value
    )
    
    # Track initial state
    ctx_size_initial = kv_mirror.get_current_size()
    print(f"Initial context size: {ctx_size_initial}")
    
    # Execute forward pass to prime the KV cache
    print("Executing initial forward pass to prime KV cache...")
    logits, past_key_values, attentions = execute_forward_pass(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,  # Use matching attention mask for priming
        get_attentions=True  # Enable attention outputs for testing
    )
    
    # Return all components needed for the test using yield
    # This allows teardown code to run after the test is complete
    yield {
        "model": model,
        "tokenizer": tokenizer,
        "diffuser_model": diffuser_model,
        "kv_mirror": kv_mirror,
        "kv_patcher": kv_patcher,
        "context_maintenance": context_maintenance,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "event_tracker": event_tracker,
        "sampler_state": sampler_state,
        "ctx_size_initial": ctx_size_initial
    }

    # Teardown: Restore original config values after test completes
    print("\n=== Restoring original config values ===\n")
    for key, value in original_config.items():
        setattr(config, key, value)

@pytest.mark.asyncio
async def test_halo_weave_run_700_steps(initialized_halo_weave_system):
    """Run Halo Weave system for 700 steps and verify correct operation using real components."""
    # Extract components from fixture
    system = initialized_halo_weave_system
    model = system['model']
    tokenizer = system['tokenizer']
    diffuser_model = system['diffuser_model']
    kv_mirror = system['kv_mirror']
    kv_patcher = system['kv_patcher']
    context_maintenance = system['context_maintenance']
    input_ids = system['input_ids']
    attention_mask = system['attention_mask']
    past_key_values = system['past_key_values']
    event_tracker = system['event_tracker']
    sampler_state = system['sampler_state']
    ctx_size_initial = system['ctx_size_initial']
    
    # Capture initial brightness state
    initial_snapshot = kv_mirror.snapshot()
    initial_tokens = initial_snapshot['tokens']
    
    # Reset the cache before starting the loop to ensure proper priming
    past_key_values = None
    
    # === Main loop: Generate tokens and maintain context ===
    num_steps = 700  # Run for 700 steps - should be enough to trigger culling and repairs
    
    # We're using the real components directly, no mocking required
    for step in range(num_steps):
        # === Step 1: Forward pass through model ===
        if step == 0 or past_key_values is None:
            # Priming pass
            model_input_ids = input_ids
            current_attention_mask = torch.ones_like(model_input_ids)  # Ensure exact match with input_ids
            current_position_ids = None
        else:
            # KV cache step - only pass the last token
            model_input_ids = input_ids[:, -1:]
            current_attention_mask = None  # Let model handle mask for cache steps
            # Calculate position ID based on cache length
            cache_seq_len = past_key_values[0][0].shape[2]  # Get sequence length from KV cache
            current_position_ids = torch.tensor([[cache_seq_len]], device=model.device)
        
        logits, new_past_key_values, attentions = execute_forward_pass(
            model=model,
            input_ids=model_input_ids,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            position_ids=current_position_ids if step > 0 else None,
            get_attentions=True  # Enable attention outputs for testing
        )
        
        # Keep past_key_values updated
        past_key_values = new_past_key_values
        
        # Prepare outputs wrapper for maintenance phase
        # Use SimpleNamespace as lightweight object with attribute access
        from types import SimpleNamespace
        outputs = SimpleNamespace()
        outputs.past_key_values = past_key_values
        outputs.attentions = attentions  # Used for brightness calculation
        outputs.logits = logits
        
        # === Step 2: Run Maintenance Phase ===
        patched_past_key_values, maintenance_events = await context_maintenance.run_phase(
            model_outputs=outputs,
            current_input_ids=input_ids,
            current_attention_mask=attention_mask,
            current_past_key_values=past_key_values,
            generation_step=step
        )
        
        # Update past_key_values for next iteration
        past_key_values = patched_past_key_values
        
        # === Step 3: Record maintenance events in the tracker ===
        if maintenance_events:
            for event in maintenance_events:
                if event['type'] == 'cull':
                    # Record culled tokens
                    for pos in event['positions']:
                        # Lookup token details from the snapshot before culling
                        token_id = event.get('token_ids', {}).get(pos, -1)
                        brightness = event.get('brightness_values', {}).get(pos, 0.0)
                        event_tracker.cull_token(pos, token_id, brightness)
                elif event['type'] == 'repair':
                    # Record repaired tokens
                    for diff in event['diffs']:
                        position = diff['position']
                        old_token = diff['old_token_id']
                        new_token = diff['new_token_id']
                        old_brightness = diff.get('old_brightness', 0.0)
                        new_brightness = config.BRIGHTNESS_SEED.get('default', 255.0)  # New tokens get default brightness
                        event_tracker.repair_token(position, old_token, new_token, old_brightness, new_brightness)
        
        # === Step 4: Select next token ===
        # === Step 3: Select next token using sampler ===
        selected_token_id, _, _ = select_next_token(
            logits=logits,
            input_ids=input_ids,
            sampler_state=sampler_state
        )
        
        # === Step 5: Update state with the new token ===
        next_token = torch.tensor([[selected_token_id]], dtype=torch.long, device=input_ids.device)  # Ensure same device
        input_ids = torch.cat([input_ids, next_token], dim=1)
        token_attention = torch.ones((1, 1), dtype=torch.long, device=attention_mask.device)  # Ensure same device
        attention_mask = torch.cat([attention_mask, token_attention], dim=1)
        
        # Record new token addition to the event tracker
        new_position = kv_mirror.get_current_size()
        event_tracker.add_token(selected_token_id, new_position)
        
        # Register the new token in the KV Mirror
        instance_id = kv_mirror.add(
            token_id=selected_token_id,
            position=new_position,
            source='llm',
            removal_bias=0.0
        )
        
        # Periodically print progress
        if step % 100 == 0:
            print(f"Step {step}: Context size: {kv_mirror.get_current_size()}")
    
    # === After the loop, generate summary and perform assertions ===
    # Get the final state
    final_snapshot = kv_mirror.snapshot()
    final_tokens = final_snapshot['tokens']
    final_kv_mirror = final_snapshot['kv_mirror']
    
    # Calculate statistics
    ctx_size_final = kv_mirror.get_current_size()
    total_add_events = sum(1 for e in event_tracker.events if e['type'] == 'add')
    total_cull_events = sum(1 for e in event_tracker.events if e['type'] == 'cull')
    total_repair_events = sum(1 for e in event_tracker.events if e['type'] == 'repair')
    
    # Calculate brightness statistics
    brightness_values = [token.brightness for token in final_tokens.values()]
    min_brightness = min(brightness_values) if brightness_values else 0
    max_brightness = max(brightness_values) if brightness_values else 0
    avg_brightness = sum(brightness_values) / len(brightness_values) if brightness_values else 0
    
    # Generate summary of events
    events_summary = event_tracker.summary(first_n=10, last_n=10)
    
    # Print summary report
    print("\n===== HALO WEAVE TEST SUMMARY =====")
    print(f"Initial context size: {ctx_size_initial}")
    print(f"Final context size: {ctx_size_final}")
    print(f"Target context size: {config.CONTEXT_WINDOW_TARGET}")
    print(f"Total tokens added: {total_add_events}")
    print(f"Total tokens culled: {total_cull_events}")
    print(f"Total tokens repaired: {total_repair_events}")
    print("\nBrightness Statistics:")
    print(f"Min brightness: {min_brightness:.2f}")
    print(f"Max brightness: {max_brightness:.2f}")
    print(f"Avg brightness: {avg_brightness:.2f}")
    
    print("\nFirst 10 Events:")
    for event in events_summary['first_events']:
        print(f"  {event}")
    
    print("\nLast 10 Events:")
    for event in events_summary['last_events']:
        print(f"  {event}")
    
    # Decode the final 50 tokens for coherence check
    final_token_ids = input_ids[0, -50:].tolist()
    final_text = tokenizer.decode(final_token_ids)
    print(f"\nFinal generated text (last 50 tokens): {final_text}")
    
    # === Assertions to verify correct behavior ===
    # 1. Context size should converge to target
    assert abs(ctx_size_final - config.CONTEXT_WINDOW_TARGET) <= 5, \
        f"Context size {ctx_size_final} did not converge to target {config.CONTEXT_WINDOW_TARGET}"
    
    # 2. Culling should have occurred
    assert total_cull_events > 0, "No culling events occurred during the test"
    
    # 3. Token repair should have been attempted
    assert total_repair_events > 0, "No repair events occurred during the test"
    
    # 4. Brightness values should be in valid range
    assert min_brightness >= 0, f"Minimum brightness {min_brightness} is below 0"
    assert max_brightness <= config.BRIGHTNESS_MAX, f"Maximum brightness {max_brightness} exceeds cap {config.BRIGHTNESS_MAX}"
    
    # 5. Average brightness should be less than initial value (decay over time)
    assert avg_brightness < config.BRIGHTNESS_SEED['default'], \
        f"Average brightness {avg_brightness} did not decrease from initial {config.BRIGHTNESS_SEED['default']}"
