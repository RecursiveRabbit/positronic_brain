"""
Brightness Engine for the Halo Weave system.

This module calculates and updates "brightness" scores for tokens based on attention patterns.
Brightness is a key metric used by the KV Mirror Culling system to identify which parts of
the context should be preserved or culled based on their attention over time.

The brightness update rule follows a Time-To-Live (TTL) approach where tokens:
1. Start with a source-dependent initial brightness (config.BRIGHTNESS_SEED)
2. Lose a fixed amount of brightness per step (config.BRIGHTNESS_DECAY_PER_TICK)
3. Gain brightness from attention (attention * config.BRIGHTNESS_GAIN_COEFFICIENT)
4. Are clamped between 0 and config.BRIGHTNESS_MAX
"""

import os
import time
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .metrics import timed_histogram, inc_counter, inc_histogram
from .kv_mirror import KVMirror


def _async_save_attention(attention_data: torch.Tensor, step: int, attention_dir: str = "attention_traces"):
    """Save attention traces to disk asynchronously (not truly async yet, but prepared for future).
    
    Args:
        attention_data: The attention tensor to save (mean across heads)
        step: The current generation step
        attention_dir: Directory to save attention traces to
    """
    # Create the directory if it doesn't exist
    os.makedirs(attention_dir, exist_ok=True)
    
    # Convert to numpy and save as JSON for now (can be optimized to parquet later)
    attention_np = attention_data.detach().cpu().numpy()
    
    # Save to JSON file
    filename = f"{attention_dir}/attention_trace_{int(time.time())}_{step}.json"
    with open(filename, "w") as f:
        json.dump({
            "step": step,
            "timestamp": time.time(),
            "attention": attention_np.tolist()
        }, f)


@timed_histogram("brightness_engine_update_seconds")
def update_brightness_scores(
    kv_mirror_manager: KVMirror,
    outputs: Any,
    generation_step: int,
    decay_per_tick: float = None,
    gain_coefficient: float = None
) -> Dict[str, int]:
    """
    Update brightness scores for all active tokens based on attention patterns.
    
    This function extracts attention scores from the model outputs, calculates new brightness
    values for each token based on the Halo Weave v0 TTL approach:
    
    b_new = max(0, min(BRIGHTNESS_MAX, b_prev - decay + int(attention * gain_coefficient)))
    
    Args:
        kv_mirror_manager: The KVMirror instance managing token state
        outputs: The outputs from model forward pass, expected to contain attentions
        generation_step: Current generation step (for attention trace saving)
        decay_per_tick: Amount to decay brightness each tick (default: from config)
        gain_coefficient: Multiplier for attention-based brightness gain (default: from config)
        
    Returns:
        Dict with statistics about the update operation
    """
    # Import config here to avoid circular imports
    from . import config

    # Use provided values or defaults from config
    if decay_per_tick is None:
        decay_per_tick = config.BRIGHTNESS_DECAY_PER_TICK
        
    if gain_coefficient is None:
        gain_coefficient = config.BRIGHTNESS_GAIN_COEFFICIENT
    
    # Add debug entry logging
    print("[DEBUG Brightness] Entering update_brightness_scores...")
    
    # Check if attentions are available in the outputs
    attentions_available = hasattr(outputs, 'attentions') and outputs.attentions is not None
    print(f"[DEBUG Brightness] outputs.attentions available: {attentions_available}")
    
    if not attentions_available:
        inc_counter("brightness_engine_missing_attentions_total")
        return {"error": "No attention data available in outputs"}
    
    try:
        # Extract attention patterns from the last layer
        # Expected shape: [batch, num_heads, seq_len, seq_len]
        # We want the attention from the latest token (last position) to all previous tokens
        attentions = outputs.attentions[-1]  # Get last layer's attention
        print(f"[DEBUG Brightness] attentions shape: {attentions.shape}")
        
        # Get attention from latest token to all previous tokens, averaging across heads
        # Shape after mean: [batch, seq_len]
        latest_token_attention = attentions.mean(dim=1)[:, -1, :]
        
        # Log attention tensor stats
        att_vec = latest_token_attention[0]  # First batch item
        print(f"[DEBUG Brightness] att_vec stats: shape={att_vec.shape}, "
              f"dtype={att_vec.dtype}, device={att_vec.device}, "
              f"min={att_vec.min().item():.6f}, max={att_vec.max().item():.6f}, "
              f"mean={att_vec.mean().item():.6f}")
        
        # Log attention distribution telemetry
        attention_values = latest_token_attention[0].detach().cpu().numpy()  # Assuming batch size 1
        if len(attention_values) > 0:
            # Calculate attention distribution statistics
            attention_mean = float(np.mean(attention_values))
            attention_median = float(np.median(attention_values))
            attention_max = float(np.max(attention_values))
            attention_min = float(np.min(attention_values)) if len(attention_values) > 0 else 0.0
            attention_std = float(np.std(attention_values))
            
            # Log metrics
            inc_histogram("attention_stats_mean", attention_mean)
            inc_histogram("attention_stats_median", attention_median)
            inc_histogram("attention_stats_max", attention_max)
            inc_histogram("attention_stats_min", attention_min)
            inc_histogram("attention_stats_std", attention_std)
        
        # Save attention traces periodically if enabled
        if config.ATTENTION_TRACE_INTERVAL > 0 and generation_step % config.ATTENTION_TRACE_INTERVAL == 0:
            _async_save_attention(latest_token_attention[0], generation_step)
        
        # Prepare brightness updates based on attention scores
        # Get a snapshot of the current KV mirror state
        mirror_snapshot = kv_mirror_manager.snapshot()
        kv_mirror = mirror_snapshot['kv_mirror']  # position -> instance_id
        tokens = mirror_snapshot['tokens']  # instance_id -> ContextToken
        
        # Calculate brightness updates for each position based on attention received
        brightness_updates = {}  # instance_id -> new_brightness
        
        # Process each position with a valid token in the KV mirror
        for position, instance_id in kv_mirror.items():
            # Skip if position is out of range of our attention scores
            if position >= latest_token_attention.shape[1]:
                continue
                
            # Get current token data
            token = tokens.get(instance_id)
            if token is None:
                continue
                
            # Get attention score for this position
            attention_score = latest_token_attention[0, position].item()  # Assuming batch size 1
            
            # Calculate new brightness using TTL approach:
            # b_new = b_prev - decay + int(attention * gain_coefficient)
            current_brightness = token.brightness
            attention_gain = int(attention_score * gain_coefficient)  # Integer gain based on attention
            new_brightness = current_brightness - decay_per_tick + attention_gain
            
            # Add detailed logging for a few specific positions
            if position == 0 or position == latest_token_attention.shape[1] - 1 or position % 100 == 0:
                print(f"[DEBUG Brightness] Pos {position}: b_prev={current_brightness:.2f}, "
                      f"decay={decay_per_tick}, att={attention_score:.6f}, "
                      f"gain={attention_gain}, b_new={new_brightness:.2f}")
            
            # Update brightness for this instance ID
            brightness_updates[instance_id] = new_brightness

        # Batch update brightness scores in the KV mirror
        print(f"[DEBUG Brightness] Sending {len(brightness_updates)} brightness updates to KVMirror.")
        
        if brightness_updates:
            result = kv_mirror_manager.batch_update_brightness(brightness_updates)
            print(f"[DEBUG Brightness] Update result: {result}")
            return result
        else:
            print("[DEBUG Brightness] No updates needed (empty brightness_updates dict).")
            return {"success": 0, "no_updates_needed": True}
            
    except Exception as e:
        inc_counter("brightness_engine_error_total")
        print(f"[Brightness Engine ERROR] Failed to update brightness scores: {type(e).__name__}: {str(e)}")
        return {"error": str(e)}
