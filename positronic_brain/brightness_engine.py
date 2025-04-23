"""
Brightness Engine for the Halo Weave system.

This module calculates and updates "brightness" scores for tokens based on attention patterns.
Brightness is a key metric used by the Compactor/Diffuser system to identify which parts of
the context need maintenance, repair, or replacement.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .metrics import timed_histogram, inc_counter
from .kv_mirror import KVMirror


@timed_histogram("brightness_engine_update_seconds")
def update_brightness_scores(
    kv_mirror_manager: KVMirror,
    outputs: Any,
    alpha: float,
    beta: float
) -> Dict[str, int]:
    """
    Update brightness scores for all active tokens based on attention patterns.
    
    This function extracts attention scores from the model outputs, calculates new brightness
    values for each token based on attention received and a decay factor, and updates these
    values in the KV mirror.
    
    The brightness update formula is:
    brightness[j] = clamp(brightness[j] + alpha * attn_to_pos_j - beta, 0, 255)
    
    Args:
        kv_mirror_manager: The KVMirror instance managing token state
        outputs: The outputs from model forward pass, expected to contain attentions
        alpha: Attention gain factor (how much to amplify attention scores)
        beta: Decay factor (constant reduction applied each step)
        
    Returns:
        Dict with statistics about the update operation
    """
    # Check if attentions are available in the outputs
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        inc_counter("brightness_engine_missing_attentions_total")
        return {"error": "No attention data available in outputs"}
    
    try:
        # Extract attention patterns from the last layer
        # Expected shape: [batch, num_heads, seq_len, seq_len]
        # We want the attention from the latest token (last position) to all previous tokens
        attentions = outputs.attentions[-1]  # Get last layer's attention
        
        # Get attention from latest token to all previous tokens, averaging across heads
        # Shape after mean: [batch, seq_len]
        latest_token_attention = attentions.mean(dim=1)[:, -1, :]
        
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
            
            # Calculate new brightness
            # brightness = current + (alpha * attention) - beta
            current_brightness = token.brightness
            new_brightness = current_brightness + (alpha * attention_score) - beta
            
            # Update brightness for this instance ID
            brightness_updates[instance_id] = new_brightness

        # Batch update brightness scores in the KV mirror
        if brightness_updates:
            return kv_mirror_manager.batch_update_brightness(brightness_updates)
        else:
            return {"success": 0, "no_updates_needed": True}
            
    except Exception as e:
        inc_counter("brightness_engine_error_total")
        print(f"[Brightness Engine ERROR] Failed to update brightness scores: {type(e).__name__}: {str(e)}")
        return {"error": str(e)}
