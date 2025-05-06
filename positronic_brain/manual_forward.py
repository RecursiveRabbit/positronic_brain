"""
Manual implementation of forward pass for TinyLlama models.
This module provides direct tensor operations for inference without using the model's forward method,
preventing unwanted mutation of cache objects.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Tuple, Optional, Union, List, Any

# Configure logger for this module
logger = logging.getLogger(__name__)

# Try to import RoPE helper functions
try:
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        LlamaRotaryEmbedding,
        rotate_half
    )
    logger.info("Successfully imported Llama RoPE components")
except ImportError:
    logger.warning("Could not import all Llama RoPE components from transformers")
    # Define minimal fallback implementations if needed
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except ImportError:
        logger.error("Failed to import apply_rotary_pos_emb - RoPE won't work!")

# Placeholder for Cache type if needed
try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    try:
        from transformers.utils import DynamicCache
        Cache = DynamicCache  # Alias for older versions
    except ImportError:
        logger.warning("Could not import Cache/DynamicCache from transformers")
        Cache = None
        DynamicCache = None


def manual_forward_step(
    model, 
    new_input_ids: torch.Tensor, 
    past_kv_cache: Any, 
    position_ids: torch.Tensor,
    get_attentions: bool = False
) -> Tuple[torch.Tensor, Optional[List]]:
    """
    Performs a single forward pass step manually using model components.
    
    This function bypasses the model's forward method to prevent any mutation
    of the past_kv_cache. It manually processes each layer's operations,
    maintaining a read-only approach to the cache.
    
    Args:
        model: The pre-trained TinyLlama model instance
        new_input_ids: Tensor containing IDs for the new token(s) only. Shape [batch_size, new_seq_len]
        past_kv_cache: The read-only KV cache from previous steps. Can be tuple format or a Cache object
        position_ids: Tensor containing position indices for the new tokens. Shape [batch_size, new_seq_len]
        get_attentions: Whether to compute and return attention weights

    Returns:
        Tuple containing:
        - logits: Output logits for the new token(s). Shape [batch_size, new_seq_len, vocab_size]
        - attentions: List of attention weights per layer, or None
    """
    logger.info("-" * 40)
    logger.info("Executing manual forward step (read-only KV cache approach)")

    # Extract model configuration parameters
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, "num_key_value_heads", num_attention_heads)
    head_dim = hidden_size // num_attention_heads
    num_layers = len(model.model.layers)
    
    logger.info(f"Model config: hidden_size={hidden_size}, num_heads={num_attention_heads}, "
                f"num_kv_heads={num_key_value_heads}, head_dim={head_dim}, num_layers={num_layers}")
    
    # Input dimensions
    batch_size, seq_len = new_input_ids.shape
    logger.info(f"Input: batch_size={batch_size}, new_seq_len={seq_len}, "
                f"position_ids={position_ids.tolist() if position_ids.numel() < 10 else '...'}")
    
    # Storage for attention weights if requested
    all_attentions = [] if get_attentions else None
    
    # Step 1: Input Embedding
    logger.info("Step 1: Input embedding")
    hidden_states = model.model.embed_tokens(new_input_ids)
    logger.debug(f"Embedded hidden_states shape: {hidden_states.shape}")
    
    # Step 2: Extract past K/V tensors from cache
    logger.info("Step 2: Extracting past K/V tensors from cache (READ-ONLY)")
    past_seq_len = 0
    past_keys = []
    past_values = []
    
    # Handle different cache types
    if Cache is not None and isinstance(past_kv_cache, Cache):
        logger.info(f"Cache type: {type(past_kv_cache).__name__} (DynamicCache or subclass)")
        # Try to access via internal attributes (might need adjustment)
        # Method 1: Try key_states and value_states lists
        if hasattr(past_kv_cache, "key_states") and hasattr(past_kv_cache, "value_states"):
            logger.debug("Accessing cache via key_states/value_states attributes")
            past_keys = past_kv_cache.key_states
            past_values = past_kv_cache.value_states
        # Method 2: Try _key_values list of dicts (common in transformers 4.37+)
        elif hasattr(past_kv_cache, "_key_values"):
            logger.debug("Accessing cache via _key_values attribute")
            for layer_idx in range(num_layers):
                layer_dict = past_kv_cache._key_values[layer_idx]
                past_keys.append(layer_dict["key"])
                past_values.append(layer_dict["value"])
        # Method 3: Other potential access methods if needed
        else:
            logger.warning("Could not determine DynamicCache structure, attempting iteration")
            past_cache_len = past_kv_cache.get_seq_length()
            logger.info(f"Cache reports sequence length: {past_cache_len}")
            
            # Try to access each layer's tensors (specifics depend on implementation)
            # This might need adjustment based on actual cache structure
            try:
                for layer_idx in range(num_layers):
                    # Different versions might have different access methods
                    layer_cache = past_kv_cache[layer_idx]
                    if isinstance(layer_cache, dict):
                        past_keys.append(layer_cache.get("key", None))
                        past_values.append(layer_cache.get("value", None))
                    elif isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                        past_keys.append(layer_cache[0])
                        past_values.append(layer_cache[1])
            except Exception as e:
                logger.error(f"Error accessing cache: {e}")
                raise ValueError(f"Could not extract K/V from cache: {e}")
    
    # Handle tuple-based caches (conventional format)
    elif isinstance(past_kv_cache, tuple):
        logger.info("Cache type: tuple (conventional KV cache format)")
        for layer_idx, layer_cache in enumerate(past_kv_cache):
            if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                past_keys.append(layer_cache[0])
                past_values.append(layer_cache[1])
            else:
                raise ValueError(
                    f"Unexpected layer cache structure at idx {layer_idx}: {type(layer_cache)}"
                )
    else:
        logger.warning(f"Unrecognized cache type: {type(past_kv_cache).__name__}")
        # Attempt to access cache via get_seq_length() method if available
        if hasattr(past_kv_cache, "get_seq_length"):
            past_seq_len = past_kv_cache.get_seq_length()
            logger.info(f"Cache reports sequence length: {past_seq_len}")
    
    # Validate extracted K/V tensors
    if past_keys and past_values:
        past_seq_len = past_keys[0].size(2)  # Shape [B, H, S, D] -> S is dim 2
        logger.info(f"Extracted {len(past_keys)} layer K/V pairs from cache")
        logger.info(f"Past sequence length: {past_seq_len}")
        logger.debug(f"Past key shape: {past_keys[0].shape}, Past value shape: {past_values[0].shape}")
    else:
        logger.warning("No past K/V extracted, treating as initial generation")
    
    # Step 3: Process through each decoder layer
    logger.info("Step 3: Processing through decoder layers")
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        logger.debug(f"Layer {layer_idx}/{num_layers}")
        
        # 3.1: Layer normalization (pre-attention)
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        
        # 3.2: Self-attention QKV projections
        query_states = decoder_layer.self_attn.q_proj(hidden_states)
        key_states = decoder_layer.self_attn.k_proj(hidden_states)
        value_states = decoder_layer.self_attn.v_proj(hidden_states)
        
        # 3.3: Reshape QKV for multi-head attention
        query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
        query_states = query_states.transpose(1, 2)  # [batch, heads, seq, dim]
        
        key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
        key_states = key_states.transpose(1, 2)  # [batch, heads, seq, dim]
        
        value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
        value_states = value_states.transpose(1, 2)  # [batch, heads, seq, dim]
        
        # 3.4: Apply rotary positional embeddings (RoPE)
        # First, check if the model already has a rotary embedding at the model level
        if hasattr(model.model, 'rotary_emb'):
            logger.debug(f"Using model's existing rotary_emb for layer {layer_idx}")
            rotary_emb = model.model.rotary_emb
        # If not at model level, try using decoder layer's rotary embedding (if available)
        elif hasattr(decoder_layer.self_attn, 'rotary_emb'):
            logger.debug(f"Using layer's existing rotary_emb for layer {layer_idx}")
            rotary_emb = decoder_layer.self_attn.rotary_emb
        # Otherwise, create our own LlamaRotaryEmbedding with minimal arguments
        else:
            logger.debug(f"Creating minimal LlamaRotaryEmbedding for layer {layer_idx}")
            
            # Extract required parameters from model config
            max_position_embeddings = getattr(model.config, "max_position_embeddings", 2048)
            rope_theta = getattr(model.config, "rope_theta", 10000.0)
            rope_dim = head_dim  # Default to head_dim (usually correct for Llama models)
            
            # If model uses partial rotary embeddings, get the specified dimension
            if hasattr(model.config, "rope_dim") and model.config.rope_dim is not None:
                rope_dim = model.config.rope_dim
                
            logger.debug(f"RoPE params: dim={rope_dim}, max_pos={max_position_embeddings}, theta={rope_theta}")
            
            # Based on error logs: LlamaRotaryEmbedding takes 2-3 positional arguments
            try:
                # Most common for this version: LlamaRotaryEmbedding(dim, max_seq, base=10000.0)
                logger.debug("Attempting minimal LlamaRotaryEmbedding constructor with 2 args + base")
                rotary_emb = LlamaRotaryEmbedding(rope_dim, max_position_embeddings, base=rope_theta)
            except Exception as e1:
                logger.warning(f"Minimal RoPE initialization failed: {e1}")
                try:
                    # Very minimal: just dimension and default max_seq
                    logger.debug("Attempting bare minimum LlamaRotaryEmbedding constructor with just dim")
                    rotary_emb = LlamaRotaryEmbedding(rope_dim)
                except Exception as e2:
                    logger.warning(f"Bare minimum RoPE initialization failed: {e2}")
                    # Last resort: use a hardcoded approach for this specific model
                    logger.warning("Using hardcoded rotary embedding as last resort")
                    rotary_emb = LlamaRotaryEmbedding(rope_dim, max_position_embeddings)
        
        # Generate cos/sin caches based on sequence length
        # This returns cos/sin values for all positions up to seq_len
        cos, sin = rotary_emb(value_states, seq_len=past_seq_len + seq_len)
        
        # Apply the rotary embeddings to the query and key states
        # position_ids tells apply_rotary_pos_emb which positions to use from cos/sin
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        
        # 3.5: Combine with past K/V (READ-ONLY access)
        if past_keys and layer_idx < len(past_keys):
            # Concatenate along sequence dimension (dim=2)
            past_key = past_keys[layer_idx]
            past_value = past_values[layer_idx]
            
            key_states_combined = torch.cat([past_key, key_states], dim=2)
            value_states_combined = torch.cat([past_value, value_states], dim=2)
            logger.debug(f"Combined K/V shapes - key: {key_states_combined.shape}, value: {value_states_combined.shape}")
        else:
            key_states_combined = key_states
            value_states_combined = value_states
            logger.debug("No past K/V to combine")
        
        # 3.6: Self-attention with scaled dot product
        # Handle GroupedQueryAttention (GQA) if needed
        is_gqa = num_attention_heads != num_key_value_heads
        
        # Try to handle GQA and causal masking appropriately
        try:
            # For PyTorch 2.0+
            attn_output = F.scaled_dot_product_attention(
                query_states,                  # [batch, q_heads, seq, dim]
                key_states_combined,           # [batch, kv_heads, seq_total, dim]
                value_states_combined,         # [batch, kv_heads, seq_total, dim]
                attn_mask=None,                # Use default causal mask
                is_causal=True,                # Enforce causality
                dropout_p=0.0                  # No dropout during inference
            )
        except (TypeError, AttributeError) as e:
            # Fallback for older PyTorch versions without is_causal param
            logger.warning(f"SDPA with is_causal failed: {e}. Falling back to version without is_causal.")
            attn_output = F.scaled_dot_product_attention(
                query_states, 
                key_states_combined,
                value_states_combined,
                attn_mask=None,
                dropout_p=0.0
            )
        
        # 3.7: Reshape attention output and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
        attn_output = decoder_layer.self_attn.o_proj(attn_output)
        
        # 3.8: Residual connection
        hidden_states = residual + attn_output
        
        # 3.9: MLP block with another residual
        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # 3.10: Optional attention weight capture
        if get_attentions:
            # Note: F.scaled_dot_product_attention doesn't return attention weights directly
            # We'd need to manually compute them which would be expensive
            # For now, we'll store None as a placeholder
            all_attentions.append(None)
    
    # Step 4: Final normalization
    logger.info("Step 4: Final normalization and LM head")
    hidden_states = model.model.norm(hidden_states)
    
    # Step 5: Apply language model head to get logits
    logits = model.lm_head(hidden_states)
    logger.info(f"Final logits shape: {logits.shape}")
    
    # Return logits and attention weights (if available)
    logger.info("Manual forward step completed successfully")
    logger.info("-" * 40)
    
    return logits, (all_attentions if get_attentions else None)
