"""
Model IO operations for the Infinite Scroll inference engine.
This module contains functions for loading models, executing forward passes,
and managing model cache operations.
"""

import torch
import logging
from typing import Optional, Tuple, Dict, Any, Union, List
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# Configure logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import DynamicCache class - newer versions call it Cache
try:
    # Adjust this import based on your exact transformers version if needed
    from transformers.cache_utils import Cache as DynamicCache
except ImportError:
    try:
        from transformers.utils import DynamicCache
    except ImportError:
        # If we can't import, define a placeholder for isinstance checks
        logger.warning("Could not import DynamicCache type hint. Proceeding without specific type check.")
        class DynamicCache:  # type: ignore
            pass
        print("Warning: Could not import DynamicCache from transformers")

from .metrics import timed_histogram


def freeze_dynamic_cache(dc: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Return a static (immutable) tuple-of-tensors view of a DynamicCache 
    without copying the underlying GPU storage.
    
    Args:
        dc: A DynamicCache object containing key-value tensors
        
    Returns:
        A tuple of tuples with tensor views, matching the legacy past_key_values format
    """
    static_layers = []
    try:
        # For transformers 4.37+ DynamicCache structure
        for layer in dc.values():
            # Each layer is a dict like {"key": K, "value": V}
            k, v = layer["key"], layer["value"]
            # We don't clone - just use the existing tensors directly
            static_layers.append((k, v))
    except (AttributeError, KeyError):
        # Fallback for other DynamicCache implementations
        print("[freeze_dynamic_cache] Warning: Using fallback mechanism for this DynamicCache structure")
        # Try to access all possible known attributes to get layers
        if hasattr(dc, "key_states") and hasattr(dc, "value_states"):
            # Some implementations store as parallel lists
            for k, v in zip(dc.key_states, dc.value_states):
                static_layers.append((k, v))
        elif hasattr(dc, "_key_states") and hasattr(dc, "_value_states"):
            # Protected attribute version
            for k, v in zip(dc._key_states, dc._value_states):
                static_layers.append((k, v))
    
    if not static_layers:
        print("[freeze_dynamic_cache] Error: Could not extract key-value tensors from DynamicCache!")
    
    return tuple(static_layers)


@timed_histogram("model_io_load_model_seconds")
def load_model(model_name: str, trust_remote_code: bool):
    """
    Load model and processor with the specified configuration.
    
    Args:
        model_name: The name or path of the model to load
        trust_remote_code: Whether to trust remote code in the model
        
    Returns:
        Tuple containing (model, processor)
    """
    # Add CUDA and cuDNN optimizations for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 4
    print("[Optimizations] TF32 and cuDNN benchmark enabled in load_model.")
    print("Loading processor...")
    # Use AutoTokenizer directly for pure text models like TinyLlama
    processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    print(f"Loaded processor type: {type(processor)}")
    
    # Access attributes directly on the processor/tokenizer object
    if processor.pad_token is None:
        if processor.eos_token is not None:
            processor.pad_token = processor.eos_token
            print(f"Set processor pad_token to eos_token ({processor.eos_token})")
        else:
            # Add a fallback if EOS is also missing
            processor.add_special_tokens({'pad_token': '[PAD]'})
            print(f"Added new pad_token: {processor.pad_token}")
    
    # Ensure pad_token_id is set if pad_token exists
    if processor.pad_token is not None and processor.pad_token_id is None:
        processor.pad_token_id = processor.convert_tokens_to_ids(processor.pad_token)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"":0},  # Force entire model onto a single GPU device (cuda:0)
        trust_remote_code=trust_remote_code,
        output_attentions=True,  # Enable attention output for attention-based KV cache management
        return_dict_in_generate=True  # Enable return dict in generate for attention scores
    )
    model.eval()
    print("Model loaded.")
    
    # Update generation config to set do_sample=True to match our sampling behavior
    # This will eliminate the UserWarning about temperature being set with do_sample=False
    try:
        if hasattr(model, "generation_config"):
            model.generation_config.do_sample = True
            # We could also set temperature/top_k here, but our manual logic overrides anyway
            print("Updated model.generation_config.do_sample = True")
        else:
            print("Model does not have generation_config attribute.")
    except Exception as e:
        print(f"Warning: Could not update model generation config: {e}")

    return model, processor


@timed_histogram("model_io_move_cache_seconds")
def move_cache_to_device(past_key_values, target_device):
    """
    Move KV cache to specified device with non-blocking transfer.
    
    Args:
        past_key_values: The KV cache to move
        target_device: The target device to move to
        
    Returns:
        The KV cache on the new device
    """
    if past_key_values is None:
        return None
        
    new_cache = []
    for layer_past in past_key_values:
        new_layer_past = tuple(
            past_tensor.to(target_device, non_blocking=True) for past_tensor in layer_past
        )
        new_cache.append(new_layer_past)
    return tuple(new_cache)


@timed_histogram("model_io_truncate_cache_seconds")
def truncate_kv_cache(past_key_values, max_len):
    """
    Truncates the KV cache to a specified maximum sequence length (robust).
    
    Args:
        past_key_values: The KV cache to truncate
        max_len: The maximum sequence length to keep
        
    Returns:
        The truncated KV cache
    """
    if past_key_values is None:
        return None
        
    new_cache = []
    for layer_past in past_key_values:
        # layer_past is usually a tuple of (key_states, value_states)
        # Tensor shape: [batch_size, num_heads, sequence_length, head_dim]
        current_cache_len = layer_past[0].shape[2]
        slice_len = min(max_len, current_cache_len)
        new_layer_past = tuple(
            past_tensor[:, :, -slice_len:, :] for past_tensor in layer_past
        )
        new_cache.append(new_layer_past)
    return tuple(new_cache)


@timed_histogram("model_io_forward_pass_seconds")
def execute_forward_pass(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Tuple] = None,
    position_ids: Optional[torch.Tensor] = None,
    get_attentions: bool = False
) -> Tuple[torch.Tensor, Optional[Tuple], Any]:  # logits, new_kv_cache, attentions
    """
    Executes a forward pass through the model.

    Args:
        model: The loaded transformer model.
        input_ids: The input token IDs for this pass.
        attention_mask: The attention mask for this pass.
        past_key_values: The KV cache from the previous step, if any.
        position_ids: Optional tensor containing the position IDs for the input tokens.
        get_attentions: Whether to return attention weights.

    Returns:
        A tuple containing:
        - logits: Raw output logits from the model.
        - new_past_key_values: The updated KV cache.
        - attentions: Attention weights (or None).
    """
    logger.info("-" * 20)
    logger.info(f"Executing forward pass...")
    logger.info(f"  Input IDs shape: {input_ids.shape}")
    pkv_type = type(past_key_values).__name__ if past_key_values is not None else "None"
    logger.info(f"  Received past_key_values type: {pkv_type}")
    if position_ids is not None:
        logger.info(f"  Position IDs: {position_ids.tolist()}")
    else:
        logger.info(f"  Position IDs: None")
    
    try:
        # Safety guard: when using past_key_values, only new tokens should be passed in input_ids
        if past_key_values is not None and input_ids.shape[1] > 1:
            raise ValueError("When supplying past_key_values feed only the new token(s).")
            
        # Defensive check to ensure attention_mask matches input_ids length if provided
        if attention_mask is not None and attention_mask.shape[1] != input_ids.shape[1]:
            attention_mask = attention_mask[:, :input_ids.shape[1]]
        
        # Use the original cache by default
        _pkv_to_pass = past_key_values
        
        # Check if it's a DynamicCache instance and needs freezing
        is_dynamic = isinstance(past_key_values, DynamicCache)
        
        if is_dynamic:
            logger.warning("  Detected DynamicCache - freezing view to prevent mutation.")
            try:
                _pkv_to_pass = freeze_dynamic_cache(past_key_values)
                logger.info(f"  Successfully created frozen view (type: {type(_pkv_to_pass).__name__}).")
            except Exception as e:
                logger.error(f"  Failed to freeze DynamicCache: {e}", exc_info=True)
                # Raise error for safer handling
                raise RuntimeError("Failed to create immutable view of DynamicCache") from e
        else:
            logger.info("  past_key_values is not DynamicCache or freezing is not needed.")
        
        # Execute the forward pass using the (potentially frozen) cache
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=_pkv_to_pass,  # Use the frozen view or original PKV
                position_ids=position_ids,
                use_cache=True,
                output_attentions=get_attentions
            )

        logits = outputs.logits
        returned_kv_cache = outputs.past_key_values
        # Use getattr for safer access to optional attributes like attentions
        attentions = getattr(outputs, 'attentions', None)
        
        # Log the type of cache returned by the model
        returned_pkv_type = type(returned_kv_cache).__name__ if returned_kv_cache is not None else "None"
        logger.info(f"  Model returned past_key_values type: {returned_pkv_type}")
        if hasattr(returned_kv_cache, 'get_seq_length'):
            logger.info(f"  Returned cache reports length: {returned_kv_cache.get_seq_length()}")
        elif isinstance(returned_kv_cache, tuple) and returned_kv_cache and len(returned_kv_cache) > 0:
            # Log tuple cache length if it has the expected structure
            if returned_kv_cache[0] and isinstance(returned_kv_cache[0], tuple) and len(returned_kv_cache[0]) > 0:
                logger.info(f"  Returned tuple cache first layer shape: {returned_kv_cache[0][0].shape}")
        
        logger.info("Forward pass execution finished.")
        logger.info("-" * 20)
        
        return logits, returned_kv_cache, attentions

    except Exception as e:
        logger.error(f"[Model IO Error] Forward pass failed: {type(e).__name__} - {e}", exc_info=True)
        # Re-raise the exception to be handled by the main loop
        raise e
