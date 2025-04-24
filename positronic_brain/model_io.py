"""
Model IO operations for the Infinite Scroll inference engine.
This module contains functions for loading models, executing forward passes,
and managing model cache operations.
"""

import torch
from typing import Optional, Tuple, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from .metrics import timed_histogram


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
        device_map="auto",  # Let the library optimize device placement
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
    attention_mask: torch.Tensor,
    past_key_values: Optional[Tuple] = None,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[Tuple], Any]:  # logits, new_kv_cache, attentions
    """
    Executes a forward pass through the model.

    Args:
        model: The loaded transformer model.
        input_ids: The input token IDs for this pass.
        attention_mask: The attention mask for this pass.
        past_key_values: The KV cache from the previous step, if any.
        position_ids: Optional tensor containing the position IDs for the input tokens.

    Returns:
        A tuple containing:
        - logits: Raw output logits from the model.
        - new_past_key_values: The updated KV cache.
        - attentions: Attention weights (or None).
    """
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,  # Add position_ids parameter
            use_cache=True,
            output_attentions=True  # Assuming we always want attentions for pruning
        )

        logits = outputs.logits
        new_kv_cache = outputs.past_key_values
        # Use getattr for safer access to optional attributes like attentions
        attentions = getattr(outputs, 'attentions', None)

        return logits, new_kv_cache, attentions

    except Exception as e:
        print(f"[Model IO Error] Forward pass failed: {type(e).__name__} - {e}")
        # Re-raise the exception to be handled by the main loop
        raise e
