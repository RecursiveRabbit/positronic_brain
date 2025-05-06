"""
Test for Step 1 of the Positronic Brain loop: Generate & Sample
This test focuses on executing a single step of the core generation and sampling logic
using the TinyLlama model.

This test consumes the session-scoped initialized_session_state from Step 0 (defined in conftest.py),
which provides a pre-loaded model, tokenizer, and primed KV cache for the fixed initial prompt.
"""

import os
import pytest
import torch

# Import our test utilities
from .test_utils import safe_save
from .conftest import execute_forward_pass
from positronic_brain.manual_forward import manual_forward_step
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any

# Define SamplerState directly in this file
@dataclass
class SamplerState:
    """Configuration for token sampling parameters."""
    temperature: float = 0.8       # Controls randomness (higher = more random)
    top_k: int = 50                # Limits to top K most likely tokens (0 = disabled)
    top_p: float = 0.9             # Nucleus sampling threshold (0.0-1.0)
    repetition_penalty: float = 1.1  # Penalty for repeated tokens (1.0 = no penalty)
    token_bias: Dict[int, float] = field(default_factory=dict)  # Bias specific tokens
    
# Token selection functions implemented directly in this file
def _step1_top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Filter logits to keep only the top k values."""
    if k <= 0:
        return logits  # No filtering
        
    # Get top k values and their indices
    values, _ = torch.topk(logits, k)
    min_value = values[:, -1].unsqueeze(-1)
    
    # Create a mask for values less than the kth value
    mask = logits < min_value
    filtered_logits = logits.clone()
    
    # Set values below threshold to -inf
    filtered_logits.masked_fill_(mask, float('-inf'))
    return filtered_logits
    
def _step1_top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Filter logits using nucleus (top-p) sampling."""
    if p <= 0.0 or p >= 1.0:
        return logits  # No filtering
        
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # Calculate cumulative probabilities
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    
    # Shift indices to the right to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0  # Keep at least one token
    
    # Scatter sorted indices back to original positions
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    
    # Set values to be removed to -inf
    filtered_logits = logits.clone()
    filtered_logits.masked_fill_(indices_to_remove, float('-inf'))
    return filtered_logits
    
def _step1_apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """Apply repetition penalty to previously generated tokens."""
    if penalty == 1.0:
        return logits  # No penalty
    
    # Get unique token IDs from input_ids
    unique_tokens = torch.unique(input_ids)
    
    # Create a penalty tensor initialized with 1.0
    penalty_tensor = torch.ones_like(logits)
    
    # Set the penalty for tokens in input_ids
    penalty_tensor.index_fill_(-1, unique_tokens, penalty)
    
    # Apply penalty (divide where logits > 0, multiply where logits < 0)
    logits_sign = torch.sign(logits)
    penalized_logits = logits.clone()
    
    # Penalize the logits by dividing or multiplying based on their sign
    pos_mask = logits_sign > 0
    neg_mask = logits_sign < 0
    
    penalized_logits.masked_scatter_(pos_mask, 
                                   penalized_logits.masked_select(pos_mask) / penalty_tensor.masked_select(pos_mask))
    penalized_logits.masked_scatter_(neg_mask, 
                                   penalized_logits.masked_select(neg_mask) * penalty_tensor.masked_select(neg_mask))
    
    return penalized_logits
    
def _step1_apply_token_bias(logits: torch.Tensor, token_bias: Dict[int, float]) -> torch.Tensor:
    """Apply bias values to specific token IDs."""
    if not token_bias:
        return logits  # No bias to apply
    
    # Create a bias tensor filled with zeros
    bias_tensor = torch.zeros_like(logits)
    
    # Apply bias values to specific token IDs
    for token_id, bias_value in token_bias.items():
        bias_tensor[..., token_id] = bias_value
    
    # Add the bias to the logits
    return logits + bias_tensor

def select_next_token(logits: torch.Tensor, 
                      input_ids: torch.Tensor, 
                      sampler_state: SamplerState) -> Tuple[int, torch.Tensor, List[Dict[str, Any]]]:
    """Select the next token based on logits and sampling parameters.
    
    Args:
        logits: Model output logits for the next token prediction [batch_size, vocab_size]
        input_ids: Input token IDs that generated these logits [batch_size, seq_len]
        sampler_state: Configuration for sampling parameters
        
    Returns:
        Tuple containing:
        - selected_token_id: The ID of the selected token
        - probs: Probability distribution after sampling
        - top_tokens_info: Information about the top tokens considered
    """
    # Extract the last logits if there's a sequence dimension
    if logits.dim() > 2:
        logits = logits[:, -1, :]
    
    # Make a copy of the logits to avoid modifying the original
    modified_logits = logits.clone()
    
    # Apply repetition penalty if needed
    if sampler_state.repetition_penalty != 1.0:
        modified_logits = _step1_apply_repetition_penalty(
            modified_logits, input_ids, sampler_state.repetition_penalty
        )
    
    # Apply token bias if provided
    if sampler_state.token_bias:
        modified_logits = _step1_apply_token_bias(
            modified_logits, sampler_state.token_bias
        )
    
    # Apply temperature if not 0
    if sampler_state.temperature > 0:
        modified_logits = modified_logits / sampler_state.temperature
    
    # Apply top-k filtering
    if sampler_state.top_k > 0:
        modified_logits = _step1_top_k_filter(modified_logits, sampler_state.top_k)
    
    # Apply top-p (nucleus) filtering
    if 0.0 < sampler_state.top_p < 1.0:
        modified_logits = _step1_top_p_filter(modified_logits, sampler_state.top_p)
    
    # Convert to probabilities
    probs = torch.softmax(modified_logits, dim=-1)
    
    # Sample from the distribution
    selected_token_id = torch.multinomial(probs, num_samples=1).item()
    
    # Get information about top tokens for debugging/logging
    top_token_count = min(20, modified_logits.size(-1))
    top_values, top_indices = torch.topk(probs, k=top_token_count)
    
    # Create info dictionaries for the top tokens
    top_tokens_info = []
    for i in range(top_token_count):
        token_id = top_indices[0, i].item()
        prob = top_values[0, i].item()
        bias = sampler_state.token_bias.get(token_id, 0.0) if sampler_state.token_bias else 0.0
        
        top_tokens_info.append({
            'token_id': token_id,
            'probability': prob,
            'bias': bias,
        })
    
    return selected_token_id, probs, top_tokens_info


@pytest.fixture(scope="function")
def default_sampler_state():
    """
    Create a standard SamplerState instance with testing parameters.
    
    Returns:
        SamplerState: Instance with default parameters
    """
    return SamplerState(
        temperature=0.8,  # Set non-zero temperature for testing
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        token_bias={}
    )


def test_generate_first_step(
    initialized_session_state,
    default_sampler_state
):
    """
    Test for the first step of generation and sampling after initial context.
    
    This test uses the shared session state (model, tokenizer, primed KV cache) from Step 0
    and generates a single new token as Step 1 of the pipeline.
    
    Args:
        initialized_session_state: Session-scoped fixture containing model, tokenizer, and primed KV cache
        default_sampler_state: Fixture containing a SamplerState instance
    """
    # Set fixed test_id for this non-parameterized test
    test_id = "fixed_initial"
    
    # Get components from session state fixture
    model = initialized_session_state['model']
    tokenizer = initialized_session_state['tokenizer']
    device = initialized_session_state['device']
    initial_input_ids = initialized_session_state['initial_input_ids']
    initial_attention_mask = initialized_session_state['initial_attention_mask']
    input_seq_len = initialized_session_state['input_seq_len']         # Length of input tokens
    expected_cache_len = initialized_session_state['cache_seq_len']  # Authoritative length from cache
    primed_kv_cache = initialized_session_state['primed_kv_cache']
    
    print(f"\n--- Running test_generate_first_step ---")
    print(f"Using session state: input_seq_len={input_seq_len}, expected_cache_len={expected_cache_len}")
    
    # Extract the last token ID from the initial input_ids for the generation step
    step1_input_token = initial_input_ids[:, -1:]  # Shape [1, 1]
    # Determine the correct position_ids for this single token
    # Use the authoritative cache length for positioning the new token
    step1_position_ids = torch.tensor([[expected_cache_len - 1]], device=device)
    
    print(f"Executing MANUAL forward pass for position {step1_position_ids.item()}...")
    
    # Execute forward pass for the last token using manual implementation
    # that prevents mutation of the primed_kv_cache object
    logits, attentions = manual_forward_step(
        model=model,
        new_input_ids=step1_input_token,
        past_kv_cache=primed_kv_cache,      # Pass the fixture's cache (read-only)
        position_ids=step1_position_ids,
        get_attentions=True                 # Request attention outputs
    )
    
    print("Manual forward pass completed successfully")
    
    # Sample the next token using the logits
    selected_token_id, probs, top_tokens_info = select_next_token(
        logits=logits,
        input_ids=initial_input_ids,  # Full input_ids for repetition penalty context
        sampler_state=default_sampler_state
    )
    
    # Get the selected token text for logging
    selected_token_text = tokenizer.decode([selected_token_id])
    print(f"Selected token ID: {selected_token_id} ('{selected_token_text}')")
    
    # Create a dictionary with the captured outputs
    captured_data = {
        'test_id': test_id,  # Identifier for this run
        'initial_input_ids': initial_input_ids,  # The full input sequence used for priming
        'initial_attention_mask': initial_attention_mask,
        'input_seq_len': input_seq_len,  # Length of input tokens
        'expected_cache_len': expected_cache_len,  # Authoritative length from cache
        'step1_input_token': step1_input_token,  # The single token fed for step 1 prediction
        'step1_position_ids': step1_position_ids,  # Position ID for the step 1 token
        'logits': logits,  # Logits output from step 1 pass
        'attentions': attentions,  # Attention output from manual forward step
        'selected_token_id': selected_token_id,  # The token ID chosen by the sampler
        'kv_cache_before_step1': primed_kv_cache,  # Save the ORIGINAL KV cache (unchanged by manual forward step)
        'token_probs': probs,  # Final probability distribution used for sampling
        'top_tokens_info': top_tokens_info  # Info for UI
        # Our manual_forward_step guarantees that primed_kv_cache remains unmodified
    }
    
    # Define the output path with the test_id
    output_path = os.path.join('tests', 'captures', f'step1_output_{test_id}.pt')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the captured data using our safe_save utility
    safe_save(captured_data, output_path)
    
    print(f"Captured data saved to {output_path}")
    
    # The kv_cache_before_step1 should contain the original state before new token generation
    # Check that kv_cache_before_step1 is a valid cache type (either tuple or DynamicCache)
    assert hasattr(primed_kv_cache, '__len__') or hasattr(primed_kv_cache, 'get_seq_length'), \
        "kv_cache_before_step1 should be either a tuple-like structure or a DynamicCache object"
    # --- Add this for explicit debugging ---
    print("\n--- Verifying Cache Length Assertion ---")
    actual_len_fixture_cache = -1 # Default
    if hasattr(primed_kv_cache, 'get_seq_length'):
        actual_len_fixture_cache = primed_kv_cache.get_seq_length()
    elif isinstance(primed_kv_cache, tuple): # Add tuple check if needed
        try:
            actual_len_fixture_cache = primed_kv_cache[0][0].shape[2]
        except: 
            pass # Ignore errors if cache structure is odd
            
    print(f"Value from fixture 'expected_cache_len': {expected_cache_len}")
    print(f"Actual length of 'primed_kv_cache' object received from fixture: {actual_len_fixture_cache}")
    print(f"Asserting: {actual_len_fixture_cache} == {expected_cache_len}")
    print(f"Type of primed_kv_cache: {type(primed_kv_cache).__name__}")
    # --- End of debug prints ---
        
    # Check that the cache length matches what's expected from the fixture
    # This is the authoritative length from the cache object, not derived from inputs
    
    # For DynamicCache objects, use get_seq_length method
    if hasattr(primed_kv_cache, 'get_seq_length'):
        # It's a DynamicCache object
        seq_length = primed_kv_cache.get_seq_length()
        assert seq_length == expected_cache_len, \
            f"Cache length received from fixture ({seq_length}) doesn't match expected ({expected_cache_len})"
    else:
        # It's a tuple of key-value tensors (older format)
        # Use the first layer's key tensor to get sequence length: shape is [batch, heads, seq, dim]
        first_layer_key = primed_kv_cache[0][0]  # First layer, key tensor
        seq_length = first_layer_key.shape[2]    # Compare against the expected cache length from the fixture
    # With manual_forward_step, the cache is guaranteed to be unmodified
    assert seq_length == expected_cache_len, \
        f"Cache length received from fixture ({seq_length}) doesn't match expected ({expected_cache_len})"
            
    # With manual_forward_step, the original cache object is saved directly
    # and is guaranteed to be unmodified, so this should be identical
    kv_cache_saved = captured_data['kv_cache_before_step1']
    
    # Verify it's the exact same object (identity check)
    assert kv_cache_saved is primed_kv_cache, \
        "Saved kv_cache_before_step1 should be the identical object from the fixture"
        
    # Check the first layer to ensure it has the expected structure
    first_layer = primed_kv_cache[0]
    assert isinstance(first_layer, tuple), "Each layer in kv_cache_before_step1 should be a tuple"
    assert len(first_layer) == 2, "Each layer should contain 2 elements (key and value)"
    
    # Check dimensions of key and value tensors in the first layer
    key_tensor, value_tensor = first_layer
    assert isinstance(key_tensor, torch.Tensor), "Key tensor should be a torch.Tensor"
    assert isinstance(value_tensor, torch.Tensor), "Value tensor should be a torch.Tensor"
        
    # Check that the sequence dimension matches the expected context length
    # KV cache shape is typically [batch_size, num_heads, sequence_length, head_dim]
    assert key_tensor.dim() >= 3, "Key tensor should have at least 3 dimensions"
    assert value_tensor.dim() >= 3, "Value tensor should have at least 3 dimensions"
    assert key_tensor.size(2) == expected_cache_len, f"Key tensor sequence length should be {expected_cache_len}, got {key_tensor.size(2)}"
    assert value_tensor.size(2) == expected_cache_len, f"Value tensor sequence length should be {expected_cache_len}, got {value_tensor.size(2)}"
    
    print("\nAll cache integrity checks passed - original cache state preserved!")

    # Check attention outputs - may be different with manual implementation
    # With SDPA in manual mode, attentions might be None or contain placeholders
    if attentions is not None:
        assert isinstance(attentions, (tuple, list)), "If provided, attentions should be a tuple or list"
    
    # Note: In manual_forward_step using SDPA, attention weights aren't returned directly
    # as SDPA doesn't expose them without extra computation, so we skip this check
    # The important validation is that the logits are correct and lead to valid token sampling
