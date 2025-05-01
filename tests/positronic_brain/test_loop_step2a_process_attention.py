"""
Test for Step 2a: Processing attention scores from the raw attention output.

This test:
1. Loads the data from Step 1 test
2. Processes the raw attention tensor to extract averaged attention scores
3. Saves the processed scores for use in Step 2b
"""
import os
import pytest
import torch
import numpy as np
import textwrap

from positronic_brain import config
from positronic_brain.model_io import load_model
from positronic_brain.serialization_utils import safe_load, safe_save

# Define test cases for attention processing
test_cases = [
    pytest.param("short_fox", 13, id="process_short_fox"),
    pytest.param("long_context_sample", 862, id="process_long_context_sample"),
]

@pytest.fixture(scope="session")
def loaded_models_and_tokenizer():
    """
    Load the LLM model and tokenizer.
    
    Returns:
        dict: Contains model, tokenizer, and device objects
    """
    print(f"\nLoading model {config.MODEL_NAME}...")
    model, tokenizer = load_model(
        model_name=config.MODEL_NAME, 
        trust_remote_code=config.TRUST_REMOTE_CODE
    )
    device = next(model.parameters()).device
    print(f"Model loaded successfully on {device}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'device': device
    }

@pytest.fixture(scope="function")
def load_captured_step1_output(loaded_models_and_tokenizer, test_id):
    """
    Load the data saved by test_generate_first_step from step 1 based on test_id.
    
    Args:
        loaded_models_and_tokenizer: Fixture containing model, tokenizer and device
        test_id: Identifier for the test case to load
    
    Returns:
        dict: The data captured in step 1 for the specified test_id
    """
    device = loaded_models_and_tokenizer['device']
    
    # Define the path to the captured data based on test_id
    output_path = os.path.join('tests', 'captures', f'step1_output_{test_id}.pt')
    
    # Ensure the file exists
    assert os.path.exists(output_path), f"Step 1 output file not found at {output_path}"
    
    # Load the captured data using our safe_load utility
    step1_data = safe_load(output_path)
    
    # Ensure it contains the expected keys
    required_keys = ['initial_input_ids', 'attentions', 'selected_token_id']
    for key in required_keys:
        assert key in step1_data, f"Required key '{key}' not found in step 1 output data"
    
    print(f"Step 1 output data for '{test_id}' loaded successfully from {output_path}")
    
    return step1_data

@pytest.mark.parametrize("test_id, expected_initial_tokens", test_cases)
def test_process_attention_scores(
    loaded_models_and_tokenizer,
    load_captured_step1_output,
    test_id,
    expected_initial_tokens
):
    """
    Process the raw attention tensor from Step 1 into averaged attention scores.
    
    Args:
        loaded_models_and_tokenizer: Fixture containing model, tokenizer and device
        test_id: Identifier for the test case to load
        expected_initial_tokens: Expected number of tokens in the initial context
    """
    # Get components from fixtures
    tokenizer = loaded_models_and_tokenizer['tokenizer']
    device = loaded_models_and_tokenizer['device']
    step1_data = load_captured_step1_output  # Use the fixture directly
    
    # Extract data from Step 1
    initial_input_ids = step1_data['initial_input_ids']
    attentions = step1_data['attentions']
    selected_token_id = step1_data['selected_token_id']
    
    # Get initial_seq_len from the Step 1 data (instead of calculating again)
    initial_seq_len = step1_data['initial_seq_len']
    assert initial_seq_len == expected_initial_tokens, \
        f"Expected {expected_initial_tokens} tokens, but got {initial_seq_len}"
    print(f"Initial context contains {initial_seq_len} tokens as expected")
    
    # No longer calculating current_context_size - we're only working with the initial context
    print(f"Working with initial context of size: {initial_seq_len}")
    
    # Process attention: extract the last layer's attention
    assert attentions is not None, "Attentions tensor is None"
    
    # Extract the last layer's attention
    # The attentions structure depends on the model, but we expect it to be the last layer's output
    # For Transformer models, it's typically the last element
    if isinstance(attentions, tuple) or isinstance(attentions, list):
        print(f"Attention tensor is a {type(attentions)} with {len(attentions)} elements")
        last_layer_attentions = attentions[-1]
    else:
        # If it's already a tensor, use it directly
        print(f"Attention tensor is a single tensor of shape {attentions.shape}")
        last_layer_attentions = attentions
    
    # Verify the shape of attention tensor
    print(f"Last layer attention shape: {last_layer_attentions.shape}")
    
    # Expected shape: [batch_size, num_heads, query_seq_len, key_seq_len]
    # Since we're interested in the last token's attention to all previous tokens:
    batch_size, num_heads, query_seq_len, key_seq_len = last_layer_attentions.shape
    
    # The key_seq_len should be initial_seq_len
    assert key_seq_len >= initial_seq_len, \
        f"Expected attention key sequence length to be at least {initial_seq_len}, but got {key_seq_len}"
    
    # Select the attention scores from the single query token to the initial context tokens
    # Shape: [batch_size, num_heads, 1, initial_seq_len]
    attention_slice = last_layer_attentions[:, :, 0, :initial_seq_len]
    
    # Average the attention scores across all heads
    # Shape: [batch_size, initial_seq_len]
    mean_attention_scores = attention_slice.mean(dim=1).cpu()
    
    # Squeeze the batch dimension if it's 1
    if mean_attention_scores.shape[0] == 1:
        mean_attention_scores = mean_attention_scores.squeeze(0)
    
    # Ensure mean_attention_scores is on CPU for easier handling
    mean_attention_scores = mean_attention_scores.cpu()
    
    # Print attention statistics
    print(f"\nAttention statistics for {initial_seq_len} tokens:")
    print(f"  Min attention: {mean_attention_scores.min().item():.5f}")
    print(f"  Max attention: {mean_attention_scores.max().item():.5f}")
    print(f"  Avg attention: {mean_attention_scores.mean().item():.5f}")
    print(f"  Std attention: {mean_attention_scores.std().item():.5f}")
    
    # Find the top tokens with highest attention
    top_k = min(5, initial_seq_len)
    top_indices = torch.topk(mean_attention_scores, top_k).indices.tolist()
    top_values = torch.topk(mean_attention_scores, top_k).values.tolist()
    
    print(f"\nTop {top_k} tokens with highest attention:")
    for i, (idx, val) in enumerate(zip(top_indices, top_values)):
        token_id = initial_input_ids[0, idx].item()
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1}. '{token_text}' (position {idx}, score: {val:.5f})")
    
    # Visualize the attention for short contexts or beginning/end of long contexts
    if initial_seq_len <= 50:
        # For short contexts, show all tokens
        print("\n--- Processed Attention Scores ---")
        tokens = [tokenizer.decode([id.item()]) for id in initial_input_ids[0]]
        formatted_output = ""
        for i, (token, score) in enumerate(zip(tokens, mean_attention_scores)):
            formatted_output += f"{token}({score:.5f}) "
            if i > 0 and (i % 8 == 0 or i == initial_seq_len - 1):
                formatted_output += "\n"
        print(formatted_output)
    else:
        # For long contexts, show beginning and end
        print("\n--- Processed Attention Scores (Beginning) ---")
        start_tokens = [tokenizer.decode([id.item()]) for id in initial_input_ids[0, :25]]
        formatted_output = ""
        for i, (token, score) in enumerate(zip(start_tokens, mean_attention_scores[:25])):
            formatted_output += f"{token}({score:.5f}) "
            if i > 0 and (i % 8 == 0 or i == 24):
                formatted_output += "\n"
        print(formatted_output)
        
        print("\n--- Processed Attention Scores (Ending) ---")
        end_tokens = [tokenizer.decode([id.item()]) for id in initial_input_ids[0, -25:]]
        formatted_output = ""
        for i, (token, score) in enumerate(zip(end_tokens, mean_attention_scores[-25:])):
            formatted_output += f"{token}({score:.5f}) "
            if i > 0 and (i % 8 == 0 or i == 24):
                formatted_output += "\n"
        print(formatted_output)
    
    # Prepare output data for Step 2b
    processed_data = {
        'test_id': test_id,
        'processed_attention_scores': mean_attention_scores,  # Shape [initial_seq_len]
        # Pass sequence length information (only passing initial_seq_len, not calculating current_context_size)
        'initial_seq_len': initial_seq_len,
        # Pass through necessary data for Step 2b and subsequent steps
        'initial_input_ids': step1_data['initial_input_ids'],
        'selected_token_id': step1_data['selected_token_id'],
        'logits': step1_data.get('logits', None)  # Keep just in case, use get() to handle possible absence
        # KV cache intentionally NOT passed through
    }
    
    # Define the output path for Step 2a
    output_path = os.path.join('tests', 'captures', f'step2a_processed_attention_{test_id}.pt')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed data
    safe_save(processed_data, output_path)
    
    print(f"Processed attention data saved to {output_path}")
    
    # Final assertions
    assert isinstance(mean_attention_scores, torch.Tensor), "Processed attention scores should be a tensor"
    assert mean_attention_scores.shape[0] == initial_seq_len, \
        f"Expected processed attention scores shape to be [{initial_seq_len}], but got {mean_attention_scores.shape}"
