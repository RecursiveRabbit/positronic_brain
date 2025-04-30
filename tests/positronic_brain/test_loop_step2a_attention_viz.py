"""
Test for Step 2a of the Positronic Brain loop: Attention Visualization
This test loads the output from Step 1 and visualizes the attention distribution
from the last generated token onto the preceding context tokens.

The test is now parameterized to handle different test cases from Step 1,
including both short and long contexts.
"""

import os
import pytest
import torch
import numpy as np
import textwrap

# Import our serialization utilities
from positronic_brain.serialization_utils import safe_load
from positronic_brain import config
from positronic_brain.model_io import load_model

# Reuse the fixtures from Step 1
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


# Define test cases for visualization
test_cases = [
    pytest.param("short_fox", id="visualize_short_fox"),
    pytest.param("resume_context", id="visualize_resume_context"),
    # Add more test cases as needed
]

@pytest.mark.parametrize("test_id", test_cases)
def test_visualize_attention_scores(
    loaded_models_and_tokenizer,
    load_captured_step1_output,
    test_id
):
    """
    Parameterized test that loads outputs from Step 1 and visualizes attention distributions.
    
    Args:
        loaded_models_and_tokenizer: Fixture containing model, tokenizer, and device
        load_captured_step1_output: Fixture containing the data captured in step 1 for the specific test_id
        test_id: Identifier for the test case to visualize
    """
    # Extract data from fixtures
    tokenizer = loaded_models_and_tokenizer['tokenizer']
    step1_data = load_captured_step1_output
    
    # Get the initial input_ids from the Step 1 data
    initial_input_ids = step1_data['initial_input_ids']
    initial_seq_len = initial_input_ids.shape[1]
    print(f"Context length: {initial_seq_len} tokens")
    
    # Get the attentions from the Step 1 output
    attentions = step1_data['attentions']
    
    # Extract the attention tensor from the last layer
    last_layer_attentions = attentions[-1]
    
    # Verify shape
    print(f"\nAttention tensor shape: {last_layer_attentions.shape}")
    batch_size, num_heads, query_seq_len, key_seq_len = last_layer_attentions.shape
    
    # Assert expected shape properties
    assert query_seq_len == 1, f"Expected query_seq_len to be 1, but got {query_seq_len}"
    
    # For the test cases with many tokens, get a subset of tokens to visualize in detail
    max_tokens_to_show = 50
    full_visualization = initial_seq_len <= max_tokens_to_show
    
    # The number of tokens to sample from the beginning and end of the sequence
    sample_size = max_tokens_to_show // 2 if not full_visualization else initial_seq_len
    
    # Select attention scores from the single query token to all key tokens
    # We want all attention scores for statistics
    attention_slice = last_layer_attentions[0, :, 0, :initial_seq_len]
    
    # Average the attention scores across all heads
    mean_attention_scores = attention_slice.mean(dim=0).cpu()
    assert mean_attention_scores.shape == (initial_seq_len,), f"Expected mean_attention_scores shape to be {(initial_seq_len,)}, but got {mean_attention_scores.shape}"
    
    # Generate visualization string
    print(f"\n--- Attention Visualization for '{test_id}' ---")
    
    # For long contexts, show a subset of tokens with the format:
    # [beginning tokens] ... [end tokens]
    if full_visualization:
        # Generate visualization for all tokens
        output_parts = []
        for i in range(initial_seq_len):
            token_id = initial_input_ids[0, i].item()
            token_text = tokenizer.decode([token_id])
            token_text = token_text.replace('\n', '\\n')
            score = mean_attention_scores[i].item()
            output_parts.append(f"{token_text}({score:.5f})")
        
        output_paragraph = " ".join(output_parts)
        print(textwrap.fill(output_paragraph, width=100))
    else:
        # Generate visualization for beginning tokens
        begin_parts = []
        for i in range(sample_size):
            token_id = initial_input_ids[0, i].item()
            token_text = tokenizer.decode([token_id])
            token_text = token_text.replace('\n', '\\n')
            score = mean_attention_scores[i].item()
            begin_parts.append(f"{token_text}({score:.5f})")
        
        # Generate visualization for end tokens
        end_parts = []
        for i in range(initial_seq_len - sample_size, initial_seq_len):
            token_id = initial_input_ids[0, i].item()
            token_text = tokenizer.decode([token_id])
            token_text = token_text.replace('\n', '\\n')
            score = mean_attention_scores[i].item()
            end_parts.append(f"{token_text}({score:.5f})")
        
        # Combine with ellipsis in the middle
        print("BEGINNING TOKENS:")
        print(textwrap.fill(" ".join(begin_parts), width=100))
        print("\n...\n")
        print("ENDING TOKENS:")
        print(textwrap.fill(" ".join(end_parts), width=100))
    
    print("-------------------------------------------")
    
    # Assertions
    assert len(mean_attention_scores) == initial_seq_len, f"Expected {initial_seq_len} attention scores, but got {len(mean_attention_scores)}"
    
    # Print the selected token
    selected_token_id = step1_data['selected_token_id']
    selected_token_text = tokenizer.decode([selected_token_id])
    print(f"\nSelected token after this context: '{selected_token_text}' (ID: {selected_token_id})")
    
    # Calculate and print statistics about the attention distribution
    min_attention = mean_attention_scores.min().item()
    max_attention = mean_attention_scores.max().item()
    avg_attention = mean_attention_scores.mean().item()
    std_attention = mean_attention_scores.std().item()
    
    print(f"\nAttention statistics for {initial_seq_len} tokens:")
    print(f"  Min attention: {min_attention:.5f}")
    print(f"  Max attention: {max_attention:.5f}")
    print(f"  Avg attention: {avg_attention:.5f}")
    print(f"  Std attention: {std_attention:.5f}")
    
    # Find token positions with highest attention
    top_k = 5
    top_k_values, top_k_indices = torch.topk(mean_attention_scores, k=min(top_k, initial_seq_len))
    
    print(f"\nTop {min(top_k, initial_seq_len)} tokens with highest attention:")
    for i, (idx, val) in enumerate(zip(top_k_indices.tolist(), top_k_values.tolist())):
        token_id = initial_input_ids[0, idx].item()
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1}. '{token_text}' (position {idx}, score: {val:.5f})")
    
    # Calculate attention distribution by token position
    # Divide the context into segments (beginning, middle, end)
    if initial_seq_len >= 3:
        segment_size = initial_seq_len // 3
        beginning_segment = mean_attention_scores[:segment_size]
        middle_segment = mean_attention_scores[segment_size:2*segment_size]
        end_segment = mean_attention_scores[2*segment_size:]
        
        beginning_avg = beginning_segment.mean().item()
        middle_avg = middle_segment.mean().item()
        end_avg = end_segment.mean().item()
        
        print(f"\nAttention by position segment:")
        print(f"  Beginning segment (tokens 0-{segment_size-1}): {beginning_avg:.5f}")
        print(f"  Middle segment (tokens {segment_size}-{2*segment_size-1}): {middle_avg:.5f}")
        print(f"  End segment (tokens {2*segment_size}-{initial_seq_len-1}): {end_avg:.5f}")
