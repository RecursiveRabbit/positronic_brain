"""
Test for Step 3a: Decide cull count based on context size.

This test:
1. Loads the brightness map data from Step 2b
2. Uses the current_context_size and target_size to determine how many tokens to cull
3. Saves the cull decision for use in Step 3b
"""
import os
import pytest
import torch

from positronic_brain import config
from positronic_brain.serialization_utils import safe_load, safe_save

# Define test cases for cull count calculation
test_cases = [
    pytest.param("short_fox", 13, id="cull_short_fox"),
    pytest.param("resume_context", 1018, id="cull_resume_context"),
]

@pytest.fixture(scope="session")
def culling_config():
    """
    Return a dictionary containing the relevant culling parameters.
    
    Returns:
        dict: Contains culling-related configuration values
    """
    return {
        'target_size': getattr(config, 'TARGET_CONTEXT_SIZE', 1024),  # Default to 1024 if not defined
        'max_cull_per_step': getattr(config, 'MAX_CULL_PER_STEP', 2)  # Default to 2 if not defined
    }

@pytest.fixture(scope="function")
def load_calculated_step2b_output(test_id):
    """
    Load the brightness map data saved by Step 2b.
    
    Args:
        test_id: Identifier for the test case to load
    
    Returns:
        dict: The brightness map data for the specified test_id
    """
    # Define the path to the brightness map data based on test_id
    output_path = os.path.join('tests', 'captures', f'step2b_brightness_map_{test_id}.pt')
    
    # Ensure the file exists
    assert os.path.exists(output_path), f"Step 2b output file not found at {output_path}"
    
    # Load the brightness map data
    calculated_data = safe_load(output_path)
    
    # Ensure it contains the expected keys (no longer checking for current_context_size)
    required_keys = ['new_brightness_map', 'initial_seq_len']
    for key in required_keys:
        assert key in calculated_data, f"Required key '{key}' not found in step 2b output data"
    
    print(f"Step 2b brightness map data for '{test_id}' loaded successfully from {output_path}")
    
    return calculated_data

@pytest.mark.parametrize("test_id, expected_initial_tokens", test_cases)
def test_decide_cull_count(
    load_calculated_step2b_output,
    culling_config,
    test_id,
    expected_initial_tokens
):
    """
    Test the culling decision logic based on context size vs. target size.
    
    This test decides how many tokens to cull based on the current context size
    and the target size configuration.
    
    Args:
        load_calculated_step2b_output: Fixture containing the brightness map data
        culling_config: Fixture containing culling-related configuration
        test_id: Identifier for this test case
        expected_initial_tokens: Expected number of tokens in the initial context
    """
    # Get the calculated data
    calculated_data = load_calculated_step2b_output
    
    # Extract necessary information
    initial_seq_len = calculated_data['initial_seq_len']
    target_size = culling_config['target_size']
    
    print(f"Initial sequence length: {initial_seq_len}")
    print(f"Target size: {target_size}")
    
    # Verify initial_seq_len matches expectations
    assert initial_seq_len == expected_initial_tokens, \
        f"Expected {expected_initial_tokens} initial tokens, but got {initial_seq_len}"
    
    # Get the target size and max cull parameters
    target_size = culling_config['target_size']
    max_cull_per_step = culling_config['max_cull_per_step']
    
    print(f"\nDeciding cull count with parameters:")
    print(f"  - Initial sequence length: {initial_seq_len}")
    print(f"  - Target context size: {target_size}")
    print(f"  - Max cull per step: {max_cull_per_step}")
    
    # Simplified culling rule based only on initial_seq_len
    if initial_seq_len < target_size:
        # Context size still below target, don't cull
        cull_count = 0
    elif initial_seq_len == target_size:
        # At target size, cull one to make room for the new token
        cull_count = 1
    else:  # initial_seq_len > target_size
        # Always cull exactly 2 tokens if we're above the target
        cull_count = 2
    
    print(f"Cull decision: {cull_count} tokens")
    
    # Prepare the data to be saved
    decision_data = {
        'test_id': test_id,
        'cull_count': cull_count,  # Number of tokens to cull
        # Pass through size information (only initial_seq_len)
        'initial_seq_len': initial_seq_len,
        # Pass through all necessary data for subsequent steps
        'new_brightness_map': calculated_data['new_brightness_map'],
        'initial_input_ids': calculated_data['initial_input_ids'],
        'selected_token_id': calculated_data['selected_token_id'],
        'processed_attention_scores': calculated_data['processed_attention_scores']
        # KV cache intentionally NOT passed through
    }
    
    # Define the output path
    output_path = os.path.join('tests', 'captures', f'step3a_cull_decision_{test_id}.pt')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the decision data
    safe_save(decision_data, output_path)
    
    print(f"\nCull decision data saved to {output_path}")
    
    # Assertions
    assert cull_count in [0, 1, 2], f"Cull count {cull_count} should be 0, 1, or 2"
    
    # Assertions based on initial context size
    if initial_seq_len < target_size:
        assert cull_count == 0, f"Expected cull_count 0 for initial_seq_len {initial_seq_len} < {target_size}, got {cull_count}"
    elif initial_seq_len == target_size:
        assert cull_count == 1, f"Expected cull_count 1 for initial_seq_len {initial_seq_len} == {target_size}, got {cull_count}"
    else:  # initial_seq_len > target_size
        assert cull_count == 2, f"Expected cull_count 2 for initial_seq_len {initial_seq_len} > {target_size}, got {cull_count}"
