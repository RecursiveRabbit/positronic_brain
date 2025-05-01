"""
Test for Step 3b: Select specific token positions to cull based on brightness.

This test:
1. Loads the cull decision data from Step 3a (cull_count)
2. Identifies the specific token positions with lowest brightness values
3. Saves the positions_to_cull list for use in subsequent steps
"""
import os
import pytest
import torch

from positronic_brain.serialization_utils import safe_load, safe_save

# Define test cases for cull candidate selection
test_cases = [
    pytest.param("short_fox", 13, id="select_cull_short_fox"),
    pytest.param("long_context_sample", 862, id="select_cull_long_context_sample"),
]

@pytest.fixture(scope="function")
def load_cull_decision_step3a_output(test_id):
    """
    Load the cull decision data saved by Step 3a.
    
    Args:
        test_id: Identifier for the test case to load
    
    Returns:
        dict: The cull decision data for the specified test_id
    """
    # Define the path to the cull decision data based on test_id
    output_path = os.path.join('tests', 'captures', f'step3a_cull_decision_{test_id}.pt')
    
    # Ensure the file exists
    assert os.path.exists(output_path), f"Step 3a output file not found at {output_path}"
    
    # Load the cull decision data
    decision_data = safe_load(output_path)
    
    # Ensure it contains the expected keys
    required_keys = ['cull_count', 'new_brightness_map']
    for key in required_keys:
        assert key in decision_data, f"Required key '{key}' not found in step 3a output data"
    
    print(f"Step 3a cull decision data for '{test_id}' loaded successfully from {output_path}")
    
    return decision_data

@pytest.mark.parametrize("test_id, expected_initial_tokens", test_cases)
def test_select_cull_candidates(
    load_cull_decision_step3a_output,
    test_id,
    expected_initial_tokens
):
    """
    Test the selection of specific token positions to cull based on brightness.
    
    This test identifies the tokens with lowest brightness values up to the
    cull_count determined in Step 3a.
    
    Args:
        load_cull_decision_step3a_output: Fixture containing the cull decision data
        test_id: Identifier for this test case
        expected_initial_tokens: Expected number of tokens in the initial context
    """
    # Get the decision data from Step 3a
    decision_data = load_cull_decision_step3a_output
    
    # Extract data (only using initial_seq_len, not current_context_size)
    initial_seq_len = decision_data['initial_seq_len']
    cull_count = decision_data['cull_count']
    new_brightness_map = decision_data['new_brightness_map']
    
    # Verify the sequence length
    assert initial_seq_len == expected_initial_tokens, \
        f"Expected {expected_initial_tokens} initial tokens, but got {initial_seq_len}"
    
    print(f"Initial sequence length: {initial_seq_len}")
    print(f"\nSelecting cull candidates with parameters:")
    print(f"  - Cull count: {cull_count}")
    print(f"  - Brightness map size: {new_brightness_map.shape}")
    
    # Initialize positions_to_cull as an empty list
    positions_to_cull = []
    
    # Only proceed if we need to cull tokens
    if cull_count > 0:
        # Ensure we have enough elements to cull
        if new_brightness_map.numel() >= cull_count:
            # Find the indices of the cull_count smallest values in the brightness map
            _, lowest_indices_tensor = torch.topk(new_brightness_map, k=cull_count, largest=False)
            positions_to_cull = lowest_indices_tensor.tolist()  # Convert tensor indices to list of ints
        else:
            # Handle edge case where context is smaller than cull_count (shouldn't happen with current rules)
            positions_to_cull = list(range(new_brightness_map.numel()))
            print(f"WARNING: Context size {new_brightness_map.numel()} is smaller than cull_count {cull_count}")
    
    print(f"Selected {len(positions_to_cull)} positions to cull: {positions_to_cull}")
    
    # For debugging, show the brightness values of the selected positions
    if positions_to_cull:
        print("\nBrightness values of selected positions:")
        for pos in positions_to_cull:
            brightness = new_brightness_map[pos].item()
            print(f"  Position {pos}: {brightness:.2f}")
            
        # Also check if these are indeed the lowest brightness values
        if len(positions_to_cull) < new_brightness_map.numel():
            # Create a mask for positions not selected for culling
            mask = torch.ones_like(new_brightness_map, dtype=torch.bool)
            mask[positions_to_cull] = False
            
            # Get the minimum brightness of tokens not being culled
            min_non_culled_brightness = new_brightness_map[mask].min().item()
            max_culled_brightness = new_brightness_map[positions_to_cull].max().item()
            
            print(f"\nVerification:")
            print(f"  Max brightness of culled tokens: {max_culled_brightness:.2f}")
            print(f"  Min brightness of non-culled tokens: {min_non_culled_brightness:.2f}")
            print(f"  All culled tokens have brightness <= non-culled tokens: {max_culled_brightness <= min_non_culled_brightness}")
    
    # Prepare the data to be saved
    cull_candidates_data = {
        'test_id': test_id,
        'cull_count': cull_count,
        'positions_to_cull': positions_to_cull,  # List of integer positions
        # Pass through necessary data for subsequent steps
        'initial_seq_len': decision_data['initial_seq_len'],
        'new_brightness_map': decision_data['new_brightness_map'],
        'initial_input_ids': decision_data['initial_input_ids'],
        'selected_token_id': decision_data['selected_token_id']
        # KV cache intentionally NOT passed through
    }
    
    # Define the output path
    output_path = os.path.join('tests', 'captures', f'step3b_cull_candidates_{test_id}.pt')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the cull candidates data
    safe_save(cull_candidates_data, output_path)
    
    print(f"\nCull candidates data saved to {output_path}")
    
    # Assertions
    assert isinstance(positions_to_cull, list), "positions_to_cull should be a list"
    assert len(positions_to_cull) == cull_count, \
        f"Expected {cull_count} positions to cull, but got {len(positions_to_cull)}"
    
    # If we're culling tokens, verify our selection has the lowest brightness values
    if cull_count > 0 and new_brightness_map.numel() > cull_count:
        # Get the maximum brightness among the culled tokens
        culled_brightness = new_brightness_map[positions_to_cull]
        max_culled_brightness = culled_brightness.max().item()
        
        # Create a mask for positions not selected for culling
        mask = torch.ones_like(new_brightness_map, dtype=torch.bool)
        mask[positions_to_cull] = False
        
        # Get the minimum brightness of tokens not being culled
        non_culled_brightness = new_brightness_map[mask]
        min_non_culled_brightness = non_culled_brightness.min().item()
        
        # The max brightness among culled tokens should be less than or equal to
        # the min brightness among non-culled tokens (allowing for ties)
        assert max_culled_brightness <= min_non_culled_brightness, \
            f"Found non-culled token with brightness {min_non_culled_brightness} lower than culled token with brightness {max_culled_brightness}"
    
    # Verify the output file was created
    assert os.path.exists(output_path), f"Output file not created at {output_path}"
