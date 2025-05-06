"""
Test for Step 3b: Select adjacent token pairs to cull based on brightness.

This test:
1. Loads the cull decision data from Step 3a (pair_cull_count)
2. Identifies adjacent token pairs with lowest maximum brightness values
3. Saves the selected_pairs list for use in subsequent steps
"""
import os
import pytest
import torch

from .test_utils import safe_load, safe_save

# Using a fixed initial prompt from session-scoped fixture

@pytest.fixture(scope="function")
def load_cull_decision_step3a_output():
    """
    Load the cull decision data saved by Step 3a.
    
    Returns:
        dict: The cull decision data with pair_cull_count
    """
    # Define the path to the cull decision data using fixed identifier
    output_path = os.path.join('tests', 'captures', 'step3a_cull_decision_fixed_initial.pt')
    
    # Ensure the file exists
    assert os.path.exists(output_path), f"Step 3a output file not found at {output_path}"
    
    # Load the cull decision data
    decision_data = safe_load(output_path)
    
    # Ensure it contains the expected keys
    required_keys = ['cull_count', 'new_brightness_map']
    for key in required_keys:
        assert key in decision_data, f"Required key '{key}' not found in step 3a output data"
    
    # Rename cull_count to pair_cull_count for clarity in the new pair-based approach
    # Note: This doesn't change the file, just renames the variable in this test
    decision_data['pair_cull_count'] = decision_data['cull_count']
    
    print(f"Step 3a cull decision data loaded successfully from {output_path}")
    print(f"Using cull_count as pair_cull_count: {decision_data['pair_cull_count']}")
    
    return decision_data

def test_select_cull_candidates(
    load_cull_decision_step3a_output
):
    """
    Test the selection of adjacent token pairs to cull based on brightness.
    
    This test identifies adjacent token pairs with lowest maximum brightness values
    up to the pair_cull_count determined in Step 3a.
    
    Args:
        load_cull_decision_step3a_output: Fixture containing the cull decision data
    """
    # Get the decision data from Step 3a
    decision_data = load_cull_decision_step3a_output
    
    # Extract data
    initial_seq_len = decision_data['initial_seq_len']
    pair_cull_count = decision_data['pair_cull_count']
    new_brightness_map = decision_data['new_brightness_map']
    
    print(f"Initial sequence length: {initial_seq_len}")
    print(f"\nSelecting adjacent token pairs for culling with parameters:")
    print(f"  - Pair cull count: {pair_cull_count}")
    print(f"  - Brightness map size: {new_brightness_map.shape}")
    
    # Initialize selected_pairs as an empty list
    selected_pairs = []
    
    # Only proceed if we need to cull token pairs
    if pair_cull_count > 0:
        # Calculate pair metrics for all possible adjacent token pairs
        pair_candidates = []
        
        # Iterate through all starting positions for adjacent pairs (0 to initial_seq_len-2)
        # Calculate the metric for each pair: max(brightness[pos], brightness[pos+1])
        for pos in range(initial_seq_len - 1):
            first_token_brightness = new_brightness_map[pos].item()
            second_token_brightness = new_brightness_map[pos + 1].item()
            
            # Use the maximum brightness of the two tokens as the pair metric
            # Lower metric = better candidate for culling
            pair_metric = max(first_token_brightness, second_token_brightness)
            
            # Store the metric and starting position
            pair_candidates.append((pair_metric, pos))
        
        # Sort pairs by their metric (ascending order - lowest max brightness first)
        pair_candidates.sort(key=lambda x: x[0])
        
        # Select the top pair_cull_count pairs with lowest maximum brightness
        for i in range(min(pair_cull_count, len(pair_candidates))):
            metric, pos = pair_candidates[i]
            selected_pairs.append((pos, pos + 1))  # Store as (start_pos, start_pos + 1)
    
    print(f"Selected {len(selected_pairs)} pairs to cull: {selected_pairs}")
    
    # For debugging, show the maximum brightness values of the selected pairs
    if selected_pairs:
        print("\nBrightness values of selected pairs:")
        for pair in selected_pairs:
            first_pos, second_pos = pair
            first_brightness = new_brightness_map[first_pos].item()
            second_brightness = new_brightness_map[second_pos].item()
            max_brightness = max(first_brightness, second_brightness)
            
            print(f"  Pair {pair}: [{first_brightness:.2f}, {second_brightness:.2f}], max = {max_brightness:.2f}")
        
        # Verify these are indeed the pairs with lowest maximum brightness
        if len(pair_candidates) > pair_cull_count:
            # Get the maximum brightness of the selected pairs
            selected_max_brightness = pair_candidates[pair_cull_count - 1][0] if pair_cull_count > 0 else 0
            
            # Get the minimum brightness of the non-selected pairs
            non_selected_min_brightness = pair_candidates[pair_cull_count][0] if pair_cull_count < len(pair_candidates) else float('inf')
            
            print(f"\nVerification:")
            print(f"  Max brightness of selected pairs: {selected_max_brightness:.2f}")
            print(f"  Min brightness of non-selected pairs: {non_selected_min_brightness:.2f}")
            print(f"  All selected pairs have max brightness <= non-selected pairs: {selected_max_brightness <= non_selected_min_brightness}")
    
    # Prepare the data to be saved
    culled_pairs_data = {
        'test_id': 'fixed_initial',  # Hardcoded to match our fixed identifier
        'pair_cull_count': pair_cull_count,
        'selected_pairs': selected_pairs,  # List of tuples (pos, pos+1)
        # Pass through necessary data for subsequent steps
        'initial_seq_len': decision_data['initial_seq_len'],
        'new_brightness_map': decision_data['new_brightness_map'],
        'initial_input_ids': decision_data['initial_input_ids'],
        'selected_token_id': decision_data['selected_token_id']
        # KV cache intentionally NOT passed through
    }
    
    # Define the output path
    output_path = os.path.join('tests', 'captures', 'step3b_culled_pairs_fixed_initial.pt')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the culled pairs data
    safe_save(culled_pairs_data, output_path)
    
    print(f"\nCulled pairs data saved to {output_path}")
    
    # Assertions
    assert isinstance(selected_pairs, list), "selected_pairs should be a list"
    assert len(selected_pairs) == pair_cull_count, \
        f"Expected {pair_cull_count} pairs to cull, but got {len(selected_pairs)}"
    
    # Check that each pair consists of adjacent tokens
    for pair in selected_pairs:
        assert len(pair) == 2, f"Each pair should have exactly 2 elements, but got {len(pair)}"
        first_pos, second_pos = pair
        assert second_pos == first_pos + 1, f"Expected adjacent positions, but got {first_pos} and {second_pos}"
    
    # If we're culling pairs, verify our selection has the lowest maximum brightness values
    if pair_cull_count > 0 and (initial_seq_len - 1) > pair_cull_count:
        # Calculate the maximum brightness for each selected pair
        selected_pair_metrics = []
        for first_pos, second_pos in selected_pairs:
            first_brightness = new_brightness_map[first_pos].item()
            second_brightness = new_brightness_map[second_pos].item()
            selected_pair_metrics.append(max(first_brightness, second_brightness))
        
        # Get the maximum metric among the selected pairs
        max_selected_pair_metric = max(selected_pair_metrics) if selected_pair_metrics else 0
        
        # Calculate metrics for all non-selected pairs
        non_selected_pair_metrics = []
        for pos in range(initial_seq_len - 1):
            # Skip if this position is the start of a selected pair
            if any(pair[0] == pos for pair in selected_pairs):
                continue
            
            first_brightness = new_brightness_map[pos].item()
            second_brightness = new_brightness_map[pos + 1].item()
            non_selected_pair_metrics.append(max(first_brightness, second_brightness))
        
        # Get the minimum metric among the non-selected pairs
        min_non_selected_pair_metric = min(non_selected_pair_metrics) if non_selected_pair_metrics else float('inf')
        
        # The max metric among selected pairs should be less than or equal to
        # the min metric among non-selected pairs (allowing for ties)
        assert max_selected_pair_metric <= min_non_selected_pair_metric, \
            f"Found non-selected pair with metric {min_non_selected_pair_metric} lower than selected pair with metric {max_selected_pair_metric}"
    
    # Verify the output file was created
    assert os.path.exists(output_path), f"Output file not created at {output_path}"
