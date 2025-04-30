"""
Test Step 3c: Prepare Diffuser Inputs

This test is responsible for extracting context windows around each token position
that has been selected for culling in Step 3b. The resulting windows will be used
as inputs to the diffuser in Step 3d.

Each window contains:
- Token IDs for the context window
- Brightness map for the context window
- Attention mask for the context window
- The local position of the token to be culled within the window
- The original token ID at the masked position

The windows are centered on the tokens to be culled, with a configurable window size.
"""

import os
import pytest
import torch
import logging

# Add safe loading utility
from positronic_brain.serialization_utils import safe_load

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test IDs for parameterization
TEST_IDS = ['short_fox', 'resume_context']

@pytest.fixture(scope="function", params=TEST_IDS)
def load_cull_candidates_step3b_output(request):
    """
    Load the cull candidates data saved by Step 3b.
    
    Args:
        request: Pytest request object containing the test_id parameter
    
    Returns:
        dict: The cull candidates data for the specified test_id
    """
    test_id = request.param
    
    # Define the path to the cull candidates data
    output_path = os.path.join('tests', 'captures', f'step3b_cull_candidates_{test_id}.pt')
    
    # Ensure the file exists
    assert os.path.exists(output_path), f"Step 3b output file not found at {output_path}"
    
    # Load the cull candidates data
    cull_candidates_data = safe_load(output_path)
    
    # Add the test_id for reference
    cull_candidates_data['test_id'] = test_id
    
    # Set expected token counts based on test_id
    if 'short_fox' in test_id:
        expected_initial_tokens = 13  # The fox prompt has 13 tokens
    else:  # 'resume_context'
        expected_initial_tokens = 1018  # The resume context has ~1018 tokens
    
    # Ensure the data contains expected keys
    required_keys = ['positions_to_cull', 'initial_seq_len', 'new_brightness_map', 'initial_input_ids']
    for key in required_keys:
        assert key in cull_candidates_data, f"Required key '{key}' not found in step 3b output data"
    
    # Verify initial_seq_len matches expectations
    assert cull_candidates_data['initial_seq_len'] == expected_initial_tokens, \
        f"Expected {expected_initial_tokens} tokens, but got {cull_candidates_data['initial_seq_len']}"
    
    return cull_candidates_data

@pytest.fixture(scope="function")
def diffuser_config():
    """
    Provide diffuser-related configuration parameters.
    
    Returns:
        dict: Configuration dictionary for the diffuser
    """
    # Use a 255-token window on each side, plus the culled token (511 total)
    half_window = 255
    window_size = half_window * 2 + 1
    
    return {
        'window_size': window_size,
        'half_window': half_window
    }

def test_prepare_diffuser_inputs(load_cull_candidates_step3b_output, diffuser_config):
    """
    Test Step 3c: Prepare Diffuser Inputs
    
    This test prepares input data for the diffuser by extracting context windows
    around each token that will be culled.
    
    Args:
        load_cull_candidates_step3b_output: Fixture providing the Step 3b output data
        diffuser_config: Fixture providing diffuser configuration parameters
    """
    # Extract data from the fixtures
    cull_candidates_data = load_cull_candidates_step3b_output
    test_id = cull_candidates_data['test_id']
    positions_to_cull = cull_candidates_data['positions_to_cull']
    initial_input_ids = cull_candidates_data['initial_input_ids']
    new_brightness_map = cull_candidates_data['new_brightness_map']
    initial_seq_len = cull_candidates_data['initial_seq_len']
    
    # Extract diffuser configuration
    window_size = diffuser_config['window_size']
    half_window = diffuser_config['half_window']
    
    # Get device from tensors
    device = initial_input_ids.device
    
    # Print test information
    print(f"\nPreparing diffuser inputs for test: {test_id}")
    print(f"Initial sequence length: {initial_seq_len}")
    print(f"Positions to cull: {positions_to_cull}")
    print(f"Window size: {window_size} (half_window: {half_window})")
    
    # Initialize the list to store diffuser inputs
    diffuser_input_list = []
    
    # Process each position to cull
    for global_cull_position in positions_to_cull:
        print(f"\nProcessing cull position: {global_cull_position}")
        
        # Calculate window boundaries
        start_pos = max(0, global_cull_position - half_window)
        end_pos = min(initial_seq_len, global_cull_position + half_window + 1)
        actual_window_len = end_pos - start_pos
        
        print(f"Window boundaries: [{start_pos}, {end_pos}) (length: {actual_window_len})")
        
        # Slice data to extract the window
        window_token_ids = initial_input_ids[0, start_pos:end_pos]
        window_brightness_map = new_brightness_map[start_pos:end_pos]
        
        # Create attention mask (all ones for now)
        window_attention_mask = torch.ones_like(window_token_ids)
        
        # Calculate the local position of the masked token within the window
        masked_position_local = global_cull_position - start_pos
        
        # Get original token ID at masked position
        original_token_id_at_mask = window_token_ids[masked_position_local].item()
        
        print(f"Masked position (local): {masked_position_local}")
        print(f"Original token ID at mask: {original_token_id_at_mask}")
        
        # Create input dictionary for this window
        window_input_data = {
            'global_cull_position': global_cull_position,
            'window_token_ids': window_token_ids.cpu(),  # Ensure CPU for saving
            'window_attention_mask': window_attention_mask.cpu(),
            'window_brightness_map': window_brightness_map.cpu(),
            'masked_position_local': masked_position_local,
            'original_token_id_at_mask': original_token_id_at_mask
        }
        
        # Add to the list
        diffuser_input_list.append(window_input_data)
    
    print(f"\nPrepared {len(diffuser_input_list)} diffuser input windows")
    
    # Prepare the data to be saved
    prepared_data = {
        'test_id': test_id,
        'diffuser_input_list': diffuser_input_list,  # List of window data dictionaries
        # Pass through necessary data from previous steps
        'initial_seq_len': initial_seq_len,
        'initial_input_ids': initial_input_ids.cpu(),  # Ensure CPU
        'positions_to_cull': positions_to_cull,
        'selected_token_id': cull_candidates_data['selected_token_id']
        # KV cache intentionally NOT passed through
    }
    
    # Define the output path
    output_path = os.path.join('tests', 'captures', f'step3c_diffuser_inputs_{test_id}.pt')
    
    # Save the prepared data
    torch.save(prepared_data, output_path)
    print(f"Saved diffuser inputs to: {output_path}")
    
    # Assertions
    
    # Check that we have the right number of diffuser inputs
    assert len(diffuser_input_list) == len(positions_to_cull), \
        f"Expected {len(positions_to_cull)} diffuser inputs, but got {len(diffuser_input_list)}"
    
    # If we have any diffuser inputs, check the structure
    if diffuser_input_list:
        first_input = diffuser_input_list[0]
        
        # Check keys
        required_keys = ['window_token_ids', 'window_attention_mask', 'window_brightness_map', 
                         'masked_position_local', 'original_token_id_at_mask', 'global_cull_position']
        for key in required_keys:
            assert key in first_input, f"Required key '{key}' not found in diffuser input"
        
        # Check tensor shapes
        actual_window_len = first_input['window_token_ids'].shape[0]
        assert first_input['window_attention_mask'].shape[0] == actual_window_len, \
            "Attention mask shape doesn't match token IDs shape"
        assert first_input['window_brightness_map'].shape[0] == actual_window_len, \
            "Brightness map shape doesn't match token IDs shape"
        
        # Check masked position is within bounds
        assert 0 <= first_input['masked_position_local'] < actual_window_len, \
            f"Masked position {first_input['masked_position_local']} is out of bounds [0, {actual_window_len-1}]"
        
        # Check original token ID matches
        assert first_input['window_token_ids'][first_input['masked_position_local']].item() == first_input['original_token_id_at_mask'], \
            "Original token ID doesn't match token ID at masked position"
    
    # Check output file was created
    assert os.path.exists(output_path), f"Output file not created at {output_path}"
