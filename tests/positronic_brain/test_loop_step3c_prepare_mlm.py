"""
Test Step 3c: Prepare MLM Inputs via Token-by-Token Text Translation

This test prepares inputs for a Masked Language Model (MLM) by translating TinyLlama tokens
to MLM-compatible inputs through a token-by-token text translation process.

For each position to cull, this test:
1. Extracts a window of TinyLlama token IDs around the position
2. Decodes each TinyLlama token individually to preserve boundaries
3. Replaces the token at the cull position with an MLM mask token
4. Re-tokenizes the resulting string with the MLM tokenizer
5. Finds the index of the mask token in the MLM sequence
6. Packages all necessary information for the diffuser in Step 3d

This approach ensures robust handling of different tokenizer vocabularies while maintaining
the mapping between the original global position and the masked token in the MLM input.
"""

import os
import pytest
import torch
import logging
from transformers import AutoTokenizer

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
    required_keys = ['positions_to_cull', 'initial_seq_len', 'initial_input_ids']
    for key in required_keys:
        assert key in cull_candidates_data, f"Required key '{key}' not found in step 3b output data"
    
    # Verify initial_seq_len matches expectations
    assert cull_candidates_data['initial_seq_len'] == expected_initial_tokens, \
        f"Expected {expected_initial_tokens} tokens, but got {cull_candidates_data['initial_seq_len']}"
    
    return cull_candidates_data

@pytest.fixture(scope="session")
def tinyllama_tokenizer():
    """
    Load the TinyLlama tokenizer.
    
    Returns:
        transformers.PreTrainedTokenizer: The TinyLlama tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        use_fast=True
    )
    
    # Ensure we have the padding token configured correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

@pytest.fixture(scope="session")
def mlm_tokenizer():
    """
    Load the MLM tokenizer.
    
    Returns:
        transformers.PreTrainedTokenizer: The BERT-based MLM tokenizer
    """
    # Use BERT tokenizer as a representative MLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        use_fast=True
    )
    
    return tokenizer

@pytest.fixture(scope="function")
def diffuser_window_config():
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

def test_prepare_mlm_inputs(load_cull_candidates_step3b_output, tinyllama_tokenizer, mlm_tokenizer, diffuser_window_config):
    """
    Test Step 3c: Prepare MLM Inputs via Token-by-Token Text Translation
    
    This test prepares input data for the MLM by extracting context windows around each token
    to be culled and translating them to MLM-compatible inputs through text.
    
    Args:
        load_cull_candidates_step3b_output: Fixture providing the Step 3b output data
        tinyllama_tokenizer: The TinyLlama tokenizer
        mlm_tokenizer: The MLM tokenizer
        diffuser_window_config: Fixture providing diffuser configuration parameters
    """
    # Extract data from the fixtures
    cull_candidates_data = load_cull_candidates_step3b_output
    test_id = cull_candidates_data['test_id']
    positions_to_cull = cull_candidates_data['positions_to_cull']
    initial_input_ids = cull_candidates_data['initial_input_ids']
    initial_seq_len = cull_candidates_data['initial_seq_len']
    
    # Extract diffuser configuration
    window_size = diffuser_window_config['window_size']
    half_window = diffuser_window_config['half_window']
    
    # Get device from tensors
    device = initial_input_ids.device
    
    # Print test information
    print(f"\nPreparing MLM inputs for test: {test_id}")
    print(f"Initial sequence length: {initial_seq_len}")
    print(f"Positions to cull: {positions_to_cull}")
    print(f"Window size: {window_size} (half_window: {half_window})")
    print(f"TinyLlama tokenizer vocabulary size: {len(tinyllama_tokenizer)}")
    print(f"MLM tokenizer vocabulary size: {len(mlm_tokenizer)}")
    print(f"MLM mask token: {mlm_tokenizer.mask_token}")
    print(f"MLM mask token ID: {mlm_tokenizer.mask_token_id}")
    
    # Initialize the list to store MLM inputs
    mlm_input_list = []
    
    # Process each position to cull
    for global_cull_position in positions_to_cull:
        print(f"\nProcessing cull position: {global_cull_position}")
        
        # Calculate window boundaries
        start_pos = max(0, global_cull_position - half_window)
        end_pos = min(initial_seq_len, global_cull_position + half_window + 1)
        actual_window_len = end_pos - start_pos
        
        print(f"Window boundaries: [{start_pos}, {end_pos}) (length: {actual_window_len})")
        
        # Extract the TinyLlama window IDs
        tinyllama_window_ids = initial_input_ids[0, start_pos:end_pos]
        
        # Calculate the local position of the masked token within the window
        masked_position_local = global_cull_position - start_pos
        
        # Get original token ID at masked position
        original_token_id_at_mask = tinyllama_window_ids[masked_position_local].item()
        
        print(f"Masked position (local): {masked_position_local}")
        print(f"Original token ID at mask: {original_token_id_at_mask}")
        
        # Decode tokens individually and reconstruct with mask
        window_token_texts = []
        for i, tid in enumerate(tinyllama_window_ids):
            if i == masked_position_local:
                # Insert MLM mask token at the culled position
                window_token_texts.append(mlm_tokenizer.mask_token)
            else:
                # Decode each TinyLlama token individually
                text = tinyllama_tokenizer.decode(
                    [tid.item()], 
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )
                window_token_texts.append(text)
        
        # Join the token texts to create the masked text string
        masked_text_string = "".join(window_token_texts)
        
        print(f"Length of masked text string: {len(masked_text_string)}")
        if len(masked_text_string) < 100:
            print(f"Masked text string: {masked_text_string}")
        else:
            print(f"Masked text string (truncated): {masked_text_string[:50]}...{masked_text_string[-50:]}")
        
        # Re-tokenize with MLM tokenizer
        mlm_inputs = mlm_tokenizer(
            masked_text_string,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=mlm_tokenizer.model_max_length
        )
        
        mlm_input_ids = mlm_inputs["input_ids"]
        mlm_attention_mask = mlm_inputs["attention_mask"]
        
        # Find the index of the mask token
        mask_indices = (mlm_input_ids == mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)
        
        # Handle cases where mask might not be found or found multiple times
        if len(mask_indices[1]) == 0:
            print(f"WARNING: Mask token not found in MLM input IDs")
            mlm_mask_index = -1
        elif len(mask_indices[1]) > 1:
            print(f"WARNING: Multiple mask tokens found in MLM input IDs at positions: {mask_indices[1].tolist()}")
            # Use the first occurrence for now
            mlm_mask_index = mask_indices[1][0].item()
        else:
            mlm_mask_index = mask_indices[1][0].item()
        
        print(f"MLM input IDs shape: {mlm_input_ids.shape}")
        print(f"MLM mask index: {mlm_mask_index}")
        
        # Create input dictionary for this window
        mlm_input_data = {
            'global_cull_position': global_cull_position,
            'original_token_id_at_mask': original_token_id_at_mask,
            'mlm_input_ids': mlm_input_ids.cpu(),  # Ensure CPU for saving
            'mlm_attention_mask': mlm_attention_mask.cpu(),
            'mlm_mask_index': mlm_mask_index,
            # Include the original TinyLlama window for reference
            'tinyllama_window_ids': tinyllama_window_ids.cpu(),
            'tinyllama_masked_position_local': masked_position_local
        }
        
        # Add to the list
        mlm_input_list.append(mlm_input_data)
    
    print(f"\nPrepared {len(mlm_input_list)} MLM input windows")
    
    # Prepare the data to be saved
    prepared_data = {
        'test_id': test_id,
        'mlm_input_list': mlm_input_list,  # List of MLM window data dictionaries
        # Pass through necessary data from previous steps
        'initial_seq_len': initial_seq_len,
        'initial_input_ids': initial_input_ids.cpu(),  # Ensure CPU
        'positions_to_cull': positions_to_cull,
        'selected_token_id': cull_candidates_data['selected_token_id']
        # KV cache intentionally NOT passed through
    }
    
    # Define the output path
    output_path = os.path.join('tests', 'captures', f'step3c_mlm_inputs_{test_id}.pt')
    
    # Save the prepared data
    torch.save(prepared_data, output_path)
    print(f"Saved MLM inputs to: {output_path}")
    
    # Assertions
    
    # Check that we have the right number of MLM inputs
    assert len(mlm_input_list) == len(positions_to_cull), \
        f"Expected {len(positions_to_cull)} MLM inputs, but got {len(mlm_input_list)}"
    
    # If we have any MLM inputs, check the structure
    if mlm_input_list:
        first_input = mlm_input_list[0]
        
        # Check keys
        required_keys = ['mlm_input_ids', 'mlm_attention_mask', 'mlm_mask_index', 
                         'global_cull_position', 'original_token_id_at_mask']
        for key in required_keys:
            assert key in first_input, f"Required key '{key}' not found in MLM input"
        
        # Check mask index is valid
        assert first_input['mlm_mask_index'] >= 0, \
            f"Mask index {first_input['mlm_mask_index']} is invalid (< 0)"
        
        # Check mask token is at the expected position
        assert first_input['mlm_input_ids'][0, first_input['mlm_mask_index']].item() == mlm_tokenizer.mask_token_id, \
            f"Token ID at mask index does not match MLM mask token ID"
    
    # Check output file was created
    assert os.path.exists(output_path), f"Output file not created at {output_path}"
