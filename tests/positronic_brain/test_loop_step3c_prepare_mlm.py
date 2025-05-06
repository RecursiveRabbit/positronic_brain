"""
Test Step 3c: Prepare MLM Inputs for Token Pairs via Token-by-Token Text Translation

This test prepares inputs for a Masked Language Model (MLM) by translating TinyLlama tokens
to MLM-compatible inputs through a token-by-token text translation process.

For each pair of adjacent tokens selected for culling, this test:
1. Extracts a window of TinyLlama token IDs around the pair
2. Decodes each TinyLlama token individually to preserve boundaries
3. Replaces the first token of the pair with an MLM mask token
4. Omits the second token of the pair entirely (skipping it in the text)
5. Re-tokenizes the resulting string with the MLM tokenizer
6. Finds the index of the mask token in the MLM sequence
7. Packages all necessary information for the diffuser in Step 3d

This approach allows us to replace two adjacent tokens with a single new token,
while ensuring robust handling of different tokenizer vocabularies.
"""

import os
import pytest
import torch
import logging
from transformers import AutoTokenizer

# Add safe loading utility
from positronic_brain.utils.serialization import safe_load

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Using a fixed initial prompt from session-scoped fixture

@pytest.fixture(scope="function")
def load_culled_pairs_step3b_output():
    """
    Load the culled pairs data saved by Step 3b.
    
    Returns:
        dict: The culled pairs data
    """
    # Define the path to the culled pairs data using fixed identifier
    output_path = os.path.join('tests', 'captures', 'step3b_culled_pairs_fixed_initial.pt')
    
    # Ensure the file exists
    assert os.path.exists(output_path), f"Step 3b output file not found at {output_path}"
    
    # Load the culled pairs data
    culled_pairs_data = safe_load(output_path)
    
    # Add the test_id for reference if not already present
    if 'test_id' not in culled_pairs_data:
        culled_pairs_data['test_id'] = 'fixed_initial'
    
    # Ensure the data contains expected keys
    required_keys = ['selected_pairs', 'initial_seq_len', 'initial_input_ids']
    for key in required_keys:
        assert key in culled_pairs_data, f"Required key '{key}' not found in step 3b output data"
    
    print(f"Step 3b culled pairs data loaded successfully with {len(culled_pairs_data['selected_pairs'])} pairs")
    
    return culled_pairs_data

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

def test_prepare_mlm_inputs(load_culled_pairs_step3b_output, tinyllama_tokenizer, mlm_tokenizer, diffuser_window_config):
    """
    Test Step 3c: Prepare MLM Inputs for Token Pairs via Token-by-Token Text Translation
    
    This test prepares input data for the MLM by extracting context windows around each pair
    of tokens to be culled and translating them to MLM-compatible inputs through text.
    The first token in each pair is replaced with a mask token, while the second token is omitted.
    
    Args:
        load_culled_pairs_step3b_output: Fixture providing the Step 3b output data with selected pairs
        tinyllama_tokenizer: The TinyLlama tokenizer
        mlm_tokenizer: The MLM tokenizer
        diffuser_window_config: Fixture providing diffuser configuration parameters
    """
    # Extract the Step 3b data
    culled_pairs_data = load_culled_pairs_step3b_output
    
    # Get the token pairs to cull
    selected_pairs = culled_pairs_data['selected_pairs']
    initial_seq_len = culled_pairs_data['initial_seq_len']
    initial_input_ids = culled_pairs_data['initial_input_ids']
    
    # Get the MLM tokenizer's mask token ID
    mlm_mask_token_id = mlm_tokenizer.mask_token_id
    mlm_mask_token = mlm_tokenizer.mask_token
    
    # Get window parameters
    half_window = diffuser_window_config['half_window']
    
    print(f"Preparing MLM inputs with parameters:")
    print(f"  - Initial sequence length: {initial_seq_len}")
    print(f"  - Number of pairs to cull: {len(selected_pairs)}")
    print(f"  - Pairs to cull: {selected_pairs}")
    print(f"  - Half window size: {half_window}")
    print(f"  - MLM mask token: {mlm_mask_token}")
    print(f"TinyLlama tokenizer vocabulary size: {len(tinyllama_tokenizer)}")
    print(f"MLM tokenizer vocabulary size: {len(mlm_tokenizer)}")
    print(f"MLM mask token: {mlm_tokenizer.mask_token}")
    print(f"MLM mask token ID: {mlm_tokenizer.mask_token_id}")
    
    # Initialize list to hold MLM inputs
    mlm_pair_input_list = []
    
    # Prepare MLM input for each selected pair to cull
    for pair_idx, (pos1, pos2) in enumerate(selected_pairs):
        # Define global target and omitted positions
        global_target_position = pos1  # First token position (will be replaced with mask)
        global_omitted_position = pos2  # Second token position (will be omitted)
        
        print(f"\nPreparing MLM input for pair {(pos1, pos2)} ({pair_idx+1}/{len(selected_pairs)})")
        
        # Calculate window boundaries centered on the target position (first token)
        start_pos = max(0, global_target_position - half_window)
        end_pos = min(initial_seq_len, global_target_position + half_window + 1)
        
        # Extract the TinyLlama token IDs for this window
        tinyllama_window_ids = initial_input_ids[0, start_pos:end_pos]
        
        # Calculate the local positions within the window
        target_pos_local = global_target_position - start_pos  # Local position of the first token (to be masked)
        omitted_pos_local = global_omitted_position - start_pos  # Local position of the second token (to be omitted)
        
        # Save original token ID at the target position
        original_token_id_at_target = tinyllama_window_ids[target_pos_local].item()
        
        print(f"Window: start_pos={start_pos}, end_pos={end_pos}, window_length={len(tinyllama_window_ids)}")
        print(f"Target position (local): {target_pos_local}, Omitted position (local): {omitted_pos_local}")
        print(f"Original token ID at target: {original_token_id_at_target}")
        
        # Build the masked string token-by-token to maintain token boundaries
        window_token_texts = []
        
        for i, token_id in enumerate(tinyllama_window_ids):
            if i == target_pos_local:
                # Replace the first token of the pair with the MLM mask token
                window_token_texts.append(mlm_mask_token)
            elif i == omitted_pos_local:
                # Omit the second token of the pair entirely
                window_token_texts.append("")  # Empty string effectively removes this token
            else:
                # Keep all other tokens as they are
                token_text = tinyllama_tokenizer.decode([token_id.item()], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                window_token_texts.append(token_text)
        
        # Join all token texts into a single string
        masked_text_string = "".join(window_token_texts)
        
        # For debugging: Show a substring of the text input
        max_debug_len = 50
        debug_text = masked_text_string[:max_debug_len] + '...' if len(masked_text_string) > max_debug_len else masked_text_string
        print(f"Masked text with omitted token: {debug_text}")
        
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
            'original_pair': (pos1, pos2),  # Store the original pair for reference
            'global_target_position': global_target_position,  # Position of the first token (to be masked)
            'global_omitted_position': global_omitted_position,  # Position of the second token (to be omitted)
            'original_token_id_at_target': original_token_id_at_target,
            'mlm_input_ids': mlm_input_ids.cpu(),  # Ensure CPU for saving
            'mlm_attention_mask': mlm_attention_mask.cpu(),
            'mlm_mask_index': mlm_mask_index,
            # Include the original TinyLlama window for reference
            'tinyllama_window_ids': tinyllama_window_ids.cpu(),
            'tinyllama_target_position_local': target_pos_local,
            'tinyllama_omitted_position_local': omitted_pos_local
        }
        
        # Add to the list
        mlm_pair_input_list.append(mlm_input_data)
    
    print(f"\nPrepared {len(mlm_pair_input_list)} MLM input windows for pairs")
    
    # Prepare the data to be saved
    prepared_data = {
        'test_id': 'fixed_initial',  # Hardcoded to match our fixed identifier
        'mlm_pair_input_list': mlm_pair_input_list,  # List of MLM window data dictionaries for pairs
        # Pass through necessary data from previous steps
        'initial_seq_len': initial_seq_len,
        'initial_input_ids': initial_input_ids.cpu(),  # Ensure CPU
        'selected_pairs': selected_pairs,
        'selected_token_id': culled_pairs_data['selected_token_id']
        # KV cache intentionally NOT passed through
    }
    
    # Define the output path
    output_path = os.path.join('tests', 'captures', 'step3c_mlm_pair_inputs_fixed_initial.pt')
    
    # Save the prepared data
    torch.save(prepared_data, output_path)
    print(f"Saved MLM pair inputs to: {output_path}")
    
    # Assertions
    
    # Check that we have the right number of MLM inputs
    assert len(mlm_pair_input_list) == len(selected_pairs), \
        f"Expected {len(selected_pairs)} MLM pair inputs, but got {len(mlm_pair_input_list)}"
    
    # If we have any MLM inputs, check the structure
    if mlm_pair_input_list:
        first_input = mlm_pair_input_list[0]
        
        # Check keys
        required_keys = ['mlm_input_ids', 'mlm_attention_mask', 'mlm_mask_index', 
                         'global_target_position', 'global_omitted_position', 
                         'original_token_id_at_target', 'original_pair']
        for key in required_keys:
            assert key in first_input, f"Required key '{key}' not found in MLM pair input"
        
        # Check mask index is valid
        assert first_input['mlm_mask_index'] >= 0, \
            f"Mask index {first_input['mlm_mask_index']} is invalid (< 0)"
        
        # Check mask token is at the expected position
        assert first_input['mlm_input_ids'][0, first_input['mlm_mask_index']].item() == mlm_tokenizer.mask_token_id, \
            f"Token ID at mask index does not match MLM mask token ID"
        
        # Verify that original_pair is correctly structured
        assert len(first_input['original_pair']) == 2, \
            f"original_pair should be a tuple of length 2, got {len(first_input['original_pair'])}"
        assert first_input['original_pair'][0] == first_input['global_target_position'], \
            f"first element of original_pair should match global_target_position"
        assert first_input['original_pair'][1] == first_input['global_omitted_position'], \
            f"second element of original_pair should match global_omitted_position"
    
    # Check output file was created
    assert os.path.exists(output_path), f"Output file not created at {output_path}"
