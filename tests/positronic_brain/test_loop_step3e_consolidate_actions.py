"""
Test for Step 3e: Consolidate actions for Step 4 based on conditional pair replacement.

This test:
1. Loads the MLM pair prediction data from Step 3d
2. Constructs a final list of actions for Step 4:
   - For each pair with a successful MLM prediction that re-tokenizes correctly:
     * Generate a single replace_pair action that replaces both original tokens
   - For pairs where re-tokenization fails, no action is taken
   - Always add the newly generated token
3. Saves the consolidated action list for use in Step 4
"""
import os
import pytest
import torch

from .test_utils import safe_load, safe_save
from transformers import AutoTokenizer

# Using a fixed initial prompt from session-scoped fixture

@pytest.fixture(scope="session")
def tinyllama_tokenizer():
    """
    Load the TinyLlama tokenizer.
    
    Returns:
        AutoTokenizer: The loaded TinyLlama tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        trust_remote_code=False,
        use_fast=True
    )
    return tokenizer

@pytest.fixture(scope="function")
def load_mlm_pair_predictions_step3d_output():
    """
    Load the MLM pair prediction data saved by Step 3d.
    
    Returns:
        dict: The MLM pair prediction data
    """
    # Define the path to the MLM pair predictions using fixed identifier
    input_path = os.path.join('tests', 'captures', 'step3d_mlm_pair_predictions_fixed_initial.pt')
    
    # Ensure the file exists
    assert os.path.exists(input_path), f"Step 3d output file not found at {input_path}"
    
    # Load the MLM pair prediction data
    mlm_data = safe_load(input_path)
    
    # Ensure it contains the expected keys
    required_keys = ['mlm_pair_predictions', 'selected_pairs', 'selected_token_id', 'initial_input_ids', 'initial_seq_len']
    for key in required_keys:
        assert key in mlm_data, f"Required key '{key}' not found in Step 3d output data"
    
    print(f"Step 3d MLM pair prediction data loaded successfully from {input_path}")
    
    return mlm_data

def test_consolidate_pair_actions(
    load_mlm_pair_predictions_step3d_output,
    tinyllama_tokenizer
):
    """
    Test the consolidation of actions for Step 4 based on conditional pair replacement.
    
    This test creates a list of actions based on the MLM pair predictions:
    - For each successfully predicted and re-tokenized pair, create a single 'replace_pair' action
      that replaces both original tokens with the new token(s)
    - If re-tokenization fails for a pair, NO action is taken for that pair (no deletions)
    - Always add the 'add' action for the token generated in Step 1
    
    This ensures we only modify the context if we have a valid replacement, preventing destructive
    deletions without repair.
    
    Args:
        load_mlm_pair_predictions_step3d_output: Fixture containing the MLM pair prediction data
        tinyllama_tokenizer: Fixture containing the TinyLlama tokenizer
    """
    # Get the MLM pair prediction data
    mlm_data = load_mlm_pair_predictions_step3d_output
    
    # Extract necessary information
    mlm_pair_predictions = mlm_data['mlm_pair_predictions']
    selected_pairs = mlm_data['selected_pairs']
    selected_token_id = mlm_data['selected_token_id']
    initial_input_ids = mlm_data['initial_input_ids']
    
    # Using fixed identifier
    test_id = 'fixed_initial'
    print(f"\nConsolidating pair-based actions for test_id: {test_id}")
    print(f"  - Number of pairs to cull: {len(selected_pairs)}")
    print(f"  - Number of MLM pair predictions: {len(mlm_pair_predictions)}")
    print(f"  - Selected token ID to add: {selected_token_id}")
    
    # Initialize the action list
    action_list = []
    replace_pair_count = 0
    
    # Create a map of predictions for easy lookup
    predictions_map = {pred['global_target_position']: pred for pred in mlm_pair_predictions}
    
    # Process each selected pair
    for pair in selected_pairs:
        pos1, pos2 = pair  # First and second position in the pair
        
        # Look up the prediction for this pair
        prediction = predictions_map.get(pos1)
        
        if prediction:
            # Get the already cleaned predicted text from Step 3d
            predicted_text = prediction['predicted_text']
            
            # Ensure the text is not empty
            if predicted_text:
                # Re-tokenize predicted text with TinyLlama tokenizer
                # Important: add_special_tokens=False to get only content tokens
                retokenized_output = tinyllama_tokenizer(predicted_text, add_special_tokens=False)
                new_tinyllama_ids = retokenized_output['input_ids']
                
                if new_tinyllama_ids:
                    # Create a replace_pair action to replace both tokens with the predicted text
                    replace_pair_action = {
                        'action': 'replace_pair',
                        'original_pos1': pos1,
                        'original_pos2': pos2,
                        'new_token_ids': new_tinyllama_ids
                    }
                    
                    # Add to action list
                    action_list.append(replace_pair_action)
                    replace_pair_count += 1
                    
                    print(f"  - Added replace_pair action for pair ({pos1}, {pos2}):")
                    print(f"      Original pair text: '{prediction['original_pair_text']}'")
                    print(f"      Replaced with: '{predicted_text}' → token IDs: {new_tinyllama_ids}")
                else:
                    print(f"Warning: Re-tokenization failed for pair ({pos1}, {pos2}), prediction '{predicted_text}'. "
                          f"No action taken for this pair.")
            else:
                print(f"Warning: Empty prediction text for pair ({pos1}, {pos2}). No action taken for this pair.")
        else:
            print(f"Warning: No MLM prediction found for pair starting at position {pos1}. "
                  f"No action taken for this pair.")
    
    # Add action for the selected token ID
    add_action = {
        'action': 'add',
        'token_id': selected_token_id
    }
    action_list.append(add_action)
    add_count = 1
    print(f"  - Added add action for token_id {selected_token_id}")
    
    # Count actions by type
    replace_pair_count = sum(1 for action in action_list if action['action'] == 'replace_pair')
    add_count = sum(1 for action in action_list if action['action'] == 'add')
    
    print(f"\nAction summary:")
    print(f"  - Replace_pair actions: {replace_pair_count}")
    print(f"  - Add actions: {add_count}")
    print(f"  - Total actions: {len(action_list)}")
    
    # Prepare the data to be saved
    output_data = {
        'test_id': test_id,
        'action_list': action_list,  # The consolidated list of actions for Step 4
        # Pass through necessary data for Step 4 execution/validation if needed
        'initial_input_ids': mlm_data['initial_input_ids'],  # Original context
        'initial_seq_len': mlm_data['initial_seq_len'],
        'selected_pairs': selected_pairs,  # Useful for validation
        'mlm_pair_predictions': mlm_pair_predictions  # Useful for validation
    }
    
    # Define the output path using fixed identifier
    output_path = os.path.join('tests', 'captures', 'step3e_actions_fixed_initial.pt')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the action data
    safe_save(output_data, output_path)
    
    print(f"\nConsolidated action data saved to {output_path}")
    
    # Assertions
    assert isinstance(action_list, list), "action_list should be a list"
    
    # Verify the only action types are 'replace_pair' and 'add'
    action_types = {action['action'] for action in action_list}
    assert action_types.issubset({'replace_pair', 'add'}), \
        f"Expected only 'replace_pair' and 'add' action types, got {action_types}"
    
    # Verify the number of replace_pair actions is at most the number of selected pairs
    assert replace_pair_count <= len(selected_pairs), \
        f"Expected at most {len(selected_pairs)} replace_pair actions, got {replace_pair_count}"
    
    # Verify the single add action
    assert add_count == 1, f"Expected exactly 1 add action, got {add_count}"
    
    # Verify there is exactly one 'add' action with the correct token_id
    add_actions = [a for a in action_list if a['action'] == 'add']
    assert len(add_actions) == 1, f"Expected exactly 1 add action, got {len(add_actions)}"
    assert add_actions[0]['token_id'] == selected_token_id, \
        f"Expected token_id {selected_token_id}, got {add_actions[0]['token_id']}"
    
    # Check structure of replace_pair actions if any exist
    if replace_pair_count > 0:
        sample_replace_pair = next(a for a in action_list if a['action'] == 'replace_pair')
        assert 'original_pos1' in sample_replace_pair, "'replace_pair' action should have 'original_pos1' key"
        assert 'original_pos2' in sample_replace_pair, "'replace_pair' action should have 'original_pos2' key"
        assert 'new_token_ids' in sample_replace_pair, "'replace_pair' action should have 'new_token_ids' key"
        assert isinstance(sample_replace_pair['original_pos1'], int), "'original_pos1' should be an integer"
        assert isinstance(sample_replace_pair['original_pos2'], int), "'original_pos2' should be an integer"
        assert isinstance(sample_replace_pair['new_token_ids'], list), "'new_token_ids' should be a list"
        assert len(sample_replace_pair['new_token_ids']) > 0, "'new_token_ids' should not be empty"
        assert all(isinstance(token_id, int) for token_id in sample_replace_pair['new_token_ids']), \
            "All token IDs should be integers"
        
        # Verify that each replace_pair action refers to a valid pair
        for action in action_list:
            if action['action'] == 'replace_pair':
                pos1 = action['original_pos1']
                pos2 = action['original_pos2']
                assert (pos1, pos2) in selected_pairs, \
                    f"replace_pair action refers to pair ({pos1}, {pos2}) which is not in selected_pairs"
    
    # Verify output file was created
    assert os.path.exists(output_path), f"Output file should exist at {output_path}"
