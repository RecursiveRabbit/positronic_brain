"""
Test for Step 2b: Update brightness scores based on processed attention.

This test:
1. Loads the processed attention data from Step 2a
2. Initializes a KVMirror for brightness tracking
3. Applies the brightness update formula using the processed attention scores
4. Saves the updated brightness map for use in subsequent steps
"""
import os
import pytest
import torch
from typing import Dict

from positronic_brain import config
from positronic_brain.model_io import load_model
from positronic_brain.serialization_utils import safe_load, safe_save
from positronic_brain.kv_mirror import KVMirror

# Define test cases for brightness updates
test_cases = [
    pytest.param("short_fox", 13, id="brightness_short_fox"),
    pytest.param("resume_context", 1018, id="brightness_resume_context"),
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
def load_processed_step2a_output(test_id):
    """
    Load the processed attention data saved by test_process_attention_scores from step 2a.
    
    Args:
        test_id: Identifier for the test case to load
    
    Returns:
        dict: The processed attention data for the specified test_id
    """
    # Define the path to the processed attention data based on test_id
    output_path = os.path.join('tests', 'captures', f'step2a_processed_attention_{test_id}.pt')
    
    # Ensure the file exists
    assert os.path.exists(output_path), f"Step 2a output file not found at {output_path}"
    
    # Load the processed attention data
    processed_data = safe_load(output_path)
    
    # Ensure it contains the expected keys
    required_keys = ['processed_attention_scores', 'initial_input_ids', 'selected_token_id']
    for key in required_keys:
        assert key in processed_data, f"Required key '{key}' not found in step 2a output data"
    
    print(f"Step 2a processed attention data for '{test_id}' loaded successfully from {output_path}")
    
    return processed_data

@pytest.fixture(scope="function")
def initialized_kv_mirror_for_brightness(load_processed_step2a_output):
    """
    Initialize a KVMirror with tokens from the initial input sequence.
    
    Args:
        load_processed_step2a_output: Fixture containing the processed attention data
    
    Returns:
        KVMirror: Initialized with tokens from initial_input_ids and max brightness
    """
    # Create a new KVMirror instance
    kv_mirror = KVMirror()
    
    # Get the initial_input_ids from the processed data
    initial_input_ids = load_processed_step2a_output['initial_input_ids']
    
    # Extract sequence as a list of token IDs
    token_ids = initial_input_ids[0].tolist()
    
    print(f"Initializing KVMirror with {len(token_ids)} tokens")
    
    # Add each token to the KVMirror with maximum brightness
    for pos, token_id in enumerate(token_ids):
        # Add token with position information
        instance_id = kv_mirror.add_token(
            token_id=token_id,
            position=pos,
            brightness=config.BRIGHTNESS_MAX
        )
        
        # Verify the instance was added successfully
        assert instance_id is not None, f"Failed to add token at position {pos} to KVMirror"
    
    # Get a snapshot of the initialized KVMirror for verification
    snapshot = kv_mirror.get_snapshot()
    
    print(f"KVMirror initialized successfully with {len(snapshot)} tokens, all at max brightness {config.BRIGHTNESS_MAX}")
    
    return kv_mirror

@pytest.mark.parametrize("test_id, expected_initial_tokens", test_cases)
def test_update_brightness(
    loaded_models_and_tokenizer,
    load_processed_step2a_output,
    initialized_kv_mirror_for_brightness,
    test_id,
    expected_initial_tokens
):
    """
    Test the brightness updating logic based on processed attention scores.
    
    Args:
        loaded_models_and_tokenizer: Fixture containing model, tokenizer, and device
        load_processed_step2a_output: Fixture containing the processed attention data
        initialized_kv_mirror_for_brightness: Fixture with a KVMirror populated with initial tokens
        test_id: Identifier for this test case
        expected_initial_tokens: Expected number of tokens in the initial context
    """
    # Extract components from fixtures
    tokenizer = loaded_models_and_tokenizer['tokenizer']
    device = loaded_models_and_tokenizer['device']
    
    # Get the processed data and KVMirror
    processed_data = load_processed_step2a_output
    kv_mirror = initialized_kv_mirror_for_brightness
    
    # Extract relevant data
    processed_attention_scores = processed_data['processed_attention_scores']
    initial_input_ids = processed_data['initial_input_ids']
    selected_token_id = processed_data['selected_token_id']
    
    # Verify the sequence length matches what we expect
    initial_seq_len = initial_input_ids.shape[1]
    assert initial_seq_len == expected_initial_tokens, \
        f"Expected {expected_initial_tokens} tokens, but got {initial_seq_len}"
    
    # Get the initial KVMirror snapshot for comparison
    initial_snapshot = kv_mirror.get_snapshot()
    
    # Prepare a dictionary to hold the brightness updates
    brightness_updates: Dict[int, float] = {}
    
    print(f"\nCalculating brightness updates for {initial_seq_len} tokens:")
    
    # Collect some tokens for detailed verification
    verification_tokens = {
        'max_attention': {'pos': -1, 'attention': 0.0, 'before': 0.0, 'after': 0.0},
        'min_attention': {'pos': -1, 'attention': float('inf'), 'before': 0.0, 'after': 0.0},
        'mid_attention': {'pos': initial_seq_len // 2, 'attention': 0.0, 'before': 0.0, 'after': 0.0}
    }
    
    # Calculate the new brightness for each token
    for pos in range(initial_seq_len):
        # Get the instance_id for this position
        instance_id = kv_mirror.get_instance_id_at_position(pos)
        assert instance_id is not None, f"No instance_id found for position {pos}"
        
        # Get current brightness from KVMirror
        current_brightness = kv_mirror.get_brightness(instance_id)
        assert current_brightness is not None, f"No brightness found for instance_id {instance_id}"
        
        # Get the attention score for this token
        attention_score = processed_attention_scores[pos].item()
        
        # Calculate attention gain
        attention_gain = attention_score * config.BRIGHTNESS_GAIN_COEFFICIENT
        
        # Calculate new brightness
        new_brightness = current_brightness - config.BRIGHTNESS_DECAY_PER_TICK + attention_gain
        
        # Clamp to valid range
        new_brightness = max(0.0, min(new_brightness, config.BRIGHTNESS_MAX))
        
        # Store in updates dictionary
        brightness_updates[instance_id] = new_brightness
        
        # Track tokens for verification
        if attention_score > verification_tokens['max_attention']['attention']:
            verification_tokens['max_attention'] = {
                'pos': pos,
                'attention': attention_score,
                'before': current_brightness,
                'after': new_brightness,
                'instance_id': instance_id
            }
        
        if attention_score < verification_tokens['min_attention']['attention']:
            verification_tokens['min_attention'] = {
                'pos': pos,
                'attention': attention_score,
                'before': current_brightness,
                'after': new_brightness,
                'instance_id': instance_id
            }
        
        if pos == verification_tokens['mid_attention']['pos']:
            verification_tokens['mid_attention'] = {
                'pos': pos,
                'attention': attention_score,
                'before': current_brightness,
                'after': new_brightness,
                'instance_id': instance_id
            }
    
    # Print verification tokens for debugging
    print("\nVerification tokens:")
    for token_type, token_info in verification_tokens.items():
        if token_info['pos'] != -1:
            token_id = initial_input_ids[0, token_info['pos']].item()
            token_text = tokenizer.decode([token_id])
            brightness_delta = token_info['after'] - token_info['before']
            print(f"  {token_type}: '{token_text}' (position {token_info['pos']})")
            print(f"    Attention: {token_info['attention']:.5f}")
            print(f"    Before: {token_info['before']:.2f}, After: {token_info['after']:.2f} (Δ: {brightness_delta:+.2f})")
    
    # Apply the brightness updates to the KVMirror
    print(f"\nApplying {len(brightness_updates)} brightness updates to KVMirror...")
    results = kv_mirror.batch_update_brightness(brightness_updates)
    
    # Check that all updates were successful
    assert len(results) == len(brightness_updates), \
        f"Expected {len(brightness_updates)} update results, but got {len(results)}"
    
    for instance_id, success in results.items():
        assert success, f"Brightness update for instance_id {instance_id} failed"
    
    # Get the updated KVMirror snapshot
    updated_snapshot = kv_mirror.get_snapshot()
    
    # Verify the updated brightness values
    for token_type, token_info in verification_tokens.items():
        if token_info['pos'] != -1:
            instance_id = token_info['instance_id']
            expected_brightness = token_info['after']
            actual_brightness = kv_mirror.get_brightness(instance_id)
            
            assert abs(actual_brightness - expected_brightness) < 1e-5, \
                f"Expected brightness {expected_brightness}, but got {actual_brightness} for {token_type} token"
    
    # Print stats about brightness changes
    brightness_values = [token.brightness for token in updated_snapshot.values()]
    avg_brightness = sum(brightness_values) / len(brightness_values) if brightness_values else 0
    min_brightness = min(brightness_values) if brightness_values else 0
    max_brightness = max(brightness_values) if brightness_values else 0
    
    print(f"\nUpdated brightness statistics:")
    print(f"  Min brightness: {min_brightness:.5f}")
    print(f"  Max brightness: {max_brightness:.5f}")
    print(f"  Avg brightness: {avg_brightness:.5f}")
    
    # Prepare the data to be saved
    brightness_data = {
        'test_id': test_id,
        'updated_kv_mirror_snapshot': updated_snapshot,
        # Pass through necessary data for subsequent steps
        'initial_input_ids': processed_data['initial_input_ids'],
        'selected_token_id': processed_data['selected_token_id'],
        'next_kv_cache': processed_data['next_kv_cache'],
        # Include the processed attention scores as well
        'processed_attention_scores': processed_data['processed_attention_scores']
    }
    
    # Define the output path
    output_path = os.path.join('tests', 'captures', f'step2b_brightness_updated_{test_id}.pt')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the brightness data
    safe_save(brightness_data, output_path)
    
    print(f"Updated brightness data saved to {output_path}")
    
    # Verify the expected number of tokens in the snapshot
    assert len(updated_snapshot) == initial_seq_len, \
        f"Expected {initial_seq_len} tokens in the updated snapshot, but got {len(updated_snapshot)}"
