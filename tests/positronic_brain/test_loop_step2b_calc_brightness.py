"""
Test for Step 2b: Calculate brightness map based on processed attention scores.

This test:
1. Loads the processed attention data from Step 2a
2. Applies the brightness calculation formula to compute the new brightness for each token
3. Saves the resulting brightness map for use in subsequent steps

NO interactions with KVMirror - pure calculation only.
"""
import os
import pytest
import torch

from positronic_brain import config
from positronic_brain.model_io import load_model
from positronic_brain.serialization_utils import safe_load, safe_save

# Define test cases for brightness calculations
test_cases = [
    pytest.param("short_fox", 13, id="brightness_short_fox"),
    pytest.param("long_context_sample", 862, id="brightness_long_context_sample"),
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
def brightness_config():
    """
    Return a dictionary containing the relevant brightness parameters.
    
    Returns:
        dict: Contains brightness-related configuration values
    """
    return {
        'max_brightness': config.BRIGHTNESS_MAX,
        'decay_per_tick': config.BRIGHTNESS_DECAY_PER_TICK,
        'gain_coefficient': config.BRIGHTNESS_GAIN_COEFFICIENT,
        'brightness_seed': getattr(config, 'BRIGHTNESS_SEED', {'system_init': 255.0, 'default': 255.0})
    }

@pytest.mark.parametrize("test_id, expected_initial_tokens", test_cases)
def test_calculate_brightness_map(
    loaded_models_and_tokenizer,
    load_processed_step2a_output,
    brightness_config,
    test_id,
    expected_initial_tokens
):
    """
    Test the brightness calculation formula using processed attention scores.
    
    This test applies the brightness update formula to calculate new brightness values
    based on processed attention scores from Step 2a and initial brightness settings.
    NO KVMirror interaction happens here.
    
    Args:
        loaded_models_and_tokenizer: Fixture containing model, tokenizer, and device
        load_processed_step2a_output: Fixture containing the processed attention data
        brightness_config: Fixture containing brightness-related configuration values
        test_id: Identifier for this test case
        expected_initial_tokens: Expected number of tokens in the initial context
    """
    # Extract components
    tokenizer = loaded_models_and_tokenizer['tokenizer']
    device = loaded_models_and_tokenizer['device']
    
    # Get the processed data
    processed_data = load_processed_step2a_output
    
    # Extract relevant data
    processed_attention_scores = processed_data['processed_attention_scores']
    initial_input_ids = processed_data['initial_input_ids']
    selected_token_id = processed_data['selected_token_id']
    
    # Extract the sequence length from step2a data
    initial_seq_len = processed_data['initial_seq_len']
    # No longer using current_context_size, only working with initial context
    
    # Verify sizes are correct
    assert initial_seq_len == expected_initial_tokens, \
        f"Expected {expected_initial_tokens} tokens, but got {initial_seq_len}"
    
    print(f"Initial sequence length: {initial_seq_len}")
    # No longer tracking current_context_size
    
    # Get brightness parameters
    max_brightness = brightness_config['max_brightness']
    decay_per_tick = brightness_config['decay_per_tick']
    gain_coefficient = brightness_config['gain_coefficient']
    brightness_seed = brightness_config['brightness_seed']
    
    # Use 'system_init' as default brightness value or fallback to max brightness
    initial_brightness_value = brightness_seed.get('system_init', max_brightness)
    
    print(f"\nInitializing brightness calculation with parameters:")
    print(f"  - Initial brightness: {initial_brightness_value:.2f}")
    print(f"  - Decay per tick: {decay_per_tick:.2f}")
    print(f"  - Gain coefficient: {gain_coefficient:.2f}")
    print(f"  - Max brightness: {max_brightness:.2f}")
    
    # Initialize the previous brightness tensor with the initial value
    # Match device of processed_attention_scores
    previous_brightness = torch.full(
        (initial_seq_len,), 
        initial_brightness_value, 
        dtype=torch.float32,
        device=processed_attention_scores.device
    )
    
    # Calculate new brightness using the formula
    print(f"Calculating new brightness values for {initial_seq_len} tokens...")
    
    # Ensure tensors are float type
    prev_b = previous_brightness.float()
    attn_scores = processed_attention_scores.float()
    
    # Calculate attention gain component
    attention_gain = attn_scores * gain_coefficient
    
    # Apply the brightness update formula
    new_brightness_unclamped = prev_b - decay_per_tick + attention_gain
    
    # Clamp values to valid range
    new_brightness_map = torch.clamp(new_brightness_unclamped, 0.0, max_brightness)
    
    # Move the brightness map to CPU for easier handling and saving
    new_brightness_map = new_brightness_map.cpu()
    
    # Print statistics about the brightness calculations
    print(f"\nBrightness update statistics:")
    print(f"  Min initial brightness: {prev_b.min().item():.2f}")
    print(f"  Max initial brightness: {prev_b.max().item():.2f}")
    print(f"  Min attention gain: {attention_gain.min().item():.2f}")
    print(f"  Max attention gain: {attention_gain.max().item():.2f}")
    print(f"  Min new brightness: {new_brightness_map.min().item():.2f}")
    print(f"  Max new brightness: {new_brightness_map.max().item():.2f}")
    print(f"  Avg new brightness: {new_brightness_map.mean().item():.2f}")
    
    # Find tokens with highest and lowest brightness changes for verification
    brightness_delta = new_brightness_map - prev_b.cpu()
    max_increase_idx = torch.argmax(brightness_delta).item()
    max_decrease_idx = torch.argmin(brightness_delta).item()
    
    # Function to decode and display token info
    def token_info(idx):
        token_id = initial_input_ids[0, idx].item()
        token_text = tokenizer.decode([token_id])
        attn = processed_attention_scores[idx].item()
        old_b = prev_b[idx].item()
        new_b = new_brightness_map[idx].item()
        delta = new_b - old_b
        return f"'{token_text}' (position {idx}): attn={attn:.5f}, brightness {old_b:.2f}→{new_b:.2f} (Δ: {delta:+.2f})"
    
    # Print token with highest brightness change (from attention)
    print(f"\nToken with max brightness increase: {token_info(max_increase_idx)}")
    
    # Print token with highest brightness decrease (from decay)
    print(f"Token with max brightness decrease: {token_info(max_decrease_idx)}")
    
    # Perform spot checks on selected tokens
    print("\nSpot check calculations:")
    
    # Check a few specific tokens (first, middle, last)
    spot_check_indices = [0, initial_seq_len // 2, initial_seq_len - 1]
    
    for idx in spot_check_indices:
        # Manual calculation for verification
        token_attn = processed_attention_scores[idx].item()
        token_prev_b = prev_b[idx].item()
        
        # Expected calculation by formula
        expected_gain = token_attn * gain_coefficient
        expected_new_b_unclamped = token_prev_b - decay_per_tick + expected_gain
        expected_new_b = max(0.0, min(max_brightness, expected_new_b_unclamped))
        
        # Actual calculated value
        actual_new_b = new_brightness_map[idx].item()
        
        # Print verification
        print(f"  Token at position {idx}:")
        print(f"    - Attention score: {token_attn:.5f}")
        print(f"    - Previous brightness: {token_prev_b:.2f}")
        print(f"    - Calculated brightness: {actual_new_b:.2f}")
        print(f"    - Expected brightness: {expected_new_b:.2f}")
        print(f"    - Matched: {abs(actual_new_b - expected_new_b) < 1e-5}")
        
        # Verify calculation is correct with small float tolerance
        assert abs(actual_new_b - expected_new_b) < 1e-5, \
            f"Brightness calculation mismatch at position {idx}"
    
    # Prepare the data to be saved
    brightness_data = {
        'test_id': test_id,
        'new_brightness_map': new_brightness_map,  # Shape [initial_seq_len]
        # Pass through size information (only initial_seq_len)
        'initial_seq_len': initial_seq_len,
        # Pass through necessary data for subsequent steps
        'initial_input_ids': processed_data['initial_input_ids'],
        'selected_token_id': processed_data['selected_token_id'],
        # Include the processed attention scores as well
        'processed_attention_scores': processed_data['processed_attention_scores']
        # KV cache intentionally NOT passed through
    }
    
    # Define the output path
    output_path = os.path.join('tests', 'captures', f'step2b_brightness_map_{test_id}.pt')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the brightness data
    safe_save(brightness_data, output_path)
    
    print(f"\nBrightness map data saved to {output_path}")
    
    # Final assertions
    assert isinstance(new_brightness_map, torch.Tensor), "new_brightness_map should be a tensor"
    assert new_brightness_map.shape[0] == initial_seq_len, \
        f"Expected new_brightness_map shape to be [{initial_seq_len}], but got {new_brightness_map.shape}"
    assert torch.all(new_brightness_map >= 0.0) and torch.all(new_brightness_map <= max_brightness), \
        f"All brightness values should be between 0 and {max_brightness}"
