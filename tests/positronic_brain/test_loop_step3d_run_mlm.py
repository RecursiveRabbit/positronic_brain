"""
Test Step 3d: Run MLM on Pairs and Predict Text

This test:
1. Loads the MLM-ready input data prepared by Step 3c for adjacent token pairs
2. For each input window (corresponding to a pair of positions to cull):
   - Performs a forward pass of the MLM using the provided input_ids and attention_mask
   - Extracts the output logits at the masked token position
   - Determines the predicted token by taking the argmax of the logits
   - Decodes the predicted token into a text string
3. Saves the predictions for use in Step 3e

This step focuses on finding a single token replacement for each pair of adjacent tokens
that were selected for culling, where the first token of the pair was masked and the second
token was omitted entirely.

NO interactions with KVMirror - pure prediction only.
"""

import os
import pytest
import torch
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Add safe loading utility
from positronic_brain import config
from positronic_brain.serialization_utils import safe_load, safe_save

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Using a fixed initial prompt from session-scoped fixture

@pytest.fixture(scope="function")
def load_mlm_pair_inputs_step3c_output():
    """
    Load the MLM pair inputs data saved by Step 3c.
    
    Returns:
        dict: The MLM pair inputs data
    """
    # Define the path to the MLM pair inputs data using fixed identifier
    output_path = os.path.join('tests', 'captures', 'step3c_mlm_pair_inputs_fixed_initial.pt')
    
    # Ensure the file exists
    assert os.path.exists(output_path), f"Step 3c output file not found at {output_path}"
    
    # Load the MLM pair inputs data
    mlm_pair_inputs_data = safe_load(output_path)
    
    # Add the test_id for reference if not already present
    if 'test_id' not in mlm_pair_inputs_data:
        mlm_pair_inputs_data['test_id'] = 'fixed_initial'
    
    # Ensure the data contains expected keys
    required_keys = ['mlm_pair_input_list', 'initial_seq_len', 'initial_input_ids', 'selected_pairs']
    for key in required_keys:
        assert key in mlm_pair_inputs_data, f"Required key '{key}' not found in step 3c output data"
    
    # Verify we have MLM pair inputs
    assert 'mlm_pair_input_list' in mlm_pair_inputs_data, "MLM pair input list not found in step 3c output data"
    
    # Verify the MLM pair input list matches the number of pairs to cull
    assert len(mlm_pair_inputs_data['mlm_pair_input_list']) == len(mlm_pair_inputs_data['selected_pairs']), \
        f"Expected {len(mlm_pair_inputs_data['selected_pairs'])} MLM pair inputs, " \
        f"but got {len(mlm_pair_inputs_data['mlm_pair_input_list'])}"
    
    print(f"Step 3c MLM pair inputs data loaded successfully with {len(mlm_pair_inputs_data['mlm_pair_input_list'])} pairs")
    
    return mlm_pair_inputs_data

@pytest.fixture(scope="session")
def loaded_mlm_model_and_tokenizer():
    """
    Load the MLM model and tokenizer.
    
    Returns:
        dict: Contains the MLM model, tokenizer, and device
    """
    # Get model name from config or use a default
    model_name = getattr(config, 'DIFFUSER_MODEL_NAME', 'distilbert-base-uncased')
    
    print(f"\nLoading MLM model {model_name}...")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the model
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model.to(device)
    
    print(f"MLM model loaded successfully on {device}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'device': device
    }

# Re-use the tinyllama_tokenizer fixture from test_loop_step3c_prepare_mlm.py
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

def test_run_mlm_predict_text(load_mlm_pair_inputs_step3c_output, loaded_mlm_model_and_tokenizer, tinyllama_tokenizer):
    """
    Test Step 3d: Run MLM and predict text for masked tokens in adjacent pairs.

    This test performs MLM forward passes on the prepared input windows and
    extracts the predictions for each masked token position, where each mask
    represents a pair of adjacent tokens (with the first token masked and the
    second token omitted). The predicted text is cleaned (removing artifacts like '##')
    and the original text of each token pair is included for comparison.

    Args:
        load_mlm_pair_inputs_step3c_output: Fixture containing the MLM pair inputs data
        loaded_mlm_model_and_tokenizer: Fixture containing the MLM model and tokenizer
        tinyllama_tokenizer: The TinyLlama tokenizer for decoding original tokens
    """
    # Extract data from fixtures
    mlm_pair_inputs_data = load_mlm_pair_inputs_step3c_output
    mlm_model = loaded_mlm_model_and_tokenizer['model']
    mlm_tokenizer = loaded_mlm_model_and_tokenizer['tokenizer']
    device = loaded_mlm_model_and_tokenizer['device']
    
    # Extract test data
    test_id = mlm_pair_inputs_data['test_id']
    mlm_pair_input_list = mlm_pair_inputs_data['mlm_pair_input_list']
    selected_pairs = mlm_pair_inputs_data['selected_pairs']
    
    print(f"Running MLM forward passes for token pairs with parameters:")
    print(f"  - Number of MLM pair input windows: {len(mlm_pair_input_list)}")
    print(f"  - Selected pairs: {selected_pairs}")
    print(f"  - Device: {device}")
    
    # Initialize list to hold pair predictions
    mlm_pair_predictions_list = []
    
    # Process each pair window
    for window_idx, window_data in enumerate(mlm_pair_input_list):
        # Extract window data
        original_pair = window_data['original_pair']
        global_target_position = window_data['global_target_position']  # First token position
        global_omitted_position = window_data['global_omitted_position']  # Second token position
        original_token_id_at_target = window_data['original_token_id_at_target']
        mlm_input_ids = window_data['mlm_input_ids'].to(device)
        mlm_attention_mask = window_data['mlm_attention_mask'].to(device)
        mlm_mask_index = window_data['mlm_mask_index']
        
        # Get the original token ID for the second token in the pair
        original_token_id_omitted = mlm_pair_inputs_data['initial_input_ids'][0, global_omitted_position].item()
        
        print(f"\nProcessing pair window {window_idx+1}/{len(mlm_pair_input_list)} "
              f"(pair: {original_pair})")
        
        # Handle invalid mask index
        if mlm_mask_index < 0:
            logger.warning(f"Invalid mask index {mlm_mask_index} for window {window_idx}. Skipping.")
            continue
        
        # Run MLM forward pass
        with torch.no_grad():
            outputs = mlm_model(input_ids=mlm_input_ids, attention_mask=mlm_attention_mask)
            logits = outputs.logits  # Shape: [1, window_len_mlm, mlm_vocab_size]
        
        # Get prediction at mask position
        mask_logits = logits[0, mlm_mask_index, :]
        predicted_mlm_token_id = torch.argmax(mask_logits).item()
        
        # Decode predicted MLM token to text and clean artifacts
        predicted_text_raw = mlm_tokenizer.decode([predicted_mlm_token_id])
        cleaned_predicted_text = predicted_text_raw.replace('##', '')  # Remove MLM artifacts
        
        # Decode original TinyLlama pair text
        original_text1 = tinyllama_tokenizer.decode([original_token_id_at_target], 
                                                   skip_special_tokens=False, 
                                                   clean_up_tokenization_spaces=False)
        original_text2 = tinyllama_tokenizer.decode([original_token_id_omitted], 
                                                   skip_special_tokens=False, 
                                                   clean_up_tokenization_spaces=False)
        original_pair_text = original_text1 + original_text2  # Concatenate the pair's text
        
        # Print prediction and original text for comparison
        print(f"  Pair {original_pair} original: '{original_pair_text}' → prediction: '{cleaned_predicted_text}' "
              f"(MLM token ID: {predicted_mlm_token_id})")
        
        # Store prediction result with original text and cleaned prediction
        prediction_result = {
            'original_pair': original_pair,                         # Original pair (pos1, pos2)
            'global_target_position': global_target_position,       # Position of first token in pair
            'global_omitted_position': global_omitted_position,     # Position of second token in pair
            'original_token_id_at_target': original_token_id_at_target,  # TinyLlama ID of first token
            'original_token_id_omitted': original_token_id_omitted, # TinyLlama ID of second token
            'original_pair_text': original_pair_text,              # Decoded text of original pair
            'predicted_mlm_token_id': predicted_mlm_token_id,      # MLM ID
            'predicted_text': cleaned_predicted_text               # MLM Decoded Text (cleaned)
        }
        
        mlm_pair_predictions_list.append(prediction_result)
    
    print(f"\nGenerated {len(mlm_pair_predictions_list)} MLM pair predictions")
    
    # Prepare output data
    output_data = {
        'test_id': test_id,
        'mlm_pair_predictions': mlm_pair_predictions_list,  # List of prediction result dictionaries for pairs
        # Pass through necessary data from previous steps
        'initial_seq_len': mlm_pair_inputs_data['initial_seq_len'],
        'initial_input_ids': mlm_pair_inputs_data['initial_input_ids'],  # TinyLlama IDs
        'selected_pairs': mlm_pair_inputs_data['selected_pairs'],
        'selected_token_id': mlm_pair_inputs_data.get('selected_token_id', None)
    }
    
    # Define the output path
    output_path = os.path.join('tests', 'captures', 'step3d_mlm_pair_predictions_fixed_initial.pt')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the output data
    safe_save(output_data, output_path)
    
    print(f"MLM predictions saved to: {output_path}")
    
    # Assertions
    
    # Check that we have the right number of MLM pair predictions
    valid_inputs = [w for w in mlm_pair_input_list if w['mlm_mask_index'] >= 0]
    assert len(mlm_pair_predictions_list) == len(valid_inputs), \
        f"Expected {len(valid_inputs)} MLM pair predictions, but got {len(mlm_pair_predictions_list)}"
    
    # If we have any MLM pair predictions, check the structure
    if mlm_pair_predictions_list:
        first_prediction = mlm_pair_predictions_list[0]
        
        # Check keys
        required_keys = ['original_pair', 'global_target_position', 'global_omitted_position',
                        'original_token_id_at_target', 'original_token_id_omitted',
                        'original_pair_text', 'predicted_mlm_token_id', 'predicted_text']
        for key in required_keys:
            assert key in first_prediction, f"Required key '{key}' not found in MLM pair prediction"
        
        # Check that the predicted text is a non-empty string and doesn't contain MLM artifacts
        assert isinstance(first_prediction['predicted_text'], str), \
            f"Predicted text should be a string, but got {type(first_prediction['predicted_text'])}"
        assert len(first_prediction['predicted_text']) > 0, "Predicted text should not be empty"
        assert '##' not in first_prediction['predicted_text'], "Predicted text should not contain '##' artifacts"
        
        # Check the original pair text
        assert isinstance(first_prediction['original_pair_text'], str), \
            f"Original pair text should be a string, but got {type(first_prediction['original_pair_text'])}"
        assert len(first_prediction['original_pair_text']) > 0, "Original pair text should not be empty"
        
        # Check the structure of the original_pair
        assert isinstance(first_prediction['original_pair'], tuple), \
            f"original_pair should be a tuple, but got {type(first_prediction['original_pair'])}"
        assert len(first_prediction['original_pair']) == 2, \
            f"original_pair should have exactly 2 elements, but got {len(first_prediction['original_pair'])}"
    
    # Check output file was created
    assert os.path.exists(output_path), f"Output file not created at {output_path}"
