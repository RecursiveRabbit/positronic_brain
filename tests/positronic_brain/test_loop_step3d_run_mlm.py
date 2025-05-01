"""
Test Step 3d: Run MLM and Predict Text

This test:
1. Loads the MLM-ready input data prepared by Step 3c
2. For each input window (corresponding to a position to cull):
   - Performs a forward pass of the MLM using the provided input_ids and attention_mask
   - Extracts the output logits at the masked token position
   - Determines the predicted token by taking the argmax of the logits
   - Decodes the predicted token into a text string
3. Saves the predictions for use in Step 3e

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

# Test IDs for parameterization
TEST_IDS = ['short_fox', 'long_context_sample']

@pytest.fixture(scope="function", params=TEST_IDS)
def load_mlm_inputs_step3c_output(request):
    """
    Load the MLM inputs data saved by Step 3c.
    
    Args:
        request: Pytest request object containing the test_id parameter
    
    Returns:
        dict: The MLM inputs data for the specified test_id
    """
    test_id = request.param
    
    # Define the path to the MLM inputs data
    output_path = os.path.join('tests', 'captures', f'step3c_mlm_inputs_{test_id}.pt')
    
    # Ensure the file exists
    assert os.path.exists(output_path), f"Step 3c output file not found at {output_path}"
    
    # Load the MLM inputs data
    mlm_inputs_data = safe_load(output_path)
    
    # Add the test_id for reference
    mlm_inputs_data['test_id'] = test_id
    
    # Ensure the data contains expected keys
    required_keys = ['mlm_input_list', 'initial_seq_len', 'initial_input_ids', 'positions_to_cull']
    for key in required_keys:
        assert key in mlm_inputs_data, f"Required key '{key}' not found in step 3c output data"
    
    # Verify we have MLM inputs
    assert 'mlm_input_list' in mlm_inputs_data, "MLM input list not found in step 3c output data"
    
    # Verify the MLM input list matches the number of positions to cull
    assert len(mlm_inputs_data['mlm_input_list']) == len(mlm_inputs_data['positions_to_cull']), \
        f"Expected {len(mlm_inputs_data['positions_to_cull'])} MLM inputs, " \
        f"but got {len(mlm_inputs_data['mlm_input_list'])}"
    
    print(f"Step 3c MLM inputs data for '{test_id}' loaded successfully")
    
    return mlm_inputs_data

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

def test_run_mlm_predict_text(load_mlm_inputs_step3c_output, loaded_mlm_model_and_tokenizer):
    """
    Test Step 3d: Run MLM and predict text for masked tokens.
    
    This test performs MLM forward passes on the prepared input windows and
    extracts the predictions for each masked token position.
    
    Args:
        load_mlm_inputs_step3c_output: Fixture containing the MLM inputs data
        loaded_mlm_model_and_tokenizer: Fixture containing the MLM model and tokenizer
    """
    # Extract data from fixtures
    mlm_inputs_data = load_mlm_inputs_step3c_output
    mlm_model = loaded_mlm_model_and_tokenizer['model']
    mlm_tokenizer = loaded_mlm_model_and_tokenizer['tokenizer']
    device = loaded_mlm_model_and_tokenizer['device']
    
    # Extract necessary data from Step 3c
    test_id = mlm_inputs_data['test_id']
    mlm_input_list = mlm_inputs_data['mlm_input_list']
    
    print(f"Processing {len(mlm_input_list)} MLM inputs for test_id: {test_id}")
    
    # Initialize list to collect MLM predictions
    mlm_predictions_list = []
    
    # Process each MLM input window
    for window_idx, window_data in enumerate(mlm_input_list):
        # Extract necessary data from window
        global_cull_position = window_data['global_cull_position']
        original_token_id_at_mask = window_data['original_token_id_at_mask']
        mlm_input_ids = window_data['mlm_input_ids'].to(device)
        mlm_attention_mask = window_data['mlm_attention_mask'].to(device)
        mlm_mask_index = window_data['mlm_mask_index']
        
        print(f"\nProcessing window {window_idx+1}/{len(mlm_input_list)} "
              f"(global position: {global_cull_position})")
        
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
        
        # Decode predicted MLM token to text
        predicted_text = mlm_tokenizer.decode([predicted_mlm_token_id])
        
        # Print prediction for debugging
        print(f"  Position {global_cull_position} prediction: "
              f"{predicted_text} (MLM token ID: {predicted_mlm_token_id})")
        
        # Store prediction result
        prediction_result = {
            'global_cull_position': global_cull_position,
            'original_token_id_at_mask': original_token_id_at_mask,  # TinyLlama ID
            'predicted_mlm_token_id': predicted_mlm_token_id,       # MLM ID
            'predicted_text': predicted_text                        # MLM Decoded Text
        }
        
        mlm_predictions_list.append(prediction_result)
    
    print(f"\nGenerated {len(mlm_predictions_list)} MLM predictions")
    
    # Prepare output data
    output_data = {
        'test_id': test_id,
        'mlm_predictions': mlm_predictions_list,  # List of prediction result dictionaries
        # Pass through necessary data from previous steps
        'initial_seq_len': mlm_inputs_data['initial_seq_len'],
        'initial_input_ids': mlm_inputs_data['initial_input_ids'],  # TinyLlama IDs
        'positions_to_cull': mlm_inputs_data['positions_to_cull'],
        'selected_token_id': mlm_inputs_data.get('selected_token_id', None)
    }
    
    # Define the output path
    output_path = os.path.join('tests', 'captures', f'step3d_mlm_predictions_{test_id}.pt')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the output data
    safe_save(output_data, output_path)
    
    print(f"MLM predictions saved to: {output_path}")
    
    # Assertions
    
    # Check that we have the right number of MLM predictions
    valid_inputs = [w for w in mlm_input_list if w['mlm_mask_index'] >= 0]
    assert len(mlm_predictions_list) == len(valid_inputs), \
        f"Expected {len(valid_inputs)} MLM predictions, but got {len(mlm_predictions_list)}"
    
    # If we have any MLM predictions, check the structure
    if mlm_predictions_list:
        first_prediction = mlm_predictions_list[0]
        
        # Check keys
        required_keys = ['global_cull_position', 'original_token_id_at_mask', 
                        'predicted_mlm_token_id', 'predicted_text']
        for key in required_keys:
            assert key in first_prediction, f"Required key '{key}' not found in MLM prediction"
        
        # Check that the predicted text is a non-empty string
        assert isinstance(first_prediction['predicted_text'], str), \
            f"Predicted text should be a string, but got {type(first_prediction['predicted_text'])}"
        assert len(first_prediction['predicted_text']) > 0, "Predicted text should not be empty"
    
    # Check output file was created
    assert os.path.exists(output_path), f"Output file not created at {output_path}"
