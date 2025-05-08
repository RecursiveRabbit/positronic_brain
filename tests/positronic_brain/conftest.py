import pytest
import torch
import threading
import time
import os
import sys

# Add project root to allow importing sibling modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the KVMirror class and other necessary components
from positronic_brain.kv_mirror import KVMirror
from positronic_brain import config

# Step 0: Model Loading stub (to be implemented in test_loop_step0_load_model.py)
def load_model(model_name, trust_remote_code=False):
    raise NotImplementedError("Torch-based explicit model loading required here.")

# Step 1: Forward Pass Execution stub (to be implemented in test_loop_step1.py)
def execute_forward_pass(model, input_ids, attention_mask, past_key_values, position_ids, get_attentions=False):
    raise NotImplementedError("Custom forward pass implementation managing KV cache explicitly.")

# Load the long context sample for a more realistic test case
def load_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

# Get the path to the long context sample file
long_context_file = os.path.join(
    os.path.dirname(__file__), 
    'mocks', 
    'long_context_sample.txt'
)

# Load the text file or fall back to a simple prompt if the file doesn't exist
if os.path.exists(long_context_file):
    INITIAL_PROMPT_TEXT = load_text_file(long_context_file)
    print(f"Loaded long context sample ({len(INITIAL_PROMPT_TEXT)} characters)")
else:
    INITIAL_PROMPT_TEXT = "The quick brown fox jumps over the lazy dog."
    print("Warning: Could not find long_context_sample.txt, using fallback prompt")

# --- Define Fixtures ---
@pytest.fixture
def empty_mirror():
    """Provide a fresh, empty KVMirror."""
    # Create a new empty mirror
    mirror = KVMirror()
    return mirror

@pytest.fixture
def populated_mirror(empty_mirror):
    """Provide a KVMirror with 10 tokens already added."""
    mirror = empty_mirror  # Start with a clean mirror
    for i in range(10):
        mirror.append_token(token_id=i + 100, plaintext=f"token_{i}")
    return mirror

@pytest.fixture(scope="session")
def initialized_session_state():
    """
    Initializes the core components (model, tokenizer, initial KV cache, KV mirror)
    once per test session.
    """
    print("\n--- (Session Fixture) Initializing Session State ---")
    
    # --- Load Model and Tokenizer ---
    print(f"(Session Fixture) Loading model would normally happen here...")
    # Stub: In the real implementation, this would load the model
    # model, tokenizer = load_model(
    #     model_name=config.MODEL_NAME,
    #     trust_remote_code=config.TRUST_REMOTE_CODE
    # )
    # device = next(model.parameters()).device
    
    # For now, we'll just create placeholder objects for testing
    model = None
    tokenizer = None
    device = "cpu"
    print(f"(Session Fixture) Stub model initialized on {device}")
    
    # --- Explicitly Disable Special Tokens in Tokenizer ---
    print("(Session Fixture) Disabling tokenizer's automatic special token additions...")
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    # Ensure pad token is set if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"(Session Fixture) Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")

    # --- Tokenize Initial Prompt Raw (without any special tokens) ---
    print(f"(Session Fixture) Tokenizing initial prompt raw...")
    encoded = tokenizer(
        INITIAL_PROMPT_TEXT, 
        return_tensors="pt",
        add_special_tokens=False,  # Explicitly disable automatic special tokens
        padding=False
    )
    raw_ids = encoded['input_ids'].to(device)
    raw_len = raw_ids.shape[1]
    print(f"(Session Fixture) Raw prompt tokenized: {raw_len} tokens")
    
    # Manually prepend BOS token (required by Llama models)
    print("(Session Fixture) Manually prepending BOS token...")
    bos_token_id_tensor = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    initial_input_ids = torch.cat([bos_token_id_tensor, raw_ids], dim=1)
    
    # Calculate input length being sent to model
    input_seq_len = initial_input_ids.shape[1]  # Should be raw_len + 1
    print(f"(Session Fixture) Final initial input IDs shape: {initial_input_ids.shape}")
    print(f"(Session Fixture) Final input_seq_len: {input_seq_len}")
    
    # Create matching attention mask
    initial_attention_mask = torch.ones_like(initial_input_ids, device=device)

    # --- Prime KV Cache ---
    print("(Session Fixture) Priming initial KV cache would normally happen here...")
    # Stub: In the real implementation, this would execute a forward pass
    # _, primed_kv_cache, _ = execute_forward_pass(
    #     model=model,
    #     input_ids=initial_input_ids,
    #     attention_mask=initial_attention_mask,
    #     past_key_values=None,
    #     position_ids=None,
    #     get_attentions=False # No need for attentions during priming
    # )
    
    # For now, we'll just create a placeholder KV cache for testing
    primed_kv_cache = {}
    print("(Session Fixture) Stub KV cache initialized.")
    
    # For the stub implementation, we'll just use the input length as cache length
    cache_seq_len = input_seq_len
    
    print(f"(Session Fixture) Primed KV cache reports length: {cache_seq_len}")
    
    # Verify if input_seq_len matches cache_seq_len
    if input_seq_len != cache_seq_len:
        print(f"*** WARNING: Sequence length mismatch! Input len {input_seq_len}, Cache len {cache_seq_len} ***")
        # Using cache_seq_len as authoritative

    # --- Initialize KVMirror ---
    print("(Session Fixture) Initializing KVMirror...")
    kv_mirror = KVMirror()
    for pos, token_id in enumerate(initial_input_ids[0].tolist()):
        # Convert token to plaintext using the tokenizer
        plaintext = tokenizer.decode([token_id])
        kv_mirror.append_token(token_id=token_id, plaintext=plaintext)
    print(f"(Session Fixture) KVMirror initialized with {len(kv_mirror.kv_index)} tokens.")

    # --- Bundle and Return State ---
    session_state = {
        'model': model,
        'tokenizer': tokenizer,
        'device': device,
        'initial_input_ids': initial_input_ids,
        'initial_attention_mask': initial_attention_mask,
        'input_seq_len': input_seq_len,     # Length of IDs fed to model
        'cache_seq_len': cache_seq_len,     # Authoritative length reported by cache
        'primed_kv_cache': primed_kv_cache, # Cache *after* initial prompt
        'kv_mirror': kv_mirror              # Initialized mirror state
    }
    print("--- (Session Fixture) Initialization Complete ---")
    return session_state
