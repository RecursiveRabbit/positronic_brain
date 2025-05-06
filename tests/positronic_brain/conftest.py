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

# Import necessary components
from positronic_brain import config
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Tuple, Union
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Direct implementation of required model functions
def load_model(model_name: str, trust_remote_code: bool = False):
    """Load a model and tokenizer directly."""
    print(f"Loading model {model_name}...")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side="left"
    )
    
    return model, tokenizer

def execute_forward_pass(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Tuple] = None,
    position_ids: Optional[torch.Tensor] = None,
    get_attentions: bool = False
):
    """Execute a forward pass through the model."""
    print("--------------------")
    print("Executing forward pass...")
    print(f"  Input IDs shape: {input_ids.shape}")
    
    # Capture start time for performance tracking
    start_time = time.perf_counter()
    
    # Run the model's forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            output_attentions=get_attentions,
            use_cache=True,
            return_dict=True
        )
    
    # Extract results
    logits = outputs.logits
    past_key_values = outputs.past_key_values
    attentions = outputs.attentions if get_attentions else None
    
    # Report cache information if available
    if hasattr(past_key_values, 'get_seq_length'):
        print(f"  Returned cache reports length: {past_key_values.get_seq_length()}")
    
    # Performance tracking
    duration = time.perf_counter() - start_time
    print(f"Forward pass execution finished in {duration:.4f}s")
    print("--------------------")
    
    return logits, past_key_values, attentions

# Simple KVMirror implementation directly in this file
class TokenInfo:
    """Information about a single token in the mirror."""
    def __init__(self, token_id: int, position: int, source: str = 'system_init'):
        self.token_id = token_id
        self.position = position
        self.source = source
        self.timestamp = time.time()

class KVMirror:
    """Simple token tracking for tests. Maps positions to token info."""
    def __init__(self):
        self._tokens: Dict[int, TokenInfo] = {}  # position -> TokenInfo
        self._lock = threading.RLock()
        
    def clear(self):
        """Reset the mirror state."""
        with self._lock:
            self._tokens.clear()
    
    def add(self, token_id: int, position: int, source: str = 'system_init') -> int:
        """Add a token to a position in the cache mirror."""
        with self._lock:
            # Create token info
            token = TokenInfo(
                token_id=token_id,
                position=position,
                source=source
            )
            # Store in mirror
            self._tokens[position] = token
            return position
    
    def get_current_size(self) -> int:
        """Get the number of tokens in the mirror."""
        with self._lock:
            return len(self._tokens)

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
    # Create a new mirror for the test
    mirror = KVMirror()
    mirror.clear()
    return mirror

@pytest.fixture
def populated_mirror(empty_mirror):
    """Provide a KVMirror with 10 tokens already added."""
    mirror = empty_mirror # Start with a clean mirror
    for i in range(10):
        mirror.add(i + 100, i)
    return mirror

@pytest.fixture(scope="session")
def initialized_session_state():
    """
    Initializes the core components (model, tokenizer, initial KV cache, KV mirror)
    once per test session.
    """
    print("\n--- (Session Fixture) Initializing Session State ---")
    
    # --- Load Model and Tokenizer ---
    print(f"(Session Fixture) Loading model {config.MODEL_NAME}...")
    model, tokenizer = load_model(
        model_name=config.MODEL_NAME,
        trust_remote_code=config.TRUST_REMOTE_CODE
    )
    device = next(model.parameters()).device
    print(f"(Session Fixture) Model loaded on {device}")
    
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
    print("(Session Fixture) Priming initial KV cache with explicit tokens...")
    _, primed_kv_cache, _ = execute_forward_pass(
        model=model,
        input_ids=initial_input_ids,
        attention_mask=initial_attention_mask,
        past_key_values=None,
        position_ids=None,
        get_attentions=False # No need for attentions during priming
    )
    print("(Session Fixture) Initial KV cache primed.")
    
    # Get authoritative cache length from the cache object itself
    try:
        # For DynamicCache objects
        cache_seq_len = primed_kv_cache.get_seq_length()
    except AttributeError:
        # For tuple caches
        if isinstance(primed_kv_cache, tuple) and primed_kv_cache and isinstance(primed_kv_cache[0], tuple):
            cache_seq_len = primed_kv_cache[0][0].shape[2]
        else:
            cache_seq_len = 0  # Fallback
            print("WARNING: Could not determine cache sequence length!")
    
    print(f"(Session Fixture) Primed KV cache reports length: {cache_seq_len}")
    
    # Verify if input_seq_len matches cache_seq_len
    if input_seq_len != cache_seq_len:
        print(f"*** WARNING: Sequence length mismatch! Input len {input_seq_len}, Cache len {cache_seq_len} ***")
        # Using cache_seq_len as authoritative

    # --- Initialize KVMirror ---
    print("(Session Fixture) Initializing KVMirror...")
    kv_mirror = KVMirror()
    for pos, token_id in enumerate(initial_input_ids[0].tolist()):
        kv_mirror.add(token_id=token_id, position=pos, source='system_init')
    print(f"(Session Fixture) KVMirror initialized with {kv_mirror.get_current_size()} tokens.")

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
