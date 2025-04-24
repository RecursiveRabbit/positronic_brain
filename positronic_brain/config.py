"""
Configuration settings for the Infinite Scroll AI system.
This module centralizes all configurable parameters to make adjustments easier.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import torch  # For device check

# --- Model Configuration ---
MODEL_NAME = "moonshotai/Kimi-VL-A3B-Thinking"
DIFFUSER_MODEL_NAME = "distilbert-base-uncased"  # Model used for token repair in the Compactor
TRUST_REMOTE_CODE = True

# --- Device Configuration ---
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = "cpu"

# --- Performance/Optimization Configuration ---
OFFLOAD_KV_CACHE_TO_CPU = False
MAX_SEQUENCE_LENGTH = 2048        # Maximum sequence length for the model
MAX_NEW_TOKENS = 8192           # Maximum number of new tokens to generate
PRUNING_INTERVAL = 20            # Interval for pruning the KV cache
MAX_BEAM_SOURCES = 1             # Maximum number of beam sources for sampling

# --- Context Window Configuration ---
CONTEXT_WINDOW_TARGET = 500  # Target size of KV cache after pruning

# --- Pruning Configuration ---
TEMPORAL_PENALTY_FACTOR = 0.005  # Factor for age-based pruning penalty

# --- Brightness Engine Configuration ---
BRIGHTNESS_ALPHA = 0.7      # Weight for new attention-based brightness
BRIGHTNESS_BETA = 0.3       # Weight for existing brightness (decay factor)
BRIGHTNESS_REPAIR_THRESHOLD = 50.0  # Only repair tokens with brightness below this threshold
INITIAL_TOKEN_BRIGHTNESS = 255.0    # Initial brightness value for newly registered tokens
MAX_REPAIR_TOKENS_PER_STEP = 5      # Maximum number of tokens to repair in a single step

# --- Compactor Configuration ---
COMPACTOR_SLEEP_INTERVAL = 5.0       # Seconds between Compactor repair cycles
COMPACTOR_BUFFER_SIZE = 10          # Maximum number of diffs that can be queued
COMPACTOR_ENABLED = True            # Whether the Compactor is enabled
COMPACTOR_WINDOW_SIZE = 16          # Size of context window around repair position
COMPACTOR_REQUEST_TIMEOUT = 3.0     # Timeout (seconds) for embedding data requests
MAX_REQUESTS_PER_CYCLE = 5          # Maximum number of embedding requests processed per main loop cycle

# --- Sampling Configuration ---
@dataclass
class SamplerState:
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    token_bias: Optional[Dict[int, float]] = field(default=None)  # {token_id: logit_delta}
    force_accept: bool = False  # emergency breaker
