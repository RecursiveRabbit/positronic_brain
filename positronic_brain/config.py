"""
Configuration settings for the Infinite Scroll AI system.
This module centralizes all configurable parameters to make adjustments easier.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import torch  # For device check

# --- Model Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DIFFUSER_MODEL_NAME = "distilbert-base-uncased"  # Model used for token repair in the Compactor
TRUST_REMOTE_CODE = False  # TinyLlama doesn't need remote code

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
# Brightness Seeding - Initial brightness based on token source
BRIGHTNESS_SEED = {
    'user': 255.0,         # User tokens start at maximum brightness
    'user_inject': 255.0,  # User-injected tokens start at maximum brightness
    'system': 255.0,       # System tokens start at maximum brightness
    'system_init': 255.0,  # Initial context tokens start at maximum brightness
    'tool': 255.0,         # Tool tokens start at maximum brightness
    'llm': 255.0,          # Model-generated tokens start at maximum brightness
    'default': 255.0       # Default for any unspecified sources
}

# Brightness mechanics
BRIGHTNESS_MAX = 255.0              # Maximum brightness value (cap)
BRIGHTNESS_DECAY_PER_TICK = 2.0     # Amount brightness decays each generation step
BRIGHTNESS_GAIN_COEFFICIENT = 10.0  # Multiplier for attention-based brightness gain
BRIGHTNESS_ALPHA = 0.5              # Scaling factor for brightness updates
BRIGHTNESS_BETA = 0.1               # Secondary scaling factor for brightness updates
# DEPRECATED - Do not use directly, instead use the BRIGHTNESS_SEED dictionary with appropriate source
# Will be removed in future versions
INITIAL_TOKEN_BRIGHTNESS = 255.0

# Attention trace settings
ATTENTION_TRACE_INTERVAL = 50       # Save attention traces every N steps (0 to disable)

# --- Diffuser Brightness-Guided Noise parameters ---
BRIGHTNESS_LOCK_THRESHOLD = 0.8     # Tokens with normalized brightness >= this value are locked (immutable)
BRIGHTNESS_NOISE_ALPHA = 1.0        # Base noise intensity multiplier

# --- Repair Configuration ---
BRIGHTNESS_REPAIR_THRESHOLD = 180.0 # Tokens with brightness below this threshold are repair candidates
MAX_REPAIR_TOKENS_PER_STEP = 2      # Maximum number of tokens to repair per maintenance phase

# --- Compactor Configuration ---
COMPACTOR_SLEEP_INTERVAL = 1.0       # Seconds between Compactor repair cycles (decreased)
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
