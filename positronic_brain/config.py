"""
Configuration settings for the Infinite Scroll AI system.
This module centralizes all configurable parameters to make adjustments easier.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import torch  # For device check

# --- Model Configuration ---
MODEL_NAME = "moonshotai/Kimi-VL-A3B-Thinking"
TRUST_REMOTE_CODE = True

# --- Device Configuration ---
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = "cpu"

# --- Performance/Optimization Configuration ---
OFFLOAD_KV_CACHE_TO_CPU = False

# --- Context Window Configuration ---
CONTEXT_WINDOW_TARGET = 500  # Target size of KV cache after pruning

# --- Pruning Configuration ---
TEMPORAL_PENALTY_FACTOR = 0.005  # Factor for age-based pruning penalty

# --- Sampling Configuration ---
@dataclass
class SamplerState:
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    token_bias: Optional[Dict[int, float]] = field(default=None)  # {token_id: logit_delta}
    force_accept: bool = False  # emergency breaker
