"""
Type definitions for the sampling module.

This module contains the data structures needed for sampling configuration,
separated to avoid circular imports between ai_core and sampler modules.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class SamplerState:
    """Configuration state for token sampling operations."""
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    token_bias: Optional[Dict[int, float]] = None

    def __post_init__(self):
        if self.token_bias is None:
            self.token_bias = {}
