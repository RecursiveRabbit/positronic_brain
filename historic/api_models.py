"""
API models for the Infinite Scroll application
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, List

class SamplerUpdate(BaseModel):
    """Model for updating sampling parameters"""
    temperature: Optional[float] = Field(None, description="Temperature for sampling (0.0-2.0)")
    top_k: Optional[int] = Field(None, description="Top-K value for sampling (0-100)")
    top_p: Optional[float] = Field(None, description="Top-P (nucleus) value (0.0-1.0)")
    repetition_penalty: Optional[float] = Field(None, description="Repetition penalty (1.0-2.0)")
    token_bias: Optional[Dict[int, float]] = Field(None, description="Token biases {token_id: bias_value}")
    force_accept: Optional[bool] = Field(None, description="Force accept tokens including EOS")


class TokenInfo(BaseModel):
    """Model for token information"""
    token: str = Field(..., description="The token string")
    token_id: int = Field(..., description="The token ID")
    probability: float = Field(..., description="The probability of this token")
    bias: Optional[float] = Field(0.0, description="The current bias value for this token")


class TokenBiasUpdate(BaseModel):
    """Model for token bias updates"""
    token_id: int = Field(..., description="Token ID to update bias for")
    bias_value: float = Field(..., description="Bias value to apply")
    

class TokenBiasPhrase(BaseModel):
    """Model for biasing a phrase by converting it to tokens"""
    phrase: str = Field(..., description="Phrase to tokenize and bias")
    bias_value: float = Field(..., description="Bias value to apply to all tokens")
    

class TopTokensResponse(BaseModel):
    """Response model for top tokens endpoint"""
    tokens: List[TokenInfo] = Field(..., description="List of top predicted tokens")
    current_context: str = Field("", description="Current context for reference")
