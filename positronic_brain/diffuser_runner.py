"""
Diffuser Runner for the Positronic Brain / Halo Weave system.

This module is responsible for running the secondary diffuser model that repairs
low-brightness tokens in the context. The diffuser is a masked language model (like DistilBERT)
that predicts replacement tokens when given a context with masked tokens.
"""

import os
import glob
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from . import config
from .metrics import timed_histogram, inc_counter


def load_diffuser_model(model_name: str = None, device: str = "cuda") -> Tuple[Any, Any]:
    """
    Load the diffuser model and tokenizer.
    
    Args:
        model_name: HuggingFace model name, defaults to config.DIFFUSER_MODEL_NAME
        device: Device to load the model on ('cuda', 'cpu')
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_name or config.DIFFUSER_MODEL_NAME
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    try:
        print(f"[Diffuser] Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        ).eval()
        
        if device.type != "cuda":
            model = model.to(device)
            
        print(f"[Diffuser] Model loaded successfully on {device}")
        return model, tokenizer
    except Exception as e:
        print(f"[Diffuser ERROR] Failed to load model: {str(e)}")
        raise


class DiffuserModel:
    """
    Wrapper for a masked language model used for token repair.
    
    This class encapsulates the diffuser model and provides methods for repairing
    tokens based on their brightness scores.
    """
    
    def __init__(self, model_name=None, device="cuda"):
        """
        Initialize the diffuser model.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on
        """
        self.model_name = model_name or config.DIFFUSER_MODEL_NAME
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = load_diffuser_model(self.model_name, self.device)
        
        # Store mask token ID and embedding
        self.mask_token_id = self.tokenizer.mask_token_id
        self.mask_token_embedding = None
        
        # Extract mask token embedding for masking operations
        if self.mask_token_id is not None:
            # Get the embedding of the mask token
            with torch.no_grad():
                mask_input = torch.tensor([self.mask_token_id], device=self.device)
                self.mask_token_embedding = self.model.get_input_embeddings()(mask_input)
    
    def get_token_embedding(self, token_id: int) -> torch.Tensor:
        """Get the embedding for a specific token ID."""
        with torch.no_grad():
            token_input = torch.tensor([token_id], device=self.device)
            return self.model.get_input_embeddings()(token_input)


@timed_histogram("diffuser_repair_token_seconds")
def repair_token(
    diffuser_model: DiffuserModel,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    repair_index: int,
    original_token_id: int
) -> Optional[int]:
    """
    Repair a single token by predicting a replacement using the diffuser model.
    
    This function masks the token at repair_index and uses the diffuser model
    to predict a replacement based on surrounding context represented by hidden states.
    
    Args:
        diffuser_model: The DiffuserModel instance
        hidden_states: Hidden states from the LLM [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask tensor [batch_size, seq_len]
        repair_index: Index of the token to repair
        original_token_id: Original token ID at the repair position
        
    Returns:
        Optional[int]: New token ID if a change is needed, None otherwise
    """
    try:
        model = diffuser_model.model
        device = diffuser_model.device
        
        # Ensure inputs are on the correct device
        hidden_states = hidden_states.to(device)
        attention_mask = attention_mask.to(device)
        
        # Use hidden states as input to the diffuser model
        # We'll create a modified version with the target position masked
        modified_hidden_states = hidden_states.clone()
        
        # Replace the token at repair_index with the mask token embedding
        if diffuser_model.mask_token_embedding is not None:
            # Get mask token embedding reshaped to match hidden dimension if needed
            mask_embed = diffuser_model.mask_token_embedding
            if mask_embed.shape[-1] != hidden_states.shape[-1]:
                # Reshape or project the mask embedding to match hidden dimension
                # For simplicity, we just zero out the position instead if dimensions don't match
                modified_hidden_states[:, repair_index] = torch.zeros_like(hidden_states[:, repair_index])
            else:
                # Expand mask embedding to match batch dimension if needed
                if mask_embed.shape[0] != modified_hidden_states.shape[0]:
                    mask_embed = mask_embed.expand(modified_hidden_states.shape[0], -1, -1)
                
                # Replace the hidden state at repair_index with mask embedding
                modified_hidden_states[:, repair_index] = mask_embed[:, 0]
        else:
            inc_counter("diffuser_missing_mask_token")
            print("[Diffuser WARNING] Mask token embedding not available")
            return None
        
        # Forward pass through the diffuser model
        with torch.no_grad():
            outputs = model(
                inputs_embeds=modified_hidden_states,
                attention_mask=attention_mask
            )
            
            # Get logits for the repaired position
            target_logits = outputs.logits[0, repair_index, :]
            
            # Find the highest probability token
            predicted_token_id = torch.argmax(target_logits).item()
            
            # Return the predicted token ID if it's different from the original
            if predicted_token_id != original_token_id:
                return predicted_token_id
            else:
                return None
    
    except Exception as e:
        inc_counter("diffuser_repair_error")
        print(f"[Diffuser ERROR] Failed to repair token: {str(e)}")
        return None


@timed_histogram("diffuser_get_repaired_tokens_seconds")
def get_repaired_tokens(
    diffuser_model: DiffuserModel,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    token_ids: List[int],
    repair_indices: List[int]
) -> List[Tuple[int, int, int]]:
    """
    Get a list of token repairs for multiple positions.
    
    Args:
        diffuser_model: The DiffuserModel instance
        hidden_states: Hidden states from the last layer of the primary model
                      These serve as context for the diffuser model to predict replacements
        attention_mask: Attention mask tensor
        token_ids: List of original token IDs corresponding to the positions
        repair_indices: List of positions to repair
        
    Returns:
        List[Tuple[int, int, int]]: List of (position, old_token_id, new_token_id) tuples
        representing the token repairs
    """
    repairs = []
    
    for i, repair_idx in enumerate(repair_indices):
        if repair_idx < 0 or repair_idx >= hidden_states.shape[1]:
            continue
            
        original_token_id = token_ids[repair_idx]
        
        # Repair the token
        new_token_id = repair_token(
            diffuser_model,
            hidden_states,
            attention_mask,
            repair_idx,
            original_token_id
        )
        
        # If a repair is suggested, add it to the list
        if new_token_id is not None:
            repairs.append((repair_idx, original_token_id, new_token_id))
    
    return repairs
