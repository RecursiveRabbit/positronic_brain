"""
Diffuser Runner for the Positronic Brain / Halo Weave system.

This module is responsible for running the secondary diffuser model that repairs
low-brightness tokens in the context. The diffuser is a masked language model (like DistilBERT)
that predicts replacement tokens when given a context with masked tokens.

Refactored to use input embeddings instead of hidden states, creating a cleaner interface
that is more aligned with how Masked Language Models typically operate.
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


def load_diffuser_model(model_name: str = None, device: str = "cuda") -> Tuple[Any, Any, int]:
    """
    Load the diffuser model, tokenizer, and mask token ID.

    Args:
        model_name: HuggingFace model name, defaults to config.DIFFUSER_MODEL_NAME
        device: Device to load the model on ('cuda', 'cpu')

    Returns:
        Tuple of (model, tokenizer, mask_token_id)
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

        # Get the mask token ID
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            print(f"[Diffuser WARNING] No mask token found in tokenizer. Using default 103.")
            mask_token_id = 103  # Default for BERT-based models

        print(f"[Diffuser] Model loaded successfully on {device}. Mask token ID: {mask_token_id}")
        return model, tokenizer, mask_token_id
    except Exception as e:
        print(f"[Diffuser ERROR] Failed to load model: {str(e)}")
        raise


class DiffuserModel:
    """
    Wrapper for a masked language model used for token repair.

    This class encapsulates the diffuser model and provides methods for repairing
    tokens based on their brightness scores. The model operates on input embeddings
    rather than hidden states, making it more compatible with standard MLM practices.
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
        self.model, self.tokenizer, self.mask_token_id = load_diffuser_model(self.model_name, self.device)

        # Store mask token embedding
        self.mask_token_embedding = None

        # Extract mask token embedding for masking operations
        if self.mask_token_id is not None:
            # Get the embedding of the mask token
            with torch.no_grad():
                mask_input = torch.tensor([self.mask_token_id], device=self.device)
                self.mask_token_embedding = self.model.get_input_embeddings()(mask_input)
                print(f"[Diffuser] Mask token embedding shape: {self.mask_token_embedding.shape}")

    def get_token_embedding(self, token_id: int) -> torch.Tensor:
        """Get the embedding for a specific token ID."""
        with torch.no_grad():
            token_input = torch.tensor([token_id], device=self.device)
            return self.model.get_input_embeddings()(token_input)

    def get_mask_embedding(self):
        """Get the mask token embedding, ensuring it's available."""
        if self.mask_token_embedding is None:
            inc_counter("diffuser_missing_mask_token")
            print("[Diffuser WARNING] Mask token embedding not available. Creating it now.")
            with torch.no_grad():
                mask_input = torch.tensor([self.mask_token_id], device=self.device)
                self.mask_token_embedding = self.model.get_input_embeddings()(mask_input)
        return self.mask_token_embedding


@timed_histogram("diffuser_predict_replacement_seconds")
def predict_replacement(
    diffuser_model: DiffuserModel,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    repair_index: int,
    original_token_id: int
) -> Optional[int]:
    """
    Predict a replacement for a token using the diffuser model.

    This function masks the token at repair_index in the input_embeddings and uses 
    the diffuser model to predict a replacement based on surrounding context.

    Args:
        diffuser_model: The DiffuserModel instance
        input_embeddings: Input embeddings [batch_size, seq_len, embed_dim]
        attention_mask: Attention mask tensor
        repair_index: Position of the token to repair
        original_token_id: Original token ID at repair_index

    Returns:
        Optional[int]: New token ID if a change is needed, None otherwise
    """
    try:
        model = diffuser_model.model
        device = diffuser_model.device

        # Ensure inputs are on the correct device
        input_embeddings = input_embeddings.to(device)
        attention_mask = attention_mask.to(device)

        # Create a copy of input embeddings to modify
        masked_embeddings = input_embeddings.clone()

        # Get the mask token embedding
        mask_embedding = diffuser_model.get_mask_embedding()

        if mask_embedding is not None:
            # Ensure mask embedding has the right shape for replacement
            if mask_embedding.shape[-1] != input_embeddings.shape[-1]:
                inc_counter("diffuser_embedding_dimension_mismatch")
                print(f"[Diffuser WARNING] Embedding dimension mismatch: {mask_embedding.shape[-1]} vs {input_embeddings.shape[-1]}")
                # Zero out the position if dimensions don't match
                masked_embeddings[:, repair_index] = torch.zeros_like(input_embeddings[:, repair_index])
            else:
                # Replace the embedding at repair_index with mask embedding
                masked_embeddings[:, repair_index] = mask_embedding[:, 0]
        else:
            inc_counter("diffuser_missing_mask_token")
            print("[Diffuser WARNING] Mask token embedding not available")
            return None

        # Forward pass through the diffuser model
        with torch.no_grad():
            outputs = model(
                inputs_embeds=masked_embeddings,
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
        inc_counter("diffuser_predict_replacement_error")
        print(f"[Diffuser ERROR] Failed to predict replacement: {str(e)}")
        return None


# For backward compatibility
@timed_histogram("diffuser_repair_token_seconds")
def repair_token(
    diffuser_model: DiffuserModel,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    repair_index: int,
    original_token_id: int
) -> Optional[int]:
    """
    Legacy function that uses hidden states instead of input embeddings.
    Kept for backward compatibility.

    Args:
        diffuser_model: The DiffuserModel instance
        hidden_states: Hidden states from the LLM [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask tensor
        repair_index: Position of the token to repair
        original_token_id: Original token ID at repair_index

    Returns:
        Optional[int]: New token ID if a change is needed, None otherwise
    """
    # In this version, we're treating hidden states as if they were input embeddings
    # This is not technically correct but maintains compatibility
    inc_counter("diffuser_using_legacy_repair")
    print("[Diffuser WARNING] Using legacy repair_token function with hidden states")
    return predict_replacement(diffuser_model, hidden_states, attention_mask, repair_index, original_token_id)


# Maximum window size for diffuser models like DistilBERT
MAX_DIFFUSER_WINDOW = 512  # 512 for DistilBERT

@timed_histogram("diffuser_compute_diff_seconds")
async def compute_diff(
    diffuser_model: DiffuserModel,
    original_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    brightness_map: torch.Tensor,
    repair_indices: List[int] = None
) -> List[Tuple[int, int, int]]:
    """
    Compute a diff of token repair suggestions using brightness-guided masking.
    
    This implements a brightness-based masked language modeling approach where:
    1. Tokens with normalized brightness >= BRIGHTNESS_LOCK_THRESHOLD are locked (kept unchanged).
    2. Lower brightness tokens are masked using the MLM mask token.
    3. The masked inputs are processed by the MLM to predict replacements.
    
    Args:
        diffuser_model: The DiffuserModel instance
        original_input_ids: Original token IDs for the sequence (shape: [1, seq_len] or [seq_len])
        attention_mask: Attention mask tensor (shape: [1, seq_len])
        brightness_map: Normalized brightness values (0-1) for each token (shape: [1, seq_len] or [seq_len])
        repair_indices: Optional list of positions to repair. If None, all positions below
                       the brightness threshold will be considered for repair.
        
    Returns:
        List[Tuple[int, int, int]]: List of (position, old_token_id, new_token_id) tuples
        representing the token repairs, in the format expected by KVMirror.apply_diff()
    """
    # Ensure model is loaded
    if diffuser_model is None:
        print("[Diffuser ERROR] Diffuser model not loaded. Cannot compute diff.")
        return []
        
    from . import config
    
    diff_list = []
    device = diffuser_model.device
    
    # Ensure inputs are on the correct device
    attention_mask = attention_mask.to(device)
    
    # Ensure original_input_ids has batch dimension and is on correct device
    if original_input_ids.dim() == 1:
        original_input_ids = original_input_ids.unsqueeze(0)  # Add batch dimension
    original_input_ids = original_input_ids.to(device)
    
    # Ensure brightness map has correct shape and device
    # Normalize brightness from [0-255] to [0-1] if needed
    b = brightness_map.to(device).float()
    if b.max() > 1.0:  # Assuming max is ~255.0
        b = b / config.BRIGHTNESS_MAX  # Normalize to [0, 1]
    
    if b.dim() == 1:
        b = b.unsqueeze(0)  # Ensure [1, seq_len]
    
    seq_len = original_input_ids.shape[1]
    
    # Check if sequence length exceeds the diffuser model window limit
    if seq_len > MAX_DIFFUSER_WINDOW:
        print(f"[Diffuser] Sequence length {seq_len} exceeds diffuser model window size {MAX_DIFFUSER_WINDOW}. Using sliding window.")
        # Calculate window start position (take the last MAX_DIFFUSER_WINDOW tokens)
        start_pos = max(0, seq_len - MAX_DIFFUSER_WINDOW)
        
        # Create windowed versions of inputs
        window_input_ids = original_input_ids[:, start_pos:]
        window_attention_mask = attention_mask[:, start_pos:]
        window_brightness = b[:, start_pos:] if b.dim() > 1 else b[start_pos:]
        
        # Map repair_indices to window positions
        if repair_indices is not None:
            window_repair_indices = [i - start_pos for i in repair_indices if i >= start_pos]
            if len(window_repair_indices) < len(repair_indices):
                print(f"[Diffuser] {len(repair_indices) - len(window_repair_indices)} repair positions were outside the sliding window")
        else:
            window_repair_indices = None
            
        # Use windowed variables from here on
        original_input_ids = window_input_ids
        attention_mask = window_attention_mask
        b = window_brightness
        repair_indices = window_repair_indices
        seq_len = original_input_ids.shape[1]  # Update sequence length to window size
        window_offset = start_pos  # Store for position remapping later
    else:
        window_offset = 0  # No windowing applied
    
    # Create a masked version of input_ids
    masked_input_ids = original_input_ids.clone()
    
    # Process tokens based on repair_indices or brightness
    repair_positions = []
    
    # If repair_indices is provided, only consider those positions
    positions_to_check = repair_indices if repair_indices is not None else range(seq_len)
    
    for pos in positions_to_check:
        # Skip positions outside of sequence range
        if pos < 0 or pos >= seq_len:
            print(f"[Diffuser WARNING] Repair index {pos} out of bounds for sequence length {seq_len}")
            continue
            
        b_val = b[0, pos].item() if b.dim() > 1 else b[pos].item()
        
        # Skip positions where brightness >= lock threshold (these tokens are "frozen")
        if b_val >= config.BRIGHTNESS_LOCK_THRESHOLD:
            continue
            
        # For tokens below the lock threshold, mask them for MLM prediction
        original_id = original_input_ids[0, pos].item()
        masked_input_ids[0, pos] = diffuser_model.mask_token_id  # Replace with mask token
        repair_positions.append(pos)
    
    try:
        if not repair_positions:
            print(f"[Diffuser] No tokens to repair - all tokens are above the brightness threshold")
            return []
            
        print(f"[Diffuser] Masked {len(repair_positions)}/{seq_len} tokens for potential repair")
        
        # Execute the MLM model with masked inputs
        with torch.no_grad():
            outputs = diffuser_model.model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask
            )
            
        # Get logits and predicted tokens
        logits = outputs.logits  # Shape: [1, seq_len, vocab_size]
        
        # Compute diff only for masked positions
        for pos in repair_positions:
            original_id = original_input_ids[0, pos].item()
            
            # Get the token with highest probability for this position
            token_logits = logits[0, pos]
            predicted_id = torch.argmax(token_logits).item()
            
            # Only add to diff if prediction differs from original
            if predicted_id != original_id:
                # Optional: get prediction probability for logging
                probs = torch.softmax(token_logits, dim=0)
                prob = probs[predicted_id].item()
                
                # Remap position to original sequence position if windowing was applied
                orig_pos = pos + window_offset
                print(f"[Diffuser] Pos {orig_pos}: {original_id} -> {predicted_id} (prob: {prob:.4f})")
                diff_list.append((orig_pos, original_id, predicted_id))
    
    except Exception as e:
        inc_counter("diffuser_compute_diff_error")
        print(f"[Diffuser ERROR] Failed to compute diff: {str(e)}")
    
    return diff_list


# For legacy support - simplified masking-based version without brightness
@timed_histogram("diffuser_compute_diff_masking_seconds")
def compute_diff_masking(
    diffuser_model: DiffuserModel,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    token_ids: List[int],
    repair_indices: List[int]
) -> List[Tuple[int, int, int]]:
    """
    Legacy version of compute_diff using simple masking approach instead of noise.
    
    This function is kept for compatibility with older code that hasn't been updated
    to use the new brightness-guided noise diffusion approach.
    
    Args:
        diffuser_model: The DiffuserModel instance
        input_embeddings: Input embeddings tensor for the sequence
        attention_mask: Attention mask tensor
        token_ids: List of original token IDs corresponding to all positions in the sequence
        repair_indices: List of positions to repair
        
    Returns:
        List[Tuple[int, int, int]]: List of (position, old_token_id, new_token_id) tuples
    """
    # Ensure model is loaded
    if diffuser_model is None:
        print("[Diffuser ERROR] Diffuser model not loaded. Cannot compute diff.")
        return []
    
    diff_list = []
    device = diffuser_model.device
    
    # Ensure inputs are on the correct device
    input_embeddings = input_embeddings.to(device)
    attention_mask = attention_mask.to(device)
    
    # Create a masked copy of input embeddings
    masked_embeddings = input_embeddings.clone()
    
    try:
        # Get the mask token embedding
        mask_embedding = diffuser_model.get_mask_embedding()
        
        if mask_embedding is None:
            inc_counter("diffuser_missing_mask_embedding")
            print("[Diffuser ERROR] Mask token embedding is None. Cannot compute diff.")
            return []
        
        # Replace all repair positions with the mask token embedding
        for repair_idx in repair_indices:
            if 0 <= repair_idx < input_embeddings.shape[1]:
                # Replace the embedding at repair_idx with mask embedding
                masked_embeddings[:, repair_idx] = mask_embedding[:, 0]
            else:
                print(f"[Diffuser WARNING] Index {repair_idx} out of bounds for input of length {input_embeddings.shape[1]}")
        
        # Forward pass through the diffuser model
        with torch.no_grad():
            outputs = diffuser_model.model(
                inputs_embeds=masked_embeddings,
                attention_mask=attention_mask
            )
        
        # Process predictions for each repair position
        for repair_idx in repair_indices:
            if 0 <= repair_idx < input_embeddings.shape[1]:
                # Get the original token ID for this position
                original_token_id = token_ids[repair_idx]
                
                # Get logits for this position
                target_logits = outputs.logits[0, repair_idx, :]
                
                # Find the highest probability token
                predicted_token_id = torch.argmax(target_logits).item()
                
                # Only include in diff if prediction differs from original
                if predicted_token_id != original_token_id:
                    diff_list.append((repair_idx, original_token_id, predicted_token_id))
    
    except Exception as e:
        inc_counter("diffuser_compute_diff_error")
        print(f"[Diffuser ERROR] Failed to compute diff: {str(e)}")
    
    return diff_list


# For backward compatibility
@timed_histogram("diffuser_get_repaired_tokens_seconds")
def get_repaired_tokens(
    diffuser_model: DiffuserModel,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    token_ids: List[int],
    repair_indices: List[int]
) -> List[Tuple[int, int, int]]:
    """
    Legacy function that uses hidden states instead of input embeddings.
    Kept for backward compatibility.
    
    Args:
        diffuser_model: The DiffuserModel instance
        hidden_states: Hidden states from the last layer of the primary model
        attention_mask: Attention mask tensor
        token_ids: List of original token IDs corresponding to the positions
        repair_indices: List of positions to repair
        
    Returns:
        List[Tuple[int, int, int]]: List of (position, old_token_id, new_token_id) tuples
    """
    inc_counter("diffuser_using_legacy_get_repaired")
    print("[Diffuser WARNING] Using legacy get_repaired_tokens function with hidden states")
    
    # Treat hidden states as input embeddings for backward compatibility
    return compute_diff_masking(diffuser_model, hidden_states, attention_mask, token_ids, repair_indices)
