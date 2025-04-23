"""
Token sampling logic for the Infinite Scroll inference engine.
This module contains functions for sampling tokens based on model output logits
with various strategies like temperature, top-k, top-p, repetition penalty, etc.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

# Import the SamplerState dataclass definition 
from .sampler_types import SamplerState

# Import metrics decorator
from .metrics import timed_histogram


def top_p_filter(logits, top_p):
    """Filter logits using nucleus (top-p) sampling"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter sorted tensors back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits


def top_k_filter(logits, top_k):
    """Filter logits using top-k sampling"""
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = torch.topk(logits, top_k)[0][..., -1, None]
    logits = torch.where(logits < indices_to_remove, 
                        torch.ones_like(logits) * float('-inf'),
                        logits)
    return logits


def apply_repetition_penalty(logits, input_ids, penalty):
    """Apply repetition penalty to logits to reduce repetition"""
    for i in range(input_ids.shape[0]):
        for token_id in set(input_ids[i].tolist()):
            # If token has appeared before, penalize it
            if token_id in input_ids[i]:
                logits[i, token_id] /= penalty
    return logits


@timed_histogram("sampler_select_next_token_seconds")
def select_next_token(
    logits: torch.Tensor,
    input_ids: torch.Tensor,  # Needed for repetition penalty
    sampler_state: SamplerState  # Pass the current state object
) -> Tuple[int, torch.Tensor, List[Dict[str, float]]]:  # Return token_id, probs, top_token_info
    """
    Applies sampling logic (temp, top-k, top-p, repetition, bias) to logits and selects the next token.

    Args:
        logits: The raw logits output from the model for the next token (shape: [batch_size, vocab_size]).
        input_ids: The input IDs tensor generated so far (needed for repetition penalty).
        sampler_state: The current SamplerState configuration object.

    Returns:
        Tuple containing:
        - The selected next token ID (int).
        - The final probability distribution tensor after sampling filters.
        - Dictionary of top ~20 tokens and their probabilities for UI display.
    """
    next_token_logits = logits[:, -1, :]  # Usually logits are [batch, seq, vocab], take last token

    # Apply temperature scaling
    # Ensure temperature is not zero to avoid division errors
    temp = max(sampler_state.temperature, 1e-3)
    next_token_logits = next_token_logits / temp

    # Apply repetition penalty if enabled
    if sampler_state.repetition_penalty != 1.0:
        next_token_logits = apply_repetition_penalty(
            next_token_logits, input_ids, sampler_state.repetition_penalty
        )

    # Apply token bias if specified
    if sampler_state.token_bias:
        for tid, delta in sampler_state.token_bias.items():
            # Ensure token ID is within vocab bounds before biasing
            if 0 <= tid < next_token_logits.shape[-1]:
                next_token_logits[:, tid] += delta
            else:
                print(f"[Sampler Warning] Token ID {tid} out of bounds for biasing.")

    # --- Capture Top Tokens for UI (Before Filtering) ---
    top_token_info_for_ui = []
    try:
        # Calculate probabilities with softmax before filtering
        pre_filter_probs = F.softmax(next_token_logits[0], dim=-1)
        top_probs, top_indices = torch.topk(pre_filter_probs, k=20)

        for token_id_tensor, prob_tensor in zip(top_indices, top_probs):
            token_id = token_id_tensor.item()
            prob = prob_tensor.item()
            # Note: We can't easily decode here without the processor.
            # The UI part will need to handle decoding later.
            top_token_info_for_ui.append({
                'token_id': token_id,
                'probability': round(prob, 4)
                # 'token': processor.tokenizer.decode([token_id]) # Decode later
            })

    except Exception as e:
        print(f"[Sampler Error] Failed getting top tokens for UI: {e}")
        top_token_info_for_ui = []  # Return empty list on error

    # Apply top-p (nucleus) sampling if enabled
    if sampler_state.top_p < 1.0:
        next_token_logits = top_p_filter(next_token_logits, sampler_state.top_p)

    # Apply top-k sampling if enabled
    if sampler_state.top_k > 0:
        next_token_logits = top_k_filter(next_token_logits, sampler_state.top_k)

    # Apply softmax to get final probabilities after filtering
    probs = F.softmax(next_token_logits, dim=-1)

    # Sample from the distribution (or use greedy if temp=0 / effectively zero)
    if sampler_state.temperature > 1e-3:
        next_token_tensor = torch.multinomial(probs, num_samples=1)
    else:
        # Greedy decoding
        next_token_tensor = torch.argmax(probs, dim=-1, keepdim=True)

    selected_token_id = next_token_tensor.item()

    # Return ID, final probabilities, and top token info for UI
    return selected_token_id, probs, top_token_info_for_ui
