"""
Helper functions for the AI Core sampling operations
"""

import torch

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
