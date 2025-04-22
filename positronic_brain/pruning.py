import torch
from typing import Optional, Tuple, Dict
# Add KVMirror import to access snapshot for age/bias calculation
from .kv_mirror import KVMirror
from .metrics import timed_histogram  # Import the decorator

# --- Constants (Move from ai_core.py if desired, or keep them in config.py later) ---
CONTEXT_WINDOW_TARGET = 500  # Target size of KV cache after pruning
TEMPORAL_PENALTY_FACTOR = 0.005  # Factor for age-based pruning penalty

@timed_histogram("pruning_calculate_indices_seconds")
def calculate_biased_attention_pruning_indices(
    current_cache_len: int,
    kv_mirror_manager: KVMirror,  # Pass the manager instance
    outputs,  # Model outputs containing attentions
    device: torch.device
) -> Optional[torch.Tensor]:
    """
    Calculates which indices to keep based on attention scores, token age, and bias.

    Args:
        current_cache_len: The current length of the KV cache sequence.
        kv_mirror_manager: The KVMirror instance to get token metadata.
        outputs: The output object from the model's forward pass (must contain attentions).
        device: The torch device where calculations should occur.

    Returns:
        A tensor of indices to keep, or None if no pruning is needed or possible.
    """
    if current_cache_len <= CONTEXT_WINDOW_TARGET:
        return None  # No pruning needed

    num_to_remove = current_cache_len - CONTEXT_WINDOW_TARGET
    print(f"[Pruning Algo] Context length {current_cache_len} exceeds target {CONTEXT_WINDOW_TARGET}. Need to remove {num_to_remove} tokens.")

    attn_scores_raw = None
    pruning_scores = None  # Initialize pruning_scores

    # --- Extract Attention Scores ---
    try:
        if hasattr(outputs, 'attentions') and outputs.attentions:
            attentions = outputs.attentions[-1]
            last_pos_attentions = attentions[:, :, -1, :]  # Attention from last query to all keys
            attn_scores_raw = last_pos_attentions.mean(dim=1)  # Average across heads [batch_size, seq_len]

            # We only consider scores for the first CONTEXT_WINDOW_TARGET positions for pruning decisions
            # Note: The original code sliced to CONTEXT_WINDOW_TARGET. Let's rethink:
            # We should probably consider scores for *all* existing tokens up to current_cache_len
            # when deciding which to remove. Let's adjust this.
            attn_scores_raw = attn_scores_raw[0, :current_cache_len]  # Use scores up to current length
            attn_scores_raw = attn_scores_raw.to(device)  # Ensure correct device

            print(f"[Pruning Algo] Extracted attention scores shape: {attn_scores_raw.shape}")

            # --- Calculate Biased Scores (Age + Manual Bias) ---
            # Get necessary data from KVMirror snapshot
            snapshot = kv_mirror_manager.snapshot()
            kv_mirror_snapshot = snapshot['kv_mirror']
            token_registry_snapshot = snapshot['tokens']
            # Need the current step count for age calculation
            current_gen_step = kv_mirror_manager._global_generation_step  # Access internal step count

            token_ages = torch.zeros_like(attn_scores_raw)
            manual_biases = torch.zeros_like(attn_scores_raw)

            for pos in range(attn_scores_raw.shape[0]):  # Iterate up to current cache length
                if pos in kv_mirror_snapshot:
                    instance_id = kv_mirror_snapshot[pos]
                    if instance_id in token_registry_snapshot:
                        token = token_registry_snapshot[instance_id]
                        # Calculate age relative to current step
                        age = max(0, current_gen_step - token.instance_id)
                        token_ages[pos] = age
                        manual_biases[pos] = token.removal_bias
                    else:  # Should not happen if snapshot is consistent
                        manual_biases[pos] = 0.0
                else:  # Position might not be in mirror if pruning happened between steps? Unlikely.
                    manual_biases[pos] = 0.0

            # Calculate temporal penalty (ensure max_age > 0)
            max_age = max(1.0, token_ages.max().item())
            temporal_penalty = -TEMPORAL_PENALTY_FACTOR * (token_ages / max_age)

            # Calculate final pruning scores (higher = more likely to keep)
            pruning_scores = attn_scores_raw + temporal_penalty + manual_biases
            print(f"[Pruning Algo] Calculated final pruning scores.")

        else:
            print(f"[Pruning Algo] No attention scores available in model output.")
            # Fallback handled below if pruning_scores remain None

    except Exception as e:
        print(f"[Pruning Algo Error] Failed calculating biased scores: {type(e).__name__} - {e}")
        # Fallback handled below

    # --- Determine Indices to Remove ---
    indices_to_remove = []
    if pruning_scores is not None:
        try:
            # Find indices with the lowest scores (most eligible for pruning)
            # Ensure k is not larger than the number of scores available
            k_remove = min(num_to_remove, len(pruning_scores))
            if k_remove > 0:
                _, indices_to_remove_tensor = torch.topk(-pruning_scores, k=k_remove, largest=True, sorted=False)
                indices_to_remove = indices_to_remove_tensor.cpu().tolist()
                print(f"[Pruning Algo] Determined {len(indices_to_remove)} indices to remove based on scores.")
            else:
                print(f"[Pruning Algo] k_remove is {k_remove}, skipping score-based removal.")

        except Exception as e:
            print(f"[Pruning Algo Error] Failed score-based index selection: {type(e).__name__} - {e}")
            indices_to_remove = []  # Reset on error
    else:
        print("[Pruning Algo] No scores calculated, preparing for fallback.")

    # No fallback pruning - we never want to do FIFO/simple truncation
    if not indices_to_remove and num_to_remove > 0:
        print(f"[Pruning Algo WARNING] No indices selected for removal based on scores, and fallback is disabled. Skipping prune this step.")
        return None  # Explicitly return None if no indices were chosen

    # --- Calculate Indices to Keep ---
    # We've already handled the case where indices_to_remove is empty above

    # Create a set of indices to remove for efficient lookup
    indices_to_remove_set = set(indices_to_remove)

    # Build the list of indices to keep
    keep_indices_list = [i for i in range(current_cache_len) if i not in indices_to_remove_set]

    # Ensure the final length matches the target
    if len(keep_indices_list) != CONTEXT_WINDOW_TARGET:
        print(f"[Pruning Algo WARNING] Final keep list length ({len(keep_indices_list)}) doesn't match target ({CONTEXT_WINDOW_TARGET}). This might indicate an issue.")
        # As a safety measure, could truncate/adjust here, but better to flag the warning.
        # For now, return the calculated list, but this needs investigation if it occurs.

    keep_indices_tensor = torch.tensor(keep_indices_list, dtype=torch.long, device=device)
    print(f"[Pruning Algo] Calculated indices to keep, shape: {keep_indices_tensor.shape}")
    return keep_indices_tensor
