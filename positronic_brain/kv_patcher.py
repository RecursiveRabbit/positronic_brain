"""
KV Cache Patcher for the Positronic Brain / Halo Weave system.

This module is responsible for applying token diffs (from the diffuser)
directly to the main LLM's past_key_values tensor. It bridges the gap between
the diffuser's repair suggestions and the actual state used for generation.
"""

import torch
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoModelForCausalLM
from . import config
from .metrics import timed_histogram, inc_counter


class KVCachePatcher:
    """
    KVCachePatcher applies token changes from diffs to the model's KV cache.
    
    This component is responsible for the crucial task of modifying the model's
    past_key_values tensor based on token replacement suggestions from the diffuser.
    It ensures that repaired tokens affect subsequent generation.
    """
    
    def __init__(self, model: AutoModelForCausalLM):
        """
        Initializes the patcher with the main LLM.

        Args:
            model: The main AutoModelForCausalLM instance.
        """
        self.model = model
        # We fetch embeddings on demand rather than caching them,
        # as the model might move between devices or have its weights updated
        self._debug_mode = config.DEBUG_MODE if hasattr(config, 'DEBUG_MODE') else False
    
    def _debug_log(self, message: str) -> None:
        """Log debug messages if debug mode is enabled."""
        if self._debug_mode:
            print(f"[KVPatcher Debug] {message}")
    
    def _get_model_projections(self, layer_idx: int) -> Dict[str, Any]:
        """
        Get the projection matrices for a specific layer.
        
        This is a placeholder implementation that attempts to access the model's
        key and value projection matrices for a specific layer. The actual
        implementation depends heavily on the model architecture.
        
        Args:
            layer_idx: Index of the layer to get projections for
            
        Returns:
            Dict containing 'k_proj' and 'v_proj' keys with the projection layers
        """
        try:
            # This is model-architecture dependent and will need to be adapted
            # based on the specific model being used
            # Common architectures:
            
            # For decoder-only transformers like GPT:
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                # GPT-2 style
                layer = self.model.transformer.h[layer_idx]
                attn = layer.attn
                return {'k_proj': attn.c_attn, 'v_proj': attn.c_attn}  # GPT uses a combined projection
                
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # LLaMA style
                layer = self.model.model.layers[layer_idx]
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    return {'k_proj': attn.k_proj, 'v_proj': attn.v_proj}
                    
            elif hasattr(self.model, 'layers'):
                # Direct layers attribute
                layer = self.model.layers[layer_idx]
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    return {'k_proj': attn.k_proj, 'v_proj': attn.v_proj}
                    
            # Fallback - architecture not recognized
            self._debug_log(f"Could not identify model architecture for layer {layer_idx}")
            return {'k_proj': None, 'v_proj': None}
            
        except Exception as e:
            self._debug_log(f"Error accessing projections for layer {layer_idx}: {e}")
            return {'k_proj': None, 'v_proj': None}

    @timed_histogram("kv_patcher_patch_seconds")
    def patch(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]],
        diff_list: List[Tuple[int, int, int]]  # [(position, old_token_id, new_token_id)]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Applies token changes from a diff list directly to the KV cache tensors.

        Args:
            past_key_values: The current KV cache tuple (layers x (key, value)).
            diff_list: A list of changes: (position_index, old_id, new_id).

        Returns:
            The *modified* past_key_values tuple.
        """
        if not diff_list:
            return past_key_values  # No changes needed

        print(f"[KVPatcher] Applying {len(diff_list)} patches to KV cache.")

        try:
            # Get embedding layer on the correct device
            embeddings = self.model.get_input_embeddings()
            device = past_key_values[0][0].device  # Get device from cache tensors

            # Create tensor of new token IDs on the correct device
            new_token_ids = torch.tensor([new_id for _, _, new_id in diff_list], device=device)
            positions_to_patch = [pos for pos, _, _ in diff_list]
            # Get new embeddings for all changed tokens in one go
            # Shape: [num_patches, hidden_dim]
            new_embeds = embeddings(new_token_ids)
            self._debug_log(f"Generated new embeddings shape: {new_embeds.shape}")

            # Create a mutable list of layers from the immutable cache tuple
            new_past_key_values_list = list(past_key_values)

            # Iterate through each layer in the cache
            for layer_idx in range(len(new_past_key_values_list)):
                # Get the key and value tensors for this layer
                key_states, value_states = new_past_key_values_list[layer_idx]
                
                # Shape: [batch_size, num_heads, sequence_length, head_dim]
                # Assume batch_size = 1 for now
                batch_size, num_heads, seq_len, head_dim = key_states.shape
                self._debug_log(f"Layer {layer_idx} cache shapes: K={key_states.shape}, V={value_states.shape}")

                # Make copies to modify if tensors are views / prevent in-place errors
                new_key_states = key_states.clone()
                new_value_states = value_states.clone()

                # --- Re-calculate K/V projections for the new embeddings ---
                # Get the K/V projection layers for this layer
                projections = self._get_model_projections(layer_idx)
                k_proj, v_proj = projections.get('k_proj'), projections.get('v_proj')

                for i, pos in enumerate(positions_to_patch):
                    # Ensure position is valid for this cache
                    if 0 <= pos < seq_len:
                        # PLACEHOLDER LOGIC - This needs to be model-specific in the future
                        # In a real implementation, we would:
                        # 1. Get the embedding for the new token
                        # 2. Apply the K and V projections for this layer
                        # 3. Reshape to match the cache tensor dimensions
                        # 4. Insert at the appropriate position
                        
                        # For now, we'll use zeros as a placeholder
                        # In the future, we'd calculate:
                        # new_k = k_proj(new_embeds[i:i+1]).view(batch_size, num_heads, 1, head_dim)
                        # new_v = v_proj(new_embeds[i:i+1]).view(batch_size, num_heads, 1, head_dim)
                        
                        self._debug_log(f"Patching position {pos} with token ID {new_token_ids[i]}")
                        new_key_states[:, :, pos, :] = 0.0  # Placeholder
                        new_value_states[:, :, pos, :] = 0.0  # Placeholder
                        inc_counter("kv_patcher_patches_applied")
                    else:
                        print(f"[KVPatcher Warning] Position {pos} out of bounds for layer {layer_idx} cache (len {seq_len})")
                        inc_counter("kv_patcher_out_of_bounds")

                # Update the layer tuple in the list
                new_past_key_values_list[layer_idx] = (new_key_states, new_value_states)

            # Convert back to tuple
            return tuple(new_past_key_values_list)

        except Exception as e:
            import traceback
            print(f"[KVPatcher Error] Failed to apply patches: {type(e).__name__} - {e}")
            print(traceback.format_exc())
            inc_counter("kv_patcher_error")
            # Return the *original* cache if patching fails to avoid corrupting state
            return past_key_values
