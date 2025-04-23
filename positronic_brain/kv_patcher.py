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
        
        # Detect model architecture for specialized handling
        self.model_type = self._detect_model_architecture()
        print(f"[KVPatcher] Detected model architecture: {self.model_type}")
        
        # Cache for architecture-specific parameters
        self.model_params = {
            'num_attention_heads': None,
            'num_key_value_heads': None,
            'head_dim': None,
            'hidden_size': None,
            'rope_theta': 10000.0,  # Default value, will be updated if available
        }
        
        # Try to extract key parameters from model config
        self._extract_model_parameters()
    
    def _debug_log(self, message: str) -> None:
        """Log debug messages if debug mode is enabled."""
        if self._debug_mode:
            print(f"[KVPatcher Debug] {message}")
            
    def _detect_model_architecture(self) -> str:
        """
        Detect the model architecture type based on various attributes.
        
        Returns:
            String identifier for the model type (e.g., 'kimi-vl', 'llama', 'mistral')
        """
        try:
            # Check config for model type information
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type'):
                model_type = self.model.config.model_type.lower()
                
                # Map to our supported architectural types
                if 'deepseek' in model_type or 'kimi' in model_type:
                    return 'kimi-vl'  # Handles Kimi-VL and DeepSeek models
                elif 'llama' in model_type:
                    return 'llama'
                elif 'mistral' in model_type:
                    return 'mistral'
                else:
                    self._debug_log(f"Using generic handling for model_type: {model_type}")
                    return 'generic'
            
            # Fallback detection based on layer structure
            if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
                # This pattern matches DeepSeek and Kimi models
                return 'kimi-vl'
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # Look for specific layer attributes to differentiate further
                if len(self.model.model.layers) > 0:
                    layer = self.model.model.layers[0]
                    # Mistral uses RMS Norm and has attention with specific structure
                    if hasattr(layer, 'input_layernorm') and hasattr(layer.self_attn, 'q_proj'):
                        if hasattr(layer, 'post_attention_layernorm'):
                            return 'mistral'  # Mistral has post_attention_layernorm
                        else:
                            return 'llama'     # LLaMa has similar structure
            
            # Fallback to generic
            return 'generic'
            
        except Exception as e:
            self._debug_log(f"Error in architecture detection: {e}")
            return 'generic'
            
    def _extract_model_parameters(self) -> None:
        """
        Extract key model parameters from the config for use in KV projection calculations.
        """
        try:
            config = self.model.config
            
            # Try to get key parameters
            param_mappings = {
                'num_attention_heads': ['num_attention_heads', 'n_head'],
                'num_key_value_heads': ['num_key_value_heads', 'n_kv_head'],
                'head_dim': ['head_dim'],
                'hidden_size': ['hidden_size', 'n_embd', 'hidden_dim'],
                'rope_theta': ['rope_theta', 'rotary_emb_base']
            }
            
            # Extract parameters using various possible attribute names
            for param, possible_names in param_mappings.items():
                for name in possible_names:
                    if hasattr(config, name):
                        self.model_params[param] = getattr(config, name)
                        break
            
            # If head_dim not found, try to calculate it
            if self.model_params['head_dim'] is None and self.model_params['hidden_size'] is not None:
                if self.model_params['num_attention_heads'] is not None:
                    self.model_params['head_dim'] = self.model_params['hidden_size'] // self.model_params['num_attention_heads']
            
            # If num_key_value_heads not found, assume equal to num_attention_heads (no GQA/MQA)
            if self.model_params['num_key_value_heads'] is None and self.model_params['num_attention_heads'] is not None:
                self.model_params['num_key_value_heads'] = self.model_params['num_attention_heads']
                
            # Debug log the extracted parameters
            self._debug_log(f"Model parameters: {self.model_params}")
            
        except Exception as e:
            self._debug_log(f"Error extracting model parameters: {e}")
    
    def _get_model_projections(self, layer_idx: int) -> Dict[str, Any]:
        """
        Get the projection matrices and related components for a specific layer.
        
        This method accesses the model's key and value projection matrices and
        any related components (like RoPE embeddings) for the specific layer,
        with handling for different model architectures.
        
        Args:
            layer_idx: Index of the layer to get projections for
            
        Returns:
            Dict containing architecture-specific projection components
        """
        try:
            result = {}
            
            # Handle different architectures based on detected model type
            if self.model_type == 'kimi-vl':
                # Kimi-VL / DeepSeek architecture with language_model.model path and low-rank KV projections
                if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
                    model_layers = self.model.language_model.model.layers
                    if layer_idx < len(model_layers):
                        layer = model_layers[layer_idx]
                        attn = layer.self_attn
                        
                        # Kimi-VL uses two-stage KV projections
                        # First stage: joint kv_a_proj with MQA
                        result['kv_a_proj'] = attn.kv_a_proj_with_mqa if hasattr(attn, 'kv_a_proj_with_mqa') else None
                        # Second stage: kv_b_proj
                        result['kv_b_proj'] = attn.kv_b_proj if hasattr(attn, 'kv_b_proj') else None
                        # Get rotary embeddings component
                        result['rotary_emb'] = attn.rotary_emb if hasattr(attn, 'rotary_emb') else None
                        # Get MQA repetition count if applicable
                        # Store size parameters (both Kimi-specific and generic)
                        if hasattr(attn, 'num_heads'):
                            result['num_heads'] = attn.num_heads
                        if hasattr(attn, 'num_kv_heads'):
                            result['num_kv_heads'] = attn.num_kv_heads
                        if hasattr(attn, 'head_dim'):
                            result['head_dim'] = attn.head_dim
                        
                        # Note: Kimi-VL applies RoPE only to part of the key vector
                        result['rope_dim'] = 64  # Default for Kimi-VL and DeepSeek
            
            elif self.model_type in ['llama', 'mistral', 'generic']:
                # LLaMA / Mistral style with direct k_proj and v_proj
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    model_layers = self.model.model.layers
                    if layer_idx < len(model_layers):
                        layer = model_layers[layer_idx]
                        if hasattr(layer, 'self_attn'):
                            attn = layer.self_attn
                            
                            # Direct K/V projections
                            result['k_proj'] = attn.k_proj if hasattr(attn, 'k_proj') else None
                            result['v_proj'] = attn.v_proj if hasattr(attn, 'v_proj') else None
                            # Get rotary embeddings component
                            result['rotary_emb'] = attn.rotary_emb if hasattr(attn, 'rotary_emb') else None
                            # Store size parameters
                            if hasattr(attn, 'num_heads'):
                                result['num_heads'] = attn.num_heads
                            if hasattr(attn, 'num_key_value_heads'): 
                                result['num_kv_heads'] = attn.num_key_value_heads
                            if hasattr(attn, 'head_dim'):
                                result['head_dim'] = attn.head_dim
                            
                            # For Mistral, we need to check for head repetition pattern (GQA)
                            if self.model_type == 'mistral':
                                if hasattr(attn, 'num_key_value_groups'):
                                    result['num_key_value_groups'] = attn.num_key_value_groups
                                    
            # Handle other models or add additional architectures here
                            
            # Fallback to global parameters if not found in the layer
            for param, global_value in self.model_params.items():
                if param not in result and global_value is not None:
                    result[param] = global_value
                    
            # Check if we got the essential components for the architecture
            if self.model_type == 'kimi-vl' and ('kv_a_proj' not in result or 'kv_b_proj' not in result):
                self._debug_log(f"Missing critical Kimi-VL projections for layer {layer_idx}")
            elif self.model_type in ['llama', 'mistral'] and ('k_proj' not in result or 'v_proj' not in result):
                self._debug_log(f"Missing critical LLaMA/Mistral projections for layer {layer_idx}")
                
            return result
            
        except Exception as e:
            self._debug_log(f"Error accessing projections for layer {layer_idx}: {e}")
            return {}

    def _apply_rotary_embedding(self, x: torch.Tensor, position: int, rotary_emb: Any, rope_dim: Optional[int] = None) -> torch.Tensor:
        """
        Apply Rotary Position Embeddings to a tensor at a specific position.
        
        Args:
            x: The tensor to apply rotations to [batch, heads, seq_len=1, head_dim]
            position: The absolute position in the sequence for this token
            rotary_emb: The rotary embedding module from the model
            rope_dim: If specified, only apply RoPE to the first rope_dim dimensions
            
        Returns:
            The tensor with rotary position embeddings applied
        """
        try:
            # Extract head dimension
            batch_size, num_heads, seq_len, head_dim = x.shape
            assert seq_len == 1, "Expected single token processing with seq_len=1"
            
            # Handle Kimi-VL/DeepSeek partial RoPE application
            if rope_dim is not None and rope_dim > 0 and rope_dim < head_dim:
                # Split the tensor into RoPE and non-RoPE parts
                x_rope_part = x[:, :, :, :rope_dim]
                x_non_rope_part = x[:, :, :, rope_dim:]
                
                # Check if rotary_emb module has a specific method for RoPE at position
                if hasattr(rotary_emb, 'apply_rotary_pos_emb_index'):
                    # Method that takes position index directly
                    x_rope_part = rotary_emb.apply_rotary_pos_emb_index(x_rope_part, position)
                elif hasattr(rotary_emb, 'apply_rotary_pos_emb'):
                    # Method that takes a cos/sin cache and position
                    # Create sin/cos for just this position
                    cos, sin = rotary_emb(x_rope_part, seq_len=1, position_ids=torch.tensor([position], device=x.device))
                    x_rope_part = rotary_emb.apply_rotary_pos_emb(x_rope_part, cos, sin)
                
                # Recombine the parts
                return torch.cat([x_rope_part, x_non_rope_part], dim=-1)
            else:
                # Apply RoPE to the full tensor
                if hasattr(rotary_emb, 'apply_rotary_pos_emb_index'):
                    # Method that takes position index directly
                    return rotary_emb.apply_rotary_pos_emb_index(x, position)
                elif hasattr(rotary_emb, 'apply_rotary_pos_emb'):
                    # Method that takes a cos/sin cache and position
                    cos, sin = rotary_emb(x, seq_len=1, position_ids=torch.tensor([position], device=x.device))
                    return rotary_emb.apply_rotary_pos_emb(x, cos, sin)
                else:
                    # Fallback when no standard method found
                    self._debug_log("No standard RoPE application method found in rotary_emb")
                    return x
        except Exception as e:
            self._debug_log(f"Error applying RoPE: {e}")
            # Return unmodified tensor as fallback
            return x
    
    def _calculate_kv_kimi(self, embedding: torch.Tensor, projections: Dict[str, Any], position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate key and value tensors for Kimi-VL/DeepSeek architecture.
        
        Args:
            embedding: The token embedding [1, hidden_size]
            projections: Dict with projection matrices and parameters
            position: The absolute position for this token (for RoPE)
            
        Returns:
            Tuple of (key_slice, value_slice) each with shape [1, num_kv_heads, 1, head_dim]
        """
        try:
            # Get the low-rank projection matrices and parameters
            kv_a_proj = projections['kv_a_proj']
            kv_b_proj = projections['kv_b_proj']
            rotary_emb = projections.get('rotary_emb')
            head_dim = projections.get('head_dim', 128)  # Default for Kimi-VL
            num_kv_heads = projections.get('num_kv_heads', 8)  # Default for Kimi-VL
            rope_dim = projections.get('rope_dim', 64)  # Partial RoPE application in Kimi-VL
            
            device = embedding.device
            dtype = embedding.dtype
            
            # Kimi-VL uses a two-stage, low-rank projection:
            # 1. First stage: joint KV projection with MQA
            kv_a = kv_a_proj(embedding)  # [1, 2*inner_dim] where inner_dim is smaller than final dim
            
            # 2. Split KV apart from the first projection
            inner_dim = kv_a.shape[-1] // 2
            k_a = kv_a[:, :inner_dim]
            v_a = kv_a[:, inner_dim:]
            
            # 3. Second stage: Project to final dimension
            k = kv_b_proj(k_a)  # [1, num_kv_heads * head_dim]
            v = kv_b_proj(v_a)  # [1, num_kv_heads * head_dim]
            
            # 4. Reshape for multi-head attention
            k = k.view(1, num_kv_heads, head_dim)
            v = v.view(1, num_kv_heads, head_dim)
            
            # 5. Add sequence dimension for compatibility with KV cache shape
            k = k.unsqueeze(2)  # [1, num_kv_heads, 1, head_dim]
            v = v.unsqueeze(2)  # [1, num_kv_heads, 1, head_dim]
            
            # 6. Apply rotary position embeddings to part of k
            if rotary_emb is not None:
                k = self._apply_rotary_embedding(k, position, rotary_emb, rope_dim)
            
            return k, v
            
        except Exception as e:
            self._debug_log(f"Error in Kimi-VL K/V calculation: {e}")
            # Return zeros as fallback with appropriate dimensions
            device = embedding.device
            dtype = embedding.dtype
            shape = (1, projections.get('num_kv_heads', 8), 1, projections.get('head_dim', 128))
            return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)
    
    def _calculate_kv_llama(self, embedding: torch.Tensor, projections: Dict[str, Any], position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate key and value tensors for LLaMA architecture.
        
        Args:
            embedding: The token embedding [1, hidden_size]
            projections: Dict with projection matrices and parameters
            position: The absolute position for this token (for RoPE)
            
        Returns:
            Tuple of (key_slice, value_slice) each with shape [1, num_kv_heads, 1, head_dim]
        """
        try:
            # Get the projection matrices and parameters
            k_proj = projections['k_proj']
            v_proj = projections['v_proj']
            rotary_emb = projections.get('rotary_emb')
            head_dim = projections.get('head_dim', 128)  # Default for LLaMA
            num_kv_heads = projections.get('num_kv_heads', 32)  # Default for LLaMA
            
            device = embedding.device
            dtype = embedding.dtype
            
            # 1. Direct projection to key/value spaces
            k = k_proj(embedding)  # [1, num_kv_heads * head_dim]
            v = v_proj(embedding)  # [1, num_kv_heads * head_dim]
            
            # 2. Reshape for multi-head attention
            k = k.view(1, num_kv_heads, head_dim)
            v = v.view(1, num_kv_heads, head_dim)
            
            # 3. Add sequence dimension for compatibility with KV cache shape
            k = k.unsqueeze(2)  # [1, num_kv_heads, 1, head_dim]
            v = v.unsqueeze(2)  # [1, num_kv_heads, 1, head_dim]
            
            # 4. Apply rotary position embeddings to the entire k vector
            if rotary_emb is not None:
                k = self._apply_rotary_embedding(k, position, rotary_emb)
            
            return k, v
            
        except Exception as e:
            self._debug_log(f"Error in LLaMA K/V calculation: {e}")
            # Return zeros as fallback with appropriate dimensions
            device = embedding.device
            dtype = embedding.dtype
            shape = (1, projections.get('num_kv_heads', 32), 1, projections.get('head_dim', 128))
            return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)
    
    def _calculate_kv_mistral(self, embedding: torch.Tensor, projections: Dict[str, Any], position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate key and value tensors for Mistral architecture (handles GQA).
        
        Args:
            embedding: The token embedding [1, hidden_size]
            projections: Dict with projection matrices and parameters
            position: The absolute position for this token (for RoPE)
            
        Returns:
            Tuple of (key_slice, value_slice) each with shape [1, num_kv_heads, 1, head_dim]
        """
        try:
            # Get the projection matrices and parameters
            k_proj = projections['k_proj']
            v_proj = projections['v_proj']
            rotary_emb = projections.get('rotary_emb')
            head_dim = projections.get('head_dim', 128)       # Default for Mistral
            num_kv_heads = projections.get('num_kv_heads', 8) # KV heads (smaller than attn heads in GQA)
            num_heads = projections.get('num_heads', 32)      # Full attention heads
            
            device = embedding.device
            dtype = embedding.dtype
            
            # Similar to LLaMA but with Grouped-Query Attention handling
            # 1. Direct projection to key/value spaces
            k = k_proj(embedding)  # [1, num_kv_heads * head_dim]
            v = v_proj(embedding)  # [1, num_kv_heads * head_dim]
            
            # 2. Reshape for multi-head attention
            k = k.view(1, num_kv_heads, head_dim)
            v = v.view(1, num_kv_heads, head_dim)
            
            # 3. Add sequence dimension for compatibility with KV cache shape
            k = k.unsqueeze(2)  # [1, num_kv_heads, 1, head_dim]
            v = v.unsqueeze(2)  # [1, num_kv_heads, 1, head_dim]
            
            # 4. Apply rotary position embeddings to the entire k vector
            if rotary_emb is not None:
                k = self._apply_rotary_embedding(k, position, rotary_emb)
            
            return k, v
            
        except Exception as e:
            self._debug_log(f"Error in Mistral K/V calculation: {e}")
            # Return zeros as fallback with appropriate dimensions
            device = embedding.device
            dtype = embedding.dtype
            shape = (1, projections.get('num_kv_heads', 8), 1, projections.get('head_dim', 128))
            return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)
    
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

    print(f"[KVPatcher] Applying {len(diff_list)} patches to KV cache using {self.model_type} architecture handler")

    try:
        # Get embedding layer on the correct device
        embeddings = self.model.get_input_embeddings()
        device = past_key_values[0][0].device  # Get device from cache tensors

        # Create tensor of new token IDs on the correct device
        new_token_ids = torch.tensor([new_id for _, _, new_id in diff_list], device=device)
        # Get new embeddings for all changed tokens in one go
        # Shape: [num_patches, hidden_dim]
        new_embeds = embeddings(new_token_ids)
        self._debug_log(f"Generated new embeddings shape: {new_embeds.shape}")
        
        # Extract positions and create mapping for lookup during patching
        # Create a position -> index mapping for faster lookup during patching
        pos_to_idx = {pos: i for i, (pos, _, _) in enumerate(diff_list)}

        # Create a mutable list of layers from the immutable cache tuple
        new_past_key_values_list = list(past_key_values)

        # Iterate through each layer in the cache
        for layer_idx in range(len(new_past_key_values_list)):
            # Get the key and value tensors for this layer
            key_states, value_states = new_past_key_values_list[layer_idx]
            
            # Check tensor shapes
            # Shape: [batch_size, num_heads, sequence_length, head_dim]
            batch_size, num_heads, seq_len, head_dim = key_states.shape
            self._debug_log(f"Layer {layer_idx} cache shapes: K={key_states.shape}, V={value_states.shape}")

            # Make copies to modify (prevents modifying views / in-place errors)
            new_key_states = key_states.clone()
            new_value_states = value_states.clone()

            # Get model-specific projection components for this layer
            projections = self._get_model_projections(layer_idx)
            
            # Optimization: Process unique positions to avoid duplicate work
            # The diffuser may rarely suggest multiple different tokens for the same position,
            # which we handle by only applying the latest change to each position.
            unique_positions = set(pos for pos, _, _ in diff_list)
            
            # Process each position
            for pos in unique_positions:
                # Skip invalid positions
                if not (0 <= pos < seq_len):
                    print(f"[KVPatcher Warning] Position {pos} out of bounds for layer {layer_idx} cache (len {seq_len})")
                    inc_counter("kv_patcher_out_of_bounds")
                    continue
                    
                # Find the index of this position in our diff_list (use the latest if multiple)
                candidates = [(i, new_id) for i, (p, _, new_id) in enumerate(diff_list) if p == pos]
                if not candidates:
                    continue  # Shouldn't happen with our unique_positions approach
                    
                # Use the last change for this position (diff_list is ordered)
                idx, new_id = candidates[-1]
                
                # Get the embedding for this token
                token_embedding = new_embeds[idx:idx+1]  # Keep batch dimension
                
                # Calculate K/V vectors using architecture-specific methods
                if self.model_type == 'kimi-vl':
                    k_slice, v_slice = self._calculate_kv_kimi(token_embedding, projections, pos)
                elif self.model_type == 'llama':
                    k_slice, v_slice = self._calculate_kv_llama(token_embedding, projections, pos)
                elif self.model_type == 'mistral':
                    k_slice, v_slice = self._calculate_kv_mistral(token_embedding, projections, pos)
                else:  # Generic fallback
                    # For unknown architectures, try the LLaMA approach first
                    try:
                        k_slice, v_slice = self._calculate_kv_llama(token_embedding, projections, pos)
                    except Exception as e:
                        self._debug_log(f"Generic fallback K/V calculation error: {e}")
                        # Create zero tensors as fallback
                        k_slice = torch.zeros((1, num_heads, 1, head_dim), device=device, dtype=key_states.dtype)
                        v_slice = torch.zeros((1, num_heads, 1, head_dim), device=device, dtype=value_states.dtype)
                
                # Handle GQA/MQA head repetition in Mistral if needed
                # In GQA, we may need to repeat each KV head across multiple query heads
                if self.model_type == 'mistral':
                    # Check if we need to handle GQA
                    num_kv_heads = projections.get('num_kv_heads', num_heads)
                    if num_kv_heads < num_heads:
                        # For Mistral with GQA, KV heads are repeated for multiple query heads
                        # If k_slice has fewer heads than the cache, we need to repeat them
                        if k_slice.shape[1] != num_heads:
                            repeats = num_heads // num_kv_heads
                            # Ensure we have the correct device and shape for efficient assignment
                            if repeats > 1:
                                self._debug_log(f"Expanding {num_kv_heads} KV heads to {num_heads} heads for GQA")
                                # For each KV head, it serves 'repeats' number of query heads
                                # We need to compute the target shape for efficient assignment later
                
                # Apply the calculated K/V slices to the cache at the target position
                try:
                    # Ensure shapes match expected dimensions
                    if k_slice.shape[1] == num_heads and v_slice.shape[1] == num_heads:
                        # Standard case - direct assignment
                        new_key_states[:, :, pos, :] = k_slice[:, :, 0, :]
                        new_value_states[:, :, pos, :] = v_slice[:, :, 0, :]
                    elif self.model_type == 'mistral' and k_slice.shape[1] < num_heads:
                        # Handle GQA case where we have fewer KV heads than attention heads
                        num_kv_heads = k_slice.shape[1]
                        # Each KV head maps to multiple attention heads
                        for i in range(num_heads):
                            kv_head_idx = i % num_kv_heads  # Map to corresponding KV head
                            new_key_states[:, i, pos, :] = k_slice[:, kv_head_idx, 0, :]
                            new_value_states[:, i, pos, :] = v_slice[:, kv_head_idx, 0, :]
                    else:
                        # Mismatch - log error but still try basic assignment
                        print(f"[KVPatcher Warning] Shape mismatch in layer {layer_idx}: cache has {num_heads} heads, calculated slices have {k_slice.shape[1]} heads")
                        # Try broadcasting assignment if sizes are compatible
                        if k_slice.shape[-1] == head_dim:
                            new_key_states[:, :, pos, :] = k_slice[:, 0, 0, :]
                            new_value_states[:, :, pos, :] = v_slice[:, 0, 0, :]
                            
                    inc_counter("kv_patcher_patches_applied")
                except Exception as e:
                    print(f"[KVPatcher Error] Failed to apply patch at position {pos} in layer {layer_idx}: {e}")
                    inc_counter("kv_patcher_error")

            # Update the layer tuple in the list
            new_past_key_values_list[layer_idx] = (new_key_states, new_value_states)
            
        # Convert back to tuple
        patched_kv_cache = tuple(new_past_key_values_list)
        print(f"[KVPatcher] Successfully applied {len(diff_list)} patches to KV cache")
        return patched_kv_cache

    except Exception as e:
        import traceback
        print(f"[KVPatcher Error] Failed to apply patches: {type(e).__name__} - {e}")
        print(traceback.format_exc())
        inc_counter("kv_patcher_error")
        # Return the *original* cache if patching fails to avoid corrupting state
        return past_key_values
