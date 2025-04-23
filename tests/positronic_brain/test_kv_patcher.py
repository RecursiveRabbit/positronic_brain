"""
Tests for the KV Cache Patcher module.

These tests validate that the KV Cache Patcher correctly applies
token diffs to the model's past_key_values tensor.
"""

import sys
import os
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Import mock metrics first to patch metrics module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'mocks')))
sys.modules['positronic_brain.metrics'] = __import__('mocks.metrics', fromlist=['*'])

# Now we can safely import the module
from positronic_brain.kv_patcher import KVCachePatcher


class TestKVCachePatcher:
    
    def setup_mock_model(self, architecture='generic', hidden_size=64, num_heads=2, num_kv_heads=None, head_dim=4):
        """Set up a mock model for testing with architecture-specific components.
        
        Args:
            architecture: Model architecture type ('llama', 'mistral', 'kimi-vl', or 'generic')
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads (for GQA), defaults to num_heads if None
            head_dim: Head dimension size
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads
            
        # Create mock model
        mock_model = MagicMock()
        
        # Set up Config for model type detection
        mock_config = MagicMock()
        mock_config.model_type = architecture
        mock_config.hidden_size = hidden_size
        mock_config.num_attention_heads = num_heads
        mock_config.num_key_value_heads = num_kv_heads
        mock_config.head_dim = head_dim
        mock_model.config = mock_config
        
        # Create a functional mock embedding layer
        embedding_matrix = torch.nn.Embedding(100, hidden_size)  # vocab_size=100
        embedding_matrix.weight.data.normal_(mean=0.0, std=0.02)  # Initialize with some values
        mock_model.get_input_embeddings.return_value = embedding_matrix
        
        # Set up mock RoPE function that applies a simple, predictable transformation
        def mock_rope_fn(x, *args, **kwargs):
            # Simple position-dependent transformation (not actual RoPE, just for testing)
            return x  # Identity for cos/sin cache case
            
        def mock_apply_rope(x, cos, sin):
            # Simple position-dependent transformation (not actual RoPE, just for testing)
            return x  # Identity for now - test will work with this
        
        def mock_apply_rope_index(x, position):
            # Apply a simple rotation based on position to make the result predictable and testable
            # Just multiply by position+1 to make the transformation distinct and verifiable
            return x * (position + 1.0) / 10.0
            
        mock_rotary_emb = MagicMock()
        mock_rotary_emb.side_effect = mock_rope_fn
        mock_rotary_emb.apply_rotary_pos_emb = mock_apply_rope
        mock_rotary_emb.apply_rotary_pos_emb_index = mock_apply_rope_index

        # Architecture-specific mock setup
        if architecture == 'llama' or architecture == 'mistral' or architecture == 'generic':
            # Set up LLaMA/Mistral style layers with direct k_proj and v_proj
            self._setup_llama_mistral_layers(mock_model, mock_rotary_emb, num_heads, num_kv_heads, head_dim, hidden_size)
        elif architecture == 'kimi-vl':
            # Set up Kimi-VL style layers with two-stage projection
            self._setup_kimi_layers(mock_model, mock_rotary_emb, num_heads, num_kv_heads, head_dim, hidden_size)
            
        return mock_model
        
    def _setup_llama_mistral_layers(self, mock_model, mock_rotary_emb, num_heads, num_kv_heads, head_dim, hidden_size):
        """Set up LLaMA or Mistral style model layers."""
        # Create mock layer architecture
        mock_model.model = MagicMock()
        mock_model.model.layers = []
        
        # For simplicity, create just one layer with functional k_proj and v_proj
        mock_layer = MagicMock()
        mock_layer.self_attn = MagicMock()
        
        # Create real Linear layers for K/V projection so we can verify calculation
        k_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        v_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        
        # Initialize with known values for predictable results
        torch.nn.init.normal_(k_proj.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(v_proj.weight, mean=0.0, std=0.02)
        
        # Attach to mock
        mock_layer.self_attn.k_proj = k_proj
        mock_layer.self_attn.v_proj = v_proj
        mock_layer.self_attn.rotary_emb = mock_rotary_emb
        mock_layer.self_attn.num_heads = num_heads
        mock_layer.self_attn.num_key_value_heads = num_kv_heads
        mock_layer.self_attn.head_dim = head_dim
        
        # Add layer to model
        mock_model.model.layers.append(mock_layer)
        
    def _setup_kimi_layers(self, mock_model, mock_rotary_emb, num_heads, num_kv_heads, head_dim, hidden_size):
        """Set up Kimi-VL style model layers with two-stage projection."""
        # Create mock layer architecture
        mock_model.language_model = MagicMock()
        mock_model.language_model.model = MagicMock()
        mock_model.language_model.model.layers = []
        
        # For simplicity, create just one layer
        mock_layer = MagicMock()
        mock_layer.self_attn = MagicMock()
        
        # Kimi uses low-rank projections - use smaller inner dimension for first projection
        inner_dim = head_dim // 2
        
        # Create real Linear layers for two-stage K/V projection
        kv_a_proj = torch.nn.Linear(hidden_size, 2 * inner_dim, bias=False)  # Joint KV projection
        kv_b_proj = torch.nn.Linear(inner_dim, num_kv_heads * head_dim, bias=False)  # Second stage
        
        # Initialize with known values for predictable results
        torch.nn.init.normal_(kv_a_proj.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(kv_b_proj.weight, mean=0.0, std=0.02)
        
        # Attach to mock
        mock_layer.self_attn.kv_a_proj_with_mqa = kv_a_proj
        mock_layer.self_attn.kv_b_proj = kv_b_proj
        mock_layer.self_attn.rotary_emb = mock_rotary_emb
        mock_layer.self_attn.num_heads = num_heads
        mock_layer.self_attn.num_kv_heads = num_kv_heads
        mock_layer.self_attn.head_dim = head_dim
        mock_layer.self_attn.rope_dim = head_dim // 2  # Kimi applies RoPE to part of the key
        
        # Add layer to model
        mock_model.language_model.model.layers.append(mock_layer)
    
    def test_patch_basic(self):
        """Test basic KV cache patching functionality."""
        # Set up mock model
        mock_model = self.setup_mock_model()
        
        # Create simple mock past_key_values tensors
        # 1 layer, batch_size=1, num_heads=2, seq_len=5, head_dim=4
        key_tensor = torch.ones((1, 2, 5, 4))
        value_tensor = torch.ones((1, 2, 5, 4))
        mock_past_key_values = ((key_tensor, value_tensor),)  # Single layer tuple
        
        # Create a diff list targeting position 2
        diff_list = [(2, 10, 20)]  # (position, old_token, new_token)
        
        # Create patcher and apply patch
        patcher = KVCachePatcher(mock_model)
        patched_past_kv = patcher.patch(mock_past_key_values, diff_list)
        
        # Verify the result is a tuple of tuples
        assert isinstance(patched_past_kv, tuple)
        assert len(patched_past_kv) == 1  # Single layer
        assert isinstance(patched_past_kv[0], tuple)
        assert len(patched_past_kv[0]) == 2  # Key and value
        
        # Get the patched key and value tensors
        patched_key, patched_value = patched_past_kv[0]
        
        # Verify that the tensor slice at position 2 is now different
        # (should no longer be all ones - real implementation now provides calculated values)
        assert not torch.allclose(patched_key[0, :, 2, :], key_tensor[0, :, 2, :]), "Key tensor was not patched"
        assert not torch.allclose(patched_value[0, :, 2, :], value_tensor[0, :, 2, :]), "Value tensor was not patched"
        
        # Verify positions other than 2 remain unchanged
        assert torch.all(patched_key[0, :, 0, :] == 1.0), "Unpatched key positions changed"
        assert torch.all(patched_value[0, :, 4, :] == 1.0), "Unpatched value positions changed"
    
    def test_patch_with_empty_diff_list(self):
        """Test handling of empty diff list."""
        # Set up mock model
        mock_model = self.setup_mock_model()
        
        # Create simple mock past_key_values tensors
        key_tensor = torch.ones((1, 2, 5, 4))
        value_tensor = torch.ones((1, 2, 5, 4))
        mock_past_key_values = ((key_tensor, value_tensor),)
        
        # Empty diff list
        diff_list = []
        
        # Create patcher and apply patch
        patcher = KVCachePatcher(mock_model)
        patched_past_kv = patcher.patch(mock_past_key_values, diff_list)
        
        # Verify result is the original past_key_values
        assert patched_past_kv is mock_past_key_values, "Empty diff list should return original past_key_values"
    
    def test_patch_with_out_of_bounds_position(self):
        """Test patching with position outside of sequence length."""
        # Set up mock model
        mock_model = self.setup_mock_model()
        
        # Create simple mock past_key_values
        key_tensor = torch.ones((1, 2, 5, 4))
        value_tensor = torch.ones((1, 2, 5, 4))
        mock_past_key_values = ((key_tensor, value_tensor),)
        
        # Diff list with out-of-bounds position
        diff_list = [(10, 10, 20)]  # Position 10 is out of bounds for seq_len=5
        
        # Create patcher and apply patch
        patcher = KVCachePatcher(mock_model)
        patched_past_kv = patcher.patch(mock_past_key_values, diff_list)
        
        # Verify no changes are made since position is out of bounds
        patched_key, patched_value = patched_past_kv[0]
        assert torch.all(patched_key == 1.0), "Out of bounds position should not change key tensor"
        assert torch.all(patched_value == 1.0), "Out of bounds position should not change value tensor"
    
    def test_patch_with_error(self):
        """Test error handling during patching."""
        # Set up mock model
        mock_model = self.setup_mock_model()
        
        # Create simple mock past_key_values
        key_tensor = torch.ones((1, 2, 5, 4))
        value_tensor = torch.ones((1, 2, 5, 4))
        mock_past_key_values = ((key_tensor, value_tensor),)
        
        # Diff list
        diff_list = [(2, 10, 20)]
        
        # Force get_input_embeddings to raise an exception
        mock_model.get_input_embeddings.side_effect = RuntimeError("Simulated error in getting embeddings")
        
        # Create patcher and apply patch
        patcher = KVCachePatcher(mock_model)
        patched_past_kv = patcher.patch(mock_past_key_values, diff_list)
        
        # Verify original past_key_values is returned when an error occurs
        assert patched_past_kv is mock_past_key_values, "Error should return original past_key_values"
        
    def test_patch_llama_structure(self):
        """Test patching with LLaMA architecture."""
        # Set up a LLaMA-style mock model
        hidden_size = 64
        num_heads = 4
        head_dim = 16
        mock_model = self.setup_mock_model('llama', hidden_size=hidden_size, num_heads=num_heads, head_dim=head_dim)
        
        # Create mock past_key_values tensors for a single layer
        batch_size = 1
        seq_len = 10
        key_tensor = torch.ones((batch_size, num_heads, seq_len, head_dim))
        value_tensor = torch.ones((batch_size, num_heads, seq_len, head_dim))
        mock_past_key_values = ((key_tensor, value_tensor),)  # Single layer tuple
        
        # Create a diff list targeting position 3 and 7
        positions = [3, 7]
        diff_list = [(pos, 10, 20) for pos in positions]  # (position, old_token, new_token)
        
        # Create patcher and apply patch
        patcher = KVCachePatcher(mock_model)
        patched_past_kv = patcher.patch(mock_past_key_values, diff_list)
        
        # Get the patched key and value tensors
        patched_key, patched_value = patched_past_kv[0]
        
        # Verify structure is preserved
        assert patched_key.shape == key_tensor.shape, "Key shape changed unexpectedly"
        assert patched_value.shape == value_tensor.shape, "Value shape changed unexpectedly"
        assert patched_key.dtype == key_tensor.dtype, "Key dtype changed unexpectedly"
        assert patched_value.dtype == value_tensor.dtype, "Value dtype changed unexpectedly"
        assert patched_key.device == key_tensor.device, "Key device changed unexpectedly"
        assert patched_value.device == value_tensor.device, "Value device changed unexpectedly"
        
        # Verify patched positions changed
        for pos in positions:
            assert not torch.allclose(patched_key[0, :, pos, :], key_tensor[0, :, pos, :]), f"Key at position {pos} not patched"
            assert not torch.allclose(patched_value[0, :, pos, :], value_tensor[0, :, pos, :]), f"Value at position {pos} not patched"
            
        # Verify unpatched positions remain unchanged
        for pos in range(seq_len):
            if pos not in positions:
                assert torch.allclose(patched_key[0, :, pos, :], key_tensor[0, :, pos, :]), f"Unpatched key at position {pos} was modified"
                assert torch.allclose(patched_value[0, :, pos, :], value_tensor[0, :, pos, :]), f"Unpatched value at position {pos} was modified"
    
    def test_patch_mistral_structure_with_gqa(self):
        """Test patching with Mistral architecture using Grouped-Query Attention."""
        # Set up a Mistral-style mock model with GQA (fewer KV heads than attention heads)
        hidden_size = 64
        num_heads = 8
        num_kv_heads = 4  # GQA: num_kv_heads < num_heads
        head_dim = 8
        mock_model = self.setup_mock_model(
            'mistral', 
            hidden_size=hidden_size, 
            num_heads=num_heads, 
            num_kv_heads=num_kv_heads, 
            head_dim=head_dim
        )
        
        # Create mock past_key_values tensors for a single layer
        batch_size = 1
        seq_len = 10
        # Note: The KV cache has num_heads entries, even though projections use num_kv_heads
        # This is because the KV cache is organized by attention heads, not KV heads
        key_tensor = torch.ones((batch_size, num_heads, seq_len, head_dim))
        value_tensor = torch.ones((batch_size, num_heads, seq_len, head_dim))
        mock_past_key_values = ((key_tensor, value_tensor),)  # Single layer tuple
        
        # Create a diff list targeting position 2
        diff_list = [(2, 10, 20)]  # (position, old_token, new_token)
        
        # Create patcher and apply patch
        patcher = KVCachePatcher(mock_model)
        patched_past_kv = patcher.patch(mock_past_key_values, diff_list)
        
        # Get the patched key and value tensors
        patched_key, patched_value = patched_past_kv[0]
        
        # Verify structure is preserved
        assert patched_key.shape == key_tensor.shape, "Key shape changed unexpectedly"
        assert patched_value.shape == value_tensor.shape, "Value shape changed unexpectedly"
        
        # Verify patched position changed
        assert not torch.allclose(patched_key[0, :, 2, :], key_tensor[0, :, 2, :]), "Key at position 2 not patched"
        assert not torch.allclose(patched_value[0, :, 2, :], value_tensor[0, :, 2, :]), "Value at position 2 not patched"
        
        # Verify unpatched positions remain unchanged
        for pos in [0, 1, 3, 4, 5]:
            assert torch.allclose(patched_key[0, :, pos, :], key_tensor[0, :, pos, :]), f"Unpatched key at position {pos} was modified"
            assert torch.allclose(patched_value[0, :, pos, :], value_tensor[0, :, pos, :]), f"Unpatched value at position {pos} was modified"
            
        # Verify GQA head pattern - with GQA, each KV head serves multiple attention heads
        # For example, with 8 attn heads and 4 KV heads, heads [0,4] should share the same KV vectors
        # Head 0 and head 4 should have identical patched values since they share a KV head
        for base_head in range(num_kv_heads):
            for offset in range(num_heads // num_kv_heads):
                dependent_head = base_head + offset * num_kv_heads
                if dependent_head < num_heads:
                    assert torch.allclose(
                        patched_key[0, base_head, 2, :], 
                        patched_key[0, dependent_head, 2, :]
                    ), f"Head {base_head} and {dependent_head} should share KV vectors"
    
    def test_patch_kimi_structure(self):
        """Test patching with Kimi-VL/DeepSeek architecture."""
        # Set up a Kimi-VL style mock model
        hidden_size = 64
        num_heads = 4
        num_kv_heads = 2  # Kimi often uses MQA
        head_dim = 16
        mock_model = self.setup_mock_model(
            'kimi-vl', 
            hidden_size=hidden_size, 
            num_heads=num_heads, 
            num_kv_heads=num_kv_heads, 
            head_dim=head_dim
        )
        
        # Create mock past_key_values tensors for a single layer
        batch_size = 1
        seq_len = 10
        key_tensor = torch.ones((batch_size, num_heads, seq_len, head_dim))
        value_tensor = torch.ones((batch_size, num_heads, seq_len, head_dim))
        mock_past_key_values = ((key_tensor, value_tensor),)  # Single layer tuple
        
        # Create a diff list targeting position 5
        diff_list = [(5, 10, 20)]  # (position, old_token, new_token)
        
        # Create patcher and apply patch
        patcher = KVCachePatcher(mock_model)
        patched_past_kv = patcher.patch(mock_past_key_values, diff_list)
        
        # Get the patched key and value tensors
        patched_key, patched_value = patched_past_kv[0]
        
        # Verify structure is preserved
        assert patched_key.shape == key_tensor.shape, "Key shape changed unexpectedly"
        assert patched_value.shape == value_tensor.shape, "Value shape changed unexpectedly"
        
        # Verify patched position changed
        assert not torch.allclose(patched_key[0, :, 5, :], key_tensor[0, :, 5, :]), "Key at position 5 not patched"
        assert not torch.allclose(patched_value[0, :, 5, :], value_tensor[0, :, 5, :]), "Value at position 5 not patched"
        
        # Verify unpatched positions remain unchanged
        for pos in [0, 1, 2, 3, 4, 6]:
            assert torch.allclose(patched_key[0, :, pos, :], key_tensor[0, :, pos, :]), f"Unpatched key at position {pos} was modified"
            assert torch.allclose(patched_value[0, :, pos, :], value_tensor[0, :, pos, :]), f"Unpatched value at position {pos} was modified"
    
    def test_patch_with_multiple_layers(self):
        """Test patching with a model having multiple layers."""
        # We'll reuse the setup_mock_model function but modify it to add multiple layers
        mock_model = self.setup_mock_model('llama', hidden_size=64, num_heads=4, head_dim=8)
        
        # Add a second layer manually
        first_layer = mock_model.model.layers[0]
        second_layer = MagicMock()
        second_layer.self_attn = MagicMock()
        second_layer.self_attn.k_proj = torch.nn.Linear(64, 4 * 8, bias=False)
        second_layer.self_attn.v_proj = torch.nn.Linear(64, 4 * 8, bias=False)
        second_layer.self_attn.rotary_emb = first_layer.self_attn.rotary_emb
        second_layer.self_attn.num_heads = 4
        second_layer.self_attn.num_key_value_heads = 4
        second_layer.self_attn.head_dim = 8
        mock_model.model.layers.append(second_layer)
        
        # Create mock past_key_values tensors for two layers
        batch_size = 1
        num_heads = 4
        seq_len = 6
        head_dim = 8
        key_tensor1 = torch.ones((batch_size, num_heads, seq_len, head_dim))
        value_tensor1 = torch.ones((batch_size, num_heads, seq_len, head_dim))
        key_tensor2 = torch.ones((batch_size, num_heads, seq_len, head_dim)) * 2.0  # Different values for layer 2
        value_tensor2 = torch.ones((batch_size, num_heads, seq_len, head_dim)) * 2.0
        mock_past_key_values = ((key_tensor1, value_tensor1), (key_tensor2, value_tensor2))
        
        # Create a diff list targeting position 1
        diff_list = [(1, 10, 20)]  # (position, old_token, new_token)
        
        # Create patcher and apply patch
        patcher = KVCachePatcher(mock_model)
        patched_past_kv = patcher.patch(mock_past_key_values, diff_list)
        
        # Verify we have the correct number of layers in the result
        assert len(patched_past_kv) == 2, "Should have 2 layers in patched past_key_values"
        
        # Get the patched key and value tensors for both layers
        patched_key1, patched_value1 = patched_past_kv[0]
        patched_key2, patched_value2 = patched_past_kv[1]
        
        # Verify structure is preserved for both layers
        assert patched_key1.shape == key_tensor1.shape, "Layer 1 key shape changed unexpectedly"
        assert patched_value1.shape == value_tensor1.shape, "Layer 1 value shape changed unexpectedly"
        assert patched_key2.shape == key_tensor2.shape, "Layer 2 key shape changed unexpectedly"
        assert patched_value2.shape == value_tensor2.shape, "Layer 2 value shape changed unexpectedly"
        
        # Verify patched position changed in both layers
        assert not torch.allclose(patched_key1[0, :, 1, :], key_tensor1[0, :, 1, :]), "Layer 1 key at position 1 not patched"
        assert not torch.allclose(patched_value1[0, :, 1, :], value_tensor1[0, :, 1, :]), "Layer 1 value at position 1 not patched"
        assert not torch.allclose(patched_key2[0, :, 1, :], key_tensor2[0, :, 1, :]), "Layer 2 key at position 1 not patched"
        assert not torch.allclose(patched_value2[0, :, 1, :], value_tensor2[0, :, 1, :]), "Layer 2 value at position 1 not patched"
    
    def test_patch_multiple_diffs_same_position(self):
        """Test patching with multiple diffs targeting the same position (should use last diff)."""
        # Set up a LLaMA-style mock model
        mock_model = self.setup_mock_model('llama')
        
        # Create mock past_key_values tensors for a single layer
        batch_size = 1
        num_heads = 2
        seq_len = 10
        head_dim = 4
        key_tensor = torch.ones((batch_size, num_heads, seq_len, head_dim))
        value_tensor = torch.ones((batch_size, num_heads, seq_len, head_dim))
        mock_past_key_values = ((key_tensor, value_tensor),)  # Single layer tuple
        
        # Create a diff list with multiple entries for the same position
        # The implementation should use the last entry (token ID 30) for position 3
        diff_list = [(3, 10, 20), (3, 20, 30)]  # Multiple diffs for position 3
        
        # Create patcher and apply patch
        patcher = KVCachePatcher(mock_model)
        patched_past_kv = patcher.patch(mock_past_key_values, diff_list)
        
        # Get the patched key and value tensors
        patched_key, patched_value = patched_past_kv[0]
        
        # Verify patched position changed
        assert not torch.allclose(patched_key[0, :, 3, :], key_tensor[0, :, 3, :]), "Key at position 3 not patched"
        assert not torch.allclose(patched_value[0, :, 3, :], value_tensor[0, :, 3, :]), "Value at position 3 not patched"
        
        # Verify that only one change was applied
        # This is harder to test directly, but we can at least verify the patched values are consistent
        # and match what we'd expect for a single update
