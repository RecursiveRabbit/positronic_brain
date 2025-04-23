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
    
    def setup_mock_model(self):
        """Set up a mock model for testing."""
        mock_model = MagicMock()
        
        # Create a mock embedding layer
        embedding_matrix = torch.nn.Embedding(100, 64)  # vocab_size=100, embed_dim=64
        mock_model.get_input_embeddings.return_value = embedding_matrix
        
        # Set up mock layer structure for _get_model_projections to find
        mock_layer = MagicMock()
        mock_attn = MagicMock()
        mock_attn.k_proj = MagicMock()
        mock_attn.v_proj = MagicMock()
        mock_layer.self_attn = mock_attn
        mock_model.layers = [mock_layer]  # Single layer for simplicity
        
        return mock_model
    
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
        
        # Since our patch implementation uses placeholder logic (zeros),
        # verify that the tensor slice at position 2 is now different
        # (should be zeros instead of ones)
        assert torch.all(patched_key[0, :, 2, :] == 0.0), "Key tensor not patched correctly"
        assert torch.all(patched_value[0, :, 2, :] == 0.0), "Value tensor not patched correctly"
        
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
