"""
Tests for the Brightness Engine module.

These tests validate that the Brightness Engine correctly calculates
and updates brightness scores for tokens based on attention patterns.
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

# Now we can safely import the modules
from positronic_brain.kv_mirror import KVMirror, ContextToken
from positronic_brain.brightness_engine import update_brightness_scores


class TestBrightnessEngine:
    def test_update_brightness_basic(self):
        """Test basic brightness updates based on attention scores."""
        # Create a mock KVMirror with some initial state
        kv_mirror = KVMirror()
        
        # Add some tokens with positions 0, 1, 2, 3, 4
        token_ids = [100, 101, 102, 103, 104]
        instance_ids = []
        for i, token_id in enumerate(token_ids):
            instance_id = kv_mirror.add(token_id, i)
            instance_ids.append(instance_id)
            
        # Mock the attentions tensor
        # Shape [batch, num_heads, seq_len, seq_len]
        # We'll use 1 batch, 2 heads, 5 sequence length
        mock_attentions = torch.zeros(1, 2, 5, 5)
        
        # Set attention values for the last token (idx 4) attending to others
        # High attention to tokens 1 and 3, low to others
        # Values across heads:
        # Token 0: [0.1, 0.2] -> avg 0.15
        # Token 1: [0.5, 0.7] -> avg 0.6
        # Token 2: [0.0, 0.1] -> avg 0.05
        # Token 3: [0.4, 0.6] -> avg 0.5
        mock_attentions[0, 0, 4, 0] = 0.1  # Head 0, token 4 attending to token 0
        mock_attentions[0, 1, 4, 0] = 0.2  # Head 1, token 4 attending to token 0
        
        mock_attentions[0, 0, 4, 1] = 0.5  # Head 0, token 4 attending to token 1
        mock_attentions[0, 1, 4, 1] = 0.7  # Head 1, token 4 attending to token 1
        
        mock_attentions[0, 0, 4, 2] = 0.0  # Head 0, token 4 attending to token 2
        mock_attentions[0, 1, 4, 2] = 0.1  # Head 1, token 4 attending to token 2
        
        mock_attentions[0, 0, 4, 3] = 0.4  # Head 0, token 4 attending to token 3
        mock_attentions[0, 1, 4, 3] = 0.6  # Head 1, token 4 attending to token 3
        
        # Create mock outputs object with attentions
        mock_outputs = MagicMock()
        mock_outputs.attentions = [mock_attentions]  # List with last layer's attention
        
        # Set test parameters
        alpha = 10.0
        beta = 0.01
        
        # Call the function
        results = update_brightness_scores(
            kv_mirror_manager=kv_mirror,
            outputs=mock_outputs,
            alpha=alpha,
            beta=beta
        )
        
        # Verify success counts
        assert results['success'] == 5  # Updates positions 0-4
        
        # Get updated brightness values
        snapshot = kv_mirror.snapshot()
        
        # Expected brightness for each token:
        # Starting with 255.0 for all
        # Token 0: 255 + (0.15 * 10) - 0.01 = 256.49 -> clamped to 255
        # Token 1: 255 + (0.6 * 10) - 0.01 = 260.99 -> clamped to 255
        # Token 2: 255 + (0.05 * 10) - 0.01 = 255.49 -> clamped to 255
        # Token 3: 255 + (0.5 * 10) - 0.01 = 259.99 -> clamped to 255
        # Due to clamping, all should still be at max brightness in this case
        
        # Check each token's brightness
        for i in range(4):
            assert snapshot['tokens'][instance_ids[i]].brightness == 255.0
        
        # Now let's modify the test to show brightness decreases
        # Reset the mirror with brightness already decreased
        kv_mirror.clear()
        instance_ids = []
        for i, token_id in enumerate(token_ids):
            # Set initial brightness to 100 (below max)
            instance_id = kv_mirror.add(token_id, i, brightness=100.0)
            instance_ids.append(instance_id)
        
        # Call again with same attention pattern
        results = update_brightness_scores(
            kv_mirror_manager=kv_mirror,
            outputs=mock_outputs,
            alpha=alpha,
            beta=beta
        )
        
        # Expected new brightness values:
        # Token 0: 100 + (0.15 * 10) - 0.01 = 101.49
        # Token 1: 100 + (0.6 * 10) - 0.01 = 105.99
        # Token 2: 100 + (0.05 * 10) - 0.01 = 100.49
        # Token 3: 100 + (0.5 * 10) - 0.01 = 104.99
        
        # Get updated values
        snapshot = kv_mirror.snapshot()
        
        # Verify with epsilon for floating point comparison
        epsilon = 0.01
        assert abs(snapshot['tokens'][instance_ids[0]].brightness - 101.49) < epsilon
        assert abs(snapshot['tokens'][instance_ids[1]].brightness - 105.99) < epsilon
        assert abs(snapshot['tokens'][instance_ids[2]].brightness - 100.49) < epsilon
        assert abs(snapshot['tokens'][instance_ids[3]].brightness - 104.99) < epsilon
        
    def test_brightness_clamping(self):
        """Test that brightness is correctly clamped to [0, 255] range."""
        # Create KVMirror
        kv_mirror = KVMirror()
        
        # Add one token at position 0, with brightness near max
        instance_id_high = kv_mirror.add(100, 0, brightness=254.0)
        
        # Add another token at position 1, with brightness near min
        instance_id_low = kv_mirror.add(101, 1, brightness=0.005)
        
        # Mock attentions tensor - very high attention to pos 0, zero to pos 1
        mock_attentions = torch.zeros(1, 1, 2, 2)
        mock_attentions[0, 0, 1, 0] = 5.0  # Extremely high attention to token 0
        mock_attentions[0, 0, 1, 1] = 0.0  # No attention to token 1
        
        # Create mock outputs
        mock_outputs = MagicMock()
        mock_outputs.attentions = [mock_attentions]
        
        # Set parameters: high alpha to push over max, high beta to push below min
        alpha = 5.0  # Will push token 0 over max
        beta = 0.01  # Will push token 1 below min
        
        # Call function
        results = update_brightness_scores(
            kv_mirror_manager=kv_mirror,
            outputs=mock_outputs,
            alpha=alpha,
            beta=beta
        )
        
        # Verify both were updated
        assert results['success'] == 2
        
        # Get updated values
        snapshot = kv_mirror.snapshot()
        
        # Token 0 should be clamped to max (255)
        # 254 + (5.0 * 5.0) - 0.01 = 278.99 -> clamped to 255
        assert snapshot['tokens'][instance_id_high].brightness == 255.0
        
        # Token 1 should be clamped to min (0)
        # 0.005 + (0.0 * 5.0) - 0.01 = -0.005 -> clamped to 0
        assert snapshot['tokens'][instance_id_low].brightness == 0.0
        
    def test_missing_attentions(self):
        """Test handling of missing attentions in the outputs object."""
        # Create KVMirror
        kv_mirror = KVMirror()
        
        # Add a token
        kv_mirror.add(100, 0)
        
        # Create mock outputs without attentions
        mock_outputs = MagicMock()
        mock_outputs.attentions = None
        
        # Call function
        results = update_brightness_scores(
            kv_mirror_manager=kv_mirror,
            outputs=mock_outputs,
            alpha=10.0,
            beta=0.01
        )
        
        # Verify function reported the error
        assert 'error' in results
        assert 'No attention data available' in results['error']
        
        # Create another mock with empty attentions list
        mock_outputs.attentions = []
        
        # This should raise an IndexError when trying to access [-1]
        results = update_brightness_scores(
            kv_mirror_manager=kv_mirror,
            outputs=mock_outputs,
            alpha=10.0,
            beta=0.01
        )
        
        # Should catch the exception and report an error
        assert 'error' in results
        
    def test_decay_without_attention(self):
        """Test brightness decay for tokens that receive no attention."""
        # Create KVMirror
        kv_mirror = KVMirror()
        
        # Add tokens with initial brightness of 100
        instance_ids = []
        for i in range(5):
            instance_id = kv_mirror.add(100 + i, i, brightness=100.0)
            instance_ids.append(instance_id)
        
        # Mock attentions tensor - zero attention to all tokens
        mock_attentions = torch.zeros(1, 1, 5, 5)
        
        # Create mock outputs
        mock_outputs = MagicMock()
        mock_outputs.attentions = [mock_attentions]
        
        # Set parameters with non-zero beta for decay
        alpha = 10.0
        beta = 1.0  # Large decay
        
        # Call function
        results = update_brightness_scores(
            kv_mirror_manager=kv_mirror,
            outputs=mock_outputs,
            alpha=alpha,
            beta=beta
        )
        
        # Verify all were updated
        assert results['success'] == 5
        
        # Get updated values
        snapshot = kv_mirror.snapshot()
        
        # All tokens should decay by beta
        # 100.0 + (0.0 * 10.0) - 1.0 = 99.0
        for instance_id in instance_ids:
            assert abs(snapshot['tokens'][instance_id].brightness - 99.0) < 0.01
