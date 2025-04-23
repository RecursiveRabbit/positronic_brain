"""
Tests for the Diffuser Runner module.

These tests validate that the Diffuser Runner correctly repairs tokens
using the diffuser model's masked language modeling capabilities.
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
from positronic_brain.diffuser_runner import (
    DiffuserModel, predict_replacement, compute_diff,
    repair_token, get_repaired_tokens  # Legacy functions kept for tests
)


class TestDiffuserRunner:
    
    def setup_mock_diffuser(self):
        """Set up a mock diffuser model for testing."""
        # Create a mock diffuser model
        mock_diffuser = MagicMock()
        mock_diffuser.device = torch.device("cpu")
        
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token_id = 103  # Standard BERT mask token ID
        mock_diffuser.tokenizer = mock_tokenizer
        mock_diffuser.mask_token_id = 103
        
        # Create mock embedding layer
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.return_value = torch.ones((1, 768), dtype=torch.float)  # Standard hidden dim for BERT
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = mock_embedding_fn
        
        # Create mock forward pass output
        mock_outputs = MagicMock()
        mock_logits = torch.zeros((1, 5, 30522))  # [batch, seq_len, vocab_size] for BERT
        
        # Set up different predictions for different positions
        # Position 1 will predict token ID 200 (same as original to test no-change)
        # Position 2 will predict token ID 1000 (different from original)
        # Position 3 will predict token ID 2000 (same as original to test no-change)
        mock_logits[0, 1, 200] = 10.0   # High logit for the same token
        mock_logits[0, 2, 1000] = 10.0  # High logit for a different token
        mock_logits[0, 3, 2000] = 10.0  # High logit for the same token
        
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        
        mock_diffuser.model = mock_model
        mask_embedding = torch.ones((1, 1, 768), dtype=torch.float)
        mock_diffuser.mask_token_embedding = mask_embedding
        mock_diffuser.get_mask_embedding = MagicMock(return_value=mask_embedding)
        
        return mock_diffuser
    
    def test_predict_replacement_basic(self):
        """Test basic token prediction functionality using new API."""
        # Set up mock diffuser model
        mock_diffuser = self.setup_mock_diffuser()
        
        # Create dummy input embeddings and attention mask
        input_embeddings = torch.ones((1, 5, 768), dtype=torch.float)  # [batch, seq_len, embed_dim]
        attention_mask = torch.ones((1, 5), dtype=torch.long)  # [batch, seq_len]
        
        # Test predicting replacement for token at position 2 with original token ID 500
        # Mock is set up to predict 1000, so should return a new token
        result = predict_replacement(
            mock_diffuser,
            input_embeddings,
            attention_mask,
            repair_index=2,
            original_token_id=500
        )
        
        # Verify result is the expected new token ID
        assert result == 1000
        
        # Test predicting replacement for token at position 3 with original token ID 2000
        # Mock is set up to predict 2000 (same), so should return None
        result = predict_replacement(
            mock_diffuser,
            input_embeddings,
            attention_mask,
            repair_index=3,
            original_token_id=2000
        )
        
        # Verify result is None (no change needed)
        assert result is None
        
    def test_legacy_repair_token(self):
        """Test that the legacy repair_token function still works."""
        # Set up mock diffuser model
        mock_diffuser = self.setup_mock_diffuser()
        
        # Create dummy input embeddings and attention mask
        hidden_states = torch.ones((1, 5, 768), dtype=torch.float)  # [batch, seq_len, hidden_dim]
        attention_mask = torch.ones((1, 5), dtype=torch.long)  # [batch, seq_len]
        
        # Test repairing token at position 2 with original token ID 500
        # Should work the same as predict_replacement
        result = repair_token(
            mock_diffuser,
            hidden_states,
            attention_mask,
            repair_index=2,
            original_token_id=500
        )
        
        # Verify result is the expected new token ID
        assert result == 1000
    
    def test_predict_replacement_with_error(self):
        """Test error handling during token prediction."""
        # Set up mock diffuser model
        mock_diffuser = self.setup_mock_diffuser()
        
        # Force model forward pass to raise an exception
        mock_diffuser.model.side_effect = RuntimeError("Simulated error in model forward pass")
        
        # Create dummy input embeddings and attention mask
        input_embeddings = torch.ones((1, 5, 768), dtype=torch.float)
        attention_mask = torch.ones((1, 5), dtype=torch.long)
        
        # Attempt to predict replacement - should handle the error and return None
        result = predict_replacement(
            mock_diffuser,
            input_embeddings,
            attention_mask,
            repair_index=2,
            original_token_id=500
        )
        
        # Verify result is None due to error
        assert result is None
    
    def test_compute_diff(self):
        """Test batch computation of token replacements."""
        # Set up mock diffuser model
        mock_diffuser = self.setup_mock_diffuser()
        
        # Create dummy input embeddings and attention mask
        input_embeddings = torch.ones((1, 5, 768), dtype=torch.float)  # [batch, seq_len, embed_dim]
        attention_mask = torch.ones((1, 5), dtype=torch.long)  # [batch, seq_len]
        
        # Original token IDs for the sequence
        token_ids = [100, 200, 500, 2000, 300]  # Position 2 will change, position 3 won't
        
        # Request repairs for positions 1, 2, 3, and an out-of-range position
        repair_indices = [1, 2, 3, 10]  # Position 10 is invalid
        
        # Get diff list using new API
        diff_list = compute_diff(
            mock_diffuser,
            input_embeddings,
            attention_mask,
            token_ids,
            repair_indices
        )
        
        # Verify results
        # Should contain a repair for position 2 where the predicted token is different
        # from the original - our mock is set up to only change token at position 2
        assert len(diff_list) == 1
        
        # Check position 2 repair
        assert diff_list[0][0] == 2  # Position
        assert diff_list[0][1] == 500  # Original token ID
        assert diff_list[0][2] == 1000  # New token ID
        
    def test_legacy_get_repaired_tokens(self):
        """Test the legacy get_repaired_tokens function."""
        # Set up mock diffuser model
        mock_diffuser = self.setup_mock_diffuser()
        
        # Create dummy input embeddings and attention mask
        hidden_states = torch.ones((1, 5, 768), dtype=torch.float)  # Using hidden states for legacy function
        attention_mask = torch.ones((1, 5), dtype=torch.long)
        
        # Original token IDs for the sequence
        token_ids = [100, 200, 500, 2000, 300]  # Position 2 will change, position 3 won't
        
        # Request repairs for positions 1, 2, 3
        repair_indices = [1, 2, 3]
        
        # Get repairs using legacy function
        repairs = get_repaired_tokens(
            mock_diffuser,
            hidden_states,
            attention_mask,
            token_ids,
            repair_indices
        )
        
        # Should work the same as compute_diff
        assert len(repairs) == 1
        assert repairs[0][0] == 2  # Position
        assert repairs[0][1] == 500  # Original token ID
        assert repairs[0][2] == 1000  # New token ID
        
    def test_compute_diff_error_handling(self):
        """Test error handling in compute_diff function."""
        # Test with None diffuser model
        diff_list = compute_diff(
            None,  # None diffuser model should be handled gracefully
            torch.ones((1, 5, 768)),
            torch.ones((1, 5)),
            [100, 200, 300, 400, 500],
            [1, 2, 3]
        )
        assert diff_list == [], "Should return empty list for None diffuser model"
        
        # Test with model that raises exception
        mock_diffuser = self.setup_mock_diffuser()
        mock_diffuser.model.side_effect = RuntimeError("Simulated error in batch processing")
        
        diff_list = compute_diff(
            mock_diffuser,
            torch.ones((1, 5, 768)),
            torch.ones((1, 5)),
            [100, 200, 300, 400, 500],
            [1, 2, 3]
        )
        assert diff_list == [], "Should return empty list when model throws exception"
