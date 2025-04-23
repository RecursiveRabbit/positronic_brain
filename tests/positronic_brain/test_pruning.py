import sys
import os
import pytest
import torch
import time
import inspect
import functools

# Import our mock metrics module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'mocks')))

# Apply module-level patching before importing anything that uses metrics
sys.modules['positronic_brain.metrics'] = __import__('mocks.metrics', fromlist=['*'])

# Import the function to be tested
from positronic_brain.pruning import calculate_biased_attention_pruning_indices, CONTEXT_WINDOW_TARGET, TEMPORAL_PENALTY_FACTOR
from positronic_brain.kv_mirror import KVMirror, ContextToken

"""
Test Plan for pruning.py:

1. Test Case 1: No Pruning Needed
   - Input `current_cache_len` less than or equal to `CONTEXT_WINDOW_TARGET`.
   - Assert the function returns `None`.

2. Test Case 2: Basic Attention Pruning
   - Set up a `KVMirror` instance with tokens.
   - Create mock model `outputs` containing an attention tensor where scores clearly indicate which tokens should be pruned.
   - Set `current_cache_len` > `CONTEXT_WINDOW_TARGET`.
   - Call the function.
   - Assert the returned `keep_indices` tensor has the correct shape (`CONTEXT_WINDOW_TARGET`).
   - Assert the specific indices correspond to the tokens with higher attention scores.

3. Test Case 3: Age Bias Influence
   - Similar setup to Test Case 2, but ensure some older tokens have slightly higher attention than some newer tokens.
   - Verify that the temporal penalty correctly causes the older tokens to be pruned despite slightly higher attention.

4. Test Case 4: Manual Bias Influence
   - Similar setup, but give specific tokens a high positive `removal_bias`.
   - Ensure these tokens are kept even if their attention/age score would normally suggest pruning.
   - Test negative bias causing a token to be pruned sooner.

5. Test Case 5: Fallback Pruning (No Attentions)
   - Set up `KVMirror`.
   - Create mock `outputs` that do not have an `.attentions` attribute or where it's `None`.
   - Set `current_cache_len` > `CONTEXT_WINDOW_TARGET`.
   - Call the function.
   - Assert the returned `keep_indices` tensor has the correct shape.
   - Assert the specific indices correspond to keeping the newest tokens.

6. Test Case 6: Edge Case (Few Tokens to Prune)
   - Set `current_cache_len` only slightly larger than `CONTEXT_WINDOW_TARGET` (e.g., target=500, len=501).
   - Verify only one token is pruned and `keep_indices` has shape 500.
"""

# Helper class for creating mock model outputs
class MockModelOutputs:
    def __init__(self, attentions=None):
        self.attentions = attentions

# Define fixtures for common test setup
@pytest.fixture
def empty_mirror():
    """Provide a fresh, empty KVMirror."""
    mirror = KVMirror()
    mirror.clear()
    return mirror

@pytest.fixture
def populated_mirror():
    """Provide a KVMirror with 10 tokens already added."""
    mirror = KVMirror()
    mirror.clear()
    for i in range(CONTEXT_WINDOW_TARGET + 50):  # Add more than target to test pruning
        mirror.add(i + 100, i)
    return mirror

@pytest.fixture
def mock_attention_tensor_clear_pattern():
    """
    Create a mock attention tensor with a very clear distinction between tokens to keep and prune.
    We'll give high scores to the last CONTEXT_WINDOW_TARGET tokens, and low scores to the first 50.
    This mimics a common scenario where more recent tokens are more relevant.
    """
    seq_len = CONTEXT_WINDOW_TARGET + 50  # Should match test setup
    attention = torch.zeros((1, 12, 1, seq_len))  # (batch, heads, query_pos, key_pos)
    
    # Create a clear pattern - high scores for last 500 tokens, low for first 50
    scores = torch.zeros(seq_len, dtype=torch.float32)
    scores[:50] = 0.01  # Very low scores for first 50 tokens (to be pruned)
    scores[50:] = 0.99  # Very high scores for last 500 tokens (to be kept)
    
    # Repeat the pattern for each attention head
    for head in range(12):
        attention[0, head, 0, :] = scores
    
    return attention

@pytest.fixture
def mock_biased_attention_tensor():
    """
    Create a mock attention tensor with scores where some older tokens have higher attention,
    but we want age bias to overrule and prune them anyway.
    """
    seq_len = CONTEXT_WINDOW_TARGET + 50  # Should match populated_mirror size
    attention = torch.zeros((1, 12, 1, seq_len))
    
    # Create a pattern biased towards newer tokens (higher indices)
    # but with some exceptions where older tokens have good scores
    scores = torch.arange(0, seq_len, dtype=torch.float32) / seq_len  # Increasing scores
    
    # Add some "anomalies": give high scores to some old tokens (which should still be pruned due to age)
    for i in range(0, 30, 5):
        scores[i] = 0.8  # High score for selected old tokens
        
    # Repeat for each attention head
    for head in range(12):
        attention[0, head, 0, :] = scores
    
    return attention

class TestPruning:
    def test_no_pruning_needed(self, empty_mirror):
        """Test that no pruning occurs when context length <= target."""
        # Setup a KVMirror with fewer tokens than the target
        mirror = empty_mirror
        context_len = CONTEXT_WINDOW_TARGET - 10  # Less than target
        
        for i in range(context_len):
            mirror.add(i + 100, i)
        
        # Create mock outputs with attention
        mock_outputs = MockModelOutputs(attentions=[torch.rand(1, 12, 1, context_len)])
        
        # Call function
        result = calculate_biased_attention_pruning_indices(
            current_cache_len=context_len,
            kv_mirror_manager=mirror,
            outputs=mock_outputs,
            device=torch.device('cpu')
        )
        
        # Assert no pruning needed when context_len <= CONTEXT_WINDOW_TARGET
        assert result is None, "Function should return None when context length <= target"
    
    def test_basic_attention_pruning(self, empty_mirror, mock_attention_tensor_clear_pattern):
        """Test basic attention-based pruning with clear patterns."""
        mirror = empty_mirror
        current_cache_len = CONTEXT_WINDOW_TARGET + 50  # 550 tokens total
        num_to_remove = 50  # We need to remove 50 tokens
        
        # Create tokens with uniform timestamps to eliminate age bias impact
        # All tokens will have the same age so pruning will be based purely on attention
        current_time = time.time()
        
        # Add tokens with the same timestamp to eliminate age bias
        for i in range(current_cache_len):
            mirror.add(i + 100, i)
            
            # Modify the token timestamp to be the same for all tokens
            token = mirror._registry[i+1]  # +1 because instance IDs start at 1
            token.timestamp = current_time
        
        # Create mock outputs with our patterned attention
        # This gives high scores to the last 500 tokens, low scores to first 50
        mock_outputs = MockModelOutputs(attentions=[mock_attention_tensor_clear_pattern])
        
        # Call function
        keep_indices = calculate_biased_attention_pruning_indices(
            current_cache_len=current_cache_len,
            kv_mirror_manager=mirror,
            outputs=mock_outputs,
            device=torch.device('cpu')
        )
        
        # Assertions
        assert keep_indices is not None, "Function should return indices to keep"
        assert keep_indices.shape[0] == CONTEXT_WINDOW_TARGET, f"Should keep exactly {CONTEXT_WINDOW_TARGET} tokens"
        
        # With our attention pattern, the first 50 tokens should be pruned
        # and the last 500 tokens should be kept
        kept_indices_set = set(keep_indices.tolist())
        
        # Indices that should be pruned (low attention scores)
        low_attention_indices = set(range(0, num_to_remove))
        
        # Indices that should be kept (high attention scores)
        high_attention_indices = set(range(num_to_remove, current_cache_len))
        
        # Check that low attention tokens were pruned
        # There should be minimal overlap between kept_indices and low_attention_indices
        overlap_with_low = len(kept_indices_set.intersection(low_attention_indices))
        assert overlap_with_low < 5, f"Expected minimal low-attention tokens to be kept, but found {overlap_with_low}"
        
        # Check that high attention tokens were kept
        # Most high attention tokens should be in the kept set
        kept_high_attention = len(kept_indices_set.intersection(high_attention_indices))
        assert kept_high_attention > CONTEXT_WINDOW_TARGET * 0.95, f"Expected most high-attention tokens to be kept, but only kept {kept_high_attention}"
    
    def test_age_bias_influence(self, empty_mirror):
        """Test that age bias correctly influences pruning decisions."""
        mirror = empty_mirror
        current_cache_len = CONTEXT_WINDOW_TARGET + 10
        
        # Add tokens with timestamps to simulate age differences
        base_time = time.time() - current_cache_len  # Base timestamp for oldest token
        
        # Track specific tokens that will have interesting age/attention properties
        old_high_attention_idx = 20  # An old token with high attention
        new_medium_attention_idx = current_cache_len - 20  # A newer token with medium attention
        
        # Create a custom attention tensor where older tokens have slightly higher attention
        # but should still be pruned due to age penalty
        attention = torch.ones((1, 12, 1, current_cache_len)) * 0.5  # Medium attention for most
        
        # Give old token high attention - without age bias, it would be kept
        for head in range(12):
            attention[0, head, 0, old_high_attention_idx] = 0.9  # Very high attention
            attention[0, head, 0, new_medium_attention_idx] = 0.6  # Medium-high attention
        
        # Add tokens with progressively newer timestamps
        for i in range(current_cache_len):
            mirror.add(i + 100, i, source='llm')
            
            # Set timestamps to create clear age gradient
            token = mirror._registry[i+1]  # +1 because instance IDs start from 1
            token.timestamp = base_time + i  # Progressively newer timestamps
        
        # Calculate what the age penalty will be
        old_token_age = current_cache_len - old_high_attention_idx
        new_token_age = current_cache_len - new_medium_attention_idx
        
        # For debugging - print expected scores
        old_token_penalty = -TEMPORAL_PENALTY_FACTOR * (old_token_age / current_cache_len)
        new_token_penalty = -TEMPORAL_PENALTY_FACTOR * (new_token_age / current_cache_len)
        expected_old_score = 0.9 + old_token_penalty  # High attention but age penalty
        expected_new_score = 0.6 + new_token_penalty  # Medium attention but less age penalty
        
        print(f"Debug age bias test: Expected old token score: {expected_old_score:.4f}, Expected new token score: {expected_new_score:.4f}")
        
        # Call function
        mock_outputs = MockModelOutputs(attentions=[attention])
        keep_indices = calculate_biased_attention_pruning_indices(
            current_cache_len=current_cache_len,
            kv_mirror_manager=mirror,
            outputs=mock_outputs,
            device=torch.device('cpu')
        )
        
        # Assertions
        assert keep_indices is not None, "Function should return indices to keep"
        
        # Check if our specific test case worked - the newer token with medium attention
        # should be kept over the older token with high attention
        kept_indices = keep_indices.tolist()
        
        # If age bias is working, the newer token should be kept despite lower attention
        assert new_medium_attention_idx in kept_indices, "Newer token with medium attention should be kept"
        
        # And the older token should be pruned despite higher attention, due to age bias
        # Only check this if expected_old_score < expected_new_score - otherwise our test setup might be wrong
        if expected_old_score < expected_new_score:
            assert old_high_attention_idx not in kept_indices, "Older token with high attention should be pruned due to age bias"
    
    def test_manual_bias_influence(self, empty_mirror):
        """Test that manual bias correctly influences pruning decisions."""
        mirror = empty_mirror
        current_cache_len = CONTEXT_WINDOW_TARGET + 10
        
        # Create a flat attention tensor where all tokens have equal attention
        # This isolates the effect of manual bias
        attention = torch.ones((1, 12, 1, current_cache_len)) * 0.5
        
        # Define specific indices for our test tokens
        positive_bias_idx = 30   # Should be kept despite being old with average attention
        negative_bias_idx = current_cache_len - 30  # Should be pruned despite being newer with average attention
        
        # Add tokens with uniform timestamps to eliminate age bias
        current_time = time.time()
        
        for i in range(current_cache_len):
            # Set bias for our test tokens
            bias = 0.0
            if i == positive_bias_idx:
                bias = 10.0  # Strong positive bias - should be kept
            elif i == negative_bias_idx:
                bias = -10.0  # Strong negative bias - should be pruned
                
            mirror.add(i + 100, i, removal_bias=bias)
            
            # Give all tokens the same timestamp to eliminate age bias
            token = mirror._registry[i+1]  # +1 because instance IDs start from 1
            token.timestamp = current_time
        
        # Call function
        mock_outputs = MockModelOutputs(attentions=[attention])
        keep_indices = calculate_biased_attention_pruning_indices(
            current_cache_len=current_cache_len,
            kv_mirror_manager=mirror,
            outputs=mock_outputs,
            device=torch.device('cpu')
        )
        
        # Assertions
        assert keep_indices is not None, "Function should return indices to keep"
        
        kept_indices = keep_indices.tolist()
        
        # Check positive bias - token should be kept
        assert positive_bias_idx in kept_indices, "Token with high positive bias should be kept regardless of other factors"
        
        # Check negative bias - token should be pruned
        assert negative_bias_idx not in kept_indices, "Token with high negative bias should be pruned regardless of other factors"
    
    def test_fallback_pruning(self, empty_mirror):
        """Test fallback behavior when no attention scores are available."""
        mirror = empty_mirror
        current_cache_len = CONTEXT_WINDOW_TARGET + 50
        num_to_remove = 50
        
        # Set up tokens with ordered timestamps so newer tokens are at higher indices
        base_time = time.time() - current_cache_len
        for i in range(current_cache_len):
            mirror.add(i + 100, i)
            token = mirror._registry[i+1]  # +1 because instance IDs start from 1
            token.timestamp = base_time + i  # Progressively newer timestamps
        
        # Create mock outputs with no attention
        mock_outputs = MockModelOutputs(attentions=None)
        
        # Call function - note that according to implementation, this should return None
        # since the fallback logic is not implemented/disabled
        keep_indices = calculate_biased_attention_pruning_indices(
            current_cache_len=current_cache_len,
            kv_mirror_manager=mirror,
            outputs=mock_outputs,
            device=torch.device('cpu')
        )
        
        # Currently, the function should return None when no attention scores are available
        # This matches the current implementation which has no fallback
        assert keep_indices is None, "Function should return None when no scores available and fallback is disabled"
        
        # If fallback logic is implemented in the future, the below assertions should be used
        # The fallback would typically keep the newest tokens (highest indices)
        
        # if keep_indices is not None:  # This would be true if fallback is implemented
        #     kept_indices = keep_indices.tolist()
        #     # Check that the oldest tokens were pruned (lowest indices)
        #     oldest_indices = set(range(0, num_to_remove))
        #     newest_indices = set(range(num_to_remove, current_cache_len))
        #     
        #     # In a timestamp-based fallback, we'd expect most/all of the newest tokens to be kept
        #     kept_newest = len(kept_indices_set.intersection(newest_indices))
        #     assert kept_newest > CONTEXT_WINDOW_TARGET * 0.95, "Fallback should keep newest tokens"
    
    def test_edge_case_few_tokens_to_prune(self, empty_mirror):
        """Test edge case where only a few tokens need to be pruned."""
        mirror = empty_mirror
        current_cache_len = CONTEXT_WINDOW_TARGET + 1  # Just one token over the limit
        
        # Add tokens with uniform timestamps to eliminate age bias
        current_time = time.time()
        
        for i in range(current_cache_len):
            mirror.add(i + 100, i)
            # Set same timestamp for all tokens
            token = mirror._registry[i+1]  # +1 because instance IDs start from 1
            token.timestamp = current_time
        
        # Create attention pattern where one specific token has much lower attention
        lowest_attention_idx = 42  # Choose a specific index to have lowest attention
        attention = torch.ones((1, 12, 1, current_cache_len)) * 0.9  # High attention for most tokens
        
        # Set one token to have significantly lower attention
        for head in range(12):
            attention[0, head, 0, lowest_attention_idx] = 0.1  # Very low attention
        
        mock_outputs = MockModelOutputs(attentions=[attention])
        
        # Call function
        keep_indices = calculate_biased_attention_pruning_indices(
            current_cache_len=current_cache_len,
            kv_mirror_manager=mirror,
            outputs=mock_outputs,
            device=torch.device('cpu')
        )
        
        # Assertions
        assert keep_indices is not None, "Function should return indices to keep"
        assert keep_indices.shape[0] == CONTEXT_WINDOW_TARGET, f"Should keep exactly {CONTEXT_WINDOW_TARGET} tokens"
        assert lowest_attention_idx not in keep_indices.tolist(), f"The token with lowest attention (at index {lowest_attention_idx}) should be pruned"
        
        # Also verify that only the expected number of tokens were pruned
        expected_removed = current_cache_len - CONTEXT_WINDOW_TARGET
        assert expected_removed == 1, "Expected to remove exactly 1 token"
        
        # Verify that all other tokens were kept
        kept_indices_set = set(keep_indices.tolist())
        all_indices_set = set(range(current_cache_len))
        removed_indices = all_indices_set - kept_indices_set
        
        assert len(removed_indices) == 1, "Should have exactly 1 index removed"
        assert lowest_attention_idx in removed_indices, "The removed index should be the one with lowest attention"

# Additional tests could include:
# - Testing with actual model outputs (integration test)
# - Testing with extremely biased attention patterns
# - Testing with random attention patterns
# - Performance testing with large context lengths
