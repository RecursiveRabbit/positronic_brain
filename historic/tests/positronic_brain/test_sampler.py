import sys
import os
import pytest
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# Import mock metrics first to patch metrics module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'mocks')))
sys.modules['positronic_brain.metrics'] = __import__('mocks.metrics', fromlist=['*'])

# Import the real SamplerState definition
from positronic_brain.sampler_types import SamplerState

# Implement the helper functions
def top_p_filter(logits, top_p):
    """Implementation of top_p_filter for testing"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    return logits

def top_k_filter(logits, top_k):
    """Implementation of top_k_filter for testing"""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    return logits

def apply_repetition_penalty(logits, input_ids, penalty):
    """Implementation of repetition penalty for testing"""
    for token_id in set(input_ids.view(-1).tolist()):
        if token_id < logits.shape[-1]:
            # If token is in vocab
            # Apply penalty: reduce logits if token has already been generated
            # Increase logits if token hasn't been seen yet
            if logits[:, token_id] > 0:
                logits[:, token_id] /= penalty
            else:
                logits[:, token_id] *= penalty
    return logits

# Now that the helper functions have been moved to sampler.py, we don't need to mock them anymore.
# Import the real functions directly
from positronic_brain.sampler import top_p_filter, top_k_filter, apply_repetition_penalty

# Now try to import the actual function to test
from positronic_brain.sampler import select_next_token

"""
Test Plan for sampler.py:

1. Test Case 1: Greedy Decoding (Temp=0)
   - Set temperature=0 (or very small)
   - Provide logits where one token has a clearly higher logit
   - Assert the returned selected_token_id is the index of that highest logit

2. Test Case 2: Temperature Scaling
   - Provide distinct logits
   - Call with temperature=1.0 and temperature=0.5
   - Assert the relative probabilities in the returned probs tensor change as expected
     (distribution becomes sharper with lower temp)
   - Use torch.allclose for comparing probability tensors

3. Test Case 3: Repetition Penalty
   - Provide input_ids containing a specific token ID
   - Provide logits where that same token ID has a high score
   - Set repetition_penalty > 1.0
   - Assert the probability of that token ID in the returned probs is significantly reduced
     compared to a run with repetition_penalty=1.0

4. Test Case 4: Token Bias
   - Provide logits
   - Set a positive token_bias for a specific token ID
   - Assert the probability of that token ID increases in the returned probs compared to a run without bias
   - Test negative bias decreasing probability

5. Test Case 5: Top-K Filtering
   - Provide logits with scores spread across many tokens
   - Set top_k=5
   - Assert that only 5 tokens have non-zero probability in the returned probs tensor
   - Assert that these correspond to the original top 5 logits

6. Test Case 6: Top-P Filtering
   - Provide logits where probabilities sum > 1.0 (e.g., after softmax)
   - Set top_p=0.9
   - Calculate the expected cumulative probability cutoff
   - Assert that only tokens within that cumulative probability mass have non-zero probability in the returned probs

7. Test Case 7: Combined Sampling
   - Apply multiple modifiers (e.g., temp, top-k, repetition) simultaneously
   - Check that the likely candidates shift appropriately (e.g., repeated tokens become less likely, top-k limits choices)

8. Test Case 8: Output Structure
   - Verify the function returns a tuple with the correct types: (int, torch.Tensor, List[Dict])
   - Check the returned top_token_info_for_ui list has the expected format and length (~20)
"""

# Define constants for tests
VOCAB_SIZE = 100  # Small test vocabulary
BATCH_SIZE = 1    # Batch size for tests


# Helper functions
def create_test_logits(high_value_indices=None, values=None):
    """
    Create a test logits tensor with specified values at given indices.
    
    Args:
        high_value_indices: List of indices to set to specific high values
        values: List of values to set at those indices
        
    Returns:
        torch.Tensor of shape [1, 1, VOCAB_SIZE] with preset values
    """
    # Create base logits with small random values
    logits = torch.randn(BATCH_SIZE, 1, VOCAB_SIZE) * 0.1
    
    # If specific indices and values are provided, set them
    if high_value_indices is not None and values is not None:
        assert len(high_value_indices) == len(values), "Indices and values must have same length"
        for idx, val in zip(high_value_indices, values):
            logits[0, 0, idx] = val
            
    return logits


def create_test_input_ids(token_ids=None):
    """Create a test input_ids tensor with the specified token IDs"""
    if token_ids is None:
        token_ids = [1, 2, 3, 4, 5]  # Default sequence
    return torch.tensor([token_ids], dtype=torch.long)


def create_default_sampler_state():
    """Create a default SamplerState for testing"""
    return SamplerState(
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        token_bias={}
    )


def get_token_probability(probs, token_id):
    """Helper to safely extract the probability for a specific token"""
    if 0 <= token_id < probs.shape[-1]:
        return probs[0][token_id].item()
    return 0.0


def count_nonzero_probs(probs):
    """Count how many tokens have non-zero probability"""
    return torch.count_nonzero(probs).item()


class TestSampler:
    def test_greedy_decoding(self):
        """Test that with temperature near zero, the highest logit token is always selected."""
        # Create logits with one clearly higher value
        highest_token_id = 42
        logits = create_test_logits([highest_token_id], [10.0])  # Very high logit at index 42
        input_ids = create_test_input_ids()
        
        # Create sampler state with effectively zero temperature
        sampler_state = create_default_sampler_state()
        sampler_state.temperature = 1e-5  # Nearly zero
        
        # Call the function
        selected_token_id, probs, top_tokens_info = select_next_token(
            logits, input_ids, sampler_state
        )
        
        # Assert the highest logit token was selected
        assert selected_token_id == highest_token_id, "Greedy decoding should select the token with highest logit"
        
        # Check that highest probability in the distribution matches the highest logit
        highest_prob_token = torch.argmax(probs, dim=-1).item()
        assert highest_prob_token == highest_token_id, "Highest probability should correspond to highest logit"
    
    def test_temperature_scaling(self):
        """Test that temperature properly scales the probability distribution."""
        # Create logits with a few distinct values
        token_indices = [10, 20, 30]
        token_values = [5.0, 3.0, 1.0]  # Decreasing values
        logits = create_test_logits(token_indices, token_values)
        input_ids = create_test_input_ids()
        
        # 1. Test with temperature = 1.0 (baseline)
        sampler_state = create_default_sampler_state()
        sampler_state.temperature = 1.0
        
        _, probs_temp_1, _ = select_next_token(logits, input_ids, sampler_state)
        
        # 2. Test with temperature = 0.5 (sharper distribution)
        sampler_state.temperature = 0.5
        _, probs_temp_0_5, _ = select_next_token(logits, input_ids, sampler_state)
        
        # Extract probabilities for our test tokens
        probs_1 = [probs_temp_1[0][idx].item() for idx in token_indices]
        probs_0_5 = [probs_temp_0_5[0][idx].item() for idx in token_indices]
        
        # Check that the distribution is sharper with lower temperature
        # Compare the ratio between highest and lowest probability
        ratio_temp_1 = probs_1[0] / probs_1[2]  # Highest / Lowest
        ratio_temp_0_5 = probs_0_5[0] / probs_0_5[2]  # Highest / Lowest
        
        assert ratio_temp_0_5 > ratio_temp_1, "Lower temperature should increase probability contrast"
        
        # Verify the order of probabilities remains the same
        assert probs_1[0] > probs_1[1] > probs_1[2], "Probability order should match logit order"
        assert probs_0_5[0] > probs_0_5[1] > probs_0_5[2], "Probability order should match logit order"
    
    def test_repetition_penalty(self):
        """Test that repetition penalty reduces probabilities of tokens already in input_ids."""
        # Token that appears in input_ids and will be penalized
        repeated_token_id = 5
        
        # Create input_ids containing the token to be penalized
        input_ids = create_test_input_ids([1, 2, 3, 4, repeated_token_id])
        
        # Create logits where the repeated token has a high score
        token_indices = [repeated_token_id, 10, 20]
        token_values = [7.0, 5.0, 3.0]  # Repeated token has highest logit
        logits = create_test_logits(token_indices, token_values)
        
        # 1. Test without repetition penalty (baseline)
        sampler_state = create_default_sampler_state()
        sampler_state.repetition_penalty = 1.0  # No penalty
        
        _, probs_no_penalty, _ = select_next_token(logits, input_ids, sampler_state)
        prob_repeated_no_penalty = get_token_probability(probs_no_penalty, repeated_token_id)
        
        # 2. Test with repetition penalty
        sampler_state.repetition_penalty = 1.5  # Apply penalty
        
        _, probs_with_penalty, _ = select_next_token(logits, input_ids, sampler_state)
        prob_repeated_with_penalty = get_token_probability(probs_with_penalty, repeated_token_id)
        
        # Assert the probability of the repeated token is reduced with penalty
        assert prob_repeated_with_penalty < prob_repeated_no_penalty, \
            "Repetition penalty should reduce probability of repeated token"
        
        # Check that non-repeated tokens maintain their relative ordering
        prob_10_no_penalty = get_token_probability(probs_no_penalty, 10)
        prob_20_no_penalty = get_token_probability(probs_no_penalty, 20)
        prob_10_with_penalty = get_token_probability(probs_with_penalty, 10)
        prob_20_with_penalty = get_token_probability(probs_with_penalty, 20)
        
        assert prob_10_no_penalty > prob_20_no_penalty, "Probability ordering without penalty"
        assert prob_10_with_penalty > prob_20_with_penalty, "Probability ordering with penalty preserved"
    
    def test_token_bias(self):
        """Test that token bias appropriately adjusts token probabilities."""
        # Create a standard set of logits
        tokens = list(range(5))  # Tokens 0-4
        values = [1.0, 1.0, 1.0, 1.0, 1.0]  # Equal logits
        logits = create_test_logits(tokens, values)
        input_ids = create_test_input_ids()
        
        # 1. Get baseline probabilities without bias
        sampler_state = create_default_sampler_state()
        _, probs_no_bias, _ = select_next_token(logits, input_ids, sampler_state)
        
        # Grab the baseline probability for token 2
        prob_token_2_baseline = get_token_probability(probs_no_bias, 2)
        
        # 2. Test with positive bias on token 2
        sampler_state.token_bias = {2: 5.0}  # Strong positive bias
        _, probs_pos_bias, _ = select_next_token(logits, input_ids, sampler_state)
        prob_token_2_pos_bias = get_token_probability(probs_pos_bias, 2)
        
        # 3. Test with negative bias on token 2
        sampler_state.token_bias = {2: -5.0}  # Strong negative bias
        _, probs_neg_bias, _ = select_next_token(logits, input_ids, sampler_state)
        prob_token_2_neg_bias = get_token_probability(probs_neg_bias, 2)
        
        # Assert that positive bias increases probability
        assert prob_token_2_pos_bias > prob_token_2_baseline, \
            "Positive token bias should increase probability"
        
        # Assert that negative bias decreases probability
        assert prob_token_2_neg_bias < prob_token_2_baseline, \
            "Negative token bias should decrease probability"
        
        # Test with out-of-bounds token ID (should handle gracefully)
        sampler_state.token_bias = {VOCAB_SIZE + 10: 5.0}  # Out of bounds
        try:
            _, _, _ = select_next_token(logits, input_ids, sampler_state)
            # If we get here, it didn't crash with out-of-bounds token
            assert True, "Function should gracefully handle out-of-bounds token IDs"
        except IndexError:
            assert False, "Function should handle out-of-bounds token IDs gracefully"
    
    def test_top_k_filtering(self):
        """Test that top-k filtering limits the number of tokens that can be selected."""
        # Create logits with values spread across many tokens
        token_range = range(10)
        token_values = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        logits = create_test_logits(token_range, token_values)
        input_ids = create_test_input_ids()
        
        # Apply top-k = 5 filtering
        sampler_state = create_default_sampler_state()
        sampler_state.top_k = 5
        
        _, probs, _ = select_next_token(logits, input_ids, sampler_state)
        
        # Count non-zero probabilities
        non_zero_count = count_nonzero_probs(probs)
        
        # Assert that only top-k (5) tokens have non-zero probability
        assert non_zero_count == 5, f"Expected 5 non-zero probabilities with top-k=5, got {non_zero_count}"
        
        # Check that the top-k tokens are the ones with highest logits
        non_zero_token_indices = torch.nonzero(probs[0]).flatten().tolist()
        expected_top_k_indices = list(range(5))  # Tokens 0-4 have highest logits
        
        assert sorted(non_zero_token_indices) == sorted(expected_top_k_indices), \
            "Non-zero probabilities should correspond to tokens with highest logits"
    
    def test_top_p_filtering(self):
        """Test that top-p (nucleus) filtering correctly limits the probability mass."""
        # Create logits with a specific probability distribution after softmax
        token_indices = list(range(10))
        # Values that will create an approximate distribution after softmax
        token_logits = [10.0, 8.0, 5.0, 3.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
        logits = create_test_logits(token_indices, token_logits)
        input_ids = create_test_input_ids()
        
        # Calculate raw probabilities without filtering
        raw_probs = F.softmax(logits[0, 0], dim=-1)
        
        # Apply top-p filtering with p=0.9
        sampler_state = create_default_sampler_state()
        sampler_state.top_p = 0.9
        
        _, filtered_probs, _ = select_next_token(logits, input_ids, sampler_state)
        
        # Count non-zero probabilities after filtering
        non_zero_count = count_nonzero_probs(filtered_probs)
        
        # Calculate how many tokens should be included to reach p=0.9
        sorted_probs, sorted_indices = torch.sort(raw_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        num_tokens_in_p90 = torch.sum(cumulative_probs <= 0.9).item() + 1  # +1 to include the token that crosses threshold
        
        # Assert that the number of non-zero probabilities is as expected
        # We need some tolerance as the implementation might handle the cutoff slightly differently
        assert abs(non_zero_count - num_tokens_in_p90) <= 1, \
            f"Expected ~{num_tokens_in_p90} non-zero probabilities with top-p=0.9, got {non_zero_count}"
        
        # Verify that tokens with highest probabilities are kept
        non_zero_token_indices = torch.nonzero(filtered_probs[0]).flatten().tolist()
        expected_indices = sorted_indices[:num_tokens_in_p90].tolist()
        
        # Check that most of the expected high-probability tokens are there
        # (implementation might vary slightly in edge cases)
        common_indices = set(non_zero_token_indices).intersection(set(expected_indices))
        assert len(common_indices) >= num_tokens_in_p90 - 1, \
            f"Top-p filtering should keep the highest probability tokens"
    
    def test_combined_sampling(self):
        """Test that multiple sampling modifiers can be applied together."""
        # Repeated token in input_ids
        repeated_token_id = 3
        
        # Create input_ids with the repeated token
        input_ids = create_test_input_ids([1, 2, repeated_token_id, 4, 5])
        
        # Create logits with specific values
        token_indices = list(range(10))
        # repeated_token_id has high logit, should get penalized
        token_logits = [5.0, 6.0, 7.0, 9.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1]
        logits = create_test_logits(token_indices, token_logits)
        
        # Apply multiple sampling modifiers
        sampler_state = create_default_sampler_state()
        sampler_state.temperature = 0.7       # Sharpen distribution
        sampler_state.repetition_penalty = 1.3  # Penalize repeated tokens
        sampler_state.top_k = 5               # Limit to top 5 tokens
        # Add a positive bias to token 0
        sampler_state.token_bias = {0: 3.0}
        
        # Get probabilities without modifiers (baseline)
        baseline_state = create_default_sampler_state()
        _, baseline_probs, _ = select_next_token(logits, input_ids, baseline_state)
        
        # Get probabilities with all modifiers
        _, combined_probs, _ = select_next_token(logits, input_ids, sampler_state)
        
        # Check effects of combined modifiers:
        
        # 1. Repetition penalty: repeated token should have lower probability
        baseline_repeated_prob = get_token_probability(baseline_probs, repeated_token_id)
        combined_repeated_prob = get_token_probability(combined_probs, repeated_token_id)
        assert combined_repeated_prob < baseline_repeated_prob, \
            "Repetition penalty should reduce probability of repeated token"
        
        # 2. Token bias: biased token should have higher probability
        biased_token_id = 0
        baseline_biased_prob = get_token_probability(baseline_probs, biased_token_id)
        combined_biased_prob = get_token_probability(combined_probs, biased_token_id)
        assert combined_biased_prob > baseline_biased_prob, \
            "Positive token bias should increase probability"
        
        # 3. Top-k filtering: only top k tokens should have non-zero probability
        non_zero_count = count_nonzero_probs(combined_probs)
        assert non_zero_count <= 5, f"Top-k=5 should limit non-zero probabilities to at most 5"
    
    def test_output_structure(self):
        """Test that the function returns the expected output structure."""
        # Create simple inputs
        logits = create_test_logits()
        input_ids = create_test_input_ids()
        sampler_state = create_default_sampler_state()
        
        # Call function
        result = select_next_token(logits, input_ids, sampler_state)
        
        # Check result is a tuple of length 3
        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 3, "Result tuple should have 3 elements"
        
        # Unpack the tuple
        selected_token_id, probs, top_token_info = result
        
        # Check types of each component
        assert isinstance(selected_token_id, int), "First element should be an integer token ID"
        assert isinstance(probs, torch.Tensor), "Second element should be a probability tensor"
        assert isinstance(top_token_info, list), "Third element should be a list of token info"
        
        # Check probs shape
        assert probs.shape == (1, VOCAB_SIZE), f"Probs tensor should have shape [1, {VOCAB_SIZE}]"
        
        # Check token info structure
        assert len(top_token_info) <= 20, "Should return up to 20 top tokens"
        
        if len(top_token_info) > 0:
            assert isinstance(top_token_info[0], dict), "Token info should be a dictionary"
            assert 'token_id' in top_token_info[0], "Token info should contain 'token_id'"
            assert 'probability' in top_token_info[0], "Token info should contain 'probability'"
            
            # Check token ID is an integer and probability is a float
            assert isinstance(top_token_info[0]['token_id'], int), "token_id should be an integer"
            assert isinstance(top_token_info[0]['probability'], float), "probability should be a float"


# Tests for the helper functions
class TestSamplerHelpers:
    """Tests for the helper functions in sampler.py."""

    def test_top_k_filter(self):
        """Test that top_k_filter keeps only the top k logits."""
        # Create a test logits tensor with known values
        values = [10.0, 8.0, 6.0, 4.0, 2.0, 0.0, -2.0, -4.0]  # in descending order
        indices = list(range(len(values)))
        logits = create_test_logits(indices, values)[:, 0, :]  # Shape [1, VOCAB_SIZE]
        
        # Set k to 3 to keep only the top 3 logits
        k = 3
        filtered_logits = top_k_filter(logits, k)
        
        # Check that only top k logits are kept (not -inf)
        for i in range(VOCAB_SIZE):
            if i < k:  # Top k indices should have original values
                assert filtered_logits[0, i] == values[i], f"Top {k} logits should be preserved"
            else:  # All other indices should be set to -inf
                assert filtered_logits[0, i] == float('-inf') or filtered_logits[0, i] < -1e5, \
                    f"Non-top-{k} logits should be set to negative infinity"
        
        # Verify that relative order is preserved
        non_inf_indices = (filtered_logits[0] > -1e5).nonzero().view(-1)
        assert len(non_inf_indices) == k, f"Exactly {k} logits should be preserved"
        sorted_indices = torch.argsort(filtered_logits[0], descending=True)[:k]
        assert torch.equal(sorted_indices, torch.tensor(range(k))), "Relative order of top logits should be preserved"

    def test_top_p_filter(self):
        """Test that top_p_filter keeps only logits within cumulative probability p."""
        # Create a test logits tensor that will result in a spread of probabilities
        values = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]  # in descending order
        indices = list(range(len(values)))
        logits = create_test_logits(indices, values)[:, 0, :]  # Shape [1, VOCAB_SIZE]
        
        # Calculate probabilities via softmax
        probs = F.softmax(logits, dim=-1)
        
        # Set p to capture approximately the first 3 tokens (cumulative prob ~0.86)
        p = 0.85
        filtered_logits = top_p_filter(logits, p)
        
        # Calculate expected cumulative probabilities
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find how many tokens should be kept based on cum_probs <= p
        expected_kept = (cum_probs[0] <= p).sum().item() + 1  # Add 1 to include the token that exceeds p
        
        # Count how many tokens are actually kept in filtered_logits
        non_inf_count = (filtered_logits[0] > -1e5).sum().item()
        
        assert non_inf_count == expected_kept, \
            f"Expected to keep {expected_kept} tokens, but kept {non_inf_count}"
            
        # Check that top tokens (sorted by original probability) are kept
        sorted_indices = torch.argsort(logits[0], descending=True)[:expected_kept]
        for idx in sorted_indices:
            assert filtered_logits[0, idx] > -1e5, f"Token {idx} should be kept"
            
        # Check that other tokens are filtered out
        filtered_out_indices = torch.argsort(logits[0], descending=True)[expected_kept:]
        for idx in filtered_out_indices:
            assert filtered_logits[0, idx] == float('-inf') or filtered_logits[0, idx] < -1e5, \
                f"Token {idx} should be filtered out"

    def test_apply_repetition_penalty(self):
        """Test that repetition penalty correctly adjusts logits of repeated tokens."""
        # Create a test logits tensor with high values for certain tokens
        values = [5.0, 4.0, 3.0, 2.0, 1.0]  # in descending order
        indices = list(range(len(values)))
        logits = create_test_logits(indices, values)[:, 0, :]  # Shape [1, VOCAB_SIZE]
        original_logits = logits.clone()
        
        # Create input_ids that contains some tokens from our high-value indices
        repeated_tokens = [1, 3]  # tokens with indices 1 and 3
        input_ids = create_test_input_ids(repeated_tokens)
        
        # Case 1: penalty > 1.0 (decrease probability of repeated tokens)
        penalty = 2.0
        penalized_logits = apply_repetition_penalty(logits.clone(), input_ids, penalty)
        
        # Check that repeated tokens have reduced logits
        for token_id in repeated_tokens:
            assert penalized_logits[0, token_id] < original_logits[0, token_id], \
                f"Logit for repeated token {token_id} should be reduced with penalty > 1"
            assert torch.isclose(penalized_logits[0, token_id], 
                                original_logits[0, token_id] / penalty), \
                f"Logit for token {token_id} should be divided by {penalty}"
        
        # Check that non-repeated tokens are unchanged
        non_repeated_tokens = [0, 2, 4]  # tokens with indices 0, 2, and 4
        for token_id in non_repeated_tokens:
            assert torch.isclose(penalized_logits[0, token_id], original_logits[0, token_id]), \
                f"Logit for non-repeated token {token_id} should remain unchanged"
                
        # Case 2: penalty < 1.0 (increase probability of repeated tokens)
        reward_penalty = 0.5
        rewarded_logits = apply_repetition_penalty(logits.clone(), input_ids, reward_penalty)
        
        # Check that repeated tokens have increased logits
        for token_id in repeated_tokens:
            assert rewarded_logits[0, token_id] > original_logits[0, token_id], \
                f"Logit for repeated token {token_id} should be increased with penalty < 1"
            assert torch.isclose(rewarded_logits[0, token_id], 
                                original_logits[0, token_id] / reward_penalty), \
                f"Logit for token {token_id} should be divided by {reward_penalty}"
        
        # Check that non-repeated tokens are unchanged
        for token_id in non_repeated_tokens:
            assert torch.isclose(rewarded_logits[0, token_id], original_logits[0, token_id]), \
                f"Logit for non-repeated token {token_id} should remain unchanged"

# Additional tests could be added for:
# - Testing with actual model outputs (integration test)
# - Testing with different batch sizes
# - Edge cases (all equal logits, very small/large logits, etc.)
# - Stability of sampling with very low temperature
# - Handling of invalid inputs (negative temperature, etc.)
