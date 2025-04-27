"""
Unit tests for brightness engine and KVMirror brightness functionality.

These tests focus on the core brightness calculation and update logic in isolation,
ensuring that brightness initialization, decay, gain, and clamping work correctly.
"""

import pytest
import torch
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import modules under test
from positronic_brain.kv_mirror import KVMirror, ContextToken
from positronic_brain.brightness_engine import update_brightness_scores
from positronic_brain import config

# Mock metrics to avoid side effects
@pytest.fixture(autouse=True)
def mock_metrics():
    """Mock all metrics-related functions to avoid side effects."""
    with patch('positronic_brain.kv_mirror.timed_histogram'), \
         patch('positronic_brain.kv_mirror.set_gauge'), \
         patch('positronic_brain.kv_mirror.inc_counter'), \
         patch('positronic_brain.brightness_engine.timed_histogram'), \
         patch('positronic_brain.brightness_engine.inc_histogram'), \
         patch('positronic_brain.brightness_engine.inc_counter'), \
         patch('positronic_brain.brightness_engine._async_save_attention', return_value=None), \
         patch('positronic_brain.brightness_engine.print'), \
         patch('positronic_brain.kv_mirror.print'):
        yield

# Custom mock for model outputs
class MockOutput:
    """Mock outputs object that mimics the structure of outputs from a model's forward pass."""
    def __init__(self, attention_shape=(1, 32, 1, 10), attention_values=None):
        """
        Args:
            attention_shape: Shape of attention tensor (batch, heads, seq_len, seq_len)
            attention_values: Custom attention values, as a numpy array matching attention_shape
                              If None, will generate uniform values
        """
        if attention_values is None:
            # Default to uniform attention
            attention_values = np.ones(attention_shape) / attention_shape[-1]
            
        self.attentions = [torch.tensor(attention_values, dtype=torch.float32)]

@pytest.fixture
def kv_mirror_fixture():
    """Create a KVMirror with controlled initial state for testing."""
    # Create a fresh KVMirror
    mirror = KVMirror()
    
    # Clear any preexisting state
    mirror.clear()
    
    return mirror

def test_brightness_initialization_by_source(kv_mirror_fixture):
    """Test initialization of brightness based on token source."""
    mirror = kv_mirror_fixture
    
    # Store original config values to restore later
    original_brightness_seed = config.BRIGHTNESS_SEED
    
    # Set up test configuration
    config.BRIGHTNESS_SEED = {
        'user_inject': 200.0,
        'llm': 150.0,
        'system_init': 255.0,
        'default': 100.0
    }
    
    try:
        # Add tokens with different sources
        id1 = mirror.add(token_id=101, position=0, source='user_inject')
        id2 = mirror.add(token_id=102, position=1, source='llm')
        id3 = mirror.add(token_id=103, position=2, source='system_init')
        id4 = mirror.add(token_id=104, position=3, source='unknown')  # Should get default
        
        # Get token registry snapshot
        snapshot = mirror.snapshot()
        tokens = snapshot['tokens']
        
        # Assert brightness was initialized correctly based on source
        assert tokens[id1].brightness == 200.0, "User injected token brightness incorrect"
        assert tokens[id2].brightness == 150.0, "LLM token brightness incorrect"
        assert tokens[id3].brightness == 255.0, "System init token brightness incorrect"
        assert tokens[id4].brightness == 100.0, "Default token brightness incorrect"
        
    finally:
        # Restore original config
        config.BRIGHTNESS_SEED = original_brightness_seed

def test_explicit_brightness_initialization(kv_mirror_fixture):
    """Test explicit brightness initialization that overrides source-based defaults."""
    mirror = kv_mirror_fixture
    
    # Add tokens with explicit brightness values
    id1 = mirror.add(token_id=101, position=0, source='llm', brightness=42.0)
    
    # Get token registry snapshot
    snapshot = mirror.snapshot()
    tokens = snapshot['tokens']
    
    # Assert explicit brightness was used
    assert tokens[id1].brightness == 42.0, "Explicit brightness value not applied"

def test_brightness_decay(kv_mirror_fixture):
    """Test decay-only path in brightness update."""
    mirror = kv_mirror_fixture
    
    # Add tokens with known brightness
    id1 = mirror.add(token_id=101, position=0, brightness=100.0)
    id2 = mirror.add(token_id=102, position=1, brightness=50.0)
    
    # Create mock outputs with zero attention (pure decay)
    mock_outputs = MockOutput(
        attention_shape=(1, 32, 1, 2),
        attention_values=np.zeros((1, 32, 1, 2))
    )
    
    # Set test decay rate
    decay_rate = 5.0
    
    # Update brightness with zero attention
    result = update_brightness_scores(
        kv_mirror_manager=mirror,
        outputs=mock_outputs,
        generation_step=1,
        decay_per_tick=decay_rate,
        gain_coefficient=10.0  # Irrelevant for this test as attention is 0
    )
    
    # Get updated token registry
    snapshot = mirror.snapshot()
    tokens = snapshot['tokens']
    
    # Assert successful update
    assert result.get('success', 0) == 2, "Expected 2 successful brightness updates"
    
    # Assert brightness decayed correctly
    assert tokens[id1].brightness == 95.0, f"Expected token 1 brightness to decay from 100.0 to 95.0"
    assert tokens[id2].brightness == 45.0, f"Expected token 2 brightness to decay from 50.0 to 45.0"

def test_brightness_gain(kv_mirror_fixture):
    """Test gain-only path in brightness update."""
    mirror = kv_mirror_fixture
    
    # Add tokens with known brightness
    id1 = mirror.add(token_id=101, position=0, brightness=100.0)
    id2 = mirror.add(token_id=102, position=1, brightness=50.0)
    
    # Set attention values to test gain
    attention_values = np.zeros((1, 32, 1, 2))
    # Set attention from latest token to token 0 and 1
    attention_values[0, :, 0, 0] = 0.5  # 50% attention to position 0
    attention_values[0, :, 0, 1] = 0.2  # 20% attention to position 1
    
    mock_outputs = MockOutput(
        attention_shape=(1, 32, 1, 2),
        attention_values=attention_values
    )
    
    # Set test parameters
    decay_rate = 0.0  # No decay for this test
    gain_coefficient = 10.0
    
    # Expected attention gains
    expected_gain_id1 = int(0.5 * gain_coefficient)  # 5
    expected_gain_id2 = int(0.2 * gain_coefficient)  # 2
    
    # Update brightness with attention
    result = update_brightness_scores(
        kv_mirror_manager=mirror,
        outputs=mock_outputs,
        generation_step=1,
        decay_per_tick=decay_rate,
        gain_coefficient=gain_coefficient
    )
    
    # Get updated token registry
    snapshot = mirror.snapshot()
    tokens = snapshot['tokens']
    
    # Assert successful update
    assert result.get('success', 0) == 2, "Expected 2 successful brightness updates"
    
    # Assert brightness increased correctly
    assert tokens[id1].brightness == 100.0 + expected_gain_id1, \
           f"Expected token 1 brightness to increase from 100.0 to {100.0 + expected_gain_id1}"
    assert tokens[id2].brightness == 50.0 + expected_gain_id2, \
           f"Expected token 2 brightness to increase from 50.0 to {50.0 + expected_gain_id2}"

def test_brightness_combined(kv_mirror_fixture):
    """Test combined decay and gain in brightness update."""
    mirror = kv_mirror_fixture
    
    # Add tokens with known brightness
    id1 = mirror.add(token_id=101, position=0, brightness=100.0)
    id2 = mirror.add(token_id=102, position=1, brightness=50.0)
    
    # Set attention values to test gain
    attention_values = np.zeros((1, 32, 1, 2))
    # Set attention from latest token to token 0 and 1
    attention_values[0, :, 0, 0] = 0.5  # 50% attention to position 0
    attention_values[0, :, 0, 1] = 0.2  # 20% attention to position 1
    
    mock_outputs = MockOutput(
        attention_shape=(1, 32, 1, 2),
        attention_values=attention_values
    )
    
    # Set test parameters
    decay_rate = 3.0
    gain_coefficient = 10.0
    
    # Expected brightness changes
    expected_gain_id1 = int(0.5 * gain_coefficient)  # 5
    expected_gain_id2 = int(0.2 * gain_coefficient)  # 2
    
    # Update brightness with attention and decay
    result = update_brightness_scores(
        kv_mirror_manager=mirror,
        outputs=mock_outputs,
        generation_step=1,
        decay_per_tick=decay_rate,
        gain_coefficient=gain_coefficient
    )
    
    # Get updated token registry
    snapshot = mirror.snapshot()
    tokens = snapshot['tokens']
    
    # Assert successful update
    assert result.get('success', 0) == 2, "Expected 2 successful brightness updates"
    
    # Assert brightness updated correctly (decay + gain)
    assert tokens[id1].brightness == 100.0 - decay_rate + expected_gain_id1, \
           f"Expected token 1 brightness to change from 100.0 to {100.0 - decay_rate + expected_gain_id1}"
    assert tokens[id2].brightness == 50.0 - decay_rate + expected_gain_id2, \
           f"Expected token 2 brightness to change from 50.0 to {50.0 - decay_rate + expected_gain_id2}"

def test_brightness_clamp_lower(kv_mirror_fixture):
    """Test clamping to minimum brightness (0)."""
    mirror = kv_mirror_fixture
    
    # Add tokens with low brightness
    id1 = mirror.add(token_id=101, position=0, brightness=2.0)
    
    # Create mock outputs with zero attention (pure decay)
    mock_outputs = MockOutput(
        attention_shape=(1, 32, 1, 1),
        attention_values=np.zeros((1, 32, 1, 1))
    )
    
    # Set decay rate higher than current brightness to force negative value
    decay_rate = 5.0  # Should result in 2.0 - 5.0 = -3.0, which should be clamped to 0
    
    # Update brightness
    result = update_brightness_scores(
        kv_mirror_manager=mirror,
        outputs=mock_outputs,
        generation_step=1,
        decay_per_tick=decay_rate,
        gain_coefficient=10.0
    )
    
    # Get updated token registry
    snapshot = mirror.snapshot()
    tokens = snapshot['tokens']
    
    # Assert brightness was clamped to 0
    assert tokens[id1].brightness == 0.0, "Expected brightness to be clamped to 0 for negative value"

def test_brightness_clamp_upper(kv_mirror_fixture):
    """Test clamping to maximum brightness (255)."""
    mirror = kv_mirror_fixture
    
    # Add tokens with high brightness
    id1 = mirror.add(token_id=101, position=0, brightness=250.0)
    
    # Set very high attention to force brightness above 255
    attention_values = np.zeros((1, 32, 1, 1))
    attention_values[0, :, 0, 0] = 1.0  # 100% attention
    
    mock_outputs = MockOutput(
        attention_shape=(1, 32, 1, 1),
        attention_values=attention_values
    )
    
    # Set parameters that would push brightness above max
    decay_rate = 0.0  # No decay
    gain_coefficient = 20.0  # High gain: 1.0 * 20.0 = 20, resulting in 250 + 20 = 270
    
    # Update brightness
    result = update_brightness_scores(
        kv_mirror_manager=mirror,
        outputs=mock_outputs,
        generation_step=1,
        decay_per_tick=decay_rate,
        gain_coefficient=gain_coefficient
    )
    
    # Get updated token registry
    snapshot = mirror.snapshot()
    tokens = snapshot['tokens']
    
    # Assert brightness was clamped to 255
    assert tokens[id1].brightness == 255.0, "Expected brightness to be clamped to 255"

def test_batch_update_brightness_clamps(kv_mirror_fixture):
    """Directly test the batch_update_brightness clamping functionality."""
    mirror = kv_mirror_fixture
    
    # Add tokens
    id1 = mirror.add(token_id=101, position=0, brightness=100.0)
    id2 = mirror.add(token_id=102, position=1, brightness=100.0)
    id3 = mirror.add(token_id=103, position=2, brightness=100.0)
    
    # Create update dictionary with values that require clamping
    updates = {
        id1: -10.0,   # Should be clamped to 0
        id2: 100.0,   # No clamping needed
        id3: 300.0    # Should be clamped to 255
    }
    
    # Apply batch update
    result = mirror.batch_update_brightness(updates)
    
    # Get updated token registry
    snapshot = mirror.snapshot()
    tokens = snapshot['tokens']
    
    # Assert successful updates
    assert result['success'] == 3, "Expected 3 successful brightness updates"
    
    # Assert values were correctly clamped
    assert tokens[id1].brightness == 0.0, "Expected negative value to be clamped to 0"
    assert tokens[id2].brightness == 100.0, "Expected in-range value to remain unchanged"
    assert tokens[id3].brightness == 255.0, "Expected >255 value to be clamped to 255"

def test_brightness_decay_with_repair_threshold(kv_mirror_fixture):
    """Test that tokens decay correctly and can drop below the repair threshold."""
    mirror = kv_mirror_fixture
    
    # Store original config value to restore later
    original_repair_threshold = getattr(config, 'BRIGHTNESS_REPAIR_THRESHOLD', 50.0)
    
    # Set repair threshold for test
    config.BRIGHTNESS_REPAIR_THRESHOLD = 50.0
    
    try:
        # Add token with brightness above repair threshold
        id1 = mirror.add(token_id=101, position=0, brightness=60.0)
        
        # Create mock outputs with zero attention (pure decay)
        mock_outputs = MockOutput(
            attention_shape=(1, 32, 1, 1),
            attention_values=np.zeros((1, 32, 1, 1))
        )
        
        # Set decay rate to push below repair threshold
        decay_rate = 15.0  # Should result in 60.0 - 15.0 = 45.0, which is below threshold
        
        # Update brightness
        result = update_brightness_scores(
            kv_mirror_manager=mirror,
            outputs=mock_outputs,
            generation_step=1,
            decay_per_tick=decay_rate,
            gain_coefficient=10.0
        )
        
        # Get updated token registry
        snapshot = mirror.snapshot()
        tokens = snapshot['tokens']
        
        # Assert brightness decayed below repair threshold
        assert tokens[id1].brightness == 45.0, "Expected brightness to decay below repair threshold"
        assert tokens[id1].brightness < config.BRIGHTNESS_REPAIR_THRESHOLD, \
               "Brightness should be below repair threshold"
    finally:
        # Restore original config
        config.BRIGHTNESS_REPAIR_THRESHOLD = original_repair_threshold

def test_update_brightness_scores_with_missing_attentions():
    """Test behavior when attentions are missing from model outputs."""
    mirror = KVMirror()
    
    # Create mock outputs without attentions
    mock_outputs = MagicMock()
    mock_outputs.attentions = None
    
    # Update brightness - should return error
    result = update_brightness_scores(
        kv_mirror_manager=mirror,
        outputs=mock_outputs,
        generation_step=1
    )
    
    # Assert error was returned
    assert 'error' in result, "Expected error when attentions are missing"
    assert "No attention data available" in result['error'], \
           "Error message should indicate missing attentions"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
