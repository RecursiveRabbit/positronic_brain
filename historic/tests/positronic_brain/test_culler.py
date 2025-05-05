"""
Tests for the Culler module of the Halo Weave v0 system.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch

from positronic_brain.culler import select_tokens_for_cull, culling_task
from positronic_brain.kv_mirror import KVMirror, ContextToken


class TestCullerRules:
    """Tests for the token culling rules."""
    
    def test_no_culling_when_below_target(self):
        """Test that no tokens are culled when below target size."""
        # Create mock KVMirror with snapshot returning fewer tokens than target
        mock_kv_mirror = MagicMock(spec=KVMirror)
        mock_kv_mirror.snapshot.return_value = {
            'kv_mirror': {0: 1, 1: 2, 2: 3},  # 3 positions
            'tokens': {
                1: MagicMock(spec=ContextToken, brightness=100.0),
                2: MagicMock(spec=ContextToken, brightness=150.0),
                3: MagicMock(spec=ContextToken, brightness=200.0),
            }
        }
        
        # Target size is 5, current size is 3
        result = select_tokens_for_cull(mock_kv_mirror, 5)
        
        # Should return empty list (no culling)
        assert result == []
    
    def test_cull_one_when_at_target(self):
        """Test that one token is culled when at target size."""
        # Create mock KVMirror with snapshot returning exactly target tokens
        mock_kv_mirror = MagicMock(spec=KVMirror)
        mock_kv_mirror.snapshot.return_value = {
            'kv_mirror': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},  # 5 positions
            'tokens': {
                1: MagicMock(spec=ContextToken, brightness=100.0),
                2: MagicMock(spec=ContextToken, brightness=150.0),
                3: MagicMock(spec=ContextToken, brightness=50.0),  # Dimmest
                4: MagicMock(spec=ContextToken, brightness=200.0),
                5: MagicMock(spec=ContextToken, brightness=175.0),
            }
        }
        
        # Target size is 5, current size is 5
        result = select_tokens_for_cull(mock_kv_mirror, 5)
        
        # Should return position 2 (dimmest token)
        assert result == [2]
    
    def test_cull_two_when_above_target(self):
        """Test that two tokens are culled when above target size."""
        # Create mock KVMirror with snapshot returning more than target tokens
        mock_kv_mirror = MagicMock(spec=KVMirror)
        mock_kv_mirror.snapshot.return_value = {
            'kv_mirror': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7},  # 7 positions
            'tokens': {
                1: MagicMock(spec=ContextToken, brightness=100.0),
                2: MagicMock(spec=ContextToken, brightness=30.0),   # 2nd dimmest
                3: MagicMock(spec=ContextToken, brightness=150.0),
                4: MagicMock(spec=ContextToken, brightness=20.0),   # Dimmest
                5: MagicMock(spec=ContextToken, brightness=200.0),
                6: MagicMock(spec=ContextToken, brightness=175.0),
                7: MagicMock(spec=ContextToken, brightness=190.0),
            }
        }
        
        # Target size is 5, current size is 7
        result = select_tokens_for_cull(mock_kv_mirror, 5)
        
        # Should return positions 3 and 1 (two dimmest tokens)
        assert sorted(result) == [1, 3]


@pytest.mark.asyncio
async def test_culling_task_once():
    """Test the culling task with once=True to verify it operates correctly."""
    # Create mock KVMirror
    mock_kv_mirror = MagicMock(spec=KVMirror)
    mock_kv_mirror.snapshot.return_value = {
        'kv_mirror': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},  # 6 positions
        'tokens': {
            1: MagicMock(spec=ContextToken, brightness=100.0),
            2: MagicMock(spec=ContextToken, brightness=50.0),  # Dimmest
            3: MagicMock(spec=ContextToken, brightness=150.0),
            4: MagicMock(spec=ContextToken, brightness=75.0),  # 2nd dimmest
            5: MagicMock(spec=ContextToken, brightness=200.0),
            6: MagicMock(spec=ContextToken, brightness=175.0),
        }
    }
    
    # After culling, the size should be smaller
    # Create token objects for the mock
    mock_tokens = {
        1: MagicMock(spec=ContextToken, brightness=100.0),
        2: MagicMock(spec=ContextToken, brightness=50.0),  # Dimmest
        3: MagicMock(spec=ContextToken, brightness=150.0),
        4: MagicMock(spec=ContextToken, brightness=75.0),  # 2nd dimmest
        5: MagicMock(spec=ContextToken, brightness=200.0),
        6: MagicMock(spec=ContextToken, brightness=175.0),
    }
    
    mock_kv_mirror.snapshot.side_effect = [
        # First call returns 6 tokens
        {
            'kv_mirror': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
            'tokens': mock_tokens
        },
        # Second call (after culling) returns 4 tokens
        {
            'kv_mirror': {0: 1, 2: 3, 4: 5, 5: 6},
            'tokens': {k: v for k, v in mock_tokens.items() if k != 2 and k != 4}
        }
    ]
    
    # Mock config
    with patch('positronic_brain.culler.config') as mock_config:
        mock_config.CONTEXT_WINDOW_TARGET = 4
        mock_config.COMPACTOR_SLEEP_INTERVAL = 0.01  # Fast for test
        
        # Run the culling task once
        shutdown_event = asyncio.Event()
        stats = await culling_task(mock_kv_mirror, shutdown_event, once=True)
        
        # Verify prune was called with the right positions
        mock_kv_mirror.prune.assert_called_once()
        
        # Verify expected stats
        assert stats["culling_cycles"] == 1
        assert stats["total_tokens_culled"] == 2  # Should have culled 2 tokens
