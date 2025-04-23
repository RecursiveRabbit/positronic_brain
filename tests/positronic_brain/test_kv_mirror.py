import sys
import os
import pytest
import torch
import threading
import time

# Import our mock metrics module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'mocks')))

# Apply module-level patching before importing KVMirror
sys.modules['positronic_brain.metrics'] = __import__('mocks.metrics', fromlist=['*'])

# Now we can safely import KVMirror
from positronic_brain.kv_mirror import KVMirror, ContextToken


class TestKVMirror:
    def test_initialization(self, empty_mirror):
        """Test that a new KVMirror is properly initialized with empty state."""
        mirror = empty_mirror
        stats = mirror.get_stats()
        
        assert stats['active_tokens'] == 0
        assert stats['total_tokens'] == 0
        
    def test_add_token(self, empty_mirror):
        """Test adding a token and verifying its presence in the mirror."""
        mirror = empty_mirror
        token_id = 123
        position = 0
        
        instance_id = mirror.add(token_id, position)
        
        # Check the return value is a positive integer
        assert instance_id > 0
        
        # Verify stats reflect the addition
        stats = mirror.get_stats()
        assert stats['active_tokens'] == 1
        assert stats['total_tokens'] == 1
        
        # Check the size method
        assert mirror.get_current_size() == 1
        
        # Verify token in snapshot
        snapshot = mirror.snapshot()
        assert snapshot['mirror_size'] == 1
        assert position in snapshot['kv_mirror']
        assert snapshot['kv_mirror'][position] == instance_id
        assert instance_id in snapshot['tokens']
        assert snapshot['tokens'][instance_id].token_id == token_id
        assert snapshot['tokens'][instance_id].position == position
        assert snapshot['tokens'][instance_id].state == 'active'
        
    def test_add_multiple_tokens(self, empty_mirror):
        """Test adding multiple tokens sequentially."""
        mirror = empty_mirror
        num_tokens = 10
        token_ids = range(100, 100 + num_tokens)
        instance_ids = []
        
        # Add tokens sequentially
        for i, token_id in enumerate(token_ids):
            instance_id = mirror.add(token_id, i)
            instance_ids.append(instance_id)
            
        # Verify all tokens are in the mirror
        assert mirror.get_current_size() == num_tokens
        
        # Verify each token's position and state
        snapshot = mirror.snapshot()
        for i, instance_id in enumerate(instance_ids):
            assert snapshot['kv_mirror'][i] == instance_id
            assert snapshot['tokens'][instance_id].token_id == token_ids[i]
            assert snapshot['tokens'][instance_id].position == i
    
    def test_unique_instance_ids(self, empty_mirror):
        """Test that each token gets a unique instance ID."""
        mirror = empty_mirror
        instance_ids = set()
        
        # Add multiple tokens with the same token_id but different positions
        for i in range(10):
            instance_id = mirror.add(42, i)
            assert instance_id not in instance_ids
            instance_ids.add(instance_id)
            
        # Verify all tokens have the same token_id but different instance_ids
        snapshot = mirror.snapshot()
        for i in range(10):
            token = snapshot['tokens'][snapshot['kv_mirror'][i]]
            assert token.token_id == 42
            assert token.instance_id in instance_ids
    
    def test_prune_tokens(self, empty_mirror):
        """Test pruning tokens from the mirror."""
        mirror = empty_mirror
        
        # Add 10 tokens
        for i in range(10):
            mirror.add(i + 100, i)
            
        # Create a keep_indices tensor that keeps every other token
        keep_indices = torch.tensor([0, 2, 4, 6, 8], dtype=torch.long)
        
        # Apply pruning
        success = mirror.prune(keep_indices)
        assert success
        
        # Verify only the kept tokens remain active
        assert mirror.get_current_size() == 5
        
        # Verify positions have been reassigned
        snapshot = mirror.snapshot()
        for new_pos, old_pos in enumerate(keep_indices.tolist()):
            assert new_pos in snapshot['kv_mirror']
            
        # Verify registry size includes both active and pruned tokens
        assert snapshot['registry_size'] == 10
    
    def test_clear(self, empty_mirror):
        """Test clearing all state from the mirror."""
        mirror = empty_mirror
        
        # Add some tokens
        for i in range(5):
            mirror.add(i + 100, i)
            
        # Verify tokens were added
        assert mirror.get_current_size() == 5
        
        # Clear the mirror
        mirror.clear()
        
        # Verify everything is reset
        assert mirror.get_current_size() == 0
        stats = mirror.get_stats()
        assert stats['active_tokens'] == 0
        assert stats['total_tokens'] == 0
        
        # Add a new token and verify it gets a fresh instance ID
        instance_id = mirror.add(500, 0)
        assert instance_id == 1  # Should reset counter
    
    def test_thread_safety(self):
        """Test thread safety of the mirror operations."""
        mirror = KVMirror()
        mirror.clear()  # Ensure clean state
        num_threads = 5
        ops_per_thread = 100
        
        def worker(thread_id):
            for i in range(ops_per_thread):
                token_id = thread_id * 1000 + i
                mirror.add(token_id, -1)  # Use -1 as a placeholder
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        
        # Start all threads
        for t in threads:
            t.start()
            
        # Wait for completion
        for t in threads:
            t.join()
            
        # Verify total tokens registered
        stats = mirror.get_stats()
        assert stats['total_tokens'] == num_threads * ops_per_thread
        
        # Verify instance IDs are unique by checking registry size
        snapshot = mirror.snapshot()
        assert snapshot['registry_size'] == num_threads * ops_per_thread
    
    def test_prune_with_empty_mirror(self, empty_mirror):
        """Test pruning an empty mirror."""
        mirror = empty_mirror
        keep_indices = torch.tensor([], dtype=torch.long)
        
        success = mirror.prune(keep_indices)
        assert success
        assert mirror.get_current_size() == 0
    
    def test_prune_all_tokens(self, empty_mirror):
        """Test pruning all tokens."""
        mirror = empty_mirror
        
        # Add 5 tokens
        for i in range(5):
            mirror.add(i + 100, i)
            
        # Create an empty keep_indices tensor
        keep_indices = torch.tensor([], dtype=torch.long)
        
        # Apply pruning
        success = mirror.prune(keep_indices)
        assert success
        
        # Verify no active tokens remain
        assert mirror.get_current_size() == 0
        
        # Verify registry still contains the now-pruned tokens
        snapshot = mirror.snapshot()
        assert snapshot['registry_size'] == 5
        assert snapshot['mirror_size'] == 0
    
    def test_add_token_without_position(self, empty_mirror):
        """Test adding a token without a position (e.g., for future 'pruned' registration)."""
        mirror = empty_mirror
        token_id = 789
        
        # Add token with position=None
        instance_id = mirror.add(token_id, None)
        
        # Check the return value is a positive integer
        assert instance_id > 0
        
        # Verify stats reflect the addition to registry, but not active mirror
        stats = mirror.get_stats()
        assert stats['active_tokens'] == 0  # Should not be in active mirror
        assert stats['total_tokens'] == 1   # Should be in registry
        
        # Check the size method
        assert mirror.get_current_size() == 0
        
        # Verify token in snapshot registry, but NOT in mirror
        snapshot = mirror.snapshot()
        assert snapshot['mirror_size'] == 0  # Should be 0
        assert instance_id in snapshot['tokens']
        assert snapshot['tokens'][instance_id].token_id == token_id
        assert snapshot['tokens'][instance_id].position is None  # Position should be None
        assert snapshot['tokens'][instance_id].state == 'active'  # Still 'active' unless explicitly pruned
        
    def test_update_token_success(self, empty_mirror):
        """Test successfully updating a token's ID."""
        mirror = empty_mirror
        token_id = 100
        new_token_id = 200
        position = 0
        
        # Add token to update
        instance_id = mirror.add(token_id, position)
        
        # Update the token
        success = mirror.update_token(position, new_token_id)
        assert success is True
        
        # Verify token ID was updated in the registry
        snapshot = mirror.snapshot()
        assert snapshot['tokens'][instance_id].token_id == new_token_id
        
        # Verify other token properties remain unchanged
        token = snapshot['tokens'][instance_id]
        assert token.position == position
        assert token.instance_id == instance_id
        assert token.state == 'active'
        
    def test_update_token_invalid_pos(self, empty_mirror):
        """Test updating a token at an invalid position."""
        mirror = empty_mirror
        
        # Add a token at position 0
        mirror.add(123, 0)
        
        # Try to update a token at a non-existent position
        non_existent_position = 5
        success = mirror.update_token(non_existent_position, 999)
        
        # Verify update failed
        assert success is False
        
        # Verify no changes to existing tokens
        snapshot = mirror.snapshot()
        assert snapshot['mirror_size'] == 1
        assert 0 in snapshot['kv_mirror']
        
    def test_apply_diff_success(self, empty_mirror):
        """Test applying a batch of token updates successfully."""
        mirror = empty_mirror
        
        # Add several tokens
        num_tokens = 5
        positions = list(range(num_tokens))
        instance_ids = []
        
        for pos in positions:
            instance_id = mirror.add(pos + 100, pos)
            instance_ids.append(instance_id)
        
        # Create a diff list to update all tokens
        diff_list = [(pos, old_id, old_id + 500) for pos, old_id in enumerate([100, 101, 102, 103, 104])]
        
        # Apply the diff
        result = mirror.apply_diff(diff_list)
        
        # Verify all updates succeeded
        assert result['success'] == num_tokens
        assert result['failed_pos_not_found'] == 0
        assert result['failed_reg_not_found'] == 0
        
        # Verify token IDs were updated
        snapshot = mirror.snapshot()
        for i, instance_id in enumerate(instance_ids):
            assert snapshot['tokens'][instance_id].token_id == 100 + i + 500
            
    def test_apply_diff_partial_fail(self, empty_mirror):
        """Test applying a batch of token updates with some invalid positions."""
        mirror = empty_mirror
        
        # Add tokens at positions 0, 1, 2
        for pos in range(3):
            mirror.add(pos + 100, pos)
        
        # Create a diff list with some valid and some invalid positions
        # Format: (position, old_token_id_ignored, new_token_id)
        diff_list = [
            (0, 100, 600),  # Valid
            (2, 102, 602),  # Valid
            (5, 105, 605),  # Invalid position
            (1, 101, 601),  # Valid
            (7, 107, 607)   # Invalid position
        ]
        
        # Apply the diff
        result = mirror.apply_diff(diff_list)
        
        # Verify correct counts of successes and failures
        assert result['success'] == 3
        assert result['failed_pos_not_found'] == 2
        assert result['failed_reg_not_found'] == 0
        
        # Verify valid updates were applied
        snapshot = mirror.snapshot()
        positions_to_check = {0: 600, 1: 601, 2: 602}
        
        for pos, expected_id in positions_to_check.items():
            instance_id = snapshot['kv_mirror'][pos]
            assert snapshot['tokens'][instance_id].token_id == expected_id
            
    def test_batch_update_brightness_success(self, empty_mirror):
        """Test batch updating brightness values for tokens."""
        mirror = empty_mirror
        
        # Add several tokens
        num_tokens = 5
        instance_ids = []
        
        for pos in range(num_tokens):
            instance_id = mirror.add(pos + 100, pos)
            instance_ids.append(instance_id)
        
        # Create brightness updates
        brightness_updates = {
            instance_ids[0]: 200.0,  # Decrease from default 255
            instance_ids[2]: 50.0,   # Significant decrease
            instance_ids[4]: 230.0   # Small decrease
        }
        
        # Apply the brightness updates
        result = mirror.batch_update_brightness(brightness_updates)
        
        # Verify all updates succeeded
        assert result['success'] == 3
        assert result['failed_instance_not_found'] == 0
        
        # Verify brightness values were updated
        snapshot = mirror.snapshot()
        assert snapshot['tokens'][instance_ids[0]].brightness == 200.0
        assert snapshot['tokens'][instance_ids[1]].brightness == 255.0  # Unchanged
        assert snapshot['tokens'][instance_ids[2]].brightness == 50.0
        assert snapshot['tokens'][instance_ids[3]].brightness == 255.0  # Unchanged
        assert snapshot['tokens'][instance_ids[4]].brightness == 230.0
        
    def test_batch_update_brightness_clamping(self, empty_mirror):
        """Test that brightness values are properly clamped to [0, 255] range."""
        mirror = empty_mirror
        
        # Add a token
        instance_id = mirror.add(123, 0)
        
        # Test values outside the valid range
        brightness_updates = {
            instance_id: 300.0  # Above maximum
        }
        
        # Apply update
        mirror.batch_update_brightness(brightness_updates)
        
        # Verify value was clamped to maximum
        snapshot = mirror.snapshot()
        assert snapshot['tokens'][instance_id].brightness == 255.0
        
        # Test negative values
        brightness_updates = {
            instance_id: -50.0  # Below minimum
        }
        
        # Apply update
        mirror.batch_update_brightness(brightness_updates)
        
        # Verify value was clamped to minimum
        snapshot = mirror.snapshot()
        assert snapshot['tokens'][instance_id].brightness == 0.0
        
    def test_batch_update_brightness_invalid_id(self, empty_mirror):
        """Test batch updating brightness with invalid instance IDs."""
        mirror = empty_mirror
        
        # Add a token
        valid_id = mirror.add(123, 0)
        
        # Invalid instance IDs
        non_existent_id = 9999
        
        # Create updates with both valid and invalid IDs
        brightness_updates = {
            valid_id: 150.0,
            non_existent_id: 100.0
        }
        
        # Apply the updates
        result = mirror.batch_update_brightness(brightness_updates)
        
        # Verify correct counts
        assert result['success'] == 1
        assert result['failed_instance_not_found'] == 1
        
        # Verify valid update was applied
        snapshot = mirror.snapshot()
        assert snapshot['tokens'][valid_id].brightness == 150.0
