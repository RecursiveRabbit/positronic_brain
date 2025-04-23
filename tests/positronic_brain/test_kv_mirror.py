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
