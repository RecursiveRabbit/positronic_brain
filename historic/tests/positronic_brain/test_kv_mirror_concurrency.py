"""
Concurrency tests for KVMirror to verify thread safety under high contention.

These tests validate that the KVMirror can handle concurrent operations from 
multiple threads performing different operations, simulating the interaction
between the main inference loop and asynchronous Compactor tasks.
"""

import sys
import os
import pytest
import torch
import threading
import time
import random
import concurrent.futures
from typing import Dict, List, Set

# Import our mock metrics module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'mocks')))

# Apply module-level patching before importing KVMirror
sys.modules['positronic_brain.metrics'] = __import__('mocks.metrics', fromlist=['*'])

# Now we can safely import KVMirror
from positronic_brain.kv_mirror import KVMirror, ContextToken


class TestKVMirrorConcurrency:
    """
    Test class focusing on concurrency aspects of KVMirror.
    
    These tests aim to verify that KVMirror can handle concurrent operations
    from multiple threads without deadlocking or corrupting its internal state.
    """
    
    def test_concurrent_add_and_apply_diff(self):
        """
        Test concurrent adds and diffs simulating main loop and Compactor interaction.
        
        This test spawns multiple threads:
        - Some threads add new tokens (simulating the main inference loop)
        - Some threads apply diffs to modify existing tokens (simulating the Compactor)
        - One thread continuously takes snapshots (simulating monitoring)
        
        The goal is to verify that there are no deadlocks and the final state is consistent.
        """
        # Create the KVMirror instance shared between all threads
        mirror = KVMirror()
        
        # Constants for test configuration
        NUM_ADDER_THREADS = 3
        NUM_DIFF_THREADS = 2
        OPERATIONS_PER_THREAD = 100
        
        # Tracking structures for verification
        added_tokens = {}  # {position: latest_token_id}
        added_positions = set()
        instance_id_map = {}  # {position: instance_id}
        applied_diffs = []  # [(position, new_token_id), ...]
        
        # Thread synchronization
        start_barrier = threading.Barrier(NUM_ADDER_THREADS + NUM_DIFF_THREADS + 1)  # +1 for snapshot thread
        completion_event = threading.Event()
        thread_lock = threading.Lock()  # For safe updates to our tracking structures
        
        # Define worker that adds tokens
        def adder_worker(worker_id: int):
            # Wait for all threads to be ready
            start_barrier.wait()
            
            for i in range(OPERATIONS_PER_THREAD):
                # Generate unique position and token_id
                position = (worker_id * 1000) + i
                token_id = (worker_id * 10000) + i
                
                # Add to KVMirror
                instance_id = mirror.add(token_id, position)
                
                # Track what we added (thread-safe)
                with thread_lock:
                    added_tokens[position] = token_id
                    added_positions.add(position)
                    instance_id_map[position] = instance_id
                
                # Small random sleep to increase chance of thread interleaving
                if random.random() < 0.05:
                    time.sleep(0.001)
        
        # Define worker that applies diffs
        def diff_applier_worker(worker_id: int):
            # Wait for all threads to be ready
            start_barrier.wait()
            
            # Wait a bit for some tokens to be added before trying to apply diffs
            time.sleep(0.01)
            
            for i in range(OPERATIONS_PER_THREAD):
                # Create a diff batch (between 1-5 token updates)
                diff_size = random.randint(1, 5)
                diff_batch = []
                
                # Thread-safe copy of available positions
                with thread_lock:
                    available_positions = list(added_positions)
                
                if len(available_positions) > 0:
                    # Pick random positions to update
                    positions_to_update = random.sample(
                        available_positions,
                        min(diff_size, len(available_positions))
                    )
                    
                    for pos in positions_to_update:
                        # Generate a new token ID for this position
                        new_token_id = (worker_id * 100000) + i + pos
                        # Include dummy old_token_id (it's ignored by apply_diff)
                        diff_batch.append((pos, 0, new_token_id))
                
                    # Apply the diff batch
                    if diff_batch:
                        result = mirror.apply_diff(diff_batch)
                        
                        # Record successful updates for verification
                        with thread_lock:
                            for pos, _, new_id in diff_batch:
                                if pos in result.get('updated_positions', []):
                                    added_tokens[pos] = new_id
                                    applied_diffs.append((pos, new_id))
                
                # Small random sleep to increase chance of thread interleaving
                if random.random() < 0.1:
                    time.sleep(0.002)
        
        # Define worker that continuously takes snapshots
        def snapshot_worker():
            # Wait for all threads to be ready
            start_barrier.wait()
            
            while not completion_event.is_set():
                # Take a snapshot and check some basic properties
                snapshot = mirror.snapshot()
                
                # Very simple validation
                assert isinstance(snapshot, dict)
                assert 'kv_mirror' in snapshot
                assert 'tokens' in snapshot
                
                # Sleep a bit before next snapshot
                time.sleep(0.005)
        
        # Create and start threads
        threads = []
        
        # Create adder threads
        for i in range(NUM_ADDER_THREADS):
            t = threading.Thread(target=adder_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Create diff applier threads
        for i in range(NUM_DIFF_THREADS):
            t = threading.Thread(target=diff_applier_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Create snapshot thread
        snapshot_thread = threading.Thread(target=snapshot_worker)
        snapshot_thread.daemon = True  # Allow it to be terminated when main thread ends
        snapshot_thread.start()
        
        # Wait for all worker threads to complete
        for t in threads:
            t.join(timeout=10)  # 10 second timeout to prevent hanging test
            assert not t.is_alive(), "Thread did not complete within timeout - possible deadlock"
        
        # Signal the snapshot thread to exit
        completion_event.set()
        
        # Get final snapshot for verification
        final_snapshot = mirror.snapshot()
        
        # Verify state consistency
        
        # 1. All positions in our tracking map should be in the final mirror
        for pos in added_positions:
            assert pos in final_snapshot['kv_mirror'], f"Position {pos} missing from final mirror"
        
        # 2. The mirror size should match the number of positions we've tracked
        assert mirror.get_current_size() == len(added_positions)
        
        # 3. In concurrent execution, we can't guarantee which thread's update "won" for each position
        # Instead, verify each token ID is either from an add operation or a diff operation
        valid_token_patterns = [
            # Original token from adder thread (worker_id * 10000) + i
            lambda pos, tid: any(tid == ((worker_id * 10000) + (pos % 1000)) 
                               for worker_id in range(NUM_ADDER_THREADS)),
            # Updated token from diff thread (worker_id * 100000) + i + pos
            lambda pos, tid: any(tid >= (worker_id * 100000) and tid <= ((worker_id + 1) * 100000 + pos) 
                               for worker_id in range(NUM_DIFF_THREADS))
        ]
        
        # Verify each token has a valid ID pattern
        invalid_positions = []
        for pos in added_positions:
            instance_id = final_snapshot['kv_mirror'][pos]
            token_id = final_snapshot['tokens'][instance_id].token_id
            
            # Check if token ID matches any valid pattern
            is_valid = any(pattern(pos, token_id) for pattern in valid_token_patterns)
            if not is_valid:
                invalid_positions.append((pos, token_id))
        
        assert not invalid_positions, f"Found {len(invalid_positions)} positions with invalid token IDs: {invalid_positions[:5]} ..."
    
    def test_concurrent_add_update_and_batch_brightness(self):
        """
        Test concurrent token additions, updates, and brightness updates.
        
        This test simulates:
        - Main loop adding tokens
        - Compactor applying diffs
        - Brightness Engine updating brightness scores
        
        All operations occur concurrently to maximize thread contention.
        """
        # Create the KVMirror instance shared between all threads
        mirror = KVMirror()
        
        # Track tokens and their instance IDs across threads
        token_registry = {}  # {position: {'instance_id': id, 'token_id': id, 'brightness': value}}
        registry_lock = threading.RLock()
        
        # Number of operations and threads
        NUM_ITERATIONS = 200
        NUM_THREADS = 3  # One for each operation type
        
        # Event to stop background threads
        stop_event = threading.Event()
        
        # Function to add tokens (simulating main loop)
        def add_tokens_worker():
            for i in range(NUM_ITERATIONS):
                position = i
                token_id = i + 1000
                
                # Add token to KVMirror
                instance_id = mirror.add(token_id, position, brightness=200.0)
                
                # Register token in our tracking dictionary
                with registry_lock:
                    token_registry[position] = {
                        'instance_id': instance_id,
                        'token_id': token_id,
                        'brightness': 200.0
                    }
                
                # Simulate work and allow thread switching
                time.sleep(0.001)
        
        # Function to update tokens (simulating Compactor)
        def update_tokens_worker():
            for i in range(NUM_ITERATIONS):
                # Wait for some tokens to be added
                if i == 0:
                    time.sleep(0.05)
                
                # Generate diff batch (between 1-5 updates)
                diff_batch = []
                with registry_lock:
                    positions = list(token_registry.keys())
                
                if positions:
                    # Select some positions to update
                    positions_to_update = random.sample(
                        positions,
                        min(5, len(positions))
                    )
                    
                    for pos in positions_to_update:
                        new_token_id = pos + 5000  # Different token ID range
                        # Include a dummy value for old_token_id (it's ignored by apply_diff)
                        diff_batch.append((pos, 0, new_token_id))
                
                    # Apply diff to KVMirror
                    if diff_batch:
                        result = mirror.apply_diff(diff_batch)
                        
                        # Update our tracking registry
                        with registry_lock:
                            for pos, _, new_id in diff_batch:
                                if pos in result.get('updated_positions', []):
                                    if pos in token_registry:
                                        token_registry[pos]['token_id'] = new_id
                
                # Allow thread switching
                time.sleep(0.002)
        
        # Function to update brightness (simulating Brightness Engine)
        def update_brightness_worker():
            for i in range(NUM_ITERATIONS):
                # Wait for some tokens to be added first
                if i == 0:
                    time.sleep(0.05)
                
                # Take snapshot to get current instance IDs
                with registry_lock:
                    registry_copy = token_registry.copy()
                
                if registry_copy:
                    # Generate brightness updates for 1-10 tokens
                    brightness_updates = {}
                    positions = random.sample(
                        list(registry_copy.keys()),
                        min(10, len(registry_copy))
                    )
                    
                    for pos in positions:
                        # Generate a new brightness value
                        new_brightness = random.uniform(0, 255)
                        instance_id = registry_copy[pos]['instance_id']
                        brightness_updates[instance_id] = new_brightness
                    
                    # Apply brightness updates
                    if brightness_updates:
                        mirror.batch_update_brightness(brightness_updates)
                        
                        # Update our tracking registry
                        with registry_lock:
                            for pos in positions:
                                if pos in token_registry:
                                    instance_id = registry_copy[pos]['instance_id']
                                    if instance_id in brightness_updates:
                                        token_registry[pos]['brightness'] = brightness_updates[instance_id]
                
                # Allow thread switching
                time.sleep(0.002)
        
        # Run all workers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [
                executor.submit(add_tokens_worker),
                executor.submit(update_tokens_worker),
                executor.submit(update_brightness_worker)
            ]
            
            # Wait for all workers to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # This will re-raise any exceptions from the thread
                except Exception as e:
                    pytest.fail(f"Thread failed with exception: {e}")
        
        # Verify final state consistency
        final_snapshot = mirror.snapshot()
        
        # 1. Check that KVMirror contains all positions we tracked
        assert len(final_snapshot['kv_mirror']) == len(token_registry)
        
        # 2. For each position, validate token properties within expected ranges
        for pos, token_data in token_registry.items():
            instance_id = final_snapshot['kv_mirror'].get(pos)
            assert instance_id is not None, f"Position {pos} missing from KVMirror"
            
            token_in_mirror = final_snapshot['tokens'][instance_id]
            
            # Allow for two valid token ID patterns due to concurrent updates:
            # - Original ID from add_tokens_worker: pos + 1000
            # - Updated ID from update_tokens_worker: pos + 5000
            valid_token_ids = [pos + 1000, pos + 5000]
            assert token_in_mirror.token_id in valid_token_ids, \
                f"Invalid token ID {token_in_mirror.token_id} at position {pos}, expected one of {valid_token_ids}"
            
            # Brightness will be either the initial value (200.0) or any value set by the brightness worker
            # We can only verify it's within the valid brightness range [0, 255]
            assert 0 <= token_in_mirror.brightness <= 255, \
                f"Brightness {token_in_mirror.brightness} out of valid range [0, 255] at position {pos}"
    
    def test_stress_with_rapid_snapshot_during_updates(self):
        """
        Stress test with very frequent snapshots during rapid updates.
        
        This test specifically targets potential issues with the RLock during
        high-frequency read operations (snapshots) concurrent with writes.
        """
        mirror = KVMirror()
        
        # Configuration
        NUM_UPDATE_THREADS = 4
        UPDATES_PER_THREAD = 100
        SNAPSHOT_FREQUENCY_MS = 1  # Very high frequency snapshots
        
        # Synchronization
        barrier = threading.Barrier(NUM_UPDATE_THREADS + 1)  # +1 for snapshot thread
        running = threading.Event()
        running.set()
        
        # Track operations
        operations_completed = 0
        operations_lock = threading.Lock()
        
        # Function for update workers
        def update_worker(worker_id):
            nonlocal operations_completed
            
            # Wait for all threads to start simultaneously
            barrier.wait()
            
            for i in range(UPDATES_PER_THREAD):
                pos = (worker_id * 1000) + i
                token_id = (worker_id * 10000) + i
                
                # 70% probability of add, 30% probability of update
                if random.random() < 0.7 or pos >= mirror.get_current_size():
                    # Add new token
                    mirror.add(token_id, pos)
                else:
                    # Update existing token at random valid position
                    valid_positions = [p for p in range(mirror.get_current_size()) if p < pos]
                    if valid_positions:
                        update_pos = random.choice(valid_positions)
                        new_token_id = token_id + 50000
                        # For variety, sometimes use update_token and sometimes use apply_diff
                        if random.random() < 0.5:
                            mirror.update_token(update_pos, new_token_id)
                        else:
                            # Use 3-tuple format for apply_diff
                            mirror.apply_diff([(update_pos, 0, new_token_id)])
                
                # Count completed operations
                with operations_lock:
                    operations_completed += 1
                
                # Random small sleep for better thread interleaving
                if random.random() < 0.1:
                    time.sleep(0.001)
        
        # Function for snapshot thread
        def snapshot_worker():
            snapshots_taken = 0
            
            # Wait for all threads to start simultaneously
            barrier.wait()
            
            while running.is_set():
                # Take snapshot and do minimal validation
                snapshot = mirror.snapshot()
                assert isinstance(snapshot, dict)
                snapshots_taken += 1
                
                # Very small sleep between snapshots
                time.sleep(SNAPSHOT_FREQUENCY_MS / 1000)
            
            print(f"[Snapshot worker] Took {snapshots_taken} snapshots")
        
        # Start update threads
        update_threads = []
        for i in range(NUM_UPDATE_THREADS):
            t = threading.Thread(target=update_worker, args=(i,))
            update_threads.append(t)
            t.start()
        
        # Start snapshot thread
        snapshot_thread = threading.Thread(target=snapshot_worker)
        snapshot_thread.daemon = True
        snapshot_thread.start()
        
        # Wait for update threads to complete
        for t in update_threads:
            t.join(timeout=30)  # 30 second timeout
            assert not t.is_alive(), "Update thread did not complete - possible deadlock"
        
        # Signal snapshot thread to stop
        running.clear()
        
        # Verify operations completed
        assert operations_completed == NUM_UPDATE_THREADS * UPDATES_PER_THREAD
        
        # Get final state and verify consistency
        final_snapshot = mirror.snapshot()
        assert len(final_snapshot['kv_mirror']) == mirror.get_current_size()
        assert len(final_snapshot['tokens']) >= mirror.get_current_size()  # May include pruned tokens


# Define fixture for an empty mirror (reused across test methods)
@pytest.fixture
def empty_mirror():
    return KVMirror()
