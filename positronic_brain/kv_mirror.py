import torch
import threading
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
from .metrics import timed_histogram, set_gauge, inc_counter


# --- Token Context Tracking ---
@dataclass
class ContextToken:
    token_id: int
    instance_id: int
    position: Optional[int] = None  # Current index in kv_mirror (0..N-1), None if pruned
    source: Literal['llm', 'user_inject', 'system_init'] = 'llm'
    state: Literal['active', 'pruned'] = 'active'
    timestamp: float = field(default_factory=time.time)
    removal_bias: float = 0.0  # Manual bias to influence pruning (positive = more likely to keep)
    brightness: float = 255.0  # Token brightness value (255 = max, 0 = min) for Halo Weave


class KVMirror:
    def __init__(self):
        # These replace the global token_registry, kv_mirror, and kv_state_lock
        self._registry: Dict[int, ContextToken] = {}  # Maps instance_id -> ContextToken
        self._pos: Dict[int, int] = {}                # Maps position -> instance_id
        self._lock = threading.RLock()                # Use RLock for thread safety

        # This replaces the global global_generation_step and its lock
        self._global_generation_step: int = 0
        self._generation_step_lock = threading.Lock()  # Separate lock for the counter

    def _get_next_instance_id(self) -> int:
        """Atomically get the next unique instance ID for a token.
        
        Returns:
            int: A unique instance ID
        """
        with self._generation_step_lock:
            self._global_generation_step += 1
            return self._global_generation_step

    @timed_histogram("kv_mirror_add_seconds")
    def add(self, token_id: int, position: int, *, source: str = 'llm', removal_bias: float = 0.0, brightness: float = 255.0) -> int:
        """Atomically add a new token to the KV mirror and registry.
        
        This is the primary function for adding new tokens to the context tracking system.
        It handles creating a unique instance ID, creating the token object, and updating
        both the token registry and KV mirror atomically.
        
        Args:
            token_id: The ID of the token to register
            position: The position in the KV cache (0 to N-1)
            source: The source of the token (llm, user_inject, system_init, etc.)
            removal_bias: Bias factor affecting pruning (positive = more likely to keep)
            brightness: Initial brightness value for the token (default: 255.0 = maximum brightness)
            
        Returns:
            instance_id: The unique ID assigned to this token instance
        """
        # Get a unique instance ID for this token
        instance_id = self._get_next_instance_id()
        
        # Create token instance
        token = ContextToken(
            token_id=token_id,
            instance_id=instance_id,
            position=position,
            source=source,
            state='active',
            timestamp=time.time(),
            removal_bias=removal_bias,
            brightness=brightness
        )
        
        # Atomically update both data structures
        with self._lock:
            # Add to registry
            self._registry[instance_id] = token
            
            # Add to kv_mirror if it has a position
            if position is not None:
                self._pos[position] = instance_id
                
            # Verify integrity
            if position is not None and position in self._pos:
                actual_id = self._pos[position]
                if actual_id != instance_id:
                    print(f"[KV Mirror WARNING] Position {position} has inconsistent instance_id: expected {instance_id}, got {actual_id}")
            
            # Update metrics
            set_gauge("kv_mirror_active_tokens", len(self._pos))
            set_gauge("kv_mirror_registry_tokens", len(self._registry))
            inc_counter("kv_mirror_tokens_added_total")
        
        return instance_id

    @timed_histogram("kv_mirror_prune_seconds")
    def prune(self, positions_to_prune: list) -> bool:
        """Atomically apply a pruning operation to the KV mirror and token registry.
        
        This function takes a list of positions to remove and updates both the KV mirror
        and token registry accordingly. It ensures the entire operation is atomic to
        maintain consistency between the two data structures.
        
        CRITICAL: Unlike the previous implementation, this "soft mode" pruning does NOT 
        reindex the remaining tokens. Their original positions in the _pos dictionary are 
        preserved. This is essential for RoPE (Rotary Position Embedding) consistency, 
        as RoPE depends on absolute position values.
        
        Args:
            positions_to_prune: List of positions to remove from the KV cache
            
        Returns:
            bool: True if pruning was successful, False otherwise
        """
        try:
            with self._lock:
                original_size = len(self._pos)  # Size before pruning
                pruned_count = 0                 # Counter for actually pruned positions
                skipped_count = 0                # Counter for positions not found
                
                # Process each position to prune
                for pos in positions_to_prune:
                    # Check if position exists in the mirror
                    if pos not in self._pos:
                        print(f"[KV Mirror WARNING] Position {pos} not found during pruning, skipping")
                        skipped_count += 1
                        continue
                    
                    # Get the instance ID for this position
                    instance_id = self._pos[pos]
                    
                    # Remove from position map
                    del self._pos[pos]
                    
                    # Update registry state
                    if instance_id in self._registry:
                        self._registry[instance_id].state = 'pruned'
                        self._registry[instance_id].position = None
                    
                    pruned_count += 1
                
                # Update metrics
                set_gauge("kv_mirror_active_tokens", len(self._pos))  # Post-prune active count
                inc_counter("kv_mirror_tokens_pruned_total", pruned_count)  # Increment by num pruned
                
                # Log summary
                if skipped_count > 0:
                    print(f"[KV Mirror INFO] Pruned {pruned_count} positions, skipped {skipped_count} non-existent positions")
                
                return True
        except Exception as e:
            print(f"[KV Mirror ERROR] Exception during pruning: {type(e).__name__}: {str(e)}")
            return False

    @timed_histogram("kv_mirror_snapshot_seconds")
    def snapshot(self) -> Dict:
        """Get an atomic snapshot of the current KV mirror state.
        
        Returns:
            Dict containing kv_mirror mapping and token information, including
            all tokens in the registry (both positioned and non-positioned).
        """
        with self._lock:
            # Create copies to avoid exposing internal state
            mirror_snapshot = self._pos.copy()
            registry_snapshot = {}
            
            # Include all tokens from the registry, not just those in the mirror
            for instance_id, token in self._registry.items():
                # Create a copy of the token to avoid external modification
                registry_snapshot[instance_id] = copy.copy(token)
            
            # Calculate sizes while still holding the lock to ensure consistency
            mirror_size = len(mirror_snapshot)
            registry_size = len(registry_snapshot)
        
        return {
            'kv_mirror': mirror_snapshot,
            'tokens': registry_snapshot,
            'mirror_size': mirror_size,
            'registry_size': registry_size
        }
        
    def get_current_size(self) -> int:
        """Get the current size of the KV mirror.
        
        Returns:
            int: Number of active tokens in the mirror
        """
        with self._lock:
            return len(self._pos)
            
    @timed_histogram("kv_mirror_clear_seconds")
    def clear(self) -> None:
        """Clear all state in the KV Mirror.
        
        This resets the internal data structures and counters to their initial empty state.
        Useful when reinitializing the mirror for a new inference session or for testing.
        
        The method also resets the global generation step counter, ensuring that
        instance IDs start from 1 after clear() has been called.
        """
        with self._lock:
            # Store metrics before clearing
            tokens_cleared = len(self._pos)
            
            # Clear data structures
            self._registry.clear()
            self._pos.clear()
            
            # Update metrics
            set_gauge("kv_mirror_active_tokens", 0)
            set_gauge("kv_mirror_registry_tokens", 0)
            inc_counter("kv_mirror_tokens_cleared_total", tokens_cleared)
        
        # Reset the generation step counter to ensure instance IDs start fresh
        # This is particularly useful for testing and initialization
        with self._generation_step_lock:
            self._global_generation_step = 0
            
    @timed_histogram("kv_mirror_update_token_seconds")
    def update_token(self, position: int, new_token_id: int) -> bool:
        """Atomically update a token's ID at the specified position.
        
        This is used by the Compactor system to repair/update tokens without changing their position.
        
        Args:
            position: The position in the KV cache (0 to N-1)
            new_token_id: The new token ID to assign
            
        Returns:
            bool: True if update was successful, False if position not found or registry inconsistent
        """
        with self._lock:
            # Check if position exists in mirror
            if position not in self._pos:
                print(f"[KV Mirror WARNING] Cannot update token: Position {position} not found in mirror")
                return False
                
            # Get the instance_id for this position
            instance_id = self._pos[position]
            
            # Check if instance_id exists in registry
            if instance_id not in self._registry:
                print(f"[KV Mirror WARNING] Inconsistency detected: Instance ID {instance_id} at position {position} not in registry")
                return False
                
            # Update the token_id
            self._registry[instance_id].token_id = new_token_id
            
            # Increment counter for updated tokens
            inc_counter("kv_mirror_tokens_updated_total")
            
            return True
            
    @timed_histogram("kv_mirror_apply_diff_seconds")
    def apply_diff(self, diff_list: list) -> Dict[str, int]:
        """Apply a batch of token updates atomically.
        
        This method applies multiple token updates in a single atomic operation, which is more
        efficient than calling update_token repeatedly. Each update is a tuple containing
        (position, old_token_id_ignored, new_token_id).
        
        Args:
            diff_list: List of tuples (position, old_token_id_ignored, new_token_id)
            
        Returns:
            Dict with counts of successful and failed updates
        """
        results = {
            'success': 0,
            'failed_pos_not_found': 0,
            'failed_reg_not_found': 0
        }
        
        with self._lock:
            for position, _, new_token_id in diff_list:
                # Check if position exists in mirror
                if position not in self._pos:
                    results['failed_pos_not_found'] += 1
                    continue
                    
                # Get the instance_id for this position
                instance_id = self._pos[position]
                
                # Check if instance_id exists in registry
                if instance_id not in self._registry:
                    results['failed_reg_not_found'] += 1
                    continue
                    
                # Update the token_id
                self._registry[instance_id].token_id = new_token_id
                results['success'] += 1
            
            # Update metrics if any successful updates
            if results['success'] > 0:
                inc_counter("kv_mirror_tokens_updated_total", results['success'])
            
        return results
    
    @timed_histogram("kv_mirror_update_brightness_seconds")
    def batch_update_brightness(self, updates: Dict[int, float]) -> Dict[str, int]:
        """Atomically update brightness scores for multiple token instances.
        
        This method is used by the Brightness Engine to apply updated brightness values
        based on attention scores. All updates are applied atomically.
        
        Args:
            updates: Dictionary mapping instance_id to new brightness value
            
        Returns:
            Dict with counts of successful and failed updates
        """
        results = {
            'success': 0,
            'failed_instance_not_found': 0
        }
        
        with self._lock:
            for instance_id, new_brightness in updates.items():
                # Check if instance_id exists in registry
                if instance_id not in self._registry:
                    results['failed_instance_not_found'] += 1
                    continue
                    
                # Clamp the brightness value to valid range [0, 255]
                clamped_brightness = max(0.0, min(255.0, new_brightness))
                
                # Update the brightness
                self._registry[instance_id].brightness = clamped_brightness
                results['success'] += 1
            
            # Update metrics if any successful updates
            if results['success'] > 0:
                inc_counter("kv_mirror_brightness_updates_total", results['success'])
            
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the KV Mirror state.
        
        Returns:
            Dict with statistics including active tokens count and total tokens tracked
        """
        with self._lock:
            return {
                'active_tokens': len(self._pos),
                'total_tokens': len(self._registry)
            }
