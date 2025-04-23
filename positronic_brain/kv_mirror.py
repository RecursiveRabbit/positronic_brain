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
    def add(self, token_id: int, position: int, *, source: str = 'llm', removal_bias: float = 0.0) -> int:
        """Atomically add a new token to the KV mirror and registry.
        
        This is the primary function for adding new tokens to the context tracking system.
        It handles creating a unique instance ID, creating the token object, and updating
        both the token registry and KV mirror atomically.
        
        Args:
            token_id: The ID of the token to register
            position: The position in the KV cache (0 to N-1)
            source: The source of the token (llm, user_inject, system_init, etc.)
            removal_bias: Bias factor affecting pruning (positive = more likely to keep)
            
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
            removal_bias=removal_bias
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
    def prune(self, keep_indices: torch.Tensor) -> bool:
        """Atomically apply a pruning operation to the KV mirror and token registry.
        
        This function takes a tensor of indices to keep and updates both the KV mirror
        and token registry accordingly. It ensures the entire operation is atomic to
        maintain consistency between the two data structures.
        
        Args:
            keep_indices: Tensor of indices to keep in the KV cache
            
        Returns:
            bool: True if pruning was successful, False otherwise
        """
        try:
            # Convert keep_indices to Python list
            keep_list = keep_indices.cpu().tolist()
            
            with self._lock:
                # Build the new mirror mapping
                new_mirror = {}
                for new_pos, old_pos in enumerate(keep_list):
                    if old_pos in self._pos:
                        instance_id = self._pos[old_pos]
                        new_mirror[new_pos] = instance_id
                        
                        # Update position in token registry
                        if instance_id in self._registry:
                            self._registry[instance_id].position = new_pos
                    else:
                        print(f"[KV Mirror WARNING] Position {old_pos} not found in kv_mirror during pruning")
                        return False
                
                # Mark tokens not in new_mirror as pruned
                for old_pos in self._pos:
                    instance_id = self._pos[old_pos]
                    if not any(new_mirror[new_pos] == instance_id for new_pos in new_mirror):
                        if instance_id in self._registry:
                            self._registry[instance_id].position = None
                            self._registry[instance_id].state = 'pruned'
                
                # Replace the mirror with the new mapping
                self._pos.clear()
                self._pos.update(new_mirror)
                
                # Calculate stats for metrics
                num_kept = len(new_mirror)
                num_pruned = len(self._pos) - num_kept  # Size *before* clear()/update()
                
                # Update metrics (after state update)
                set_gauge("kv_mirror_active_tokens", len(new_mirror))  # Post-prune active count
                inc_counter("kv_mirror_tokens_pruned_total", num_pruned)  # Increment by num pruned
                
                # Verify integrity
                if len(keep_list) != len(self._pos):
                    print(f"[KV Mirror WARNING] Size mismatch after pruning: expected {len(keep_list)}, got {len(self._pos)}")
                    return False
                
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
        
        return {
            'kv_mirror': mirror_snapshot,
            'tokens': registry_snapshot,
            'mirror_size': len(mirror_snapshot),
            'registry_size': len(self._registry)
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
