"""
Culler module for the Positronic Brain / Halo Weave system.

This module is responsible for deterministically removing tokens from the context
based on brightness values to maintain a target context window size.
It implements the "Halo Weave v0" approach with fixed-size context management.
"""

import asyncio
import time
from typing import List, Dict, Set, Tuple, Optional

from . import config
from .kv_mirror import KVMirror
from .metrics import timed_histogram, inc_counter, set_gauge


def select_tokens_for_cull(kv_mirror_manager: KVMirror, target_size: int) -> List[int]:
    """
    Deterministically select tokens for culling based on brightness and target size.
    
    The culling rules are:
    - If len < target: Cull 0 tokens
    - If len == target: Cull 1 token (lowest brightness)
    - If len > target: Cull 2 tokens per step until len == target
    
    Args:
        kv_mirror_manager: The KVMirror instance containing token state
        target_size: The target context window size to maintain
        
    Returns:
        List of positions to be culled from the KV mirror
    """
    # Get current state snapshot
    snapshot = kv_mirror_manager.snapshot()
    kv_mirror = snapshot['kv_mirror']   # Dictionary {position: instance_id}
    active_tokens = snapshot['tokens']  # Dictionary {instance_id: ContextToken}
    
    # Calculate current size
    current_size = len(kv_mirror)
    
    # Determine how many tokens to cull based on size rule
    if current_size <= target_size:
        # If we're at or below target, just cull one token (if we're at the target)
        num_to_cull = 1 if current_size == target_size else 0
    else:
        # If we're above target, cull 2 tokens per step (or however many needed to reach target)
        num_to_cull = min(2, current_size - target_size)
    
    # If no tokens to cull, return empty list
    if num_to_cull == 0:
        return []
    
    # Create brightness map {position: brightness}
    brightness_map = {}
    for position, instance_id in kv_mirror.items():
        if instance_id in active_tokens:
            token_info = active_tokens[instance_id]
            brightness_map[position] = token_info.brightness
    
    # Sort positions by brightness (ascending)
    sorted_positions = sorted(brightness_map.keys(), key=lambda pos: brightness_map[pos])
    
    # Select the dimmest tokens up to num_to_cull
    return sorted_positions[:num_to_cull]


@timed_histogram("culler_task_iteration_seconds")
async def culling_task(
    kv_mirror_manager: KVMirror,
    shutdown_event: asyncio.Event,
    once: bool = False,  # For testing: run once and exit
) -> Dict[str, int]:
    """
    Asynchronous task that culls tokens based on brightness to maintain target context size.
    
    This function runs as a separate async task, periodically checking if the context
    size exceeds the target and culling the dimmest tokens to maintain the target size.
    
    Args:
        kv_mirror_manager: The KVMirror instance containing token state
        shutdown_event: Event signaling when the culler should shut down
        once: If True, run only one iteration and return stats (for testing)
        
    Returns:
        Dictionary with statistics about culling operations (only if once=True)
    """
    print("[Culler] Task started.")
    
    # Track statistics
    stats = {
        "culling_cycles": 0,
        "total_tokens_culled": 0,
    }
    
    # If once=True, we'll return after one iteration
    run_once = once
    
    while not shutdown_event.is_set():
        try:
            # Sleep at the beginning of each iteration to avoid tight loops
            await asyncio.sleep(config.COMPACTOR_SLEEP_INTERVAL)
            
            # Use deterministic culling strategy
            positions_to_cull = select_tokens_for_cull(
                kv_mirror_manager,
                config.CONTEXT_WINDOW_TARGET
            )
            
            # If there are positions to cull, prune them from the KV mirror
            if positions_to_cull:
                # Prune the tokens in a single operation
                kv_mirror_manager.prune(positions_to_cull)
                
                # Log culling event
                stats["culling_cycles"] += 1
                stats["total_tokens_culled"] += len(positions_to_cull)
                
                print(f"[Culler] Cycle {stats['culling_cycles']}: "
                      f"Culled {len(positions_to_cull)} tokens to maintain target size of {config.CONTEXT_WINDOW_TARGET}")
                
                # Update metrics
                inc_counter("culler_cycles")
                inc_counter("culler_tokens_culled", len(positions_to_cull))
                
                # Get new size after culling
                current_size = len(kv_mirror_manager.snapshot()['kv_mirror'])
                set_gauge("context_window_current_size", current_size)
            
            # If we're only supposed to run once, break after the first iteration
            if run_once:
                break
                
        except asyncio.CancelledError:
            print("[Culler] Task cancelled.")
            break
        except Exception as e:
            print(f"[Culler Error] Unexpected error: {type(e).__name__} - {e}")
            inc_counter("culler_errors")
            await asyncio.sleep(5)  # Wait longer after an error
            
            # Don't count this as a completed iteration if we're only supposed to run once
            if not run_once:
                continue
            else:
                break
    
    # Final statistics on shutdown
    if not run_once:
        print(f"[Culler] Task finished. Total cycles: {stats['culling_cycles']}, "
              f"Total tokens culled: {stats['total_tokens_culled']}")
    
    # Return stats if we're only running once (used for testing)
    return stats if run_once else None
