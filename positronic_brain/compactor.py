"""
Compactor module for the Positronic Brain / Halo Weave system.

This module is responsible for asynchronously monitoring token brightness
and scheduling token repairs using the diffuser model. It sits between
the KVMirror, Diffuser Runner, and the main inference loop.
"""

import asyncio
import time
import torch
from typing import List, Tuple, Dict, Optional, Any
import traceback

from . import config
from .kv_mirror import KVMirror
from .diffuser_runner import DiffuserModel, compute_diff
from .metrics import timed_histogram, inc_counter


@timed_histogram("compactor_task_iteration_seconds")
async def compactor_task(
    kv_mirror_manager: KVMirror,
    diffuser_model: DiffuserModel,
    pending_diffs_queue: asyncio.Queue,
    shutdown_event: asyncio.Event
):
    """
    Asynchronous task that monitors token brightness and schedules diffuser repairs.
    
    This function runs as a separate async task, periodically checking for tokens
    that have fallen below the brightness threshold and scheduling them for repair.
    
    Args:
        kv_mirror_manager: The KVMirror instance containing token state
        diffuser_model: The DiffuserModel instance for token repair
        pending_diffs_queue: Queue where repair diffs are placed for the main loop to apply
        shutdown_event: Event signaling when the compactor should shut down
    """
    print("[Compactor] Task started.")
    
    # Track statistics
    repair_cycles = 0
    total_candidates_identified = 0
    total_tokens_selected = 0
    
    while not shutdown_event.is_set():
        try:
            # Sleep at the beginning of each iteration to avoid tight loops
            await asyncio.sleep(config.COMPACTOR_SLEEP_INTERVAL)
            
            # --- 1. Identify Repair Candidates ---
            # Get current state snapshot (including brightness)
            snapshot = kv_mirror_manager.snapshot()
            active_tokens = snapshot['tokens']  # Dictionary {instance_id: ContextToken}
            kv_mirror = snapshot['kv_mirror']   # Dictionary {position: instance_id}
            
            if not kv_mirror:  # Skip if mirror is empty
                continue
            
            candidates = []
            for position, instance_id in kv_mirror.items():
                if instance_id in active_tokens:
                    token_info = active_tokens[instance_id]
                    if token_info.brightness < config.BRIGHTNESS_REPAIR_THRESHOLD:
                        # Store (position, brightness, instance_id, original_token_id)
                        candidates.append((position, token_info.brightness, instance_id, token_info.token_id))
            
            if not candidates:
                # Optional: Verbose logging
                # print("[Compactor] No candidates for repair.")
                continue
            
            # --- 2. Schedule/Select Which Candidates to Repair ---
            # Simple strategy: Repair the N dimmest tokens
            candidates.sort(key=lambda x: x[1])  # Sort by brightness (ascending)
            num_to_repair = min(len(candidates), config.MAX_REPAIR_TOKENS_PER_STEP)
            selected_candidates = candidates[:num_to_repair]
            repair_indices = [pos for pos, _, _, _ in selected_candidates]
            original_token_ids = [orig_id for _, _, _, orig_id in selected_candidates]
            
            # Update statistics
            repair_cycles += 1
            total_candidates_identified += len(candidates)
            total_tokens_selected += len(selected_candidates)
            
            avg_brightness = sum(b for _, b, _, _ in selected_candidates) / len(selected_candidates) if selected_candidates else 0
            print(f"[Compactor] Cycle {repair_cycles}: Identified {len(candidates)} candidates "
                  f"(avg brightness: {avg_brightness:.2f}). "
                  f"Selecting {len(selected_candidates)} for repair.")
            
            # --- 3. Prepare Input for Diffuser ---
            # PLACEHOLDER: This is where we'd get the embeddings for the diffuser
            # The challenge is how the Compactor gets access to the LLM's embeddings:
            # Option A: KVMirror also stores embeddings (Memory intensive!)
            # Option B: ai_core sends segments to Compactor queue (Complex data transfer)
            # Option C: Compactor reads directly from GPU memory (Requires shared memory/careful sync)
            # Option D: Simplify - For now, skip repair step and just log candidates
            
            # --- PLACEHOLDER for Diffuser Call ---
            # In a real implementation:
            # 1. Get relevant embedding segment & attention mask
            # 2. Call diffuser_runner.compute_diff(...)
            # 3. Enqueue results
            
            # For now, just log what we would attempt to repair
            if selected_candidates:
                print(f"[Compactor] Would repair indices: {repair_indices}")
                print(f"[Compactor] Tokens: {[token_id for _, _, _, token_id in selected_candidates]}")
                print(f"[Compactor] Brightness values: {[brightness for _, brightness, _, _ in selected_candidates]}")
                
                # Increment counter for monitoring
                inc_counter("compactor_repair_cycles")
                inc_counter("compactor_identified_candidates", len(candidates))
                inc_counter("compactor_selected_tokens", len(selected_candidates))
            
        except asyncio.CancelledError:
            print("[Compactor] Task cancelled.")
            break
        except Exception as e:
            print(f"[Compactor Error] Unexpected error: {type(e).__name__} - {e}")
            print(traceback.format_exc())
            inc_counter("compactor_errors")
            await asyncio.sleep(5)  # Wait longer after an error
    
    # Final statistics on shutdown
    print(f"[Compactor] Task finished. Total cycles: {repair_cycles}, "
          f"Total candidates: {total_candidates_identified}, "
          f"Total selected: {total_tokens_selected}")


# Future extension: Implement a CompactorManager class if more sophisticated
# scheduling and coordination is needed.
# 
# class CompactorManager:
#     def __init__(self, kv_mirror_manager, diffuser_model, ...):
#         ...
#     
#     async def start(self):
#         ...
#     
#     async def stop(self):
#         ...
