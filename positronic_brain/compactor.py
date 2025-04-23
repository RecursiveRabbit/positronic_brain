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
    compactor_request_queue: asyncio.Queue,
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
            
            # --- 3. Prepare Requests for Embedding Data ---
            if selected_candidates:
                print(f"[Compactor] Requesting data for repair indices: {repair_indices}")
                print(f"[Compactor] Tokens: {[token_id for _, _, _, token_id in selected_candidates]}")
                print(f"[Compactor] Brightness values: {[brightness for _, brightness, _, _ in selected_candidates]}")
                
                # Create futures and requests for each candidate
                futures = []
                window_size = config.COMPACTOR_WINDOW_SIZE  # Window size for context around repair position
                
                # Create request objects for each position
                for idx, (position, _, instance_id, original_token_id) in enumerate(selected_candidates):
                    future = asyncio.Future()
                    futures.append(future)
                    
                    # Put request on queue: (reply_future, position, window_size, original_token_id)
                    request = (future, position, window_size, original_token_id)
                    try:
                        await compactor_request_queue.put(request)
                    except Exception as e:
                        print(f"[Compactor Error] Failed to queue request: {e}")
                        future.cancel()  # Cancel this future since we couldn't queue it
                        futures[idx] = None  # Mark as None to skip it later
                
                # Wait for responses with timeout
                valid_futures = [f for f in futures if f is not None]
                if not valid_futures:
                    print("[Compactor] No valid futures to process")
                    continue
                    
                try:
                    # Wait for all futures with timeout
                    results = await asyncio.gather(
                        *valid_futures, 
                        return_exceptions=True,
                        timeout=config.COMPACTOR_REQUEST_TIMEOUT
                    )
                    
                    # --- 4. Process Results and Call Diffuser ---
                    success_count = 0
                    diff_count = 0
                    
                    for result in results:
                        # Skip exceptions or timeouts
                        if isinstance(result, Exception):
                            print(f"[Compactor] Request failed: {result}")
                            continue
                            
                        try:
                            # Extract data from result
                            input_embeddings_segment = result["input_embeddings_segment"]
                            attention_mask_segment = result["attention_mask_segment"]
                            repair_index_in_segment = result["repair_index_in_segment"]
                            original_token_id = result["original_token_id"]
                            global_position_start = result["global_position_start"]
                            
                            # We need to convert the repair_index from global to segment-local
                            repair_indices_local = [repair_index_in_segment]
                            token_ids_local = [original_token_id]
                            
                            # Call diffuser to compute diffs
                            diff_list = compute_diff(
                                diffuser_model=diffuser_model,
                                input_embeddings=input_embeddings_segment,
                                attention_mask=attention_mask_segment,
                                token_ids=token_ids_local,
                                repair_indices=repair_indices_local
                            )
                            
                            # Process the diffs if any were found
                            if diff_list:
                                # Adjust indices to global positions
                                global_diffs = []
                                for local_pos, old_id, new_id in diff_list:
                                    # Convert from segment-local to global position
                                    global_pos = global_position_start + local_pos
                                    global_diffs.append((global_pos, old_id, new_id))
                                
                                # Put each diff on the queue
                                for diff in global_diffs:
                                    await pending_diffs_queue.put(diff)
                                    diff_count += 1
                                    
                                success_count += 1
                        except Exception as proc_e:
                            print(f"[Compactor Error] Failed to process result: {proc_e}")
                            print(traceback.format_exc())
                    
                    # Log repair statistics
                    print(f"[Compactor] Successfully computed {success_count}/{len(results)} repairs, generated {diff_count} diffs")
                    
                    # Update metrics
                    inc_counter("compactor_repair_cycles")
                    inc_counter("compactor_identified_candidates", len(candidates))
                    inc_counter("compactor_selected_tokens", len(selected_candidates))
                    inc_counter("compactor_successful_repairs", success_count)
                    inc_counter("compactor_generated_diffs", diff_count)
                    
                except asyncio.TimeoutError:
                    print(f"[Compactor Warning] Timed out waiting for embedding data after {config.COMPACTOR_REQUEST_TIMEOUT}s")
                    inc_counter("compactor_request_timeouts")
                except Exception as e:
                    print(f"[Compactor Error] Failed while gathering results: {e}")
                    print(traceback.format_exc())
                    inc_counter("compactor_gather_errors")
                    
                # Cancel any futures that didn't complete
                for future in valid_futures:
                    if not future.done():
                        future.cancel()
            
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
