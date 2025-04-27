"""
Context Maintenance module for the Positronic Brain / Halo Weave system.

This module encapsulates the synchronous maintenance phase that runs after each token 
generation, including brightness updates, culling, and token repair via diffusion.
"""

from .kv_mirror import KVMirror
from .diffuser_runner import DiffuserModel, compute_diff
from .kv_patcher import KVCachePatcher
from .brightness_engine import update_brightness_scores
from .culler import select_tokens_for_cull
from . import config
import torch
from typing import Tuple, Optional, List, Dict, Any, Set
import asyncio
import time
import sys

class ContextMaintenance:
    """
    Manages synchronous context maintenance operations for the Halo Weave system.
    
    This class encapsulates the sequential steps of the maintenance phase:
    1. Update token brightness based on attention
    2. Select and cull tokens based on brightness
    3. Select tokens for repair (diffusion)
    4. Apply repairs to both KVMirror and KV cache
    
    This is a synchronous, event-driven approach that runs immediately after 
    each token generation, replacing the previous asynchronous polling tasks.
    """
    
    def __init__(self,
                 kv_mirror_manager: KVMirror,
                 diffuser: DiffuserModel,
                 kv_patcher: KVCachePatcher,
                 main_model: Any,
                 processor: Any):
        """
        Initialize the Context Maintenance handler.
        
        Args:
            kv_mirror_manager: The KVMirror instance managing token state
            diffuser: The diffuser model for token repair
            kv_patcher: The KV cache patcher for applying repairs
            main_model: Reference to the main language model
            processor: Reference to the tokenizer/processor
        """
        self.kv = kv_mirror_manager
        self.diffuser = diffuser
        self.kv_patcher = kv_patcher
        self.main_model = main_model
        self.processor = processor
        self.events = []  # Internal log for maintenance events
        
        # Statistics tracking
        self.stats = {
            "brightness_updates": 0,
            "culling_operations": 0,
            "tokens_culled": 0,
            "repair_operations": 0,
            "tokens_repaired": 0
        }
        
        print("[ContextMaintenance] Initialized")
        
    async def run_phase(self,
                        model_outputs: Any,
                        current_input_ids: torch.Tensor,
                        current_attention_mask: torch.Tensor,
                        current_past_key_values: Optional[Tuple],
                        generation_step: int
                       ) -> Tuple[Optional[Tuple], List[Dict]]:
        """
        Runs the sequential maintenance steps after a token generation.
        
        Args:
            model_outputs: The outputs from the model's forward pass
            current_input_ids: The current input IDs tensor
            current_attention_mask: The current attention mask tensor
            current_past_key_values: The current KV cache
            generation_step: The current generation step number
            
        Returns:
            A tuple containing:
            - Potentially patched KV cache
            - List of maintenance events
        """
        self.events.clear()
        patched_past_key_values = current_past_key_values  # Start with current cache
        
        try:
            # --- 1. Update Brightness ---
            print(f"[Maintenance] Updating brightness for step {generation_step}...", file=sys.stderr)
            brightness_update_result = update_brightness_scores(
                kv_mirror_manager=self.kv,
                outputs=model_outputs,
                generation_step=generation_step,
                decay_per_tick=config.BRIGHTNESS_DECAY_PER_TICK,
                gain_coefficient=config.BRIGHTNESS_GAIN_COEFFICIENT
            )
            
            if "error" not in brightness_update_result:
                self.stats["brightness_updates"] += 1
                self.events.append({
                    "type": "brightness_update",
                    "step": generation_step,
                    "timestamp": time.time(),
                    "details": brightness_update_result
                })
            
            # --- 2. Culling ---
            print("[Maintenance] Selecting tokens for culling...", file=sys.stderr)
            positions_to_cull = select_tokens_for_cull(
                self.kv,
                config.CONTEXT_WINDOW_TARGET
            )
            
            if positions_to_cull:
                print(f"[Maintenance] Culling {len(positions_to_cull)} tokens", file=sys.stderr)
                
                # Log Culling Events BEFORE pruning
                snapshot = self.kv.snapshot()  # Get state before prune
                culled_tokens = []
                
                for pos in positions_to_cull:
                    instance_id = snapshot['kv_mirror'].get(pos)
                    if instance_id and instance_id in snapshot['tokens']:
                        token_info = snapshot['tokens'][instance_id]
                        token_id = token_info.token_id
                        token_text = ""
                        try:
                            token_text = self.processor.decode([token_id])
                        except:
                            token_text = f"<ID:{token_id}>"
                            
                        culled_tokens.append({
                            "position": pos,
                            "token_id": token_id,
                            "token_text": token_text,
                            "brightness": token_info.brightness
                        })
                
                # Prune from KVMirror (RoPE Safe)
                prune_success = self.kv.prune(positions_to_cull)
                print(f"[Maintenance] KVMirror prune success: {prune_success}", file=sys.stderr)
                
                self.stats["culling_operations"] += 1
                self.stats["tokens_culled"] += len(positions_to_cull)
                
                self.events.append({
                    "type": "culling",
                    "step": generation_step,
                    "timestamp": time.time(),
                    "tokens_culled": len(positions_to_cull),
                    "culled_tokens": culled_tokens
                })
                
                # NOTE: We are NOT pruning the KV Cache tensors here, only the mirror.
                # This assumes the main loop continues with the *full* cache,
                # but subsequent position_ids/attention ignores culled tokens.
            
            # --- 3. Repair (Diffusion) ---
            repair_candidates = []
            snapshot = self.kv.snapshot()  # Get potentially updated snapshot after culling
            active_tokens = snapshot['tokens']
            kv_mirror = snapshot['kv_mirror']

            # Identify repair candidates based on brightness
            for position, instance_id in kv_mirror.items():
                if instance_id in active_tokens:
                    token_info = active_tokens[instance_id]
                    # Select if brightness is below REPAIR threshold but above potential CULL threshold
                    # And also below the LOCK threshold
                    if (config.BRIGHTNESS_REPAIR_THRESHOLD is not None and
                        token_info.brightness < config.BRIGHTNESS_REPAIR_THRESHOLD and
                        token_info.brightness < config.BRIGHTNESS_LOCK_THRESHOLD * 255.0):  # Convert normalized threshold to absolute
                        # TODO: Add lower bound check if separate CULL threshold exists
                        repair_candidates.append((position, token_info.brightness, instance_id, token_info.token_id))

            diff_list = []
            if repair_candidates:
                # Simple strategy: Repair the N dimmest *repairable* tokens
                repair_candidates.sort(key=lambda x: x[1])  # Sort by brightness
                num_to_repair = min(len(repair_candidates), config.MAX_REPAIR_TOKENS_PER_STEP)
                selected_to_repair = repair_candidates[:num_to_repair]
                positions_to_repair = [pos for pos, _, _, _ in selected_to_repair]
                original_token_ids = [token_id for _, _, _, token_id in selected_to_repair]

                print(f"[Maintenance] Selecting {len(selected_to_repair)} tokens for repair at positions: {positions_to_repair}", file=sys.stderr)

                # Get the full sequence input IDs from current_input_ids
                # We need to convert the tensor to a flat list/tensor for compute_diff
                input_ids = current_input_ids[0].detach().cpu()  # Shape: [seq_len]

                # Get embeddings for all tokens in the sequence
                # For this synchronous version, we'll extract embeddings directly from the model
                with torch.no_grad():
                    # Use the model's embedding layer to get embeddings for all tokens
                    input_embeddings = self.main_model.get_input_embeddings()(current_input_ids)

                # Create a brightness map - normalize brightness values to 0-1 range for diffuser
                brightness_map = torch.zeros(current_input_ids.shape[1], device=current_input_ids.device)
                for pos, instance_id in kv_mirror.items():
                    if pos < brightness_map.shape[0] and instance_id in active_tokens:
                        # Normalize from 0-255 to 0-1 range
                        brightness_map[pos] = active_tokens[instance_id].brightness / 255.0

                try:
                    # Call compute_diff with the repair indices
                    diff_list = await compute_diff(
                        diffuser_model=self.diffuser,
                        input_embeddings=input_embeddings,
                        attention_mask=current_attention_mask,
                        brightness_map=brightness_map,
                        original_input_ids=input_ids,
                        repair_indices=positions_to_repair
                    )
                    print(f"[Maintenance] compute_diff returned {len(diff_list)} changes", file=sys.stderr)
                except Exception as diff_e:
                    print(f"[Maintenance ERROR] Diffuser compute_diff failed: {diff_e}", file=sys.stderr)
                    diff_list = []  # Ensure empty list on error
            
            # --- 4. Apply Repair Diffs (if any) ---
            if diff_list:
                print(f"[Maintenance] Applying {len(diff_list)} repair diffs...", file=sys.stderr)
                
                # Apply to KVMirror
                update_summary = self.kv.apply_diff(diff_list)
                print(f"[Maintenance] KVMirror apply_diff summary: {update_summary}", file=sys.stderr)
                
                # Apply to KV Cache Tensors
                if patched_past_key_values is not None:
                    patched_past_key_values = await self.kv_patcher.patch(
                        patched_past_key_values,
                        diff_list
                    )
                    print(f"[Maintenance] KV Cache patched", file=sys.stderr)
                
                self.stats["repair_operations"] += 1
                self.stats["tokens_repaired"] += len(diff_list)
                
                # Create a token repair event with the correct format expected by tests
                self.events.append({
                    "type": "token_repair",
                    "step": generation_step,
                    "timestamp": time.time(),
                    "tokens_repaired": len(diff_list),
                    "repair_info": diff_list  # Use repair_info as expected by tests
                })
            
        except Exception as e:
            print(f"[Maintenance ERROR] Exception during maintenance phase: {type(e).__name__}: {str(e)}", file=sys.stderr)
            self.events.append({
                "type": "error",
                "step": generation_step,
                "timestamp": time.time(),
                "error": f"{type(e).__name__}: {str(e)}"
            })
        
        return patched_past_key_values, self.events
        
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about maintenance operations.
        
        Returns:
            Dictionary with operation counts
        """
        return self.stats.copy()
