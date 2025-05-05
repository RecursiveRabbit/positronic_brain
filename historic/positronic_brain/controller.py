import asyncio
import sys
from typing import List, Tuple, Optional, Dict, Any
import torch
from .metrics import timed_histogram

class SimpleContextController:
    """Manages context injection into the ongoing generation stream"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pending_events = []
        self.lock = asyncio.Lock()  # For safe asynchronous access

    @timed_histogram("controller_inject_event_seconds")
    async def inject_event(self, text_event: str, source: str = "USER", shared_state=None):
        """Injects a text event into the queue."""
        async with self.lock:
            # Add formatting to distinguish injected text
            # Example: [USER]: Hello there!
            # Example: [SYSTEM]: Time tick update.
            formatted_event = f"\n[{source.upper()}]: {text_event}\n"  # Add newlines for separation
            print(f"\n[Controller Queueing: '{formatted_event.strip()}']\n", file=sys.stderr)
            self.pending_events.append(formatted_event)
            
            # Check if generation is paused and needs to be resumed
            if shared_state and 'resume_generation_event' in shared_state:
                async with shared_state['lock']:
                    resume_event = shared_state['resume_generation_event']
                    if not resume_event.is_set():
                        resume_event.set()
                        print("[Controller] Signaling generation loop to resume", file=sys.stderr)

    @timed_histogram("controller_process_updates_seconds")
    async def process_pending_updates(self, device):
        """
        Checks for pending events and tokenizes them without modifying the main context.
        Returns (update_tokens, update_attention_mask, update_applied_flag)
        """
        updates_to_apply_text = []
        async with self.lock:
            if not self.pending_events:
                return None, None, False  # No updates

            # Get all pending events and clear queue
            updates_to_apply_text = self.pending_events[:]
            self.pending_events.clear()

        if not updates_to_apply_text:  # Should not happen due to initial check, but safe
            return None, None, False

        # --- Process and Tokenize Events ---
        full_update_text = "".join(updates_to_apply_text)
        print(f"\n[Controller Processing: '{full_update_text.strip()}']\n", file=sys.stderr)

        # Tokenize the combined update text
        # Important: add_special_tokens=False prevents adding BOS/EOS tokens mid-stream
        update_tokens = self.tokenizer(
            full_update_text,
            return_tensors="pt",
            add_special_tokens=False  # Crucial! Don't add BOS/EOS here
        ).input_ids.to(device)  # Move to the correct device
        
        # For KV mirror, store the tokens for registration later
        update_token_ids = update_tokens[0].cpu().tolist()

        # Create attention mask for the new tokens (all 1s)
        update_attention_mask = torch.ones_like(update_tokens).to(device)

        # Check if any tokens were produced
        if update_tokens.shape[1] > 0:  
            print(f"[Controller] Prepared update tokens. Length: {update_tokens.shape[1]}", file=sys.stderr)
            return update_tokens, update_attention_mask, True
        else:
            # No tokens were generated from the update text (e.g., empty string)
            print("[Controller] Warning: Update text produced no tokens.", file=sys.stderr)
            return None, None, False
