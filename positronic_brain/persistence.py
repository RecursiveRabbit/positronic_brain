import os
import json
import torch
import threading
from typing import Dict, Optional, List, Tuple, Any
from transformers import PreTrainedTokenizer

# Note: Depends on global 'token_map' from ai_core.py (Legacy)
# TODO: Refactor persistence away from token_map later.
# Note: Using 'context_window_target' consistently to avoid confusion with 'max_context_window'.

def save_context(input_ids, processor, file_path="context_history.txt", token_map_path=None):
    """Save the current context to a file and optionally save the token map to a separate file"""
    try:
        # Decode current context
        context_text = processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        # Save context to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(context_text)
        print(f"[Persistence] Saved context ({input_ids.shape[1]} tokens) to {file_path}")
        
        # Save token map if path is provided
        if token_map_path:
            # Create a thread-safe copy of the global token map using a lock
            # We need to handle this carefully to work in both async and non-async contexts
            token_map_copy = None
            
            # For thread safety, we'll use a regular threading lock instead of asyncio.Lock
            # This avoids using 'async with' or 'await' at the module level
            token_map_copy_lock = threading.Lock()
            
            # Direct import - use with caution to avoid circular imports
            import ai_core
            
            with token_map_copy_lock:
                # Make a direct copy of the token map
                token_map_copy = ai_core.token_map[:]
                
            print("[Persistence] Copied token map for saving")
            
            # Convert None values to null for JSON serialization
            serializable_map = [None if x is None else int(x) for x in token_map_copy]
            
            # Save the token map to a JSON file
            with open(token_map_path, "w", encoding="utf-8") as f:
                json.dump(serializable_map, f)
            print(f"[Persistence] Saved token map to {token_map_path}")
        
        return True
    except Exception as e:
        print(f"[Persistence] Error saving context: {e}")
        return False


def load_context(processor, context_window_target, file_path="context_history.txt", token_map_path=None, device="cuda"):
    """Load context from a file and return tokenized tensors and token map"""
    # Direct import - use with caution to avoid circular imports
    import ai_core
    
    if not os.path.exists(file_path):
        print(f"[Persistence] Context file {file_path} not found.")
        return None, None, None
        
    try:
        # Load text content
        with open(file_path, "r", encoding="utf-8") as f:
            loaded_text = f.read()
        print(f"[Persistence] Loaded context from {file_path}")
        
        # Re-tokenize the saved text
        re_tokenized = processor.tokenizer(
            loaded_text,
            return_tensors="pt",
            max_length=context_window_target,  # Use context_window_target consistently
            truncation=True,  # Truncate if somehow longer
            add_special_tokens=False  # Avoid adding BOS/EOS if they weren't saved
        )
        input_ids = re_tokenized["input_ids"].to(device)
        
        # Create attention mask (all ones matching input_ids shape)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # Check length
        if input_ids.shape[1] < context_window_target:
            print(f"[Persistence Warning] Loaded context shorter ({input_ids.shape[1]}) than context_window_target ({context_window_target}).")
        elif input_ids.shape[1] > context_window_target:
            print(f"[Persistence Warning] Re-tokenized context longer ({input_ids.shape[1]}) than context_window_target ({context_window_target}). Truncating.")
            input_ids = input_ids[:, -context_window_target:]
            attention_mask = attention_mask[:, -context_window_target:]
        
        # Load token map if available
        loaded_token_map = None
        if token_map_path and os.path.exists(token_map_path):
            try:
                with open(token_map_path, "r", encoding="utf-8") as f:
                    loaded_token_map = json.load(f)
                print(f"[Persistence] Loaded token map from {token_map_path}")
                
                # Check if loaded map matches the max context window size
                if len(loaded_token_map) != context_window_target:
                    print(f"[Persistence Warning] Loaded token map length ({len(loaded_token_map)}) doesn't match context_window_target ({context_window_target}). Using reconstructed map.")
                    loaded_token_map = None
                    
            except Exception as e:
                print(f"[Persistence] Error loading token map: {e}")
                loaded_token_map = None
        
        # If we couldn't load the token map, reconstruct it from input_ids (best effort)
        if loaded_token_map is None:
            print("[Persistence] Reconstructing token map from input_ids")
            loaded_token_map = [None] * context_window_target  # Use context_window_target consistently
            for i in range(min(input_ids.shape[1], context_window_target)):
                loaded_token_map[i] = int(input_ids[0, i].item())
        
        # Update the global token map with a regular threading lock instead of asyncio lock
        # This avoids async with syntax which can't be used at module level
        temp_lock = threading.Lock()
        
        try:
            print("[Persistence] Updating global token_map with loaded data...")
            with temp_lock:
                ai_core.token_map[:] = loaded_token_map[:]
            print("[Persistence] Successfully updated global token_map.")
        except Exception as e:
            print(f"[Persistence] Error updating global token_map: {e}")
            
        return input_ids, attention_mask, loaded_token_map
        
    except Exception as e:
        print(f"[Persistence] Error loading context: {e}")
        return None, None, None
