import os
import torch
from typing import Dict, Optional, Tuple, Any
from transformers import PreTrainedTokenizer

# Note: Using 'context_window_target' consistently to avoid confusion with 'max_context_window'.

def save_context(input_ids, processor, file_path="context_history.txt"):
    """Save the current context to a text file"""
    try:
        # Decode current context
        context_text = processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        # Save context to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(context_text)
        print(f"[Persistence] Saved context ({input_ids.shape[1]} tokens) to {file_path}")
        return True
    except Exception as e:
        print(f"[Persistence] Error saving context: {e}")
        return False


def load_context(processor, context_window_target, file_path="context_history.txt", device="cuda"):
    """Load context from a file and return tokenized tensors"""
    if not os.path.exists(file_path):
        print(f"[Persistence] Context file {file_path} not found.")
        return None, None
        
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
        
        return input_ids, attention_mask
        
    except Exception as e:
        print(f"[Persistence] Error loading context: {e}")
        return None, None
