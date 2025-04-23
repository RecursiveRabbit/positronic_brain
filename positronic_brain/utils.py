import asyncio
import os
import sys
from typing import List, Dict, Any

# State for Top Predicted Tokens (Moved from ai_core.py)
_top_predicted_tokens: List[Dict[str, Any]] = []
_top_tokens_lock = asyncio.Lock()

async def get_top_tokens(count=10):
    """Get the top predicted tokens with their probabilities.
    
    Returns:
        List of dicts, each with 'token', 'token_id', and 'probability' keys
    """
    # Access module-level state directly
    async with _top_tokens_lock:
        # Return a copy to avoid modification during iteration
        return _top_predicted_tokens[:count] if _top_predicted_tokens else []


async def update_top_tokens(new_token_list: List[Dict[str, Any]]):
    """Atomically update the top predicted tokens list."""
    global _top_predicted_tokens  # Needed to modify module-level list
    async with _top_tokens_lock:
        _top_predicted_tokens = new_token_list


async def _async_save_context_text(text_to_save, file_path):
    """Asynchronously save text content to a file"""
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Write the text to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_to_save)
        return True
    except Exception as e:
        print(f"[Resume] Error saving context: {e}", file=sys.stderr)
        return False
