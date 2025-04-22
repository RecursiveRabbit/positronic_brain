import asyncio
import os
import sys
from typing import List, Dict, Any

# Note: get_top_tokens currently depends on global variables
# 'top_predicted_tokens' and 'top_tokens_lock' defined in ai_core.py
# TODO: Refactor state management for top tokens later.

async def get_top_tokens(count=10):
    """Get the top predicted tokens with their probabilities.
    
    Returns:
        List of dicts, each with 'token', 'token_id', and 'probability' keys
    """
    # Direct import - use with caution to avoid circular imports
    import ai_core
    
    async with ai_core.top_tokens_lock:
        # Return a copy to avoid modification during iteration
        return ai_core.top_predicted_tokens[:count] if ai_core.top_predicted_tokens else []


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
