"""
Shared test utilities for positronic_brain tests.

This module contains simple utility functions used across multiple test files.
"""

import os
import torch
import tempfile
import shutil
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def safe_save(obj: Any, filename: str) -> str:
    """
    Atomically save an object to a file using a temporary file approach.
    
    Args:
        obj: The object to save (typically a tensor, dict, or other serializable object)
        filename: The target filename to save to
        
    Returns:
        The filename where the object was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Create a temporary file in the same directory
    temp_file = f"{filename}.tmp"
    
    try:
        # Save to the temporary file first
        torch.save(obj, temp_file)
        
        # Rename to target filename (atomic operation)
        if os.path.exists(filename):
            os.unlink(filename)
        os.rename(temp_file, filename)
        
        logger.debug(f"Successfully saved to {filename}")
        return filename
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        logger.error(f"Error saving to {filename}: {e}")
        raise
        
def safe_load(filename: str, default: Optional[Any] = None, device: Optional[torch.device] = None) -> Any:
    """
    Safely load an object from a file with error handling.
    
    Args:
        filename: The file to load from
        default: Default value to return if file doesn't exist
        device: Optional device to map tensors to (if loading torch tensors)
        
    Returns:
        The loaded object or default if file doesn't exist
    """
    if not os.path.exists(filename):
        logger.warning(f"File not found: {filename}, returning default")
        return default
        
    try:
        if device is not None:
            return torch.load(filename, map_location=device)
        else:
            return torch.load(filename)
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        if default is not None:
            logger.warning(f"Returning default value instead")
            return default
        raise
