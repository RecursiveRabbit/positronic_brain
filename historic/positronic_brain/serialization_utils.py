"""
Utility functions for serialization in Positronic Brain.
"""
import torch
from typing import Any, Optional
import pickle
import os

def safe_save(data: Any, path: str) -> None:
    """
    Save data to a file with compatibility for various PyTorch versions.
    
    Args:
        data: The data to save
        path: The file path to save to
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        # Try using pickle directly for better control
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
    except Exception as e:
        print(f"Pickle save failed with error: {e}, falling back to torch.save")
        try:
            # Try torch.save with protocol specification
            torch.save(data, path, pickle_protocol=4)
        except TypeError:
            # Fallback for older PyTorch versions that don't support pickle_protocol
            torch.save(data, path)


def safe_load(path: str) -> Any:
    """
    Load data from a file with compatibility for various PyTorch versions.
    
    Args:
        path: The file path to load from
        
    Returns:
        The loaded data
    """
    # First try to register DynamicCache as a safe global if possible
    try:
        from transformers.cache_utils import DynamicCache
        try:
            torch.serialization.add_safe_globals([DynamicCache])
            print("Registered DynamicCache as a safe global for serialization")
        except (AttributeError, ImportError):
            print("PyTorch version does not support add_safe_globals, using fallback methods")
    except ImportError:
        print("Could not import DynamicCache, using fallback methods")
    
    # Ensure the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}")
    
    # Try different loading approaches
    try:
        # First try with default settings
        return torch.load(path)
    except Exception as e1:
        try:
            # If that fails, try with weights_only=False (PyTorch 2.6+)
            return torch.load(path, weights_only=False)
        except Exception as e2:
            try:
                # If torch.load fails completely, try with pickle directly
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e3:
                # If all methods fail, raise a comprehensive error
                raise RuntimeError(
                    f"Failed to load file {path} with multiple methods:\n"
                    f"1. torch.load: {e1}\n"
                    f"2. torch.load(weights_only=False): {e2}\n"
                    f"3. pickle.load: {e3}"
                )
