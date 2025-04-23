import pytest
import torch
import threading
import time
# Import the KVMirror class
from positronic_brain.kv_mirror import KVMirror

# --- Define Fixtures ---
@pytest.fixture
def empty_mirror():
    """Provide a fresh, empty KVMirror."""
    # Ensure a clean state for each test using this fixture
    mirror = KVMirror()
    mirror.clear() # This should reset the internal state including _global_generation_step
    return mirror

@pytest.fixture
def populated_mirror(empty_mirror):
    """Provide a KVMirror with 10 tokens already added."""
    mirror = empty_mirror # Start with a clean mirror
    for i in range(10):
        mirror.add(i + 100, i)
    return mirror
