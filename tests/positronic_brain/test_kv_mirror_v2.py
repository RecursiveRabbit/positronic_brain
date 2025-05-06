"""
Unit tests for the KVMirror v2 implementation.

Tests the core functionality of the new KVMirror class, ensuring
it correctly maintains a ledger of token history and a KV index
that mirrors the real KV cache structure.
"""

import pytest
from positronic_brain.kv_mirror import KVMirror


def test_kv_mirror_init():
    """Test that the KVMirror initializes with empty ledger and KV index."""
    mirror = KVMirror()
    assert isinstance(mirror.ledger, dict)
    assert isinstance(mirror.kv_index, list)
    assert len(mirror.ledger) == 0
    assert len(mirror.kv_index) == 0


def test_append_token():
    """Test appending tokens to the KVMirror."""
    mirror = KVMirror()
    
    # Append some tokens
    ledger_id1 = mirror.append_token(123, "hello", 255.0)
    ledger_id2 = mirror.append_token(456, "world", 200.0)
    
    # Check ledger contains the tokens
    assert len(mirror.ledger) == 2
    assert ledger_id1 in mirror.ledger
    assert ledger_id2 in mirror.ledger
    
    # Check token metadata
    assert mirror.ledger[ledger_id1]["token_id"] == 123
    assert mirror.ledger[ledger_id1]["plaintext"] == "hello"
    assert mirror.ledger[ledger_id1]["brightness"] == 255.0
    assert mirror.ledger[ledger_id1]["replacement_history"] == []
    
    # Check KV index matches
    assert len(mirror.kv_index) == 2
    assert mirror.kv_index[0] == ledger_id1
    assert mirror.kv_index[1] == ledger_id2


def test_replace_token():
    """Test replacing tokens in the KVMirror."""
    mirror = KVMirror()
    
    # Add initial tokens
    ledger_id1 = mirror.append_token(123, "hello", 255.0)
    ledger_id2 = mirror.append_token(456, "world", 200.0)
    
    # Replace the first token
    success = mirror.replace_token(0, 789, "goodbye")
    assert success == True
    
    # The KV index should still have the same length
    assert len(mirror.kv_index) == 2
    
    # But the ledger ID at position 0 should be different
    assert mirror.kv_index[0] != ledger_id1
    
    # The ledger should now have 3 entries (2 original + 1 replacement)
    assert len(mirror.ledger) == 3
    
    # Check the replacement token
    new_ledger_id = mirror.kv_index[0]
    assert mirror.ledger[new_ledger_id]["token_id"] == 789
    assert mirror.ledger[new_ledger_id]["plaintext"] == "goodbye"
    
    # The replacement history should be recorded
    assert len(mirror.ledger[new_ledger_id]["replacement_history"]) == 1
    old_id, new_id, _ = mirror.ledger[new_ledger_id]["replacement_history"][0]
    assert old_id == 123
    assert new_id == 789


def test_prune_tokens():
    """Test pruning tokens from the KVMirror."""
    mirror = KVMirror()
    
    # Add initial tokens
    for i in range(5):
        mirror.append_token(i+100, f"token_{i}", 200.0)
    
    # Initial state checks
    assert len(mirror.kv_index) == 5
    assert len(mirror.ledger) == 5
    
    # Prune tokens at positions 1 and 3
    success = mirror.prune_tokens([1, 3])
    assert success == True
    
    # KV index should now have 3 entries
    assert len(mirror.kv_index) == 3
    
    # Ledger should still have all 5 entries
    assert len(mirror.ledger) == 5
    
    # The remaining positions should have the correct tokens
    assert mirror.ledger[mirror.kv_index[0]]["token_id"] == 100  # First token
    assert mirror.ledger[mirror.kv_index[1]]["token_id"] == 102  # Was at position 2
    assert mirror.ledger[mirror.kv_index[2]]["token_id"] == 104  # Was at position 4


def test_snapshot():
    """Test creating a snapshot of the KVMirror state."""
    mirror = KVMirror()
    
    # Add some tokens
    for i in range(3):
        mirror.append_token(i+100, f"token_{i}", 200.0)
    
    # Replace the middle token
    mirror.replace_token(1, 999, "replaced")
    
    # Get a snapshot
    snapshot = mirror.snapshot()
    
    # Check the snapshot structure
    assert snapshot["kv_length"] == 3
    assert snapshot["ledger_size"] == 4  # 3 original + 1 replacement
    assert len(snapshot["tokens"]) == 3
    
    # Check the token details
    tokens = snapshot["tokens"]
    assert tokens[0]["position"] == 0
    assert tokens[0]["token_id"] == 100
    assert tokens[0]["plaintext"] == "token_0"
    assert tokens[0]["has_replacements"] == False
    
    assert tokens[1]["position"] == 1
    assert tokens[1]["token_id"] == 999
    assert tokens[1]["plaintext"] == "replaced"
    assert tokens[1]["has_replacements"] == True
    
    assert tokens[2]["position"] == 2
    assert tokens[2]["token_id"] == 102
    assert tokens[2]["plaintext"] == "token_2"


def test_invalid_operations():
    """Test handling of invalid operations."""
    mirror = KVMirror()
    
    # Add a token
    mirror.append_token(123, "token", 255.0)
    
    # Try to replace a non-existent position
    success = mirror.replace_token(999, 456, "invalid")
    assert success == False
    
    # Try to prune invalid positions
    success = mirror.prune_tokens([5, 10])
    assert success == False
    
    # The mirror state should be unchanged
    assert len(mirror.kv_index) == 1
    assert len(mirror.ledger) == 1
