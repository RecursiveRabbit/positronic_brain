"""
KV Mirror - A data structure that maintains a mirror of the KV cache with token metadata.

This implementation follows a flat, explicit design with two primary data structures:
1. Ledger: Dictionary mapping token IDs to metadata (persistent history of all tokens)
2. KV Index List: Maps positions in the KV cache to ledger token IDs (mirrors cache structure)
"""

from typing import Dict, List, Any, Optional, Tuple
import time

class KVMirror:
    """
    KVMirror maintains a mirror of the KV cache content with metadata about each token.
    
    The mirror consists of two data structures:
    1. ledger: A dictionary mapping unique ledger IDs to token metadata.
    2. kv_index: A list mapping KV cache positions to ledger IDs.
    
    This design maintains token history in the ledger while allowing the kv_index
    to be manipulated exactly like the real KV cache is manipulated.
    """
    
    def __init__(self):
        """Initialize an empty KV Mirror."""
        self.ledger: Dict[int, Dict[str, Any]] = {}  # Maps ledger_id -> token metadata
        self.kv_index: List[int] = []                # Maps KV position -> ledger_id
        self._next_ledger_id: int = 1                # Start IDs at 1 (0 reserved for special cases)
    
    def append_token(self, token_id: int, plaintext: str, brightness: float = 255.0) -> int:
        """
        Add a new token to the mirror.
        
        Args:
            token_id: The token ID from the tokenizer
            plaintext: The decoded string representation of the token
            brightness: Initial brightness value for the token (default: 255.0)
            
        Returns:
            The assigned ledger ID for the token
        """
        # Create a unique ledger ID for this token
        ledger_id = self._next_ledger_id
        self._next_ledger_id += 1
        
        # Create token metadata in the ledger
        self.ledger[ledger_id] = {
            "token_id": token_id,
            "plaintext": plaintext,
            "brightness": brightness,
            "replacement_history": []
        }
        
        # Add the ledger ID to the KV index
        self.kv_index.append(ledger_id)
        
        return ledger_id
    
    def replace_token(self, kv_position: int, new_token_id: int, new_plaintext: str) -> bool:
        """
        Replace a token at the specified KV cache position with a new token.
        
        Args:
            kv_position: Position in the KV cache (0-indexed)
            new_token_id: New token ID to replace with
            new_plaintext: Plaintext representation of the new token
            
        Returns:
            True if replacement was successful, False otherwise
        """
        # Validate position
        if kv_position < 0 or kv_position >= len(self.kv_index):
            return False
        
        # Get the ledger ID for the token at this position
        old_ledger_id = self.kv_index[kv_position]
        
        # Verify the old ledger ID exists
        if old_ledger_id not in self.ledger:
            return False
        
        # Get the old token's metadata for the replacement history
        old_token = self.ledger[old_ledger_id]
        old_token_id = old_token["token_id"]
        
        # Create a new ledger entry for the replacement token
        new_ledger_id = self._next_ledger_id
        self._next_ledger_id += 1
        
        # Copy the brightness from the old token
        brightness = old_token["brightness"]
        
        # Create the new token entry
        self.ledger[new_ledger_id] = {
            "token_id": new_token_id,
            "plaintext": new_plaintext,
            "brightness": brightness,
            "replacement_history": []
        }
        
        # Record this replacement in the history
        replacement_record = (old_token_id, new_token_id, time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.ledger[new_ledger_id]["replacement_history"].append(replacement_record)
        
        # Update the KV index to point to the new ledger ID
        self.kv_index[kv_position] = new_ledger_id
        
        return True
    
    def prune_tokens(self, kv_positions: List[int]) -> bool:
        """
        Remove tokens at the specified KV cache positions.
        
        Args:
            kv_positions: List of KV cache positions to remove
            
        Returns:
            True if pruning was successful, False if any position was invalid
        """
        # Validate positions
        for pos in kv_positions:
            if pos < 0 or pos >= len(self.kv_index):
                return False
        
        # Sort positions in descending order to avoid index shifting issues
        sorted_positions = sorted(kv_positions, reverse=True)
        
        # Remove each position from the KV index
        for pos in sorted_positions:
            self.kv_index.pop(pos)
        
        return True
    
    def snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of the current KV mirror state.
        
        Returns:
            Dictionary containing the current state of the KV mirror
        """
        # Collect token details from the KV index
        token_details = []
        for pos, ledger_id in enumerate(self.kv_index):
            if ledger_id in self.ledger:
                token = self.ledger[ledger_id]
                token_details.append({
                    "position": pos,
                    "ledger_id": ledger_id,
                    "token_id": token["token_id"],
                    "plaintext": token["plaintext"],
                    "brightness": token["brightness"],
                    "has_replacements": len(token["replacement_history"]) > 0
                })
        
        return {
            "kv_length": len(self.kv_index),
            "ledger_size": len(self.ledger),
            "next_ledger_id": self._next_ledger_id,
            "tokens": token_details
        }
