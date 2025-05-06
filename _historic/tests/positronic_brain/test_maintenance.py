"""
Test script for the new event-driven context maintenance architecture.

This script verifies the integration of the ContextMaintenance class which
replaces the separate background tasks for brightness updates, culling, and repair.
"""

import os
import sys
import asyncio
import torch
import time

# Add parent directory to path to import positronic_brain modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

from positronic_brain import config
from positronic_brain.kv_mirror import KVMirror
from positronic_brain.diffuser_runner import DiffuserModel
from positronic_brain.kv_patcher import KVCachePatcher
from positronic_brain.model_io import load_model
from positronic_brain.context_maintenance import ContextMaintenance

# For logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def simulate_token_generation(maintenance_handler, model, processor, device):
    """
    Simulate token generation and test the maintenance handler's ability 
    to update brightness, cull tokens, and repair when needed.
    """
    # Initialize KV Mirror
    kv_mirror = KVMirror()
    
    # Create test input text
    test_text = "This is a test to verify the event-driven context maintenance architecture works correctly."
    logger.info(f"Test text: {test_text}")
    
    # Create a small input sequence for testing
    inputs = processor(test_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    past_key_values = None
    
    logger.info(f"Input tokens: {input_ids.shape[1]}")
    
    # Register initial tokens with KV Mirror
    for i in range(input_ids.shape[1]):
        token_id = input_ids[0, i].item()
        kv_mirror.add(token_id=token_id, position=i, source='system_init')
    
    # Generate tokens and test maintenance
    num_steps = 50
    logger.info(f"Generating {num_steps} tokens and testing maintenance...")
    
    for step in range(num_steps):
        # Execute forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=True,
                use_cache=True
            )
        
        # Get logits and KV cache
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        # Sample next token (simplified)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        
        # Append to input_ids and update attention_mask
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        ], dim=1)
        
        # Register token in KV Mirror
        position = input_ids.shape[1] - 1
        token_id = next_token_id.item()
        kv_mirror.add(token_id=token_id, position=position, source='llm')
        
        # Print generated token
        token_text = processor.decode([token_id])
        logger.info(f"Step {step}, Generated: {token_text}, Position: {position}")
        
        # Run maintenance phase
        logger.info(f"Running maintenance for step {step}...")
        patched_cache, events = await maintenance_handler.run_phase(
            model_outputs=outputs,
            current_input_ids=input_ids,
            current_attention_mask=attention_mask,
            current_past_key_values=past_key_values,
            generation_step=step
        )
        
        # Use the patched cache if available
        if patched_cache is not None:
            past_key_values = patched_cache
        
        # Log maintenance events
        if events:
            for event in events:
                logger.info(f"Maintenance Event: {event['type']}")
                if event['type'] == 'culling' and 'culled_tokens' in event:
                    for token in event['culled_tokens']:
                        logger.info(f"  Culled token: pos={token['position']}, "
                                   f"text={token['token_text']}, brightness={token['brightness']}")
        
        # Print KV Mirror stats
        stats = kv_mirror.get_stats()
        logger.info(f"KV Mirror stats: {stats}")
        
        # Simulate delay between iterations
        await asyncio.sleep(0.1)
    
    logger.info("Token generation simulation completed!")
    return True

async def main():
    """Main test function"""
    logger.info("Starting context maintenance test...")
    
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    logger.info(f"Loading model: {model_name}")
    
    model, processor = load_model(model_name, trust_remote_code=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize KV Mirror
    kv_mirror = KVMirror()
    
    # Initialize KV Patcher
    kv_patcher = KVCachePatcher(model)
    
    # Initialize Diffuser
    diffuser = DiffuserModel(model_name=config.DIFFUSER_MODEL_NAME)
    
    # Initialize Context Maintenance handler
    maintenance_handler = ContextMaintenance(
        kv_mirror_manager=kv_mirror,
        diffuser=diffuser,
        kv_patcher=kv_patcher,
        main_model=model,
        processor=processor
    )
    
    # Run the test
    success = await simulate_token_generation(
        maintenance_handler, model, processor, device
    )
    
    if success:
        logger.info("✅ Test completed successfully!")
    else:
        logger.error("❌ Test failed!")

if __name__ == "__main__":
    asyncio.run(main())
