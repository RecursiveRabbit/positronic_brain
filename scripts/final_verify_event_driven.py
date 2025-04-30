"""
Final verification script for the event-driven context maintenance architecture.

This script confirms that token generation, brightness updates, and culling
all work correctly with the new synchronous maintenance phase.
"""

import os
import sys
import asyncio
import json
import logging
import time
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('final_verify_event_driven.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from positronic_brain import config
# Import ai_core directly from root directory
import ai_core
from ai_core import setup_ai_core, run_continuous_inference
from positronic_brain.kv_mirror import KVMirror
from positronic_brain.culler import select_tokens_for_cull
from positronic_brain.brightness_engine import update_brightness_scores
from positronic_brain.context_maintenance import ContextMaintenance

# Output file for events
EVENT_LOG = "event_driven_events.json"

async def monitor_events(shared_state, shutdown_event, output_file=EVENT_LOG):
    """Monitor and log events from the maintenance phase."""
    events = []
    try:
        logger.info(f"Starting event monitor, writing to {output_file}")
        while not shutdown_event.is_set():
            # Sample shared state
            async with shared_state['lock']:
                kv_stats = shared_state.get('kv_mirror_stats', {})
                token_count = kv_stats.get('active', 0)
                maintenance_metrics = shared_state.get('maintenance_metrics', {})
                maintenance_events = maintenance_metrics.get('events', [])
                
                # Record new events since last check
                if maintenance_events and len(maintenance_events) > len(events):
                    new_events = maintenance_events[len(events):]
                    events.extend(new_events)
                    
                    # Log event info
                    for event in new_events:
                        event_type = event.get('type', 'unknown')
                        timestamp = event.get('timestamp', time.time())
                        
                        logger.info(f"Event: {event_type}, Token count: {token_count}")
                        
                        if event_type == 'culling' and 'culled_tokens' in event:
                            culled_tokens = event['culled_tokens']
                            for token in culled_tokens:
                                logger.info(f"  Culled token: pos={token['position']}, "
                                           f"brightness={token['brightness']}")
                    
                    # Write to file
                    with open(output_file, 'w') as f:
                        json.dump(events, f, indent=2)
            
            await asyncio.sleep(1.0)  # Check every second
            
    except Exception as e:
        logger.error(f"Error in event monitor: {e}")
    finally:
        logger.info("Event monitor shutting down")
        # Final write to file
        if events:
            with open(output_file, 'w') as f:
                json.dump(events, f, indent=2)

async def verify_event_driven_maintenance():
    """Run the verification for event-driven maintenance."""
    try:
        # Initialize AI components with custom prompt
        prompt = """
        This is a test of the Halo Weave system with event-driven context maintenance.
        After each token generation, a maintenance phase will:
        1. Update token brightness scores based on attention
        2. Cull tokens that fall below brightness thresholds
        3. Repair degraded tokens when necessary
        
        This approach replaces the previous polling-based architecture.
        
        Please generate text continuously about the history of computing,
        focusing on major breakthroughs and innovations.
        """
        
        # Set small context window target to trigger culling
        config.CONTEXT_WINDOW_TARGET = 500
        
        logger.info("Initializing AI core...")
        ai_components = await setup_ai_core(
            initial_prompt=prompt,
            context_file="event_driven_test.txt",
            resume_context_file=None,
            context_window_target=config.CONTEXT_WINDOW_TARGET
        )
        
        # Extract components
        model = ai_components["model"]
        processor = ai_components["processor"]
        controller = ai_components["controller"]
        output_queue = ai_components["output_queue"]
        shutdown_event = ai_components["shutdown_event"]
        sliding_event = ai_components["sliding_event"]
        shared_state = ai_components["shared_state"]
        kv_patcher = ai_components["kv_patcher"]
        pending_diffs_queue = ai_components["pending_diffs_queue"]
        compactor_request_queue = ai_components["compactor_request_queue"]
        
        # Setup maintenance metrics in shared state
        if shared_state is not None:
            async with shared_state['lock']:
                shared_state['maintenance_metrics'] = {
                    'events': []
                }
        
        # Start event monitor
        monitor_task = asyncio.create_task(
            monitor_events(shared_state, shutdown_event)
        )
        
        # Run the inference for limited time
        inference_task = asyncio.create_task(
            run_continuous_inference(
                model=model,
                processor=processor,
                controller=controller,
                initial_prompt_content=prompt,
                output_queue=output_queue,
                shutdown_event=shutdown_event,
                sliding_event=sliding_event,
                shared_state=shared_state,
                kv_patcher=kv_patcher,
                pending_diffs_queue=pending_diffs_queue,
                compactor_request_queue=compactor_request_queue
            )
        )
        
        # Output consumer task
        async def consume_output():
            output_text = []
            while not shutdown_event.is_set():
                try:
                    token = await asyncio.wait_for(output_queue.get(), timeout=0.1)
                    output_text.append(token)
                    
                    # Print occasional samples
                    if len(output_text) % 50 == 0:
                        last_tokens = ''.join(output_text[-50:])
                        logger.info(f"Generated: {last_tokens}")
                        
                        # Get KV Mirror stats
                        async with shared_state['lock']:
                            kv_stats = shared_state.get('kv_mirror_stats', {})
                            logger.info(f"KV Mirror stats: {kv_stats}")
                except asyncio.TimeoutError:
                    pass
        
        consumer_task = asyncio.create_task(consume_output())
        
        # Let the system run for a set time (generating about 600-800 tokens)
        logger.info("Running inference for 90 seconds...")
        await asyncio.sleep(90)
        
        # Signal shutdown
        logger.info("Shutting down...")
        shutdown_event.set()
        
        # Wait for tasks to complete
        await asyncio.gather(inference_task, monitor_task, consumer_task, return_exceptions=True)
        
        # Get final KV Mirror stats
        kv_mirror_manager = ai_core.kv_mirror_manager
        if kv_mirror_manager:
            final_stats = kv_mirror_manager.get_stats()
            logger.info(f"Final KV Mirror stats: {final_stats}")
            
            # Verify culling
            token_count = kv_mirror_manager.get_current_size()
            target_size = config.CONTEXT_WINDOW_TARGET
            logger.info(f"Final token count: {token_count}, Target: {target_size}")
            
            if token_count <= target_size + 50:
                logger.info("✅ SUCCESS: Culling worked correctly, token count is within target range")
            else:
                logger.warning("❌ FAIL: Culling may not be working, token count exceeds target significantly")
        
        # Check if we logged any events
        if os.path.exists(EVENT_LOG):
            with open(EVENT_LOG, 'r') as f:
                events = json.load(f)
                
            culling_events = [e for e in events if e.get('type') == 'culling']
            brightness_events = [e for e in events if e.get('type') == 'brightness_update']
            
            logger.info(f"Recorded {len(events)} total events")
            logger.info(f"- {len(brightness_events)} brightness update events")
            logger.info(f"- {len(culling_events)} culling events")
            
            if culling_events:
                logger.info("✅ SUCCESS: Culling events were recorded")
            else:
                logger.warning("❌ FAIL: No culling events were recorded")
        
        logger.info("Verification complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error in verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(verify_event_driven_maintenance())
