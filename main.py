"""
FastAPI Application for Infinite Scroll
Serves a web interface and API for interacting with the AI stream
"""

import asyncio
import sys
import uvicorn
from contextlib import asynccontextmanager  # Import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import ai_core
from api_models import SamplerUpdate, TokenBiasUpdate, TokenBiasPhrase, TokenInfo, TopTokensResponse
from positronic_brain.metrics import init_metrics_server
from positronic_brain.compactor import compactor_task

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    global ai_components, inference_task, broadcast_task, compactor_task
    print("Application startup...")
    
    # Initialize the Prometheus metrics server
    init_metrics_server(port=9100)
    
    # Initialize AI components
    ai_components = await ai_core.setup_ai_core()
    
    # Start token broadcaster task
    broadcast_task = asyncio.create_task(broadcast_tokens())
    broadcast_task.add_done_callback(_log_task_result)
    
    # Start the Compactor task if enabled
    compactor_task = None
    if hasattr(ai_core.config, 'COMPACTOR_ENABLED') and ai_core.config.COMPACTOR_ENABLED:
        # Get the required components
        kv_mirror_manager = ai_core.kv_mirror_manager
        diffuser_model = ai_core.diffuser_model
        pending_diffs_queue = ai_components.get('pending_diffs_queue')
        compactor_request_queue = ai_components.get('compactor_request_queue')
        shutdown_event = ai_components.get('shutdown_event')
        
        if kv_mirror_manager and diffuser_model and pending_diffs_queue and compactor_request_queue and shutdown_event:
            print("Starting Compactor task...")
            compactor_task = asyncio.create_task(
                compactor_task(
                    kv_mirror_manager=kv_mirror_manager,
                    diffuser_model=diffuser_model,
                    pending_diffs_queue=pending_diffs_queue,
                    compactor_request_queue=compactor_request_queue,
                    shutdown_event=shutdown_event
                )
            )
            compactor_task.add_done_callback(_log_task_result)
            print("Compactor task started")
        else:
            print("Cannot start Compactor: missing required components")
    
    # Start inference task with explicit keyword arguments
    try:
        inference_task = asyncio.create_task(
            ai_core.run_continuous_inference(
                # Pass required components by keyword
                model=ai_components["model"],
                processor=ai_components["processor"],
                controller=ai_components["controller"],
                # Pass optional components explicitly by keyword
                initial_prompt_content=ai_components["initial_prompt_content"],
                output_queue=ai_components["output_queue"],
                shutdown_event=ai_components["shutdown_event"],
                sliding_event=ai_components["sliding_event"],
                resume_context_file=ai_components.get("resume_context_file"),
                shared_state=ai_components["shared_state"],
                kv_patcher=ai_components.get("kv_patcher"),
                pending_diffs_queue=ai_components.get("pending_diffs_queue"),
                compactor_request_queue=ai_components.get("compactor_request_queue")
            )
        )
    except Exception:
        print("Failed to launch inference loop:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise  # re-raise so FastAPI still aborts cleanly
    inference_task.add_done_callback(_log_task_result)
    print("AI inference task started")
    
    # Application runs
    yield
    
    # Code to run on shutdown
    print("Server shutdown initiated - cleaning up tasks...")
    
    if ai_components and ai_components["shutdown_event"]:
        # Signal the inference task to stop
        ai_components["shutdown_event"].set()
    
    # Cancel and await all tasks with proper error handling
    for task_name, task in [("Inference", inference_task), ("Broadcast", broadcast_task), ("Compactor", compactor_task)]:
        if task and not task.done():
            try:
                # Cancel the task
                task.cancel()
                # Wait with a timeout
                await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
                print(f"{task_name} task shut down cleanly")
            except asyncio.TimeoutError:
                print(f"{task_name} task didn't shut down within timeout")
            except asyncio.CancelledError:
                print(f"{task_name} task cancelled successfully")
            except Exception as e:
                print(f"Error during {task_name} task shutdown: {e}")
    
    print("Server shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(title="Infinite Scroll AI Stream", lifespan=lifespan)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables to track state
ai_components = None
inference_task = None
broadcast_task = None
compactor_task = None
active_connections = set()

# Create a connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Will be removed on next failed send
                pass


manager = ConnectionManager()


# Model for event injection
class InjectRequest(BaseModel):
    message: str
    source: str = "API"


# FastAPI startup event
# Task exception handler for better error visibility
def _log_task_result(task: asyncio.Task):
    try:
        task.result()  # Re-raises exception if task crashed
    except asyncio.CancelledError:
        pass  # Task cancellation is expected on shutdown
    except Exception as exc:
        import traceback
        import sys
        print(f"--- CRITICAL: Background task crashed! ---", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        print(f"--- End of background task crash report ---", file=sys.stderr)

# Startup and shutdown handling has been moved to the lifespan context manager above


# Broadcaster task that sends tokens from the queue to all connected clients
async def broadcast_tokens():
    MAX_QUEUE_SIZE = 100  # Max number of tokens to buffer before dropping
    
    while True:
        try:
            if ai_components and ai_components["output_queue"] and not ai_components["shutdown_event"].is_set():
                # Check if we have active connections before blocking on queue.get()
                if not manager.active_connections:
                    # No active clients, discard a token if the queue has items
                    # This prevents buffers from building up when no one is listening
                    if not ai_components["output_queue"].empty():
                        token = await ai_components["output_queue"].get()
                        ai_components["output_queue"].task_done()
                        # Log the token discard for better visibility
                        print(f"[Broadcast] Discarded token: No clients connected.", file=sys.stderr)
                    # Short wait before checking again
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if queue is getting too large (backpressure)
                if ai_components["output_queue"].qsize() > MAX_QUEUE_SIZE:
                    print(f"Warning: Token queue size exceeded {MAX_QUEUE_SIZE}, dropping older tokens")
                    # Clear out older tokens to prevent memory buildup
                    while ai_components["output_queue"].qsize() > MAX_QUEUE_SIZE//2:
                        _ = await ai_components["output_queue"].get()
                        ai_components["output_queue"].task_done()
                
                # Get a token from the queue
                token = await ai_components["output_queue"].get()
                
                # Broadcast to all WebSocket connections
                await manager.broadcast(token)
                
                # Mark task as done
                ai_components["output_queue"].task_done()
            else:
                # If not initialized yet, wait a bit
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error in token broadcaster: {e}")
            await asyncio.sleep(1)  # Sleep a bit on error


# WebSocket endpoint for real-time token streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Log connection
    client_info = f"{websocket.client.host}:{websocket.client.port}"
    print(f"WebSocket client connected: {client_info}")
    
    try:
        while True:
            try:
                # Wait for messages from the client
                data = await websocket.receive_json()
            except ValueError as json_err:
                # Handle malformed JSON separately
                print(f"Error parsing JSON from client {client_info}: {json_err}")
                await websocket.send_json({
                    "status": "error", 
                    "message": "Invalid JSON format"
                })
                continue
            except Exception as receive_err:
                # For any other error receiving data, we need to break the loop
                print(f"Error receiving data from client {client_info}: {receive_err}")
                break
            
            try:
                # Parse the JSON message
                cmd = data.get("cmd", "")
                
                # Handle different command types
                if cmd == "ping":
                    # Respond to ping with pong (lightweight heartbeat)
                    await websocket.send_json({"cmd": "pong"})
                    
                elif cmd == "set_sampler":
                    # Extract sampler parameters from the message
                    updates = {}
                    
                    # Check for each parameter and add to updates if present
                    if "temperature" in data:
                        updates["temperature"] = float(data["temperature"])
                    if "top_k" in data:
                        updates["top_k"] = int(data["top_k"])
                    if "top_p" in data:
                        updates["top_p"] = float(data["top_p"])
                    if "repetition_penalty" in data:
                        updates["repetition_penalty"] = float(data["repetition_penalty"])
                    if "force_accept" in data:
                        updates["force_accept"] = bool(data["force_accept"])
                    if "token_bias" in data and isinstance(data["token_bias"], dict):
                        # Convert string keys to integers if they're token IDs
                        token_bias = {}
                        for k, v in data["token_bias"].items():
                            try:
                                token_id = int(k)
                                token_bias[token_id] = float(v)
                            except ValueError:
                                # Skip invalid token IDs
                                pass
                        updates["token_bias"] = token_bias
                    
                    # Update the sampler state
                    for param, value in updates.items():
                        setattr(ai_core.sampler_state, param, value)
                    
                    # Send confirmation to the client
                    result = {
                        "status": "success",
                        "message": "Sampler parameters updated",
                        "updated": list(updates.keys())
                    }
                    await websocket.send_json(result)
                    
                    # Log the change
                    print(f"[WebSocket] Sampler parameters updated: {updates}")
                
                elif "message" in data:
                    # Handle regular message inject (default behavior)
                    message = data.get("message", "")
                    source = data.get("source", "USER")
                    
                    if message.strip():
                        # Format the user message for display
                        formatted_message = f"\n[{source}]: {message}\n"
                        
                        # Broadcast the user message to all connected clients
                        await manager.broadcast(formatted_message)
                        
                        # Send acknowledgment to the sender
                        await websocket.send_text("✓")
                        
                        # Now pass the data to the AI core
                        await ai_components["controller"].inject_event(
                            message, 
                            source=source,
                            shared_state=ai_components["shared_state"]
                        )
            except Exception as cmd_err:
                # Handle errors in the command processing
                print(f"Error processing command from client {client_info}: {cmd_err}")
                try:
                    await websocket.send_json({
                        "status": "error",
                        "message": f"Error processing command: {str(cmd_err)}"
                    })
                except:
                    # If we can't even send the error message, the connection is likely gone
                    break
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# REST API endpoint for injecting events
@app.post("/inject")
async def inject_event(request: InjectRequest):
    if not ai_components or not ai_components["controller"]:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    
    await ai_components["controller"].inject_event(
        request.message, 
        source=request.source,
        shared_state=ai_components["shared_state"]
    )
    
    return {"status": "queued"}


# API endpoint for updating sampling parameters
@app.post("/update_sampler")
async def update_sampler(update: SamplerUpdate):
    """Update the sampling parameters in real-time"""
    if not ai_components:
        raise HTTPException(status_code=503, detail="AI service not initialized")
    
    # Create a dictionary of the updated fields (excluding None values)
    updates = {k: v for k, v in update.dict().items() if v is not None}
    
    if not updates:
        return {"status": "no change", "message": "No parameters provided"}
    
    # Update the sampler state with the new values
    for param, value in updates.items():
        setattr(ai_core.sampler_state, param, value)
    
    # Log the change
    print(f"[API] Sampler parameters updated: {updates}")
    
    # Return the current state after update
    current_state = {
        "temperature": ai_core.sampler_state.temperature,
        "top_k": ai_core.sampler_state.top_k,
        "top_p": ai_core.sampler_state.top_p,
        "repetition_penalty": ai_core.sampler_state.repetition_penalty,
        "force_accept": ai_core.sampler_state.force_accept,
        # Don't return token_bias in the response as it could be large
    }
    
    return {
        "status": "success",
        "message": "Sampler parameters updated",
        "current_state": current_state
    }


# API endpoint for getting top token predictions
@app.get("/top_tokens")
async def get_top_token_predictions():
    """Get the current top token predictions from the model."""
    if not ai_components:
        raise HTTPException(status_code=503, detail="AI service not initialized")
    
    # Get top tokens from the utils module
    from positronic_brain.utils import get_top_tokens
    tokens = await get_top_tokens(count=20)
    
    # Add current bias information to each token
    for token in tokens:
        token_id = token["token_id"]
        if (ai_core.sampler_state.token_bias and 
            token_id in ai_core.sampler_state.token_bias):
            token["bias"] = ai_core.sampler_state.token_bias[token_id]
        else:
            token["bias"] = 0.0
    
    # Prepare the response
    response = {
        "tokens": tokens,
        "current_context": ""  # Could add context here if needed
    }
    
    return response


# API endpoint for updating token bias
@app.post("/update_token_bias")
async def update_token_bias(bias_update: TokenBiasUpdate):
    """Update the bias for a specific token ID."""
    if not ai_components:
        raise HTTPException(status_code=503, detail="AI service not initialized")
    
    # Initialize token_bias dict if it doesn't exist
    if ai_core.sampler_state.token_bias is None:
        ai_core.sampler_state.token_bias = {}
    
    # Update the bias for the specified token
    token_id = bias_update.token_id
    bias_value = bias_update.bias_value
    
    # If bias is very close to zero, remove the token from the dict to save memory
    if abs(bias_value) < 1e-4:
        if token_id in ai_core.sampler_state.token_bias:
            del ai_core.sampler_state.token_bias[token_id]
    else:
        ai_core.sampler_state.token_bias[token_id] = bias_value
    
    # Get the token text for the response
    token_text = ai_components["processor"].tokenizer.decode([token_id])
    
    return {
        "status": "success",
        "message": f"Bias for token '{token_text}' (ID: {token_id}) updated to {bias_value}",
        "token_id": token_id,
        "token": token_text,
        "bias_value": bias_value
    }


# API endpoint for biasing a phrase (multiple tokens)
@app.post("/bias_phrase")
async def bias_phrase(phrase_bias: TokenBiasPhrase):
    """Convert a phrase to tokens and apply bias to all."""
    if not ai_components:
        raise HTTPException(status_code=503, detail="AI service not initialized")
    
    # Initialize token_bias dict if it doesn't exist
    if ai_core.sampler_state.token_bias is None:
        ai_core.sampler_state.token_bias = {}
    
    # Tokenize the phrase
    tokenizer = ai_components["processor"].tokenizer
    token_ids = tokenizer.encode(phrase_bias.phrase)
    
    # Remove special tokens if any
    if tokenizer.bos_token_id in token_ids:
        token_ids.remove(tokenizer.bos_token_id)
    if tokenizer.eos_token_id in token_ids:
        token_ids.remove(tokenizer.eos_token_id)
    
    # Apply bias to all tokens
    bias_value = phrase_bias.bias_value
    biased_tokens = []
    
    for token_id in token_ids:
        # If bias is zero-ish, remove token from dict
        if abs(bias_value) < 1e-4:
            if token_id in ai_core.sampler_state.token_bias:
                del ai_core.sampler_state.token_bias[token_id]
        else:
            ai_core.sampler_state.token_bias[token_id] = bias_value
        
        # Add token info to response
        biased_tokens.append({
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "bias_value": bias_value
        })
    
    return {
        "status": "success",
        "message": f"Applied bias {bias_value} to {len(token_ids)} tokens for phrase '{phrase_bias.phrase}'",
        "biased_tokens": biased_tokens
    }


# API endpoint to clear all token biases
@app.post("/clear_token_biases")
async def clear_token_biases():
    """Clear all token biases."""
    if not ai_components:
        raise HTTPException(status_code=503, detail="AI service not initialized")
    
    # Count biases before clearing
    bias_count = len(ai_core.sampler_state.token_bias) if ai_core.sampler_state.token_bias else 0
    
    # Clear the biases
    ai_core.sampler_state.token_bias = {}
    
    return {
        "status": "success",
        "message": f"Cleared {bias_count} token biases"
    }


# API endpoint for retrieving current context information
@app.get("/context_info")
async def get_context_info():
    """Get the current context window information.
    
    Returns both the token map (showing which token occupies each KV cache slot)
    and the chronological context (the actual text content the model is using)
    """
    if not ai_components:
        raise HTTPException(status_code=503, detail="AI service not initialized")
    
    if not ai_components.get("shared_state"):
        raise HTTPException(status_code=503, detail="Shared state not initialized")
    
    # Get processor for decoding
    processor = ai_components["processor"]
    
    # Get both token map and chronological context
    context_info = await ai_core.get_token_map(
        processor=processor,
        shared_state=ai_components["shared_state"]
    )
    
    return context_info


# Serve the static HTML file at the root endpoint
@app.get("/", response_class=HTMLResponse)
async def get_html():
    return FileResponse("static/index.html")


# Main entry point to run the server directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
