import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import asyncio
import sys
import gc
import aioconsole  # For async terminal input

# --- Configuration ---
MODEL_NAME = "moonshotai/Kimi-VL-A3B-Thinking"
TRUST_REMOTE_CODE = True

# --- Device Configuration ---
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = "cpu"
print(f"Using GPU device: {GPU_DEVICE}")
print(f"Using CPU device for potential cache offload: {CPU_DEVICE}")

# --- Option: Control KV Cache Offloading ---
OFFLOAD_KV_CACHE_TO_CPU = False

# --- Sampling Configuration ---
USE_SAMPLING = True
TEMPERATURE = 0.6
TOP_K = 50

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(f"Set tokenizer pad_token to eos_token ({processor.tokenizer.eos_token})")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=TRUST_REMOTE_CODE,
)
model.eval()
print("Model loaded.")

def move_cache_to_device(past_key_values, target_device):
    if past_key_values is None: return None
    new_cache = []
    for layer_past in past_key_values:
        new_layer_past = tuple(
            past_tensor.to(target_device, non_blocking=True) for past_tensor in layer_past
        )
        new_cache.append(new_layer_past)
    return tuple(new_cache)

def truncate_kv_cache(past_key_values, max_len):
    if past_key_values is None: return None
    new_cache = []
    for layer_past in past_key_values:
        current_cache_len = layer_past[0].shape[2]
        slice_len = min(max_len, current_cache_len)
        new_layer_past = tuple(
            past_tensor[:, :, -slice_len:, :] for past_tensor in layer_past
        )
        new_cache.append(new_layer_past)
    return tuple(new_cache)

class SimpleContextController:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pending_events = []
        self.lock = asyncio.Lock() # For safe asynchronous access

    async def inject_event(self, text_event: str, source: str = "USER"):
        """Injects a text event into the queue."""
        async with self.lock:
            # Add formatting to distinguish injected text
            # Example: [USER]: Hello there!
            # Example: [SYSTEM]: Time tick update.
            formatted_event = f"\n[{source.upper()}]: {text_event}\n" # Add newlines for separation
            print(f"\n[Controller Queueing: '{formatted_event.strip()}']", file=sys.stderr)
            self.pending_events.append(formatted_event)

    async def apply_pending_updates(self, current_input_ids, current_attention_mask, device):
        """
        Checks for pending events, tokenizes them, and concatenates them
        to the input_ids and attention_mask.
        Returns (updated_input_ids, updated_attention_mask, update_applied_flag)
        """
        updates_to_apply_text = []
        async with self.lock:
            if not self.pending_events:
                return current_input_ids, current_attention_mask, False # No updates

            # Get all pending events and clear queue
            updates_to_apply_text = self.pending_events[:]
            self.pending_events.clear()

        if not updates_to_apply_text: # Should not happen due to initial check, but safe
             return current_input_ids, current_attention_mask, False

        # --- Process and Tokenize Events ---
        full_update_text = "".join(updates_to_apply_text)
        print(f"\n[Controller Applying: '{full_update_text.strip()}']", file=sys.stderr)

        # Tokenize the combined update text
        # Important: add_special_tokens=False prevents adding BOS/EOS tokens mid-stream
        update_tokens = self.tokenizer(
            full_update_text,
            return_tensors="pt",
            add_special_tokens=False # Crucial! Don't add BOS/EOS here
        ).input_ids.to(device) # Move to the correct device

        # Create attention mask for the new tokens (all 1s)
        update_attention_mask = torch.ones_like(update_tokens).to(device)

        # --- Concatenate with existing context ---
        if update_tokens.shape[1] > 0: # Only concatenate if tokens were produced
            updated_input_ids = torch.cat([current_input_ids.to(device), update_tokens], dim=1)
            updated_attention_mask = torch.cat([current_attention_mask.to(device), update_attention_mask], dim=1)
            print(f"[Controller] Context updated. New sequence length: {updated_input_ids.shape[1]}", file=sys.stderr)
            return updated_input_ids, updated_attention_mask, True
        else:
            # No tokens were generated from the update text (e.g., empty string)
            print("[Controller] Warning: Update text produced no tokens.", file=sys.stderr)
            return current_input_ids, current_attention_mask, False

async def run_continuous_inference(model, processor, controller, initial_prompt_content):
    try:
        conf = model.config
        max_length = getattr(conf, 'seq_length', None) or getattr(conf, 'max_position_embeddings', 131072)
        print(f"Model configured max sequence length: {max_length}")
    except Exception:
        print("Warning: Could not determine max length accurately. Using 128k.", file=sys.stderr)
        max_length = 131072
    window_size = min(max_length - 512, 100000)
    print(f"--- Using sliding window target size: {window_size} (truncation infrequent) ---", flush=True)
    if OFFLOAD_KV_CACHE_TO_CPU:
        print("--- KV Cache configured for CPU offload (if needed) ---", flush=True)

    # --- Kimi-style Initial Prompt Handling ---
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": initial_prompt_content}],
        },
    ]
    # Get the formatted text string first
    prompt_text_formatted = processor.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    # Tokenize the formatted string using the processor (handles multimodal if needed)
    prompt_inputs = processor(text=prompt_text_formatted, images=None, return_tensors="pt").to(model.device)

    input_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
    past_key_values = None
    generated_token_count = 0
    print(f"Initial prompt token length: {input_ids.shape[1]}") # Debug print

    print(f"\n--- Starting stream (Model: {MODEL_NAME}) ---", flush=True)

    with torch.no_grad():
        while True:
            # --- Step 0: Apply Context Updates ---
            input_ids, attention_mask, update_applied = await controller.apply_pending_updates(
                input_ids, attention_mask, model.device
            )
            if update_applied:
                past_key_values = None
                print("[Inference] Context update applied, KV cache reset.", file=sys.stderr)

            current_length = input_ids.shape[1]
            if current_length > window_size:
                keep_length = window_size
                input_ids = input_ids[:, -keep_length:]
                attention_mask = attention_mask[:, -keep_length:]
                if past_key_values is not None:
                    past_key_values = truncate_kv_cache(past_key_values, keep_length)
                current_length = input_ids.shape[1]

            if OFFLOAD_KV_CACHE_TO_CPU and past_key_values is not None:
                past_key_values = move_cache_to_device(past_key_values, model.device)

            if past_key_values is None:
                model_input_ids = input_ids
                current_attention_mask = attention_mask
            else:
                model_input_ids = input_ids[:, -1:]
                current_attention_mask = attention_mask

            try:
                outputs = model(
                    input_ids=model_input_ids.to(model.device),
                    attention_mask=current_attention_mask.to(model.device),
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values
                if OFFLOAD_KV_CACHE_TO_CPU and past_key_values is not None:
                    past_key_values = move_cache_to_device(past_key_values, CPU_DEVICE)
                next_token_logits = logits[:, -1, :]
                if USE_SAMPLING:
                    scaled_logits = next_token_logits / TEMPERATURE
                    if TOP_K > 0:
                        v, _ = torch.topk(scaled_logits, TOP_K)
                        scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                token_to_decode = next_token_id[0].item()
                eos_token_ids = [processor.tokenizer.eos_token_id]
                if processor.tokenizer.pad_token_id is not None and processor.tokenizer.pad_token_id != processor.tokenizer.eos_token_id:
                    eos_token_ids.append(processor.tokenizer.pad_token_id)
                if token_to_decode in eos_token_ids:
                    print("\n[EOS]", end="", flush=True)
                new_token_text = processor.tokenizer.decode(token_to_decode, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                print(new_token_text, end="", flush=True)
                generated_token_count += 1
                input_ids = torch.cat([input_ids.to(model.device), next_token_id.to(model.device)], dim=-1)
                attention_mask = torch.cat([attention_mask.to(model.device), torch.ones_like(next_token_id).to(model.device)], dim=-1)
            except RuntimeError as e:
                print(f"\n\n--- FATAL RUNTIME ERROR CAUGHT ---", file=sys.stderr)
                print(f"Error Type: {type(e)}", file=sys.stderr)
                print(f"Error Message: {e}", file=sys.stderr)
                print(f"\n--- State at time of error ---", file=sys.stderr)
                print(f"Generated token count: {generated_token_count}", file=sys.stderr)
                print(f"Current sequence length before model call attempt: {current_length}", file=sys.stderr)
                print(f"Model Input IDs shape fed to model: {model_input_ids.shape}", file=sys.stderr)
                print(f"Attention Mask shape fed to model: {current_attention_mask.shape}", file=sys.stderr)
                if past_key_values:
                    try:
                        kv_len = past_key_values[0][0].shape[2]
                        print(f"KV Cache sequence length: {kv_len}", file=sys.stderr)
                        if kv_len != current_attention_mask.shape[1] - model_input_ids.shape[1] and model_input_ids.shape[1] == 1:
                            print(f"WARNING: KV Cache length ({kv_len}) might be inconsistent with attention mask length ({current_attention_mask.shape[1]}) when using single token input.", file=sys.stderr)
                    except Exception as inspect_e:
                        print(f"Could not inspect KV Cache details: {inspect_e}", file=sys.stderr)
                else:
                    print("KV Cache was None", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"\n\n--- UNEXPECTED ERROR CAUGHT ---", file=sys.stderr)
                print(f"Error Type: {type(e)}", file=sys.stderr)
                print(f"Error Message: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.exit(1)
            await asyncio.sleep(0.005)

async def user_input_handler(controller):
    """Asynchronously waits for user input and injects it."""
    print("\n--- Enter text below to inject into the AI's stream (Ctrl+C in main terminal to exit) ---")
    while True:
        try:
            # Use aioconsole for non-blocking input
            line = await aioconsole.ainput("You: ")
            if line.strip(): # Ignore empty lines
                await controller.inject_event(line.strip(), source="USER")
        except (EOFError, KeyboardInterrupt): # Handle Ctrl+D or if terminal closes
            print("\n--- User input handler stopped ---", file=sys.stderr)
            break
        except Exception as e:
            print(f"\n[Input Error] {e}", file=sys.stderr)
            break # Exit on other errors

async def main():
    # Controller uses the processor's tokenizer
    controller = SimpleContextController(processor.tokenizer)
    initial_prompt_content = "The simulation awakens. A stream of consciousness begins to flow. What thoughts emerge?"

    # Start the inference task
    inference_task = asyncio.create_task(
        run_continuous_inference(model, processor, controller, initial_prompt_content)
    )

    # Start the user input handler task
    input_task = asyncio.create_task(
        user_input_handler(controller)
    )

    # Wait for either task to complete (inference runs forever unless error/interrupted)
    # Or for user input task to exit (e.g., on error)
    done, pending = await asyncio.wait(
        [inference_task, input_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    print("\n--- Main loop detected task completion ---", file=sys.stderr)

    # Cancel any remaining tasks
    for task in pending:
        print(f"Cancelling pending task: {task.get_name()}", file=sys.stderr)
        task.cancel()
        try:
            await task # Allow cancellation to process
        except asyncio.CancelledError:
            pass # Expected

    # Check if inference task exited with an error
    for task in done:
        if task == inference_task and task.exception():
            print(f"Inference task exited with exception: {task.exception()}", file=sys.stderr)

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting continuous AI stream (Model: {MODEL_NAME}). Press Ctrl+C to stop.")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram exiting due to KeyboardInterrupt.", file=sys.stderr)
    finally:
        # Optional: Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleanup finished.", file=sys.stderr)