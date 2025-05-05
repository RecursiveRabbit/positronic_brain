import asyncio
import torch
import pynvml
from .metrics import timed_histogram, set_gauge

async def vram_monitor_task(
    threshold_percent: float, 
    check_interval: int, 
    shutdown_event: asyncio.Event,
    sliding_event: asyncio.Event  # Added as a parameter instead of using global
):
    """
    Monitors GPU VRAM usage across all available GPUs and sets the sliding_event if any GPU exceeds the threshold.
    Works with multi-GPU setups including those using device_map="auto".
    """
    # Early exit if CUDA is not available
    if not torch.cuda.is_available():
        print("[VRAM Monitor] No CUDA devices available. VRAM monitoring disabled.")
        return
    
    try:
        print(f"[VRAM Monitor] Initializing NVML...")
        pynvml.nvmlInit()
        
        # Get all available GPU devices
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            print("[VRAM Monitor] No NVML devices found. VRAM monitoring disabled.")
            return
            
        # Get handles for all GPUs
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
        
        # Log monitored devices
        gpu_names = [pynvml.nvmlDeviceGetName(handle) for handle in handles]
        print(f"[VRAM Monitor] Monitoring {device_count} GPUs with threshold {threshold_percent}%:")
        for i, name in enumerate(gpu_names):
            print(f"[VRAM Monitor]   GPU {i}: {name}")

        while not shutdown_event.is_set():
            if sliding_event.is_set():
                # Sliding has already been triggered, no need to monitor further
                print("[VRAM Monitor] Sliding already enabled. Stopping monitoring.")
                break

            try:
                # Check all GPUs in the system
                for i, handle in enumerate(handles):
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    usage_percent = (mem_info.used / mem_info.total) * 100
                    
                    # Add metrics for this GPU
                    gpu_label = f"gpu_{i}"  # Create a label for the GPU index
                    set_gauge("vram_usage_percent", usage_percent, {"gpu": gpu_label})
                    set_gauge("vram_used_bytes", mem_info.used, {"gpu": gpu_label})
                    set_gauge("vram_total_bytes", mem_info.total, {"gpu": gpu_label})
                    
                    # Optional debugging output (uncomment if needed)
                    # print(f"[VRAM Monitor] GPU {i} usage: {usage_percent:.1f}%")
                    
                    # If any GPU exceeds the threshold, trigger sliding window
                    if usage_percent > threshold_percent:
                        print(f"[VRAM Monitor] ALERT: GPU {i} ({gpu_names[i]}) usage {usage_percent:.1f}% "
                              f"exceeded threshold {threshold_percent}%. Enabling sliding window.")
                        sliding_event.set()  # Signal the inference loop!
                        break  # Exit the GPU checking loop
                
                # If sliding_event was set by any GPU, exit the monitoring loop
                if sliding_event.is_set():
                    break

            except pynvml.NVMLError as error:
                print(f"[VRAM Monitor] NVML Error: {error}. Retrying...")
                # Avoid busy-looping on persistent errors
                await asyncio.sleep(check_interval * 2)
                continue  # Try again next interval

            # Wait for the next check interval
            await asyncio.sleep(check_interval)

    except pynvml.NVMLError as error:
        print(f"[VRAM Monitor] Failed to initialize NVML or get handle: {error}")
        print("[VRAM Monitor] VRAM monitoring disabled.")
    except Exception as e:
        print(f"[VRAM Monitor] Unexpected error: {e}")
    finally:
        try:
            print("[VRAM Monitor] Shutting down NVML...")
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass  # Ignore shutdown errors if already failed
        print("[VRAM Monitor] Task finished.")
