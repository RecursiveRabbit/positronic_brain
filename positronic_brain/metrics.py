import time
import functools
import inspect

# Add iscoroutinefunction to functools if it's not there (older Python versions or testing environments)
if not hasattr(functools, 'iscoroutinefunction'):
    functools.iscoroutinefunction = inspect.iscoroutinefunction

from prometheus_client import Histogram, Gauge, Counter, start_http_server, REGISTRY
from prometheus_client.exposition import generate_latest

# --- Prometheus Metrics Definitions ---

# Using default buckets for histograms initially, can customize later
_histograms: dict[str, Histogram] = {}
_gauges: dict[str, Gauge] = {}
_counters: dict[str, Counter] = {}

# --- Decorator for Timing Histograms ---
def timed_histogram(name: str, description: str = ''):
    """Decorator to record the duration of a function call in a Prometheus Histogram."""
    # Ensure histogram is created if it doesn't exist
    hist = _histograms.setdefault(
        name,
        Histogram(name, description or f'Duration of {name} in seconds')
    )

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                hist.observe(duration)
        # Handle async functions
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                hist.observe(duration)

        if functools.iscoroutinefunction(fn):
             return async_wrapper
        else:
             return wrapper
    return decorator

# --- Functions to interact with Gauges and Counters ---
def set_gauge(name: str, value: float, description: str = ''):
    """Set the value of a Prometheus Gauge."""
    # Get or create the gauge (using our cached dictionary)
    gauge = _gauges.get(name)
    if gauge is None:
        gauge = _gauges.setdefault(
            name,
            Gauge(name, description or f'Value of {name}')
        )
    # Set the value
    gauge.set(value)

def inc_counter(name: str, value: float = 1.0, description: str = ''):
    """Increment a Prometheus Counter."""
    # Get or create the counter (using our cached dictionary)
    counter = _counters.get(name)
    if counter is None:
        counter = _counters.setdefault(
            name,
            Counter(name, description or f'Count of {name}')
        )
    # Increment the counter
    counter.inc(value)

# --- Metrics Server Initialization ---
def init_metrics_server(port: int = 9100):
    """Starts the Prometheus HTTP server on the specified port."""
    try:
        start_http_server(port)
        print(f"[Metrics] Prometheus metrics server started on port {port}")
    except OSError as e:
        print(f"[Metrics] Warning: Could not start Prometheus server on port {port} (maybe already running?): {e}")
    except Exception as e:
        print(f"[Metrics] Error starting Prometheus server: {e}")

# --- Optional: Function to get metrics as text (for debugging/non-server use) ---
def get_metrics_text() -> str:
    """Generate the current metrics in Prometheus text format."""
    return generate_latest(REGISTRY).decode('utf-8')

# --- Predefined Metrics (Examples - Add specific ones as needed) ---
# Example: Define gauges or counters that might be updated directly elsewhere
# set_gauge("kv_mirror_active_tokens", 0, "Number of active tokens in the KV mirror")
# inc_counter("tokens_processed_total", 0, "Total number of tokens processed")
