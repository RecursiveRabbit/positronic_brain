"""
Metrics module - simplified version for the refactored codebase.

This provides a minimal implementation of the metrics functionality needed by other modules.
"""

import time
import functools
from typing import Callable, Any, TypeVar, Dict, Optional

F = TypeVar('F', bound=Callable[..., Any])

def timed_histogram(name: str, description: str = '') -> Callable[[F], F]:
    """
    Decorator that provides timing metrics functionality.
    
    In this simplified version, it just logs the function execution time without
    actually recording metrics to Prometheus.
    
    Args:
        name: The name of the metric (used for logging only in this version)
        description: Optional description
        
    Returns:
        A decorator function
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                print(f"[METRICS] {name}: {duration:.4f}s")
        
        # Return the wrapped function
        return wrapper  # type: ignore
    
    return decorator

def set_gauge(name: str, value: float, description: str = '') -> None:
    """Stub for setting a gauge metric."""
    print(f"[METRICS] Set gauge {name} to {value}")
    
def inc_counter(name: str, value: float = 1.0, description: str = '') -> None:
    """Stub for incrementing a counter metric."""
    print(f"[METRICS] Increment counter {name} by {value}")
