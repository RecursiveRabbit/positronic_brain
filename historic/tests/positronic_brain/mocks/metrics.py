# Mock metrics module for testing
import functools
import inspect  # Need this for iscoroutinefunction

class MockMetric:
    def observe(self, value): pass
    def inc(self, value=1.0): pass
    def set(self, value): pass

_mock_metrics = {}
_histograms = {}
_gauges = {}
_counters = {}

# Add iscoroutinefunction to functools if it's not there
if not hasattr(functools, 'iscoroutinefunction'):
    functools.iscoroutinefunction = inspect.iscoroutinefunction

def timed_histogram(name, description=''):
    """Mock decorator that replaces the actual metrics timing decorator"""
    # Ensure histogram is created if it doesn't exist
    hist = _histograms.setdefault(
        name,
        MockMetric()
    )

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # No actual timing, just call the function
            return fn(*args, **kwargs)

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            # No actual timing, just call the function
            return await fn(*args, **kwargs)

        if functools.iscoroutinefunction(fn):
            return async_wrapper
        else:
            return wrapper
    return decorator

def set_gauge(name, value, description='', labelvalues=None):
    """Mock function for setting gauge metrics"""
    gauge = _gauges.setdefault(name, MockMetric())
    gauge.set(value)

def inc_counter(name, value=1.0, description='', labelvalues=None):
    """Mock function for incrementing counter metrics"""
    counter = _counters.setdefault(name, MockMetric())
    counter.inc(value)
