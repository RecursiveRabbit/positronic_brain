# ── disable_weights_only.py ────────────────────────────────────────────────────
import torch
import functools
import inspect

# 1) grab the original
_orig_torch_load = torch.load
_sig = inspect.signature(_orig_torch_load)

@functools.wraps(_orig_torch_load)
def patched_torch_load(f, *args, **kwargs):
    # if torch.load actually has a weights_only parameter, force it to False
    if 'weights_only' in _sig.parameters:
        kwargs['weights_only'] = False
    # otherwise, just drop any weights_only in kwargs
    else:
        kwargs.pop('weights_only', None)

    return _orig_torch_load(f, *args, **kwargs)

# 2) override globally
torch.load = patched_torch_load
