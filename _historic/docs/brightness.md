# Brightness Mechanics in Halo Weave v0

This document describes the brightness management system for tokens in the Halo Weave context management system.

## Core Concepts

In Halo Weave v0, token "brightness" is a measure of how important or relevant a token is to the current context. It follows a Time-To-Live (TTL) approach where tokens:

1. Start with a source-dependent initial brightness
2. Naturally decay over time
3. Gain brightness when they receive attention
4. Get culled when they are the dimmest tokens in a full context window

## Brightness Update Formula

The brightness update rule works as follows:

```
b_new = max(0, min(BRIGHTNESS_MAX, b_prev - decay + int(attention * gain_coefficient)))
```

Where:
- `b_prev` is the token's current brightness
- `decay` is the fixed amount of brightness decay per generation step
- `attention` is the amount of attention the token received in the current step
- `gain_coefficient` is a multiplier for converting attention to brightness gains
- `BRIGHTNESS_MAX` is the maximum allowable brightness value

## Configuration Parameters

These parameters can be adjusted in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BRIGHTNESS_SEED` | Dictionary | Initial brightness values for tokens based on source |
| `BRIGHTNESS_MAX` | 255.0 | Maximum brightness value |
| `BRIGHTNESS_DECAY_PER_TICK` | 2.0 | Amount brightness decays each generation step |
| `BRIGHTNESS_GAIN_COEFFICIENT` | 10.0 | Multiplier for attention-based brightness gain |
| `ATTENTION_TRACE_INTERVAL` | 50 | Save attention traces every N steps (0 to disable) |

## Token Seeding

Tokens are initialized with different brightness values based on their source:

```python
BRIGHTNESS_SEED = {
    'user': 255.0,     # User tokens start at maximum brightness
    'system': 255.0,   # System tokens start at maximum brightness
    'tool': 255.0,     # Tool tokens start at maximum brightness
    'llm': 200.0,      # Model-generated tokens start at slightly lower brightness
    'default': 200.0   # Default for any unspecified sources
}
```

This ensures that user input and system prompts have higher priority in the context window than model-generated text.

## Context Window Management

The culling mechanism maintains a fixed context window size following these rules:

1. If `len < target`: Cull 0 tokens (context not full yet)
2. If `len == target`: Cull 1 token (lowest brightness)
3. If `len > target`: Cull 2 tokens per step (lowest brightness) until `len == target`

This ensures deterministic behavior and stable context management.

## Telemetry

The brightness engine collects metrics on attention distributions, including:
- Mean attention scores
- Median attention scores
- Maximum and minimum attention scores
- Standard deviation of attention

These metrics can be used to monitor model behavior and tune brightness parameters.

## Attention Trace Saving

When enabled (`ATTENTION_TRACE_INTERVAL > 0`), the system periodically saves attention distribution traces to disk for analysis. This can be helpful for debugging and optimizing the attention-based brightness mechanics.
