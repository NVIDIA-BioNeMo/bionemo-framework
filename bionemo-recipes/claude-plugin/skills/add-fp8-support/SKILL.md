---
name: add-fp8-support
description: >
  Add FP8, MXFP8, or NVFP4 quantization support to a TransformerEngine model.
  Triggers when user asks about FP8, FP4, quantization, mixed precision,
  or low-precision training.
allowed-tools: Read, Grep, Glob, Write, Edit, Bash, Agent
argument-hint: '[fp8|mxfp8|nvfp4]'
---

# Add FP8/FP4 Quantization Support

You are adding quantization support to a TransformerEngine model. Read the reference files first.

## Reference Files

- `reference/quantization.py` — Layer-wise precision assignment
- `reference/fp8_config_example.py` — FP8 recipe setup in training

## Steps

### 1. Add Config Fields

Add these fields to the NV config class:

- `layer_precision: list[str | None] | None = None` — Per-layer precision ("fp8", "fp4", None)
- `use_quantized_model_init: bool = False` — Initialize weights directly in quantized format

Validate in `__init__`:

```python
if layer_precision is not None:
    assert len(layer_precision) == self.num_hidden_layers
    for p in layer_precision:
        assert p in {"fp8", "fp4", None}
```

### 2. Pad Vocabulary Size

FP8 requires tensor dimensions divisible by 16. Pad vocab:

```python
self.padded_vocab_size = padded_vocab_size or self.vocab_size
# Round up to next multiple of 16
if self.padded_vocab_size % 16 != 0:
    self.padded_vocab_size = ((self.padded_vocab_size + 15) // 16) * 16
```

Update embedding and LM head to use `padded_vocab_size`. Truncate logits back to `vocab_size` in forward pass.

### 3. Implement `get_autocast_context()`

This method returns the appropriate TE context manager for each layer:

```python
from contextlib import nullcontext
import transformer_engine.pytorch as te


def get_autocast_context(self, layer_number, init=False, outer=False):
    if self.config.layer_precision is None:
        return nullcontext()

    # Outer context wraps entire encoder for recipe post-processing
    if outer:
        if "fp8" not in self.config.layer_precision:
            return nullcontext()
        return te.autocast(enabled=True, recipe=self._fp8_recipe)

    precision = self.config.layer_precision[layer_number]
    recipe = {"fp8": self._fp8_recipe, "fp4": self._fp4_recipe}.get(precision)

    # During init: use quantized_model_init for weight initialization
    if init and self.config.use_quantized_model_init:
        if precision in ("fp8", "fp4"):
            return te.quantized_model_init(recipe=recipe)
        return nullcontext()

    # During forward: use autocast for precision control
    if precision in ("fp8", "fp4"):
        return te.autocast(enabled=True, recipe=recipe)
    return te.autocast(enabled=False)  # Explicitly disable for BF16 layers
```

### 4. Use Contexts in Model

During layer creation:

```python
for i in range(config.num_hidden_layers):
    with self.get_autocast_context(i, init=True):
        layers.append(te.TransformerLayer(...))
```

During forward pass:

```python
with self.get_autocast_context(None, outer=True):
    for layer_idx, layer in enumerate(self.layers):
        with self.get_autocast_context(layer_idx):
            hidden_states = layer(hidden_states, ...)
```

### 5. Keep LM Head in Higher Precision

```python
with te.autocast(enabled=False):
    logits = self.lm_head(hidden_states)
```

### 6. Set Up FP8 Recipes

In training script:

```python
from transformer_engine.common.recipe import DelayedScaling, Format

fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID)
model = MyTEModel(config, fp8_recipe=fp8_recipe)
```

Available recipes:

- `DelayedScaling` — Classic FP8, computes scaling factors with delay
- `Float8CurrentScaling` — Per-tensor current scaling
- `Float8BlockScaling` — Block-wise scaling (MXFP8)
- `NVFP4BlockScaling` — 4-bit quantization

### 7. Layer-wise Precision Assignment

Use `resolve_layer_precision()` from reference to assign layers:

```python
# In config: fp8_layers=[1,2,3], fp4_layers=[4,5,6] (1-indexed)
# Returns: ["fp8","fp8","fp8","fp4","fp4","fp4"] (0-indexed)
```
