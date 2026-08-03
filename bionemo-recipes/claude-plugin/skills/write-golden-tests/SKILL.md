---
name: write-golden-tests
description: >
  Write golden value tests and conversion tests for a TransformerEngine model.
  Triggers when user asks to test TE conversion, write golden tests,
  validate model equivalence, or verify conversion correctness.
allowed-tools: Read, Grep, Glob, Write, Edit, Bash, Agent
argument-hint: '[test-dir-path]'
---

# Write Golden Value and Conversion Tests

You are writing tests that prove a TransformerEngine model produces identical outputs to the original HuggingFace model. Read the reference test files first.

## Reference Files

- `reference/test_esm2_example.py` — Encoder model test pattern
- `reference/test_llama3_example.py` — Decoder model test pattern

## Test Categories

### 1. Golden Value Test (Most Important)

Proves numerical equivalence between HF and TE models:

```python
def test_golden_values(self):
    model_hf = OriginalModel.from_pretrained(model_id, dtype=torch.bfloat16).cuda()
    model_te = convert_hf_to_te(model_hf)
    model_te.to("cuda")

    input_data = self.prepare_test_input()

    with torch.no_grad():
        hf_out = model_hf(**input_data)
        te_out = model_te(**input_data)

    # Loss should be very close
    torch.testing.assert_close(te_out.loss, hf_out.loss, atol=1e-2, rtol=1e-3)
    # Logits may have larger absolute differences but small relative error
    torch.testing.assert_close(te_out.logits, hf_out.logits, atol=2.0, rtol=1e-4)
```

### 2. Roundtrip Conversion Test

Proves HF->TE->HF preserves weights:

```python
def test_roundtrip_conversion(self):
    model_hf_orig = OriginalModel.from_pretrained(model_id)
    model_te = convert_hf_to_te(model_hf_orig)
    model_hf_back = convert_te_to_hf(model_te)

    for (name_orig, param_orig), (name_back, param_back) in zip(
        model_hf_orig.named_parameters(), model_hf_back.named_parameters()
    ):
        torch.testing.assert_close(
            param_orig, param_back, msg=f"Mismatch in {name_orig}"
        )
```

### 3. Forward/Backward Smoke Test

```python
def test_forward_backward(self):
    model_te = create_te_model()
    input_data = self.prepare_test_input()
    output = model_te(**input_data)
    output.loss.backward()
    # Verify gradients exist
    for name, param in model_te.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
```

### 4. FP8 Smoke Test

```python
def test_fp8_forward_backward(self):
    from transformer_engine.common.recipe import DelayedScaling, Format

    config = create_config(layer_precision=["fp8"] * num_layers)
    recipe = DelayedScaling(fp8_format=Format.HYBRID)
    model = MyTEModel(config, fp8_recipe=recipe).cuda()
    input_data = self.prepare_test_input()
    output = model(**input_data)
    output.loss.backward()
```

### 5. Meta Device Init Test

```python
def test_meta_device_init(self):
    with torch.device("meta"):
        model = MyTEModel(config)
    model.init_empty_weights()
    # Verify no meta tensors remain
    for name, param in model.named_parameters():
        assert not param.is_meta, f"{name} still on meta device"
```

## Test Tolerances

- **Loss**: atol=1e-2, rtol=1e-3 (should be very close)
- **Logits**: atol=2.0, rtol=1e-4 (larger absolute due to accumulated numerical differences)
- **Hidden states**: atol=0.1, rtol=0.05
- **FP8 loss**: atol=0.1, rtol=0.05 (FP8 introduces more error)
