# Qwen2.5 / Qwen3 Optimized with NVIDIA TransformerEngine

This folder contains source code and tests for Qwen2.5 and Qwen3 style models that inherit from the transformers
`PreTrainedModel` class and use TransformerEngine layers. Users can convert existing Qwen checkpoints from HuggingFace
using the provided conversion utilities.

## Feature support

The Qwen implementations natively support the following TransformerEngine-provided optimizations:

| Feature                                 | Support                                                                       |
| --------------------------------------- | ----------------------------------------------------------------------------- |
| **FP8**                                 | Supported on compute capacity 9.0 and above (Hopper+)                         |
| **MXFP8**                               | Supported on compute capacity 10.0 and 10.3 (Blackwell), 12.0 support pending |
| **Sequence Packing / THD input format** | Supported                                                                     |
| **FP8 with THD input format**           | Supported where FP8 is supported                                              |
| **Import from HuggingFace checkpoints** | Supported                                                                     |
| **Export to HuggingFace checkpoints**   | Supported                                                                     |
| **KV-cache inference**                  | Supported (including beam search)                                             |

## Inference Examples

### Quick start: convert and run (Qwen3)

> **Note:** The snippets below use bare imports (e.g., `from convert_qwen3 import ...`). Run them from the
> `models/qwen` directory, or install dependencies first with `pip install -r requirements.txt`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert_qwen3 import convert_qwen3_hf_to_te

# Load the original HuggingFace Qwen3 model
model_hf = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16
)

# Convert to TransformerEngine
model_te = convert_qwen3_hf_to_te(model_hf)
model_te.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("The quick brown fox", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model_te.generate(**inputs, max_new_tokens=16, use_cache=False)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### Quick start: convert and run (Qwen2.5)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert_qwen2 import convert_qwen2_hf_to_te

# Load the original HuggingFace Qwen2.5 model
model_hf = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16
)

# Convert to TransformerEngine
model_te = convert_qwen2_hf_to_te(model_hf)
model_te.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("The quick brown fox", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model_te.generate(**inputs, max_new_tokens=16, use_cache=False)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Running with Low Precision (FP8/FP4)

The TE-optimized Qwen models support per-layer quantization via two mechanisms: a **config-level**
`layer_precision` list that declares which layers use which precision, and **constructor-level** recipe
objects (`fp8_recipe`, `fp4_recipe`) that control the quantization behaviour.

### Configuration: `layer_precision`

`NVQwen2Config.layer_precision` (and `NVQwen3Config.layer_precision`) is a list of length
`num_hidden_layers` where each element is `"fp8"`, `"fp4"`, or `None` (BF16 fallback). When set, it
controls the `te.autocast` context used for each transformer layer during both initialization and
forward pass.

```python
from modeling_qwen3_te import NVQwen3Config, NVQwen3ForCausalLM

# All layers in FP8
config = NVQwen3Config.from_pretrained(
    "Qwen/Qwen3-0.6B",
    layer_precision=["fp8"] * 28,
)
```

If you pass an `fp8_recipe` to the model constructor **without** setting `layer_precision`, it
defaults to `["fp8"] * num_hidden_layers` (all layers FP8). You can also mix precisions, for example
running most layers in FP8 but keeping the first and last layers in BF16:

```python
layer_precision = [None] + ["fp8"] * 26 + [None]
config = NVQwen3Config.from_pretrained(
    "Qwen/Qwen3-0.6B",
    layer_precision=layer_precision,
)
```

### Constructor arguments: `fp8_recipe` and `fp4_recipe`

The model classes (`NVQwen2Model`, `NVQwen2ForCausalLM`, `NVQwen3Model`, `NVQwen3ForCausalLM`)
accept `fp8_recipe` and `fp4_recipe` keyword arguments. These are
`transformer_engine.common.recipe.Recipe` objects that configure the quantization algorithm (e.g.,
delayed scaling, block scaling, MXFP8).

```python
import transformer_engine.common.recipe as te_recipe

from modeling_qwen3_te import NVQwen3Config, NVQwen3ForCausalLM

fp8_recipe = te_recipe.DelayedScaling()

config = NVQwen3Config.from_pretrained(
    "Qwen/Qwen3-0.6B",
    layer_precision=["fp8"] * 28,
)
model = NVQwen3ForCausalLM(config, fp8_recipe=fp8_recipe)
```

For FP4 (NVFP4) quantization, pass an `fp4_recipe` instead and set the corresponding layers to
`"fp4"` in `layer_precision`:

```python
fp4_recipe = te_recipe.NVFP4BlockScaling()

config = NVQwen3Config.from_pretrained(
    "Qwen/Qwen3-0.6B",
    layer_precision=["fp4"] * 28,
)
model = NVQwen3ForCausalLM(config, fp4_recipe=fp4_recipe)
```

You can also mix FP8 and FP4 layers by providing both recipes and a mixed `layer_precision` list.

The same pattern applies to Qwen2.5 models using `NVQwen2Config` and `NVQwen2ForCausalLM`.

### Quantized model initialization: `use_quantized_model_init`

When `use_quantized_model_init=True` is set in the config, layers are created inside a
`te.quantized_model_init` context. This tells TransformerEngine to initialize weights directly in
the target quantized format, avoiding a separate quantization step after initialization.

```python
config = NVQwen3Config.from_pretrained(
    "Qwen/Qwen3-0.6B",
    layer_precision=["fp4"] * 28,
    use_quantized_model_init=True,
)
model = NVQwen3ForCausalLM(config, fp4_recipe=te_recipe.NVFP4BlockScaling())
```

### Notes

- The `lm_head` always runs in higher precision (`te.autocast(enabled=False)`) regardless of
  `layer_precision`, to avoid numerical instability in the output logits.
- FP8 requires compute capability 9.0+ (Hopper). MXFP8 requires compute capability 10.0+
  (Blackwell).
- If an `fp8_recipe` is provided without `layer_precision`, all layers default to FP8. Providing
  both `fp8_recipe` and `fp4_recipe` without `layer_precision` raises a `RuntimeError`.
- An FP4 layer **requires** an `fp4_recipe`; omitting it raises a `RuntimeError`.

## Converting Between Model Formats

This section explains how to convert between Hugging Face Transformers and Transformer Engine (TE) Qwen model formats.
The process demonstrates bidirectional conversion: from Transformers to TE format for optimized training and inference,
and back to Hugging Face Transformers format for sharing and deployment.

### Converting from HF Transformers to TE (Qwen3)

```python
from transformers import AutoModelForCausalLM

from convert_qwen3 import convert_qwen3_hf_to_te

model_hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model_te = convert_qwen3_hf_to_te(model_hf)
model_te.save_pretrained("/path/to/te_checkpoint")
```

### Converting from TE back to HF Transformers (Qwen3)

```python
from convert_qwen3 import convert_qwen3_te_to_hf
from modeling_qwen3_te import NVQwen3ForCausalLM

model_te = NVQwen3ForCausalLM.from_pretrained("/path/to/te_checkpoint")
model_hf = convert_qwen3_te_to_hf(model_te)
model_hf.save_pretrained("/path/to/hf_checkpoint")
```

### Converting from HF Transformers to TE (Qwen2.5)

```python
from transformers import AutoModelForCausalLM

from convert_qwen2 import convert_qwen2_hf_to_te

model_hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model_te = convert_qwen2_hf_to_te(model_hf)
model_te.save_pretrained("/path/to/te_checkpoint")
```

### Converting from TE back to HF Transformers (Qwen2.5)

```python
from convert_qwen2 import convert_qwen2_te_to_hf
from modeling_qwen2_te import NVQwen2ForCausalLM

model_te = NVQwen2ForCausalLM.from_pretrained("/path/to/te_checkpoint")
model_hf = convert_qwen2_te_to_hf(model_te)
model_hf.save_pretrained("/path/to/hf_checkpoint")
```

Once converted back to HF format, the model can be loaded by any library that supports Qwen, such as
[vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang).

### Validating Converted Models

To validate the converted models, refer to the commands in [Inference Examples](#inference-examples) above to load and
test both the original and converted models to ensure loss and logit values are similar. Additionally, refer to the
golden value tests in [test_modeling_qwen2_te.py](tests/test_modeling_qwen2_te.py) and
[test_modeling_qwen3_te.py](tests/test_modeling_qwen3_te.py).

## Developer Guide

### Running tests

To run tests locally, run `recipes_local_test.py` from the repository root with the model directory as an argument.

```bash
./ci/scripts/recipes_local_test.py models/qwen/
```

### Exporting to Hugging Face Hub

The model directory includes an `export.py` script that bundles all files needed for Hugging Face Hub distribution. To
create the export bundle, run from the model directory:

```bash
python export.py
```

Before publishing, validate the export by running the local test suite via
`ci/scripts/recipes_local_test.py`.

### Development container

To use the provided devcontainer, use "Dev Containers: Reopen in Container" from the VSCode menu, and choose the
"BioNeMo Recipes Dev Container" option. To run the tests inside the container, first install the dependencies with
`pip install -r requirements.txt`, then run `pytest -v .` in the model directory.
