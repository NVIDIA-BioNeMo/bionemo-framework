# Llama-3.1 Optimized with NVIDIA TransformerEngine

This folder contains source code and tests for Llama-3.\* style models that inherit from the transformers
`PreTrainedModel` class and uses TransformerEngine layers. Unlike the ESM-2 model, we do not currently distribute
pre-converted TE checkpoints on HuggingFace Hub. Instead, users can convert existing Llama 3 checkpoints from
HuggingFace using the provided conversion utilities.

## Feature support

The Llama-3 implementation natively supports the following TransformerEngine-provided optimizations:

| Feature                                 | Support                                                                          |
| --------------------------------------- | -------------------------------------------------------------------------------- |
| **FP8**                                 | ✅ Supported on compute capacity 9.0 and above (Hopper+)                         |
| **MXFP8**                               | ✅ Supported on compute capacity 10.0 and 10.3 (Blackwell), 12.0 support pending |
| **Sequence Packing / THD input format** | ✅ Supported                                                                     |
| **FP8 with THD input format**           | ✅ Supported where FP8 is supported                                              |
| **Import from HuggingFace checkpoints** | ✅ Supported                                                                     |
| **Export to HuggingFace checkpoints**   | ✅ Supported                                                                     |
| **KV-cache inference**                  | ✅ Supported (including beam search)                                             |
| **Context Parallelism**                 | ✅ Supported                                                                     |
| **Tensor Parallelism**                  | 🚧 Under development                                                             |

Refer to [BioNeMo Recipes](../../recipes/llama3_native_te/README.md) for more details on how to use these features to accelerate model
training and inference with native PyTorch training loops.

## Inference Examples

### Quick start: convert and run

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert import convert_llama_hf_to_te

# Load the original HuggingFace Llama 3 model
model_hf = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.bfloat16
)

# Convert to TransformerEngine.
model_te = convert_llama_hf_to_te(model_hf)
model_te.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("The quick brown fox", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model_te.generate(**inputs, max_new_tokens=16, use_cache=False)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### Inference with KV-cache

For efficient autoregressive generation, use the TE-provided `InferenceParams` KV-cache:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_engine.pytorch.attention import InferenceParams

from convert import convert_llama_hf_to_te

model_hf = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16
)
model_te = convert_llama_hf_to_te(
    model_hf, attn_input_format="thd", self_attn_mask_type="padding_causal"
)
model_te.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

inputs = tokenizer("The quick brown fox", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Allocate KV-cache
past_key_values = InferenceParams(
    max_batch_size=1,
    max_sequence_length=256,
    num_heads_kv=model_te.config.num_key_value_heads,
    head_dim_k=model_te.config.hidden_size // model_te.config.num_attention_heads,
    dtype=torch.bfloat16,
    qkv_format="thd",
    max_ctx_len=256,
)

for layer_number in range(1, model_te.config.num_hidden_layers + 1):
    past_key_values.allocate_memory(layer_number)

with torch.no_grad():
    output_ids = model_te.generate(
        **inputs,
        max_new_tokens=16,
        use_cache=True,
        past_key_values=past_key_values,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Recipe Links

Training recipes are available in the `recipes/` directory:

- **[llama3_native_te](../../recipes/llama3_native_te/)** - Demonstrates training with a native PyTorch training loop
  using FSDP2, including FP8, sequence packing, and context parallelism.

## Running with Low Precision (FP8/FP4)

The TE-optimized Llama model supports per-layer quantization via two mechanisms: a **config-level**
`layer_precision` list that declares which layers use which precision, and **constructor-level** recipe
objects (`fp8_recipe`, `fp4_recipe`) that control the quantization recipe.

### Configuration: `layer_precision`

`NVLlamaConfig.layer_precision` is a list of length `num_hidden_layers` where each element is `"fp8"`,
`"fp4"`, or `None` (BF16 fallback). When set, it controls the `te.autocast` context used for each
transformer layer during both initialization and forward pass.

```python
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM

# All layers in FP8
config = NVLlamaConfig.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    layer_precision=["fp8"] * 16,
)
```

If you pass an `fp8_recipe` to the model constructor **without** setting `layer_precision`, it
defaults to `["fp8"] * num_hidden_layers` (all layers FP8). You can also mix precisions, for example
running most layers in FP8 but keeping the first and last layers in BF16:

```python
layer_precision = [None] + ["fp8"] * 14 + [None]
config = NVLlamaConfig.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    layer_precision=layer_precision,
)
```

### Constructor arguments: `fp8_recipe` and `fp4_recipe`

The model classes (`NVLlamaModel`, `NVLlamaForCausalLM`) accept `fp8_recipe` and `fp4_recipe`
keyword arguments. These are `transformer_engine.common.recipe.Recipe` objects that configure the
quantization algorithm (e.g., delayed scaling, block scaling, MXFP8).

```python
import torch
import transformer_engine.common.recipe as te_recipe
from transformers import AutoModelForCausalLM

from convert import convert_llama_hf_to_te
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM

fp8_recipe = te_recipe.DelayedScaling()

config = NVLlamaConfig.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    layer_precision=["fp8"] * 16,
)
model = NVLlamaForCausalLM(config, fp8_recipe=fp8_recipe)
```

For FP4 (NVFP4) quantization, pass an `fp4_recipe` instead and set the corresponding layers to
`"fp4"` in `layer_precision`:

```python
fp4_recipe = te_recipe.NVFP4()

config = NVLlamaConfig.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    layer_precision=["fp4"] * 16,
)
model = NVLlamaForCausalLM(config, fp4_recipe=fp4_recipe)
```

You can also mix FP8 and FP4 layers by providing both recipes and a mixed `layer_precision` list.

### Quantized model initialization: `use_quantized_model_init`

When `use_quantized_model_init=True` is set in the config, layers are created inside a
`te.quantized_model_init` context. This tells TransformerEngine to initialize weights directly in
the target quantized format, avoiding a separate quantization step after initialization.

```python
config = NVLlamaConfig.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    layer_precision=["fp4"] * 16,
    use_quantized_model_init=True,
)
model = NVLlamaForCausalLM(config, fp4_recipe=te_recipe.NVFP4())
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

This section explains how to convert between Hugging Face Transformers and Transformer Engine (TE) Llama 3 model
formats. The process demonstrates bidirectional conversion: from Transformers to TE format for optimized training and
inference, and back to Hugging Face Transformers format for sharing and deployment.

### Converting from HF Transformers to TE

```python
from transformers import AutoModelForCausalLM

from convert import convert_llama_hf_to_te

model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model_te = convert_llama_hf_to_te(model_hf)
model_te.save_pretrained("/path/to/te_checkpoint")
```

### Converting from TE back to HF Transformers

```python
from convert import convert_llama_te_to_hf
from modeling_llama_te import NVLlamaForCausalLM

model_te = NVLlamaForCausalLM.from_pretrained("/path/to/te_checkpoint")
model_hf = convert_llama_te_to_hf(model_te)
model_hf.save_pretrained("/path/to/hf_checkpoint")
```

Once converted back to HF format, the model can be loaded by any library that supports Llama 3, such as
[vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang).

### Validating Converted Models

To validate the converted models, refer to the commands in [Inference Examples](#inference-examples) above to load and
test both the original and converted models to ensure loss and logit values are similar. Additionally, refer to the
golden value tests in [test_modeling_llama_te.py](tests/test_modeling_llama_te.py).

## Developer Guide

### Running tests

To run tests locally, run `recipes_local_test.py` from the repository root with the model directory as an argument.

```bash
./ci/scripts/recipes_local_test.py models/llama3/
```

### Development container

To use the provided devcontainer, use "Dev Containers: Reopen in Container" from the VSCode menu, and choose the
"BioNeMo Recipes Dev Container" option. To run the tests inside the container, first install the model package in
editable mode with `pip install -e .`, then run `pytest -v .` in the model directory.
