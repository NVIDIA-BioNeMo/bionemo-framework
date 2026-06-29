# ESM-2 Optimized with NVIDIA TransformerEngine

This folder contains source code and tests for an ESM-2 model that inherits from the transformers `PreTrainedModel`
class and uses TransformerEngine layers. Users don't need to install this package directly, but can load the
model directly from HuggingFace Hub using the standard transformers API. For more information, refer to [Inference Examples](#inference-examples).

## Feature support

The ESM-2 implementation natively supports the following TransformerEngine-provided optimizations:

| Feature                                 | Support                                                                          |
| --------------------------------------- | -------------------------------------------------------------------------------- |
| **FP8**                                 | ✅ Supported on compute capacity 9.0 and above (Hopper+)                         |
| **MXFP8**                               | ✅ Supported on compute capacity 10.0 and 10.3 (Blackwell), 12.0 support pending |
| **Sequence Packing / THD input format** | ✅ Supported                                                                     |
| **FP8 with THD input format**           | ✅ Supported where FP8 is supported                                              |
| **Import from HuggingFace checkpoints** | ✅ Supported                                                                     |
| **Export to HuggingFace checkpoints**   | ✅ Supported                                                                     |

Refer to [BioNemo Recipes](../../recipes/README.md) for more details on how to use these features to accelerate model
training and inference.

## Links to HF checkpoints

Pre-trained ESM-2 models converted from the original Facebook weights are available on HuggingFace as part of the NVIDIA
[BioNeMo collection](https://huggingface.co/collections/nvidia/bionemo-686d3faf75aa1edde8c118d9) on the HuggingFace Hub:

**Available Models:**

- [`nvidia/esm2_t6_8M_UR50D`](https://huggingface.co/nvidia/esm2_t6_8M_UR50D) (8M parameters)
- [`nvidia/esm2_t12_35M_UR50D`](https://huggingface.co/nvidia/esm2_t12_35M_UR50D) (35M parameters)
- [`nvidia/esm2_t30_150M_UR50D`](https://huggingface.co/nvidia/esm2_t30_150M_UR50D) (150M parameters)
- [`nvidia/esm2_t33_650M_UR50D`](https://huggingface.co/nvidia/esm2_t33_650M_UR50D) (650M parameters)
- [`nvidia/esm2_t36_3B_UR50D`](https://huggingface.co/nvidia/esm2_t36_3B_UR50D) (3B parameters)
- [`nvidia/esm2_t48_15B_UR50D`](https://huggingface.co/nvidia/esm2_t48_15B_UR50D) (15B parameters)

## Runtime Requirements

We recommend using the latest [NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
for optimal performance and compatibility. Refer to the provided Dockerfile for details.

## Inference Examples

Quick start example using HuggingFace transformers:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("nvidia/esm2_t6_8M_UR50D")
tokenizer = AutoTokenizer.from_pretrained("nvidia/esm2_t6_8M_UR50D")

gfp_P42212 = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
    "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
    "HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

inputs = tokenizer(gfp_P42212, return_tensors="pt")
output = model(**inputs)
```

## Recipe Links

Training recipes are available in the `recipes/` directory:

- **[esm2_native_te](../../recipes/esm2_native_te/)** - Demonstrates training with a simple native PyTorch training
  loop.
- **[esm2_accelerate_te](../../recipes/esm2_accelerate_te/)** - Trains the model using HuggingFace
  [Accelerate](https://huggingface.co/docs/accelerate/index).
- **[vllm_inference/esm2](../../recipes/vllm_inference/esm2/)** - Demonstrates inference with
  [vLLM](https://github.com/vllm-project/vllm).

## Running with Low Precision (FP8/FP4)

The TE-optimized ESM-2 model supports per-layer quantization via two mechanisms: a **config-level**
`layer_precision` list that declares which layers use which precision, and **constructor-level** recipe
objects (`fp8_recipe`, `fp4_recipe`) that control the quantization behaviour.

### Configuration: `layer_precision`

`NVEsmConfig.layer_precision` is a list of length `num_hidden_layers` where each element is `"fp8"`,
`"fp4"`, or `None` (BF16 fallback). When set, it controls the `te.autocast` context used for each
transformer layer during both initialization and forward pass.

```python
from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM

# All layers in FP8
config = NVEsmConfig.from_pretrained(
    "nvidia/esm2_t6_8M_UR50D",
    layer_precision=["fp8"] * 6,
)
```

If you pass an `fp8_recipe` to the model constructor **without** setting `layer_precision`, it
defaults to `["fp8"] * num_hidden_layers` (all layers FP8). You can also mix precisions, for example
running most layers in FP8 but keeping the first and last layers in BF16:

```python
layer_precision = [None] + ["fp8"] * 4 + [None]
config = NVEsmConfig.from_pretrained(
    "nvidia/esm2_t6_8M_UR50D",
    layer_precision=layer_precision,
)
```

### Constructor arguments: `fp8_recipe` and `fp4_recipe`

The model classes (`NVEsmModel`, `NVEsmForMaskedLM`, `NVEsmForTokenClassification`) accept
`fp8_recipe` and `fp4_recipe` keyword arguments. These are `transformer_engine.common.recipe.Recipe`
objects that configure the quantization algorithm (e.g., delayed scaling, block scaling, MXFP8).

```python
import transformer_engine.common.recipe as te_recipe

from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM

fp8_recipe = te_recipe.DelayedScaling()

config = NVEsmConfig.from_pretrained(
    "nvidia/esm2_t6_8M_UR50D",
    layer_precision=["fp8"] * 6,
)
model = NVEsmForMaskedLM(config, fp8_recipe=fp8_recipe)
```

For FP4 (NVFP4) quantization, pass an `fp4_recipe` instead and set the corresponding layers to
`"fp4"` in `layer_precision`:

```python
fp4_recipe = te_recipe.NVFP4BlockScaling()

config = NVEsmConfig.from_pretrained(
    "nvidia/esm2_t6_8M_UR50D",
    layer_precision=["fp4"] * 6,
)
model = NVEsmForMaskedLM(config, fp4_recipe=fp4_recipe)
```

You can also mix FP8 and FP4 layers by providing both recipes and a mixed `layer_precision` list.

### Quantized model initialization: `use_quantized_model_init`

When `use_quantized_model_init=True` is set in the config, layers are created inside a
`te.quantized_model_init` context. This tells TransformerEngine to initialize weights directly in
the target quantized format, avoiding a separate quantization step after initialization.

```python
config = NVEsmConfig.from_pretrained(
    "nvidia/esm2_t6_8M_UR50D",
    layer_precision=["fp4"] * 6,
    use_quantized_model_init=True,
)
model = NVEsmForMaskedLM(config, fp4_recipe=te_recipe.NVFP4BlockScaling())
```

### Notes

- The `lm_head` (and `dense` projection in `NVEsmLMHead`) always runs in higher precision
  (`te.autocast(enabled=False)`) regardless of `layer_precision`, to avoid numerical instability in
  the output logits.
- FP8 requires compute capability 9.0+ (Hopper). MXFP8 requires compute capability 10.0+
  (Blackwell).
- If an `fp8_recipe` is provided without `layer_precision`, all layers default to FP8. Providing
  both `fp8_recipe` and `fp4_recipe` without `layer_precision` raises a `RuntimeError`.
- An FP4 layer **requires** an `fp4_recipe`; omitting it raises a `RuntimeError`.

## Converting Between Model Formats

This section explains how to convert between Hugging Face Transformers and Transformer Engine (TE) ESM2 model formats.
The process demonstrates bidirectional conversion: from Transformers to TE format for optimized inference, and back to
Hugging Face Transformers format for sharing and deployment. The workflow involves several key steps:

### Converting from HF Transformers to TE

```python
from transformers import AutoModelForMaskedLM

from convert import convert_esm_hf_to_te

hf_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
te_model = convert_esm_hf_to_te(hf_model)
te_model.save_pretrained("/path/to/te_checkpoint")
```

This loads the pre-trained ESM2 model that will serve as our reference for comparison.

### Converting from TE back to HF Transformers

```python
from convert import convert_esm_te_to_hf
from modeling_esm_te import NVEsmForMaskedLM

te_model = NVEsmForMaskedLM.from_pretrained("/path/to/te_checkpoint")
hf_model = convert_esm_te_to_hf(te_model)
hf_model.save_pretrained("/path/to/hf_checkpoint")
```

### Loading and Testing the Exported Model

Load the exported model and perform validation:

```python
from transformers import AutoTokenizer

model_hf_exported = AutoModelForMaskedLM.from_pretrained("/path/to/hf_checkpoint")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
```

### Validating Converted Models

To validate the converted models, refer to the commands in [Inference Examples](#inference-examples) above to load and test both the original and converted
models to ensure loss and logit values are similar. Additionally, refer to the golden value tests in
[test_modeling_esm_te.py](tests/test_modeling_esm_te.py) and [test_export.py](tests/test_export.py).

## Developer Guide

### Running tests

To run tests locally, run `recipes_local_test.py` from the repository root with the model directory as an argument.

```bash
./ci/scripts/recipes_local_test.py models/esm2/
```

### Development container

To use the provided devcontainer, use "Dev Containers: Reopen in Container" from the VSCode menu, and choose the
"BioNeMo Recipes Dev Container" option. To run the tests inside the container, first install the dependencies with
`pip install -r requirements.txt`, then run `pytest -v .` in the model directory.

### Deploying converted checkpoints to HuggingFace Hub

First, generate converted ESM-2 checkpoints from existing HuggingFace transformers checkpoints:

```bash
mkdir -p checkpoint_export
docker build -t esm2 .
docker run --rm -it --gpus all \
  -v $PWD/checkpoint_export/:/workspace/bionemo/checkpoint_export \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface \
  esm2 python export.py
```

Now deploy the converted checkpoints to the HuggingFace Hub by running the following command for each model:

```bash
huggingface-cli upload nvidia/${MODEL_NAME} $PWD/checkpoint_export/${MODEL_NAME}
```

You can also upload all models at once with:

```bash
cd checkpoint_export
for dir in */; do hf upload --repo-type model nvidia/$(basename "$dir") "$dir/"; done
```
