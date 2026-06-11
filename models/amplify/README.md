# AMPLIFY Optimized with NVIDIA TransformerEngine

This folder contains source code and tests for an AMPLIFY model that inherits from the transformers `PreTrainedModel`
class and uses TransformerEngine layers. Users do not need to install this package directly, but can load the
model directly from HuggingFace Hub using the standard transformers API. For more information, refer to [Inference Examples](#inference-examples).

## Feature support

The AMPLIFY implementation natively supports the following TransformerEngine-provided optimizations:

| Feature                                 | Support                    |
| --------------------------------------- | -------------------------- |
| **FP8**                                 | 🚧 Under development       |
| **MXFP8**                               | ❌ Not currently supported |
| **Sequence Packing / THD input format** | 🚧 Under development       |
| **FP8 with THD input format**           | 🚧 Under development       |
| **Import from HuggingFace checkpoints** | ✅ Supported               |
| **Export to HuggingFace checkpoints**   | 🚧 Under development       |

Refer to [BioNeMo Recipes](../../recipes/README.md) for more details on how to use these features to accelerate model
training and inference.

## Links to HF checkpoints

Pre-trained AMPLIFY models are available on HuggingFace as part of the NVIDIA
[BioNeMo collection](https://huggingface.co/collections/nvidia/bionemo-686d3faf75aa1edde8c118d9) on the HuggingFace Hub:

**Available Models:**

- [`nvidia/AMPLIFY_120M`](https://huggingface.co/nvidia/AMPLIFY_120M) (120M parameters)
- [`nvidia/AMPLIFY_350M`](https://huggingface.co/nvidia/AMPLIFY_350M) (350M parameters)

## Runtime Requirements

We recommend using the latest [NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
for optimal performance and compatibility. Refer to the provided Dockerfile for details.

## Inference Examples

Quick start example using HuggingFace transformers:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("nvidia/AMPLIFY_120M")
tokenizer = AutoTokenizer.from_pretrained("nvidia/AMPLIFY_120M")

# Example protein sequence
protein_sequence = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
    "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
    "HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

inputs = tokenizer(protein_sequence, return_tensors="pt")
output = model(**inputs)
```

## Recipe Links

Training recipes are available in the `recipes/` directory. AMPLIFY can be trained using the same
recipes as ESM-2, simply by switching the model_tag to reference the AMPLIFY model, such as `nvidia/AMPLIFY_120M`, and
changing the dataset as appropriate.

- **[esm2_native_te](../../recipes/esm2_native_te/)** - Demonstrates training with a simple native PyTorch training
  loop.
- **[esm2_accelerate_te](../../recipes/esm2_accelerate_te/)** - Trains the model using HuggingFace
  [Accelerate](https://huggingface.co/docs/accelerate/index).

## Commands for converting checkpoints

### HF Transformers to TE conversion

Generate converted AMPLIFY checkpoints from existing HuggingFace transformers checkpoints:

```bash
mkdir -p checkpoint_export
docker build -t amplify .
docker run --rm -it --gpus all \
  -v $PWD/checkpoint_export/:/workspace/bionemo/checkpoint_export \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface \
  amplify python export.py
```

### TE to HF Transformers conversion

(Coming soon)

## Developer Guide

### Running tests

To run tests locally, run `recipes_local_test.py` from the repository root with the model directory as an argument.

```bash
./ci/scripts/recipes_local_test.py models/amplify/
```

### Development container

To use the provided devcontainer, use "Dev Containers: Reopen in Container" from the VSCode menu, and choose the
"BioNeMo Recipes Dev Container" option. To run the tests inside the container, first install the model package in
editable mode with `pip install -e .`, then run `pytest -v .` in the model directory.

### Deploying converted checkpoints to HuggingFace Hub

After running the checkpoint conversion steps listed in [Commands for converting checkpoints](#commands-for-converting-checkpoints),
you can deploy the converted checkpoints to the HuggingFace Hub by running the following command:

```bash
huggingface-cli upload nvidia/${MODEL_NAME} $PWD/checkpoint_export/${MODEL_NAME}
```

Or, upload all models at once with:

```bash
for dir in *; do huggingface-cli upload nvidia/$(basename "$dir") "$dir/"; done
```
