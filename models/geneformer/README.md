# Geneformer Implemented with Transformer Engine

This repository contains an optimized implementation of Geneformer using NVIDIA Transformer Engine (TE) layers for improved performance on NVIDIA GPUs.

## Overview

Geneformer is a transformer-based model pre-trained on large-scale single-cell transcriptomic data for learning context-aware gene embeddings. This implementation leverages NVIDIA's Transformer Engine to provide:

- **Accelerated inference** with optimized CUDA kernels
- **Memory efficiency** through fused operations
- **FP8 precision support** for faster computation (on supported hardware)
- **Full compatibility** with original Hugging Face checkpoints

## Available Model Variants

| Model                         | Parameters | Input Size | Vocabulary | Training Data                                |
| ----------------------------- | ---------- | ---------- | ---------- | -------------------------------------------- |
| `Geneformer-V1-10M`           | 10M        | 2048       | ~25K genes | ~30M human single cells                      |
| `Geneformer-V2-104M`          | 104M       | 4096       | ~20K genes | ~104M human single cells                     |
| `Geneformer-V2-316M`          | 316M       | 4096       | ~20K genes | ~104M human single cells                     |
| `Geneformer-V2-104M_CLcancer` | 104M       | 4096       | ~20K genes | ~104M human single cells + cancer cell lines |

## Quick Start

### Converting Models to TE Format

Use the export script to convert Geneformer models to the optimized Transformer Engine format:

**Convert a specific model:**

```bash
python export.py --model Geneformer-V1-10M
```

**Convert all available models:**

```bash
python export.py
```

**Specify a custom output directory:**

```bash
python export.py --model Geneformer-V2-104M --output-path /path/to/output
```

**Using Docker:**

```bash
docker run --rm -it --gpus all \
  -v /path/to/checkpoint_export/:/workspace/checkpoint_export \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface \
  geneformer python export.py --model Geneformer-V2-104M --output-path /workspace/checkpoint_export
```

## Converting Between Model Formats

Geneformer supports bidirectional conversion between Hugging Face Transformers format and the optimized Transformer Engine format.

### Loading a Pre-trained Model

Load any Geneformer model variant from Hugging Face:

```python
from transformers import AutoModelForMaskedLM

# Load the default model (Geneformer-V2-316M)
model = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")

# Or load a specific variant
model = AutoModelForMaskedLM.from_pretrained(
    "ctheodoris/Geneformer", subfolder="Geneformer-V2-104M"
)
```

### Converting from HF Transformers to TE

For more control over the conversion process, you can use the low-level conversion API:

```python
from transformers import AutoModelForMaskedLM
from geneformer.convert import convert_geneformer_hf_to_te

# Load the original HF model
model_hf = AutoModelForMaskedLM.from_pretrained(
    "ctheodoris/Geneformer", subfolder="Geneformer-V2-104M"
)

# Convert to TE format
model_te = convert_geneformer_hf_to_te(model_hf)

# Save the TE model
model_te.save_pretrained("./te_checkpoint")
```

### Converting from TE back to HF Transformers

Convert TE-optimized checkpoints back to standard Hugging Face format:

#### Using the High-Level Export API (Recommended)

```python
from geneformer.export import export_te_checkpoint

# Convert TE checkpoint back to HF format
export_te_checkpoint(
    te_checkpoint_path="./te_checkpoint", output_path="./hf_checkpoint"
)
```

This will:

1. Load the TE checkpoint
2. Unpack fused QKV parameters
3. Convert to standard HF format
4. Save as a standard Hugging Face checkpoint

#### Using the Low-Level Conversion API

```python
from geneformer import BertForMaskedLM
from geneformer.convert import convert_geneformer_te_to_hf

# Load the TE model
model_te = BertForMaskedLM.from_pretrained("./te_checkpoint")

# Convert back to HF format
model_hf = convert_geneformer_te_to_hf(model_te)

# Save as HF checkpoint
model_hf.save_pretrained("./hf_checkpoint")
```

### What Happens During Conversion?

**HF → TE Conversion:**

- QKV weights are packed into fused parameters for efficient attention computation
- Layer structure is adapted to use TE's optimized CUDA kernels
- Configuration is extended with TE-specific settings (dtype, layer config, etc.)

**TE → HF Conversion:**

- Fused QKV parameters are unpacked to separate Q, K, V weights
- Layer structure is converted back to standard BERT format
- TE-specific configuration options are removed

## Installation

### Using pip

```bash
cd models/geneformer
pip install -e .
```

### For Development

```bash
cd models/geneformer
pip install -e .[test]
```

## Testing

### Running Tests with Docker

```bash
docker build -t geneformer .
docker run --rm -it --gpus all geneformer pytest tests/
```

### Running Tests Locally

```bash
cd models/geneformer
pytest tests/
```

Install development dependencies:

```bash
cd models/geneformer
PIP_CONSTRAINT= pip install -e .[test]
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
