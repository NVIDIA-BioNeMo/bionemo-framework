# BioNeMo Recipes Claude Plugin

A Claude Code plugin for converting HuggingFace models to NVIDIA TransformerEngine,
adding FP8/FP4 quantization support, writing golden value tests, and setting up
FSDP distributed training. All skills use real BioNeMo Recipes as reference implementations.

## Installation

```bash
claude --add-dir /path/to/bionemo-recipes/claude-plugin
```

## Available Skills

| Skill                  | Description                                                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `/te-convert-model`    | Convert a HuggingFace `PreTrainedModel` to use TransformerEngine layers with bidirectional weight conversion (HF \<-> TE). |
| `/add-fp8-support`     | Add FP8 or FP4 quantized training support to an existing TransformerEngine model.                                          |
| `/write-golden-tests`  | Generate golden value tests that verify a TE model produces identical outputs to the original HF reference model.          |
| `/setup-fsdp-training` | Scaffold a complete FSDP training recipe with Hydra configs, distributed launcher, and Docker environment.                 |
| `/export-to-hf-hub`    | Create an export script that bundles model weights, tokenizer, and config for publishing to the Hugging Face Hub.          |

## Usage Examples

### Convert a HuggingFace model to TransformerEngine

```
/te-convert-model facebook/esm2_t33_650M_UR50D
```

Generates a TE-backed `PreTrainedModel` class with `convert_hf_to_te()` and
`convert_te_to_hf()` functions, following the pattern in `bionemo-recipes/models/`.

### Add FP8 quantized training

```
/add-fp8-support --precision fp8
```

Adds FP8 recipe configuration, `DelayedScaling` setup, and the `fp8_autocast`
context manager to your training loop.

### Write golden value tests

```
/write-golden-tests --model esm2 --reference facebook/esm2_t33_650M_UR50D
```

Creates pytest tests that load both the HF reference and TE model, run a forward
pass with fixed inputs, and assert outputs match within tolerance.

### Set up FSDP distributed training

```
/setup-fsdp-training --model esm2 --framework native_te
```

Scaffolds a self-contained recipe directory with a Dockerfile, training script,
Hydra configs, and a sample data loader.

### Export model to Hugging Face Hub

```
/export-to-hf-hub --model esm2
```

Generates an `export.py` script that packages weights, config, and tokenizer
files for upload to Hugging Face Hub.

## Links

- [BioNeMo Framework](https://github.com/NVIDIA/bionemo-framework)
- [BioNeMo Recipes README](../README.md)
