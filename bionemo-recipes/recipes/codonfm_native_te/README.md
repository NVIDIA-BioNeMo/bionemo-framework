# TransformerEngine-accelerated CodonFM training with native PyTorch training loop

This folder demonstrates how to train TE-accelerated
[CodonFM](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf) with a native PyTorch
training loop, including sequence packing, FP8/NVFP4 precision with layer-wise control, using fully sharded data
parallel (FSDP2) for distributed training.

CodonFM is a suite of foundation models trained directly on codon sequences to learn contextual codon representations and
enable downstream codon-aware tasks. This recipe uses the "non-exact" TransformerEngine implementation, which employs
TE's standard `TransformerLayer` rather than a custom reproduction of the original research architecture. Despite the
slight architectural difference, this variant converges on par with the original. For the original PyTorch Lightning
based recipe (with both "exact" and "non-exact" TE modes), see
[codonfm_ptl_te](../codonfm_ptl_te/).

## How to use this recipe

This folder contains an independent, minimal training example. It does not depend on any other code in the top-level
bionemo-framework repository.

## Supported Training Features

| Feature                | Status        |
| ---------------------- | ------------- |
| BF16                   | Supported     |
| FP8 (DelayedScaling)   | Supported [1] |
| NVFP4 (BlockScaling)   | Supported [2] |
| Layer-wise precision   | Supported     |
| THD sequence packing   | Supported     |
| FSDP2                  | Supported     |
| Checkpoint save/resume | Supported     |

\[1\]: Requires [compute capability](https://developer.nvidia.com/cuda-gpus) 9.0 and above (Hopper+) <br/>
\[2\]: Requires [compute capability](https://developer.nvidia.com/cuda-gpus) 10.0 and above (Blackwell+) <br/>

## Pre-Trained Models

The HuggingFace-compatible model definition lives in
[`bionemo-recipes/models/codonfm/`](../../models/codonfm/). Pre-trained checkpoints for the "exact" TE architecture are
available on the Hugging Face Hub (see the [codonfm_ptl_te README](../codonfm_ptl_te/README.md#pre-trained-models) for
links). A native_te checkpoint trained with this recipe will be uploaded in the future.

## Performance Benchmarks

Under development. For TE acceleration benchmarks comparing against the original Xformers-based implementation, see the
[codonfm_ptl_te benchmarks](../codonfm_ptl_te/README.md#nvidia-transformerengine-optimization-benchmarks).

## Repository Structure

```
codonfm_native_te/
├── modeling_codonfm_te.py    — HF-compatible CodonFM model with TE layers
├── train_fsdp2.py            — FSDP2 training script
├── dataset.py                — data loading and collation (BSHD + THD)
├── tokenizer.py              — codon tokenizer
├── checkpoint.py             — checkpoint save/load utilities
├── perf_logger.py            — performance and metrics logging
├── quantization.py           — FP8/FP4 quantization utilities
├── scheduler.py              — learning rate scheduler
├── distributed_config.py     — distributed training configuration
├── hydra_config/             — Hydra configuration files
│   ├── defaults.yaml         — default training configuration
│   └── L0_sanity.yaml        — quick sanity check configuration
├── train.parquet             — sample training data
├── requirements.txt          — Python dependencies
└── tests/                    — unit and integration tests
```

## Installing Dependencies

The easiest way to get started is to use the devcontainer provided in the top-level repository. Alternatively, install
dependencies manually in an environment with CUDA support:

```bash
pip install -r requirements.txt
```

## Commands to Launch Training

To run single-process training on one GPU:

```bash
python train_fsdp2.py
```

To run multi-process training locally on 2+ GPUs:

```bash
torchrun --nproc_per_node=2 train_fsdp2.py
```

The default configuration (`L0_sanity.yaml`) runs a quick sanity check with 250 steps using the included sample data.
For real training, create a custom Hydra config or override parameters from the command line.

### Quantized Training (FP8 / NVFP4)

To run training with FP8:

```bash
python train_fsdp2.py fp8_config.enabled=true
```

To train with NVFP4 quantization:

```bash
python train_fsdp2.py fp4_config.enabled=true
```

Additional recipe parameters (e.g., switching to `MXFP8BlockScaling`) can be set via the Hydra configuration.

### Layer-Wise Precision

You can control which transformer layers use FP8 or FP4 by specifying 1-indexed layer numbers via `fp8_layers` and
`fp4_layers`. Layers not assigned to either format will run in BF16.

For example, to run layers 1-3 in FP8, layers 4-6 in FP4, and the rest in BF16:

```bash
python train_fsdp2.py \
  fp8_config.enabled=true \
  fp4_config.enabled=true \
  'fp8_layers=[1,2,3]' \
  'fp4_layers=[4,5,6]'
```

When both `fp8_config` and `fp4_config` are enabled but only one layer list is provided, the other format automatically
claims the remaining layers.

### Sequence Packing (THD input format)

Enable sequence packing with:

```bash
python train_fsdp2.py use_sequence_packing=true
```

### FP8 and Sequence Packing

To combine FP8 training with sequence packing:

```bash
python train_fsdp2.py fp8_config.enabled=true use_sequence_packing=true
```

### Quantization Stats Debugging

To enable quantization statistics logging:

```bash
python train_fsdp2.py \
  quant_stats_config.enabled=true \
  quant_stats_config.quant_log_dir=./logs/quant_stats \
  quant_stats_config.quant_stats_file=./fp8_debugging_stats.yaml \
  fp8_config.enabled=true
```

The config file structure [fp8_debugging_stats.yaml](fp8_debugging_stats.yaml) is explained in the
[NVIDIA Transformer Engine config file documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/debug/2_config_file_structure.html).

## Saving and Loading Checkpoints

To enable checkpoint saving, ensure that `checkpoint.ckpt_dir` is set to a writable directory:

```bash
python train_fsdp2.py \
  checkpoint.ckpt_dir=/path/to/ckpt_dir \
  checkpoint.save_every_n_steps=100
```

To resume from the latest checkpoint:

```bash
python train_fsdp2.py \
  checkpoint.ckpt_dir=/path/to/ckpt_dir \
  checkpoint.resume_from_checkpoint=true
```

A final model suitable for uploading to the Hugging Face Hub can be exported at the end of training by setting
`checkpoint.save_final_model=true`.

## Developer Guide

### Running Tests

To run tests locally inside the devcontainer:

```bash
cd bionemo-recipes/recipes/codonfm_native_te
pytest -v tests/
```

### Hydra Tips

[Hydra](https://hydra.cc/) is used for configuration management. Parameters can be overridden from the command line,
e.g., `python train_fsdp2.py fp8_config.enabled=true`. For verbose logging, use `hydra.verbose=true`.

## License

Refer to the [bionemo-recipes LICENSE](https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main/bionemo-recipes/LICENSE).
