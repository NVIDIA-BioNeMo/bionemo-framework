# TransformerEngine-accelerated Mixtral training with a native PyTorch training loop

This folder demonstrates how to train TE-accelerated Mixtral with a native PyTorch training loop using FSDP2 for
distributed training. The recipe mirrors the structure and conventions of `llama3_native_te`, and includes a Lingua-style
configuration for natural-language pre-training on DCLM Baseline 1.0.

## Commands

Single GPU sanity run:

```bash
python train_fsdp2.py --config-name L0_sanity
```

Single GPU Lingua smoke run:

```bash
python train_fsdp2.py --config-name L2_lingua_8x1B num_train_steps=20 checkpoint.ckpt_dir=./checkpoints
```

Cluster or multi-GPU run:

```bash
torchrun --standalone --nproc_per_node=2 train_fsdp2.py --config-name L2_lingua_8x1B
```

## Notes

- The Lingua config uses the `meta-llama/Meta-Llama-3-8B` tokenizer and streams `mlfoundations/dclm-baseline-1.0`.
- `expert_parallel_size` remains `1` in this v1 recipe so it matches the existing Llama3 Lingua recipe structure.
- Use `HF_TOKEN` for Hugging Face access and `WANDB_KEY` for Weights & Biases logging.
