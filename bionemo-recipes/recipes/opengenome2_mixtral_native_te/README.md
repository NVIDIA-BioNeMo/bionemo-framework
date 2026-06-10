# TransformerEngine-accelerated Mixtral training for OpenGenome2

This folder demonstrates how to train TE-accelerated Mixtral with a native PyTorch training loop for autoregressive DNA
token prediction on the OpenGenome2 metagenome subset. It follows the same recipe conventions as
`opengenome2_llama_native_te` for dataset loading, genomic masking, validation, checkpointing, and W&B logging.

## Supported features

- FSDP2 training
- THD sequence packing
- nucleotide tokenizer packaged with the recipe
- genomic label masking
- FP32 master weights through FSDP mixed precision policy
- validation logging during training

## Not supported in this v1 recipe

- context parallelism
- Llama-specific OG2 initialization features such as Spike-No-More and Megatron scaled residual init

## Commands

Single-GPU sanity run:

```bash
python train_fsdp2.py --config-name L0_sanity
```

Single-GPU bounded OG2 smoke run:

```bash
python train_fsdp2.py --config-name og2_small_thd_moe \
  num_train_steps=20 \
  checkpoint.ckpt_dir=./checkpoints
```

Cluster handoff:

```bash
torchrun --standalone --nproc_per_node=2 train_fsdp2.py --config-name og2_small_thd_moe
```

## Data

Download a bounded OpenGenome2 subset for local runs:

```bash
hf download arcinstitute/opengenome2 --repo-type dataset \
  --include "json/pretraining_or_both_phases/metagenomes/data_metagenomics_train_chunk1.jsonl.gz" \
  --include "json/pretraining_or_both_phases/metagenomes/data_metagenomics_valid_chunk1.jsonl.gz" \
  --local-dir /data/opengenome2
```

Use `WANDB_KEY` for Weights & Biases logging.
