#!/bin/bash
# Tensor-parallel training of a ~1M-latent SAE across 8 H100s (latent-sharded).
#
# TP shards the SAE latents across all 8 GPUs, so the model is trained FROM A CACHE
# (extract once first -- the 30B model can't co-reside with 8-way SAE shards).
# expansion_factor 384 * 2688 = 1,032,192 latents (129,024 per rank).
set -e

MODEL=/data/jwilber/checkpoints/Nemotron-3-Nano-30B-A3B-Base-BF16
CACHE=.cache/activations/nemotron_l39
OUT=outputs/tp_1m

# 1) Extract activations once (uses all GPUs via device_map="auto").
python3 scripts/extract.py \
    activations.model_name="${MODEL}" \
    activations.cache_dir="${CACHE}" \
    activations.layer=39 \
    data.max_samples=200000

# 2) Tensor-parallel SAE training across 8 GPUs (latent-sharded, Triton decoder).
torchrun --nproc_per_node=8 scripts/train.py \
    activations.model_name="${MODEL}" \
    activations.cache_dir="${CACHE}" \
    activations.layer=39 \
    parallel.tp_size=8 \
    model.expansion_factor=384 \
    model.top_k=32 \
    model.normalize_input=true \
    model.decoder_impl=triton \
    model.auxk=512 \
    model.dead_tokens_threshold=10000000 \
    training.lr=3e-4 \
    training.max_steps=250000 \
    training.batch_size=4096 \
    checkpoint.dir="${OUT}/checkpoints" \
    wandb.enabled=false
