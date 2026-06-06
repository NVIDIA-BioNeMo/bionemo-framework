#!/bin/bash
# PoC: multi-GPU producer-consumer streaming (no disk).
#
# 6 GPUs extract activations in parallel, 2 GPUs train the SAE via DDP:
#   torchrun --nproc_per_node=2  -> 2 training ranks (SAE-DDP on cuda:0, cuda:1)
#   streaming.extract_devices=[2..7] -> 6 model replicas, 3 per rank
#     rank 0: SAE cuda:0, extractor replicas on cuda:2,3,4
#     rank 1: SAE cuda:1, extractor replicas on cuda:5,6,7
# Producers run concurrently (PyTorch releases the GIL during CUDA), feeding a
# shared bounded queue per rank; DDP all-reduces SAE grads across the 2 ranks.
#
# This is a short smoke (max_steps=200). Bump data.max_samples / training.max_steps
# for a real run. Loading 6x ~59GB replicas takes a few minutes.
set -e

MODEL=/data/jwilber/checkpoints/Nemotron-3-Nano-30B-A3B-Base-BF16

torchrun --nproc_per_node=2 scripts/train.py \
    +experiments=topk_k32_8x \
    activations.model_name="${MODEL}" \
    streaming.enabled=true \
    parallel.dp_size=2 \
    'streaming.extract_devices=[2,3,4,5,6,7]' \
    data.max_samples=20000 \
    training.max_steps=200 \
    checkpoint.dir=outputs/poc_multigpu/checkpoints \
    checkpoint.steps=null \
    wandb.enabled=false

# Quick wiring check first (2 extractors, ~1 min to load):
#   torchrun --nproc_per_node=2 scripts/train.py +experiments=topk_k32_8x \
#     activations.model_name=$MODEL streaming.enabled=true parallel.dp_size=2 \
#     'streaming.extract_devices=[2,3]' data.max_samples=2000 training.max_steps=20 \
#     checkpoint.dir=outputs/poc_smoke/checkpoints wandb.enabled=false
