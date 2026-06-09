#!/bin/bash
# Full Nemotron-3-Nano SAE workflow: extract -> train -> eval.
#
# The 30B model does not fit on one GPU, so every step runs as a SINGLE process
# with device_map="auto" (model sharded across all visible GPUs). Do NOT use
# torchrun here. Set CUDA_VISIBLE_DEVICES to choose GPUs.
#
# Edit the variables below, then: bash scripts/nemotron_30b.sh
set -e

LAYER=39
EXPERIMENT=topk_k32_8x                       # 8x baseline; swap for topk_k32_16x / _32x
CACHE=.cache/activations/nemotron_l${LAYER}  # extract once, reuse across SAE configs
OUT=outputs/${EXPERIMENT}
MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16

echo "============================================================"
echo "STEP 1: Extract activations from Nemotron-3-Nano (layer ${LAYER})"
echo "============================================================"
# Idempotent: skips if the cache already exists.
python scripts/extract.py \
    activations.model_name="${MODEL}" \
    activations.layer=${LAYER} \
    activations.cache_dir="${CACHE}" \
    data.max_samples=50000

echo ""
echo "============================================================"
echo "STEP 2: Train SAE on cached activations (${EXPERIMENT})"
echo "============================================================"
python scripts/train.py +experiments=${EXPERIMENT} \
    activations.model_name="${MODEL}" \
    activations.layer=${LAYER} \
    activations.cache_dir="${CACHE}" \
    checkpoint.dir="${OUT}/checkpoints" \
    output_dir="${OUT}"

echo ""
echo "============================================================"
echo "STEP 3: Evaluate SAE (reconstruction + loss recovered)"
echo "============================================================"
python scripts/eval.py +experiments=${EXPERIMENT} \
    activations.model_name="${MODEL}" \
    activations.layer=${LAYER} \
    checkpoint.dir="${OUT}/checkpoints" \
    output_dir="${OUT}"

echo ""
echo "============================================================"
echo "DONE -> ${OUT}"
echo "============================================================"
