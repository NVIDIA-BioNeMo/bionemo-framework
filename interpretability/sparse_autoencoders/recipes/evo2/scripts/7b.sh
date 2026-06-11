#!/bin/bash
# Evo2 7B layer-26 SAE recipe: chunk FASTA -> stream-extract activations -> train SAE.
# This reproduces the layer26_7B (normalize_input) run.
#
# Prerequisites (this recipe does NOT download or convert the model):
#   - An Evo2 7B *MBridge* checkpoint directory (CKPT_DIR). Obtain it from NGC, e.g.:
#         ngc registry model download-version "nvidia/clara/evo2:7b_<ver>" --dest "${WORK_ROOT}/checkpoints"
#     (or convert a nemo2 checkpoint to MBridge with the evo2_megatron converter).
#   - recipes/evo2_megatron built (.ci_build.sh) with its .venv active,
#     providing `predict_evo2`.
#   - The `sae` workspace package importable in that same venv.
#
# Override any of these by exporting before invocation.

set -euo pipefail

EVO2_MEGATRON_DIR="${EVO2_MEGATRON_DIR:-/workspace/bionemo-framework/recipes/evo2_megatron}"
RECIPE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LAYER="${LAYER:-26}"
# Context length the activations were extracted at (the model is context-extended; we
# trained the SAE on 8192-bp chunks).
CHUNK_BP="${CHUNK_BP:-8192}"

# An Evo2 7B MBridge checkpoint directory (see prerequisites above).
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to an Evo2 7B MBridge checkpoint directory (see header)}"
FASTA="${FASTA:?Set FASTA to the (prok+euk) input sequences}"
WORK_ROOT="${WORK_ROOT:-/data/interp/evo2}"

NPROC="${NPROC:-8}"            # GPUs / DP ranks
MAX_TOKENS="${MAX_TOKENS:-1000000000}"

PARQUET_DIR="${WORK_ROOT}/activations/evo2_7b_layer${LAYER}_parquet"
OUTPUT_DIR="${WORK_ROOT}/sae/evo2_7b_layer${LAYER}"

source "${EVO2_MEGATRON_DIR}/.venv/bin/activate"

echo "============================================================"
echo "STEP 0: Chunk FASTA to <=${CHUNK_BP} bp"
echo "============================================================"
INPUT_STEM="$(basename "$FASTA")"; INPUT_STEM="${INPUT_STEM%.gz}"; INPUT_STEM="${INPUT_STEM%.fasta}"
CHUNKED_FASTA="${WORK_ROOT}/scratch/${INPUT_STEM}_chunked${CHUNK_BP}.fasta"
if [[ -f "$CHUNKED_FASTA" ]]; then
    echo "Reusing existing chunked FASTA: $CHUNKED_FASTA"
else
    python "${RECIPE_DIR}/scripts/chunk_fasta.py" --input "$FASTA" --output "$CHUNKED_FASTA" --window "$CHUNK_BP"
fi

echo "============================================================"
echo "STEP 1: Stream-extract layer-${LAYER} activations -> parquet ActivationStore (no .pt)"
echo "============================================================"
if [[ -f "${PARQUET_DIR}/metadata.json" ]]; then
    echo "Reusing existing parquet shards at $PARQUET_DIR"
else
    torchrun --nproc_per_node="$NPROC" "${RECIPE_DIR}/scripts/extract.py" \
        --ckpt-dir "$CKPT_DIR" \
        --embedding-layer "$LAYER" \
        --fasta "$CHUNKED_FASTA" \
        --activation-store-dir "$PARQUET_DIR" \
        --max-tokens "$MAX_TOKENS" \
        --micro-batch-size 4 \
        --dtype fp32
fi

echo "============================================================"
echo "STEP 2: Train TopK SAE (layer26_7B normalize_input config)"
echo "============================================================"
# unset a leaked key so ~/.netrc wins; clara-discovery is the wandb entity.
unset WANDB_API_KEY || true
export WANDB_ENTITY="${WANDB_ENTITY:-clara-discovery}"
torchrun --nproc_per_node="$NPROC" "${RECIPE_DIR}/scripts/train.py" \
    --cache-dir "$PARQUET_DIR" \
    --model-path "$CKPT_DIR" \
    --layer "$LAYER" \
    --model-type topk \
    --expansion-factor 16 --top-k 128 \
    --normalize-input \
    --auxk 2048 --auxk-coef 0.03125 \
    --dead-tokens-threshold 10000000 \
    --init-pre-bias \
    --n-epochs 1 \
    --batch-size 1024 \
    --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --warmup-steps 1000 \
    --max-grad-norm 1.0 \
    --mix-shards 10 \
    --dp-size "$NPROC" \
    --log-interval 100 \
    --wandb --wandb-project evo2-sae-v2-diverse --wandb-run-name "layer${LAYER}_7B_normalize_input" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-dir "${OUTPUT_DIR}/checkpoints" \
    --checkpoint-steps 2000

echo "============================================================"
echo "DONE: SAE checkpoint at ${OUTPUT_DIR}/checkpoints/checkpoint_final.pt"
echo "============================================================"
