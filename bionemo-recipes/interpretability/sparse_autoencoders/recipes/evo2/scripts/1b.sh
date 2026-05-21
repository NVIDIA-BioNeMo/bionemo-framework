#!/bin/bash
# Evo2 1B SAE pipeline: convert -> predict_evo2 -> pt_to_parquet -> train.
#
# Assumes:
#   - bionemo-recipes/recipes/evo2_megatron has been built (.ci_build.sh) and
#     its .venv is active, providing predict_evo2 + evo2_convert_savanna_to_mbridge.
#   - The sae workspace package is importable in that same venv.
#   - HF_TOKEN is set if Savanna checkpoint repo is gated.
#
# Override any of these by exporting before invocation.

set -euo pipefail

EVO2_MEGATRON_DIR="${EVO2_MEGATRON_DIR:-/workspace/bionemo-framework/bionemo-recipes/recipes/evo2_megatron}"
RECIPE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${MODEL:-arcinstitute/savanna_evo2_1b_base}"
MODEL_SIZE="${MODEL_SIZE:-evo2_1b_base}"
LAYER="${LAYER:-12}"
# Trained context length. 1B = 8192. Bump for 7B/40B (context-extended).
CHUNK_BP="${CHUNK_BP:-8192}"

FASTA="${FASTA:-/data/interp/evo2/OpenGenome2/fasta/organelles/organelle_sequences.fasta.gz}"
WORK_ROOT="${WORK_ROOT:-/data/interp/evo2}"

CKPT_DIR="${WORK_ROOT}/checkpoints/${MODEL_SIZE}_mbridge"
PREDICT_DIR="${WORK_ROOT}/activations/${MODEL_SIZE}_layer${LAYER}_pt"
PARQUET_DIR="${WORK_ROOT}/activations/${MODEL_SIZE}_layer${LAYER}_parquet"
OUTPUT_DIR="${WORK_ROOT}/sae/${MODEL_SIZE}_layer${LAYER}"

source "${EVO2_MEGATRON_DIR}/.venv/bin/activate"

echo "============================================================"
echo "STEP 0: Chunk FASTA to <=${CHUNK_BP} bp (model trained context)"
echo "============================================================"
# chunk_fasta.py reads .gz directly and writes plain .fasta; no separate gunzip needed.
INPUT_STEM="$(basename "$FASTA")"
INPUT_STEM="${INPUT_STEM%.gz}"
INPUT_STEM="${INPUT_STEM%.fasta}"
CHUNKED_FASTA="${WORK_ROOT}/scratch/${INPUT_STEM}_chunked${CHUNK_BP}.fasta"
if [[ -f "$CHUNKED_FASTA" ]]; then
    echo "Reusing existing chunked FASTA: $CHUNKED_FASTA"
else
    python "${RECIPE_DIR}/scripts/chunk_fasta.py" \
        --input "$FASTA" \
        --output "$CHUNKED_FASTA" \
        --window "$CHUNK_BP"
fi
FASTA="$CHUNKED_FASTA"

echo "============================================================"
echo "STEP 1: Convert Savanna -> MBridge"
echo "============================================================"
if [[ ! -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    evo2_convert_savanna_to_mbridge \
        --savanna-ckpt-path "$MODEL" \
        --mbridge-ckpt-dir "$CKPT_DIR" \
        --model-size "$MODEL_SIZE" \
        --tokenizer-path "${EVO2_MEGATRON_DIR}/tokenizers/nucleotide_fast_tokenizer_512"
else
    echo "Reusing existing checkpoint at $CKPT_DIR"
fi

echo "============================================================"
echo "STEP 2: Extract layer-${LAYER} embeddings (predict_evo2)"
echo "============================================================"
mkdir -p "$PREDICT_DIR"
if compgen -G "${PREDICT_DIR}/predictions__*.pt" > /dev/null; then
    echo "Reusing existing .pt files in $PREDICT_DIR"
else
    predict_evo2 \
        --fasta "$FASTA" \
        --ckpt-dir "$CKPT_DIR" \
        --output-dir "$PREDICT_DIR" \
        --embedding-layer "$LAYER" \
        --micro-batch-size 1 \
        --devices 1 \
        --write-interval batch
fi

echo "============================================================"
echo "STEP 3: Convert .pt -> parquet ActivationStore"
echo "============================================================"
if [[ -f "${PARQUET_DIR}/metadata.json" ]]; then
    echo "Reusing existing parquet shards at $PARQUET_DIR"
else
    python "${RECIPE_DIR}/scripts/pt_to_parquet.py" \
        --predict-dir "$PREDICT_DIR" \
        --output "$PARQUET_DIR" \
        --model-name "$MODEL" \
        --layer "$LAYER"
fi

echo "============================================================"
echo "STEP 4: Train TopK SAE"
echo "============================================================"
python "${RECIPE_DIR}/scripts/train.py" \
    --cache-dir "$PARQUET_DIR" \
    --model-path "$MODEL" \
    --layer "$LAYER" \
    --model-type topk \
    --expansion-factor 8 --top-k 32 \
    --auxk 64 --auxk-coef 0.03125 \
    --init-pre-bias \
    --n-epochs 3 \
    --batch-size 4096 \
    --lr 3e-4 \
    --log-interval 50 \
    --no-wandb \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-dir "${OUTPUT_DIR}/checkpoints" \
    --checkpoint-steps 999999

echo "============================================================"
echo "DONE: SAE checkpoint at ${OUTPUT_DIR}/checkpoints/checkpoint_final.pt"
echo "============================================================"
