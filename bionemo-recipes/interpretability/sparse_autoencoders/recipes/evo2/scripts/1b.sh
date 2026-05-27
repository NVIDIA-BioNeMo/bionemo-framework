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

# Default output paths can be overridden per-run. Set RUN_TAG to suffix all three
# at once (e.g. RUN_TAG=100M_mixed -> ..._pt_100M_mixed), or override each path
# individually if you want them in different locations entirely.
RUN_TAG="${RUN_TAG:-}"
_SUFFIX="${RUN_TAG:+_${RUN_TAG}}"

CKPT_DIR="${CKPT_DIR:-${WORK_ROOT}/checkpoints/${MODEL_SIZE}_mbridge}"
PREDICT_DIR="${PREDICT_DIR:-${WORK_ROOT}/activations/${MODEL_SIZE}_layer${LAYER}_pt${_SUFFIX}}"
PARQUET_DIR="${PARQUET_DIR:-${WORK_ROOT}/activations/${MODEL_SIZE}_layer${LAYER}_parquet${_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/sae/${MODEL_SIZE}_layer${LAYER}${_SUFFIX}}"

source "${EVO2_MEGATRON_DIR}/.venv/bin/activate"

if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
    echo "============================================================"
    echo "TRAIN_ONLY=1 — skipping chunk / convert / predict / shim;"
    echo "expecting an existing parquet at: $PARQUET_DIR"
    echo "============================================================"
    if [[ ! -f "${PARQUET_DIR}/metadata.json" ]]; then
        echo "ERROR: TRAIN_ONLY=1 but no parquet at $PARQUET_DIR"
        exit 1
    fi
    # Jump straight to step 4 by sourcing the rest after step 3.
    _SKIP_STEPS_0_TO_3=1
else
    _SKIP_STEPS_0_TO_3=0
fi

if [[ "$_SKIP_STEPS_0_TO_3" != "1" ]]; then

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
    torchrun --nproc_per_node "${DEVICES:-4}" --no-python \
        predict_evo2 \
        --fasta "$FASTA" \
        --ckpt-dir "$CKPT_DIR" \
        --output-dir "$PREDICT_DIR" \
        --embedding-layer "$LAYER" \
        --micro-batch-size "${MICRO_BATCH:-4}" \
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

fi  # end if _SKIP_STEPS_0_TO_3

echo "============================================================"
echo "STEP 4: Train TopK SAE"
echo "============================================================"
# Wandb is enabled iff WANDB_API_KEY is in the env. WANDB_PROJECT/RUN can be overridden.
WANDB_FLAGS=("--no-wandb")
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_FLAGS=(
        "--wandb"
        "--wandb-project" "${WANDB_PROJECT:-evo2-sae}"
    )
    if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
        WANDB_FLAGS+=("--wandb-run-name" "$WANDB_RUN_NAME")
    fi
fi

torchrun --nproc_per_node "${DEVICES:-4}" "${RECIPE_DIR}/scripts/train.py" \
    --cache-dir "$PARQUET_DIR" \
    --model-path "$MODEL" \
    --layer "$LAYER" \
    --model-type topk \
    --expansion-factor "${EXPANSION_FACTOR:-8}" \
    --top-k "${TOP_K:-32}" \
    --auxk "${AUXK:-512}" \
    --auxk-coef "${AUXK_COEF:-0.03125}" \
    --dead-tokens-threshold "${DEAD_TOKENS_THRESHOLD:-500000}" \
    --init-pre-bias \
    --n-epochs "${N_EPOCHS:-3}" \
    --batch-size 4096 \
    --dp-size "${DEVICES:-4}" \
    --lr 3e-4 \
    --log-interval 50 \
    "${WANDB_FLAGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-dir "${OUTPUT_DIR}/checkpoints" \
    --checkpoint-steps 999999

echo "============================================================"
echo "DONE: SAE checkpoint at ${OUTPUT_DIR}/checkpoints/checkpoint_final.pt"
echo "============================================================"
