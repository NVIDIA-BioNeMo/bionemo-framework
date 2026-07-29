#!/usr/bin/env bash

set -euo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$BENCHMARK_DIR/mixtral_8x7b_8xB200.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-${TMPDIR:-/tmp}/mixtral_automodel_8xB200}"
export HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/mixtral_automodel_hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="$BENCHMARK_DIR${PYTHONPATH:+:$PYTHONPATH}"

DCLM_REPO="codelion/dclm-baseline-1B"
DCLM_REVISION="2b7b056aae2fde089e234563fb32c678caea6bca"
MODEL_REPO="${MODEL_REPO:-mistralai/Mixtral-8x7B-v0.1}"
MODEL_REVISION="${MODEL_REVISION:-fc7ac94680e38d7348cfa806e51218e6273104b0}"
# DP8 is the only memory-safe Mixtral layout in the 26.06 image. TP layouts use
# AutoModel's generic fallback plan, which replicates the experts; they remain
# available as explicit experiments for future images.
LAYOUTS="${LAYOUTS:-dp8tp1}"
PRECISIONS="${PRECISIONS:-bf16 mxfp8}"
MXFP8_DENSE="${MXFP8_DENSE:-true}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-60}"
WARMUP_STEPS="${WARMUP_STEPS:-30}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "${DATA_FILE:-}" ]]; then
    DATA_FILE="$(
        DCLM_REPO="$DCLM_REPO" DCLM_REVISION="$DCLM_REVISION" python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


paths = [
    hf_hub_download(
        repo_id=os.environ["DCLM_REPO"],
        filename=f"data/train-{index:05d}-of-00009.parquet",
        repo_type="dataset",
        revision=os.environ["DCLM_REVISION"],
    )
    for index in range(9)
]
print(Path(paths[0]).parent / "*.parquet")
PY
    )"
fi
export DCLM_DATA_FILE="$DATA_FILE"

if [[ -z "${MODEL_PATH:-}" ]]; then
    MODEL_PATH="$(
        MODEL_REPO="$MODEL_REPO" MODEL_REVISION="$MODEL_REVISION" python - <<'PY'
import os

from huggingface_hub import snapshot_download


print(
    snapshot_download(
        repo_id=os.environ["MODEL_REPO"],
        revision=os.environ["MODEL_REVISION"],
        allow_patterns=[
            "*.json",
            "*.model",
            "*.safetensors",
            "tokenizer*",
            "special_tokens_map.json",
        ],
    )
)
PY
    )"
fi
export MIXTRAL_AUTOMODEL_MODEL="$MODEL_PATH"

# All network activity is complete before measured processes start.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

for layout in $LAYOUTS; do
    if [[ "$layout" =~ ^dp([1248])tp([1248])$ ]]; then
        dp_size="${BASH_REMATCH[1]}"
        tp_size="${BASH_REMATCH[2]}"
    else
        echo "Invalid layout '$layout'; expected dp8tp1, dp4tp2, dp2tp4, or dp1tp8" >&2
        exit 2
    fi
    if ((dp_size * tp_size != 8)); then
        echo "Invalid 8-GPU layout '$layout'" >&2
        exit 2
    fi

    for precision in $PRECISIONS; do
        case "$precision" in
            bf16)
                experts_implementation=grouped_mm
                mxfp8_dense=false
                peak_tflops=2250
                ;;
            mxfp8)
                experts_implementation=mxfp8_grouped_mm
                mxfp8_dense="$MXFP8_DENSE"
                peak_tflops=4500
                ;;
            *)
                echo "Invalid precision '$precision'; expected bf16 or mxfp8" >&2
                exit 2
                ;;
        esac

        export BENCHMARK_JSON_OUTPUT="$OUTPUT_DIR/mixtral_8x7b_dp${dp_size}_tp${tp_size}_${precision}.json"
        echo "Running NeMo AutoModel Mixtral-8x7B with DP=$dp_size TP=$tp_size precision=$precision"
        rm -f "$BENCHMARK_JSON_OUTPUT"
        set +e
        torchrun --standalone --nproc-per-node=8 -m nemo_automodel.cli.app "$CONFIG" \
            --model.pretrained_model_name_or_path="$MIXTRAL_AUTOMODEL_MODEL" \
            --model.experts_implementation="$experts_implementation" \
            --model.mxfp8_dense="$mxfp8_dense" \
            --distributed.dp_size="$dp_size" \
            --distributed.tp_size="$tp_size" \
            --step_scheduler.max_steps="$NUM_TRAIN_STEPS" \
            --benchmark.warmup_steps="$WARMUP_STEPS" \
            --benchmark.peak_tflops="$peak_tflops" \
            2>&1 | tee "$OUTPUT_DIR/dp${dp_size}_tp${tp_size}_${precision}.log"
        rc="${PIPESTATUS[0]}"
        set -e
        if ((rc != 0)); then
            echo "Layout DP=$dp_size TP=$tp_size precision=$precision failed with rc=$rc" >&2
        fi
    done
done

shopt -s nullglob
json_results=("$OUTPUT_DIR"/mixtral_8x7b_dp*_tp*_{bf16,mxfp8}.json)
if ((${#json_results[@]} == 0)); then
    echo "No benchmark layout completed successfully" >&2
    exit 1
fi
python "$BENCHMARK_DIR/summarize.py" "${json_results[@]}" | tee "$OUTPUT_DIR/mixtral_8x7b_8xB200.csv"
