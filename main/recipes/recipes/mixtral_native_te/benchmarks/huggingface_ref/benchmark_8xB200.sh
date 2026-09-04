#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${TMPDIR:-/tmp}/mixtral_hf_ref_8xB200}"
HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/mixtral_hf_ref_cache}"
MODEL_ID="${MODEL_ID:-mistralai/Mixtral-8x7B-v0.1}"
MODEL_REVISION="${MODEL_REVISION:-fc7ac94680e38d7348cfa806e51218e6273104b0}"
DCLM_REPO="${DCLM_REPO:-codelion/dclm-baseline-1B}"
DCLM_REVISION="${DCLM_REVISION:-2b7b056aae2fde089e234563fb32c678caea6bca}"
PRECISIONS="${PRECISIONS:-bf16 mxfp8}"
ATTENTION="${ATTENTION:-sdpa}"
EXPERTS="${EXPERTS:-grouped_mm}"
EP_SIZES="${EP_SIZES:-1 8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-30}"
COMPILE_MODE="${COMPILE_MODE:-default}"
RESHARD_AFTER_FORWARD="${RESHARD_AFTER_FORWARD:-1}"

export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HF_HOME/torchinductor}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

mkdir -p "$OUTPUT_DIR" "$HF_HOME"

readarray -t PREPARED_PATHS < <(
    MODEL_ID="$MODEL_ID" MODEL_REVISION="$MODEL_REVISION" DCLM_REPO="$DCLM_REPO" \
    DCLM_REVISION="$DCLM_REVISION" python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

model_path = snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
)
data_paths = [
    hf_hub_download(
        repo_id=os.environ["DCLM_REPO"],
        filename=f"data/train-{index:05d}-of-00009.parquet",
        repo_type="dataset",
        revision=os.environ["DCLM_REVISION"],
    )
    for index in range(9)
]
print(model_path)
print(Path(data_paths[0]).parent / "*.parquet")
PY
)
MODEL_PATH="${PREPARED_PATHS[0]}"
DATA_FILE="${DATA_FILE:-${PREPARED_PATHS[1]}}"

# All artifacts are local before torchrun, so network and download variance cannot enter timings.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

for ep_size in $EP_SIZES; do
    for precision in $PRECISIONS; do
        echo "Running upstream HF Mixtral: ep_size=$ep_size precision=$precision attention=$ATTENTION experts=$EXPERTS"
        extra_args=()
        if [[ "$RESHARD_AFTER_FORWARD" == "0" ]]; then
            extra_args+=(--no-reshard-after-forward)
        fi
        torchrun --standalone --nproc_per_node=8 "$HERE/train.py" \
            --model "$MODEL_PATH" \
            --tokenizer "$MODEL_PATH" \
            --data-file "$DATA_FILE" \
            --seq-len 4096 \
            --micro-batch-size "$MICRO_BATCH_SIZE" \
            --ep-size "$ep_size" \
            --precision "$precision" \
            --attention "$ATTENTION" \
            --experts "$EXPERTS" \
            --compile-mode "$COMPILE_MODE" \
            --warmup-steps "$WARMUP_STEPS" \
            --benchmark-steps "$BENCHMARK_STEPS" \
            --output-json "$OUTPUT_DIR/ep${ep_size}_${precision}_${ATTENTION}_${EXPERTS}.json" \
            "${extra_args[@]}" \
            2>&1 | tee "$OUTPUT_DIR/ep${ep_size}_${precision}_${ATTENTION}_${EXPERTS}.log"
    done
done
