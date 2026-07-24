#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

set -euo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(dirname "$BENCHMARK_DIR")"
export HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/mixtral_native_te_hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

# Pin the nine parquet shards from a 1B-token sample of DCLM so data is finite, local, and identical
# across runs. Downloading happens before torchrun so network I/O is not part of the measured
# intervals, and each rank can consume an independent local shard.
DCLM_REPO="codelion/dclm-baseline-1B"
DCLM_REVISION="2b7b056aae2fde089e234563fb32c678caea6bca"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-60}"
LOGGER_FREQUENCY="${LOGGER_FREQUENCY:-10}"
PRECISIONS="${PRECISIONS:-fp8 bf16}"
PARALLEL_CONFIGS="${PARALLEL_CONFIGS:-1,8 2,4 4,2 8,1}"
export MIXTRAL_TE_PRETRAINED_CHECKPOINT="$HF_HOME/te_checkpoints/mixtral_8x7b_fused_bf16.pt"

GPU_NAME="${GPU_NAME:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)}"
case "$GPU_NAME" in
    *B300*)
        GPU_SHORT_NAME="B300"
        DEFAULT_MAX_SEQ_LENGTH=8192
        DEFAULT_TOKEN_MICRO_BATCH_SIZE=16384
        DEFAULT_FP8_PEAK_PFLOPS=5.0
        DEFAULT_BF16_PEAK_PFLOPS=2.5
        ;;
    *B200*)
        GPU_SHORT_NAME="B200"
        DEFAULT_MAX_SEQ_LENGTH=4096
        DEFAULT_TOKEN_MICRO_BATCH_SIZE=4096
        DEFAULT_FP8_PEAK_PFLOPS=4.5
        DEFAULT_BF16_PEAK_PFLOPS=2.25
        ;;
    *)
        echo "Unsupported GPU '$GPU_NAME'; set GPU_NAME to a supported B200 or B300 model." >&2
        exit 2
        ;;
esac

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-$DEFAULT_MAX_SEQ_LENGTH}"
TOKEN_MICRO_BATCH_SIZE="${TOKEN_MICRO_BATCH_SIZE:-$DEFAULT_TOKEN_MICRO_BATCH_SIZE}"
FP8_PEAK_PFLOPS="${FP8_PEAK_PFLOPS:-$DEFAULT_FP8_PEAK_PFLOPS}"
BF16_PEAK_PFLOPS="${BF16_PEAK_PFLOPS:-$DEFAULT_BF16_PEAK_PFLOPS}"
OUTPUT_DIR="${OUTPUT_DIR:-${TMPDIR:-/tmp}/mixtral_native_te_8x${GPU_SHORT_NAME}}"
RESULTS_CSV="${RESULTS_CSV:-$OUTPUT_DIR/mixtral_8x7b_8x${GPU_SHORT_NAME}.csv}"

if [[ ! -e "$MIXTRAL_TE_PRETRAINED_CHECKPOINT" ]]; then
    echo "Missing converted pretrained checkpoint: $MIXTRAL_TE_PRETRAINED_CHECKPOINT" >&2
    echo "Create it from mistralai/Mixtral-8x7B-v0.1 with" >&2
    echo "models/mixtral/export.export_hf_state_dict and expert_ffn_mode=fused_grouped_mlp." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$(dirname "$RESULTS_CSV")"

echo "Benchmarking on 8x $GPU_NAME"
echo "max_seq=$MAX_SEQ_LENGTH token_mb=$TOKEN_MICRO_BATCH_SIZE"

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

echo "Preparing the tokenizer cache"
(
    cd "$RECIPE_DIR"
    python - <<'PY'
from transformers import AutoTokenizer


AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
PY
)

# DCLM is streamed from the local parquet file, and the tokenizer is now local. Keep rank-local
# startup from issuing network requests; tokenization and packing remain part of dataloader timing.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

COMMON_OVERRIDES=(
    --config-name L1_8x7B_B200
    num_train_steps="$NUM_TRAIN_STEPS"
    logger.frequency="$LOGGER_FREQUENCY"
    dataset.max_seq_length="$MAX_SEQ_LENGTH"
    dataset.token_micro_batch_size="$TOKEN_MICRO_BATCH_SIZE"
)

run_benchmark() {
    local dp_size="$1"
    local ep_size="$2"
    local precision="$3"
    local fp8_enabled=false
    local quantized_model_init=false
    local store_param_remainders=true
    local start_time

    case "$precision" in
        fp8)
            fp8_enabled=true
            quantized_model_init=true
            # Qinit preserves and seeds a full FP32 master value for persistent MXFP8 parameters.
            store_param_remainders=false
            ;;
        # For BF16, FusedAdam represents the FP32 main weight as the BF16 parameter plus its
        # 16-bit remainder, retaining FP32 precision without allocating a separate FP32 copy.
        bf16) ;;
        *)
            echo "Unsupported precision: $precision (expected fp8 or bf16)" >&2
            return 2
            ;;
    esac

    echo "Running Mixtral-8x7B dp=$dp_size ep=$ep_size precision=$precision on 8 GPUs"
    start_time="$(date +%s)"
    (
        cd "$RECIPE_DIR"
        torchrun --standalone --nproc_per_node=8 train_fsdp2_ep.py \
            "${COMMON_OVERRIDES[@]}" \
            parallelism.dp_size="$dp_size" \
            parallelism.ep_size="$ep_size" \
            fp8_config.enabled="$fp8_enabled" \
            fp8_config.quantized_model_init_kwargs.enabled="$quantized_model_init" \
            optimizer_store_param_remainders="$store_param_remainders"
    ) 2>&1 | tee "$OUTPUT_DIR/dp${dp_size}_ep${ep_size}_${precision}.log"
    echo "$(($(date +%s) - start_time))" >"$OUTPUT_DIR/dp${dp_size}_ep${ep_size}_${precision}.elapsed"
}

for parallel_config in $PARALLEL_CONFIGS; do
    IFS=, read -r dp_size ep_size <<<"$parallel_config"
    for precision in $PRECISIONS; do
        run_benchmark "$dp_size" "$ep_size" "$precision"
    done
done

echo
echo "Steady-state summary (last three reported windows):"
OUTPUT_DIR="$OUTPUT_DIR" PRECISIONS="$PRECISIONS" PARALLEL_CONFIGS="$PARALLEL_CONFIGS" \
    FP8_PEAK_PFLOPS="$FP8_PEAK_PFLOPS" BF16_PEAK_PFLOPS="$BF16_PEAK_PFLOPS" \
    TOKEN_MICRO_BATCH_SIZE="$TOKEN_MICRO_BATCH_SIZE" python - <<'PY' | tee "$RESULTS_CSV"
import os
import re
from pathlib import Path
from statistics import mean


metric_pattern = re.compile(r"([a-z_]+): ([0-9.e+-]+)")
num_active_parameters = 12_748_587_008
peak_pflops = {
    "fp8": float(os.environ["FP8_PEAK_PFLOPS"]),
    "bf16": float(os.environ["BF16_PEAK_PFLOPS"]),
}
print(
    "dp,ep,precision,token_mb,tokens_per_s_per_gpu,pflops_per_gpu,mfu_pct,"
    "step_time_s,mem_gb,last_loss,n_steady,elapsed_s,rc"
)
for precision in os.environ["PRECISIONS"].split():
    for parallel_config in os.environ["PARALLEL_CONFIGS"].split():
        dp_size, ep_size = parallel_config.split(",")
        stem = Path(os.environ["OUTPUT_DIR"]) / f"dp{dp_size}_ep{ep_size}_{precision}"
        rows = []
        for line in stem.with_suffix(".log").read_text().splitlines():
            if "[perf_logger][INFO]" in line and "step_time: " in line:
                rows.append({key: float(value) for key, value in metric_pattern.findall(line)})
        steady = rows[-3:]
        if len(steady) != 3:
            raise RuntimeError(f"{stem}.log has {len(steady)} steady windows; expected 3")
        tokens_per_second = mean(row["tokens_per_second_per_gpu"] for row in steady)
        pflops = 6 * num_active_parameters * tokens_per_second / 1e15
        mfu = 100 * pflops / peak_pflops[precision]
        elapsed = int(stem.with_suffix(".elapsed").read_text())
        print(
            f"{dp_size},{ep_size},{precision},{os.environ['TOKEN_MICRO_BATCH_SIZE']},"
            f"{tokens_per_second:.0f},{pflops:.4f},{mfu:.2f},"
            f"{mean(row['step_time'] for row in steady):.3f},"
            f"{mean(row['gpu_memory_allocated_mean_gb'] for row in steady):.1f},"
            f"{steady[-1]['loss']:.2f},{len(steady)},{elapsed},0"
        )
PY
echo "Wrote $RESULTS_CSV"
