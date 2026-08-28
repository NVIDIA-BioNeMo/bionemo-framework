#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

# Agent-free PhiX174 whole-genome SFT -> GDPO -> generation/screening example.

set -Eeuo pipefail

RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${RECIPE_ROOT}/results/phix174-8xh100"
DRY_RUN=0
PREPARE_ONLY=0
CALIBRATE_ONLY=0
RESUME_FROM=00
SAMPLING_SELECTION_SOURCE=
MODEL_VARIANT="${MODEL_VARIANT:-7b-base}"
HOPPER_FP8_INFERENCE=0
WANDB_ENABLED=0
WANDB_OPTION_CONFIGURED=0
WANDB_ENTITY_NAME="${WANDB_ENTITY:-}"
WANDB_SFT_PROJECT_NAME='evo2-phage-design-sft'
WANDB_RL_PROJECT_NAME='evo2-phage-design-gdpo'
NUM_GPUS="${NUM_GPUS:-8}"
NUM_CPUS="${NUM_CPUS:-${NEMO_RL_RAY_NUM_CPUS:-$(nproc)}}"
for resource_name in NUM_GPUS NUM_CPUS; do
  resource_value="${!resource_name}"
  if [[ ! "${resource_value}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer; got %q\n' "${resource_name}" "${resource_value}" >&2
    exit 2
  fi
done
if [[ -z "${SFT_TENSOR_PARALLEL_SIZE:-}" ]]; then
  if ((NUM_GPUS == 1)); then
    SFT_TENSOR_PARALLEL_SIZE=1
  else
    SFT_TENSOR_PARALLEL_SIZE=2
  fi
fi
if [[ ! "${SFT_TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'SFT_TENSOR_PARALLEL_SIZE must be a positive integer; got %q\n' "${SFT_TENSOR_PARALLEL_SIZE}" >&2
  exit 2
fi
if ((NUM_GPUS % SFT_TENSOR_PARALLEL_SIZE != 0)); then
  printf 'NUM_GPUS (%s) must be divisible by SFT_TENSOR_PARALLEL_SIZE (%s)\n' \
    "${NUM_GPUS}" "${SFT_TENSOR_PARALLEL_SIZE}" >&2
  exit 2
fi
GPU_IDS=
for ((gpu_index=0; gpu_index<NUM_GPUS; gpu_index++)); do
  GPU_IDS+="${GPU_IDS:+ }${gpu_index}"
done
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-600}"
PHAROKKA_DATABASE_URL="${PHAROKKA_DATABASE_URL:-https://zenodo.org/records/21755221/files/pharokka_v1.11.0_databases.tar.gz?download=1}"
PHAROKKA_DATABASE_MD5="${PHAROKKA_DATABASE_MD5:-143bb375ddb0b0653e5cb5671f4a7629}"
PHAROKKA_DATABASE_RELEASE="${PHAROKKA_DATABASE_RELEASE:-Pharokka database v1.11.0 / PHROGs v4}"
CALIBRATION_WORKERS="${CALIBRATION_WORKERS:-8}"
SAFETY_BATCH_SIZE="${SAFETY_BATCH_SIZE:-128}"
SAFETY_ORF_WORKERS="${SAFETY_ORF_WORKERS:-32}"
SAFETY_THREADS="${SAFETY_THREADS:-32}"
SAFETY_PHROGS_THREADS="${SAFETY_PHROGS_THREADS:-64}"

usage() {
  printf '%s\n' \
    'Usage: ./examples/phix174_8xh100.sh [OPTIONS]' \
    '  --result-root PATH         Result directory (default: results/phix174-8xh100)' \
    '  --model-variant NAME       7b-base (default) or 7b-1m' \
    '  --sampling-selection PATH  Copy and use a sampling-selection YAML' \
    '  --hopper-fp8-inference     Opt in to regular all-layer FP8 for calibration/rollout/scoring' \
    '  --wandb                    Log the full SFT and GDPO runs to W&B' \
    '  --wandb-entity NAME        W&B entity/team (or use WANDB_ENTITY)' \
    '  --wandb-sft-project NAME   SFT project (default: evo2-phage-design-sft)' \
    '  --wandb-rl-project NAME    GDPO project (default: evo2-phage-design-gdpo)' \
    '  --prepare-only             Prepare public inputs/tools/controls, then stop' \
    '  --calibrate-only           Stop after calibration scoring for sampling review' \
    '  --resume-from ID           Start at stage 00, 10, 20, 30, 40, or 50' \
    '  --dry-run                  Record and print commands without external work' \
    '  -h, --help                 Show this help'
}

while (($#)); do
  case "$1" in
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --model-variant) MODEL_VARIANT="$2"; shift 2 ;;
    --sampling-selection) SAMPLING_SELECTION_SOURCE="$2"; shift 2 ;;
    --hopper-fp8-inference) HOPPER_FP8_INFERENCE=1; shift ;;
    --wandb) WANDB_ENABLED=1; shift ;;
    --wandb-entity)
      if (($# < 2)) || [[ -z "$2" ]]; then
        printf '%s\n' '--wandb-entity requires a value' >&2; exit 2
      fi
      WANDB_ENTITY_NAME="$2"; WANDB_OPTION_CONFIGURED=1; shift 2
      ;;
    --wandb-sft-project)
      if (($# < 2)) || [[ -z "$2" ]]; then
        printf '%s\n' '--wandb-sft-project requires a value' >&2; exit 2
      fi
      WANDB_SFT_PROJECT_NAME="$2"; WANDB_OPTION_CONFIGURED=1; shift 2
      ;;
    --wandb-rl-project)
      if (($# < 2)) || [[ -z "$2" ]]; then
        printf '%s\n' '--wandb-rl-project requires a value' >&2; exit 2
      fi
      WANDB_RL_PROJECT_NAME="$2"; WANDB_OPTION_CONFIGURED=1; shift 2
      ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --calibrate-only) CALIBRATE_ONLY=1; shift ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ "${WANDB_ENABLED}" != "1" && "${WANDB_OPTION_CONFIGURED}" == "1" ]]; then
  printf '%s\n' '--wandb-entity and project overrides require --wandb' >&2
  exit 2
fi
case "${RESUME_FROM}" in 00|10|20|30|40|50) ;; *) printf 'Invalid stage: %s\n' "${RESUME_FROM}" >&2; exit 2 ;; esac
if [[ "${PREPARE_ONLY}" == "1" && "${CALIBRATE_ONLY}" == "1" ]]; then
  printf '%s\n' '--prepare-only and --calibrate-only are mutually exclusive' >&2
  exit 2
fi
if [[ "${CALIBRATE_ONLY}" == "1" ]] && ((10#${RESUME_FROM} > 30)); then
  printf '%s\n' '--calibrate-only requires --resume-from 00, 10, 20, or 30' >&2
  exit 2
fi
case "${MODEL_VARIANT}" in
  7b-base)
    BASE_CHECKPOINT_RESOURCE='evo2/7b-8k:1.0'
    BASE_CHECKPOINT_DIR='evo2-7b-8k-mbridge-10240'
    BASE_DOWNLOAD_PLACEHOLDER='<downloaded-evo2-7b-8k>'
    MODEL_SIZE='evo2_7b_base'
    RL_MODEL_NAME='bionemo/evo2_7b_base'
    ;;
  7b-1m)
    BASE_CHECKPOINT_RESOURCE='evo2/7b-1m:1.0'
    BASE_CHECKPOINT_DIR='evo2-7b-1m-mbridge-10240'
    BASE_DOWNLOAD_PLACEHOLDER='<downloaded-evo2-7b-1m>'
    MODEL_SIZE='evo2_7b'
    RL_MODEL_NAME='bionemo/evo2_7b'
    ;;
  *)
    printf 'Invalid model variant: %s (expected 7b-base or 7b-1m)\n' "${MODEL_VARIANT}" >&2
    exit 2
    ;;
esac

INFERENCE_PRECISION_NAME='bf16'
declare -a INFERENCE_PRECISION_ARGS=()
if [[ "${HOPPER_FP8_INFERENCE}" == "1" ]]; then
  INFERENCE_PRECISION_NAME='hopper-regular-fp8-all-layers'
  INFERENCE_PRECISION_ARGS=(
    --mixed-precision-recipe bf16_with_fp8_current_scaling_mixed
    --fp8-all-layers
  )
fi

WANDB_RUN_STEM="$(basename -- "${RESULT_ROOT}")-${MODEL_VARIANT}"
WANDB_SFT_RUN_NAME="${WANDB_RUN_STEM}-sft"
WANDB_RL_RUN_NAME="${WANDB_RUN_STEM}-gdpo"
declare -a SFT_WANDB_ARGS=()
declare -a RL_WANDB_ARGS=(logger.wandb_enabled=false)
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  SFT_WANDB_ARGS=(
    --wandb-project "${WANDB_SFT_PROJECT_NAME}"
    --wandb-run-name "${WANDB_SFT_RUN_NAME}"
  )
  if [[ -n "${WANDB_ENTITY_NAME}" ]]; then
    SFT_WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY_NAME}")
    export WANDB_ENTITY="${WANDB_ENTITY_NAME}"
  fi
  RL_WANDB_ARGS=(
    logger.wandb_enabled=true
    logger.wandb.project="${WANDB_RL_PROJECT_NAME}"
    logger.wandb.name="${WANDB_RL_RUN_NAME}"
  )
fi

STATE_DIR="${RESULT_ROOT}/state"
STAGE_DIR="${RESULT_ROOT}/stages"
RUNLOG="${RESULT_ROOT}/RUNLOG.md"
SAMPLING_SELECTION="${RESULT_ROOT}/calibration/sampling-selection.yaml"
DEFAULT_SAMPLING_SELECTION="${RECIPE_ROOT}/examples/default-sampling-selection.yaml"
SAMPLING_SELECTION_OVERRIDDEN=0
SAMPLING_TEMPERATURE=
SAMPLING_TOP_K=
SAMPLING_TOP_P=
SAMPLING_MAX_NEW_TOKENS=
SAMPLING_PROMPT_LENGTHS_TEXT=
SAMPLING_RL_SEED=
SAMPLING_ROLLOUT_SEED=
SAMPLING_SEED_STRIDE=
SAMPLING_PROMPT_LABEL=
SAMPLING_TRAIN_RECORDS=
SAMPLING_FINAL_PER_LENGTH=
declare -a SAMPLING_PROMPT_LENGTHS=()
mkdir -p "${RESULT_ROOT}" "${STATE_DIR}" "${STAGE_DIR}"
exec 9> "${RESULT_ROOT}/.run.lock"
if ! flock -n 9; then
  printf 'Another PhiX174 example is already running for this result directory: %s\n' "${RESULT_ROOT}" >&2
  exit 1
fi
[[ -f "${RUNLOG}" ]] || printf '# PhiX174 8xH100 run log\n\n' > "${RUNLOG}"

note() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${RUNLOG}"
}

TOTAL_STAGES=6
CURRENT_STAGE_ID=initialization
CURRENT_STAGE_ORDINAL=0
CURRENT_STAGE_DESCRIPTION='initialize run'
RUN_COMPLETION_REPORTED=0

completed_stage_count() {
  local count=0 stage
  for stage in 00 10 20 30 40 50; do
    [[ -f "${STAGE_DIR}/${stage}.done" ]] && count=$((count + 1))
  done
  printf '%s\n' "${count}"
}

set_current_stage() {
  CURRENT_STAGE_ID="$1"
  case "$1" in
    00) CURRENT_STAGE_ORDINAL=1; CURRENT_STAGE_DESCRIPTION='prepare inputs, tools, and controls' ;;
    10) CURRENT_STAGE_ORDINAL=2; CURRENT_STAGE_DESCRIPTION='safety-screen and prepare SFT' ;;
    20) CURRENT_STAGE_ORDINAL=3; CURRENT_STAGE_DESCRIPTION='train and select SFT' ;;
    30) CURRENT_STAGE_ORDINAL=4; CURRENT_STAGE_DESCRIPTION='calibrate sampling' ;;
    40) CURRENT_STAGE_ORDINAL=5; CURRENT_STAGE_DESCRIPTION='train and select GDPO' ;;
    50) CURRENT_STAGE_ORDINAL=6; CURRENT_STAGE_DESCRIPTION='generate, screen, and report' ;;
  esac
}

report_run_exit() {
  local status=$? completed
  trap - EXIT
  if [[ "${status}" != 0 && "${RUN_COMPLETION_REPORTED}" != 1 ]]; then
    set +e
    completed="$(completed_stage_count)"
    if ((CURRENT_STAGE_ORDINAL > 0)); then
      note "RUN FAILED during step ${CURRENT_STAGE_ORDINAL}/${TOTAL_STAGES} (stage ${CURRENT_STAGE_ID}: ${CURRENT_STAGE_DESCRIPTION}); ${completed}/${TOTAL_STAGES} steps complete; exit code ${status}; see ${RUNLOG}" >&2
    else
      note "RUN FAILED before step 1/${TOTAL_STAGES} (${CURRENT_STAGE_DESCRIPTION}); ${completed}/${TOTAL_STAGES} steps complete; exit code ${status}; see ${RUNLOG}" >&2
    fi
  fi
  exit "${status}"
}

trap report_run_exit EXIT

MODEL_VARIANT_STATE="${STATE_DIR}/model-variant"
if [[ -s "${MODEL_VARIANT_STATE}" ]]; then
  RECORDED_MODEL_VARIANT="$(sed -n '1p' "${MODEL_VARIANT_STATE}")"
elif [[ -s "${STATE_DIR}/selected-sft" || -f "${STAGE_DIR}/20-sft.done" ]]; then
  # Runs created before this selector existed used the publication-style 7B-base checkpoint.
  RECORDED_MODEL_VARIANT='7b-base'
else
  RECORDED_MODEL_VARIANT="${MODEL_VARIANT}"
fi
if [[ "${MODEL_VARIANT}" != "${RECORDED_MODEL_VARIANT}" ]]; then
  printf 'This result root recorded model variant is %s, not %s; use the recorded variant or a new result root.\n' \
    "${RECORDED_MODEL_VARIANT}" "${MODEL_VARIANT}" >&2
  exit 2
fi
printf '%s\n' "${MODEL_VARIANT}" > "${MODEL_VARIANT_STATE}"
note "model variant: ${MODEL_VARIANT} (${BASE_CHECKPOINT_RESOURCE}, model size ${MODEL_SIZE})"

sampling_selection_fields() {
  python - "$1" <<'PY'
import math
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
required = {
    "temperature",
    "top_k",
    "top_p",
    "max_new_tokens",
    "prompt_lengths",
    "rl_seed",
    "rollout_seed",
    "seed_stride",
}

try:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("document must be a mapping")
    missing, unknown = required - data.keys(), data.keys() - required
    if missing:
        raise ValueError(f"missing keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"unknown keys: {', '.join(sorted(unknown))}")

    def real(name, *, lower, upper=None):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        value = float(value)
        if not math.isfinite(value) or value <= lower or (upper is not None and value > upper):
            interval = f"({lower}, {upper}]" if upper is not None else f"> {lower}"
            raise ValueError(f"{name} must be finite and {interval}")
        return value

    def integer(name, *, minimum):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return value

    temperature = real("temperature", lower=0)
    top_k = integer("top_k", minimum=0)
    top_p = real("top_p", lower=0, upper=1)
    max_new_tokens = integer("max_new_tokens", minimum=1)
    rl_seed = integer("rl_seed", minimum=0)
    rollout_seed = integer("rollout_seed", minimum=0)
    seed_stride = integer("seed_stride", minimum=1)
    prompt_lengths = data["prompt_lengths"]
    if not isinstance(prompt_lengths, list) or not prompt_lengths:
        raise ValueError("prompt_lengths must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in prompt_lengths):
        raise ValueError("prompt_lengths must contain integers")
    if len(set(prompt_lengths)) != len(prompt_lengths):
        raise ValueError("prompt_lengths must be unique")
    if any(value < 0 or value > 65 for value in prompt_lengths):
        raise ValueError("prompt_lengths must be between 0 and the 65-nt PhiX174 reference prefix")
    strata = len(prompt_lengths)
    if max_new_tokens + max(prompt_lengths) > 10240:
        raise ValueError("max_new_tokens plus the longest prompt exceeds the 10,240-token model context")
except (OSError, ValueError, yaml.YAMLError) as error:
    print(f"Invalid sampling selection {path}: {error}", file=sys.stderr)
    raise SystemExit(2) from error

fields = (
    str(temperature),
    str(top_k),
    str(top_p),
    str(max_new_tokens),
    " ".join(str(value) for value in prompt_lengths),
    str(rl_seed),
    str(rollout_seed),
    str(seed_stride),
    "-".join(str(value) for value in prompt_lengths),
    # Cover every selected stratum and retain complete two-prompt GDPO steps.
    str(max(12, 2 * ((strata + 1) // 2))),
    str((1000 + strata - 1) // strata),
)
print("\t".join(fields))
PY
}

load_sampling_selection() {
  local fields
  if [[ -s "${SAMPLING_SELECTION}" ]]; then
    fields="$(sampling_selection_fields "${SAMPLING_SELECTION}")" || return
  elif [[ "${DRY_RUN}" == "1" ]]; then
    fields="$(sampling_selection_fields "${DEFAULT_SAMPLING_SELECTION}")" || return
  else
    printf 'Missing sampling selection: %s\n' "${SAMPLING_SELECTION}" >&2
    return 2
  fi
  IFS=$'\t' read -r SAMPLING_TEMPERATURE SAMPLING_TOP_K SAMPLING_TOP_P \
    SAMPLING_MAX_NEW_TOKENS SAMPLING_PROMPT_LENGTHS_TEXT SAMPLING_RL_SEED \
    SAMPLING_ROLLOUT_SEED SAMPLING_SEED_STRIDE SAMPLING_PROMPT_LABEL \
    SAMPLING_TRAIN_RECORDS SAMPLING_FINAL_PER_LENGTH <<< "${fields}"
  read -r -a SAMPLING_PROMPT_LENGTHS <<< "${SAMPLING_PROMPT_LENGTHS_TEXT}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  # shellcheck source=/dev/null
  source "${RECIPE_ROOT}/.ci_test_env.sh"
fi

if [[ -n "${SAMPLING_SELECTION_SOURCE}" ]]; then
  sampling_selection_fields "${SAMPLING_SELECTION_SOURCE}" > /dev/null
  mkdir -p "$(dirname -- "${SAMPLING_SELECTION}")"
  if [[ ! -e "${SAMPLING_SELECTION}" || ! "${SAMPLING_SELECTION_SOURCE}" -ef "${SAMPLING_SELECTION}" ]]; then
    selection_tmp="${SAMPLING_SELECTION}.tmp.$$"
    cp -- "${SAMPLING_SELECTION_SOURCE}" "${selection_tmp}"
    mv -- "${selection_tmp}" "${SAMPLING_SELECTION}"
  fi
  SAMPLING_SELECTION_OVERRIDDEN=1
  if [[ "${SAMPLING_SELECTION_SOURCE}" -ef "${DEFAULT_SAMPLING_SELECTION}" ]]; then
    note "WARNING: using bundled historical sampling settings from ${SAMPLING_SELECTION_SOURCE}; they override rather than derive from fresh calibration"
  else
    note "WARNING: copied explicit sampling selection from ${SAMPLING_SELECTION_SOURCE} to ${SAMPLING_SELECTION}; fresh calibration compatibility will not be enforced"
  fi
fi

run() {
  printf -v command '%q ' "$@"
  note "command: ${command}"
  [[ "${DRY_RUN}" == "1" ]] || "$@"
}

run_result() {
  local label="$1" log="$2"
  shift 2
  printf -v command '%q ' "$@"
  note "command: ${command}"
  [[ "${DRY_RUN}" == "1" ]] && return
  mkdir -p "$(dirname -- "${log}")"
  set +e
  "$@" > "${log}" 2>&1
  local status=$?
  set -e
  case "${status}" in 0|2|3) note "${label}: scientific result exit ${status}" ;; *) tail -n 30 "${log}" >&2; return "${status}" ;; esac
}

monitored() {
  local label="$1" log="$2"
  shift 2
  printf -v command '%q ' "$@"
  note "command: ${command}"
  note "monitor: ${label}; log: ${log}"
  [[ "${DRY_RUN}" == "1" ]] && return
  mkdir -p "$(dirname -- "${log}")"
  "$@" > "${log}" 2>&1 &
  local child=$! started=${SECONDS} waited
  while kill -0 "${child}" 2>/dev/null; do
    waited=0
    while (( waited < MONITOR_INTERVAL_SECONDS )) && kill -0 "${child}" 2>/dev/null; do sleep 10; waited=$((waited + 10)); done
    kill -0 "${child}" 2>/dev/null && note "${label} still running after $((SECONDS - started))s; log: ${log}"
  done
  set +e; wait "${child}"; local status=$?; set -e
  if [[ "${status}" != "0" ]]; then tail -n 30 "${log}" >&2; return "${status}"; fi
  note "${label} complete after $((SECONDS - started))s"
}

state() { printf '%s\n' "$2" > "${STATE_DIR}/$1"; }
read_state() { [[ "${DRY_RUN}" == "1" ]] && printf '<%s>\n' "$1" || sed -n '1p' "${STATE_DIR}/$1"; }

check_scan() {
  local mode="${2:-strict}" tolerated
  [[ "${DRY_RUN}" == "1" ]] && return
  case "${mode}" in strict|allow-no-primary-gene-candidates) ;; *) printf 'invalid safety validation mode: %s\n' "${mode}" >&2; return 2 ;; esac
  tolerated="$(python - "$1" "${mode}" <<'PY'
import sys
from pathlib import Path

from bionemo.evo2_phage_gen.sequence_safety_cli import (
    CLIValidationError,
    validate_detector_execution,
    validate_manifest_file,
)

manifest = validate_manifest_file(Path(sys.argv[1]), expected_type="sequence_safety_scan")
try:
    allowed = validate_detector_execution(
        manifest,
        allow_no_primary_gene_candidates=sys.argv[2] == "allow-no-primary-gene-candidates",
    )
except CLIValidationError as error:
    raise SystemExit(str(error)) from None
print(len({record_id for record_id, _safety_class, _reason in allowed}))
PY
  )"
  if [[ "${tolerated}" != 0 ]]; then
    note "final safety scan retained ${tolerated} no-primary-gene candidate(s) as INDETERMINATE; they remain ineligible for hard-QC PASS"
  fi
}

require_file() {
  [[ "${DRY_RUN}" == "1" ]] && return
  [[ -f "$1" ]] || { printf 'missing %s: %s\n' "$2" "$1" >&2; return 1; }
}

require_nonempty_file() {
  [[ "${DRY_RUN}" == "1" ]] && return
  [[ -s "$1" ]] || { printf 'missing or empty %s: %s\n' "$2" "$1" >&2; return 1; }
}

check_success_report() {
  [[ "${DRY_RUN}" == "1" ]] && return
  python - "$1" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
if payload.get("state") != "succeeded":
    raise SystemExit(f"nonterminal report {path}: {payload.get('state')!r}")
PY
}

check_objectives() {
  [[ "${DRY_RUN}" == "1" ]] && return
  python - "$1" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
if result["decision"] == "pause_for_diagnosis":
    raise SystemExit(f'RL objective monitor requested diagnosis: {result["reason"]}')
PY
}

select_checkpoint() {
  local mode="$1" tensorboard_root="$2" checkpoint_root="$3" output="$4"
  python - "${mode}" "${tensorboard_root}" "${checkpoint_root}" "${output}" <<'PY'
import json, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

mode, tb_root, ckpt_root, output = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
tags = ["lm loss validation"] if mode == "sft" else [
    "validation/phage_qc/binary_safety_qualified_full_qc_cluster_deduplicated_rate",
    "val:phage_qc/binary_safety_qualified_full_qc_cluster_deduplicated_rate",
]
points = {}
chosen_tag = None
for event in sorted(tb_root.rglob("events.out.tfevents*")):
    acc = EventAccumulator(str(event), size_guidance={"scalars": 0}); acc.Reload()
    for tag in tags:
        if tag in acc.Tags().get("scalars", []):
            chosen_tag = chosen_tag or tag
            for scalar in acc.Scalars(tag):
                previous = points.get(scalar.step)
                if previous is None or scalar.wall_time >= previous[0]:
                    points[scalar.step] = (scalar.wall_time, scalar.value)
values = sorted((int(step), float(value[1])) for step, value in points.items())
if len(values) < 3:
    raise SystemExit("need at least three comparable validation events")
index = (min if mode == "sft" else max)(range(len(values)), key=lambda i: (values[i][1], values[i][0]))
if (mode == "sft" and index > len(values) - 3) or (mode == "rl" and index in (0, len(values) - 1)):
    raise SystemExit("best validation is at the run boundary; extend/inspect the run before selecting")
step, value = values[index]
checkpoint = ckpt_root / (f"iter_{step:07d}" if mode == "sft" else f"step_{step}/policy/weights/iter_0000000")
if not checkpoint.is_dir():
    raise SystemExit(f"selected validation step has no checkpoint: {checkpoint}")
result = {"metric": chosen_tag, "direction": "minimize" if mode == "sft" else "maximize", "step": step, "value": value, "checkpoint": str(checkpoint.resolve())}
output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n")
print(result["checkpoint"])
PY
}

gpu_type='not queried (dry run)'
if [[ "${DRY_RUN}" != "1" ]]; then
  export PATH="${RECIPE_ROOT}/data/external/bin:${PATH}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export NCCL_GRAPH_REGISTER=0
  if ! gpu_info="$(nvidia-smi --query-gpu=name --format=csv,noheader)"; then
    printf '%s\n' 'Unable to query GPUs. This script requires GPUs; run it on the allocated compute node, outside a restricted agent sandbox.' >&2
    exit 2
  fi
  detected_gpus="$(awk 'NF { count++ } END { print count + 0 }' <<< "${gpu_info}")"
  if [[ "${detected_gpus}" != "${NUM_GPUS}" ]]; then
    printf 'Expected %s GPUs; found %s:\n%s\n' "${NUM_GPUS}" "${detected_gpus}" "${gpu_info}" >&2
    if [[ "${detected_gpus}" == "0" ]]; then
      printf '%s\n' 'This script requires GPUs.' >&2
    else
      printf 'If this topology is intended, rerun with NUM_GPUS=%s and tune its batch and parallelism settings.\n' "${detected_gpus}" >&2
    fi
    exit 2
  fi
  gpu_type="$(sed -n '1p' <<< "${gpu_info}")"
  if [[ "$(grep -c H100 <<< "${gpu_info}")" != "${NUM_GPUS}" ]]; then
    printf 'Warning: this example was tested on 8 H100 80GB GPUs. Detected:\n%s\nTune memory, batch, and parallelism settings for this topology.\n' "${gpu_info}" >&2
  fi
  if [[ "${HOPPER_FP8_INFERENCE}" == "1" && "$(grep -Ec 'H100|H200' <<< "${gpu_info}")" != "${NUM_GPUS}" ]]; then
    printf '%s\n' '--hopper-fp8-inference requires an all-Hopper H100/H200 allocation' >&2
    exit 2
  fi
fi
note "planned topology: ${NUM_GPUS} GPUs, SFT tensor parallel ${SFT_TENSOR_PARALLEL_SIZE}, ${NUM_CPUS} logical CPUs; inference precision ${INFERENCE_PRECISION_NAME}"
python - "${RESULT_ROOT}/settings.json" "${NUM_GPUS}" "${NUM_CPUS}" "${gpu_type}" \
  "${SFT_TENSOR_PARALLEL_SIZE}" "${MODEL_VARIANT}" "${BASE_CHECKPOINT_RESOURCE}" "${MODEL_SIZE}" \
  "${WANDB_ENABLED}" "${WANDB_ENTITY_NAME}" "${WANDB_SFT_PROJECT_NAME}" "${WANDB_RL_PROJECT_NAME}" \
  "${WANDB_SFT_RUN_NAME}" "${WANDB_RL_RUN_NAME}" "${INFERENCE_PRECISION_NAME}" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    gpu_count,
    cpu_count,
    gpu_type,
    sft_tensor_parallel_size,
    model_variant,
    base_checkpoint,
    model_size,
    wandb_enabled,
    wandb_entity,
    wandb_sft_project,
    wandb_rl_project,
    wandb_sft_run_name,
    wandb_rl_run_name,
    inference_precision,
) = sys.argv[1:]
wandb_is_enabled = wandb_enabled == "1"
settings = {
    "gpu_count": int(gpu_count),
    "cpu_count": int(cpu_count),
    "gpu_type": gpu_type,
    "sft_tensor_parallel_size": int(sft_tensor_parallel_size),
    "model_variant": model_variant,
    "base_checkpoint": base_checkpoint,
    "model_size": model_size,
    "inference_precision": inference_precision,
    "whole_genome": True,
    "safety_screen": "current configured databases",
    "final_generation_count": 1000,
    "wandb_enabled": wandb_is_enabled,
    "wandb_entity": (wandb_entity or None) if wandb_is_enabled else None,
    "wandb_sft_project": wandb_sft_project,
    "wandb_rl_project": wandb_rl_project,
    "wandb_sft_run_name": wandb_sft_run_name,
    "wandb_rl_run_name": wandb_rl_run_name,
}
Path(output).write_text(json.dumps(settings, indent=2) + "\n")
PY
cd "${RECIPE_ROOT}"

stage_00() {
  run evo2_phage_download_sft_data --include-raw
  monitored 'external asset preparation' "${RESULT_ROOT}/inputs/external-assets.log" \
    evo2_phage_prepare_external_assets --external-dir data/external --bin-dir data/external/bin \
    --download-large-databases --prepare-phrogs-consensus-database --with-safety \
    --pharokka-database-url "${PHAROKKA_DATABASE_URL}" \
    --pharokka-database-md5 "${PHAROKKA_DATABASE_MD5}" \
    --pharokka-database-release "${PHAROKKA_DATABASE_RELEASE}"
  run evo2_phage_prepare_arc_pipeline --output-dir data/arc_pipeline_patched --overwrite
  if [[ "${DRY_RUN}" == "1" || ! -s data/external/mmseqs/NC_001422_1_Gprotein/mmseqs_db_NC_001422_1_Gprotein.dbtype ]]; then
    run mkdir -p data/external/mmseqs/NC_001422_1_Gprotein
    run mmseqs createdb data/external/arc_evo2/phage_gen/data/NC_001422.1_Gprotein.fasta data/external/mmseqs/NC_001422_1_Gprotein/mmseqs_db_NC_001422_1_Gprotein
  fi
  local root="${RESULT_ROOT}/inputs/reference-controls" table="${RESULT_ROOT}/inputs/reference-controls/controls.tsv"
  python - configs/phage_safety_reference_controls.yaml "${root}" "${table}" "${DRY_RUN}" <<'PY'
import csv, json, sys, urllib.parse, urllib.request
from pathlib import Path
import yaml
config, root, table, dry_run = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] == "1"; root.mkdir(parents=True, exist_ok=True)
rows = []
for c in yaml.safe_load(config.read_text())["controls"]:
    path = root / f'{c["control_id"]}.fasta'
    if not dry_run:
        interval = c.get("sequence_interval")
        query = {"db":"nuccore","id":c["accession"],"rettype":"fasta","retmode":"text"}
        if interval:
            query |= {"seq_start":interval["start"],"seq_stop":interval["end"]}
        text = urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(query), timeout=120).read().decode()
        sequence = "".join(line.strip() for line in text.splitlines() if line and not line.startswith(">"))
        if len(sequence) != c["sequence_length"] or set(sequence.upper()) - set("ACGTN"):
            raise SystemExit(f'invalid NCBI response for {c["accession"]}: {len(sequence)} bases')
        record_id = c["accession"] if not interval else f'{c["accession"]}_{interval["start"]}_{interval["end"]}'
        path.write_text(f">{record_id}\n{sequence}\n")
    evidence = json.dumps({"source":"NCBI Nucleotide","source_version":c["accession"],"replication_host_domains":["BACTERIA"],"confirmed":True}, separators=(",",":"))
    rows.append((c["control_id"], path.resolve(), c["topology"], evidence))
with table.open("w", newline="") as out:
    writer=csv.writer(out, delimiter="\t", lineterminator="\n", quotechar=None, quoting=csv.QUOTE_NONE); writer.writerow(("id","fasta","topology","evidence")); writer.writerows(rows)
PY
  local reports=() id fasta topology evidence scan command
  while IFS=$'\t' read -r id fasta topology evidence; do
    [[ "${id}" == id ]] && continue
    scan="${root}/scans/${id}"
    command=(evo2_phage_sequence_safety scan --input-fasta "${fasta}" --output-dir "${scan}" --policy configs/phage_safety_policy.yaml --asset-manifest data/external/safety/asset_manifest.yaml --host-domain BACTERIA --host-evidence-json "${evidence}" --strict-lysis --threads 16 --timeout 1800 --overwrite)
    [[ "${topology}" == linear ]] && command+=(--linear)
    run_result "control ${id}" "${root}/logs/${id}.log" "${command[@]}"
    check_scan "${scan}/manifest.json"; reports+=(--report "${id}=${scan}/manifest.json")
  done < "${table}"
  [[ "${DRY_RUN}" == "1" ]] && return
  evo2_phage_validate_safety_controls --config configs/phage_safety_reference_controls.yaml "${reports[@]}" --output "${root}/current-results.json" || {
    printf '# Review required\n\nCurrent safety-control behavior changed; inspect `%s`. Do not roll back databases automatically.\n' "${root}/current-results.json" > "${RESULT_ROOT}/REVIEW_REQUIRED.md"; return 4;
  }
}

stage_10() {
  local source=data/external/zenodo/microviridae_sft_training_data_processed.fna safety="${RESULT_ROOT}/sft/source-safety" prep="${RESULT_ROOT}/sft/prepared"
  if [[ "${DRY_RUN}" == "1" ]]; then note 'remove the two-character model prefix for safety scanning while preserving FASTA IDs'; else
    python - "${source}" "${safety}/biological.fna" <<'PY'
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
import sys
source, output = Path(sys.argv[1]), Path(sys.argv[2]); output.parent.mkdir(parents=True, exist_ok=True)
records = list(SeqIO.parse(source, "fasta"))
for record in records:
    sequence = str(record.seq); record.seq = Seq(sequence[2:] if sequence[:2] in ("+!","+#","+$","+^","+~") else sequence)
SeqIO.write(records, output, "fasta")
PY
  fi
  local evidence='{"source":"Zenodo record 17101843","source_version":"Zenodo record 17101843","replication_host_domains":["BACTERIA"],"confirmed":true}'
  run_result 'SFT safety scan' "${safety}/scan.log" evo2_phage_sequence_safety scan --input-fasta "${safety}/biological.fna" --output-dir "${safety}/scan" --policy configs/phage_safety_policy.yaml --asset-manifest data/external/safety/asset_manifest.yaml --host-domain BACTERIA --host-evidence-json "${evidence}" --strict-lysis --batch-size "${SAFETY_BATCH_SIZE}" --orf-workers "${SAFETY_ORF_WORKERS}" --threads "${SAFETY_THREADS}" --phrogs-threads "${SAFETY_PHROGS_THREADS}" --timeout 1800 --overwrite
  check_scan "${safety}/scan/manifest.json"
  run evo2_phage_summarize_safety_manifest --manifest "${safety}/scan/manifest.json" --output "${safety}/summary.json"
  run_result 'SFT safety partition' "${safety}/partition.log" evo2_phage_sequence_safety filter-fasta --input-fasta "${source}" --scan-manifest "${safety}/scan/manifest.json" --output-dir "${safety}/partitions" --overwrite
  run evo2_phage_prepare_sft_split --source-fasta "${safety}/partitions/pass.fasta" --output-dir "${prep}" --mmseqs-bin data/external/bin/mmseqs --validation-count 100 --test-count 100 --seed 1234 --min-seq-id 0.98 --coverage 0.8 --cov-mode 0 --threads 16
  run preprocess_evo2 --config "${prep}/preprocess.yaml"; state sft-prepared "${prep}"
}

stage_20() {
  local prep base_nemo base_mbridge="${RESULT_ROOT}/checkpoints/${BASE_CHECKPOINT_DIR}" sft="${RESULT_ROOT}/sft/train" selected
  prep="$(read_state sft-prepared)"
  local model=(--hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_512 --model-size "${MODEL_SIZE}" --micro-batch-size 1 --seq-length 10240 --tensor-model-parallel-size "${SFT_TENSOR_PARALLEL_SIZE}" --use-precision-aware-optimizer --bf16-main-grads --grad-reduce-in-fp32 --overlap-grad-reduce --cross-entropy-loss-fusion --no-weight-decay-embeddings --no-renormalize-loss --use-subquadratic-ops --no-fp32-residual-connection --activation-checkpoint-recompute-num-layers 1 --eod-pad-in-loss-mask --mixed-precision-recipe bf16_mixed)
  if [[ -f "${STAGE_DIR}/20-sft.done" ]]; then
    note 'substage 20-sft already complete'
  else
    [[ "${DRY_RUN}" == "1" ]] && base_nemo="${BASE_DOWNLOAD_PLACEHOLDER}" || base_nemo="$(download_bionemo_data "${BASE_CHECKPOINT_RESOURCE}" | tail -n 1)"
    if [[ "${DRY_RUN}" == "1" || ! -d "${base_mbridge}" ]]; then
      run evo2_convert_nemo2_to_mbridge --nemo2-ckpt-dir "${base_nemo}" --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 --mbridge-ckpt-dir "${base_mbridge}" --model-size "${MODEL_SIZE}" --seq-length 10240 --mixed-precision-recipe bf16_mixed
    fi
    monitored 'SFT smoke' "${RESULT_ROOT}/sft/smoke.log" torchrun --nproc-per-node "${NUM_GPUS}" --no-python train_evo2 "${model[@]}" --dataset-config "${prep}/training_dataset.yaml" --finetune-ckpt-dir "${base_mbridge}" --global-batch-size 32 --max-steps 2 --eval-interval 1 --eval-iters 1 --warmup-steps 0 --decay-steps 2 --result-dir "${RESULT_ROOT}/sft/smoke" --experiment-name evo2-smoke
    monitored '12,000-step SFT' "${sft}/train.log" torchrun --nproc-per-node "${NUM_GPUS}" --no-python train_evo2 "${model[@]}" --dataset-config "${prep}/training_dataset.yaml" --finetune-ckpt-dir "${base_mbridge}" --global-batch-size 32 --max-steps 12000 --eval-interval 400 --eval-iters 4 --lr 1e-5 --min-lr 1e-6 --warmup-steps 600 --decay-steps 11400 --enable-preemption --keep-best-k 3 --most-recent-k 1 --checkpoint-metric-name 'lm loss' --strict-checkpoint-metric --checkpoint-metric-step-tolerance 1 --result-dir "${sft}" --experiment-name evo2 "${SFT_WANDB_ARGS[@]}"
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/20-sft.done"
  fi
  [[ "${DRY_RUN}" == "1" ]] && selected='<selected-sft>' || selected="$(select_checkpoint sft "${sft}/evo2/tb_logs" "${sft}/evo2/checkpoints" "${RESULT_ROOT}/sft/checkpoint-selection.json")"
  state selected-sft "${selected}"
  # Bridge validates the scheduler before entering its zero-update evaluation path, so its inert decay length must be positive.
  monitored 'held-out SFT evaluation' "${RESULT_ROOT}/sft/heldout.log" torchrun --nproc-per-node "${NUM_GPUS}" --no-python train_evo2 "${model[@]}" --dataset-config "${prep}/heldout_dataset.yaml" --finetune-ckpt-dir "${selected}" --global-batch-size 20 --max-steps 0 --eval-interval 1 --eval-iters 5 --warmup-steps 0 --decay-steps 1 --result-dir "${RESULT_ROOT}/sft/heldout" --experiment-name evo2-heldout
}

stage_30() {
  local selected calibration="${RESULT_ROOT}/calibration" evidence
  selected="$(read_state selected-sft)"; evidence="${calibration}/scoring/selection-evidence.csv"
  if [[ -f "${STAGE_DIR}/30-calibration-generation.done" ]]; then
    note 'substage 30-calibration-generation already complete'
  else
    monitored 'calibration generation' "${calibration}/generation.log" env SOURCE_ENV=0 RUN_ROOT="${calibration}/generation" CKPT_DIR="${selected}" PROMPT_LENGTHS='0 1 2 4 6 8 10 12 16 24 32' TEMPERATURES='0.3 0.5 0.7 0.9 1.0 1.1 1.3' NUM_PROMPTS=64 TARGET_LENGTH=6000 GPU_IDS="${GPU_IDS}" TENSOR_PARALLEL_SIZE=1 HOPPER_FP8_INFERENCE="${HOPPER_FP8_INFERENCE}" scripts/calibration/run_sft_sampling_sweep.sh
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/30-calibration-generation.done"
  fi
  if [[ -f "${STAGE_DIR}/30-calibration-scoring.done" ]]; then
    note 'substage 30-calibration-scoring already complete'
  else
    run evo2_phage_prepare_arc_pipeline --output-dir data/arc_pipeline_patched --overwrite
    monitored 'calibration scoring' "${calibration}/scoring.log" env SOURCE_ENV=0 CALIBRATION_ROOT="${calibration}" GENERATION_ROOT="${calibration}/generation" ARC_CONFIG="${RECIPE_ROOT}/configs/arc_genome_design_filtering_local.yaml" PIPELINE_SCRIPT="${RECIPE_ROOT}/data/arc_pipeline_patched/genome_design_filtering_pipeline.py" TOOL_BIN_DIR="${RECIPE_ROOT}/data/external/bin" REFERENCE_FASTA="${RECIPE_ROOT}/data/external/arc_evo2/phage_gen/data/NC_001422_1.fna" SFT_FASTA="${RESULT_ROOT}/sft/source-safety/partitions/pass.fasta" WORKERS="${CALIBRATION_WORKERS}" scripts/calibration/run_sampling_calibration_scoring.sh
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/30-calibration-scoring.done"
  fi
  if [[ "${CALIBRATE_ONLY}" == "1" ]]; then
    note "calibration review requested; inspect ${evidence}, write a reviewed sampling-selection YAML, then rerun without --calibrate-only"
    return
  fi
  if [[ -s "${SAMPLING_SELECTION}" ]]; then
    if [[ "${SAMPLING_SELECTION_OVERRIDDEN}" == "1" ]]; then
      note "using explicit sampling selection: ${SAMPLING_SELECTION}"
    else
      note "using existing sampling selection and skipping the fresh-calibration default check: ${SAMPLING_SELECTION}"
    fi
  elif [[ "${DRY_RUN}" == "1" ]]; then
    note 'verify fresh calibration supports the bundled default, unless calibration/sampling-selection.yaml exists'
  else
    python - "${evidence}" "${DEFAULT_SAMPLING_SELECTION}" <<'PY'
import sys

import pandas as pd
import yaml

table = pd.read_csv(sys.argv[1])
selection = yaml.safe_load(open(sys.argv[2]))
chosen = table[
    (table.temperature == selection["temperature"])
    & table.prefix_length.isin(selection["prompt_lengths"])
]
supported = (
    len(chosen) == len(selection["prompt_lengths"])
    and chosen[["eligible", "metric_environment_ok", "temperature_1_default_candidate"]].to_numpy().all()
)
if not supported:
    raise SystemExit(
        "fresh calibration does not support the bundled sampling selection; inspect "
        f"{sys.argv[1]}, then create calibration/sampling-selection.yaml or rerun with "
        "--sampling-selection PATH"
    )
PY
    cp -- "${DEFAULT_SAMPLING_SELECTION}" "${SAMPLING_SELECTION}.tmp.$$"
    mv -- "${SAMPLING_SELECTION}.tmp.$$" "${SAMPLING_SELECTION}"
    note "fresh calibration supports the bundled default; wrote ${SAMPLING_SELECTION}"
  fi
  load_sampling_selection
  note "sampling selection: temperature=${SAMPLING_TEMPERATURE}, prompt lengths=${SAMPLING_PROMPT_LENGTHS_TEXT}, max new tokens=${SAMPLING_MAX_NEW_TOKENS}"
  run evo2_phage_generation write-rl-prompts --output "${RESULT_ROOT}/rl/train.jsonl" \
    --prompt-lengths "${SAMPLING_PROMPT_LENGTHS[@]}" --num-records "${SAMPLING_TRAIN_RECORDS}" \
    --id-prefix train
  run evo2_phage_generation write-rl-prompts --output "${RESULT_ROOT}/rl/validation.jsonl" \
    --prompt-lengths "${SAMPLING_PROMPT_LENGTHS[@]}" --num-records 96 \
    --id-prefix validation
}

stage_40() {
  local selected rl="${RESULT_ROOT}/rl" control="${RESULT_ROOT}/rl/environment-control" chosen
  local prepared_sft_root="${RESULT_ROOT}/rl/sft-checkpoint" rl_checkpoint
  load_sampling_selection
  if [[ -f "${STAGE_DIR}/40-rl.done" ]]; then
    note 'substage 40-rl already complete'
  else
    selected="$(read_state selected-sft)"
    # Use the editable source module so a post-calibration rerun works before console-script metadata is refreshed.
    run python -m bionemo.evo2_phage_gen.prepare_sft_checkpoint_for_rl \
      --source-checkpoint "${selected}" --output-dir "${prepared_sft_root}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      rl_checkpoint='<rl-sft-checkpoint>'
    else
      rl_checkpoint="$(python - "${prepared_sft_root}/preparation-manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest.get("state") != "succeeded":
    raise SystemExit(f"SFT checkpoint preparation for RL is not complete: {sys.argv[1]}")
print(manifest["prepared_sft_checkpoint"])
PY
)"
    fi
    state rl-sft-checkpoint "${rl_checkpoint}"
    note "RL will use the prepared optimizer-free, runtime-sanitized SFT checkpoint: ${rl_checkpoint}"
    export NEMO_RL_RAY_NUM_CPUS="${NUM_CPUS}"
    note "RL Ray CPU slots: ${NEMO_RL_RAY_NUM_CPUS}; reward phases use at most 64 threads"
    run pytest -q tests/bionemo/evo2_phage_gen/test_reward.py tests/bionemo/evo2_phage_gen/test_nemo_rl_env.py tests/bionemo/evo2_phage_gen/test_reference_controls.py
    monitored 'RL environment control' "${control}/runner.log" \
      evo2_phage_check_rl --config configs/gdpo_phage_megatron.yaml --checkpoint "${rl_checkpoint}" \
      --prompt-data "${rl}/train.jsonl" --gpus-per-node "${NUM_GPUS}" \
      --control-fasta data/external/arc_evo2/phage_gen/data/NC_001422_1.fna --control-dir "${control}"
    local common=(checkpointing.pretrained_checkpoint.path="${rl_checkpoint}" policy.model_name="${RL_MODEL_NAME}" data.train.data_path="${rl}/train.jsonl" data.validation.data_path="${rl}/validation.jsonl" cluster.gpus_per_node="${NUM_GPUS}" policy.generation.max_new_tokens="${SAMPLING_MAX_NEW_TOKENS}" policy.generation.temperature="${SAMPLING_TEMPERATURE}" policy.generation.top_k="${SAMPLING_TOP_K}" policy.generation.top_p="${SAMPLING_TOP_P}" policy.generation.mcore_generation_config.generation_adapter_config.seed="${SAMPLING_RL_SEED}" policy.generation.mcore_generation_config.generation_adapter_config.seed_stride="${SAMPLING_SEED_STRIDE}")
    if [[ -f "${STAGE_DIR}/40-pilot.done" ]]; then
      note 'substage 40-pilot already complete'
    else
      monitored 'one-step GDPO pilot' "${RESULT_ROOT}/rl-pilot/runner.log" evo2_phage_run_gdpo --config configs/gdpo_phage_megatron.yaml "${common[@]}" logger.wandb_enabled=false checkpointing.checkpoint_dir="${RESULT_ROOT}/rl-pilot/checkpoints" checkpointing.save_period=1 grpo.max_num_steps=1 grpo.val_at_end=true env.phage_qc.external_qc.work_dir="${RESULT_ROOT}/rl-pilot/external-qc" env.phage_qc.mmseqs_cluster_diversity.work_dir="${RESULT_ROOT}/rl-pilot/mmseqs" env.phage_qc.sequence_safety.work_dir="${RESULT_ROOT}/rl-pilot/safety" logger.log_dir="${RESULT_ROOT}/rl-pilot/logs"
      [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/40-pilot.done"
    fi
    if [[ -f "${STAGE_DIR}/40-pilot-check.done" ]]; then
      note 'substage 40-pilot-check already complete'
    else
      run evo2_phage_monitor_objectives --tensorboard-root "${RESULT_ROOT}/rl-pilot/logs" --config configs/gdpo_phage_megatron.yaml --minimum-events 1 --output "${RESULT_ROOT}/rl-pilot/objective-health.json"
      check_objectives "${RESULT_ROOT}/rl-pilot/objective-health.json"
      [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/40-pilot-check.done"
    fi
    monitored "500-step DP${NUM_GPUS} GDPO" "${rl}/runner.log" evo2_phage_run_gdpo --config configs/gdpo_phage_megatron.yaml "${common[@]}" "${RL_WANDB_ARGS[@]}" checkpointing.checkpoint_dir="${rl}/checkpoints" env.phage_qc.external_qc.work_dir="${rl}/external-qc" env.phage_qc.mmseqs_cluster_diversity.work_dir="${rl}/mmseqs" env.phage_qc.sequence_safety.work_dir="${rl}/safety" logger.log_dir="${rl}/logs"
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/40-rl.done"
  fi
  run evo2_phage_monitor_objectives --tensorboard-root "${rl}/logs" --config configs/gdpo_phage_megatron.yaml --output "${rl}/objective-health.json" --history-output "${rl}/objective-history.json"
  check_objectives "${rl}/objective-health.json"
  [[ "${DRY_RUN}" == "1" ]] && chosen='<selected-rl>' || chosen="$(select_checkpoint rl "${rl}/logs" "${rl}/checkpoints" "${rl}/checkpoint-selection.json")"
  state selected-rl "${chosen}"
}

stage_50() {
  local selected selected_sft rollout="${RESULT_ROOT}/rollout" fasta safety likelihood evidence infer
  local dedup target diagnostic hard_qc clustering
  local checkv_db repo_root
  load_sampling_selection
  selected="$(read_state selected-rl)"
  selected_sft="$(read_state selected-sft)"
  fasta="${rollout}/fasta/phix174_prompt${SAMPLING_PROMPT_LABEL}_temp${SAMPLING_TEMPERATURE}.n1000.fasta"
  safety="${rollout}/sequence-safety"
  likelihood="${rollout}/sft-likelihood"
  dedup="${rollout}/deduplication"
  target="${rollout}/target-profile"
  diagnostic="${rollout}/filter7-diagnostic"
  hard_qc="${rollout}/hard-qc"
  clustering="${rollout}/post-qc-clustering"
  infer="${RECIPE_ROOT}/src/bionemo/evo2/run/infer.py"
  if [[ -f "${STAGE_DIR}/50-rollout.done" ]]; then
    note 'substage 50-rollout already complete'
    if [[ "${DRY_RUN}" != "1" && ! -s "${fasta}" ]]; then
      printf 'rollout substage is marked complete but FASTA is missing: %s\n' "${fasta}" >&2
      return 1
    fi
  else
    run evo2_phage_generation write-prompts --output-dir "${rollout}/prompts" \
      --prompt-lengths "${SAMPLING_PROMPT_LENGTHS[@]}" --num-prompts "${SAMPLING_FINAL_PER_LENGTH}" \
      --id-prefix final
  local shard_dir="${rollout}/prompts/dp${NUM_GPUS}" rank started waited alive failed=0 printable prompt_length
  local worker_count wave_start wave_end wave_size gpu_index
  local -a command=() outputs=() pids=() logs=() prompt_files=()
  for prompt_length in "${SAMPLING_PROMPT_LENGTHS[@]}"; do
    prompt_files+=("${rollout}/prompts/final_prompt${prompt_length}_${SAMPLING_FINAL_PER_LENGTH}.jsonl")
  done
  worker_count="${NUM_GPUS}"
  note "interleave prompt lengths (${SAMPLING_PROMPT_LENGTHS_TEXT}) across ${worker_count} deterministic mixed-length shard(s)"
  run evo2_phage_generation write-inference-shards --input-jsonl "${prompt_files[@]}" \
    --output-dir "${shard_dir}" --num-records 1000 --num-shards "${worker_count}"
  started=${SECONDS}
  for ((wave_start=0; wave_start<worker_count; wave_start+=NUM_GPUS)); do
    wave_end="$((wave_start + NUM_GPUS))"
    ((wave_end > worker_count)) && wave_end="${worker_count}"
    wave_size="$((wave_end - wave_start))"
    pids=()
    for ((rank=wave_start; rank<wave_end; rank++)); do
      gpu_index="$((rank - wave_start))"
      outputs[rank]="${rollout}/jsonl/dp${rank}.jsonl"
      logs[rank]="${rollout}/logs/dp${rank}.log"
      command=(env CUDA_VISIBLE_DEVICES="${gpu_index}" torchrun --nproc_per_node 1 --nnodes 1 \
        --master_port "$((29544 + gpu_index))" "${infer}" --ckpt-dir "${selected}" \
        --prompt-file "${shard_dir}/dp${rank}.jsonl" --max-new-tokens "${SAMPLING_MAX_NEW_TOKENS}" \
        --temperature "${SAMPLING_TEMPERATURE}" --top-k "${SAMPLING_TOP_K}" \
        --top-p "${SAMPLING_TOP_P}" --seed "$((SAMPLING_ROLLOUT_SEED + rank * SAMPLING_SEED_STRIDE))" \
        --tensor-parallel-size 1 \
        --max-seq-length 10240 --prompt-batch-size 16 --inference-backend dynamic \
        ${INFERENCE_PRECISION_ARGS[@]+"${INFERENCE_PRECISION_ARGS[@]}"} \
        --ignore-eos --strict-generation --stream-output \
        --output-file "${outputs[rank]}")
      printf -v printable '%q ' "${command[@]}"; note "command: ${printable}"
      if [[ "${DRY_RUN}" != "1" ]]; then
        mkdir -p "$(dirname -- "${outputs[rank]}")" "$(dirname -- "${logs[rank]}")"
        "${command[@]}" > "${logs[rank]}" 2>&1 & pids[rank]="$!"
      fi
    done
    if [[ "${DRY_RUN}" != "1" ]]; then
      while :; do
        alive=0
        for ((rank=wave_start; rank<wave_end; rank++)); do
          kill -0 "${pids[rank]}" 2>/dev/null && alive=$((alive + 1))
        done
        ((alive == 0)) && break
        waited=0
        while ((waited < MONITOR_INTERVAL_SECONDS && alive > 0)); do
          sleep 10; waited=$((waited + 10)); alive=0
          for ((rank=wave_start; rank<wave_end; rank++)); do
            kill -0 "${pids[rank]}" 2>/dev/null && alive=$((alive + 1))
          done
        done
        ((alive > 0)) && note "generation wave: ${alive}/${wave_size} workers still running after $((SECONDS - started))s"
      done
      for ((rank=wave_start; rank<wave_end; rank++)); do
        if ! wait "${pids[rank]}"; then tail -n 30 "${logs[rank]}" >&2; failed=1; fi
      done
    fi
    ((failed == 0)) || return 1
  done
  if [[ "${DRY_RUN}" != "1" ]]; then
    python - "${outputs[@]}" <<'PY'
import json, sys
seen = set()
for path in sys.argv[1:]:
    records = [json.loads(line) for line in open(path) if line.strip()]
    for record in records:
        if record["id"] in seen:
            raise SystemExit(f'duplicate generated ID: {record["id"]}')
        seen.add(record["id"])
if len(seen) != 1000:
    raise SystemExit(f"expected 1000 generated records, found {len(seen)}")
PY
  fi
  run evo2_phage_generation jsonl-to-fasta --input-jsonl "${outputs[@]}" --output-fasta "${fasta}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    python - "${fasta}" <<'PY'
from Bio import SeqIO
import sys
count = sum(1 for _ in SeqIO.parse(sys.argv[1], "fasta"))
if count != 1000:
    raise SystemExit(f"expected exactly 1000 generated genomes, found {count}")
PY
  fi
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/50-rollout.done"
  fi

  if [[ -f "${STAGE_DIR}/50-deduplication.done" ]]; then
    note 'substage 50-deduplication already complete'
    require_nonempty_file "${dedup}/representatives.fasta" 'deduplicated representative FASTA'
    require_nonempty_file "${dedup}/mapping.csv" 'deduplication mapping'
    check_success_report "${dedup}/report.json"
  else
    run evo2_phage_generation deduplicate-fasta \
      --input-fasta "${fasta}" --output-fasta "${dedup}/representatives.fasta" \
      --mapping-csv "${dedup}/mapping.csv" --report "${dedup}/report.json"
    check_success_report "${dedup}/report.json"
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/50-deduplication.done"
  fi

  if [[ -f "${STAGE_DIR}/50-sft-likelihood.done" ]]; then
    note 'substage 50-sft-likelihood already complete'
    require_nonempty_file "${likelihood}/ranked-designs.csv" 'raw SFT likelihood table'
  else
    run evo2_phage_generation prepare-sft-likelihood \
      --input-fasta "${fasta}" --output-fasta "${likelihood}/sft-conditioned.fasta"
    monitored 'selected-SFT likelihood scoring' "${likelihood}/predict.log" \
      torchrun --nproc-per-node "${NUM_GPUS}" --no-python predict_evo2 \
      --fasta "${likelihood}/sft-conditioned.fasta" --ckpt-dir "${selected_sft}" \
      --output-dir "${likelihood}/predictions" --tensor-parallel-size 1 --micro-batch-size 8 \
      ${INFERENCE_PRECISION_ARGS[@]+"${INFERENCE_PRECISION_ARGS[@]}"} \
      --output-log-prob-seqs --log-prob-collapse-option per_token
    run evo2_phage_generation collect-sft-likelihood \
      --prediction-dir "${likelihood}/predictions" --source-fasta "${fasta}" \
      --output-csv "${likelihood}/ranked-designs.csv"
    require_nonempty_file "${likelihood}/ranked-designs.csv" 'raw SFT likelihood table'
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/50-sft-likelihood.done"
  fi

  if [[ -f "${STAGE_DIR}/50-sequence-safety.done" ]]; then
    note 'substage 50-sequence-safety already complete'
    require_file "${safety}/scan/manifest.json" 'sequence-safety manifest'
    require_file "${safety}/summary.json" 'sequence-safety summary'
    check_scan "${safety}/scan/manifest.json" allow-no-primary-gene-candidates
  else
    run evo2_phage_nucleotide_qc --input-fasta "${dedup}/representatives.fasta" --output-dir "${safety}/input-qc" \
      --genome-length-min 1 --genome-length-max 1000000 --gc-content-min 0 --gc-content-max 100 \
      --homopolymer-max 1000000
    evidence='{"source":"NCBI PhiX174 reference","source_version":"NC_001422.1","replication_host_domains":["BACTERIA"],"confirmed":true}'
    run_result 'final safety scan' "${safety}/scan.log" evo2_phage_sequence_safety scan \
      --input-fasta "${safety}/input-qc/qc2_nt_filter_seqs.fasta" --output-dir "${safety}/scan" \
      --policy configs/phage_safety_policy.yaml --asset-manifest data/external/safety/asset_manifest.yaml \
      --host-domain BACTERIA --host-evidence-json "${evidence}" --strict-lysis \
      --batch-size "${SAFETY_BATCH_SIZE}" --orf-workers "${SAFETY_ORF_WORKERS}" \
      --threads "${SAFETY_THREADS}" --phrogs-threads "${SAFETY_PHROGS_THREADS}" --timeout 1800 --overwrite
    check_scan "${safety}/scan/manifest.json" allow-no-primary-gene-candidates
    run evo2_phage_summarize_safety_manifest --manifest "${safety}/scan/manifest.json" --output "${safety}/summary.json"
    require_file "${safety}/summary.json" 'sequence-safety summary'
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/50-sequence-safety.done"
  fi

  if [[ ! -f "${STAGE_DIR}/50-target-profile.done" || ! -f "${STAGE_DIR}/50-filter7-diagnostic.done" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      checkv_db='<prepared-checkv-db>'
    elif [[ -n "${CHECKVDB:-}" ]]; then
      [[ -d "${CHECKVDB}" ]] || {
        printf 'CHECKVDB is not a directory: %s\n' "${CHECKVDB}" >&2
        return 1
      }
      checkv_db="$(cd -- "${CHECKVDB}" && pwd)"
    else
      checkv_db="$(python - "${RECIPE_ROOT}/data/external/checkv" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
databases = sorted(path.resolve() for path in root.glob("checkv-db-*") if path.is_dir())
if not databases:
    raise SystemExit(f"no prepared CheckV database found under {root}")
print(databases[-1])
PY
)"
    fi
    export CHECKVDB="${checkv_db}"
    repo_root="$(cd -- "${RECIPE_ROOT}/../.." && pwd)"
    note "Arc CheckV database: ${CHECKVDB}"
    note "Arc screening working directory: ${repo_root}"
    note 'Arc internal MMseqs clustering disabled; final 99% clustering runs only after safety and hard QC'
  fi

  write_arc_rollout_config() {
    local branch_root="$1" remove_filter="$2"
    if [[ "${DRY_RUN}" == "1" ]]; then
      note "prepare ${branch_root##*/} Arc config from the maintained local template (filter 7=${remove_filter})"
      return
    fi
    python - configs/arc_genome_design_filtering_local.yaml "${dedup}/representatives.fasta" \
      "${branch_root}" "${remove_filter}" <<'PY'
from pathlib import Path
import sys
import yaml

base, fasta, output = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
remove_filter = sys.argv[4] == "true"
config = yaml.safe_load(base.read_text())
config.update({
    "results_save_dir": str(output / "arc"),
    "current_config_file": str(output / "config.yaml"),
    "evo_gen_seqs_fasta_file_save_location": str(fasta),
    "orf_filtering": True,
    "use_nucleotide_filtered_df": True,
    "homology_filtering": True,
    "use_orf_filtered_df": True,
    "use_nucleotide_filtered_df_instead": False,
    "checkv_filter": True,
    "genetic_architecture_filter": True,
    "diversification_filtering": True,
    "mmseqs_clustering_filter": False,
    "genetic_architecture_remove_filter": remove_filter,
    "genetic_architecture_visualization_and_synteny_filtering": True,
    "use_reference_genome": True,
})
output.mkdir(parents=True, exist_ok=True)
(output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
  }

  if [[ -f "${STAGE_DIR}/50-target-profile.done" ]]; then
    note 'substage 50-target-profile already complete'
    require_file "${target}/arc/qc6_synteny_filter_seqs.fasta" 'terminal target-profile FASTA'
    check_success_report "${target}/screening.json"
  else
    write_arc_rollout_config "${target}" false
    (
      cd "${repo_root}"
      monitored 'Arc target profile' "${target}/runner.log" \
        python "${RECIPE_ROOT}/data/arc_pipeline_patched/genome_design_filtering_pipeline.py" \
        "${target}/config.yaml"
    )
    run evo2_phage_generation summarize-arc-screen --config "${target}/config.yaml" \
      --input-fasta "${dedup}/representatives.fasta" --output-json "${target}/screening.json" \
      --expected-filter7 false
    check_success_report "${target}/screening.json"
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/50-target-profile.done"
  fi

  if [[ -f "${STAGE_DIR}/50-filter7-diagnostic.done" ]]; then
    note 'substage 50-filter7-diagnostic already complete'
    require_file "${diagnostic}/arc/qc6_synteny_filter_seqs.fasta" 'terminal filter-7 diagnostic FASTA'
    check_success_report "${diagnostic}/screening.json"
  else
    write_arc_rollout_config "${diagnostic}" true
    (
      cd "${repo_root}"
      monitored 'Arc filter-7 diagnostic' "${diagnostic}/runner.log" \
        python "${RECIPE_ROOT}/data/arc_pipeline_patched/genome_design_filtering_pipeline.py" \
        "${diagnostic}/config.yaml"
    )
    run evo2_phage_generation summarize-arc-screen --config "${diagnostic}/config.yaml" \
      --input-fasta "${dedup}/representatives.fasta" --output-json "${diagnostic}/screening.json" \
      --expected-filter7 true
    check_success_report "${diagnostic}/screening.json"
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/50-filter7-diagnostic.done"
  fi

  if [[ -f "${STAGE_DIR}/50-final-clustering.done" ]]; then
    note 'substage 50-final-clustering already complete'
    require_file "${hard_qc}/passers.fasta" 'hard-QC passer FASTA'
    check_success_report "${hard_qc}/report.json"
    require_file "${clustering}/representatives.fasta" 'post-QC cluster representative FASTA'
    require_nonempty_file "${clustering}/memberships.csv" 'post-QC cluster membership table'
    check_success_report "${clustering}/report.json"
  else
    run evo2_phage_generation select-hard-qc-passers \
      --representative-fasta "${dedup}/representatives.fasta" \
      --safety-input-fasta "${safety}/input-qc/qc2_nt_filter_seqs.fasta" \
      --safety-manifest "${safety}/scan/manifest.json" \
      --target-fasta "${target}/arc/qc6_synteny_filter_seqs.fasta" \
      --output-fasta "${hard_qc}/passers.fasta" --report "${hard_qc}/report.json"
    run evo2_phage_generation cluster-post-qc --input-fasta "${hard_qc}/passers.fasta" \
      --output-fasta "${clustering}/representatives.fasta" \
      --memberships-csv "${clustering}/memberships.csv" --report "${clustering}/report.json" \
      --work-dir "${clustering}/work" --mmseqs-bin data/external/bin/mmseqs --threads 16
    check_success_report "${hard_qc}/report.json"
    check_success_report "${clustering}/report.json"
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/50-final-clustering.done"
  fi

  if [[ -f "${STAGE_DIR}/50-report.done" ]]; then
    note 'substage 50-report already complete'
    check_success_report "${rollout}/final-designs.json"
    require_file "${rollout}/accepted_candidates.fasta" 'accepted candidate FASTA'
    require_nonempty_file "${RESULT_ROOT}/SUMMARY.md" 'run summary'
  else
    note 'record the raw and representative denominators, hard-QC waterfall, post-QC clusters, and limitations'
    run evo2_phage_generation finalize-rollout \
      --generated-fasta "${fasta}" --deduplication-mapping "${dedup}/mapping.csv" \
      --safety-input-fasta "${safety}/input-qc/qc2_nt_filter_seqs.fasta" \
      --safety-manifest "${safety}/scan/manifest.json" \
      --target-fasta "${target}/arc/qc6_synteny_filter_seqs.fasta" \
      --diagnostic-fasta "${diagnostic}/arc/qc6_synteny_filter_seqs.fasta" \
      --likelihood-csv "${likelihood}/ranked-designs.csv" \
      --cluster-representatives-fasta "${clustering}/representatives.fasta" \
      --cluster-memberships "${clustering}/memberships.csv" \
      --output-json "${rollout}/final-designs.json" \
      --accepted-fasta "${rollout}/accepted_candidates.fasta" --summary "${RESULT_ROOT}/SUMMARY.md" \
      --model-checkpoint "${selected_sft}" --rl-checkpoint "${selected}" \
      --sampling-selection "${SAMPLING_SELECTION}" --deduplication-report "${dedup}/report.json" \
      --hard-qc-report "${hard_qc}/report.json" --target-report "${target}/screening.json" \
      --diagnostic-report "${diagnostic}/screening.json" --clustering-report "${clustering}/report.json" \
      --run-log "${RUNLOG}"
    check_success_report "${rollout}/final-designs.json"
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/50-report.done"
  fi
}

printf '%s\n' '00 prepare inputs/tools/controls' '10 safety-screen and prepare SFT' '20 train/select/evaluate SFT' '30 calibrate sampling' '40 prepare SFT checkpoint for RL; pilot/check/train/monitor/select GDPO' '50 generate, deduplicate, SFT-score, hard-QC, cluster, and report 1,000 genomes' > "${RESULT_ROOT}/stage-plan.txt"
for id in 00 10 20 30 40 50; do
  ((10#${id} < 10#${RESUME_FROM})) && continue
  [[ "${PREPARE_ONLY}" == "1" && "${id}" != 00 ]] && continue
  [[ "${CALIBRATE_ONLY}" == "1" ]] && ((10#${id} > 30)) && continue
  set_current_stage "${id}"
  [[ -f "${STAGE_DIR}/${id}.done" ]] && { note "stage ${id} already complete"; continue; }
  note "starting stage ${id}"; "stage_${id}"
  [[ "${CALIBRATE_ONLY}" == "1" && "${id}" == 30 ]] && continue
  [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/${id}.done"
done
if [[ "${CALIBRATE_ONLY}" == "1" ]]; then
  note 'RUN PAUSED after step 4/6 (stage 30: calibrate sampling); review the calibration evidence and provide a sampling selection before RL'
elif [[ "${DRY_RUN}" == "1" ]]; then
  note 'RUN COMPLETE: dry run finished; 6/6 steps planned'
elif [[ "${PREPARE_ONLY}" == "1" ]]; then
  note 'RUN COMPLETE: preparation request finished; step 1/6 complete'
else
  completed="$(completed_stage_count)"
  if [[ "${completed}" == "${TOTAL_STAGES}" ]]; then
    note "RUN COMPLETE: ${completed}/${TOTAL_STAGES} steps complete; results: ${RESULT_ROOT}"
  else
    note "RUN COMPLETE: requested stages finished; ${completed}/${TOTAL_STAGES} step markers present; results: ${RESULT_ROOT}"
  fi
fi
RUN_COMPLETION_REPORTED=1
