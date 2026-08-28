#!/usr/bin/env bash
# Run a resumable selected-SFT temperature/prefix sweep across independent GPU replicas.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RECIPE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RECIPE_ROOT="${RECIPE_ROOT:-${DEFAULT_RECIPE_ROOT}}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
CKPT_DIR="${CKPT_DIR:?CKPT_DIR is required}"

PROMPT_LENGTHS="${PROMPT_LENGTHS:-0 1 2 4 6 8 10 12 16 24 32}"
TEMPERATURES="${TEMPERATURES:-0.3 0.5 0.7 0.9 1.0 1.1 1.3}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
TARGET_LENGTH="${TARGET_LENGTH:-6000}"
MARKER="${MARKER:-+~}"
TOP_K="${TOP_K:-4}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-7}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-16}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-10240}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29680}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CELL_TIMEOUT_SECONDS="${CELL_TIMEOUT_SECONDS:-7200}"
SOURCE_ENV="${SOURCE_ENV:-1}"
DRY_RUN="${DRY_RUN:-0}"
HOPPER_FP8_INFERENCE="${HOPPER_FP8_INFERENCE:-0}"

if [[ "${HOPPER_FP8_INFERENCE}" != "0" && "${HOPPER_FP8_INFERENCE}" != "1" ]]; then
  echo "HOPPER_FP8_INFERENCE must be 0 or 1" >&2
  exit 2
fi
declare -a CALIBRATION_PRECISION_ARGS=()
if [[ "${HOPPER_FP8_INFERENCE}" == "1" ]]; then
  CALIBRATION_PRECISION_ARGS=(--hopper-fp8)
fi

if [[ "${SOURCE_ENV}" == "1" ]]; then
  # shellcheck source=/dev/null
  source "${RECIPE_ROOT}/.ci_test_env.sh"
fi

read -r -a PREFIX_ARRAY <<< "${PROMPT_LENGTHS}"
read -r -a TEMPERATURE_ARRAY <<< "${TEMPERATURES}"
read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} == 0 || ${#GPU_ARRAY[@]} % TENSOR_PARALLEL_SIZE != 0 )); then
  echo "GPU count must be non-zero and divisible by TENSOR_PARALLEL_SIZE" >&2
  exit 2
fi
REPLICA_COUNT=$(( ${#GPU_ARRAY[@]} / TENSOR_PARALLEL_SIZE ))

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runtime"
python -m bionemo.evo2_phage_gen.sampling_calibration materialize \
  --run-root "${RUN_ROOT}" \
  --checkpoint "${CKPT_DIR}" \
  --prefix-lengths "${PREFIX_ARRAY[@]}" \
  --temperatures "${TEMPERATURE_ARRAY[@]}" \
  --num-prompts "${NUM_PROMPTS}" \
  --marker "${MARKER}" \
  --gpu-ids "${GPU_ARRAY[@]}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --target-length "${TARGET_LENGTH}" \
  --top-k "${TOP_K}" \
  --top-p "${TOP_P}" \
  --seed "${SEED}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  "${CALIBRATION_PRECISION_ARGS[@]}" \
  > "${RUN_ROOT}/logs/materialize.log"

if [[ "${DRY_RUN}" == "1" ]]; then
  touch "${RUN_ROOT}/DRY_RUN_COMPLETE"
  exit 0
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_DIR}" >&2
  exit 2
fi

INFER_SCRIPT="${RECIPE_ROOT}/src/bionemo/evo2/run/infer.py"
if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "Inference script not found: ${INFER_SCRIPT}" >&2
  exit 2
fi

run_worker() {
  local slot="$1"
  local visible_gpus="$2"
  local worker_manifest="${RUN_ROOT}/logs/worker-${slot}.tsv"
  local worker_log="${RUN_ROOT}/logs/worker-${slot}.log"
  printf 'cell\tattempt\tstatus\tfinished_at\n' > "${worker_manifest}"

  while IFS=$'\t' read -r -u 3 cell_index cell_key prefix_length temperature prompt_file output_file; do
    [[ "${cell_index}" == "index" ]] && continue
    (( cell_index % REPLICA_COUNT == slot )) || continue

    if [[ -s "${output_file}" ]] && python -m bionemo.evo2_phage_gen.sampling_calibration validate-cell \
      --output "${output_file}" --prompts "${prompt_file}" --expected-records "${NUM_PROMPTS}" \
      >> "${worker_log}" 2>&1; then
      printf '%s\t0\tSKIP\t%s\n' "${cell_key}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${worker_manifest}"
      continue
    fi
    if [[ -e "${output_file}" ]]; then
      mv "${output_file}" "${output_file}.invalid.$(date -u +%Y%m%dT%H%M%SZ)"
    fi

    local max_new_tokens=$(( TARGET_LENGTH - prefix_length ))
    if (( max_new_tokens <= 0 )); then
      echo "${cell_key}: TARGET_LENGTH=${TARGET_LENGTH} must exceed prefix_length=${prefix_length}" >&2
      return 2
    fi
    local -a inference_command=()
    mapfile -d '' -t inference_command < <(
      python -m bionemo.evo2_phage_gen.sampling_calibration print-command \
        --infer-script "${INFER_SCRIPT}" \
        --checkpoint "${CKPT_DIR}" \
        --prompt-file "${prompt_file}" \
        --output-file "${output_file}" \
        --prefix-length "${prefix_length}" \
        --temperature "${temperature}" \
        --target-length "${TARGET_LENGTH}" \
        --seed "${SEED}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --master-port "$(( MASTER_PORT_BASE + slot ))" \
        --prompt-batch-size "${PROMPT_BATCH_SIZE}" \
        --max-seq-length "${MAX_SEQ_LENGTH}" \
        --top-k "${TOP_K}" \
        --top-p "${TOP_P}" \
        "${CALIBRATION_PRECISION_ARGS[@]}"
    )
    if (( ${#inference_command[@]} == 0 )); then
      echo "${cell_key}: command builder returned no arguments" >&2
      return 2
    fi

    local attempt=0
    local succeeded=0
    while (( attempt <= MAX_RETRIES )); do
      local log="${RUN_ROOT}/logs/${cell_key}.attempt-${attempt}.log"
      local slot_runtime="${RUN_ROOT}/runtime/slot-${slot}"
      mkdir -p "${slot_runtime}/xdg/cache" "${slot_runtime}/xdg/config" "${slot_runtime}/matplotlib" "${slot_runtime}/flashinfer"
      if {
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start ${cell_key} attempt=${attempt} GPUs=${visible_gpus}"
        CUDA_VISIBLE_DEVICES="${visible_gpus}" \
        XDG_CACHE_HOME="${slot_runtime}/xdg/cache" \
        XDG_CONFIG_HOME="${slot_runtime}/xdg/config" \
        MPLCONFIGDIR="${slot_runtime}/matplotlib" \
        FLASHINFER_WORKSPACE_BASE="${slot_runtime}/flashinfer" \
        timeout --signal=TERM --kill-after=60s "${CELL_TIMEOUT_SECONDS}" \
          "${inference_command[@]}"
      } > "${log}" 2>&1; then
        local rc=0
      else
        local rc=$?
      fi
      if [[ "${rc}" == "0" ]] && python -m bionemo.evo2_phage_gen.sampling_calibration validate-cell \
        --output "${output_file}" --prompts "${prompt_file}" --expected-records "${NUM_PROMPTS}" \
        >> "${log}" 2>&1; then
        succeeded=1
        printf '%s\t%s\t0\t%s\n' "${cell_key}" "${attempt}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${worker_manifest}"
        break
      fi
      printf '%s\t%s\t%s\t%s\n' "${cell_key}" "${attempt}" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${worker_manifest}"
      attempt=$(( attempt + 1 ))
    done
    if [[ "${succeeded}" != "1" ]]; then
      echo "${cell_key} failed after bounded retries" >> "${worker_log}"
      return 1
    fi
  done 3< "${RUN_ROOT}/cells.tsv"
}

pids=()
for (( slot=0; slot<REPLICA_COUNT; slot++ )); do
  group=()
  for (( offset=0; offset<TENSOR_PARALLEL_SIZE; offset++ )); do
    group+=( "${GPU_ARRAY[$(( slot * TENSOR_PARALLEL_SIZE + offset ))]}" )
  done
  visible_gpus="$(IFS=,; echo "${group[*]}")"
  run_worker "${slot}" "${visible_gpus}" &
  pids+=( "$!" )
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "One or more sweep workers failed; rerun resumes completed cells." >&2
  exit 1
fi

python -m bionemo.evo2_phage_gen.sampling_calibration validate-all --run-root "${RUN_ROOT}" \
  > "${RUN_ROOT}/validation_summary.json"
touch "${RUN_ROOT}/SUCCEEDED"
