#!/usr/bin/env bash
# Run the Evo2 Microviridae prompt/temperature generation sweep.
#
# Default output layout:
#   recipes/evo2_phage_gen/data/checkpoints/generation/paper_hpo_batched_20260629/
#     prompts/
#     jsonl/
#     logs/
#     hpo_generation_manifest.tsv
#
# The script is resumable: a cell with at least NUM_PROMPTS JSONL records is skipped.
#
# By default this runs the paper's useful prompt/temperature region:
#   PROMPT_LENGTHS="4 5 6 7 8 9"
#   TEMPERATURES="0.7 0.9"
#
# To run the full 55-cell paper grid instead:
#   PROMPT_LENGTHS="1 2 3 4 5 6 7 8 9 10 11" \
#   TEMPERATURES="0.3 0.5 0.7 0.9 1.1" \
#   recipes/evo2_phage_gen/scripts/run_paper_hpo_generation.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/evo2_phage_gen"

if [[ "${SOURCE_ENV:-1}" == "1" ]]; then
  # shellcheck source=/dev/null
  source "${RECIPE_DIR}/.ci_test_env.sh"
fi

RUN_NAME="${RUN_NAME:-paper_hpo_batched_20260629}"
RUN_ROOT="${RUN_ROOT:-${RECIPE_DIR}/data/checkpoints/generation/${RUN_NAME}}"
CKPT_DIR="${CKPT_DIR:-${RECIPE_DIR}/data/checkpoints/evo2_7b_microviridae_mbridge}"

PROMPT_LENGTHS="${PROMPT_LENGTHS:-4 5 6 7 8 9}"
TEMPERATURES="${TEMPERATURES:-0.7 0.9}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
TARGET_LENGTH="${TARGET_LENGTH:-6000}"
ID_PREFIX="${ID_PREFIX:-phix174_hpo}"

TOP_K="${TOP_K:-4}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MASTER_PORT="${MASTER_PORT:-29641}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-10240}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-64}"
OVERWRITE="${OVERWRITE:-0}"

PROMPT_DIR="${RUN_ROOT}/prompts"
JSONL_DIR="${RUN_ROOT}/jsonl"
LOG_DIR="${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/hpo_generation_manifest.tsv"

mkdir -p "${PROMPT_DIR}" "${JSONL_DIR}" "${LOG_DIR}"

cd "${REPO_ROOT}"

count_jsonl_records() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    printf '0\n'
    return
  fi
  python - "$path" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
print(sum(1 for line in path.open() if line.startswith("{")))
PY
}

printf 'Writing prompt files under %s\n' "${PROMPT_DIR}"
evo2_phage_generation write-prompts \
  --output-dir "${PROMPT_DIR}" \
  --prompt-lengths ${PROMPT_LENGTHS} \
  --num-prompts "${NUM_PROMPTS}" \
  --id-prefix "${ID_PREFIX}"

if [[ ! -s "${MANIFEST}" || "${OVERWRITE}" == "1" ]]; then
  printf 'prompt_len\ttemperature\tstatus\trecords\tjsonl\tlog\tstarted_at_utc\tfinished_at_utc\n' > "${MANIFEST}"
fi

RUNNER_LOG="${LOG_DIR}/hpo_runner.log"
printf 'Started paper HPO generation sweep at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUNNER_LOG}"
printf 'run_root=%s\nckpt_dir=%s\nprompt_batch_size=%s\nmax_seq_length=%s\n' \
  "${RUN_ROOT}" "${CKPT_DIR}" "${PROMPT_BATCH_SIZE}" "${MAX_SEQ_LENGTH}" | tee -a "${RUNNER_LOG}"

for temp in ${TEMPERATURES}; do
  for prompt_len in ${PROMPT_LENGTHS}; do
    prompt_file="${PROMPT_DIR}/${ID_PREFIX}_prompt${prompt_len}_${NUM_PROMPTS}.jsonl"
    out="${JSONL_DIR}/phix174_prompt${prompt_len}_temp${temp}.jsonl"
    log="${LOG_DIR}/phix174_prompt${prompt_len}_temp${temp}.infer.log"
    max_new_tokens=$((TARGET_LENGTH - prompt_len))

    if (( max_new_tokens <= 0 )); then
      printf 'ERROR: TARGET_LENGTH=%s must exceed prompt_len=%s\n' "${TARGET_LENGTH}" "${prompt_len}" >&2
      exit 2
    fi

    existing_records="$(count_jsonl_records "${out}")"
    if [[ "${OVERWRITE}" != "1" && "${existing_records}" -ge "${NUM_PROMPTS}" ]]; then
      now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf '[%s] SKIP prompt_len=%s temp=%s records=%s\n' "${now}" "${prompt_len}" "${temp}" "${existing_records}" | tee -a "${RUNNER_LOG}"
      printf '%s\t%s\tSKIP\t%s\t%s\t%s\t%s\t%s\n' \
        "${prompt_len}" "${temp}" "${existing_records}" "${out}" "${log}" "${now}" "${now}" >> "${MANIFEST}"
      continue
    fi

    if [[ "${OVERWRITE}" == "1" ]]; then
      rm -f "${out}" "${log}"
    fi

    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '\n[%s] START prompt_len=%s temp=%s out=%s\n' "${started_at}" "${prompt_len}" "${temp}" "${out}" | tee -a "${RUNNER_LOG}"

    set +e
    torchrun \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --nnodes 1 \
      --master_port "${MASTER_PORT}" \
      recipes/evo2_megatron/src/bionemo/evo2/run/infer.py \
      --ckpt-dir "${CKPT_DIR}" \
      --prompt-file "${prompt_file}" \
      --max-new-tokens "${max_new_tokens}" \
      --temperature "${temp}" \
      --top-k "${TOP_K}" \
      --top-p "${TOP_P}" \
      --seed "${SEED}" \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
      --max-seq-length "${MAX_SEQ_LENGTH}" \
      --prompt-batch-size "${PROMPT_BATCH_SIZE}" \
      --stream-output \
      --output-file "${out}" \
      > "${log}" 2>&1
    status=$?
    set -e

    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    records="$(count_jsonl_records "${out}")"
    printf '[%s] DONE prompt_len=%s temp=%s status=%s records=%s log=%s\n' \
      "${finished_at}" "${prompt_len}" "${temp}" "${status}" "${records}" "${log}" | tee -a "${RUNNER_LOG}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${prompt_len}" "${temp}" "${status}" "${records}" "${out}" "${log}" "${started_at}" "${finished_at}" >> "${MANIFEST}"

    if [[ "${status}" -ne 0 ]]; then
      printf '[%s] STOP after failure prompt_len=%s temp=%s\n' "${finished_at}" "${prompt_len}" "${temp}" | tee -a "${RUNNER_LOG}"
      exit "${status}"
    fi
  done
done

printf 'Completed paper HPO generation sweep at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUNNER_LOG}"
