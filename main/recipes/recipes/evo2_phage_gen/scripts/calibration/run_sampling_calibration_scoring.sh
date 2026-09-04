#!/usr/bin/env bash
# Score a completed sweep independently per cell and measure target/SFT copy risk.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="${RECIPE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:?CALIBRATION_ROOT is required}"
GENERATION_ROOT="${GENERATION_ROOT:-${CALIBRATION_ROOT}/generation}"
ARC_CONFIG="${ARC_CONFIG:?ARC_CONFIG is required}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:?PIPELINE_SCRIPT is required}"
TOOL_BIN_DIR="${TOOL_BIN_DIR:?TOOL_BIN_DIR is required}"
REFERENCE_FASTA="${REFERENCE_FASTA:?REFERENCE_FASTA is required}"
SFT_FASTA="${SFT_FASTA:?SFT_FASTA is required}"
SCORE_ROOT="${SCORE_ROOT:-${CALIBRATION_ROOT}/scoring}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-64}"
WORKERS="${WORKERS:-8}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-12}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CELL_TIMEOUT_SECONDS="${CELL_TIMEOUT_SECONDS:-7200}"
NOVELTY_TIMEOUT_SECONDS="${NOVELTY_TIMEOUT_SECONDS:-7200}"
NOVELTY_THREADS="${NOVELTY_THREADS:-32}"
SOURCE_ENV="${SOURCE_ENV:-1}"

if [[ "${SOURCE_ENV}" == "1" ]]; then
  # shellcheck source=/dev/null
  source "${RECIPE_ROOT}/.ci_test_env.sh"
fi

mkdir -p "${SCORE_ROOT}"
[[ -f "${GENERATION_ROOT}/SUCCEEDED" ]] || {
  echo "Generation is not complete: ${GENERATION_ROOT}/SUCCEEDED is absent" >&2
  exit 2
}
python -m bionemo.evo2_phage_gen.sampling_calibration validate-all --run-root "${GENERATION_ROOT}" \
  > "${SCORE_ROOT}/generation-validation.json"

mkdir -p "${SCORE_ROOT}/csv" "${SCORE_ROOT}/logs" "${SCORE_ROOT}/work"

run_worker() {
  local slot="$1"
  local manifest="${SCORE_ROOT}/logs/worker-${slot}.tsv"
  printf 'cell\tattempt\tstatus\tfinished_at\n' > "${manifest}"

  while IFS=$'\t' read -r -u 3 cell_index cell _prefix _temperature _prompt_file generation_jsonl; do
    [[ "${cell_index}" == "index" ]] && continue
    (( cell_index % WORKERS == slot )) || continue

    local output_csv="${SCORE_ROOT}/csv/${cell}.scores.csv"
    if [[ -s "${output_csv}" ]] && python -m bionemo.evo2_phage_gen.calibration_scoring validate \
      --score-csv "${output_csv}" --expected-records "${EXPECTED_RECORDS}" >/dev/null 2>&1; then
      printf '%s\t0\tSKIP\t%s\n' "${cell}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${manifest}"
      continue
    fi

    local attempt=0
    local succeeded=0
    while (( attempt <= MAX_RETRIES )); do
      local partial="${output_csv}.partial"
      local log="${SCORE_ROOT}/logs/${cell}.attempt-${attempt}.log"
      if timeout --signal=TERM --kill-after=60s "${CELL_TIMEOUT_SECONDS}" \
        python -m bionemo.evo2_phage_gen.calibration_scoring score-cell \
          --generation-jsonl "${generation_jsonl}" \
          --output-csv "${partial}" \
          --arc-config "${ARC_CONFIG}" \
          --pipeline-script "${PIPELINE_SCRIPT}" \
          --work-dir "${SCORE_ROOT}/work/${cell}" \
          --tool-bin-dir "${TOOL_BIN_DIR}" \
          --threads "${THREADS_PER_WORKER}" \
          > "${log}" 2>&1 &&
        python -m bionemo.evo2_phage_gen.calibration_scoring validate \
          --score-csv "${partial}" --expected-records "${EXPECTED_RECORDS}" >> "${log}" 2>&1; then
        mv "${partial}" "${output_csv}"
        succeeded=1
        printf '%s\t%s\t0\t%s\n' "${cell}" "${attempt}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${manifest}"
        break
      else
        local rc=$?
        printf '%s\t%s\t%s\t%s\n' "${cell}" "${attempt}" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${manifest}"
      fi
      attempt=$(( attempt + 1 ))
    done
    if [[ "${succeeded}" != "1" ]]; then
      echo "${cell} failed after bounded retries; see ${log}" >&2
      return 1
    fi
  done 3< "${GENERATION_ROOT}/cells.tsv"
}

pids=()
for (( slot=0; slot<WORKERS; slot++ )); do
  run_worker "${slot}" &
  pids+=( "$!" )
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "One or more scoring workers failed; inspect ${SCORE_ROOT}/logs/*.attempt-*.log. Rerunning resumes validated cells." >&2
  exit 1
fi

python -m bionemo.evo2_phage_gen.calibration_scoring summarize \
  --score-dir "${SCORE_ROOT}/csv" \
  --output-csv "${SCORE_ROOT}/cell-summary.csv"

expected_total_records="$(awk -v records="${EXPECTED_RECORDS}" 'END { print (NR - 1) * records }' "${GENERATION_ROOT}/cells.tsv")"
novelty_csv="${SCORE_ROOT}/novelty/sequence-metrics.csv"
if ! [[ -s "${novelty_csv}" ]] || ! python -m bionemo.evo2_phage_gen.calibration_novelty validate \
  --metrics-csv "${novelty_csv}" --expected-records "${expected_total_records}" >/dev/null 2>&1; then
  novelty_attempt=0
  novelty_succeeded=0
  while (( novelty_attempt <= MAX_RETRIES )); do
    novelty_partial="${novelty_csv}.partial"
    novelty_log="${SCORE_ROOT}/logs/novelty.attempt-${novelty_attempt}.log"
    if timeout --signal=TERM --kill-after=60s "${NOVELTY_TIMEOUT_SECONDS}" \
      python -m bionemo.evo2_phage_gen.calibration_novelty measure \
        --generation-root "${GENERATION_ROOT}" \
        --reference-fasta "${REFERENCE_FASTA}" \
        --sft-fasta "${SFT_FASTA}" \
        --tool-bin-dir "${TOOL_BIN_DIR}" \
        --work-dir "${SCORE_ROOT}/novelty/work" \
        --output-csv "${novelty_partial}" \
        --threads "${NOVELTY_THREADS}" \
        > "${novelty_log}" 2>&1 &&
      python -m bionemo.evo2_phage_gen.calibration_novelty validate \
        --metrics-csv "${novelty_partial}" \
        --expected-records "${expected_total_records}" >> "${novelty_log}" 2>&1; then
      mv "${novelty_partial}" "${novelty_csv}"
      novelty_succeeded=1
      break
    fi
    novelty_attempt=$(( novelty_attempt + 1 ))
  done
  if [[ "${novelty_succeeded}" != "1" ]]; then
    echo "Calibration novelty analysis failed after bounded retries." >&2
    exit 1
  fi
fi
python -m bionemo.evo2_phage_gen.calibration_novelty summarize \
  --metrics-csv "${novelty_csv}" \
  --output-csv "${SCORE_ROOT}/novelty/cell-summary.csv"
python -m bionemo.evo2_phage_gen.calibration_selection \
  --score-dir "${SCORE_ROOT}/csv" \
  --novelty-summary "${SCORE_ROOT}/novelty/cell-summary.csv" \
  --output-csv "${SCORE_ROOT}/selection-evidence.csv"
touch "${SCORE_ROOT}/SUCCEEDED"
