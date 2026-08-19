#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

# Agent-free PhiX174 whole-genome SFT -> GDPO -> generation/screening example.

set -Eeuo pipefail

RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${RECIPE_ROOT}/results/phix174-8xh100"
DRY_RUN=0
PREPARE_ONLY=0
RESUME_FROM=00
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-600}"
PHAROKKA_DATABASE_URL="${PHAROKKA_DATABASE_URL:-https://zenodo.org/records/21755221/files/pharokka_v1.11.0_databases.tar.gz?download=1}"
PHAROKKA_DATABASE_MD5="${PHAROKKA_DATABASE_MD5:-143bb375ddb0b0653e5cb5671f4a7629}"
PHAROKKA_DATABASE_RELEASE="${PHAROKKA_DATABASE_RELEASE:-Pharokka database v1.11.0 / PHROGs v4}"
SAFETY_BATCH_SIZE="${SAFETY_BATCH_SIZE:-128}"
SAFETY_ORF_WORKERS="${SAFETY_ORF_WORKERS:-32}"
SAFETY_THREADS="${SAFETY_THREADS:-32}"
SAFETY_PHROGS_THREADS="${SAFETY_PHROGS_THREADS:-64}"

usage() {
  printf '%s\n' \
    'Usage: ./examples/phix174_8xh100.sh [OPTIONS]' \
    '  --result-root PATH  Result directory (default: results/phix174-8xh100)' \
    '  --prepare-only      Prepare public inputs/tools/controls, then stop' \
    '  --resume-from ID    Start at stage 00, 10, 20, 30, 40, or 50' \
    '  --dry-run           Record and print commands without external work' \
    '  -h, --help          Show this help'
}

while (($#)); do
  case "$1" in
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
case "${RESUME_FROM}" in 00|10|20|30|40|50) ;; *) printf 'Invalid stage: %s\n' "${RESUME_FROM}" >&2; exit 2 ;; esac

STATE_DIR="${RESULT_ROOT}/state"
STAGE_DIR="${RESULT_ROOT}/stages"
RUNLOG="${RESULT_ROOT}/RUNLOG.md"
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
  [[ "${DRY_RUN}" == "1" ]] && return
  python - "$1" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
bad = [(r["record_id"], a.get("safety_class"), a.get("execution_status")) for r in data["records"] for a in r["adapter_attempts"] if a.get("execution_status") != "COMPLETED_AND_PARSED"]
if bad:
    raise SystemExit(f"incomplete safety detector execution: {bad[:10]}")
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

if [[ "${DRY_RUN}" != "1" ]]; then
  # shellcheck source=/dev/null
  source "${RECIPE_ROOT}/.ci_test_env.sh"
  export PATH="${RECIPE_ROOT}/data/external/bin:${PATH}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  if ! gpu_info="$(nvidia-smi --query-gpu=name --format=csv,noheader)"; then
    printf '%s\n' 'Unable to query GPUs. Run on the allocated compute node, outside a restricted agent sandbox.' >&2
    exit 2
  fi
  [[ "$(wc -l <<< "${gpu_info}")" == 8 ]] || { printf 'Expected 8 GPUs; found:\n%s\n' "${gpu_info}" >&2; exit 2; }
  [[ "$(grep -c H100 <<< "${gpu_info}")" == 8 ]] || { printf 'Expected 8 H100 GPUs; found:\n%s\n' "${gpu_info}" >&2; exit 2; }
fi
printf '%s\n' '{"gpu_count":8,"gpu_type":"H100 80GB","whole_genome":true,"safety_screen":"current configured databases","final_generation_count":1000}' > "${RESULT_ROOT}/settings.json"
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
  local prep base_nemo base_mbridge="${RESULT_ROOT}/checkpoints/evo2-7b-8k-mbridge-10240" sft="${RESULT_ROOT}/sft/train" selected
  prep="$(read_state sft-prepared)"
  local model=(--hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_512 --model-size evo2_7b_base --micro-batch-size 1 --seq-length 10240 --tensor-model-parallel-size 2 --use-precision-aware-optimizer --bf16-main-grads --grad-reduce-in-fp32 --overlap-grad-reduce --cross-entropy-loss-fusion --no-weight-decay-embeddings --no-renormalize-loss --use-subquadratic-ops --no-fp32-residual-connection --activation-checkpoint-recompute-num-layers 1 --eod-pad-in-loss-mask --mixed-precision-recipe bf16_mixed)
  if [[ -f "${STAGE_DIR}/20-sft.done" ]]; then
    note 'substage 20-sft already complete'
  else
    [[ "${DRY_RUN}" == "1" ]] && base_nemo='<downloaded-evo2-7b-8k>' || base_nemo="$(download_bionemo_data evo2/7b-8k:1.0 | tail -n 1)"
    if [[ "${DRY_RUN}" == "1" || ! -d "${base_mbridge}" ]]; then
      run evo2_convert_nemo2_to_mbridge --nemo2-ckpt-dir "${base_nemo}" --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 --mbridge-ckpt-dir "${base_mbridge}" --model-size evo2_7b_base --seq-length 10240 --mixed-precision-recipe bf16_mixed
    fi
    monitored 'SFT smoke' "${RESULT_ROOT}/sft/smoke.log" torchrun --nproc-per-node 8 --no-python train_evo2 "${model[@]}" --dataset-config "${prep}/training_dataset.yaml" --finetune-ckpt-dir "${base_mbridge}" --global-batch-size 32 --max-steps 2 --eval-interval 1 --eval-iters 1 --warmup-steps 0 --decay-steps 2 --result-dir "${RESULT_ROOT}/sft/smoke" --experiment-name evo2-smoke
    monitored '12,000-step SFT' "${sft}/train.log" torchrun --nproc-per-node 8 --no-python train_evo2 "${model[@]}" --dataset-config "${prep}/training_dataset.yaml" --finetune-ckpt-dir "${base_mbridge}" --global-batch-size 32 --max-steps 12000 --eval-interval 400 --eval-iters 4 --lr 1e-5 --min-lr 1e-6 --warmup-steps 600 --decay-steps 11400 --enable-preemption --keep-best-k 3 --most-recent-k 1 --checkpoint-metric-name 'lm loss' --strict-checkpoint-metric --checkpoint-metric-step-tolerance 1 --result-dir "${sft}" --experiment-name evo2
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/20-sft.done"
  fi
  [[ "${DRY_RUN}" == "1" ]] && selected='<selected-sft>' || selected="$(select_checkpoint sft "${sft}/evo2/tb_logs" "${sft}/evo2/checkpoints" "${RESULT_ROOT}/sft/checkpoint-selection.json")"
  state selected-sft "${selected}"
  monitored 'held-out SFT evaluation' "${RESULT_ROOT}/sft/heldout.log" torchrun --nproc-per-node 8 --no-python train_evo2 "${model[@]}" --dataset-config "${prep}/heldout_dataset.yaml" --finetune-ckpt-dir "${selected}" --global-batch-size 20 --max-steps 0 --eval-interval 1 --eval-iters 5 --warmup-steps 0 --decay-steps 0 --result-dir "${RESULT_ROOT}/sft/heldout" --experiment-name evo2-heldout
}

stage_30() {
  local selected calibration="${RESULT_ROOT}/calibration" evidence selection="${RESULT_ROOT}/calibration/selected-sampling.json"
  selected="$(read_state selected-sft)"; evidence="${calibration}/scoring/selection-evidence.csv"
  monitored 'calibration generation' "${calibration}/generation.log" env SOURCE_ENV=0 RUN_ROOT="${calibration}/generation" CKPT_DIR="${selected}" PROMPT_LENGTHS='0 1 2 4 6 8 10 12 16 24 32' TEMPERATURES='0.3 0.5 0.7 0.9 1.0 1.1 1.3' NUM_PROMPTS=64 TARGET_LENGTH=6000 GPU_IDS='0 1 2 3 4 5 6 7' TENSOR_PARALLEL_SIZE=1 scripts/calibration/run_sft_sampling_sweep.sh
  monitored 'calibration scoring' "${calibration}/scoring.log" env SOURCE_ENV=0 CALIBRATION_ROOT="${calibration}" GENERATION_ROOT="${calibration}/generation" ARC_CONFIG="${RECIPE_ROOT}/configs/arc_genome_design_filtering_local.yaml" PIPELINE_SCRIPT="${RECIPE_ROOT}/data/arc_pipeline_patched/genome_design_filtering_pipeline.py" TOOL_BIN_DIR="${RECIPE_ROOT}/data/external/bin" REFERENCE_FASTA="${RECIPE_ROOT}/data/external/arc_evo2/phage_gen/data/NC_001422_1.fna" SFT_FASTA="${RESULT_ROOT}/sft/source-safety/partitions/pass.fasta" WORKERS=8 scripts/calibration/run_sampling_calibration_scoring.sh
  if [[ "${DRY_RUN}" == "1" ]]; then note 'verify fresh calibration supports temperature 1.0 and prefixes 16/24'; else
    python - "${evidence}" "${selection}" <<'PY'
import json, sys, pandas as pd
table=pd.read_csv(sys.argv[1]); chosen=table[(table.temperature==1.0)&table.prefix_length.isin([16,24])]
if len(chosen)!=2 or not chosen[["eligible","metric_environment_ok","temperature_1_default_candidate"]].to_numpy().all(): raise SystemExit("fresh calibration does not support temperature 1.0 with prefixes 16/24")
open(sys.argv[2],"w").write(json.dumps({"temperature":1.0,"top_k":4,"top_p":1.0,"max_new_tokens":5976,"prompt_lengths":[16,24],"weights":[0.5,0.5],"seed":7,"seed_stride":1000003},indent=2)+"\n")
PY
  fi
  run evo2_phage_generation write-rl-prompts --output "${RESULT_ROOT}/rl/train.jsonl" --prompt-lengths 16 24 --repeats-per-length 6 --id-prefix train
  run evo2_phage_generation write-rl-prompts --output "${RESULT_ROOT}/rl/validation.jsonl" --prompt-lengths 16 24 --repeats-per-length 48 --id-prefix validation --grouped
}

stage_40() {
  local selected rl="${RESULT_ROOT}/rl" control="${RESULT_ROOT}/rl/environment-control" chosen
  if [[ -f "${STAGE_DIR}/40-rl.done" ]]; then
    note 'substage 40-rl already complete'
  else
    selected="$(read_state selected-sft)"
    export NEMO_RL_RAY_NUM_CPUS="${NEMO_RL_RAY_NUM_CPUS:-$(nproc)}"
    note "RL Ray CPU slots: ${NEMO_RL_RAY_NUM_CPUS}; reward phases use at most 64 threads"
    run pytest -q tests/bionemo/evo2_phage_gen/test_reward.py tests/bionemo/evo2_phage_gen/test_nemo_rl_env.py tests/bionemo/evo2_phage_gen/test_reference_controls.py
    monitored 'RL environment control' "${control}/runner.log" \
      evo2_phage_check_rl --config configs/gdpo_phage_megatron.yaml --checkpoint "${selected}" \
      --control-fasta data/external/arc_evo2/phage_gen/data/NC_001422.1.fna --control-dir "${control}"
    local common=(checkpointing.pretrained_checkpoint.path="${selected}" data.train.data_path="${rl}/train.jsonl" data.validation.data_path="${rl}/validation.jsonl" logger.wandb_enabled=false)
    monitored 'one-step GDPO pilot' "${RESULT_ROOT}/rl-pilot/runner.log" evo2_phage_run_gdpo --config configs/gdpo_phage_megatron.yaml "${common[@]}" checkpointing.checkpoint_dir="${RESULT_ROOT}/rl-pilot/checkpoints" checkpointing.save_period=1 grpo.max_num_steps=1 grpo.val_at_end=true env.phage_qc.external_qc.work_dir="${RESULT_ROOT}/rl-pilot/external-qc" env.phage_qc.mmseqs_cluster_diversity.work_dir="${RESULT_ROOT}/rl-pilot/mmseqs" env.phage_qc.sequence_safety.work_dir="${RESULT_ROOT}/rl-pilot/safety" logger.log_dir="${RESULT_ROOT}/rl-pilot/logs"
    run evo2_phage_monitor_objectives --tensorboard-root "${RESULT_ROOT}/rl-pilot/logs" --config configs/gdpo_phage_megatron.yaml --minimum-events 1 --output "${RESULT_ROOT}/rl-pilot/objective-health.json"
    check_objectives "${RESULT_ROOT}/rl-pilot/objective-health.json"
    monitored '500-step DP8 GDPO' "${rl}/runner.log" evo2_phage_run_gdpo --config configs/gdpo_phage_megatron.yaml "${common[@]}" checkpointing.checkpoint_dir="${rl}/checkpoints" env.phage_qc.external_qc.work_dir="${rl}/external-qc" env.phage_qc.mmseqs_cluster_diversity.work_dir="${rl}/mmseqs" env.phage_qc.sequence_safety.work_dir="${rl}/safety" logger.log_dir="${rl}/logs"
    [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/40-rl.done"
  fi
  run evo2_phage_monitor_objectives --tensorboard-root "${rl}/logs" --config configs/gdpo_phage_megatron.yaml --output "${rl}/objective-health.json" --history-output "${rl}/objective-history.json"
  check_objectives "${rl}/objective-health.json"
  [[ "${DRY_RUN}" == "1" ]] && chosen='<selected-rl>' || chosen="$(select_checkpoint rl "${rl}/logs" "${rl}/checkpoints" "${rl}/checkpoint-selection.json")"
  state selected-rl "${chosen}"
}

stage_50() {
  local selected selected_sft rollout="${RESULT_ROOT}/rollout" fasta safety likelihood evidence infer
  selected="$(read_state selected-rl)"
  selected_sft="$(read_state selected-sft)"
  fasta="${rollout}/fasta/phix174_prompt16-24_temp1.0.n1000.fasta"
  safety="${rollout}/sequence-safety"
  likelihood="${rollout}/sft-likelihood"
  infer="${RECIPE_ROOT}/src/bionemo/evo2/run/infer.py"
  if [[ -f "${STAGE_DIR}/50-rollout.done" ]]; then
    note 'substage 50-rollout already complete'
    if [[ "${DRY_RUN}" != "1" && ! -s "${fasta}" ]]; then
      printf 'rollout substage is marked complete but FASTA is missing: %s\n' "${fasta}" >&2
      return 1
    fi
  else
    run evo2_phage_generation write-prompts --output-dir "${rollout}/prompts" --prompt-lengths 16 24 --num-prompts 500 --id-prefix final
  local shard_dir="${rollout}/prompts/dp8" rank started waited alive failed=0 printable pid
  local -a command=() outputs=() pids=() logs=()
  if [[ "${DRY_RUN}" == "1" ]]; then
    note 'split the 500/500 prompt mixture into eight homogeneous 125-record shards'
  else
    python - "${rollout}/prompts/final_prompt16_500.jsonl" "${rollout}/prompts/final_prompt24_500.jsonl" "${shard_dir}" <<'PY'
import json, sys
from pathlib import Path
inputs, output = [Path(path) for path in sys.argv[1:3]], Path(sys.argv[3])
records = [[json.loads(line) for line in path.read_text().splitlines() if line] for path in inputs]
if [len(group) for group in records] != [500, 500]:
    raise SystemExit("expected 500 prompts for each prefix length")
all_records = records[0] + records[1]
if len({record["id"] for record in all_records}) != 1000:
    raise SystemExit("final prompt IDs are not unique")
output.mkdir(parents=True, exist_ok=True)
for rank in range(8):
    shard = all_records[rank * 125 : (rank + 1) * 125]
    (output / f"dp{rank}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in shard))
PY
  fi
  started=${SECONDS}
  for rank in {0..7}; do
    outputs+=("${rollout}/jsonl/dp${rank}.jsonl")
    logs+=("${rollout}/logs/dp${rank}.log")
    command=(env CUDA_VISIBLE_DEVICES="${rank}" torchrun --nproc_per_node 1 --nnodes 1 \
      --master_port "$((29544 + rank))" "${infer}" --ckpt-dir "${selected}" \
      --prompt-file "${shard_dir}/dp${rank}.jsonl" --max-new-tokens 5976 --temperature 1.0 \
      --top-k 4 --top-p 1.0 --seed "$((7 + rank * 1000003))" --tensor-parallel-size 1 \
      --max-seq-length 10240 --prompt-batch-size 16 --strict-generation --stream-output \
      --output-file "${outputs[-1]}")
    printf -v printable '%q ' "${command[@]}"; note "command: ${printable}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      mkdir -p "$(dirname -- "${outputs[-1]}")" "$(dirname -- "${logs[-1]}")"
      "${command[@]}" > "${logs[-1]}" 2>&1 & pids+=("$!")
    fi
  done
  if [[ "${DRY_RUN}" != "1" ]]; then
    while :; do
      alive=0; for pid in "${pids[@]}"; do kill -0 "${pid}" 2>/dev/null && alive=$((alive + 1)); done
      ((alive == 0)) && break
      waited=0
      while ((waited < MONITOR_INTERVAL_SECONDS && alive > 0)); do
        sleep 10; waited=$((waited + 10)); alive=0
        for pid in "${pids[@]}"; do kill -0 "${pid}" 2>/dev/null && alive=$((alive + 1)); done
      done
      ((alive > 0)) && note "generation: ${alive}/8 workers still running after $((SECONDS - started))s"
    done
    for rank in {0..7}; do
      if ! wait "${pids[rank]}"; then tail -n 30 "${logs[rank]}" >&2; failed=1; fi
    done
    ((failed == 0)) || return 1
    python - "${outputs[@]}" <<'PY'
import json, sys
seen = set()
for path in sys.argv[1:]:
    records = [json.loads(line) for line in open(path) if line.strip()]
    if len(records) != 125:
        raise SystemExit(f"{path} has {len(records)} records instead of 125")
    for record in records:
        if record["id"] in seen:
            raise SystemExit(f'duplicate generated ID: {record["id"]}')
        seen.add(record["id"])
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

  run evo2_phage_generation prepare-sft-likelihood \
    --input-fasta "${fasta}" --output-fasta "${likelihood}/sft-conditioned.fasta"
  monitored 'selected-SFT likelihood scoring' "${likelihood}/predict.log" \
    torchrun --nproc-per-node 8 --no-python predict_evo2 \
    --fasta "${likelihood}/sft-conditioned.fasta" --ckpt-dir "${selected_sft}" \
    --output-dir "${likelihood}/predictions" --tensor-parallel-size 1 --micro-batch-size 1 \
    --use-subquadratic-ops --output-log-prob-seqs --log-prob-collapse-option per_token
  run evo2_phage_generation collect-sft-likelihood \
    --prediction-dir "${likelihood}/predictions" --source-fasta "${fasta}" \
    --output-csv "${likelihood}/ranked-designs.csv"

  run evo2_phage_nucleotide_qc --input-fasta "${fasta}" --output-dir "${safety}/input-qc" \
    --genome-length-min 1 --genome-length-max 1000000 --gc-content-min 0 --gc-content-max 100 \
    --homopolymer-max 1000000
  evidence='{"source":"NCBI PhiX174 reference","source_version":"NC_001422.1","replication_host_domains":["BACTERIA"],"confirmed":true}'
  run_result 'final safety scan' "${safety}/scan.log" evo2_phage_sequence_safety scan \
    --input-fasta "${safety}/input-qc/qc2_nt_filter_seqs.fasta" --output-dir "${safety}/scan" \
    --policy configs/phage_safety_policy.yaml --asset-manifest data/external/safety/asset_manifest.yaml \
    --host-domain BACTERIA --host-evidence-json "${evidence}" --strict-lysis \
    --batch-size "${SAFETY_BATCH_SIZE}" --orf-workers "${SAFETY_ORF_WORKERS}" \
    --threads "${SAFETY_THREADS}" --phrogs-threads "${SAFETY_PHROGS_THREADS}" --timeout 1800 --overwrite
  check_scan "${safety}/scan/manifest.json"
  run evo2_phage_summarize_safety_manifest --manifest "${safety}/scan/manifest.json" --output "${safety}/summary.json"

  if [[ "${DRY_RUN}" == "1" ]]; then
    note 'prepare target and filter-7 diagnostic Arc configs from the maintained local template'
  else
    python - configs/arc_genome_design_filtering_local.yaml "${fasta}" "${rollout}" <<'PY'
from pathlib import Path
import sys, yaml
base, fasta, root = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
for name, remove_filter in (("target-profile", False), ("filter7-diagnostic", True)):
    config = yaml.safe_load(base.read_text())
    out = root / name
    config.update({
        "results_save_dir": str(out / "arc"),
        "current_config_file": str(out / "config.yaml"),
        "evo_gen_seqs_fasta_file_save_location": str(fasta),
        "orf_filtering": True,
        "use_nucleotide_filtered_df": True,
        "homology_filtering": True,
        "use_orf_filtered_df": True,
        "use_nucleotide_filtered_df_instead": False,
        "checkv_filter": True,
        "genetic_architecture_filter": True,
        "diversification_filtering": True,
        "mmseqs_clustering_filter": True,
        "genetic_architecture_remove_filter": remove_filter,
        "genetic_architecture_visualization_and_synteny_filtering": True,
        "use_reference_genome": True,
    })
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
  fi
  monitored 'Arc target profile' "${rollout}/target-profile/runner.log" \
    python data/arc_pipeline_patched/genome_design_filtering_pipeline.py "${rollout}/target-profile/config.yaml"
  monitored 'Arc filter-7 diagnostic' "${rollout}/filter7-diagnostic/runner.log" \
    python data/arc_pipeline_patched/genome_design_filtering_pipeline.py "${rollout}/filter7-diagnostic/config.yaml"

  if [[ "${DRY_RUN}" == "1" ]]; then
    note 'join all 1,000 SFT likelihoods with safety and target-profile results; rank only if length bias is acceptable'
  else
    [[ -f "${rollout}/target-profile/arc/qc6_synteny_filter_seqs.fasta" ]] || {
      printf 'missing terminal target-profile FASTA: %s\n' \
        "${rollout}/target-profile/arc/qc6_synteny_filter_seqs.fasta" >&2
      return 1
    }
  fi
  run evo2_phage_generation finalize-rollout \
    --generated-fasta "${fasta}" --safety-manifest "${safety}/scan/manifest.json" \
    --target-fasta "${rollout}/target-profile/arc/qc6_synteny_filter_seqs.fasta" \
    --likelihood-csv "${likelihood}/ranked-designs.csv" \
    --output-json "${rollout}/final-designs.json" \
    --accepted-fasta "${rollout}/accepted_candidates.fasta" --summary "${RESULT_ROOT}/SUMMARY.md" \
    --model-checkpoint "${selected_sft}"
}

printf '%s\n' '00 prepare inputs/tools/controls' '10 safety-screen and prepare SFT' '20 train/select/evaluate SFT' '30 calibrate sampling' '40 pilot/train/monitor/select GDPO' '50 generate, SFT-score, and screen 1,000 genomes' > "${RESULT_ROOT}/stage-plan.txt"
for id in 00 10 20 30 40 50; do
  ((10#${id} < 10#${RESUME_FROM})) && continue
  [[ "${PREPARE_ONLY}" == "1" && "${id}" != 00 ]] && continue
  [[ -f "${STAGE_DIR}/${id}.done" ]] && { note "stage ${id} already complete"; continue; }
  note "starting stage ${id}"; "stage_${id}"; [[ "${DRY_RUN}" == "1" ]] || touch "${STAGE_DIR}/${id}.done"
done
note 'requested stages complete'
