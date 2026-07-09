#!/usr/bin/env bash
# Run full Arc-style filtering for a paper HPO generation root.
#
# Typical usage:
#   RUN_NAME=paper_hpo_useful_20260629 \
#     recipes/evo2_phage_gen/scripts/run_paper_hpo_full_arc_scoring.sh
#
# Useful controls:
#   CELL_GLOB='phix174_prompt4_temp0.7.manifest1000.fasta'  # run one cell
#   DRY_RUN=1                                               # write configs only
#   OVERWRITE=1                                             # rerun existing Arc outputs
#   GENETIC_ARCHITECTURE_REMOVE_FILTER=0                    # retain architecture-similar representatives

set -Eeuo pipefail

HPO_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HPO_REPO_ROOT="$(cd -- "${HPO_SCRIPT_DIR}/../../.." && pwd)"
HPO_RECIPE_DIR="${HPO_REPO_ROOT}/recipes/evo2_phage_gen"

if [[ "${SOURCE_ENV:-1}" == "1" ]]; then
  # shellcheck source=/dev/null
  source "${HPO_RECIPE_DIR}/.ci_test_env.sh"
fi

RUN_NAME="${RUN_NAME:-paper_hpo_useful_20260629}"
RUN_ROOT="${RUN_ROOT:-${HPO_RECIPE_DIR}/data/checkpoints/generation/${RUN_NAME}}"
TARGET_RECORDS="${TARGET_RECORDS:-1000}"
CELL_GLOB="${CELL_GLOB:-phix174_prompt*_temp*.manifest${TARGET_RECORDS}.fasta}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
GENETIC_ARCHITECTURE_REMOVE_FILTER="${GENETIC_ARCHITECTURE_REMOVE_FILTER:-1}"
if [[ "${GENETIC_ARCHITECTURE_REMOVE_FILTER}" != "0" && "${GENETIC_ARCHITECTURE_REMOVE_FILTER}" != "1" ]]; then
  printf 'ERROR: GENETIC_ARCHITECTURE_REMOVE_FILTER must be 0 or 1, got %s\n' "${GENETIC_ARCHITECTURE_REMOVE_FILTER}" >&2
  exit 2
fi

BASE_CONFIG="${BASE_CONFIG:-${HPO_RECIPE_DIR}/configs/arc_genome_design_filtering_local.yaml}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${HPO_RECIPE_DIR}/data/arc_pipeline_patched/genome_design_filtering_pipeline.py}"
FASTA_DIR="${FASTA_DIR:-${RUN_ROOT}/fasta}"
ARC_CONFIG_DIR="${ARC_CONFIG_DIR:-${RUN_ROOT}/arc_configs}"
ARC_ROOT="${ARC_ROOT:-${RUN_ROOT}/arc_filtering}"
ARC_LOG_DIR="${ARC_LOG_DIR:-${RUN_ROOT}/arc_logs}"

mkdir -p "${ARC_CONFIG_DIR}" "${ARC_ROOT}" "${ARC_LOG_DIR}"
cd "${HPO_REPO_ROOT}"

printf 'Scoring/converting HPO JSONL before full Arc filtering...\n'
python "${HPO_SCRIPT_DIR}/score_paper_hpo_generation.py" \
  --run-root "${RUN_ROOT}" \
  --target-records "${TARGET_RECORDS}"

printf 'Preparing full Arc configs under %s\n' "${ARC_CONFIG_DIR}"
python - "$BASE_CONFIG" "$FASTA_DIR" "$ARC_CONFIG_DIR" "$ARC_ROOT" "$CELL_GLOB" "$GENETIC_ARCHITECTURE_REMOVE_FILTER" <<'PY'
from pathlib import Path
import sys
import yaml

base_config = Path(sys.argv[1])
fasta_dir = Path(sys.argv[2])
arc_config_dir = Path(sys.argv[3])
arc_root = Path(sys.argv[4])
cell_glob = sys.argv[5]
genetic_architecture_remove_filter = sys.argv[6] == "1"

template = yaml.safe_load(base_config.read_text())

for fasta_path in sorted(fasta_dir.glob(cell_glob)):
    cell = fasta_path.name.removesuffix(".fasta")
    config = dict(template)
    config["results_save_dir"] = str(arc_root / cell)
    config["current_config_file"] = str(arc_config_dir / f"{cell}.yaml")
    config["evo_gen_seqs_fasta_file_save_location"] = str(fasta_path)

    config["nucleotide_filtering"] = True
    config["orf_filtering"] = True
    config["use_nucleotide_filtered_df"] = True
    config["prodigal_based_filters"] = True
    config["orf_count_filter"] = True
    config["orf_lengths_filter"] = True
    config["coding_density_filter"] = True
    config["aminoacid_homopolymer_length_filter"] = True

    config["homology_filtering"] = True
    config["use_orf_filtered_df"] = True
    config["use_nucleotide_filtered_df_instead"] = False
    config["protein_database_hit_count_filter"] = True
    config["checkv_filter"] = True
    config["genetic_architecture_filter"] = True
    config["tropism_protein_sequence_identity_filter"] = True

    config["diversification_filtering"] = True
    config["use_homology_filtered_df"] = True
    config["use_orf_filtered_df_instead"] = False
    config["use_nucleotide_filtered_df_instead_2"] = False
    config["mmseqs_clustering_filter"] = True
    config["genetic_architecture_remove_filter"] = genetic_architecture_remove_filter

    config["genetic_architecture_visualization_and_synteny_filtering"] = True
    config["use_reference_genome"] = True
    config["average_protein_sequence_identity_filter"] = True
    config["required_genes_filter"] = True
    config["syntenic_gene_count_filter"] = True

    output_path = arc_config_dir / f"{cell}.yaml"
    output_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(output_path)
PY

shopt -s nullglob
CONFIGS=( "${ARC_CONFIG_DIR}"/${CELL_GLOB%.fasta}.yaml )
IFS=$'\n' CONFIGS=( $(printf '%s\n' "${CONFIGS[@]}" | sort) )
unset IFS
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  printf 'ERROR: no Arc configs matched %s under %s\n' "${CELL_GLOB%.fasta}.yaml" "${ARC_CONFIG_DIR}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN=1; prepared %s configs and stopping before Arc execution.\n' "${#CONFIGS[@]}"
  exit 0
fi

for config in "${CONFIGS[@]}"; do
  cell="$(basename "${config}" .yaml)"
  result_dir="${ARC_ROOT}/${cell}"
  log="${ARC_LOG_DIR}/${cell}.pipeline.log"
  counts="${result_dir}/qc6_synteny_filter_counts.csv"
  if [[ "${OVERWRITE}" != "1" && -s "${counts}" ]]; then
    printf 'SKIP %s; found %s\n' "${cell}" "${counts}"
    continue
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    rm -rf "${result_dir}" "${log}"
  fi
  printf 'START full Arc scoring cell=%s config=%s log=%s\n' "${cell}" "${config}" "${log}"
  python "${PIPELINE_SCRIPT}" "${config}" > "${log}" 2>&1
  printf 'DONE full Arc scoring cell=%s\n' "${cell}"
done

python "${HPO_SCRIPT_DIR}/summarize_paper_hpo_arc_scores.py" \
  --arc-root "${ARC_ROOT}" \
  --output-csv "${RUN_ROOT}/scores/hpo_full_arc_summary.csv" \
  --output-md "${RUN_ROOT}/scores/hpo_full_arc_summary.md"
