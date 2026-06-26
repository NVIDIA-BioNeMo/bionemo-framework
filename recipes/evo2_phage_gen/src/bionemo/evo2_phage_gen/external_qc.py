# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prerequisite checks for Arc's external phage QC pipeline."""

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA = (
    "/large_storage/hielab/samuelking/phage_design/data/phix174_only/microviridae_genomes_NC_001422_1.fna"
)


@dataclass(frozen=True)
class QCPrerequisiteCheck:
    """Single external-QC prerequisite result."""

    name: str
    ok: bool
    required: bool
    detail: str


def _path_exists(path_like: str | Path) -> bool:
    """Return true when ``path_like`` points to an existing filesystem path."""
    return Path(path_like).exists()


def _tool_exists(tool: str) -> bool:
    """Return true when an executable is discoverable on PATH."""
    return shutil.which(tool) is not None


def _check_path(name: str, config: dict[str, Any], key: str, *, required: bool) -> QCPrerequisiteCheck:
    """Create a path-existence check from a config key."""
    value = config.get(key, "")
    ok = bool(value) and _path_exists(value)
    detail = str(value) if value else f"missing config key: {key}"
    return QCPrerequisiteCheck(name=name, ok=ok, required=required, detail=detail)


def _check_tool(name: str, tool: str, *, required: bool) -> QCPrerequisiteCheck:
    """Create a PATH tool check."""
    ok = _tool_exists(tool)
    detail = shutil.which(tool) or f"{tool} not found on PATH"
    return QCPrerequisiteCheck(name=name, ok=ok, required=required, detail=detail)


def check_arc_qc_prerequisites(
    config_path: Path,
    *,
    genetic_architecture_import_fasta: str | Path = ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA,
) -> list[QCPrerequisiteCheck]:
    """Check path and tool prerequisites for the local Arc pipeline config."""
    config = yaml.safe_load(Path(config_path).read_text())
    homology_required = bool(config.get("homology_filtering"))
    reference_identity_required = homology_required and bool(config.get("reference_genome_sequence_identity_filter"))
    genetic_architecture_required = homology_required and bool(config.get("genetic_architecture_filter"))
    tropism_required = homology_required and bool(config.get("tropism_protein_sequence_identity_filter"))
    checks = [
        _check_path("input_fasta", config, "evo_gen_seqs_fasta_file_save_location", required=True),
        _check_path("reference_genome_fasta", config, "reference_genome_fasta", required=reference_identity_required),
        _check_path(
            "genetic_architecture_reference_genome",
            config,
            "genetic_architecture_reference_genome",
            required=genetic_architecture_required,
        ),
        _check_path("reference_tropism_protein", config, "reference_tropism_protein", required=tropism_required),
        QCPrerequisiteCheck(
            name="arc_genetic_architecture_import_fasta",
            ok=_path_exists(genetic_architecture_import_fasta),
            required=True,
            detail=str(genetic_architecture_import_fasta),
        ),
    ]

    orf_required = bool(config.get("orf_filtering"))
    checks.append(_check_tool("prodigal", "prodigal", required=orf_required))

    checks.extend(
        [
            _check_tool("orfipy", "orfipy", required=homology_required),
            _check_tool("mmseqs", "mmseqs", required=homology_required),
            _check_path(
                "phrogs_mmseqs_db",
                config,
                "mmseqs_db_protein_database",
                required=homology_required and bool(config.get("protein_database_hit_count_filter")),
            ),
            _check_path(
                "tropism_mmseqs_db",
                config,
                "mmseqs_db_tropism_protein",
                required=homology_required and bool(config.get("tropism_protein_sequence_identity_filter")),
            ),
            _check_path(
                "training_data_genomes_fasta",
                config,
                "training_data_genomes_fasta",
                required=homology_required and bool(config.get("training_data_sequence_identity_filter")),
            ),
        ]
    )

    checkv_required = homology_required and bool(config.get("checkv_filter"))
    checks.append(_check_tool("checkv", "checkv", required=checkv_required))

    diversification_required = bool(config.get("diversification_filtering"))
    checks.append(_check_tool("mmseqs_for_diversification", "mmseqs", required=diversification_required))

    visualization_required = bool(config.get("genetic_architecture_visualization_and_synteny_filtering"))
    checks.extend(
        [
            _check_path(
                "genetic_architecture_visualization_script",
                config,
                "genetic_architecture_visualization_script",
                required=visualization_required,
            ),
            _check_path("protein_annotation_file", config, "protein_annotation_file", required=visualization_required),
            _check_path(
                "reference_genome_gff_file",
                config,
                "reference_genome_gff_file_save_location",
                required=visualization_required and bool(config.get("use_reference_genome")),
            ),
        ]
    )
    return checks


def _print_text_report(checks: list[QCPrerequisiteCheck]) -> None:
    """Print a concise human-readable prerequisite report."""
    for check in checks:
        severity = "required" if check.required else "optional"
        status = "ok" if check.ok else "missing"
        print(f"{status:7} {severity:8} {check.name}: {check.detail}")


def main() -> None:
    """CLI entry point for Arc external-QC prerequisite checks."""
    parser = argparse.ArgumentParser(description="Check prerequisites for Arc's phage genome filtering config")
    parser.add_argument("--config", type=Path, required=True, help="Arc genome-design filtering YAML config")
    parser.add_argument(
        "--genetic-architecture-import-fasta",
        type=Path,
        default=Path(ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA),
        help="PhiX174 FASTA path read by Arc genetic_architecture.py at import time",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--warn-only", action="store_true", help="Report missing required checks without failing")
    args = parser.parse_args()

    checks = check_arc_qc_prerequisites(
        args.config,
        genetic_architecture_import_fasta=args.genetic_architecture_import_fasta,
    )
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        _print_text_report(checks)

    missing_required = [check for check in checks if check.required and not check.ok]
    if missing_required and not args.warn_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
