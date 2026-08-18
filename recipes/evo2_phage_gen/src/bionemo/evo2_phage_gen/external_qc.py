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
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


RECIPE_ROOT = Path(__file__).resolve().parents[3]
ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA = str(
    RECIPE_ROOT / "data" / "external" / "arc_evo2" / "phage_gen" / "data" / "NC_001422_1.fna"
)
DEFAULT_CHECKV_DB = RECIPE_ROOT / "data" / "external" / "checkv" / "checkv-db-v1.5"


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


def _check_path(name: str, config: dict[str, Any], key: str, *, required: bool) -> QCPrerequisiteCheck:
    """Create a path-existence check from a config key."""
    value = config.get(key, "")
    ok = bool(value) and _path_exists(value)
    detail = str(value) if value else f"missing config key: {key}"
    return QCPrerequisiteCheck(name=name, ok=ok, required=required, detail=detail)


def _check_tool(name: str, tool: str, *, required: bool, search_path: str | None = None) -> QCPrerequisiteCheck:
    """Create a PATH tool check."""
    resolved = shutil.which(tool, path=search_path) if search_path is not None else shutil.which(tool)
    ok = resolved is not None
    detail = resolved or f"{tool} not found on scorer PATH"
    return QCPrerequisiteCheck(name=name, ok=ok, required=required, detail=detail)


def check_arc_qc_prerequisites(
    config_path: Path,
    *,
    genetic_architecture_import_fasta: str | Path = ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA,
    checkv_db_path: str | Path = DEFAULT_CHECKV_DB,
    tool_bin_dir: str | Path | None = None,
) -> list[QCPrerequisiteCheck]:
    """Check path and tool prerequisites for the local Arc pipeline config."""
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, Mapping):
        raise ValueError("Arc QC config must be a mapping")
    search_path = None
    if tool_bin_dir is not None:
        search_path = os.pathsep.join((str(Path(tool_bin_dir).resolve()), os.environ.get("PATH", "")))
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
    checks.append(_check_tool("prodigal", "prodigal", required=orf_required, search_path=search_path))

    checks.extend(
        [
            _check_tool("orfipy", "orfipy", required=homology_required, search_path=search_path),
            _check_tool("mmseqs", "mmseqs", required=homology_required, search_path=search_path),
            _check_path(
                "phrogs_consensus_db",
                config,
                "mmseqs_db_protein_database",
                required=homology_required and bool(config.get("protein_database_hit_count_filter")),
            ),
            _check_path(
                "tropism_mmseqs_db",
                config,
                "mmseqs_db_tropism_protein",
                required=tropism_required,
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
    checks.append(_check_tool("checkv", "checkv", required=checkv_required, search_path=search_path))
    checks.append(_check_tool("hmmsearch", "hmmsearch", required=checkv_required, search_path=search_path))
    checks.append(_check_tool("diamond", "diamond", required=checkv_required, search_path=search_path))
    checks.append(
        QCPrerequisiteCheck(
            name="checkv_database",
            ok=_path_exists(checkv_db_path),
            required=checkv_required,
            detail=str(checkv_db_path),
        )
    )

    diversification_required = bool(config.get("diversification_filtering"))
    checks.append(
        _check_tool(
            "mmseqs_for_diversification",
            "mmseqs",
            required=diversification_required,
            search_path=search_path,
        )
    )

    visualization_required = bool(config.get("genetic_architecture_visualization_and_synteny_filtering"))
    checks.append(_check_tool("lovis4u", "lovis4u", required=visualization_required, search_path=search_path))
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
    parser.add_argument(
        "--checkv-db",
        type=Path,
        default=DEFAULT_CHECKV_DB,
        help="CheckV database directory exported as CHECKVDB for CheckV runs",
    )
    parser.add_argument("--tool-bin-dir", type=Path, default=None, help="Prepend the run-specific tool directory")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--warn-only", action="store_true", help="Report missing required checks without failing")
    args = parser.parse_args()

    checks = check_arc_qc_prerequisites(
        args.config,
        genetic_architecture_import_fasta=args.genetic_architecture_import_fasta,
        checkv_db_path=args.checkv_db,
        tool_bin_dir=args.tool_bin_dir,
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
