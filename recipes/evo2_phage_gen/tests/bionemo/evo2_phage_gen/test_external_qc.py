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

"""Tests for ``bionemo.evo2_phage_gen.external_qc``."""

from pathlib import Path

import pytest
import yaml

from bionemo.evo2_phage_gen.external_qc import check_arc_qc_prerequisites, main


def _write_config(tmp_path: Path, **overrides) -> Path:
    input_fasta = tmp_path / "generated.fasta"
    reference = tmp_path / "NC_001422_1.fna"
    g_protein = tmp_path / "NC_001422.1_Gprotein.fasta"
    for path in [input_fasta, reference, g_protein]:
        path.write_text(">seq\nACGT\n")

    config = {
        "evo_gen_seqs_fasta_file_save_location": str(input_fasta),
        "reference_genome_fasta": str(reference),
        "genetic_architecture_reference_genome": str(reference),
        "reference_tropism_protein": str(g_protein),
        "orf_filtering": False,
        "homology_filtering": False,
        "protein_database_hit_count_filter": True,
        "mmseqs_db_protein_database": str(tmp_path / "missing_phrogs_db"),
        "tropism_protein_sequence_identity_filter": True,
        "mmseqs_db_tropism_protein": str(tmp_path / "missing_tropism_db"),
        "training_data_sequence_identity_filter": False,
        "training_data_genomes_fasta": str(tmp_path / "missing_training.fna"),
        "checkv_filter": False,
        "diversification_filtering": False,
        "genetic_architecture_visualization_and_synteny_filtering": False,
        "genetic_architecture_visualization_script": str(tmp_path / "missing_viz.py"),
        "protein_annotation_file": str(tmp_path / "missing_annotations.tsv"),
        "use_reference_genome": False,
        "reference_genome_gff_file_save_location": str(tmp_path / "missing.gff"),
    }
    config.update(overrides)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _write_import_fasta(tmp_path: Path) -> Path:
    import_fasta = tmp_path / "legacy_import_phiX174.fna"
    import_fasta.write_text(">NC_001422.1\nACGT\n")
    return import_fasta


def test_external_qc_checker_allows_missing_optional_external_tools(tmp_path):
    """Safe-by-default config should only require local FASTA/reference files."""
    checks = check_arc_qc_prerequisites(
        _write_config(tmp_path),
        genetic_architecture_import_fasta=_write_import_fasta(tmp_path),
    )

    missing_required = [check.name for check in checks if check.required and not check.ok]
    assert missing_required == []
    assert any(check.name == "phrogs_mmseqs_db" and not check.required and not check.ok for check in checks)


def test_external_qc_checker_requires_enabled_stage_inputs(tmp_path):
    """Enabling homology should promote MMseqs databases to required checks."""
    checks = check_arc_qc_prerequisites(
        _write_config(tmp_path, homology_filtering=True),
        genetic_architecture_import_fasta=_write_import_fasta(tmp_path),
    )

    missing_required = {check.name for check in checks if check.required and not check.ok}
    assert "phrogs_mmseqs_db" in missing_required
    assert "tropism_mmseqs_db" in missing_required
    assert "mmseqs" in missing_required
    assert "orfipy" in missing_required


def test_external_qc_cli_warn_only_does_not_fail_on_missing_required(tmp_path, monkeypatch, capsys):
    """``--warn-only`` should support planning runs before generated FASTA exists."""
    missing_input = tmp_path / "missing.fasta"
    config_path = _write_config(tmp_path, evo_gen_seqs_fasta_file_save_location=str(missing_input))
    monkeypatch.setattr(
        "sys.argv",
        ["evo2_phage_check_external_qc", "--config", str(config_path), "--warn-only"],
    )

    main()

    assert "missing required input_fasta" in capsys.readouterr().out


def test_external_qc_cli_accepts_import_fasta_override(tmp_path, monkeypatch, capsys):
    """The CLI should support patched Arc workdirs with repo-local import FASTA paths."""
    import_fasta = _write_import_fasta(tmp_path)
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evo2_phage_check_external_qc",
            "--config",
            str(config_path),
            "--genetic-architecture-import-fasta",
            str(import_fasta),
        ],
    )

    main()

    assert "ok      required arc_genetic_architecture_import_fasta" in capsys.readouterr().out


def test_external_qc_cli_fails_on_missing_required_without_warn_only(tmp_path, monkeypatch):
    """Missing required checks should make the CLI fail by default."""
    missing_input = tmp_path / "missing.fasta"
    config_path = _write_config(tmp_path, evo_gen_seqs_fasta_file_save_location=str(missing_input))
    monkeypatch.setattr("sys.argv", ["evo2_phage_check_external_qc", "--config", str(config_path)])

    with pytest.raises(SystemExit):
        main()
