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

from bionemo.evo2_phage_gen.external_qc import (
    ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA,
    RECIPE_ROOT,
    check_arc_qc_prerequisites,
    main,
)


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


def test_external_qc_default_import_fasta_is_recipe_local():
    """The portable prerequisite default must not retain Arc's author-specific cluster path."""
    expected = RECIPE_ROOT / "data" / "external" / "arc_evo2" / "phage_gen" / "data" / "NC_001422_1.fna"

    assert Path(ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA) == expected
    assert Path(ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA).is_relative_to(RECIPE_ROOT)


def test_external_qc_checker_allows_missing_optional_external_tools(tmp_path):
    """Safe-by-default config should only require local FASTA/reference files."""
    checks = check_arc_qc_prerequisites(
        _write_config(tmp_path),
        genetic_architecture_import_fasta=_write_import_fasta(tmp_path),
    )

    missing_required = [check.name for check in checks if check.required and not check.ok]
    assert missing_required == []
    assert any(check.name == "phrogs_consensus_db" and not check.required and not check.ok for check in checks)


@pytest.mark.parametrize("payload", ["", "- one\n- two\n", "42\n"])
def test_external_qc_checker_rejects_non_mapping_yaml(tmp_path, payload):
    """Empty, sequence, and scalar YAML cannot be interpreted as Arc configuration."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload)

    with pytest.raises(ValueError, match="mapping"):
        check_arc_qc_prerequisites(config_path)


def test_external_qc_checker_requires_enabled_stage_inputs(tmp_path, monkeypatch):
    """Enabling homology should promote MMseqs databases to required checks."""
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: None)
    checks = check_arc_qc_prerequisites(
        _write_config(tmp_path, homology_filtering=True),
        genetic_architecture_import_fasta=_write_import_fasta(tmp_path),
    )

    missing_required = {check.name for check in checks if check.required and not check.ok}
    assert "phrogs_consensus_db" in missing_required
    assert "tropism_mmseqs_db" in missing_required
    assert "mmseqs" in missing_required
    assert "orfipy" in missing_required


def test_external_qc_checker_uses_explicit_run_tool_directory(tmp_path, monkeypatch):
    """Prerequisite discovery should use the same configured tool directory as scoring."""
    tool_bin_dir = tmp_path / "fresh-external" / "bin"
    tool_bin_dir.mkdir(parents=True)
    mmseqs = tool_bin_dir / "mmseqs"
    mmseqs.write_text("#!/usr/bin/env bash\n")
    mmseqs.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda tool, path=None: str(mmseqs) if tool == "mmseqs" and path else None)

    checks = check_arc_qc_prerequisites(
        _write_config(tmp_path, homology_filtering=True),
        genetic_architecture_import_fasta=_write_import_fasta(tmp_path),
        tool_bin_dir=tool_bin_dir,
    )

    mmseqs_check = next(check for check in checks if check.name == "mmseqs")
    assert mmseqs_check.ok
    assert mmseqs_check.detail == str(mmseqs)


def test_external_qc_checker_requires_lovis4u_for_visualization_stage(tmp_path, monkeypatch):
    """The exact Arc visualization/synteny stage needs LoVis4u on PATH."""
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: None)
    checks = check_arc_qc_prerequisites(
        _write_config(tmp_path, genetic_architecture_visualization_and_synteny_filtering=True),
        genetic_architecture_import_fasta=_write_import_fasta(tmp_path),
    )

    missing_required = {check.name for check in checks if check.required and not check.ok}
    assert "lovis4u" in missing_required


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
