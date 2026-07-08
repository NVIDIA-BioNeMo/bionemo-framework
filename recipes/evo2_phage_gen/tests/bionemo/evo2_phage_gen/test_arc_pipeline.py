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

"""Tests for ``bionemo.evo2_phage_gen.arc_pipeline``."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from bionemo.evo2_phage_gen.arc_pipeline import (
    ARC_EVO2_GIT_URL,
    ARC_EVO2_REV,
    ARC_LEGACY_CHECKV_ENV,
    ARC_LEGACY_EMPTY_DIVERSIFICATION_ANCHOR,
    ARC_LEGACY_EMPTY_HOMOLOGY_ANCHOR,
    ARC_LEGACY_EMPTY_ORF_ANCHOR,
    ARC_LEGACY_EMPTY_SYNTENY_ANCHOR,
    ARC_LEGACY_LOVIS4U_CONDA_WRAPPER,
    ARC_LEGACY_LOVIS4U_PARALLEL_CONFIG,
    ARC_LEGACY_LOVIS4U_PDF_COLLECTION,
    ARC_LEGACY_MMSEQS_EMPTY_GUARD_ANCHOR,
    ARC_LEGACY_PRODIGAL_CMD,
    ARC_PIPELINE_FILES,
    DEFAULT_ARC_PIPELINE_PATCH,
    DEFAULT_ARC_PIPELINE_SOURCE_DIR,
    DEFAULT_PHIX174_FASTA,
    PATCHED_CHECKV_ENV,
    PATCHED_EMPTY_DIVERSIFICATION_GUARD,
    PATCHED_EMPTY_HOMOLOGY_GUARD,
    PATCHED_EMPTY_ORF_GUARD,
    PATCHED_EMPTY_SYNTENY_GUARD,
    PATCHED_LOVIS4U_CONDA_WRAPPER,
    PATCHED_LOVIS4U_PARALLEL_CONFIG,
    PATCHED_LOVIS4U_PDF_COLLECTION,
    PATCHED_MMSEQS_EMPTY_GUARD,
    PATCHED_PRODIGAL_CMD,
    _assert_arc_source_revision,
    prepare_arc_pipeline_workdir,
)
from bionemo.evo2_phage_gen.external_qc import ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA


def _load_prepared_arc_pipeline(tmp_path: Path, module_name: str):
    """Prepare Arc into tmp_path and import the patched pipeline module from there."""
    if not DEFAULT_ARC_PIPELINE_SOURCE_DIR.exists() or not DEFAULT_PHIX174_FASTA.exists():
        pytest.skip("Arc source assets are not available")
    workdir = tmp_path / "patched_arc"
    prepare_arc_pipeline_workdir(
        DEFAULT_ARC_PIPELINE_SOURCE_DIR,
        workdir,
        phix174_fasta=DEFAULT_PHIX174_FASTA,
    )
    pipeline_path = workdir / "genome_design_filtering_pipeline.py"
    sys.path.insert(0, str(pipeline_path.parent))
    try:
        spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_prepare_arc_pipeline_workdir_patches_legacy_reference_path(tmp_path):
    """The prepared Arc workdir should not depend on Arc's legacy absolute path."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for filename in ARC_PIPELINE_FILES:
        content = "print('ok')\n"
        if filename == "genetic_architecture.py":
            content = f'fasta_file = "{ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA}"\n'
        if filename == "genetic_architecture_visualization.py":
            content = ARC_LEGACY_LOVIS4U_PARALLEL_CONFIG
        if filename == "genome_design_filtering_pipeline.py":
            content = (
                f"{ARC_LEGACY_PRODIGAL_CMD}\n"
                f"{ARC_LEGACY_CHECKV_ENV}\n"
                f"{ARC_LEGACY_LOVIS4U_CONDA_WRAPPER}\n"
                f"{ARC_LEGACY_LOVIS4U_PDF_COLLECTION}\n"
                f"{ARC_LEGACY_MMSEQS_EMPTY_GUARD_ANCHOR}\n"
                f"{ARC_LEGACY_EMPTY_ORF_ANCHOR}\n"
                f"{ARC_LEGACY_EMPTY_HOMOLOGY_ANCHOR}\n"
                f"{ARC_LEGACY_EMPTY_DIVERSIFICATION_ANCHOR}\n"
                f"{ARC_LEGACY_EMPTY_SYNTENY_ANCHOR}\n"
            )
        (source_dir / filename).write_text(content)
    phix174_fasta = tmp_path / "NC_001422_1.fna"
    phix174_fasta.write_text(">NC_001422.1\nACGT\n")

    written_paths = prepare_arc_pipeline_workdir(
        source_dir,
        tmp_path / "patched",
        phix174_fasta=phix174_fasta,
        pipeline_patch=None,
    )

    assert [path.name for path in written_paths] == list(ARC_PIPELINE_FILES)
    patched_text = (tmp_path / "patched" / "genetic_architecture.py").read_text()
    assert ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA not in patched_text
    assert str(phix174_fasta) in patched_text

    pipeline_text = (tmp_path / "patched" / "genome_design_filtering_pipeline.py").read_text()
    assert ARC_LEGACY_PRODIGAL_CMD not in pipeline_text
    assert ARC_LEGACY_CHECKV_ENV not in pipeline_text
    assert ARC_LEGACY_LOVIS4U_CONDA_WRAPPER not in pipeline_text
    assert ARC_LEGACY_LOVIS4U_PDF_COLLECTION not in pipeline_text
    assert PATCHED_PRODIGAL_CMD in pipeline_text
    assert PATCHED_CHECKV_ENV in pipeline_text
    assert PATCHED_LOVIS4U_CONDA_WRAPPER in pipeline_text
    assert PATCHED_LOVIS4U_PDF_COLLECTION in pipeline_text
    assert PATCHED_MMSEQS_EMPTY_GUARD in pipeline_text
    assert PATCHED_EMPTY_ORF_GUARD in pipeline_text
    assert PATCHED_EMPTY_HOMOLOGY_GUARD in pipeline_text
    assert PATCHED_EMPTY_DIVERSIFICATION_GUARD in pipeline_text
    assert PATCHED_EMPTY_SYNTENY_GUARD in pipeline_text
    visualization_text = (tmp_path / "patched" / "genetic_architecture_visualization.py").read_text()
    assert ARC_LEGACY_LOVIS4U_PARALLEL_CONFIG not in visualization_text
    assert PATCHED_LOVIS4U_PARALLEL_CONFIG in visualization_text


def test_prepare_arc_pipeline_workdir_checks_pinned_arc_revision(tmp_path, monkeypatch):
    """The maintained Arc patch should only apply to the pinned Arc source revision."""
    source_dir = tmp_path / "source" / "phage_gen" / "pipelines"
    source_dir.mkdir(parents=True)
    for filename in ARC_PIPELINE_FILES:
        content = "print('ok')\n"
        if filename == "genetic_architecture.py":
            content = f'fasta_file = "{ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA}"\n'
        (source_dir / filename).write_text(content)
    phix174_fasta = tmp_path / "NC_001422_1.fna"
    phix174_fasta.write_text(">NC_001422.1\nACGT\n")
    patch_file = tmp_path / "arc.patch"
    patch_file.write_text("")

    monkeypatch.setattr("bionemo.evo2_phage_gen.arc_pipeline._git_head", lambda path: "wrong-revision")

    with pytest.raises(RuntimeError, match=f"{ARC_EVO2_GIT_URL}@{ARC_EVO2_REV}"):
        prepare_arc_pipeline_workdir(
            source_dir,
            tmp_path / "patched",
            phix174_fasta=phix174_fasta,
            pipeline_patch=patch_file,
        )


def test_prepare_arc_pipeline_workdir_applies_maintained_patch(tmp_path):
    """The real Arc source should be patched from the tracked maintained patch."""
    if not DEFAULT_ARC_PIPELINE_SOURCE_DIR.exists() or not DEFAULT_PHIX174_FASTA.exists():
        pytest.skip("Arc source assets are not available")

    _assert_arc_source_revision(DEFAULT_ARC_PIPELINE_SOURCE_DIR, ARC_EVO2_REV)
    workdir = tmp_path / "patched"
    prepare_arc_pipeline_workdir(
        DEFAULT_ARC_PIPELINE_SOURCE_DIR,
        workdir,
        phix174_fasta=DEFAULT_PHIX174_FASTA,
    )

    pipeline_text = (workdir / "genome_design_filtering_pipeline.py").read_text()
    assert DEFAULT_ARC_PIPELINE_PATCH.exists()
    assert "missing_synteny_output" in pipeline_text
    assert "save_mmseqs_pident_metrics" in pipeline_text
    assert "metrics_df.to_csv(metrics_csv, index=False)" in pipeline_text


def test_patched_arc_synteny_missing_lovis4u_output_fails_closed(tmp_path):
    """Missing LoVis4u files should zero synteny metrics instead of aborting RL reward scoring."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_for_test")

    metadata_dir = tmp_path / "metadata"
    gff_dir = tmp_path / "gff"
    (metadata_dir / "genome_1").mkdir(parents=True)
    gff_dir.mkdir()
    (gff_dir / "genome_1.gff").write_text("contig\ttool\tCDS\t1\t90\t.\t+\t0\tID=ORF.1;product=major spike protein\n")
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame({"id_prompt": ["umi1"], "genome_id": ["genome_1"], "total_num_genes": [1]}).to_csv(
        input_csv,
        index=False,
    )

    module.count_syntenic_genes_all(
        root_dir=str(metadata_dir),
        gff_dir=str(gff_dir),
        input_csv=str(input_csv),
        output_csv=str(output_csv),
        reference_gff_path=None,
    )

    output = pd.read_csv(output_csv)
    assert output["num_syntenic_genes"].tolist() == [0]
    assert output["non_syntenic_genes"].fillna("").tolist() == [""]
    assert output["missing_synteny_output"].tolist() == [True]


def test_patched_arc_synteny_producer_consumer_contract_tracks_positive_and_missing_outputs(tmp_path):
    """LoVis4u consumer should score real clustering output and mark missing artifacts per input."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_contract_test")

    metadata_dir = tmp_path / "metadata"
    gff_dir = tmp_path / "gff"
    positive_mmseqs_dir = metadata_dir / "genome_1" / "mmseqs"
    (metadata_dir / "genome_2").mkdir(parents=True)
    positive_mmseqs_dir.mkdir(parents=True)
    gff_dir.mkdir()

    positive_mmseqs = positive_mmseqs_dir / "mmseqs_clustering.tsv"
    positive_mmseqs.write_text("genome_1-ORF.1\treference-ORF.1\ngenome_1-ORF.2\treference-ORF.2\n")
    assert positive_mmseqs.exists()

    (gff_dir / "genome_1.gff").write_text(
        "contig\ttool\tCDS\t1\t90\t.\t+\t0\tID=ORF.1;product=major spike protein\n"
        "contig\ttool\tCDS\t100\t180\t.\t+\t0\tID=ORF.2;product=minor capsid protein\n"
    )
    (gff_dir / "genome_2.gff").write_text(
        "contig\ttool\tCDS\t1\t90\t.\t+\t0\tID=ORF.1;product=negative control protein\n"
    )
    (gff_dir / "genome_3.gff").write_text(
        "contig\ttool\tCDS\t1\t90\t.\t+\t0\tID=ORF.1;product=no lovis output protein\n"
    )
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "id_prompt": ["genome_1", "genome_2", "genome_3"],
            "genome_id": ["genome_1", "genome_2", "genome_3"],
        }
    ).to_csv(input_csv, index=False)

    module.count_syntenic_genes_all(
        root_dir=str(metadata_dir),
        gff_dir=str(gff_dir),
        input_csv=str(input_csv),
        output_csv=str(output_csv),
        reference_gff_path=None,
    )
    module.count_total_num_genes(str(gff_dir), str(output_csv))

    output = pd.read_csv(output_csv)
    assert output["id_prompt"].tolist() == ["genome_1", "genome_2", "genome_3"]
    assert output["num_syntenic_genes"].tolist() == [2, 0, 0]
    assert output["total_num_genes"].tolist() == [2, 1, 1]
    assert output["missing_synteny_output"].tolist() == [False, True, True]


def test_patched_arc_mmseqs_protein_search_fails_closed(tmp_path, monkeypatch):
    """Failed MMseqs protein searches should produce an empty hit table for online reward scoring."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_mmseqs_test")

    query_fasta = tmp_path / "query.fasta"
    query_fasta.write_text(">umi1_ORF.1\nM\n")
    mmseqs_db = tmp_path / "mmseqs_db"
    mmseqs_db.mkdir()
    output_csv = tmp_path / "hits.csv"

    def fail_mmseqs(*_args, **_kwargs):
        raise module.subprocess.CalledProcessError(returncode=1, cmd="mmseqs")

    monkeypatch.setattr(module.subprocess, "run", fail_mmseqs)

    hits = module.run_mmseqs_search_proteins(
        query_fasta=str(query_fasta),
        mmseqs_db=str(mmseqs_db),
        results_dir=str(tmp_path / "mmseqs_results"),
        output_csv=str(output_csv),
        descriptive_prefix="protein_database",
    )

    assert hits.empty
    assert output_csv.exists()


def test_patched_arc_mmseqs_protein_search_allows_successful_empty_hits(tmp_path, monkeypatch):
    """A successful MMseqs run with no hits should still produce an empty hit table."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_empty_mmseqs_test")
    query_fasta = tmp_path / "query.fasta"
    query_fasta.write_text(">umi1_ORF.1\nM\n")
    mmseqs_db = tmp_path / "mmseqs_db"
    mmseqs_db.mkdir()
    output_csv = tmp_path / "hits.csv"
    mmseqs_out = tmp_path / "mmseqs_results" / "mmseqs_out.tsv"
    mmseqs_out.parent.mkdir()
    mmseqs_out.write_text("")

    monkeypatch.setattr(module, "mmseqs_search_proteins", lambda *_args, **_kwargs: str(mmseqs_out))

    hits = module.run_mmseqs_search_proteins(
        query_fasta=str(query_fasta),
        mmseqs_db=str(mmseqs_db),
        results_dir=str(tmp_path / "mmseqs_results"),
        output_csv=str(output_csv),
        descriptive_prefix="protein_database",
    )

    assert hits.empty
    assert list(hits.columns) == [
        "id_prompt",
        "sequence",
        "protein_database_mmseqs_target",
        "protein_database_mmseqs_e_value",
        "protein_database_mmseqs_percent_identity",
    ]
    assert output_csv.exists()
