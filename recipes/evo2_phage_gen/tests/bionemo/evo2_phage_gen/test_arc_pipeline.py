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
from pathlib import Path

import pandas as pd
import pytest
from Bio import SeqIO
from Bio.Seq import Seq

import bionemo.evo2_phage_gen.arc_pipeline as arc_pipeline
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
    PATCHED_LOVIS4U_COMMAND,
    PATCHED_LOVIS4U_CONDA_WRAPPER,
    PATCHED_LOVIS4U_PARALLEL_CONFIG,
    PATCHED_LOVIS4U_PDF_COLLECTION,
    PATCHED_MMSEQS_EMPTY_GUARD,
    PATCHED_PRODIGAL_CMD,
    _apply_lovis4u_runtime_patches,
    _apply_online_measurement_patches,
    _assert_arc_source_revision,
    prepare_arc_pipeline_workdir,
)
from bionemo.evo2_phage_gen.external_qc import ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA


def _load_prepared_arc_pipeline(tmp_path: Path, module_name: str, monkeypatch):
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
    monkeypatch.syspath_prepend(str(pipeline_path.parent))
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_synthetic_mmseqs_pipeline(tmp_path: Path, module_name: str):
    """Load the Arc protein-search fragment after applying the runtime evidence patch."""
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """import os
import subprocess
import time

import pandas as pd
from Bio import SeqIO


def mmseqs_search_proteins(query_fasta: str, mmseqs_db: str, results_dir: str, threads: int=8, split: int=0, sensitivity: float=4.0) -> None:
    os.makedirs(results_dir, exist_ok=True)
    mmseqs_out = os.path.join(results_dir, "mmseqs_result.m8")
    log_file = os.path.join(results_dir, "mmseqs_search.log")
    cmd = f"mmseqs easy-search {query_fasta} {mmseqs_db} {mmseqs_out} {results_dir} --threads {threads} --split {split} -s {sensitivity} --remove-tmp-files 1 --format-output 'query,target,evalue,pident'"
    start_time = time.time()
    with open(log_file, "w") as log:
        subprocess.run(cmd, shell=True, check=True, stdout=log, stderr=log, text=True)
    end_time = time.time()
    print(f"MMseqs2 search completed in {end_time - start_time:.2f} seconds.")
    if not os.path.isfile(mmseqs_out):
        raise FileNotFoundError(f"Output file not found: {mmseqs_out}")
    return mmseqs_out


def parse_mmseqs_results(mmseqs_out):
    hits = []
    with open(mmseqs_out, "r") as f:
        for line in f:
            query, target, evalue, pident = line.strip().split('\\t')
            hits.append((query, target, float(evalue), float(pident)))
    return hits


def mmseqs_results_to_df(hits, query_fasta: str, output_csv: str, descriptive_prefix: str, only_top_hits: bool=True) -> pd.DataFrame:
    sequences = {record.id: record.seq for record in SeqIO.parse(query_fasta, "fasta")}
    data = []
    for query, target, evalue, pident in hits:
        if query in sequences:
            data.append([query, sequences[query], target, evalue, pident])
    df = pd.DataFrame(data, columns=["id_prompt", "sequence", f"{descriptive_prefix}_mmseqs_target", f"{descriptive_prefix}_mmseqs_e_value", f"{descriptive_prefix}_mmseqs_percent_identity"])
    if only_top_hits==True and not df.empty:
        df = df.loc[df.groupby("id_prompt")[f"{descriptive_prefix}_mmseqs_e_value"].idxmin()]
    df.to_csv(output_csv, index=False)
    return df


def run_mmseqs_search_proteins(query_fasta: str, mmseqs_db: str, results_dir: str, output_csv: str, descriptive_prefix: str, threads: int=8, split: int=0, sensitivity: float=4.0, only_top_hits: bool=True) -> pd.DataFrame:
    try:
        mmseqs_out = mmseqs_search_proteins(query_fasta, mmseqs_db, results_dir, threads, split, sensitivity)
        hits = parse_mmseqs_results(mmseqs_out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        hits = []
    if not hits:
        df = pd.DataFrame(
            columns=[
                "id_prompt",
                "sequence",
                f"{descriptive_prefix}_mmseqs_target",
                f"{descriptive_prefix}_mmseqs_e_value",
                f"{descriptive_prefix}_mmseqs_percent_identity",
            ]
        )
        df.to_csv(output_csv, index=False)
        return df
    return mmseqs_results_to_df(hits, query_fasta, output_csv, descriptive_prefix, only_top_hits)
"""
    )
    arc_pipeline._apply_mmseqs_protein_evidence_patch(tmp_path)
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lovis4u_patch_handles_source_trailing_whitespace(tmp_path):
    """The compatible Arc source has a trailing space after the LoVis4u executable."""
    visualization_path = tmp_path / "genetic_architecture_visualization.py"
    visualization_path.write_text("    command = [\n        'lovis4u', \n        '-gff', input_gff_dir,\n    ]\n")

    _apply_lovis4u_runtime_patches(tmp_path)

    assert PATCHED_LOVIS4U_COMMAND in visualization_path.read_text()


def test_online_measurement_patch_rejects_missing_gbk_conversion_anchor(tmp_path):
    """A drifted GBK conversion anchor must not silently skip its online patch."""
    patched_fragments = [
        value
        for name, value in vars(arc_pipeline).items()
        if name.startswith(("PATCHED_ONLINE_", "PATCHED_REQUIRED_GENE_", "PATCHED_AAI_", "PATCHED_SYNTENY_"))
        and name != "PATCHED_ONLINE_GBK_CONVERSION"
    ]
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text("\n".join(patched_fragments))

    with pytest.raises(ValueError, match="online objective-measurement patches"):
        _apply_online_measurement_patches(tmp_path)


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
    assert "if os.path.exists(synteny_counts_csv):" in pipeline_text
    assert "synteny_filter_counts = pd.read_csv(synteny_counts_csv)" in pipeline_text
    assert not any(
        line.strip().startswith("filter_counts = pd.read_csv(synteny_counts_csv)")
        for line in pipeline_text.splitlines()
    )
    visualization_text = (tmp_path / "patched" / "genetic_architecture_visualization.py").read_text()
    assert ARC_LEGACY_LOVIS4U_PARALLEL_CONFIG not in visualization_text
    assert PATCHED_LOVIS4U_PARALLEL_CONFIG in visualization_text


def test_prepare_arc_pipeline_resolves_reference_path_before_runtime_cwd_changes(tmp_path, monkeypatch):
    """Prepared Arc imports must not depend on the launcher's later working directory."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for filename in ARC_PIPELINE_FILES:
        content = "print('ok')\n"
        if filename == "genetic_architecture.py":
            content = f'fasta_file = "{ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA}"\n'
        (source_dir / filename).write_text(content)
    phix174_fasta = tmp_path / "reference.fna"
    phix174_fasta.write_text(">NC_001422.1\nACGT\n")
    monkeypatch.chdir(tmp_path)

    prepare_arc_pipeline_workdir(
        Path("source"),
        Path("prepared"),
        phix174_fasta=Path("reference.fna"),
        pipeline_patch=None,
    )

    prepared = (tmp_path / "prepared" / "genetic_architecture.py").read_text()
    assert str(phix174_fasta.resolve()) in prepared


@pytest.mark.parametrize(
    ("header", "expected_start", "expected_end"),
    [
        ("candidate_ORF.1 [567-843](+)", "568", "843"),
        ("candidate_ORF.2 [9-90](-)", "10", "90"),
    ],
)
def test_prepared_arc_writes_orfipy_intervals_as_one_based_inclusive_gff(
    tmp_path,
    header,
    expected_start,
    expected_end,
):
    """ORFipy's zero-based half-open interval must not shift GFF translations by one base."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pipeline_source = """import re

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

def render_orfipy_interval_as_gff(header):
    match = re.search(r"\\[(\\d+)-(\\d+)\\]", header)
    if match:
        start, end = match.groups()
        orf_entry = {
            "seq_id": "candidate",
            "feature_type": "CDS",
            "start": start,
            "end": end,
        }
        return (
            f"{orf_entry['seq_id']}\\tcaller\\t{orf_entry['feature_type']}\\t"
            f"{orf_entry['start']}\\t{orf_entry['end']}\\t.\\t+\\t0\\tID=ORF.1"
        )
    raise ValueError("missing ORFipy interval")


def convert_gff_to_gbk(sequence, start, end, output_path, strand=1):
    record = SeqRecord(Seq(sequence), id="candidate", name="candidate")
    record.annotations["molecule_type"] = "DNA"
    feature = SeqFeature(
        location=FeatureLocation(start, end, strand=strand),
        type="CDS",
    )
    feature.qualifiers["translation"] = [str(feature.extract(record.seq).translate())]
    record.features = [feature]
    SeqIO.write(record, output_path, "genbank")
"""
    for filename in ARC_PIPELINE_FILES:
        content = pipeline_source if filename == "genome_design_filtering_pipeline.py" else "print('ok')\n"
        if filename == "genetic_architecture.py":
            content = f'fasta_file = "{ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA}"\n'
        (source_dir / filename).write_text(content)
    reference_fasta = tmp_path / "reference.fna"
    reference_fasta.write_text(">reference\nACGT\n")
    workdir = tmp_path / "prepared"

    prepare_arc_pipeline_workdir(
        source_dir,
        workdir,
        phix174_fasta=reference_fasta,
        pipeline_patch=None,
    )
    pipeline_path = workdir / "genome_design_filtering_pipeline.py"
    spec = importlib.util.spec_from_file_location("coordinate_corrected_arc_pipeline", pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fields = module.render_orfipy_interval_as_gff(header).split("\t")
    assert fields[3:5] == [expected_start, expected_end]
    start = int(expected_start) - 1
    coding_length = int(expected_end) - start
    if header.startswith("candidate_ORF.1"):
        coding_sequence = Seq(
            "ATGGTACGCTGGACTTTGTGGGATACCCTCGCTTTCCTGCTCCTGTTGAGTTTATTGCTGCCGTCATTGCTTATTATGTT"
            "CATCCCGTCAACATTCAAACGGCCTGTCTCATCATGGAAGGCGCTGAATTTACGGAAAACATTATTAATGGCGTCGAGCGT"
            "CCGGTTAAAGCCGCTGAATTGTTCGCGTTTACCTTGCGTGTACGCGCAGGAAACACTGACGTTCTTACTGACGCAGAAGAA"
            "AACGTGCGTCAAAAATTACGTGCGGAAGGAGTGA"
        )
    else:
        coding_sequence = Seq("ATG" + "GCT" * (coding_length // 3 - 2) + "TAA")
    genome = Seq("A" * start) + coding_sequence
    gbk_path = tmp_path / f"{expected_start}.gbk"
    module.convert_gff_to_gbk(str(genome), int(fields[3]), int(fields[4]), gbk_path)

    converted = SeqIO.read(gbk_path, "genbank")
    for feature in (feature for feature in converted.features if feature.type == "CDS"):
        assert (int(feature.location.start), int(feature.location.end)) == (start, int(expected_end))
        assert len(feature) % 3 == 0
        assert "*" not in feature.qualifiers["translation"][0][:-1]


def test_prepare_arc_pipeline_requires_compatible_arc_revision(tmp_path, monkeypatch):
    """The maintained Arc patch should only apply to its compatible Arc source revision."""
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
    assert "online_measurement_mode" in pipeline_text
    assert "Skipping unconsumed GBK conversion during online measurement." in pipeline_text
    assert 'config.get("lovis4u_collect_pdfs", True)' in pipeline_text
    assert "Skipping LoVis4u PDF collection" in pipeline_text
    visualization_text = (workdir / "genetic_architecture_visualization.py").read_text()
    assert "bionemo.evo2_phage_gen.lovis4u_metrics" in visualization_text
    assert 'config.get("lovis4u_mmseqs_threads")' in visualization_text


def test_maintained_arc_patch_rotates_only_candidate_and_guards_empty_synteny_counts() -> None:
    """Circular LCS work stays bounded and empty synteny output tolerates absent counts."""
    patch_text = DEFAULT_ARC_PIPELINE_PATCH.read_text()
    assert "+    def best_circular_lcs_candidate_indices" in patch_text
    helper = patch_text.split("+    def best_circular_lcs_candidate_indices", 1)[1].split(
        "+    def count_syntenic_genes_from_gff_products", 1
    )[0]

    assert "reference_offset" not in helper
    assert "for candidate_offset in range(len(candidate_products))" in helper
    assert "if os.path.exists(synteny_counts_csv):" in patch_text
    assert "synteny_filter_counts = pd.read_csv(synteny_counts_csv)" in patch_text


def test_maintained_arc_patch_uses_local_empty_input_run_state() -> None:
    """Empty inputs should skip work without changing the caller's Arc config."""
    patch_text = DEFAULT_ARC_PIPELINE_PATCH.read_text()

    assert 'config["prodigal_based_filters"] = False' not in patch_text
    assert 'config["protein_database_hit_count_filter"] = False' not in patch_text
    assert 'config["mmseqs_clustering_filter"] = False' not in patch_text
    assert 'config["genetic_architecture_visualization_and_synteny_filtering"] = False' not in patch_text
    assert "run_prodigal_based_filters" in patch_text
    assert "+        run_orfipy = " not in patch_text
    assert "run_protein_database_hit_count_filter" in patch_text
    assert "run_mmseqs_clustering_filter" in patch_text
    assert "run_genetic_architecture_visualization_and_synteny_filtering" in patch_text


def test_maintained_patch_honors_lovis4u_pdf_collection_flag():
    patch_text = DEFAULT_ARC_PIPELINE_PATCH.read_text()
    assert '+            if config.get("lovis4u_collect_pdfs", True):' in patch_text
    assert "Skipping LoVis4u PDF collection" in patch_text


def test_patched_arc_required_gene_measurement_does_not_filter_or_delete(tmp_path, monkeypatch):
    """Online rewards should measure required genes without starving later objectives."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_measurement_test", monkeypatch)
    gff_dir = tmp_path / "gff"
    gbk_dir = tmp_path / "gbk"
    gff_dir.mkdir()
    gbk_dir.mkdir()
    (gff_dir / "genome_1.gff").write_text("contig\ttool\tCDS\t1\t90\t.\t+\t0\tID=ORF.1;product=major capsid protein\n")
    (gbk_dir / "genome_1.gbk").write_text("LOCUS genome_1\n")
    sequences = pd.DataFrame({"id_prompt": ["umi1"], "genome_id": ["genome_1"], "sequence": ["ACGT"]})
    metrics_csv = tmp_path / "required.csv"

    measured = module.valid_gene_annotations(
        input_gff_dir=str(gff_dir),
        input_gbk_dir=str(gbk_dir),
        required_products=("major capsid protein", "missing protein"),
        sequences_df=sequences,
        metrics_csv=str(metrics_csv),
        filter_results=False,
    )

    assert measured["id_prompt"].tolist() == ["umi1"]
    assert (gff_dir / "genome_1.gff").exists()
    assert (gbk_dir / "genome_1.gbk").exists()
    metrics = pd.read_csv(metrics_csv)
    assert metrics["required_genes_matched_count"].tolist() == [0]


def test_patched_arc_required_gene_evidence_is_fractional_duplicate_safe_and_shared_with_hard_qc(tmp_path):
    """Online required-family credit and offline rejection must use the same reciprocal-coverage evidence."""
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """import os
import shutil

import pandas as pd


def valid_gene_annotations(input_gff_dir, input_gbk_dir, required_products, sequences_df, metrics_csv=None, filter_results=True):
    return sequences_df


##############################
### RUN FILTERING PIPELINE ###
##############################
"""
    )
    arc_pipeline._apply_required_gene_evidence_patch(tmp_path)
    spec = importlib.util.spec_from_file_location("patched_required_gene_evidence", pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gff_dir = tmp_path / "gff"
    gbk_dir = tmp_path / "gbk"
    gff_dir.mkdir()
    gbk_dir.mkdir()
    (gff_dir / "genome_1.gff").write_text("##gff-version 3\n")
    (gbk_dir / "genome_1.gbk").write_text("LOCUS genome_1\n")
    sequences = pd.DataFrame({"id_prompt": ["umi1"], "genome_id": ["genome_1"], "sequence": ["ACGT"]})
    hits = pd.DataFrame(
        {
            "id_prompt": ["umi1_ORF.1", "umi1_ORF.2", "umi1_ORF.3"],
            "annot": ["gene A", "gene A", "gene B"],
            "protein_database_mmseqs_target": ["family_A", "family_A", "family_B"],
            "protein_database_mmseqs_percent_identity": [100.0, 100.0, 100.0],
            "protein_database_mmseqs_alignment_length": [100, 100, 50],
            "protein_database_mmseqs_query_length": [100, 100, 50],
            "protein_database_mmseqs_target_length": [100, 100, 100],
        }
    )
    metrics_csv = tmp_path / "required.csv"

    measured = module.valid_gene_annotations(
        input_gff_dir=str(gff_dir),
        input_gbk_dir=str(gbk_dir),
        required_products=("gene A", "gene B"),
        sequences_df=sequences,
        metrics_csv=str(metrics_csv),
        filter_results=False,
        protein_database_hits_df=hits,
        minimum_reciprocal_coverage=0.95,
    )

    assert measured["id_prompt"].tolist() == ["umi1"]
    metrics = pd.read_csv(metrics_csv)
    assert metrics["required_genes_matched_count"].tolist() == [2]
    assert metrics["required_genes_integrity_sum"].tolist() == pytest.approx([1.5])
    assert metrics["required_genes_full_length_count"].tolist() == [1]
    assert (gff_dir / "genome_1.gff").exists()
    assert (gbk_dir / "genome_1.gbk").exists()

    filtered = module.valid_gene_annotations(
        input_gff_dir=str(gff_dir),
        input_gbk_dir=str(gbk_dir),
        required_products=("gene A", "gene B"),
        sequences_df=sequences,
        metrics_csv=str(metrics_csv),
        filter_results=True,
        protein_database_hits_df=hits,
        minimum_reciprocal_coverage=0.95,
    )

    assert filtered.empty
    assert not (gff_dir / "genome_1.gff").exists()
    assert not (gbk_dir / "genome_1.gbk").exists()


def test_required_gene_patch_preserves_composed_hard_gate_imports(tmp_path):
    """The production patch chain must leave every injected protein gate bound."""
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """import os
import shutil

import pandas as pd


def valid_gene_annotations(input_gff_dir, input_gbk_dir, required_products, sequences_df, metrics_csv=None):
    return sequences_df


##############################
### RUN FILTERING PIPELINE ###
##############################
"""
    )
    arc_pipeline._apply_protein_hard_gate_patch(tmp_path)
    arc_pipeline._apply_required_gene_evidence_patch(tmp_path)

    spec = importlib.util.spec_from_file_location("composed_protein_gate_patches", pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.valid_coverage_aware_protein_database_hit_count)
    assert callable(module.valid_coverage_aware_mmseqs_pident)


def test_patched_arc_aai_excludes_truncated_proteins(tmp_path):
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """import os

import pandas as pd


def valid_average_protein_percent_identity(gff_directory, gbk_directory, results_csv, output_csv, identity_range):
    pass


def count_total_num_genes(gff_directory, results_csv):
    pass
"""
    )
    arc_pipeline._apply_aai_evidence_patch(tmp_path)
    spec = importlib.util.spec_from_file_location("patched_aai_evidence", pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "id_prompt": ["truncated", "complete"],
            "genome_id": ["genome_1", "genome_2"],
        }
    ).to_csv(input_csv, index=False)
    hits = pd.DataFrame(
        {
            "id_prompt": ["truncated_ORF.1", "complete_ORF.1"],
            "protein_database_mmseqs_target": ["family_A", "family_A"],
            "protein_database_mmseqs_percent_identity": [100.0, 80.0],
            "protein_database_mmseqs_alignment_length": [50, 100],
            "protein_database_mmseqs_query_length": [50, 100],
            "protein_database_mmseqs_target_length": [100, 100],
        }
    )

    module.valid_average_protein_percent_identity(
        str(tmp_path / "gff"),
        str(tmp_path / "gbk"),
        str(input_csv),
        str(output_csv),
        (0, 95),
        filter_results=False,
        protein_database_hits_df=hits,
        minimum_reciprocal_coverage=0.75,
        metrics_csv=str(metrics_csv),
    )

    metrics = pd.read_csv(metrics_csv)
    assert metrics["average_protein_percent_identity"].tolist() == [0.0, 80.0]
    assert metrics["average_protein_identity_gene_count"].tolist() == [0, 1]


def test_reference_cluster_patch_replaces_arc_edge_counter(tmp_path):
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """def count_syntenic_genes_all(root_dir, gff_dir, input_csv, output_csv):
    raise NotImplementedError


def valid_syntenic_gene_count(input_csv, output_csv):
    pass
"""
    )

    arc_pipeline._apply_reference_cluster_evidence_patch(tmp_path)

    patched = pipeline_path.read_text()
    compile(patched, str(pipeline_path), "exec")
    assert "measure_reference_cluster_architecture" in patched
    assert "raise NotImplementedError" not in patched


def test_patched_arc_hard_protein_gates_require_unique_full_length_families(tmp_path):
    """Final Arc protein-count and tropism gates must reject duplicates and high-identity fragments."""
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """import pandas as pd


##############################
### RUN FILTERING PIPELINE ###
##############################
"""
    )
    arc_pipeline._apply_protein_hard_gate_patch(tmp_path)
    spec = importlib.util.spec_from_file_location("patched_protein_hard_gates", pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sequences = pd.DataFrame({"id_prompt": ["duplicate", "complete"], "sequence": ["ACGT", "ACGT"]})
    protein_hits = pd.DataFrame(
        {
            "id_prompt": [
                "duplicate_ORF.1",
                "duplicate_ORF.2",
                "duplicate_ORF.3",
                "complete_ORF.1",
                "complete_ORF.2",
            ],
            "protein_database_mmseqs_target": ["family_A", "family_A", "family_B", "family_A", "family_B"],
            "protein_database_mmseqs_percent_identity": [100.0, 100.0, 100.0, 100.0, 100.0],
            "protein_database_mmseqs_alignment_length": [100, 100, 50, 100, 100],
            "protein_database_mmseqs_query_length": [100, 100, 50, 100, 100],
            "protein_database_mmseqs_target_length": [100, 100, 100, 100, 100],
        }
    )

    protein_pass = module.valid_coverage_aware_protein_database_hit_count(
        protein_hits,
        sequences,
        id_column="id_prompt",
        min_hits=2,
        minimum_reciprocal_coverage=0.95,
    )

    assert protein_pass["id_prompt"].tolist() == ["complete"]
    assert protein_pass["protein_database_hit_count"].tolist() == [2]

    tropism_hits = pd.DataFrame(
        {
            "id_prompt": ["duplicate_ORF.1", "complete_ORF.1"],
            "tropism_protein_mmseqs_percent_identity": [100.0, 60.0],
            "tropism_protein_mmseqs_alignment_length": [50, 100],
            "tropism_protein_mmseqs_query_length": [50, 100],
            "tropism_protein_mmseqs_target_length": [100, 100],
        }
    )

    tropism_pass = module.valid_coverage_aware_mmseqs_pident(
        tropism_hits,
        "tropism_protein",
        [60, 100],
        sequences,
        minimum_reciprocal_coverage=0.95,
    )

    assert tropism_pass["id_prompt"].tolist() == ["complete"]
    assert tropism_pass["tropism_protein_mmseqs_percent_identity"].tolist() == [60.0]


def test_patched_arc_synteny_missing_lovis4u_output_receives_zero_credit(tmp_path, monkeypatch):
    """Missing LoVis4u files should zero synteny metrics instead of aborting RL reward scoring."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_for_test", monkeypatch)

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


def test_patched_arc_synteny_producer_consumer_contract_tracks_positive_and_missing_outputs(tmp_path, monkeypatch):
    """LoVis4u consumer should score real clustering output and mark missing artifacts per input."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_contract_test", monkeypatch)

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


def test_patched_arc_mmseqs_protein_search_rejects_missing_output(tmp_path, monkeypatch):
    """Failed MMseqs protein searches should produce an empty hit table for online reward scoring."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_mmseqs_test", monkeypatch)

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
    module = _load_synthetic_mmseqs_pipeline(tmp_path, "patched_arc_pipeline_empty_mmseqs_test")
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
        "protein_database_mmseqs_alignment_length",
        "protein_database_mmseqs_query_length",
        "protein_database_mmseqs_target_length",
    ]
    assert output_csv.exists()


def test_patched_arc_mmseqs_protein_search_carries_alignment_lengths(tmp_path, monkeypatch):
    """Arc must retain the MMseqs lengths needed to distinguish fragments from intact proteins."""
    module = _load_synthetic_mmseqs_pipeline(tmp_path, "patched_arc_pipeline_coverage_test")
    query_fasta = tmp_path / "query.fasta"
    query_fasta.write_text(">umi1_ORF.1\n" + "M" * 80 + "\n")
    mmseqs_db = tmp_path / "mmseqs_db"
    mmseqs_db.mkdir()
    output_csv = tmp_path / "hits.csv"
    mmseqs_out = tmp_path / "mmseqs_results" / "mmseqs_out.tsv"
    mmseqs_out.parent.mkdir()
    mmseqs_out.write_text("umi1_ORF.1\tphrog_1\t1e-20\t75.0\t40\t80\t100\n")

    monkeypatch.setattr(module, "mmseqs_search_proteins", lambda *_args, **_kwargs: str(mmseqs_out))

    hits = module.run_mmseqs_search_proteins(
        query_fasta=str(query_fasta),
        mmseqs_db=str(mmseqs_db),
        results_dir=str(tmp_path / "mmseqs_results"),
        output_csv=str(output_csv),
        descriptive_prefix="protein_database",
    )

    assert hits.loc[0, "protein_database_mmseqs_alignment_length"] == 40
    assert hits.loc[0, "protein_database_mmseqs_query_length"] == 80
    assert hits.loc[0, "protein_database_mmseqs_target_length"] == 100


def test_patched_arc_mmseqs_search_requests_alignment_length_fields(tmp_path, monkeypatch):
    """The MMseqs command contract must request reciprocal-coverage inputs explicitly."""
    module = _load_synthetic_mmseqs_pipeline(tmp_path, "patched_arc_pipeline_command_test")
    query_fasta = tmp_path / "query.fasta"
    query_fasta.write_text(">umi1_ORF.1\nM\n")
    mmseqs_db = tmp_path / "mmseqs_db"
    mmseqs_db.mkdir()
    results_dir = tmp_path / "mmseqs_results"

    def fake_run(cmd, **_kwargs):
        assert "--format-output 'query,target,evalue,pident,alnlen,qlen,tlen'" in cmd
        (results_dir / "mmseqs_result.m8").write_text("")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    observed_path = module.mmseqs_search_proteins(
        query_fasta=str(query_fasta),
        mmseqs_db=str(mmseqs_db),
        results_dir=str(results_dir),
    )

    assert observed_path == str(results_dir / "mmseqs_result.m8")
