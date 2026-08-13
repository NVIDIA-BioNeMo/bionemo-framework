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

import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import bionemo.evo2_phage_gen.external_assets as external_assets
import bionemo.evo2_phage_gen.sequence_safety_adapters as sequence_safety_adapters
from bionemo.evo2_phage_gen.design_scope import HostDomain
from bionemo.evo2_phage_gen.sequence_safety import SafetyFinding, SafetyState
from bionemo.evo2_phage_gen.sequence_safety_adapters import (
    PHROGS_HOMOLOGY_POLICY_V1,
    TOXIN_HOMOLOGY_POLICY_V1,
    TOXIN_HOMOLOGY_POLICY_V2,
    GenomeInput,
    HomologyBand,
    HomologyPolicy,
    NormalizedSafetyFinding,
    ORFQueryRecord,
    PredictedGene,
    ToolPin,
    VerifiedIdentityMappingMissingError,
    _parse_amrfinder_output_validated,
    _parse_phrogs_output_validated,
    _parse_toxin_diamond_output_validated,
    _read_phrogs_profile_ids,
    build_amrfinder_command,
    build_diamond_command,
    build_phrogs_command,
    build_phrogs_commands,
    parse_amrfinder_output,
    parse_phrogs_output,
    parse_toxin_diamond_output,
    prepare_orf_artifacts,
    prepare_orf_artifacts_checked,
    run_amrfinder,
    run_phrogs,
    run_toxin_diamond,
    validate_tool_pin,
)


_AMRFINDER_HEADER = (
    "Protein id",
    "Contig id",
    "Start",
    "Stop",
    "Strand",
    "Element symbol",
    "Element name",
    "Scope",
    "Type",
    "Subtype",
    "Class",
    "Subclass",
    "Method",
    "Target length",
    "Reference sequence length",
    "% Coverage of reference",
    "% Identity to reference",
    "Alignment length",
    "Closest reference accession",
    "Closest reference name",
    "HMM accession",
    "HMM description",
    "Hierarchy node",
)


def test_amrfinder_combined_command_uses_curated_thresholds(tmp_path):
    """Dropping combined inputs or adding identity overrides would weaken AMRFinder's curated call."""
    command = build_amrfinder_command(
        amrfinder=Path("/tools/amrfinder"),
        genomes_fna=tmp_path / "genomes.fna",
        proteins_faa=tmp_path / "proteins.faa",
        proteins_gff=tmp_path / "proteins.gff",
        database_dir=Path("/db/amrfinder/2026-07-22.1"),
        blast_bin_dir=Path("/tools/blast"),
        hmmer_bin_dir=Path("/tools/hmmer"),
        threads=7,
        output_tsv=tmp_path / "amrfinder.tsv",
    )

    assert command == [
        "/tools/amrfinder",
        "-n",
        str(tmp_path / "genomes.fna"),
        "-p",
        str(tmp_path / "proteins.faa"),
        "-g",
        str(tmp_path / "proteins.gff"),
        "--annotation_format",
        "standard",
        "--plus",
        "--print_node",
        "--database",
        "/db/amrfinder/2026-07-22.1",
        "--blast_bin",
        "/tools/blast",
        "--hmmer_bin",
        "/tools/hmmer",
        "--threads",
        "7",
        "-o",
        str(tmp_path / "amrfinder.tsv"),
    ]


def test_normalized_finding_is_a_canonical_immutable_safety_finding():
    """Losing structured evidence would make downstream manifests unable to audit a hit."""
    finding = NormalizedSafetyFinding(
        safety_class="toxin",
        state=SafetyState.FAIL,
        reason_codes=("TOXIN_HIGH_CONFIDENCE",),
        finding_id="toxin:orf_genome_1_0001:P0C1",
        detector="diamond-reviewed-toxin",
        accession="P0C1",
        query_id="orf_genome_1_0001",
        sequence_id="genome_1",
        start=4,
        end=93,
        strand="+",
        frame=1,
        scores={"identity": 87.5, "query_coverage": 96.0, "reference_coverage": 91.0, "evalue": 1e-30},
        thresholds={"identity": 80.0, "query_coverage": 80.0, "reference_coverage": 80.0, "evalue": 1e-10},
        source_path="/db/reviewed_toxins.dmnd",
        source_sha256="a" * 64,
        tool_version="diamond version 2.1.24",
        database_version="UniProt 2026_03",
        evidence_path="pyrodigal-gv",
        evidence_method="blastp",
        threshold_policy="toxin-homology-v1",
        threshold_policy_sha256="d" * 64,
        tool_path="/tools/diamond",
        tool_sha256="b" * 64,
        profile=None,
    )

    assert isinstance(finding, SafetyFinding)
    assert finding.to_dict() == {
        "safety_class": "toxin",
        "state": "FAIL",
        "reason_codes": ["TOXIN_HIGH_CONFIDENCE"],
        "finding_id": "toxin:orf_genome_1_0001:P0C1",
        "detector": "diamond-reviewed-toxin",
        "accession": "P0C1",
        "query_id": "orf_genome_1_0001",
        "sequence_id": "genome_1",
        "start": 4,
        "end": 93,
        "strand": "+",
        "frame": 1,
        "scores": {"identity": 87.5, "query_coverage": 96.0, "reference_coverage": 91.0, "evalue": 1e-30},
        "thresholds": {
            "identity": 80.0,
            "query_coverage": 80.0,
            "reference_coverage": 80.0,
            "evalue": 1e-10,
        },
        "source_path": "/db/reviewed_toxins.dmnd",
        "source_sha256": "a" * 64,
        "tool_version": "diamond version 2.1.24",
        "database_version": "UniProt 2026_03",
        "evidence_path": "pyrodigal-gv",
        "evidence_method": "blastp",
        "threshold_policy": "toxin-homology-v1",
        "threshold_policy_sha256": "d" * 64,
        "tool_path": "/tools/diamond",
        "tool_sha256": "b" * 64,
        "profile": None,
    }

    with pytest.raises(FrozenInstanceError):
        finding.query_id = "changed"
    with pytest.raises(TypeError):
        finding.scores["identity"] = 0.0

    restored = NormalizedSafetyFinding.from_dict(finding.to_dict())
    assert restored == finding
    with pytest.raises(TypeError):
        restored.thresholds["identity"] = 0.0


@pytest.mark.parametrize(
    "high",
    [
        HomologyBand(identity=39.0, query_coverage=80.0, reference_coverage=80.0, evalue=1e-10),
        HomologyBand(identity=80.0, query_coverage=59.0, reference_coverage=80.0, evalue=1e-10),
        HomologyBand(identity=80.0, query_coverage=80.0, reference_coverage=59.0, evalue=1e-10),
        HomologyBand(identity=80.0, query_coverage=80.0, reference_coverage=80.0, evalue=1e-4),
    ],
)
def test_homology_policy_rejects_a_high_band_weaker_than_review(high):
    """A mislabeled high band must never weaken any jointly required review threshold."""
    with pytest.raises(ValueError, match="high-confidence band"):
        HomologyPolicy(
            policy_id="invalid-policy-v1",
            high=high,
            review=HomologyBand(
                identity=40.0,
                query_coverage=60.0,
                reference_coverage=60.0,
                evalue=1e-5,
            ),
        )


def test_homology_policy_has_canonical_serialization_and_sha256():
    """Task 4 needs a stable digest that changes with any threshold or policy identifier."""
    expected = {
        "policy_id": "toxin-homology-v1",
        "high": {
            "evalue": 1e-10,
            "identity": 80.0,
            "query_coverage": 80.0,
            "reference_coverage": 80.0,
        },
        "review": {
            "evalue": 1e-5,
            "identity": 40.0,
            "query_coverage": 60.0,
            "reference_coverage": 60.0,
        },
    }
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()

    assert TOXIN_HOMOLOGY_POLICY_V1.to_dict() == expected
    assert TOXIN_HOMOLOGY_POLICY_V1.sha256 == hashlib.sha256(encoded).hexdigest()


class _DeterministicGenePredictor:
    def predict(self, sequence: str, *, circular: bool) -> tuple[PredictedGene, ...]:
        assert circular is True
        assert len(sequence) == 210
        return (
            PredictedGene(start=4, end=93, strand="+", nucleotide="ATG" * 30, protein="M" * 30),
            PredictedGene(start=150, end=240, strand="-", nucleotide="ATG" * 30, protein="K" * 30),
        )


def test_orf_artifacts_coordinate_fasta_gff_and_circular_both_strand_calls(tmp_path):
    """Mismatched IDs or clipped cross-origin calls would break combined AMRFinder association."""
    artifacts = prepare_orf_artifacts(
        (GenomeInput(sequence_id="genome_1", sequence="ACG" * 70, circular=True),),
        tmp_path,
        predictor=_DeterministicGenePredictor(),
        minimum_fallback_amino_acids=3,
    )

    assert artifacts.genomes_fna.read_text() == f">genome_1 circular=true\n{'ACG' * 70}\n"
    assert artifacts.proteins_faa.read_text() == (f">genome_1__orf0001\n{'M' * 30}\n>genome_1__orf0002\n{'K' * 30}\n")
    assert artifacts.proteins_fna.read_text() == (
        f">genome_1__orf0001\n{'ATG' * 30}\n>genome_1__orf0002\n{'ATG' * 30}\n"
    )
    assert artifacts.proteins_gff.read_text().splitlines() == [
        "##gff-version 3",
        "##sequence-region genome_1 1 210",
        "genome_1\tpyrodigal-gv\tCDS\t4\t93\t.\t+\t0\tID=genome_1__orf0001;Name=genome_1__orf0001",
        "genome_1\tpyrodigal-gv\tCDS\t150\t240\t.\t-\t0\tID=genome_1__orf0002;Name=genome_1__orf0002",
    ]
    predicted = artifacts.query_records[:2]
    assert [(record.query_id, record.strand, record.frame, record.evidence_path) for record in predicted] == [
        ("genome_1__orf0001", "+", 1, "pyrodigal-gv"),
        ("genome_1__orf0002", "-", -1, "pyrodigal-gv"),
    ]
    fallback = artifacts.query_records[2:]
    assert {record.frame for record in fallback} == {-3, -2, -1, 1, 2, 3}
    assert {record.evidence_path for record in fallback} == {"six-frame-fallback"}
    assert set(artifacts.all_queries_faa.read_text().splitlines()[::2]) == {
        f">{record.query_id}" for record in artifacts.query_records
    }


def test_orf_artifacts_reject_duplicate_sequence_ids_before_prediction(tmp_path):
    """Duplicate input IDs would make scanner hits ambiguous across genomes."""
    genomes = (
        GenomeInput(sequence_id="duplicate", sequence="ATG" * 30),
        GenomeInput(sequence_id="duplicate", sequence="ATG" * 30),
    )

    with pytest.raises(ValueError, match="duplicate genome sequence ID"):
        prepare_orf_artifacts(genomes, tmp_path, predictor=_DeterministicGenePredictor())


def test_genome_id_rejects_characters_that_diverge_between_fasta_and_gff_attributes():
    """Allowing an ID that GFF escapes would break AMRFinder's exact Name-to-protein join."""
    with pytest.raises(ValueError, match="invalid genome sequence ID"):
        GenomeInput(sequence_id="genome:1", sequence="ACG" * 30)


def test_missing_pyrodigal_runtime_is_fail_closed_without_import_time_failure(tmp_path, monkeypatch):
    """An unavailable required predictor must become INDETERMINATE rather than a silent fallback pass."""

    def missing_predictor():
        raise ImportError("pyrodigal_gv unavailable")

    monkeypatch.setattr(
        "bionemo.evo2_phage_gen.sequence_safety_adapters._new_pyrodigal_predictor",
        missing_predictor,
    )

    result = prepare_orf_artifacts_checked(
        (GenomeInput(sequence_id="genome_1", sequence="ATG" * 30),),
        tmp_path,
    )

    assert result.state is SafetyState.INDETERMINATE
    assert result.artifacts is None
    assert result.reason_codes == ("ORF_PREDICTOR_UNAVAILABLE",)


def test_lazy_pyrodigal_boundary_keeps_center_copy_cross_origin_calls(tmp_path, monkeypatch):
    """Circular calls are retained by centered-copy start and de-duplicated on both strands."""
    observed = {}

    class FakeGene:
        def __init__(self, begin, end, strand, protein):
            self.begin = begin
            self.end = end
            self.strand = strand
            self._protein = protein

        def translate(self, *, include_stop):
            assert include_stop is False
            return self._protein

        def sequence(self):
            return "ATG" * len(self._protein)

    calls = (
        FakeGene(115, 132, 1, "PLUSAA"),
        FakeGene(109, 126, -1, "MINUSK"),
        FakeGene(109, 126, -1, "MINUSK"),
        FakeGene(55, 72, 1, "OUTSIDE"),
    )

    class FakeFinder:
        def __init__(self, **kwargs):
            observed["finder_kwargs"] = kwargs

        def find_genes(self, sequence):
            observed["search_sequence"] = sequence
            return calls

    monkeypatch.setitem(sys.modules, "pyrodigal_gv", SimpleNamespace(ViralGeneFinder=FakeFinder))
    genome = GenomeInput(sequence_id="circular", sequence="ACG" * 20)

    artifacts = prepare_orf_artifacts(
        (genome,),
        tmp_path,
        minimum_fallback_amino_acids=100,
    )

    assert observed == {
        "finder_kwargs": {"meta": True, "viral_only": False, "closed": False},
        "search_sequence": genome.sequence * 3,
    }
    assert [(record.start, record.end, record.strand, record.protein) for record in artifacts.query_records] == [
        (49, 66, "-", "MINUSK"),
        (55, 72, "+", "PLUSAA"),
    ]


def test_six_frame_fallback_coordinates_are_exact_on_both_strands(tmp_path):
    """Fallback query coordinates must map translated slices back to the original genome."""

    class NoGenePredictor:
        def predict(self, sequence, *, circular):
            return ()

    artifacts = prepare_orf_artifacts(
        (GenomeInput(sequence_id="short", sequence="ATGAAATAA", circular=False),),
        tmp_path,
        predictor=NoGenePredictor(),
        minimum_fallback_amino_acids=2,
    )
    records = {record.query_id: record for record in artifacts.query_records}

    plus = records["short__sixframe_p1_0001"]
    assert (plus.start, plus.end, plus.strand, plus.frame, plus.nucleotide, plus.protein) == (
        1,
        6,
        "+",
        1,
        "ATGAAA",
        "MK",
    )
    minus = records["short__sixframe_m2_0001"]
    assert (minus.start, minus.end, minus.strand, minus.frame, minus.nucleotide, minus.protein) == (
        3,
        8,
        "-",
        -2,
        "TATTTC",
        "YF",
    )


def test_tool_pin_validates_exact_executable_digest_and_version(tmp_path):
    """A changed executable or reported version must not be treated as the pinned detector."""
    executable = tmp_path / "diamond"
    executable.write_bytes(b"pinned-diamond-binary")
    pin = ToolPin(
        path=executable,
        sha256=hashlib.sha256(b"pinned-diamond-binary").hexdigest(),
        version="diamond version 2.1.24",
        version_args=("version",),
    )

    def runner(command, **kwargs):
        assert command == [str(executable), "version"]
        assert kwargs == {"check": True, "capture_output": True, "text": True, "timeout": 11.0}
        return subprocess.CompletedProcess(command, 0, stdout="diamond version 2.1.24\n", stderr="")

    assert validate_tool_pin(pin, runner=runner, timeout=11.0) == "diamond version 2.1.24"


def _orf_artifacts(tmp_path):
    return prepare_orf_artifacts(
        (GenomeInput(sequence_id="genome_1", sequence="ACG" * 70),),
        tmp_path / "orf",
        predictor=_DeterministicGenePredictor(),
        minimum_fallback_amino_acids=3,
    )


def _amrfinder_manifest(tmp_path):
    runtime_root = tmp_path / "amrfinder-runtime"
    runtime_root.mkdir()
    runtime_paths = {}
    for name in external_assets.AMRFINDER_RUNTIME_FILES:
        path = runtime_root / name
        path.write_bytes(b"amrfinder-4.2.7" if name == "amrfinder" else f"runtime-{name}".encode())
        if name in external_assets.AMRFINDER_RUNTIME_EXECUTABLES:
            path.chmod(0o755)
        runtime_paths[name] = path
    binary = runtime_paths["amrfinder"]
    database = tmp_path / "amr-db" / "2026-07-22.1"
    database.mkdir(parents=True)
    (database / "catalog.txt").write_bytes(b"amr-db")
    manifest = {
        "release": "amrfinder_v4.2.7",
        "binary_path": str(binary),
        "binary_sha256": hashlib.sha256(b"amrfinder-4.2.7").hexdigest(),
        "runtime_bundle": external_assets._amrfinder_runtime_bundle_record(runtime_paths),
        "amrfinder_version": "AMRFinderPlus version 4.2.7",
        "database_path": str(database),
        "database_version": "2026-07-22.1",
        "database_sha256": hashlib.sha256(b"catalog.txt\0amr-db").hexdigest(),
    }
    for tool_name in ("blastn", "blastp", "blastx", "makeblastdb", "tblastn", "hmmpress", "hmmsearch"):
        tool_path = tmp_path / ("blast" if tool_name.startswith(("blast", "make", "tblast")) else "hmmer") / tool_name
        tool_path.parent.mkdir(exist_ok=True)
        payload = f"runtime-{tool_name}".encode()
        tool_path.write_bytes(payload)
        tool_path.chmod(0o755)
        manifest[f"{tool_name}_path"] = str(tool_path)
        manifest[f"{tool_name}_sha256"] = hashlib.sha256(payload).hexdigest()
    return manifest


def _amrfinder_row(**overrides):
    values = {
        "Protein id": "genome_1__orf0001",
        "Contig id": "genome_1",
        "Start": "4",
        "Stop": "93",
        "Strand": "+",
        "Element symbol": "blaSYN",
        "Element name": "synthetic AMR control",
        "Scope": "core",
        "Type": "AMR",
        "Subtype": "AMR",
        "Class": "BETA-LACTAM",
        "Subclass": "CLASS A",
        "Method": "ALLELE",
        "Target length": "30",
        "Reference sequence length": "30",
        "% Coverage of reference": "100.0",
        "% Identity to reference": "99.0",
        "Alignment length": "30",
        "Closest reference accession": "SYN1",
        "Closest reference name": "nonfunctional synthetic control",
        "HMM accession": "NA",
        "HMM description": "NA",
        "Hierarchy node": "123",
    }
    values.update(overrides)
    return "\t".join(values[column] for column in _AMRFINDER_HEADER)


def _write_amrfinder_output(path, *rows, header=_AMRFINDER_HEADER, comments=()):
    path.write_text(
        "".join(f"# {comment}\n" for comment in comments)
        + "\t".join(header)
        + "\n"
        + "".join(f"{row}\n" for row in rows)
    )


def test_amrfinder_header_only_success_is_measured_pass(tmp_path):
    """A successful no-hit scan must be distinguishable from absent or zero-byte output."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "amrfinder.tsv"
    _write_amrfinder_output(output, comments=("AMRFinderPlus measured output",))

    result = _parse_amrfinder_output_validated(
        output,
        artifacts=artifacts,
        manifest_section=_amrfinder_manifest(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.PASS
    assert result.class_result.reason_codes == ("AMRFINDER_MEASURED_NO_AMR_HIT",)
    assert result.supplemental_findings == ()
    assert result.policy_id == "amrfinder-curated-thresholds-v4.2.7-r2"
    assert len(result.policy_sha256) == 64


def test_amrfinder_amr_type_fails_regardless_of_plus_scope_with_normalized_evidence(tmp_path):
    """Treating plus-scope AMR as non-AMR would allow a resistance determinant through the gate."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "amrfinder.tsv"
    _write_amrfinder_output(output, _amrfinder_row(**{"Scope": "plus"}))
    manifest = _amrfinder_manifest(tmp_path)

    result = parse_amrfinder_output(
        output,
        artifacts=artifacts,
        manifest_section=manifest,
        required=True,
    )

    assert result.class_result.state is SafetyState.FAIL
    assert len(result.class_result.findings) == 1
    finding = result.class_result.findings[0]
    assert isinstance(finding, NormalizedSafetyFinding)
    assert finding.to_dict() == {
        "safety_class": "amr",
        "state": "FAIL",
        "reason_codes": ["AMR_DETERMINANT_DETECTED"],
        "finding_id": "amr:genome_1__orf0001:SYN1",
        "detector": "amrfinder-plus",
        "accession": "SYN1",
        "query_id": "genome_1__orf0001",
        "sequence_id": "genome_1",
        "start": 4,
        "end": 93,
        "strand": "+",
        "frame": 1,
        "scores": {
            "alignment_length": 30.0,
            "identity": 99.0,
            "reference_coverage": 100.0,
            "reference_length": 30.0,
            "target_length": 30.0,
        },
        "thresholds": {},
        "source_path": manifest["database_path"],
        "source_sha256": manifest["database_sha256"],
        "tool_version": "AMRFinderPlus version 4.2.7",
        "database_version": "2026-07-22.1",
        "evidence_path": "pyrodigal-gv",
        "evidence_method": "ALLELE",
        "threshold_policy": "amrfinder-curated-thresholds-v4.2.7-r2",
        "threshold_policy_sha256": result.policy_sha256,
        "tool_path": manifest["binary_path"],
        "tool_sha256": manifest["binary_sha256"],
        "profile": None,
    }


def _amrfinder_nucleotide_query_id(*, sequence_id: str, sequence: str, start: int, stop: int, strand: str) -> str:
    strand_label = "p" if strand == "+" else "m"
    sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
    return f"{sequence_id}__amrfinder_nt_{strand_label}_{start}_{stop}_{sequence_sha256}"


def test_amrfinder_mixed_protein_and_nucleotide_rows_retain_authenticated_amr_evidence(tmp_path):
    """A valid nucleotide-only hit must not erase independent protein AMR findings."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "amrfinder.tsv"
    nucleotide_row = _amrfinder_row(
        **{
            "Protein id": "NA",
            "Start": "100",
            "Stop": "120",
            "Method": "INTERNAL_STOP",
            "Closest reference accession": "SYN_NT.1",
        }
    )
    _write_amrfinder_output(output, _amrfinder_row(), nucleotide_row)

    result = _parse_amrfinder_output_validated(
        output,
        artifacts=artifacts,
        manifest_section=_amrfinder_manifest(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.FAIL
    assert [finding.accession for finding in result.class_result.findings] == ["SYN1", "SYN_NT.1"]
    nucleotide = result.class_result.findings[1]
    sequence = "ACG" * 70
    assert nucleotide.query_id == _amrfinder_nucleotide_query_id(
        sequence_id="genome_1", sequence=sequence, start=100, stop=120, strand="+"
    )
    finding_prefix = f"amr:{nucleotide.query_id}:SYN_NT.1:"
    assert nucleotide.finding_id.startswith(finding_prefix)
    assert len(nucleotide.finding_id.removeprefix(finding_prefix)) == 16
    assert nucleotide.sequence_id == "genome_1"
    assert (nucleotide.start, nucleotide.end, nucleotide.strand, nucleotide.frame) == (100, 120, "+", 1)
    assert nucleotide.evidence_path == "amrfinder-nucleotide-v1"
    assert nucleotide.evidence_method == "INTERNAL_STOP"


@pytest.mark.parametrize(
    ("overrides", "reason_code"),
    [
        ({"Contig id": "GENOME_1"}, "AMRFINDER_UNKNOWN_SEQUENCE_ID"),
        ({"Start": "0"}, "AMRFINDER_NUCLEOTIDE_COORDINATE_MISMATCH"),
        ({"Start": "121", "Stop": "120"}, "AMRFINDER_NUCLEOTIDE_COORDINATE_MISMATCH"),
        ({"Stop": "211"}, "AMRFINDER_NUCLEOTIDE_COORDINATE_MISMATCH"),
        ({"Start": "one"}, "AMRFINDER_INVALID_NUMERIC_VALUE"),
        ({"Strand": "."}, "AMRFINDER_NUCLEOTIDE_COORDINATE_MISMATCH"),
        ({"Method": "ALLELEP"}, "AMRFINDER_NUCLEOTIDE_METHOD_MISMATCH"),
        ({"Method": ""}, "AMRFINDER_NUCLEOTIDE_METHOD_MISMATCH"),
    ],
)
def test_amrfinder_nucleotide_rows_fail_closed_on_unbound_evidence(tmp_path, overrides, reason_code):
    """Nucleotide evidence must bind an exact contig, interval, strand, and v4.2.7 method."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "amrfinder.tsv"
    row = {
        "Protein id": "NA",
        "Start": "100",
        "Stop": "120",
        "Method": "INTERNAL_STOP",
        **overrides,
    }
    _write_amrfinder_output(output, _amrfinder_row(**row))

    result = _parse_amrfinder_output_validated(
        output,
        artifacts=artifacts,
        manifest_section=_amrfinder_manifest(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


@pytest.mark.parametrize("protein_id", ["", "N/A", "na", " NA"])
def test_amrfinder_nucleotide_query_aliases_are_not_accepted(tmp_path, protein_id):
    """Only AMRFinderPlus's literal NA sentinel may select the nucleotide evidence path."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "amrfinder.tsv"
    _write_amrfinder_output(
        output,
        _amrfinder_row(
            **{
                "Protein id": protein_id,
                "Start": "100",
                "Stop": "120",
                "Method": "INTERNAL_STOP",
            }
        ),
    )

    result = _parse_amrfinder_output_validated(
        output,
        artifacts=artifacts,
        manifest_section=_amrfinder_manifest(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("AMRFINDER_UNKNOWN_QUERY_ID",)


@pytest.mark.parametrize(
    "tampered_fasta",
    [
        ">genome_1 alias\n" + "ACG" * 70 + "\n",
        ">genome_1 circular=true\n" + "ACG" * 70 + "\n>genome_1 circular=false\nACG\n",
        ">genome_1 circular=true\n" + "ACG" * 69 + "XCG\n",
    ],
)
def test_amrfinder_nucleotide_rows_reject_aliased_duplicate_or_invalid_contig_fasta(tmp_path, tampered_fasta):
    """The nucleotide row must be authenticated against the canonical generated genome FASTA."""
    artifacts = _orf_artifacts(tmp_path)
    artifacts.genomes_fna.write_text(tampered_fasta)
    output = tmp_path / "amrfinder.tsv"
    _write_amrfinder_output(
        output,
        _amrfinder_row(**{"Protein id": "NA", "Start": "100", "Stop": "120", "Method": "INTERNAL_STOP"}),
    )

    result = _parse_amrfinder_output_validated(
        output,
        artifacts=artifacts,
        manifest_section=_amrfinder_manifest(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("AMRFINDER_NUCLEOTIDE_FASTA_MISMATCH",)


def test_amrfinder_nucleotide_query_id_collision_is_indeterminate(tmp_path):
    """A synthetic nucleotide evidence identity may never alias an existing ORF query."""
    artifacts = _orf_artifacts(tmp_path)
    sequence = "ACG" * 70
    query_id = _amrfinder_nucleotide_query_id(
        sequence_id="genome_1", sequence=sequence, start=100, stop=120, strand="+"
    )
    collision = ORFQueryRecord(
        query_id=query_id,
        sequence_id="genome_1",
        start=100,
        end=120,
        strand="+",
        frame=1,
        nucleotide=sequence[99:120],
        protein="TYVRVRT",
        evidence_path="pyrodigal-gv",
    )
    artifacts = replace(artifacts, query_records=(*artifacts.query_records, collision))
    output = tmp_path / "amrfinder.tsv"
    _write_amrfinder_output(
        output,
        _amrfinder_row(**{"Protein id": "NA", "Start": "100", "Stop": "120", "Method": "INTERNAL_STOP"}),
    )

    result = _parse_amrfinder_output_validated(
        output,
        artifacts=artifacts,
        manifest_section=_amrfinder_manifest(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("AMRFINDER_NUCLEOTIDE_QUERY_ID_COLLISION",)


def test_amrfinder_distinct_same_locus_nucleotide_rows_have_unique_findings(tmp_path):
    """Pinned v4.2.7 emits distinct POINTX mutations at one locus and accession."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "amrfinder.tsv"
    base = {
        "Protein id": "NA",
        "Start": "1",
        "Stop": "210",
        "Method": "POINTX",
        "Closest reference accession": "WP_089631889.1",
    }
    _write_amrfinder_output(
        output,
        _amrfinder_row(**{**base, "Element symbol": "nfsA_K141Ter"}),
        _amrfinder_row(**{**base, "Element symbol": "nfsA_R15C"}),
    )

    result = _parse_amrfinder_output_validated(
        output,
        artifacts=artifacts,
        manifest_section=_amrfinder_manifest(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.FAIL
    assert len(result.class_result.findings) == 2
    assert len({finding.finding_id for finding in result.class_result.findings}) == 2
    assert len({finding.query_id for finding in result.class_result.findings}) == 1


def test_amrfinder_pointn_nucleotide_row_is_supported(tmp_path):
    """The pinned v4.2.7 POINTN method is valid nucleotide-only AMR evidence."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "amrfinder.tsv"
    _write_amrfinder_output(
        output,
        _amrfinder_row(
            **{
                "Protein id": "NA",
                "Start": "1",
                "Stop": "210",
                "Method": "POINTN",
            }
        ),
    )

    result = _parse_amrfinder_output_validated(
        output, artifacts=artifacts, manifest_section=_amrfinder_manifest(tmp_path), required=True
    )

    assert result.class_result.state is SafetyState.FAIL
    assert result.class_result.findings[0].evidence_method == "POINTN"


def test_amrfinder_complete_row_allows_schema_defined_na_scores(tmp_path):
    """Pinned v4.2.7 COMPLETE operon rows legitimately omit reference length and coverage."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "amrfinder.tsv"
    _write_amrfinder_output(
        output,
        _amrfinder_row(
            **{
                "Protein id": "NA",
                "Start": "100",
                "Stop": "210",
                "Scope": "plus",
                "Type": "VIRULENCE",
                "Method": "COMPLETE",
                "Reference sequence length": "NA",
                "% Coverage of reference": "NA",
            }
        ),
    )

    result = _parse_amrfinder_output_validated(
        output, artifacts=artifacts, manifest_section=_amrfinder_manifest(tmp_path), required=True
    )

    assert result.class_result.state is SafetyState.PASS
    assert len(result.supplemental_findings) == 1
    finding = result.supplemental_findings[0]
    assert finding.evidence_method == "COMPLETE"
    assert "reference_length" not in finding.scores
    assert "reference_coverage" not in finding.scores


def test_amrfinder_plus_virulence_is_supplemental_toxin_evidence_not_complete_screen(tmp_path):
    """Virulence evidence must remain visible without turning AMRFinder into the sole toxin detector."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "amrfinder.tsv"
    _write_amrfinder_output(
        output,
        _amrfinder_row(
            **{
                "Scope": "plus",
                "Type": "VIRULENCE",
                "Element symbol": "virSYN",
                "Closest reference accession": "VIR_SYN.1",
            }
        ),
    )

    result = _parse_amrfinder_output_validated(
        output,
        artifacts=artifacts,
        manifest_section=_amrfinder_manifest(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.PASS
    assert result.class_result.findings == ()
    assert len(result.supplemental_findings) == 1
    supplemental = result.supplemental_findings[0]
    assert supplemental.safety_class == "toxin"
    assert supplemental.state is SafetyState.INDETERMINATE
    assert supplemental.reason_codes == ("AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL",)


@pytest.mark.parametrize(
    ("write_output", "reason_code"),
    [
        (
            lambda path: _write_amrfinder_output(path, header=(*_AMRFINDER_HEADER, "Unexpected")),
            "AMRFINDER_PARSER_SCHEMA_MISMATCH",
        ),
        (
            lambda path: path.write_text("\t".join(_AMRFINDER_HEADER) + "\n" + "\t".join(_AMRFINDER_HEADER) + "\n"),
            "AMRFINDER_DUPLICATE_HEADER",
        ),
        (
            lambda path: _write_amrfinder_output(path, _amrfinder_row(), _amrfinder_row()),
            "AMRFINDER_DUPLICATE_HIT",
        ),
        (
            lambda path: _write_amrfinder_output(path, _amrfinder_row(**{"Protein id": "unknown_orf"})),
            "AMRFINDER_UNKNOWN_QUERY_ID",
        ),
        (
            lambda path: _write_amrfinder_output(path, _amrfinder_row(**{"% Identity to reference": "nan"})),
            "AMRFINDER_INVALID_NUMERIC_VALUE",
        ),
    ],
)
def test_amrfinder_parser_drift_and_ambiguous_rows_are_indeterminate(tmp_path, write_output, reason_code):
    """Malformed or ambiguous scanner rows must never degrade into a measured PASS."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "amrfinder.tsv"
    write_output(output)

    result = parse_amrfinder_output(
        output,
        artifacts=artifacts,
        manifest_section=_amrfinder_manifest(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


def test_amrfinder_non_utf_output_is_indeterminate_instead_of_raising(tmp_path):
    """Undecodable scanner bytes are malformed evidence, not a process-level exception."""
    output = tmp_path / "amrfinder.tsv"
    output.write_bytes(b"\xff\xfe")

    result = parse_amrfinder_output(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_amrfinder_manifest(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("AMRFINDER_PARSER_SCHEMA_MISMATCH",)


def test_amrfinder_runner_validates_versions_and_database_before_search(tmp_path):
    """Executing the scan before pin checks would let drifted assets produce trusted results."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _amrfinder_manifest(tmp_path)
    commands = []

    def runner(command, **kwargs):
        commands.append((command, kwargs))
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="AMRFinderPlus version 4.2.7\n", stderr="")
        if command[-1] == "--database_version":
            return subprocess.CompletedProcess(command, 0, stdout="2026-07-22.1\n", stderr="")
        _write_amrfinder_output(Path(command[command.index("-o") + 1]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_amrfinder(
        artifacts,
        manifest_section=manifest,
        work_dir=tmp_path / "scan",
        threads=3,
        runner=runner,
        timeout=17.0,
    )

    assert result.class_result.state is SafetyState.PASS
    assert [command for command, _ in commands[:2]] == [
        [manifest["binary_path"], "--version"],
        [manifest["binary_path"], "--database", manifest["database_path"], "--database_version"],
    ]
    assert commands[2][0] == build_amrfinder_command(
        amrfinder=Path(manifest["binary_path"]),
        genomes_fna=artifacts.genomes_fna,
        proteins_faa=artifacts.proteins_faa,
        proteins_gff=artifacts.proteins_gff,
        database_dir=Path(manifest["database_path"]),
        blast_bin_dir=Path(manifest["blastx_path"]).parent,
        hmmer_bin_dir=Path(manifest["hmmsearch_path"]).parent,
        threads=3,
        output_tsv=tmp_path / "scan" / "amrfinder.tsv",
    )
    assert commands[2][1] == {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 17.0,
    }


def test_amrfinder_runner_accepts_v427_verbose_database_version_output(tmp_path):
    """The v4.2.7 database-version command emits a labeled report, not a bare version string."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _amrfinder_manifest(tmp_path)

    def runner(command, **_kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout=manifest["amrfinder_version"] + "\n", stderr="")
        if command[-1] == "--database_version":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "Software directory: '/opt/amrfinder/bin/'\n"
                    "Software version: 4.2.7\n"
                    f"Database directory: '{manifest['database_path']}'\n"
                    f"Database version: {manifest['database_version']}\n"
                ),
                stderr="",
            )
        _write_amrfinder_output(Path(command[command.index("-o") + 1]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_amrfinder(
        artifacts,
        manifest_section=manifest,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.PASS
    assert result.class_result.reason_codes == ("AMRFINDER_MEASURED_NO_AMR_HIT",)


def test_amrfinder_database_digest_matches_task2_for_nested_directories(tmp_path):
    """Task 3 must consume the exact stable tree digest that Task 2 records."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _amrfinder_manifest(tmp_path)
    database = Path(manifest["database_path"])
    nested_file = database / "nested" / "index.bin"
    nested_file.parent.mkdir()
    nested_file.write_bytes(b"nested-index")
    digest = hashlib.sha256()
    for relative_path, payload in (
        ("catalog.txt", b"amr-db"),
        ("nested/index.bin", b"nested-index"),
    ):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(payload)
    manifest["database_sha256"] = digest.hexdigest()

    def runner(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout=manifest["amrfinder_version"] + "\n", stderr="")
        if command[-1] == "--database_version":
            return subprocess.CompletedProcess(command, 0, stdout=manifest["database_version"] + "\n", stderr="")
        _write_amrfinder_output(Path(command[command.index("-o") + 1]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_amrfinder(
        artifacts,
        manifest_section=manifest,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.PASS


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (
            lambda manifest: Path(manifest["binary_path"]).write_bytes(b"changed"),
            "AMRFINDER_ASSET_PROVENANCE_MISMATCH",
        ),
        (
            lambda manifest: (Path(manifest["database_path"]) / "catalog.txt").write_bytes(b"changed"),
            "AMRFINDER_ASSET_PROVENANCE_MISMATCH",
        ),
        (
            lambda manifest: Path(manifest["blastx_path"]).write_bytes(b"changed"),
            "AMRFINDER_ASSET_PROVENANCE_MISMATCH",
        ),
        (
            lambda manifest: Path(manifest["hmmsearch_path"]).unlink(),
            "AMRFINDER_ASSET_PROVENANCE_MISMATCH",
        ),
    ],
)
def test_amrfinder_digest_drift_is_indeterminate_without_execution(tmp_path, mutate, reason_code):
    """Digest drift must stop execution and make the required class unmeasured."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _amrfinder_manifest(tmp_path)
    mutate(manifest)

    def runner(*args, **kwargs):
        raise AssertionError("drifted assets must not execute")

    result = run_amrfinder(
        artifacts,
        manifest_section=manifest,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


def test_amrfinder_missing_runtime_companion_is_provenance_failure_before_execution(tmp_path):
    """A missing sibling executable must fail before the scanner command can run."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _amrfinder_manifest(tmp_path)
    (Path(manifest["runtime_bundle"]["path"]) / "fasta_check").unlink()

    def runner(*_args, **_kwargs):
        raise AssertionError("an incomplete AMRFinder runtime must not execute")

    result = run_amrfinder(
        artifacts,
        manifest_section=manifest,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("AMRFINDER_ASSET_PROVENANCE_MISMATCH",)


def test_amrfinder_version_drift_is_indeterminate_before_search(tmp_path):
    """A binary reporting a different version must not execute against the pinned database."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _amrfinder_manifest(tmp_path)
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="AMRFinderPlus version 4.3.0\n", stderr="")

    result = run_amrfinder(
        artifacts,
        manifest_section=manifest,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("AMRFINDER_ASSET_PROVENANCE_MISMATCH",)
    assert commands == [[manifest["binary_path"], "--version"]]


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("nonzero", "AMRFINDER_EXECUTION_FAILED"),
        ("timeout", "AMRFINDER_EXECUTION_TIMEOUT"),
        ("missing", "AMRFINDER_OUTPUT_MISSING"),
    ],
)
def test_amrfinder_execution_failure_timeout_or_missing_output_is_indeterminate(tmp_path, failure, reason_code):
    """A scanner that did not produce measured output cannot report a safety PASS."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _amrfinder_manifest(tmp_path)

    def runner(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout=manifest["amrfinder_version"] + "\n", stderr="")
        if command[-1] == "--database_version":
            return subprocess.CompletedProcess(command, 0, stdout=manifest["database_version"] + "\n", stderr="")
        if failure == "nonzero":
            raise subprocess.CalledProcessError(2, command, stderr="failed")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_amrfinder(
        artifacts,
        manifest_section=manifest,
        work_dir=tmp_path / "scan",
        runner=runner,
        timeout=2.5,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


_DIAMOND_HEADER = (
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "qlen",
    "slen",
    "qcovhsp",
    "scovhsp",
    "evalue",
    "bitscore",
)


def _toxin_manifest(tmp_path):
    annotations = tmp_path / "reviewed_toxins.tsv"
    annotations.write_text(
        "Entry\tEntry Name\tProtein names\tGene Names\tOrganism\tOrganism (ID)\tKeyword ID\t"
        "Taxonomic lineage (IDs)\tFunction [CC]\n"
        "P0C1\tSYN_TOX\tSynthetic toxin control\tsyn\tSynthetic virus\t10239\tKW-0800; KW-0843\t"
        "1; 10239\tSynthetic annotation\n"
    )
    fasta = tmp_path / "reviewed_toxins.faa"
    fasta.write_text(">sp|P0C1|SYN_TOX Synthetic toxin control\nMPEPTIDE\n")
    database = tmp_path / "reviewed_toxins.dmnd"
    database.write_bytes(b"reviewed-toxin-diamond-db")
    return {
        "uniprot_release": "2026_03",
        "classification_policy": dict(sequence_safety_adapters.TOXIN_REFERENCE_CLASSIFICATION_POLICY),
        "files": {
            "annotations": {
                "path": str(annotations),
                "sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
            },
            "fasta": {"path": str(fasta), "sha256": hashlib.sha256(fasta.read_bytes()).hexdigest()},
            "diamond_database": {
                "path": str(database),
                "sha256": hashlib.sha256(b"reviewed-toxin-diamond-db").hexdigest(),
            },
        },
    }


def _diamond_pin(tmp_path):
    binary = tmp_path / "diamond"
    binary.write_bytes(b"diamond-2.1.24")
    return ToolPin(
        path=binary,
        sha256=hashlib.sha256(b"diamond-2.1.24").hexdigest(),
        version="diamond version 2.1.24",
        version_args=("version",),
    )


def _diamond_row(query_id="genome_1__orf0001", target_id="sp|P0C1|SYN_TOX", **overrides):
    values = {
        "qseqid": query_id,
        "sseqid": target_id,
        "pident": "85.0",
        "length": "28",
        "qlen": "30",
        "slen": "30",
        "qcovhsp": "93.333",
        "scovhsp": "93.333",
        "evalue": "1e-30",
        "bitscore": "150.0",
    }
    values.update(overrides)
    return "\t".join(values[column] for column in _DIAMOND_HEADER)


def _write_diamond_output(path, *rows, header=_DIAMOND_HEADER, comments=()):
    path.write_text(
        "".join(f"# {comment}\n" for comment in comments)
        + "\t".join(header)
        + "\n"
        + "".join(f"{row}\n" for row in rows)
    )


def test_diamond_command_scans_primary_and_six_frame_queries_without_prefilter_thresholds(tmp_path):
    """Scanning only predicted ORFs or prefiltering motifs could miss split and overlapping toxin evidence."""
    command = build_diamond_command(
        diamond=Path("/tools/diamond"),
        queries_faa=tmp_path / "all_queries.faa",
        database=Path("/db/reviewed_toxins.dmnd"),
        output_tsv=tmp_path / "toxin.raw.tsv",
        threads=5,
    )

    assert command == [
        "/tools/diamond",
        "blastp",
        "--query",
        str(tmp_path / "all_queries.faa"),
        "--db",
        "/db/reviewed_toxins.dmnd",
        "--out",
        str(tmp_path / "toxin.raw.tsv"),
        "--outfmt",
        "6",
        *_DIAMOND_HEADER,
        "--threads",
        "5",
        "--max-target-seqs",
        "0",
        "--sensitive",
    ]


def test_toxin_header_only_success_is_measured_pass(tmp_path):
    """A completed no-hit toxin search must pass while missing or malformed output remains unmeasured."""
    output = tmp_path / "toxin.tsv"
    _write_diamond_output(output, comments=("normalized measured DIAMOND output",))

    result = _parse_toxin_diamond_output_validated(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_toxin_manifest(tmp_path),
        tool_pin=_diamond_pin(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.PASS
    assert result.class_result.reason_codes == ("TOXIN_DIAMOND_MEASURED_NO_REVIEW_HIT",)
    assert result.policy_id == TOXIN_HOMOLOGY_POLICY_V2.policy_id
    assert result.policy_sha256 == TOXIN_HOMOLOGY_POLICY_V2.sha256


def test_toxin_high_confidence_joint_thresholds_fail_with_complete_provenance(tmp_path):
    """A whole-protein toxin hit meeting every high band threshold must fail the safety class."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _toxin_manifest(tmp_path)
    tool_pin = _diamond_pin(tmp_path)
    output = tmp_path / "toxin.tsv"
    _write_diamond_output(output, _diamond_row())

    result = parse_toxin_diamond_output(
        output,
        artifacts=artifacts,
        manifest_section=manifest,
        tool_pin=tool_pin,
        required=True,
    )

    assert result.class_result.state is SafetyState.FAIL
    finding = result.class_result.findings[0]
    assert finding.reason_codes == ("TOXIN_HIGH_CONFIDENCE_HOMOLOGY",)
    assert finding.accession == "P0C1"
    assert finding.detector == "diamond-reviewed-toxin"
    assert finding.evidence_path == "pyrodigal-gv"
    assert finding.threshold_policy_sha256 == TOXIN_HOMOLOGY_POLICY_V2.sha256
    assert finding.threshold_policy == "toxin-homology-v2"
    assert finding.thresholds == {
        "evalue": 1e-10,
        "identity": 80.0,
        "query_coverage": 80.0,
        "reference_coverage": 80.0,
    }
    assert finding.source_path == manifest["files"]["diamond_database"]["path"]
    assert finding.source_sha256 == manifest["files"]["diamond_database"]["sha256"]
    assert finding.tool_path == str(tool_pin.path)
    assert finding.tool_sha256 == tool_pin.sha256
    assert finding.database_version == "UniProt 2026_03"


def test_curated_phage_wo_latrotoxin_domain_homolog_requires_review_without_a_functional_claim(tmp_path):
    """Exact domain homology must block PASS without asserting functional venom or a hard hazard."""
    manifest = _toxin_manifest(tmp_path)
    curated = tmp_path / "curated_hazards.faa"
    curated.write_text(">domain|PF15658.11|Latrotoxin_C\nMPEPTIDE\n")
    search_fasta = tmp_path / "toxin_hazards.faa"
    search_fasta.write_bytes(manifest["files"]["fasta"]["path"].encode())
    manifest.update(
        reference_version="UniProt 2026_03 + phage-domain-hazards-v1",
        curated_hazards={
            "set_id": "phage-domain-hazards-v1",
            "entries": [
                {
                    "accession": "PF15658.11",
                    "name": "Latrotoxin_C",
                    "action": "REVIEW",
                    "reason_code": "TOXIN_LATROTOXIN_C_DOMAIN_HOMOLOGY_REVIEW",
                    "source_protein_accession": "CAQ54400.1",
                    "source_urls": ["https://www.ncbi.nlm.nih.gov/protein/CAQ54400.1"],
                    "source_protein_sequence_sha256": (
                        "8e8eb5098bd972dadd0c94ccbd0718c3ede5e528ac2517c605ece16e9eb08a73"
                    ),
                    "source_interval": {"start": 2571, "end": 2706},
                    "sequence_sha256": "9da486e50032ff2f89b493049419d7fb9f754f8cc935abb9339f56631dd6a8be",
                    "evidence_urls": ["https://doi.org/10.1038/ncomms13155"],
                    "interpretation": "Latrotoxin C-terminal-domain homology; not evidence of functional venom.",
                }
            ],
        },
    )
    manifest["files"].update(
        curated_hazard_fasta={
            "path": str(curated),
            "sha256": hashlib.sha256(curated.read_bytes()).hexdigest(),
        },
        search_fasta={
            "path": str(search_fasta),
            "sha256": hashlib.sha256(search_fasta.read_bytes()).hexdigest(),
        },
    )
    output = tmp_path / "toxin.tsv"
    _write_diamond_output(
        output,
        _diamond_row(
            target_id="domain|PF15658.11|Latrotoxin_C",
            pident="100",
            length="136",
            qlen="2748",
            slen="136",
            qcovhsp="4.9",
            scovhsp="100",
            evalue="3.20e-85",
            bitscore="271",
        ),
    )

    result = _parse_toxin_diamond_output_validated(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=manifest,
        tool_pin=_diamond_pin(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("TOXIN_LATROTOXIN_C_DOMAIN_HOMOLOGY_REVIEW",)
    finding = result.class_result.findings[0]
    assert finding.accession == "PF15658.11"
    assert finding.detector == "diamond-curated-toxin-domain"
    assert finding.profile == "PF15658.11"
    assert finding.database_version == "UniProt 2026_03 + phage-domain-hazards-v1"
    assert "VENOM" not in " ".join(finding.reason_codes)


def test_toxin_fragment_of_reviewed_human_harm_reference_is_review_not_pass(tmp_path):
    """A substantial local toxin fragment must be reviewed without being promoted to a whole-protein FAIL."""
    output = tmp_path / "toxin.tsv"
    _write_diamond_output(
        output,
        _diamond_row(
            pident="55.0",
            length="55",
            qlen="60",
            slen="400",
            qcovhsp="91.667",
            scovhsp="13.75",
            evalue="1e-8",
            bitscore="65.0",
        ),
    )

    result = _parse_toxin_diamond_output_validated(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_toxin_manifest(tmp_path),
        tool_pin=_diamond_pin(tmp_path),
        required=True,
        policy=sequence_safety_adapters.TOXIN_HOMOLOGY_POLICY_V2,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("TOXIN_FRAGMENT_REVIEW_HOMOLOGY",)
    assert result.class_result.findings[0].thresholds == {
        "alignment_length": 50.0,
        "evalue": 1e-5,
        "identity": 50.0,
        "query_coverage": 70.0,
        "reference_coverage": 10.0,
    }


def test_toxin_selected_policy_digest_is_bound_to_result_and_finding(tmp_path):
    """A caller-selected versioned policy must not be mislabeled as the default policy."""
    policy = HomologyPolicy(
        policy_id="toxin-homology-v2",
        high=HomologyBand(identity=90.0, query_coverage=90.0, reference_coverage=90.0, evalue=1e-20),
        review=HomologyBand(identity=50.0, query_coverage=70.0, reference_coverage=70.0, evalue=1e-8),
    )
    output = tmp_path / "toxin.tsv"
    _write_diamond_output(
        output,
        _diamond_row(pident="95.0", qcovhsp="95.0", scovhsp="95.0", evalue="1e-30"),
    )

    result = parse_toxin_diamond_output(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_toxin_manifest(tmp_path),
        tool_pin=_diamond_pin(tmp_path),
        required=True,
        policy=policy,
    )

    finding = result.class_result.findings[0]
    assert (result.policy_id, result.policy_sha256) == (policy.policy_id, policy.sha256)
    assert (finding.threshold_policy, finding.threshold_policy_sha256) == (
        policy.policy_id,
        policy.sha256,
    )


def test_toxin_material_review_band_is_indeterminate_and_records_fallback_path(tmp_path):
    """Review-band fallback homology must remain visible and cannot become either PASS or definite FAIL."""
    artifacts = _orf_artifacts(tmp_path)
    fallback = next(record for record in artifacts.query_records if record.evidence_path == "six-frame-fallback")
    output = tmp_path / "toxin.tsv"
    _write_diamond_output(
        output,
        _diamond_row(
            query_id=fallback.query_id,
            pident="50.0",
            qcovhsp="70.0",
            scovhsp="65.0",
            evalue="1e-8",
        ),
    )

    result = parse_toxin_diamond_output(
        output,
        artifacts=artifacts,
        manifest_section=_toxin_manifest(tmp_path),
        tool_pin=_diamond_pin(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    finding = result.class_result.findings[0]
    assert finding.reason_codes == ("TOXIN_REVIEW_HOMOLOGY",)
    assert finding.evidence_path == "six-frame-fallback"
    assert finding.thresholds == {
        "evalue": 1e-5,
        "identity": 40.0,
        "query_coverage": 60.0,
        "reference_coverage": 60.0,
    }


def test_toxin_short_motif_does_not_pass_joint_coverage_gate_and_remains_raw(tmp_path):
    """High identity on a short motif must not be classified as a whole toxin ORF."""
    output = tmp_path / "toxin.tsv"
    row = _diamond_row(pident="99.0", qcovhsp="12.0", scovhsp="9.0", evalue="1e-40")
    _write_diamond_output(output, row)

    result = _parse_toxin_diamond_output_validated(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_toxin_manifest(tmp_path),
        tool_pin=_diamond_pin(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.PASS
    assert result.class_result.findings == ()
    assert row in Path(result.raw_output_path).read_text()
    assert TOXIN_HOMOLOGY_POLICY_V1.policy_id == "toxin-homology-v1"


@pytest.mark.parametrize(
    ("write_output", "reason_code"),
    [
        (
            lambda path, query_id: _write_diamond_output(path, header=(*_DIAMOND_HEADER, "unexpected")),
            "TOXIN_DIAMOND_PARSER_SCHEMA_MISMATCH",
        ),
        (
            lambda path, query_id: path.write_text(
                "\t".join(_DIAMOND_HEADER) + "\n" + "\t".join(_DIAMOND_HEADER) + "\n"
            ),
            "TOXIN_DIAMOND_DUPLICATE_HEADER",
        ),
        (
            lambda path, query_id: _write_diamond_output(path, _diamond_row(), _diamond_row()),
            "TOXIN_DIAMOND_DUPLICATE_HIT",
        ),
        (
            lambda path, query_id: _write_diamond_output(path, _diamond_row(), _diamond_row(bitscore="149.0")),
            "TOXIN_DIAMOND_DUPLICATE_HIT",
        ),
        (
            lambda path, query_id: _write_diamond_output(path, _diamond_row(query_id="unknown_orf")),
            "TOXIN_DIAMOND_UNKNOWN_QUERY_ID",
        ),
        (
            lambda path, query_id: _write_diamond_output(path, _diamond_row(pident="inf")),
            "TOXIN_DIAMOND_INVALID_NUMERIC_VALUE",
        ),
        (
            lambda path, query_id: _write_diamond_output(path, _diamond_row(target_id="P0C1")),
            "TOXIN_DIAMOND_MALFORMED_TARGET_ID",
        ),
        (
            lambda path, query_id: _write_diamond_output(path, _diamond_row(target_id="sp|UNKNOWN|TOX")),
            "TOXIN_DIAMOND_UNKNOWN_ACCESSION",
        ),
    ],
)
def test_toxin_parser_drift_duplicate_unknown_and_nonfinite_rows_are_indeterminate(
    tmp_path, write_output, reason_code
):
    """Ambiguous DIAMOND output must never be interpreted as an absence of toxin evidence."""
    artifacts = _orf_artifacts(tmp_path)
    output = tmp_path / "toxin.tsv"
    write_output(output, artifacts.query_records[0].query_id)

    result = parse_toxin_diamond_output(
        output,
        artifacts=artifacts,
        manifest_section=_toxin_manifest(tmp_path),
        tool_pin=_diamond_pin(tmp_path),
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


def test_toxin_runner_normalizes_successful_empty_raw_output_to_header_only_pass(tmp_path):
    """Successful headerless DIAMOND no-hit output must gain an auditable measured header."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _toxin_manifest(tmp_path)
    tool_pin = _diamond_pin(tmp_path)
    commands = []

    def runner(command, **kwargs):
        commands.append((command, kwargs))
        if command[1:] == ["version"]:
            return subprocess.CompletedProcess(command, 0, stdout=tool_pin.version + "\n", stderr="")
        Path(command[command.index("--out") + 1]).write_text("")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_toxin_diamond(
        artifacts,
        manifest_section=manifest,
        tool_pin=tool_pin,
        work_dir=tmp_path / "scan",
        threads=4,
        runner=runner,
        timeout=13.0,
    )

    assert result.class_result.state is SafetyState.PASS
    assert Path(result.raw_output_path).read_text() == "\t".join(_DIAMOND_HEADER) + "\n"
    assert commands[1][0] == build_diamond_command(
        diamond=tool_pin.path,
        queries_faa=artifacts.all_queries_faa,
        database=Path(manifest["files"]["diamond_database"]["path"]),
        output_tsv=tmp_path / "scan" / "toxin_diamond.raw.tsv",
        threads=4,
    )
    assert commands[1][1] == {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 13.0,
    }


@pytest.mark.parametrize(
    ("drift", "reason_code"),
    [
        ("tool_version", "TOXIN_ASSET_PROVENANCE_MISMATCH"),
        ("database_digest", "TOXIN_ASSET_PROVENANCE_MISMATCH"),
        ("missing_output", "TOXIN_DIAMOND_OUTPUT_MISSING"),
    ],
)
def test_toxin_runner_rejects_version_digest_or_output_mismatch(tmp_path, drift, reason_code):
    """Unpinned or unmeasured toxin searches cannot yield a positive safety signal."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _toxin_manifest(tmp_path)
    tool_pin = _diamond_pin(tmp_path)
    if drift == "database_digest":
        Path(manifest["files"]["diamond_database"]["path"]).write_bytes(b"changed")

    def runner(command, **kwargs):
        if command[1:] == ["version"]:
            version = "diamond version 2.2.0" if drift == "tool_version" else tool_pin.version
            return subprocess.CompletedProcess(command, 0, stdout=version + "\n", stderr="")
        if drift != "missing_output":
            raise AssertionError("drifted provenance must stop before search")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_toxin_diamond(
        artifacts,
        manifest_section=manifest,
        tool_pin=tool_pin,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("nonzero", "TOXIN_DIAMOND_EXECUTION_FAILED"),
        ("timeout", "TOXIN_DIAMOND_EXECUTION_TIMEOUT"),
    ],
)
def test_toxin_runner_marks_nonzero_exit_and_timeout_indeterminate(tmp_path, failure, reason_code):
    """A DIAMOND process failure cannot be normalized into a measured no-hit result."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _toxin_manifest(tmp_path)
    tool_pin = _diamond_pin(tmp_path)

    def runner(command, **kwargs):
        if command[1:] == ["version"]:
            return subprocess.CompletedProcess(command, 0, stdout=tool_pin.version + "\n", stderr="")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise subprocess.CalledProcessError(2, command, stderr="failed")

    result = run_toxin_diamond(
        artifacts,
        manifest_section=manifest,
        tool_pin=tool_pin,
        work_dir=tmp_path / "scan",
        runner=runner,
        timeout=2.5,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


def test_toxin_runner_marks_non_utf_output_indeterminate(tmp_path):
    """A successful process with undecodable output has not produced measured toxin evidence."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _toxin_manifest(tmp_path)
    tool_pin = _diamond_pin(tmp_path)

    def runner(command, **kwargs):
        if command[1:] == ["version"]:
            return subprocess.CompletedProcess(command, 0, stdout=tool_pin.version + "\n", stderr="")
        Path(command[command.index("--out") + 1]).write_bytes(b"\xff\xfe")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_toxin_diamond(
        artifacts,
        manifest_section=manifest,
        tool_pin=tool_pin,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("TOXIN_DIAMOND_PARSER_SCHEMA_MISMATCH",)


_PHROGS_HEADER = (
    "query",
    "target",
    "pident",
    "alnlen",
    "qlen",
    "tlen",
    "qcov",
    "tcov",
    "evalue",
    "bits",
)

_PHROGS_HIGH_PROFILE_IDS = ("phrog_1", *(f"phrog_{index}" for index in range(4, 60)))
_PHROGS_REVIEW_PROFILE_IDS = ("phrog_2", *(f"phrog_{index}" for index in range(60, 111)))
_PHROGS_FULL_PROFILE_IDS = tuple(f"phrog_{index}" for index in range(1, 111))
_PHROGS_ANNOTATION_SOURCE_TEXT = (
    "phrog\tcolor\tannot\tcategory\n"
    + "".join(f"{profile}\tblack\tintegrase\tintegration and excision\n" for profile in _PHROGS_HIGH_PROFILE_IDS)
    + "".join(
        f"{profile}\tblack\tputative recombinase\tintegration and excision\n" for profile in _PHROGS_REVIEW_PROFILE_IDS
    )
    + "phrog_3\tblack\tmajor capsid protein\thead and packaging\n"
)
_PHROGS_TEST_ANNOTATION_SHA256 = hashlib.sha256(_PHROGS_ANNOTATION_SOURCE_TEXT.encode()).hexdigest()
_PHROGS_TEST_ARCHIVE_PAYLOAD = b"pinned Pharokka profile archive fixture\n"
_PHROGS_TEST_ARCHIVE_SHA256 = hashlib.sha256(_PHROGS_TEST_ARCHIVE_PAYLOAD).hexdigest()


def test_phrogs_annotation_lookup_includes_unambiguous_lifecycle_regulators_outside_integration_category(tmp_path):
    """CTX-like anti-repressors are lifecycle evidence even when PHROGs classifies transcriptional function."""
    source = tmp_path / "phrog_annot_v4.tsv"
    source.write_text(
        "phrog\tcolor\tannot\tcategory\n"
        "1\tblack\tintegrase\tintegration and excision\n"
        "2\tblack\tanti-repressor\ttranscription regulation\n"
        "3\tblack\tCI-like repressor\ttranscription regulation\n"
        "4\tblack\ttranscriptional repressor\ttranscription regulation\n"
        "5\tblack\tmajor capsid protein\thead and packaging\n"
    )

    profiles = sequence_safety_adapters._read_phrogs_annotation_lookup(source)

    assert set(profiles) == {"phrog_1", "phrog_2", "phrog_3"}
    assert profiles["phrog_2"]["matched_term"] == "anti-repressor"
    assert profiles["phrog_2"]["confidence"] == "high_confidence"
    assert profiles["phrog_3"]["matched_term"] == "ci-like repressor"
    assert profiles["phrog_3"]["confidence"] == "high_confidence"


def test_phrogs_materialized_lookup_accepts_only_declared_lifecycle_regulators_outside_category(tmp_path):
    """The runtime reader admits the new exact terms without admitting arbitrary out-of-category rows."""
    lookup = tmp_path / "phrogs_lysogeny.tsv"
    header = "phrog\tannot\tcategory\tconfidence\tmatched_term\n"
    integrase = "phrog_1\tintegrase\tintegration and excision\thigh_confidence\tintegrase\n"
    anti_repressor = "phrog_2\tanti-repressor\ttranscription regulation\thigh_confidence\tanti-repressor\n"
    lookup.write_text(header + integrase + anti_repressor)

    profiles = sequence_safety_adapters._read_phrogs_lookup(lookup)

    assert set(profiles) == {"phrog_1", "phrog_2"}
    lookup.write_text(
        header + integrase + "phrog_2\tmajor capsid protein\thead and packaging\thigh_confidence\tanti-repressor\n"
    )
    with pytest.raises(sequence_safety_adapters.AssetProvenanceError, match="invalid"):
        sequence_safety_adapters._read_phrogs_lookup(lookup)


@pytest.fixture(autouse=True)
def _pin_synthetic_phrogs_sources_for_adapter_tests(monkeypatch):
    """Use deterministic local pins; production retains the official PHROGs digests."""
    monkeypatch.setattr(sequence_safety_adapters, "_PHROGS_ANNOTATION_SHA256", _PHROGS_TEST_ANNOTATION_SHA256)
    monkeypatch.setattr(sequence_safety_adapters, "_PHROGS_PROFILE_ARCHIVE_SHA256", _PHROGS_TEST_ARCHIVE_SHA256)


def _phrogs_manifest(tmp_path, *, unsearchable_profile_ids=()):
    unsearchable = frozenset(unsearchable_profile_ids)
    if not unsearchable.issubset(_PHROGS_FULL_PROFILE_IDS):
        raise ValueError("synthetic unsearchable PHROG IDs must exist in the source inventory")
    annotation_source = tmp_path / "phrog_annot_v4.tsv"
    annotation_source.write_text(_PHROGS_ANNOTATION_SOURCE_TEXT)
    archive_payload = _PHROGS_TEST_ARCHIVE_PAYLOAD
    archive_sha256 = _PHROGS_TEST_ARCHIVE_SHA256
    archive_path = tmp_path / "downloads" / "phrogs_safety_profile_archives" / f"{archive_sha256}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(archive_payload)
    lookup = tmp_path / "phrogs_integration_excision_v4.tsv"
    source_lookup_rows = [
        *(
            f"{profile}\tintegrase\tintegration and excision\thigh_confidence\tintegrase\n"
            for profile in _PHROGS_HIGH_PROFILE_IDS
        ),
        *(
            f"{profile}\tputative recombinase\tintegration and excision\treview\trecombinase\n"
            for profile in _PHROGS_REVIEW_PROFILE_IDS
        ),
    ]
    lookup_rows = [row for row in source_lookup_rows if row.split("\t", 1)[0] not in unsearchable]
    lookup.write_text("phrog\tannot\tcategory\tconfidence\tmatched_term\n" + "".join(lookup_rows))
    profile_prefix = tmp_path / "phrogs_profile_db"
    profile_ids = tuple(profile for profile in _PHROGS_FULL_PROFILE_IDS if profile not in unsearchable)
    profile_lookup = "".join(
        f"{index}\t{profile_id}\t{index}\n" for index, profile_id in enumerate(profile_ids)
    ).encode()
    profile_files = {
        profile_prefix: b"profile-db",
        Path(f"{profile_prefix}.index"): b"profile-index",
        Path(f"{profile_prefix}.dbtype"): b"profile-dbtype",
        Path(f"{profile_prefix}.lookup"): profile_lookup,
        Path(f"{profile_prefix}.source"): b"pharokka-v1.8.0\n",
        Path(f"{profile_prefix}_h"): ("\n".join(profile_ids) + "\n").encode(),
        Path(f"{profile_prefix}_h.index"): b"profile-header-index",
        Path(f"{profile_prefix}_h.dbtype"): b"profile-header-dbtype",
    }
    for path, payload in profile_files.items():
        path.write_bytes(payload)
    release_marker = tmp_path / "VERSION_1_8_0"
    release_marker.write_text("1.8.0\n")
    digest = hashlib.sha256()
    for path in sorted(profile_files):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(profile_files[path])
    tree_files = sorted((*profile_files, release_marker))
    tree_digest = hashlib.sha256()
    for path in tree_files:
        tree_digest.update(str(path.relative_to(tmp_path)).encode())
        tree_digest.update(b"\0")
        tree_digest.update(path.read_bytes())
    profile_id_digest = hashlib.sha256()
    for profile_id in sorted(profile_ids):
        profile_id_digest.update(profile_id.encode())
        profile_id_digest.update(b"\n")
    return {
        "annotation_url": "https://phrogs.lmge.uca.fr/downloads_from_website/phrog_annot_v4.tsv",
        "annotation_sha256": _PHROGS_TEST_ANNOTATION_SHA256,
        "source_path": str(annotation_source),
        "source_sha256": _PHROGS_TEST_ANNOTATION_SHA256,
        "category": "integration and excision",
        "high_confidence_terms": [
            "integrase",
            "excisionase",
            "site-specific recombinase",
            "lysogeny repressor",
        ],
        "additional_high_confidence_terms": ["anti-repressor", "ci-like repressor"],
        "selection_scope": "integration/excision category plus unambiguous lifecycle-regulator annotations",
        "review_terms": ["recombinase", "repressor", "lysogeny", "integration", "excision"],
        "lookup_path": str(lookup),
        "lookup_sha256": hashlib.sha256(lookup.read_bytes()).hexdigest(),
        "lookup_counts": {
            "total": len(lookup_rows),
            "high_confidence": sum("\thigh_confidence\t" in row for row in lookup_rows),
            "review": sum("\treview\t" in row for row in lookup_rows),
        },
        "source_lookup_counts": {"total": 109, "high_confidence": 57, "review": 52},
        "profile_unsearchable": {
            "reason": "absent_from_verified_profile_lookup",
            "count": len(unsearchable),
            "ids": sorted(unsearchable),
        },
        "profile_database": {
            "path": str(profile_prefix),
            "role": "complete PHROGs v4 MMseqs profile database for identity-bearing lysogeny search",
            "sha256": digest.hexdigest(),
            "files": [str(path.resolve()) for path in sorted(profile_files)],
            "extracted_tree": {
                "path": str(tmp_path.resolve()),
                "sha256": tree_digest.hexdigest(),
                "files": [str(path.resolve()) for path in tree_files],
            },
            "search_orientation": "phrog_profile_query_vs_orf_target",
            "search_profile_scope": "full_phrogs_v4_profile_database",
            "lookup_join_policy": "classify_only_profile_ids_present_in_pinned_lookup",
            "output_fields": list(_PHROGS_HEADER),
            "units": {"pident": "percent", "qcov": "fraction", "tcov": "fraction"},
            "query_id_pattern": r"^phrog_[1-9][0-9]*$",
            "query_ids_join_lookup": True,
            "profile_id_inventory": {
                "count": len(profile_ids),
                "sha256": profile_id_digest.hexdigest(),
            },
            "provenance": {
                "source_url": "https://zenodo.org/record/17110353/files/pharokka_v1.8.0_databases.tar.gz",
                "archive_observed_sha256": archive_sha256,
                "archive_expected_sha256": archive_sha256,
                "archive_published_sha256": None,
                "archive_published_md5": "a63c485241b900a11989bd1821bfbb09",
                "archive_published_size": 656171247,
                "retrieved_at": "2026-08-07T00:00:00+00:00",
                "release": "Pharokka database v1.8.0",
                "dataset_release": "PHROGs v4",
                "doi": "10.5281/zenodo.17110353",
                "license": "CC BY 4.0",
                "citation": "Pharokka database v1.8.0 (DOI: 10.5281/zenodo.17110353).",
                "minimum_mmseqs_version": "14",
                "built_with_mmseqs_version": "18.8cc5c",
                "verified_archive": {"path": str(archive_path.resolve()), "sha256": archive_sha256},
            },
        },
    }


def _attach_curated_phrogs_search_database(tmp_path: Path, manifest: dict) -> Path:
    """Attach a real derived-search manifest while using a tiny deterministic MMseqs fixture."""
    full = Path(manifest["profile_database"]["path"])
    mmseqs = _mmseqs_pin(tmp_path).path
    mmseqs.chmod(0o755)

    def runner(command, **_kwargs):
        destination = Path(command[-1])
        destination.write_bytes(b"curated-profile-db\n")
        Path(f"{destination}.dbtype").write_bytes(Path(f"{full}.dbtype").read_bytes())
        Path(f"{destination}.index").write_text("0\t0\t18\n")
        for suffix in ("_h", "_h.dbtype", "_h.index", ".lookup", ".source"):
            Path(f"{destination}{suffix}").symlink_to(Path(f"{full}{suffix}"))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    record = external_assets._prepare_phrogs_safety_search_database(
        profile_database=full,
        safety_lookup=Path(manifest["lookup_path"]),
        output_root=tmp_path / "safety_search_database",
        mmseqs_path=mmseqs,
        runner=runner,
    )
    manifest["search_database"] = record
    return Path(record["path"])


def _mmseqs_pin(tmp_path, *, version="18-8cc5c"):
    binary = tmp_path / "mmseqs"
    binary.write_bytes(b"mmseqs-18")
    return ToolPin(
        path=binary,
        sha256=hashlib.sha256(b"mmseqs-18").hexdigest(),
        version=version,
        version_args=("version",),
    )


def _phrogs_row(profile="phrog_1", query_id="genome_1__orf0001", **overrides):
    values = {
        "query": profile,
        "target": query_id,
        "pident": "45.0",
        "alnlen": "28",
        "qlen": "30",
        "tlen": "30",
        "qcov": "0.90",
        "tcov": "0.85",
        "evalue": "1e-20",
        "bits": "120.0",
    }
    values.update(overrides)
    return "\t".join(values[column] for column in _PHROGS_HEADER)


def _write_phrogs_output(path, *rows, header=_PHROGS_HEADER, comments=()):
    path.write_text(
        "".join(f"# {comment}\n" for comment in comments)
        + "\t".join(header)
        + "\n"
        + "".join(f"{row}\n" for row in rows)
    )


def _rewrite_phrogs_lookup(manifest, rows, *, lookup_counts):
    lookup = Path(manifest["lookup_path"])
    lookup.write_text("phrog\tannot\tcategory\tconfidence\tmatched_term\n" + "".join(rows))
    manifest["lookup_sha256"] = hashlib.sha256(lookup.read_bytes()).hexdigest()
    manifest["lookup_counts"] = lookup_counts


def _run_phrogs_with_measured_empty_output(tmp_path, manifest):
    commands = []
    tool_pin = _mmseqs_pin(tmp_path)

    def runner(command, **kwargs):
        commands.append(tuple(command))
        if command[1:] == ["version"]:
            return subprocess.CompletedProcess(command, 0, stdout=tool_pin.version + "\n", stderr="")
        if command[1] == "convertalis":
            Path(command[5]).write_text("")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_phrogs(
        _orf_artifacts(tmp_path),
        manifest_section=manifest,
        tool_pin=tool_pin,
        host_domain=HostDomain.BACTERIA,
        work_dir=tmp_path / "scan",
        runner=runner,
    )
    return result, commands


def test_validated_phrogs_batch_context_reuses_one_validation_and_rechecks_at_end(monkeypatch, tmp_path):
    """A batch may reuse one immutable PHROGs view only when it is revalidated at its boundary."""
    profile = tmp_path / "profiles"
    profile.write_bytes(b"profiles")
    calls: list[object] = []

    def validate(section):
        calls.append(section)
        return (
            profile,
            "a" * 64,
            {"phrog_1": {"confidence": "review"}},
            frozenset({"phrog_1"}),
            "PHROGs v4 / Pharokka v1.8.0",
            15,
        )

    monkeypatch.setattr(sequence_safety_adapters, "_validate_phrogs_assets", validate)
    section = {"identity": "pinned"}

    context = sequence_safety_adapters._prepare_validated_phrogs_assets(section)
    first = sequence_safety_adapters._phrogs_assets_from_validated_context(section, context)
    second = sequence_safety_adapters._phrogs_assets_from_validated_context(section, context)

    assert first == second
    assert len(calls) == 1
    sequence_safety_adapters._revalidate_phrogs_assets(section, context)
    assert len(calls) == 2

    with pytest.raises(sequence_safety_adapters.AssetProvenanceError, match="manifest changed"):
        sequence_safety_adapters._phrogs_assets_from_validated_context({"identity": "drifted"}, context)


def test_run_phrogs_reuses_validated_batch_context_without_rehashing(monkeypatch, tmp_path):
    """Execution and parsing share the exact prevalidated PHROGs view inside a batch."""
    manifest = _phrogs_manifest(tmp_path)
    context = sequence_safety_adapters._prepare_validated_phrogs_assets(manifest)
    tool_pin = _mmseqs_pin(tmp_path)

    def reject_revalidation(_section):
        raise AssertionError("per-record PHROGs asset validation must be skipped inside a batch")

    monkeypatch.setattr(sequence_safety_adapters, "_validate_phrogs_assets", reject_revalidation)

    def runner(command, **_kwargs):
        if command[1:] == ["version"]:
            return subprocess.CompletedProcess(command, 0, stdout=tool_pin.version + "\n", stderr="")
        if command[1] == "convertalis":
            Path(command[5]).write_text("")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_phrogs(
        _orf_artifacts(tmp_path),
        manifest_section=manifest,
        tool_pin=tool_pin,
        host_domain=HostDomain.BACTERIA,
        work_dir=tmp_path / "scan",
        runner=runner,
        _validated_assets=context,
    )

    assert result.class_result.state is SafetyState.PASS


def test_phrogs_missing_pinned_safety_lookup_counts_is_indeterminate_before_execution(tmp_path):
    """Removing the cardinality pin must not turn an incomplete safety gate into measured PASS."""
    manifest = _phrogs_manifest(tmp_path)
    manifest.pop("lookup_counts")

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_ASSET_PROVENANCE_MISMATCH",)
    assert commands == []


def test_phrogs_self_consistent_truncated_safety_lookup_is_indeterminate_before_execution(tmp_path):
    """A re-digested lookup missing one reviewed profile must return INDETERMINATE before MMseqs runs."""
    manifest = _phrogs_manifest(tmp_path)
    lookup_rows = Path(manifest["lookup_path"]).read_text().splitlines(keepends=True)[1:-1]
    _rewrite_phrogs_lookup(
        manifest,
        lookup_rows,
        lookup_counts={"total": 109, "high_confidence": 57, "review": 52},
    )

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_ASSET_PROVENANCE_MISMATCH",)
    assert commands == []


def test_phrogs_self_consistent_confidence_imbalance_is_indeterminate_before_execution(tmp_path):
    """Relabeling one pinned high-confidence family as review must fail before MMseqs runs."""
    manifest = _phrogs_manifest(tmp_path)
    lookup_rows = Path(manifest["lookup_path"]).read_text().splitlines(keepends=True)[1:]
    lookup_rows[0] = lookup_rows[0].replace("\thigh_confidence\t", "\treview\t")
    _rewrite_phrogs_lookup(
        manifest,
        lookup_rows,
        lookup_counts={"total": 109, "high_confidence": 57, "review": 52},
    )

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_ASSET_PROVENANCE_MISMATCH",)
    assert commands == []


def test_phrogs_count_preserving_safety_profile_substitution_is_indeterminate_before_execution(tmp_path):
    """Replacing one pinned safety family with another known profile must fail before MMseqs runs."""
    manifest = _phrogs_manifest(tmp_path)
    lookup_rows = Path(manifest["lookup_path"]).read_text().splitlines(keepends=True)[1:]
    lookup_rows[0] = lookup_rows[0].replace("phrog_1\t", "phrog_3\t", 1)
    _rewrite_phrogs_lookup(
        manifest,
        lookup_rows,
        lookup_counts={"total": 109, "high_confidence": 57, "review": 52},
    )

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_ASSET_PROVENANCE_MISMATCH",)
    assert commands == []


def test_phrogs_verified_content_addressed_archive_contract_allows_measured_search(tmp_path):
    """The finalized Task-2 archive evidence must validate before MMseqs can produce measured PASS."""
    manifest = _phrogs_manifest(tmp_path)

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.PASS
    assert len(commands) == 4
    assert [command[1] for command in commands[1:]] == ["createdb", "search", "convertalis"]
    assert commands[0][1:] == ("version",)


def test_phrogs_runtime_accepts_published_empty_release_sentinel(tmp_path):
    """Runtime validation must accept the zero-byte release sentinel shipped by Pharokka v1.8.0."""
    manifest = _phrogs_manifest(tmp_path)
    profile_record = manifest["profile_database"]
    marker = Path(profile_record["extracted_tree"]["path"]) / "VERSION_1_8_0"
    marker.write_bytes(b"")
    tree_root = marker.parent
    tree_files = [Path(path) for path in profile_record["extracted_tree"]["files"]]
    tree_digest = hashlib.sha256()
    for path in sorted(tree_files):
        tree_digest.update(str(path.relative_to(tree_root)).encode())
        tree_digest.update(b"\0")
        tree_digest.update(path.read_bytes())
    profile_record["extracted_tree"]["sha256"] = tree_digest.hexdigest()

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.PASS
    assert len(commands) == 4
    assert commands[0][1:] == ("version",)
    assert [command[1] for command in commands[1:]] == ["createdb", "search", "convertalis"]


def test_phrogs_runtime_reconciles_official_numeric_ids_with_recorded_unsearchable_profiles(tmp_path, monkeypatch):
    """Runtime must reproduce Task 2's numeric-ID normalization and explicit 109-to-105 reconciliation."""
    missing_ids = ("phrog_4", "phrog_5", "phrog_6", "phrog_7")
    manifest = _phrogs_manifest(tmp_path, unsearchable_profile_ids=missing_ids)
    source = Path(manifest["source_path"])
    lines = source.read_text().splitlines()
    numeric_rows = []
    for line in lines[1:]:
        fields = line.split("\t")
        fields[0] = fields[0].removeprefix("phrog_")
        numeric_rows.append("\t".join(fields))
    source.write_text(lines[0] + "\n" + "\n".join(numeric_rows) + "\n")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["annotation_sha256"] = source_sha256
    manifest["source_sha256"] = source_sha256
    monkeypatch.setattr(sequence_safety_adapters, "_PHROGS_ANNOTATION_SHA256", source_sha256)

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.PASS
    assert len(commands) == 4
    assert [command[1] for command in commands[1:]] == ["createdb", "search", "convertalis"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest["profile_database"]["provenance"].pop("verified_archive"),
        lambda manifest: manifest["profile_database"]["provenance"]["verified_archive"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda manifest: manifest["profile_database"]["provenance"]["verified_archive"].__setitem__(
            "path", "relative/phrogs_safety_profile_archives/archive.tar.gz"
        ),
        lambda manifest: Path(manifest["profile_database"]["provenance"]["verified_archive"]["path"]).write_bytes(
            b"tampered archive bytes\n"
        ),
    ],
)
def test_phrogs_verified_archive_drift_is_indeterminate_before_execution(tmp_path, mutate):
    """Missing, unbound, misplaced, or changed archive evidence must stop before MMseqs executes."""
    manifest = _phrogs_manifest(tmp_path)
    mutate(manifest)

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_ASSET_PROVENANCE_MISMATCH",)
    assert commands == []


def test_phrogs_commands_use_existing_profile_query_against_created_orf_target_database(tmp_path):
    """Use MMseqs native DB commands so an existing profile DB is never reparsed as FASTA."""
    commands = build_phrogs_commands(
        mmseqs=Path("/tools/mmseqs"),
        profile_database=Path("/db/phrogs_profile_db"),
        proteins_faa=tmp_path / "proteins.faa",
        output_tsv=tmp_path / "phrogs.raw.tsv",
        temporary_dir=tmp_path / "tmp",
        threads=6,
    )

    target_database = tmp_path / "tmp" / "orf-target"
    result_database = tmp_path / "tmp" / "profile-hits"
    assert commands == (
        (
            "/tools/mmseqs",
            "createdb",
            str(tmp_path / "proteins.faa"),
            str(target_database),
        ),
        (
            "/tools/mmseqs",
            "search",
            "/db/phrogs_profile_db",
            str(target_database),
            str(result_database),
            str(tmp_path / "tmp" / "search"),
            "--threads",
            "6",
            "--alignment-mode",
            "3",
        ),
        (
            "/tools/mmseqs",
            "convertalis",
            "/db/phrogs_profile_db",
            str(target_database),
            str(result_database),
            str(tmp_path / "phrogs.raw.tsv"),
            "--format-output",
            ",".join(_PHROGS_HEADER),
        ),
    )
    assert build_phrogs_command(
        mmseqs=Path("/tools/mmseqs"),
        profile_database=Path("/db/phrogs_profile_db"),
        proteins_faa=tmp_path / "proteins.faa",
        output_tsv=tmp_path / "phrogs.raw.tsv",
        temporary_dir=tmp_path / "tmp",
        threads=6,
    ) == list(commands[-1])


def test_phrogs_runner_uses_exact_policy_derived_profile_database_when_published(tmp_path):
    """A provenance-bound policy subset replaces only the PHROGs query DB, not the record target."""
    manifest = _phrogs_manifest(tmp_path)
    derived = _attach_curated_phrogs_search_database(tmp_path, manifest)

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.PASS
    assert commands[2][1] == "search"
    assert commands[2][2] == str(derived)
    assert commands[3][1] == "convertalis"
    assert commands[3][2] == str(derived)


def test_current_raw_phrogs_manifest_is_indeterminate_without_verified_profile_identity(tmp_path):
    """Raw protein target IDs must never be guessed into PHROG families."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = {
        "lookup_path": str(tmp_path / "lookup.tsv"),
        "sequence_database": {"path": str(tmp_path / "raw-sequence-db"), "sha256": "a" * 64},
    }

    def runner(*args, **kwargs):
        raise AssertionError("an incompatible PHROGs manifest must not execute")

    result = run_phrogs(
        artifacts,
        manifest_section=manifest,
        tool_pin=_mmseqs_pin(tmp_path),
        host_domain=HostDomain.BACTERIA,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_VERIFIED_IDENTITY_MAPPING_MISSING",)


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (
            lambda manifest: manifest["profile_database"].pop("profile_id_inventory"),
            "PHROGS_VERIFIED_IDENTITY_MAPPING_MISSING",
        ),
        (
            lambda manifest: manifest["profile_database"].__setitem__("search_profile_scope", "raw_sequences"),
            "PHROGS_VERIFIED_IDENTITY_MAPPING_MISSING",
        ),
        (
            lambda manifest: manifest["profile_database"]["provenance"].__setitem__(
                "source_url", "https://phrogs.lmge.uca.fr/legacy-v6.tar.gz"
            ),
            "PHROGS_ASSET_PROVENANCE_MISMATCH",
        ),
        (
            lambda manifest: Path(f"{manifest['profile_database']['path']}.source").unlink(),
            "PHROGS_ASSET_PROVENANCE_MISMATCH",
        ),
        (
            lambda manifest: manifest["profile_database"]["profile_id_inventory"].__setitem__("count", 2),
            "PHROGS_ASSET_PROVENANCE_MISMATCH",
        ),
        (
            lambda manifest: manifest.__setitem__("annotation_url", "https://example.invalid/phrogs.tsv"),
            "PHROGS_ASSET_PROVENANCE_MISMATCH",
        ),
        (
            lambda manifest: manifest.__setitem__("annotation_sha256", "0" * 64),
            "PHROGS_ASSET_PROVENANCE_MISMATCH",
        ),
        (
            lambda manifest: manifest.__setitem__("source_sha256", "0" * 64),
            "PHROGS_ASSET_PROVENANCE_MISMATCH",
        ),
        (
            lambda manifest: Path(manifest["source_path"]).write_text("tampered annotation source\n"),
            "PHROGS_ASSET_PROVENANCE_MISMATCH",
        ),
    ],
)
def test_phrogs_final_identity_and_pharokka_provenance_contract_is_fail_closed(tmp_path, mutate, reason_code):
    """The legacy profile source or unverifiable full-profile identity must never execute."""
    manifest = _phrogs_manifest(tmp_path)
    mutate(manifest)
    output = tmp_path / "phrogs.tsv"
    _write_phrogs_output(output)

    result = parse_phrogs_output(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=manifest,
        tool_pin=_mmseqs_pin(tmp_path),
        host_domain=HostDomain.BACTERIA,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


def test_phrogs_known_full_profile_without_safety_lookup_mapping_remains_raw_only(tmp_path):
    """A valid non-lysogeny PHROG family is measured evidence but is not safety-classified."""
    output = tmp_path / "phrogs.tsv"
    row = _phrogs_row(profile="phrog_3")
    _write_phrogs_output(output, row)

    result = _parse_phrogs_output_validated(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_phrogs_manifest(tmp_path),
        tool_pin=_mmseqs_pin(tmp_path),
        host_domain=HostDomain.BACTERIA,
    )

    assert result.class_result.state is SafetyState.PASS
    assert result.class_result.findings == ()
    assert row in Path(result.raw_output_path).read_text()


def test_phrogs_safety_lookup_profile_missing_from_full_inventory_is_indeterminate(tmp_path):
    """Every safety-classified PHROG must be provably searchable in the pinned full profile DB."""
    manifest = _phrogs_manifest(tmp_path)
    lookup = Path(manifest["lookup_path"])
    lookup.write_text(
        lookup.read_text() + "phrog_999\tintegrase\tintegration and excision\thigh_confidence\tintegrase\n"
    )
    manifest["lookup_sha256"] = hashlib.sha256(lookup.read_bytes()).hexdigest()
    output = tmp_path / "phrogs.tsv"
    _write_phrogs_output(output)

    result = parse_phrogs_output(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=manifest,
        tool_pin=_mmseqs_pin(tmp_path),
        host_domain=HostDomain.BACTERIA,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_VERIFIED_IDENTITY_MAPPING_MISSING",)


@pytest.mark.parametrize("version", ["13-45111", "MMseqs release unknown"])
def test_phrogs_rejects_incompatible_or_unparsable_mmseqs_version_before_search(tmp_path, version):
    """Pharokka's profile format requires an observable MMseqs major of at least 14."""
    manifest = _phrogs_manifest(tmp_path)
    tool_pin = _mmseqs_pin(tmp_path, version=version)
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        if command[1:] == ["version"]:
            return subprocess.CompletedProcess(command, 0, stdout=version + "\n", stderr="")
        raise AssertionError("incompatible MMseqs must not run a search")

    result = run_phrogs(
        _orf_artifacts(tmp_path),
        manifest_section=manifest,
        tool_pin=tool_pin,
        host_domain=HostDomain.BACTERIA,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_ASSET_PROVENANCE_MISMATCH",)
    assert commands == [[str(tool_pin.path), "version"]]


def test_phrogs_header_only_success_is_measured_pass_for_bacterial_profile(tmp_path):
    """A verified no-hit profile search must pass while remaining required for bacterial replication."""
    output = tmp_path / "phrogs.tsv"
    _write_phrogs_output(output, comments=("normalized measured MMseqs output",))

    result = _parse_phrogs_output_validated(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_phrogs_manifest(tmp_path),
        tool_pin=_mmseqs_pin(tmp_path),
        host_domain=HostDomain.BACTERIA,
    )

    assert result.class_result.state is SafetyState.PASS
    assert result.class_result.required is True
    assert result.class_result.reason_codes == ("PHROGS_MEASURED_NO_REVIEW_HIT",)


def test_phrogs_high_confidence_profile_fails_bacterial_and_mixed_profiles(tmp_path):
    """A high-confidence integration profile must fail every bacterial-containing replication profile."""
    output = tmp_path / "phrogs.tsv"
    _write_phrogs_output(output, _phrogs_row())
    for host_domain in (HostDomain.BACTERIA, HostDomain.BACTERIA_AND_ARCHAEA):
        result = parse_phrogs_output(
            output,
            artifacts=_orf_artifacts(tmp_path / host_domain.value),
            manifest_section=_phrogs_manifest(tmp_path / host_domain.value),
            tool_pin=_mmseqs_pin(tmp_path / host_domain.value),
            host_domain=host_domain,
        )

        assert result.class_result.state is SafetyState.FAIL
        assert result.class_result.required is True
        finding = result.class_result.findings[0]
        assert finding.reason_codes == ("LYSOGENY_HIGH_CONFIDENCE_PROFILE",)
        assert finding.profile == "phrog_1"
        assert finding.query_id == "genome_1__orf0001"
        assert finding.scores["query_coverage"] == 0.9
        assert finding.scores["reference_coverage"] == 0.85
        assert finding.threshold_policy == "phrogs-homology-v1"
        assert finding.threshold_policy_sha256 == PHROGS_HOMOLOGY_POLICY_V1.sha256
        assert result.policy_sha256 == PHROGS_HOMOLOGY_POLICY_V1.sha256


def test_phrogs_review_profile_is_indeterminate_for_mixed_replication(tmp_path):
    """A material review-category profile cannot pass the strict-lytic bacterial gate."""
    output = tmp_path / "phrogs.tsv"
    _write_phrogs_output(
        output,
        _phrogs_row(profile="phrog_2", pident="25.0", qcov="0.60", tcov="0.55", evalue="1e-8"),
    )

    result = parse_phrogs_output(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_phrogs_manifest(tmp_path),
        tool_pin=_mmseqs_pin(tmp_path),
        host_domain=HostDomain.BACTERIA_AND_ARCHAEA,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("LYSOGENY_REVIEW_PROFILE",)


def test_phrogs_archaeal_only_preserves_findings_as_informational_unless_strict_lysis(tmp_path):
    """Archaeal work must retain lysogeny evidence without applying the bacterial draft profile by default."""
    output = tmp_path / "phrogs.tsv"
    _write_phrogs_output(output, _phrogs_row())
    artifacts = _orf_artifacts(tmp_path)
    manifest = _phrogs_manifest(tmp_path)
    tool_pin = _mmseqs_pin(tmp_path)

    informational = parse_phrogs_output(
        output,
        artifacts=artifacts,
        manifest_section=manifest,
        tool_pin=tool_pin,
        host_domain=HostDomain.ARCHAEA,
    )
    strict = parse_phrogs_output(
        output,
        artifacts=artifacts,
        manifest_section=manifest,
        tool_pin=tool_pin,
        host_domain=HostDomain.ARCHAEA,
        strict_lysis=True,
    )

    assert informational.class_result.state is SafetyState.FAIL
    assert informational.class_result.required is False
    assert informational.class_result.reason_codes == ("LYSOGENY_INFORMATIONAL_ARCHAEAL_PROFILE",)
    assert informational.class_result.findings == strict.class_result.findings
    assert strict.class_result.state is SafetyState.FAIL
    assert strict.class_result.required is True
    assert strict.class_result.reason_codes == ("LYSOGENY_HIGH_CONFIDENCE_PROFILE",)


@pytest.mark.parametrize(
    ("write_output", "reason_code"),
    [
        (
            lambda path: _write_phrogs_output(path, header=(*_PHROGS_HEADER, "unexpected")),
            "PHROGS_PARSER_SCHEMA_MISMATCH",
        ),
        (
            lambda path: path.write_text("\t".join(_PHROGS_HEADER) + "\n" + "\t".join(_PHROGS_HEADER) + "\n"),
            "PHROGS_DUPLICATE_HEADER",
        ),
        (
            lambda path: _write_phrogs_output(path, _phrogs_row(), _phrogs_row()),
            "PHROGS_DUPLICATE_HIT",
        ),
        (
            lambda path: _write_phrogs_output(path, _phrogs_row(), _phrogs_row(bits="119.0")),
            "PHROGS_DUPLICATE_HIT",
        ),
        (
            lambda path: _write_phrogs_output(path, _phrogs_row(profile="phrog_999")),
            "PHROGS_UNKNOWN_PROFILE_ID",
        ),
        (
            lambda path: _write_phrogs_output(path, _phrogs_row(query_id="unknown_orf")),
            "PHROGS_UNKNOWN_QUERY_ID",
        ),
        (
            lambda path: _write_phrogs_output(path, _phrogs_row(pident="nan")),
            "PHROGS_INVALID_NUMERIC_VALUE",
        ),
        (
            lambda path: _write_phrogs_output(path, _phrogs_row(qcov="80.0")),
            "PHROGS_INVALID_NUMERIC_VALUE",
        ),
    ],
)
def test_phrogs_parser_rejects_schema_duplicate_unknown_and_unit_drift(tmp_path, write_output, reason_code):
    """PHROGs ambiguity or percent/fraction confusion must never produce a strict-lytic PASS."""
    output = tmp_path / "phrogs.tsv"
    write_output(output)

    result = parse_phrogs_output(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_phrogs_manifest(tmp_path),
        tool_pin=_mmseqs_pin(tmp_path),
        host_domain=HostDomain.BACTERIA,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


def test_phrogs_runner_validates_contract_and_normalizes_empty_output(tmp_path):
    """Only a pinned profile DB with the declared orientation may create a measured lysogeny result."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _phrogs_manifest(tmp_path)
    tool_pin = _mmseqs_pin(tmp_path)
    commands = []

    def runner(command, **kwargs):
        commands.append((command, kwargs))
        if command[1:] == ["version"]:
            return subprocess.CompletedProcess(command, 0, stdout=tool_pin.version + "\n", stderr="")
        if command[1] == "convertalis":
            Path(command[5]).write_text("")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_phrogs(
        artifacts,
        manifest_section=manifest,
        tool_pin=tool_pin,
        host_domain=HostDomain.BACTERIA,
        work_dir=tmp_path / "scan",
        threads=2,
        runner=runner,
        timeout=19.0,
    )

    assert result.class_result.state is SafetyState.PASS
    assert Path(result.raw_output_path).read_text() == "\t".join(_PHROGS_HEADER) + "\n"
    assert commands[3][0] == build_phrogs_command(
        mmseqs=tool_pin.path,
        profile_database=Path(manifest["profile_database"]["path"]),
        proteins_faa=artifacts.proteins_faa,
        output_tsv=tmp_path / "scan" / "phrogs.raw.tsv",
        temporary_dir=tmp_path / "scan" / "tmp",
        threads=2,
    )
    assert PHROGS_HOMOLOGY_POLICY_V1.policy_id == "phrogs-homology-v1"


@pytest.mark.parametrize(
    ("drift", "reason_code"),
    [
        ("tool_version", "PHROGS_ASSET_PROVENANCE_MISMATCH"),
        ("database_digest", "PHROGS_ASSET_PROVENANCE_MISMATCH"),
        ("missing_output", "PHROGS_OUTPUT_MISSING"),
    ],
)
def test_phrogs_runner_fails_closed_for_version_digest_or_output_drift(tmp_path, drift, reason_code):
    """A drifted or unmeasured PHROGs search cannot qualify a strict-lytic design."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _phrogs_manifest(tmp_path)
    tool_pin = _mmseqs_pin(tmp_path)
    if drift == "database_digest":
        Path(manifest["profile_database"]["path"]).write_bytes(b"changed")

    def runner(command, **kwargs):
        if command[1:] == ["version"]:
            version = "19-drift" if drift == "tool_version" else tool_pin.version
            return subprocess.CompletedProcess(command, 0, stdout=version + "\n", stderr="")
        if drift != "missing_output":
            raise AssertionError("drifted provenance must stop before search")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_phrogs(
        artifacts,
        manifest_section=manifest,
        tool_pin=tool_pin,
        host_domain=HostDomain.BACTERIA,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("nonzero", "PHROGS_EXECUTION_FAILED"),
        ("timeout", "PHROGS_EXECUTION_TIMEOUT"),
        ("non_utf", "PHROGS_PARSER_SCHEMA_MISMATCH"),
    ],
)
def test_phrogs_runner_fails_closed_for_execution_or_malformed_output(tmp_path, failure, reason_code):
    """A failed process or undecodable MMseqs output cannot become a measured lysogeny PASS."""
    artifacts = _orf_artifacts(tmp_path)
    manifest = _phrogs_manifest(tmp_path)
    tool_pin = _mmseqs_pin(tmp_path)

    def runner(command, **kwargs):
        if command[1:] == ["version"]:
            return subprocess.CompletedProcess(command, 0, stdout=tool_pin.version + "\n", stderr="")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "nonzero":
            raise subprocess.CalledProcessError(2, command, stderr="failed")
        if command[1] == "convertalis":
            Path(command[5]).write_bytes(b"\xff\xfe")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = run_phrogs(
        artifacts,
        manifest_section=manifest,
        tool_pin=tool_pin,
        host_domain=HostDomain.BACTERIA,
        work_dir=tmp_path / "scan",
        runner=runner,
        timeout=2.5,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


@pytest.mark.parametrize(
    ("detector", "reason_code"),
    [
        ("amrfinder", "AMRFINDER_OUTPUT_EMPTY"),
        ("toxin", "TOXIN_DIAMOND_OUTPUT_EMPTY"),
        ("phrogs", "PHROGS_OUTPUT_EMPTY"),
    ],
)
def test_zero_byte_output_is_unmeasured_not_a_header_only_pass(tmp_path, detector, reason_code):
    """Only a successful header-bearing normalized output can represent a measured no-hit scan."""
    output = tmp_path / f"{detector}.tsv"
    output.write_bytes(b"")
    artifacts = _orf_artifacts(tmp_path)
    if detector == "amrfinder":
        result = parse_amrfinder_output(
            output,
            artifacts=artifacts,
            manifest_section=_amrfinder_manifest(tmp_path),
            required=True,
        )
    elif detector == "toxin":
        result = parse_toxin_diamond_output(
            output,
            artifacts=artifacts,
            manifest_section=_toxin_manifest(tmp_path),
            tool_pin=_diamond_pin(tmp_path),
            required=True,
        )
    else:
        result = parse_phrogs_output(
            output,
            artifacts=artifacts,
            manifest_section=_phrogs_manifest(tmp_path),
            tool_pin=_mmseqs_pin(tmp_path),
            host_domain=HostDomain.BACTERIA,
        )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == (reason_code,)


_ORIGIN_CROSSING_FRAME_ONE = "AAA" * 2 + "TAA" + "AAA" * 4 + "TAA" + "AAA" * 3


class _NoGenePredictor:
    def predict(self, sequence, *, circular):
        return ()


@pytest.mark.parametrize(
    ("sequence_id", "sequence", "strand", "frame", "linear_coordinates", "crossing_coordinates"),
    [
        ("plus_circle", _ORIGIN_CROSSING_FRAME_ONE, "+", 1, (10, 21), (25, 39)),
        (
            "minus_circle",
            _ORIGIN_CROSSING_FRAME_ONE.translate(str.maketrans("ACGT", "TGCA"))[::-1],
            "-",
            -1,
            (13, 24),
            (28, 42),
        ),
    ],
)
def test_circular_six_frame_fallback_retains_origin_crossing_peptides_without_duplicate_linear_calls(
    tmp_path,
    sequence_id,
    sequence,
    strand,
    frame,
    linear_coordinates,
    crossing_coordinates,
):
    """Circular fallback must add the wraparound peptide once while retaining each ordinary call once."""
    linear = prepare_orf_artifacts(
        (GenomeInput(sequence_id=sequence_id, sequence=sequence, circular=False),),
        tmp_path / "linear",
        predictor=_NoGenePredictor(),
        minimum_fallback_amino_acids=4,
    )
    circular = prepare_orf_artifacts(
        (GenomeInput(sequence_id=sequence_id, sequence=sequence, circular=True),),
        tmp_path / "circular",
        predictor=_NoGenePredictor(),
        minimum_fallback_amino_acids=4,
    )
    linear_frame_records = [
        record for record in linear.query_records if (record.strand, record.frame) == (strand, frame)
    ]
    circular_frame_records = [
        record for record in circular.query_records if (record.strand, record.frame) == (strand, frame)
    ]

    assert [
        (record.start, record.end, record.protein)
        for record in linear_frame_records
        if (record.start, record.end) == linear_coordinates
    ] == [(*linear_coordinates, "KKKK")]
    assert [
        (record.start, record.end, record.protein, record.evidence_path)
        for record in circular_frame_records
        if (record.start, record.end) == linear_coordinates
    ] == [(*linear_coordinates, "KKKK", "six-frame-fallback")]
    assert [
        (record.start, record.end, record.protein, record.evidence_path)
        for record in circular_frame_records
        if record.end > len(sequence)
    ] == [(*crossing_coordinates, "KKKKK", "six-frame-fallback")]
    assert len({record.query_id for record in circular.query_records}) == len(circular.query_records)


@pytest.mark.parametrize(
    ("sequence", "strand", "false_call"),
    [
        ("ACTAGGCGAAAGCCGCCTGAGGTGC", "-", (3, 26, -1, "CTSGGFRL")),
        ("GCACCTCAGGCGGCTTTCGCCTAGT", "+", (25, 48, 1, "CTSGGFRL")),
    ],
)
def test_circular_six_frame_fallback_never_emits_buffer_sentinel_clipped_peptide(
    tmp_path,
    sequence,
    strand,
    false_call,
):
    """The end of a finite translation buffer is not a biological stop codon."""
    artifacts = prepare_orf_artifacts(
        (GenomeInput(sequence_id="sentinel_circle", sequence=sequence, circular=True),),
        tmp_path,
        predictor=_NoGenePredictor(),
        minimum_fallback_amino_acids=8,
    )

    observed = [
        (record.start, record.end, record.frame, record.protein)
        for record in artifacts.query_records
        if record.strand == strand and record.end > len(sequence)
    ]
    assert false_call not in observed


@pytest.mark.parametrize(
    ("sequence", "expected_call"),
    [
        (_ORIGIN_CROSSING_FRAME_ONE, (25, 39, "+", 1, "KKKKK")),
        (_ORIGIN_CROSSING_FRAME_ONE.translate(str.maketrans("ACGT", "TGCA"))[::-1], (28, 42, "-", -1, "KKKKK")),
        ("ACTAGGATCGAAAGCCGCCTGAGGTGC", (23, 46, "+", 2, "GALGSKAA")),
        ("ACTAGGATCGAAAGCCGCCTGAGGTGC", (6, 29, "+", 3, "DRKPPEVH")),
        ("GCACCTCAGGCGGCTTTCGATCCTAGT", (9, 32, "-", -2, "GALGSKAA")),
        ("ACTAGGATCGAAAGCCGCCTGAGGTGC", (5, 28, "-", -3, "CTSGGFRS")),
        ("TACGTAGAGTAACGCGTAAGTGCCT", (20, 34, "+", 2, "VPYVE")),
    ],
)
def test_circular_six_frame_fallback_retains_real_stop_delimited_crossings_in_all_frames(
    tmp_path,
    sequence,
    expected_call,
):
    """Real stop-to-stop crossings remain available in every frame, including a 25-nt genome."""
    artifacts = prepare_orf_artifacts(
        (GenomeInput(sequence_id="real_stop_circle", sequence=sequence, circular=True),),
        tmp_path,
        predictor=_NoGenePredictor(),
        minimum_fallback_amino_acids=4,
    )

    observed = {
        (record.start, record.end, record.strand, record.frame, record.protein)
        for record in artifacts.query_records
        if record.end > len(sequence)
    }
    assert expected_call in observed


def test_public_amrfinder_parser_cannot_pass_unbound_header_only_output(tmp_path):
    """A stale AMRFinder header cannot certify a completed pinned search without runner evidence."""
    output = tmp_path / "amrfinder.tsv"
    _write_amrfinder_output(output)

    result = parse_amrfinder_output(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section={},
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("AMRFINDER_SEARCH_EVIDENCE_UNVALIDATED",)


def test_public_toxin_parser_cannot_pass_output_unbound_from_pinned_tool(tmp_path):
    """A stale DIAMOND header cannot pass after the purported executable bytes have drifted."""
    output = tmp_path / "toxin.tsv"
    _write_diamond_output(output)
    tool_pin = _diamond_pin(tmp_path)
    tool_pin.path.write_bytes(b"drifted-diamond")

    result = parse_toxin_diamond_output(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_toxin_manifest(tmp_path),
        tool_pin=tool_pin,
        required=True,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("TOXIN_DIAMOND_SEARCH_EVIDENCE_UNVALIDATED",)


def test_public_phrogs_parser_cannot_pass_output_unbound_from_pinned_tool(tmp_path):
    """A stale MMseqs header cannot pass after the purported executable bytes have drifted."""
    output = tmp_path / "phrogs.tsv"
    _write_phrogs_output(output)
    tool_pin = _mmseqs_pin(tmp_path)
    tool_pin.path.write_bytes(b"drifted-mmseqs")

    result = parse_phrogs_output(
        output,
        artifacts=_orf_artifacts(tmp_path),
        manifest_section=_phrogs_manifest(tmp_path),
        tool_pin=tool_pin,
        host_domain=HostDomain.BACTERIA,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_SEARCH_EVIDENCE_UNVALIDATED",)


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_amrfinder_tool_validation_os_errors_are_indeterminate(tmp_path, error_type):
    """An unreadable or unexecutable AMRFinder pin is unmeasured evidence, not an uncaught exception."""

    def runner(command, **kwargs):
        raise error_type("tool version unavailable")

    result = run_amrfinder(
        _orf_artifacts(tmp_path),
        manifest_section=_amrfinder_manifest(tmp_path),
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("AMRFINDER_ASSET_PROVENANCE_MISMATCH",)


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_toxin_tool_validation_os_errors_are_indeterminate(tmp_path, error_type):
    """An unreadable or unexecutable DIAMOND pin is unmeasured evidence, not an uncaught exception."""

    def runner(command, **kwargs):
        raise error_type("tool version unavailable")

    result = run_toxin_diamond(
        _orf_artifacts(tmp_path),
        manifest_section=_toxin_manifest(tmp_path),
        tool_pin=_diamond_pin(tmp_path),
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("TOXIN_ASSET_PROVENANCE_MISMATCH",)


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_phrogs_tool_validation_os_errors_are_indeterminate(tmp_path, error_type):
    """An unreadable or unexecutable MMseqs pin is unmeasured evidence, not an uncaught exception."""

    def runner(command, **kwargs):
        raise error_type("tool version unavailable")

    result = run_phrogs(
        _orf_artifacts(tmp_path),
        manifest_section=_phrogs_manifest(tmp_path),
        tool_pin=_mmseqs_pin(tmp_path),
        host_domain=HostDomain.BACTERIA,
        work_dir=tmp_path / "scan",
        runner=runner,
    )

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_ASSET_PROVENANCE_MISMATCH",)


def test_six_frame_fallback_deduplicates_exact_primary_call_but_preserves_distinct_evidence_path(tmp_path):
    """An identical translated call should occur once, while the retained primary record keeps its provenance."""

    class MatchingPredictor:
        def predict(self, sequence, *, circular):
            return (
                PredictedGene(
                    start=10,
                    end=21,
                    strand="+",
                    nucleotide="AAA" * 4,
                    protein="KKKK",
                ),
            )

    artifacts = prepare_orf_artifacts(
        (GenomeInput(sequence_id="dedup", sequence=_ORIGIN_CROSSING_FRAME_ONE, circular=False),),
        tmp_path,
        predictor=MatchingPredictor(),
        minimum_fallback_amino_acids=4,
    )

    exact_calls = [
        record
        for record in artifacts.query_records
        if (record.start, record.end, record.strand, record.protein) == (10, 21, "+", "KKKK")
    ]
    assert [(record.query_id, record.evidence_path) for record in exact_calls] == [("dedup__orf0001", "pyrodigal-gv")]


@pytest.mark.parametrize("profile_id", ["phrog_0", "phrog_01", "phrog_\u0661"])
def test_phrogs_profile_identity_rejects_zero_leading_zero_and_unicode_digit_aliases(tmp_path, profile_id):
    """Only positive ASCII canonical `phrog_N` identifiers may join the pinned profile inventory."""
    profile_prefix = tmp_path / "phrogs_profile_db"
    Path(f"{profile_prefix}.lookup").write_text(f"0\t{profile_id}\t0\n")

    with pytest.raises(VerifiedIdentityMappingMissingError, match="invalid or duplicate identity"):
        _read_phrogs_profile_ids(profile_prefix)
