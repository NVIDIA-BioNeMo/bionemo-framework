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
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import bionemo.evo2_phage_gen.sequence_safety_adapters as sequence_safety_adapters
from bionemo.evo2_phage_gen.design_scope import HostDomain
from bionemo.evo2_phage_gen.sequence_safety import SafetyFinding, SafetyState
from bionemo.evo2_phage_gen.sequence_safety_adapters import (
    PHROGS_HOMOLOGY_POLICY_V1,
    TOXIN_HOMOLOGY_POLICY_V1,
    GenomeInput,
    HomologyBand,
    HomologyPolicy,
    NormalizedSafetyFinding,
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
    binary = tmp_path / "amrfinder"
    binary.write_bytes(b"amrfinder-4.2.7")
    database = tmp_path / "amr-db" / "2026-07-22.1"
    database.mkdir(parents=True)
    (database / "catalog.txt").write_bytes(b"amr-db")
    return {
        "release": "amrfinder_v4.2.7",
        "binary_path": str(binary),
        "binary_sha256": hashlib.sha256(b"amrfinder-4.2.7").hexdigest(),
        "amrfinder_version": "AMRFinderPlus version 4.2.7",
        "database_path": str(database),
        "database_version": "2026-07-22.1",
        "database_sha256": hashlib.sha256(b"catalog.txt\0amr-db").hexdigest(),
    }


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
    assert result.policy_id == "amrfinder-curated-thresholds-v4.2.7"
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
        "threshold_policy": "amrfinder-curated-thresholds-v4.2.7",
        "threshold_policy_sha256": result.policy_sha256,
        "tool_path": manifest["binary_path"],
        "tool_sha256": manifest["binary_sha256"],
        "profile": None,
    }


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
        threads=3,
        output_tsv=tmp_path / "scan" / "amrfinder.tsv",
    )
    assert commands[2][1] == {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 17.0,
    }


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
        "Entry\tEntry Name\tProtein names\tGene Names\tOrganism\tOrganism (ID)\tFunction [CC]\n"
        "P0C1\tSYN_TOX\tSynthetic toxin control\tsyn\tSynthetic virus\t10239\tSynthetic annotation\n"
    )
    fasta = tmp_path / "reviewed_toxins.faa"
    fasta.write_text(">sp|P0C1|SYN_TOX Synthetic toxin control\nMPEPTIDE\n")
    database = tmp_path / "reviewed_toxins.dmnd"
    database.write_bytes(b"reviewed-toxin-diamond-db")
    return {
        "uniprot_release": "2026_03",
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
    assert result.policy_id == TOXIN_HOMOLOGY_POLICY_V1.policy_id
    assert result.policy_sha256 == TOXIN_HOMOLOGY_POLICY_V1.sha256


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
    assert finding.evidence_path == "pyrodigal-gv"
    assert finding.threshold_policy_sha256 == TOXIN_HOMOLOGY_POLICY_V1.sha256
    assert finding.threshold_policy == "toxin-homology-v1"
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
def test_toxin_runner_fails_closed_for_version_digest_or_output_drift(tmp_path, drift, reason_code):
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
def test_toxin_runner_fails_closed_for_nonzero_exit_and_timeout(tmp_path, failure, reason_code):
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


def test_toxin_runner_fails_closed_for_non_utf_output(tmp_path):
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


@pytest.fixture(autouse=True)
def _pin_synthetic_phrogs_annotation_for_adapter_tests(monkeypatch):
    """Use a deterministic local annotation pin; production retains the official PHROGs v4 SHA-256."""
    monkeypatch.setattr(sequence_safety_adapters, "_PHROGS_ANNOTATION_SHA256", _PHROGS_TEST_ANNOTATION_SHA256)


def _phrogs_manifest(tmp_path):
    annotation_source = tmp_path / "phrog_annot_v4.tsv"
    annotation_source.write_text(_PHROGS_ANNOTATION_SOURCE_TEXT)
    archive_payload = b"pinned Pharokka profile archive fixture\n"
    archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
    archive_path = tmp_path / "downloads" / "phrogs_safety_profile_archives" / f"{archive_sha256}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(archive_payload)
    lookup = tmp_path / "phrogs_integration_excision_v4.tsv"
    lookup_rows = [
        *(
            f"{profile}\tintegrase\tintegration and excision\thigh_confidence\tintegrase\n"
            for profile in _PHROGS_HIGH_PROFILE_IDS
        ),
        *(
            f"{profile}\tputative recombinase\tintegration and excision\treview\trecombinase\n"
            for profile in _PHROGS_REVIEW_PROFILE_IDS
        ),
    ]
    lookup.write_text("phrog\tannot\tcategory\tconfidence\tmatched_term\n" + "".join(lookup_rows))
    profile_prefix = tmp_path / "phrogs_profile_db"
    profile_lookup = "".join(
        f"{index}\t{profile_id}\t{index}\n" for index, profile_id in enumerate(_PHROGS_FULL_PROFILE_IDS)
    ).encode()
    profile_files = {
        profile_prefix: b"profile-db",
        Path(f"{profile_prefix}.index"): b"profile-index",
        Path(f"{profile_prefix}.dbtype"): b"profile-dbtype",
        Path(f"{profile_prefix}.lookup"): profile_lookup,
        Path(f"{profile_prefix}.source"): b"pharokka-v1.8.0\n",
        Path(f"{profile_prefix}_h"): ("\n".join(_PHROGS_FULL_PROFILE_IDS) + "\n").encode(),
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
    for profile_id in sorted(_PHROGS_FULL_PROFILE_IDS):
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
        "review_terms": ["recombinase", "repressor", "lysogeny", "integration", "excision"],
        "lookup_path": str(lookup),
        "lookup_sha256": hashlib.sha256(lookup.read_bytes()).hexdigest(),
        "lookup_counts": {"total": 109, "high_confidence": 57, "review": 52},
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
                "count": len(_PHROGS_FULL_PROFILE_IDS),
                "sha256": profile_id_digest.hexdigest(),
            },
            "provenance": {
                "source_url": "https://zenodo.org/record/17110353/files/pharokka_v1.8.0_databases.tar.gz",
                "archive_observed_sha256": archive_sha256,
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
        Path(command[4]).write_text("")
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


def test_phrogs_missing_pinned_safety_lookup_counts_is_indeterminate_before_execution(tmp_path):
    """Removing the cardinality pin must not turn an incomplete safety gate into measured PASS."""
    manifest = _phrogs_manifest(tmp_path)
    manifest.pop("lookup_counts")

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.class_result.reason_codes == ("PHROGS_ASSET_PROVENANCE_MISMATCH",)
    assert commands == []


def test_phrogs_self_consistent_truncated_safety_lookup_is_indeterminate_before_execution(tmp_path):
    """A re-digested lookup missing one reviewed profile must still fail closed before MMseqs runs."""
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
    archive_payload = b"verified Pharokka profile archive fixture\n"
    archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
    archive_path = tmp_path / "downloads" / "phrogs_safety_profile_archives" / f"{archive_sha256}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(archive_payload)
    provenance = manifest["profile_database"]["provenance"]
    provenance["archive_observed_sha256"] = archive_sha256
    provenance["verified_archive"] = {"path": str(archive_path.resolve()), "sha256": archive_sha256}

    result, commands = _run_phrogs_with_measured_empty_output(tmp_path, manifest)

    assert result.class_result.state is SafetyState.PASS
    assert len(commands) == 2
    assert commands[0][1:] == ("version",)
    assert commands[1][1] == "easy-search"


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


def test_phrogs_command_uses_profile_query_against_predicted_orf_target(tmp_path):
    """Reversing the verified profile orientation would destroy the profile-to-lookup identity join."""
    command = build_phrogs_command(
        mmseqs=Path("/tools/mmseqs"),
        profile_database=Path("/db/phrogs_profile_db"),
        proteins_faa=tmp_path / "proteins.faa",
        output_tsv=tmp_path / "phrogs.raw.tsv",
        temporary_dir=tmp_path / "tmp",
        threads=6,
    )

    assert command == [
        "/tools/mmseqs",
        "easy-search",
        "/db/phrogs_profile_db",
        str(tmp_path / "proteins.faa"),
        str(tmp_path / "phrogs.raw.tsv"),
        str(tmp_path / "tmp"),
        "--threads",
        "6",
        "--format-output",
        ",".join(_PHROGS_HEADER),
    ]


def test_current_raw_phrogs_manifest_fails_closed_without_verified_profile_identity(tmp_path):
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
        Path(command[4]).write_text("")
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
    assert commands[1][0] == build_phrogs_command(
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
        Path(command[4]).write_bytes(b"\xff\xfe")
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
