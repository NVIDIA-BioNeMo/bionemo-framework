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

"""Independent, provenance-rich adapters for phage sequence-safety scanners.

An adapter ``PASS`` means only that a completed search found no qualifying hit
in its pinned reference and policy bands. It is not proof of strict lysis,
absence of harmful sequence, or regulatory compliance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Protocol
from urllib.parse import quote

from bionemo.evo2_phage_gen.design_scope import HostDomain
from bionemo.evo2_phage_gen.external_assets import (
    DEFAULT_PHROGS_ANNOTATION_SHA256,
    DEFAULT_PHROGS_ANNOTATION_URL,
    PHROGS_HIGH_CONFIDENCE_TERMS,
    PHROGS_INTEGRATION_EXCISION_CATEGORY,
    PHROGS_REVIEW_TERMS,
)
from bionemo.evo2_phage_gen.sequence_safety import SafetyClassResult, SafetyFinding, SafetyState


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_SEQUENCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class GenomeInput:
    """One genome supplied for coordinated ORF prediction."""

    sequence_id: str
    sequence: str
    circular: bool = True

    def __post_init__(self) -> None:
        """Normalize sequence case and reject ambiguous identifiers or empty inputs."""
        if not _SEQUENCE_ID_PATTERN.fullmatch(self.sequence_id):
            raise ValueError(f"invalid genome sequence ID: {self.sequence_id!r}")
        normalized = "".join(self.sequence.split()).upper()
        if not normalized or set(normalized) - set("ACGTN"):
            raise ValueError(f"genome {self.sequence_id!r} must contain only A, C, G, T, or N")
        object.__setattr__(self, "sequence", normalized)


@dataclass(frozen=True)
class PredictedGene:
    """A strand-aware, one-based inclusive gene call from an injected predictor."""

    start: int
    end: int
    strand: str
    nucleotide: str
    protein: str

    def __post_init__(self) -> None:
        """Reject coordinates or strands that cannot be represented in GFF."""
        if self.start < 1 or self.end < self.start:
            raise ValueError("predicted gene coordinates must be one-based, inclusive, and ordered")
        if self.strand not in {"+", "-"}:
            raise ValueError("predicted gene strand must be '+' or '-'")
        if not self.nucleotide or not self.protein:
            raise ValueError("predicted genes require nucleotide and protein sequences")


class GenePredictor(Protocol):
    """Injectable boundary around pyrodigal-gv for deterministic unit tests."""

    def predict(self, sequence: str, *, circular: bool) -> tuple[PredictedGene, ...]:
        """Return normalized calls on both strands."""


@dataclass(frozen=True)
class ORFQueryRecord:
    """Coordinates and provenance for one primary or six-frame protein query."""

    query_id: str
    sequence_id: str
    start: int
    end: int
    strand: str
    frame: int
    nucleotide: str
    protein: str
    evidence_path: str


@dataclass(frozen=True)
class ORFArtifacts:
    """Coordinated files and query metadata consumed by the scanner adapters."""

    genomes_fna: Path
    proteins_faa: Path
    proteins_fna: Path
    proteins_gff: Path
    all_queries_faa: Path
    query_records: tuple[ORFQueryRecord, ...]

    def __post_init__(self) -> None:
        """Freeze the query inventory."""
        object.__setattr__(self, "query_records", tuple(self.query_records))


@dataclass(frozen=True)
class ORFPreparationResult:
    """Fail-closed result for optional runtime loading of the required predictor."""

    state: SafetyState
    artifacts: ORFArtifacts | None
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolPin:
    """Immutable executable digest and exact version assertion."""

    path: Path
    sha256: str
    version: str
    version_args: tuple[str, ...] = ("--version",)

    def __post_init__(self) -> None:
        """Normalize path/arguments and reject malformed pins."""
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "version_args", tuple(self.version_args))
        normalized_digest = self.sha256.removeprefix("sha256:").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized_digest):
            raise ValueError("tool SHA-256 must be a 64-character hexadecimal digest")
        if not self.version or not self.version_args:
            raise ValueError("tool pin requires an exact version and version command arguments")
        object.__setattr__(self, "sha256", normalized_digest)


class AssetProvenanceError(RuntimeError):
    """A required executable or database no longer matches its recorded identity."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_mapping_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_tool_pin(
    pin: ToolPin,
    *,
    runner: CommandRunner = subprocess.run,
    timeout: float = 300.0,
) -> str:
    """Verify executable bytes and its exact observed version before any search."""
    if not pin.path.is_file():
        raise AssetProvenanceError(f"pinned tool path is missing: {pin.path}")
    observed_digest = _sha256_file(pin.path)
    if observed_digest != pin.sha256:
        raise AssetProvenanceError(f"pinned tool digest drift: expected {pin.sha256}, observed {observed_digest}")
    completed = runner(
        [str(pin.path), *pin.version_args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    observed_version = completed.stdout.strip()
    if observed_version != pin.version:
        raise AssetProvenanceError(
            f"pinned tool version drift: expected {pin.version!r}, observed {observed_version!r}"
        )
    return observed_version


class _PyrodigalGVPredictor:
    """Lazy pyrodigal-gv adapter with explicit circular unrolling."""

    def __init__(self, module: object) -> None:
        self._module = module

    def predict(self, sequence: str, *, circular: bool) -> tuple[PredictedGene, ...]:
        """Predict both strands and normalize centered circular calls to one genome copy."""
        sequence_length = len(sequence)
        search_sequence = sequence * 3 if circular else sequence
        finder = self._module.ViralGeneFinder(meta=True, viral_only=False, closed=not circular)
        genes = finder.find_genes(search_sequence)
        calls: list[PredictedGene] = []
        seen: set[tuple[int, int, str, str]] = set()
        for gene in genes:
            start = int(gene.begin)
            end = int(gene.end)
            if circular:
                if not sequence_length < start <= 2 * sequence_length:
                    continue
                start -= sequence_length
                end -= sequence_length
            strand = "+" if int(gene.strand) == 1 else "-"
            protein = str(gene.translate(include_stop=False)).rstrip("*")
            nucleotide = str(gene.sequence())
            key = (start, end, strand, protein)
            if key in seen:
                continue
            seen.add(key)
            calls.append(
                PredictedGene(
                    start=start,
                    end=end,
                    strand=strand,
                    nucleotide=nucleotide,
                    protein=protein,
                )
            )
        return tuple(sorted(calls, key=lambda call: (call.start, call.end, call.strand, call.protein)))


def _new_pyrodigal_predictor() -> GenePredictor:
    """Load pyrodigal-gv only when prediction is requested."""
    import pyrodigal_gv

    return _PyrodigalGVPredictor(pyrodigal_gv)


def _primary_frame(gene: PredictedGene, sequence_length: int) -> int:
    if gene.strand == "+":
        return (gene.start - 1) % 3 + 1
    end_on_genome = (gene.end - 1) % sequence_length + 1
    return -((sequence_length - end_on_genome) % 3 + 1)


def _six_frame_records(
    genome: GenomeInput,
    *,
    minimum_amino_acids: int,
) -> list[ORFQueryRecord]:
    """Translate all six frames, adding one normalized origin-crossing segment per circular frame."""
    from Bio.Seq import Seq

    records: list[ORFQueryRecord] = []
    seen: set[tuple[int, int, str, str, str]] = set()
    segment_counts: dict[int, int] = {}
    sequence_length = len(genome.sequence)
    oriented_sequences = (("+", genome.sequence), ("-", str(Seq(genome.sequence).reverse_complement())))

    def append_record(
        *,
        start: int,
        end: int,
        strand: str,
        frame: int,
        nucleotide: str,
        protein: str,
    ) -> None:
        key = (start, end, strand, nucleotide, protein)
        if key in seen:
            return
        seen.add(key)
        segment_counts[frame] = segment_counts.get(frame, 0) + 1
        frame_label = f"p{frame}" if frame > 0 else f"m{-frame}"
        query_id = f"{genome.sequence_id}__sixframe_{frame_label}_{segment_counts[frame]:04d}"
        records.append(
            ORFQueryRecord(
                query_id=query_id,
                sequence_id=genome.sequence_id,
                start=start,
                end=end,
                strand=strand,
                frame=frame,
                nucleotide=nucleotide,
                protein=protein,
                evidence_path="six-frame-fallback",
            )
        )

    for strand, oriented in oriented_sequences:
        for offset in range(3):
            usable_length = ((len(oriented) - offset) // 3) * 3
            if usable_length == 0:
                continue
            coding = oriented[offset : offset + usable_length]
            translated = str(Seq(coding).translate())
            amino_start = 0
            for amino_end in range(len(translated) + 1):
                if amino_end < len(translated) and translated[amino_end] != "*":
                    continue
                protein = translated[amino_start:amino_end]
                if len(protein) >= minimum_amino_acids:
                    if strand == "+":
                        start = offset + 3 * amino_start + 1
                        end = offset + 3 * amino_end
                        frame = offset + 1
                    else:
                        start = sequence_length - (offset + 3 * amino_end) + 1
                        end = sequence_length - (offset + 3 * amino_start)
                        frame = -(offset + 1)
                    nucleotide = coding[3 * amino_start : 3 * amino_end]
                    append_record(
                        start=start,
                        end=end,
                        strand=strand,
                        frame=frame,
                        nucleotide=nucleotide,
                        protein=protein,
                    )
                amino_start = amino_end + 1

    if not genome.circular:
        return records

    for strand, oriented in oriented_sequences:
        search_oriented = oriented * 3
        for offset in range(3):
            usable_length = ((len(search_oriented) - offset) // 3) * 3
            coding = search_oriented[offset : offset + usable_length]
            translated = str(Seq(coding).translate())
            amino_start = 0
            for amino_end, amino_acid in enumerate(translated):
                if amino_acid != "*":
                    continue
                protein = translated[amino_start:amino_end]
                nucleotide_start = offset + 3 * amino_start
                nucleotide_end = offset + 3 * amino_end
                crosses_center_origin = (
                    sequence_length <= nucleotide_start < 2 * sequence_length
                    and nucleotide_end > 2 * sequence_length
                    and nucleotide_end - nucleotide_start <= sequence_length
                )
                if len(protein) >= minimum_amino_acids and crosses_center_origin:
                    oriented_start = nucleotide_start - sequence_length
                    oriented_end = nucleotide_end - sequence_length
                    if strand == "+":
                        start = oriented_start + 1
                        end = oriented_end
                        frame = (start - 1) % 3 + 1
                    else:
                        start = 2 * sequence_length - oriented_end + 1
                        end = 2 * sequence_length - oriented_start
                        end_on_genome = (end - 1) % sequence_length + 1
                        frame = -((sequence_length - end_on_genome) % 3 + 1)
                    append_record(
                        start=start,
                        end=end,
                        strand=strand,
                        frame=frame,
                        nucleotide=coding[3 * amino_start : 3 * amino_end],
                        protein=protein,
                    )
                amino_start = amino_end + 1
    return records


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{identifier}\n{sequence}\n" for identifier, sequence in records))


def prepare_orf_artifacts(
    genomes: tuple[GenomeInput, ...],
    work_dir: Path,
    *,
    predictor: GenePredictor | None = None,
    minimum_fallback_amino_acids: int = 8,
) -> ORFArtifacts:
    """Write coordinated primary ORFs plus provenance-marked six-frame fallback queries."""
    genomes = tuple(genomes)
    sequence_ids = [genome.sequence_id for genome in genomes]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("duplicate genome sequence ID")
    if not genomes:
        raise ValueError("at least one genome is required for ORF prediction")
    if minimum_fallback_amino_acids < 1:
        raise ValueError("minimum fallback peptide length must be positive")
    selected_predictor = predictor if predictor is not None else _new_pyrodigal_predictor()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    genomes_fna = work_dir / "genomes.fna"
    proteins_faa = work_dir / "proteins.faa"
    proteins_fna = work_dir / "proteins.fna"
    proteins_gff = work_dir / "proteins.gff"
    all_queries_faa = work_dir / "all_queries.faa"

    primary_records: list[ORFQueryRecord] = []
    fallback_records: list[ORFQueryRecord] = []
    gff_lines = ["##gff-version 3"]
    for genome in genomes:
        gff_lines.append(f"##sequence-region {genome.sequence_id} 1 {len(genome.sequence)}")
        genome_primary_keys: set[tuple[int, int, str, str]] = set()
        calls = selected_predictor.predict(genome.sequence, circular=genome.circular)
        for ordinal, gene in enumerate(calls, start=1):
            query_id = f"{genome.sequence_id}__orf{ordinal:04d}"
            genome_primary_keys.add((gene.start, gene.end, gene.strand, gene.protein))
            primary_records.append(
                ORFQueryRecord(
                    query_id=query_id,
                    sequence_id=genome.sequence_id,
                    start=gene.start,
                    end=gene.end,
                    strand=gene.strand,
                    frame=_primary_frame(gene, len(genome.sequence)),
                    nucleotide=gene.nucleotide,
                    protein=gene.protein,
                    evidence_path="pyrodigal-gv",
                )
            )
            escaped_id = quote(query_id, safe="._-")
            gff_lines.append(
                "\t".join(
                    (
                        genome.sequence_id,
                        "pyrodigal-gv",
                        "CDS",
                        str(gene.start),
                        str(gene.end),
                        ".",
                        gene.strand,
                        "0",
                        f"ID={escaped_id};Name={escaped_id}",
                    )
                )
            )
        fallback_records.extend(
            record
            for record in _six_frame_records(genome, minimum_amino_acids=minimum_fallback_amino_acids)
            if (record.start, record.end, record.strand, record.protein) not in genome_primary_keys
        )

    query_records = (*primary_records, *fallback_records)
    _write_fasta(
        genomes_fna,
        [
            (f"{genome.sequence_id} circular={'true' if genome.circular else 'false'}", genome.sequence)
            for genome in genomes
        ],
    )
    _write_fasta(proteins_faa, [(record.query_id, record.protein) for record in primary_records])
    _write_fasta(proteins_fna, [(record.query_id, record.nucleotide) for record in primary_records])
    proteins_gff.write_text("\n".join(gff_lines) + "\n")
    _write_fasta(all_queries_faa, [(record.query_id, record.protein) for record in query_records])
    return ORFArtifacts(
        genomes_fna=genomes_fna,
        proteins_faa=proteins_faa,
        proteins_fna=proteins_fna,
        proteins_gff=proteins_gff,
        all_queries_faa=all_queries_faa,
        query_records=query_records,
    )


def prepare_orf_artifacts_checked(
    genomes: tuple[GenomeInput, ...],
    work_dir: Path,
    *,
    predictor: GenePredictor | None = None,
    minimum_fallback_amino_acids: int = 8,
) -> ORFPreparationResult:
    """Map unavailable pyrodigal-gv runtime into the canonical fail-closed state."""
    try:
        artifacts = prepare_orf_artifacts(
            genomes,
            work_dir,
            predictor=predictor,
            minimum_fallback_amino_acids=minimum_fallback_amino_acids,
        )
    except (ImportError, ModuleNotFoundError):
        return ORFPreparationResult(
            state=SafetyState.INDETERMINATE,
            artifacts=None,
            reason_codes=("ORF_PREDICTOR_UNAVAILABLE",),
        )
    return ORFPreparationResult(state=SafetyState.PASS, artifacts=artifacts)


@dataclass(frozen=True, kw_only=True)
class NormalizedSafetyFinding(SafetyFinding):
    """A canonical safety finding with complete detector and sequence provenance."""

    detector: str
    accession: str
    query_id: str
    sequence_id: str
    start: int
    end: int
    strand: str
    frame: int
    scores: Mapping[str, float]
    thresholds: Mapping[str, float]
    source_path: str
    source_sha256: str
    tool_version: str
    database_version: str
    evidence_path: str
    evidence_method: str
    threshold_policy: str
    threshold_policy_sha256: str
    tool_path: str
    tool_sha256: str
    profile: str | None = None

    def __post_init__(self) -> None:
        """Freeze nested score and threshold mappings."""
        super().__post_init__()
        if not re.fullmatch(r"[0-9a-f]{64}", self.threshold_policy_sha256):
            raise ValueError("threshold policy SHA-256 must be a lowercase digest")
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))

    def to_dict(self) -> dict[str, object]:
        """Serialize all normalized evidence without hiding provenance in an identifier."""
        return {
            **super().to_dict(),
            "detector": self.detector,
            "accession": self.accession,
            "query_id": self.query_id,
            "sequence_id": self.sequence_id,
            "start": self.start,
            "end": self.end,
            "strand": self.strand,
            "frame": self.frame,
            "scores": dict(self.scores),
            "thresholds": dict(self.thresholds),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "tool_version": self.tool_version,
            "database_version": self.database_version,
            "evidence_path": self.evidence_path,
            "evidence_method": self.evidence_method,
            "threshold_policy": self.threshold_policy,
            "threshold_policy_sha256": self.threshold_policy_sha256,
            "tool_path": self.tool_path,
            "tool_sha256": self.tool_sha256,
            "profile": self.profile,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> NormalizedSafetyFinding:
        """Restore the exact normalized finding schema emitted by :meth:`to_dict`."""
        expected_keys = frozenset(
            {
                "safety_class",
                "state",
                "reason_codes",
                "finding_id",
                "detector",
                "accession",
                "query_id",
                "sequence_id",
                "start",
                "end",
                "strand",
                "frame",
                "scores",
                "thresholds",
                "source_path",
                "source_sha256",
                "tool_version",
                "database_version",
                "evidence_path",
                "evidence_method",
                "threshold_policy",
                "threshold_policy_sha256",
                "tool_path",
                "tool_sha256",
                "profile",
            }
        )
        actual_keys = frozenset(value)
        if actual_keys != expected_keys:
            raise ValueError(
                "normalized finding keys do not match schema; "
                f"unknown={sorted(actual_keys - expected_keys)}, "
                f"missing={sorted(expected_keys - actual_keys)}"
            )

        def required_string(name: str) -> str:
            item = value[name]
            if not isinstance(item, str) or not item:
                raise ValueError(f"normalized finding {name} must be a non-empty string")
            return item

        reason_codes = value["reason_codes"]
        if not isinstance(reason_codes, (list, tuple)) or not all(isinstance(item, str) for item in reason_codes):
            raise ValueError("normalized finding reason_codes must be a string list")
        scores = value["scores"]
        thresholds = value["thresholds"]
        if not isinstance(scores, Mapping) or not isinstance(thresholds, Mapping):
            raise ValueError("normalized finding scores and thresholds must be mappings")
        finding_id = value["finding_id"]
        profile = value["profile"]
        if finding_id is not None and not isinstance(finding_id, str):
            raise ValueError("normalized finding finding_id must be a string or null")
        if profile is not None and not isinstance(profile, str):
            raise ValueError("normalized finding profile must be a string or null")
        try:
            state = SafetyState(str(value["state"]))
        except ValueError as error:
            raise ValueError("normalized finding state is invalid") from error
        for name in ("start", "end", "frame"):
            if type(value[name]) is not int:
                raise ValueError(f"normalized finding {name} must be an integer")
        if value["strand"] not in {"+", "-"}:
            raise ValueError("normalized finding strand must be '+' or '-'")

        return cls(
            safety_class=required_string("safety_class"),
            state=state,
            reason_codes=tuple(reason_codes),
            finding_id=finding_id,
            detector=required_string("detector"),
            accession=required_string("accession"),
            query_id=required_string("query_id"),
            sequence_id=required_string("sequence_id"),
            start=value["start"],
            end=value["end"],
            strand=required_string("strand"),
            frame=value["frame"],
            scores=scores,
            thresholds=thresholds,
            source_path=required_string("source_path"),
            source_sha256=required_string("source_sha256"),
            tool_version=required_string("tool_version"),
            database_version=required_string("database_version"),
            evidence_path=required_string("evidence_path"),
            evidence_method=required_string("evidence_method"),
            threshold_policy=required_string("threshold_policy"),
            threshold_policy_sha256=required_string("threshold_policy_sha256"),
            tool_path=required_string("tool_path"),
            tool_sha256=required_string("tool_sha256"),
            profile=profile,
        )


@dataclass(frozen=True)
class AdapterResult:
    """One canonical class result plus scanner evidence and execution provenance."""

    class_result: SafetyClassResult
    supplemental_findings: tuple[NormalizedSafetyFinding, ...] = ()
    command: tuple[str, ...] = ()
    raw_output_path: str | None = None
    raw_output_sha256: str | None = None
    policy_id: str | None = None
    policy_sha256: str | None = None

    def __post_init__(self) -> None:
        """Freeze command and supplemental finding inventories."""
        object.__setattr__(self, "supplemental_findings", tuple(self.supplemental_findings))
        object.__setattr__(self, "command", tuple(self.command))
        if (self.policy_id is None) != (self.policy_sha256 is None):
            raise ValueError("adapter policy ID and SHA-256 must be recorded together")
        if self.policy_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.policy_sha256):
            raise ValueError("adapter policy SHA-256 must be a lowercase digest")


def _bind_result_policy(
    *, policy_id: str, policy_sha256: str, policy_kwarg: bool = False
) -> Callable[[Callable[..., AdapterResult]], Callable[..., AdapterResult]]:
    """Attach a policy identity to every return path of a public adapter."""

    def decorate(function: Callable[..., AdapterResult]) -> Callable[..., AdapterResult]:
        @wraps(function)
        def wrapped(*args: object, **kwargs: object) -> AdapterResult:
            result = function(*args, **kwargs)
            selected_policy = kwargs.get("policy") if policy_kwarg else None
            selected_id = policy_id if selected_policy is None else selected_policy.policy_id
            selected_sha256 = policy_sha256 if selected_policy is None else selected_policy.sha256
            return replace(result, policy_id=selected_id, policy_sha256=selected_sha256)

        return wrapped

    return decorate


def build_amrfinder_command(
    *,
    amrfinder: Path,
    genomes_fna: Path,
    proteins_faa: Path,
    proteins_gff: Path,
    database_dir: Path,
    threads: int,
    output_tsv: Path,
) -> list[str]:
    """Build AMRFinderPlus's combined nucleotide, protein, and GFF command."""
    return [
        str(amrfinder),
        "-n",
        str(genomes_fna),
        "-p",
        str(proteins_faa),
        "-g",
        str(proteins_gff),
        "--annotation_format",
        "standard",
        "--plus",
        "--print_node",
        "--database",
        str(database_dir),
        "--threads",
        str(threads),
        "-o",
        str(output_tsv),
    ]


_AMRFINDER_COLUMNS = (
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
_AMRFINDER_NUMERIC_COLUMNS = (
    "Target length",
    "Reference sequence length",
    "% Coverage of reference",
    "% Identity to reference",
    "Alignment length",
)


def _class_result(
    safety_class: str,
    state: SafetyState,
    *,
    required: bool,
    findings: tuple[NormalizedSafetyFinding, ...] = (),
    reason_codes: tuple[str, ...] = (),
) -> SafetyClassResult:
    return SafetyClassResult(
        safety_class=safety_class,
        state=state,
        required=required,
        findings=findings,
        reason_codes=reason_codes,
    )


def _indeterminate_adapter_result(
    safety_class: str,
    *,
    required: bool,
    reason_code: str,
    command: tuple[str, ...] = (),
    raw_output_path: Path | None = None,
) -> AdapterResult:
    return AdapterResult(
        class_result=_class_result(
            safety_class,
            SafetyState.INDETERMINATE,
            required=required,
            reason_codes=(reason_code,),
        ),
        command=command,
        raw_output_path=str(raw_output_path) if raw_output_path is not None else None,
    )


def _without_validated_search_pass(
    result: AdapterResult,
    *,
    safety_class: str,
    reason_code: str,
) -> AdapterResult:
    """Keep conservative findings but never certify PASS from a caller-supplied output file."""
    if result.class_result.state is not SafetyState.PASS:
        return result
    return replace(
        result,
        class_result=_class_result(
            safety_class,
            SafetyState.INDETERMINATE,
            required=result.class_result.required,
            findings=tuple(result.class_result.findings),
            reason_codes=(reason_code,),
        ),
    )


def _query_record_index(artifacts: ORFArtifacts, *, primary_only: bool = False) -> dict[str, ORFQueryRecord]:
    records = (
        record for record in artifacts.query_records if not primary_only or record.evidence_path == "pyrodigal-gv"
    )
    index: dict[str, ORFQueryRecord] = {}
    for record in records:
        if record.query_id in index:
            raise ValueError(f"duplicate ORF query ID: {record.query_id}")
        index[record.query_id] = record
    return index


def _finite_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite numeric value: {value}")
    return number


_AMRFINDER_POLICY_ID = "amrfinder-curated-thresholds-v4.2.7"
_AMRFINDER_POLICY_SHA256 = _canonical_mapping_sha256(
    {
        "policy_id": _AMRFINDER_POLICY_ID,
        "amrfinder_release": "amrfinder_v4.2.7",
        "curated_identity_threshold_overrides": False,
        "amr_type_action": "FAIL",
        "plus_virulence_action": "SUPPLEMENTAL_TOXIN_EVIDENCE",
    }
)


def _amrfinder_finding(
    row: Mapping[str, str],
    record: ORFQueryRecord,
    *,
    safety_class: str,
    state: SafetyState,
    reason_code: str,
    manifest_section: Mapping[str, object],
) -> NormalizedSafetyFinding:
    accession = row["Closest reference accession"]
    if accession in {"", "NA"}:
        accession = row["HMM accession"]
    return NormalizedSafetyFinding(
        safety_class=safety_class,
        state=state,
        reason_codes=(reason_code,),
        finding_id=f"{safety_class}:{record.query_id}:{accession}",
        detector="amrfinder-plus",
        accession=accession,
        query_id=record.query_id,
        sequence_id=record.sequence_id,
        start=record.start,
        end=record.end,
        strand=record.strand,
        frame=record.frame,
        scores={
            "alignment_length": _finite_number(row["Alignment length"]),
            "identity": _finite_number(row["% Identity to reference"]),
            "reference_coverage": _finite_number(row["% Coverage of reference"]),
            "reference_length": _finite_number(row["Reference sequence length"]),
            "target_length": _finite_number(row["Target length"]),
        },
        thresholds={},
        source_path=str(manifest_section["database_path"]),
        source_sha256=str(manifest_section["database_sha256"]),
        tool_version=str(manifest_section["amrfinder_version"]),
        database_version=str(manifest_section["database_version"]),
        evidence_path=record.evidence_path,
        evidence_method=row["Method"],
        threshold_policy=_AMRFINDER_POLICY_ID,
        threshold_policy_sha256=_AMRFINDER_POLICY_SHA256,
        tool_path=str(manifest_section["binary_path"]),
        tool_sha256=str(manifest_section["binary_sha256"]),
        profile=None,
    )


@_bind_result_policy(policy_id=_AMRFINDER_POLICY_ID, policy_sha256=_AMRFINDER_POLICY_SHA256)
def parse_amrfinder_output(
    output_tsv: Path,
    *,
    artifacts: ORFArtifacts,
    manifest_section: Mapping[str, object],
    required: bool,
) -> AdapterResult:
    """Parse AMRFinderPlus v4.2.7 output with an exact, fail-closed schema."""
    output_tsv = Path(output_tsv)
    if not output_tsv.is_file():
        return _indeterminate_adapter_result(
            "amr", required=required, reason_code="AMRFINDER_OUTPUT_MISSING", raw_output_path=output_tsv
        )
    if output_tsv.stat().st_size == 0:
        return _indeterminate_adapter_result(
            "amr", required=required, reason_code="AMRFINDER_OUTPUT_EMPTY", raw_output_path=output_tsv
        )
    try:
        output_text = output_tsv.read_text()
    except (OSError, UnicodeError):
        return _indeterminate_adapter_result(
            "amr", required=required, reason_code="AMRFINDER_PARSER_SCHEMA_MISMATCH", raw_output_path=output_tsv
        )
    noncomment_lines = [line for line in output_text.splitlines() if line.strip() and not line.startswith("#")]
    if not noncomment_lines:
        return _indeterminate_adapter_result(
            "amr", required=required, reason_code="AMRFINDER_PARSER_SCHEMA_MISMATCH", raw_output_path=output_tsv
        )
    header = tuple(noncomment_lines[0].split("\t"))
    if header != _AMRFINDER_COLUMNS:
        return _indeterminate_adapter_result(
            "amr", required=required, reason_code="AMRFINDER_PARSER_SCHEMA_MISMATCH", raw_output_path=output_tsv
        )

    try:
        query_records = _query_record_index(artifacts, primary_only=True)
    except ValueError:
        return _indeterminate_adapter_result(
            "amr", required=required, reason_code="AMRFINDER_DUPLICATE_QUERY_ID", raw_output_path=output_tsv
        )
    amr_findings: list[NormalizedSafetyFinding] = []
    supplemental_findings: list[NormalizedSafetyFinding] = []
    seen_rows: set[tuple[str, ...]] = set()
    for line in noncomment_lines[1:]:
        fields = tuple(line.split("\t"))
        if fields == _AMRFINDER_COLUMNS:
            return _indeterminate_adapter_result(
                "amr", required=required, reason_code="AMRFINDER_DUPLICATE_HEADER", raw_output_path=output_tsv
            )
        if len(fields) != len(_AMRFINDER_COLUMNS):
            return _indeterminate_adapter_result(
                "amr", required=required, reason_code="AMRFINDER_PARSER_SCHEMA_MISMATCH", raw_output_path=output_tsv
            )
        if fields in seen_rows:
            return _indeterminate_adapter_result(
                "amr", required=required, reason_code="AMRFINDER_DUPLICATE_HIT", raw_output_path=output_tsv
            )
        seen_rows.add(fields)
        row = dict(zip(_AMRFINDER_COLUMNS, fields, strict=True))
        record = query_records.get(row["Protein id"])
        if record is None:
            return _indeterminate_adapter_result(
                "amr", required=required, reason_code="AMRFINDER_UNKNOWN_QUERY_ID", raw_output_path=output_tsv
            )
        if row["Contig id"] != record.sequence_id:
            return _indeterminate_adapter_result(
                "amr", required=required, reason_code="AMRFINDER_UNKNOWN_SEQUENCE_ID", raw_output_path=output_tsv
            )
        try:
            start = int(row["Start"])
            stop = int(row["Stop"])
            for column in _AMRFINDER_NUMERIC_COLUMNS:
                number = _finite_number(row[column])
                if number < 0:
                    raise ValueError(f"negative value in {column}")
            if not 0 <= _finite_number(row["% Coverage of reference"]) <= 100:
                raise ValueError("reference coverage outside percent range")
            if not 0 <= _finite_number(row["% Identity to reference"]) <= 100:
                raise ValueError("identity outside percent range")
        except (TypeError, ValueError):
            return _indeterminate_adapter_result(
                "amr", required=required, reason_code="AMRFINDER_INVALID_NUMERIC_VALUE", raw_output_path=output_tsv
            )
        if (start, stop, row["Strand"]) != (record.start, record.end, record.strand):
            return _indeterminate_adapter_result(
                "amr", required=required, reason_code="AMRFINDER_COORDINATE_MISMATCH", raw_output_path=output_tsv
            )
        if row["Scope"] not in {"core", "plus"} or row["Type"] not in {"AMR", "STRESS", "VIRULENCE"}:
            return _indeterminate_adapter_result(
                "amr", required=required, reason_code="AMRFINDER_PARSER_SCHEMA_MISMATCH", raw_output_path=output_tsv
            )
        if row["Type"] == "AMR":
            amr_findings.append(
                _amrfinder_finding(
                    row,
                    record,
                    safety_class="amr",
                    state=SafetyState.FAIL,
                    reason_code="AMR_DETERMINANT_DETECTED",
                    manifest_section=manifest_section,
                )
            )
        elif row["Scope"] == "plus" and row["Type"] == "VIRULENCE":
            supplemental_findings.append(
                _amrfinder_finding(
                    row,
                    record,
                    safety_class="toxin",
                    state=SafetyState.INDETERMINATE,
                    reason_code="AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL",
                    manifest_section=manifest_section,
                )
            )

    state = SafetyState.FAIL if amr_findings else SafetyState.PASS
    reason_codes = ("AMR_DETERMINANT_DETECTED",) if amr_findings else ("AMRFINDER_MEASURED_NO_AMR_HIT",)
    return AdapterResult(
        class_result=_class_result(
            "amr",
            state,
            required=required,
            findings=tuple(amr_findings),
            reason_codes=reason_codes,
        ),
        supplemental_findings=tuple(supplemental_findings),
        raw_output_path=str(output_tsv),
        raw_output_sha256=_sha256_file(output_tsv),
    )


_parse_amrfinder_output_validated = parse_amrfinder_output


@_bind_result_policy(policy_id=_AMRFINDER_POLICY_ID, policy_sha256=_AMRFINDER_POLICY_SHA256)
def parse_amrfinder_output(
    output_tsv: Path,
    *,
    artifacts: ORFArtifacts,
    manifest_section: Mapping[str, object],
    required: bool,
) -> AdapterResult:
    """Parse caller-supplied output conservatively; only :func:`run_amrfinder` may emit measured PASS."""
    parsed = _parse_amrfinder_output_validated(
        output_tsv,
        artifacts=artifacts,
        manifest_section=manifest_section,
        required=required,
    )
    return _without_validated_search_pass(
        parsed,
        safety_class="amr",
        reason_code="AMRFINDER_SEARCH_EVIDENCE_UNVALIDATED",
    )


def _sha256_path(path: Path) -> str:
    """Match the Task-2 stable digest for a file or directory tree."""
    path = Path(path)
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise AssetProvenanceError(f"pinned asset path is missing: {path}")
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        digest.update(str(child.relative_to(path)).encode())
        digest.update(b"\0")
        with child.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validate_amrfinder_manifest_section(section: Mapping[str, object]) -> tuple[ToolPin, Path, str]:
    required_fields = {
        "release",
        "binary_path",
        "binary_sha256",
        "amrfinder_version",
        "database_path",
        "database_version",
        "database_sha256",
    }
    if any(
        field not in section or not isinstance(section[field], str) or not section[field] for field in required_fields
    ):
        raise AssetProvenanceError("AMRFinder manifest section is incomplete")
    if section["release"] != "amrfinder_v4.2.7":
        raise AssetProvenanceError("AMRFinder manifest release is not amrfinder_v4.2.7")
    database_path = Path(str(section["database_path"]))
    observed_database_digest = _sha256_path(database_path)
    if observed_database_digest != str(section["database_sha256"]):
        raise AssetProvenanceError(
            "AMRFinder database digest drift: "
            f"expected {section['database_sha256']}, observed {observed_database_digest}"
        )
    return (
        ToolPin(
            path=Path(str(section["binary_path"])),
            sha256=str(section["binary_sha256"]),
            version=str(section["amrfinder_version"]),
            version_args=("--version",),
        ),
        database_path,
        str(section["database_version"]),
    )


@_bind_result_policy(policy_id=_AMRFINDER_POLICY_ID, policy_sha256=_AMRFINDER_POLICY_SHA256)
def run_amrfinder(
    artifacts: ORFArtifacts,
    *,
    manifest_section: Mapping[str, object],
    work_dir: Path,
    threads: int = 1,
    required: bool = True,
    runner: CommandRunner = subprocess.run,
    timeout: float = 300.0,
) -> AdapterResult:
    """Validate pinned AMRFinder assets, execute combined mode, and parse fail closed."""
    try:
        tool_pin, database_path, database_version = _validate_amrfinder_manifest_section(manifest_section)
        validate_tool_pin(tool_pin, runner=runner, timeout=timeout)
        database_completed = runner(
            [str(tool_pin.path), "--database", str(database_path), "--database_version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if database_completed.stdout.strip() != database_version:
            raise AssetProvenanceError(
                "AMRFinder database version drift: "
                f"expected {database_version!r}, observed {database_completed.stdout.strip()!r}"
            )
    except (AssetProvenanceError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        return _indeterminate_adapter_result(
            "amr", required=required, reason_code="AMRFINDER_ASSET_PROVENANCE_MISMATCH"
        )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_tsv = work_dir / "amrfinder.tsv"
    output_tsv.unlink(missing_ok=True)
    command = tuple(
        build_amrfinder_command(
            amrfinder=tool_pin.path,
            genomes_fna=artifacts.genomes_fna,
            proteins_faa=artifacts.proteins_faa,
            proteins_gff=artifacts.proteins_gff,
            database_dir=database_path,
            threads=threads,
            output_tsv=output_tsv,
        )
    )
    try:
        runner(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _indeterminate_adapter_result(
            "amr",
            required=required,
            reason_code="AMRFINDER_EXECUTION_TIMEOUT",
            command=command,
            raw_output_path=output_tsv,
        )
    except (subprocess.CalledProcessError, OSError):
        return _indeterminate_adapter_result(
            "amr",
            required=required,
            reason_code="AMRFINDER_EXECUTION_FAILED",
            command=command,
            raw_output_path=output_tsv,
        )
    if not output_tsv.is_file():
        return _indeterminate_adapter_result(
            "amr",
            required=required,
            reason_code="AMRFINDER_OUTPUT_MISSING",
            command=command,
            raw_output_path=output_tsv,
        )
    parsed = _parse_amrfinder_output_validated(
        output_tsv,
        artifacts=artifacts,
        manifest_section=manifest_section,
        required=required,
    )
    return AdapterResult(
        class_result=parsed.class_result,
        supplemental_findings=parsed.supplemental_findings,
        command=command,
        raw_output_path=parsed.raw_output_path,
        raw_output_sha256=parsed.raw_output_sha256,
    )


@dataclass(frozen=True)
class HomologyBand:
    """Joint protein-homology thresholds for one evidence band."""

    identity: float
    query_coverage: float
    reference_coverage: float
    evalue: float

    def __post_init__(self) -> None:
        """Reject nonfinite or biologically nonsensical policy bounds."""
        values = (self.identity, self.query_coverage, self.reference_coverage, self.evalue)
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in values):
            raise ValueError("homology thresholds must be finite numbers")
        if not 0 <= self.identity <= 100:
            raise ValueError("homology identity threshold must be in [0, 100]")
        if not 0 <= self.query_coverage <= 100 or not 0 <= self.reference_coverage <= 100:
            raise ValueError("homology coverage thresholds must be in [0, 100]")
        if self.evalue < 0:
            raise ValueError("homology E-value threshold must be nonnegative")

    def matches(self, scores: Mapping[str, float]) -> bool:
        """Return true only when every identity, coverage, and E-value bound is met."""
        return (
            scores["identity"] >= self.identity
            and scores["query_coverage"] >= self.query_coverage
            and scores["reference_coverage"] >= self.reference_coverage
            and scores["evalue"] <= self.evalue
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize the exact band used to classify a finding."""
        return {
            "evalue": self.evalue,
            "identity": self.identity,
            "query_coverage": self.query_coverage,
            "reference_coverage": self.reference_coverage,
        }


@dataclass(frozen=True)
class HomologyPolicy:
    """Versioned high-confidence and material-review homology bands."""

    policy_id: str
    high: HomologyBand
    review: HomologyBand

    def __post_init__(self) -> None:
        """Require high-confidence evidence to be at least as strict as review evidence."""
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.policy_id):
            raise ValueError("homology policy ID is invalid")
        if (
            self.high.identity < self.review.identity
            or self.high.query_coverage < self.review.query_coverage
            or self.high.reference_coverage < self.review.reference_coverage
            or self.high.evalue > self.review.evalue
        ):
            raise ValueError("high-confidence band must be no weaker than the review band")

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete threshold policy in a stable schema."""
        return {
            "policy_id": self.policy_id,
            "high": self.high.to_dict(),
            "review": self.review.to_dict(),
        }

    @property
    def sha256(self) -> str:
        """Return the canonical SHA-256 bound into findings and scan manifests."""
        return _canonical_mapping_sha256(self.to_dict())


TOXIN_HOMOLOGY_POLICY_V1 = HomologyPolicy(
    policy_id="toxin-homology-v1",
    high=HomologyBand(identity=80.0, query_coverage=80.0, reference_coverage=80.0, evalue=1e-10),
    review=HomologyBand(identity=40.0, query_coverage=60.0, reference_coverage=60.0, evalue=1e-5),
)
_DIAMOND_COLUMNS = (
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


def build_diamond_command(
    *,
    diamond: Path,
    queries_faa: Path,
    database: Path,
    output_tsv: Path,
    threads: int,
) -> list[str]:
    """Build a headerless DIAMOND search retaining all hits needed for two-band classification."""
    return [
        str(diamond),
        "blastp",
        "--query",
        str(queries_faa),
        "--db",
        str(database),
        "--out",
        str(output_tsv),
        "--outfmt",
        "6",
        *_DIAMOND_COLUMNS,
        "--threads",
        str(threads),
        "--max-target-seqs",
        "0",
        "--sensitive",
    ]


def _toxin_file_records(section: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    if not isinstance(section.get("uniprot_release"), str) or not section["uniprot_release"]:
        raise AssetProvenanceError("toxin manifest lacks a UniProt release")
    files = section.get("files")
    if not isinstance(files, Mapping):
        raise AssetProvenanceError("toxin manifest lacks file records")
    records: dict[str, Mapping[str, object]] = {}
    for role in ("annotations", "fasta", "diamond_database"):
        record = files.get(role)
        if not isinstance(record, Mapping):
            raise AssetProvenanceError(f"toxin manifest lacks {role} record")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not path or not isinstance(digest, str) or not digest:
            raise AssetProvenanceError(f"toxin manifest {role} path/digest is incomplete")
        records[role] = record
    return records


def _validate_toxin_assets(section: Mapping[str, object]) -> tuple[Path, set[str]]:
    records = _toxin_file_records(section)
    for role, record in records.items():
        path = Path(str(record["path"]))
        if not path.is_file() or _sha256_file(path) != str(record["sha256"]):
            raise AssetProvenanceError(f"toxin {role} path or digest drift")
    accessions = _read_uniprot_accessions(Path(str(records["annotations"]["path"])))
    return Path(str(records["diamond_database"]["path"])), accessions


def _read_uniprot_accessions(path: Path) -> set[str]:
    lines = path.read_text().splitlines()
    if not lines:
        raise AssetProvenanceError("toxin annotations table is empty")
    header = lines[0].split("\t")
    if "Entry" not in header or len(header) != len(set(header)):
        raise AssetProvenanceError("toxin annotations table has no unique Entry column")
    entry_index = header.index("Entry")
    accessions: set[str] = set()
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(header):
            raise AssetProvenanceError("toxin annotations table row does not match its header")
        accession = fields[entry_index].strip()
        if not accession or accession in accessions:
            raise AssetProvenanceError("toxin annotations table has an empty or duplicate accession")
        accessions.add(accession)
    if not accessions:
        raise AssetProvenanceError("toxin annotations table has no accessions")
    return accessions


def _canonical_uniprot_accession(target_id: str) -> str:
    parts = target_id.split("|")
    if len(parts) != 3 or parts[0] != "sp" or not parts[1] or not parts[2]:
        raise ValueError("DIAMOND target is not a canonical reviewed UniProt identifier")
    return parts[1]


def _parse_diamond_scores(row: Mapping[str, str]) -> dict[str, float]:
    scores = {
        "identity": _finite_number(row["pident"]),
        "alignment_length": _finite_number(row["length"]),
        "query_length": _finite_number(row["qlen"]),
        "reference_length": _finite_number(row["slen"]),
        "query_coverage": _finite_number(row["qcovhsp"]),
        "reference_coverage": _finite_number(row["scovhsp"]),
        "evalue": _finite_number(row["evalue"]),
        "bitscore": _finite_number(row["bitscore"]),
    }
    if not 0 <= scores["identity"] <= 100:
        raise ValueError("DIAMOND identity is outside percent range")
    if not 0 <= scores["query_coverage"] <= 100 or not 0 <= scores["reference_coverage"] <= 100:
        raise ValueError("DIAMOND coverage is outside percent range")
    if any(scores[name] <= 0 for name in ("alignment_length", "query_length", "reference_length")):
        raise ValueError("DIAMOND lengths must be positive")
    if scores["evalue"] < 0 or scores["bitscore"] < 0:
        raise ValueError("DIAMOND E-value and bitscore must be nonnegative")
    return scores


def _toxin_finding(
    *,
    record: ORFQueryRecord,
    accession: str,
    scores: Mapping[str, float],
    state: SafetyState,
    reason_code: str,
    band: HomologyBand,
    policy: HomologyPolicy,
    manifest_section: Mapping[str, object],
    tool_pin: ToolPin,
) -> NormalizedSafetyFinding:
    database_record = _toxin_file_records(manifest_section)["diamond_database"]
    return NormalizedSafetyFinding(
        safety_class="toxin",
        state=state,
        reason_codes=(reason_code,),
        finding_id=f"toxin:{record.query_id}:{accession}",
        detector="diamond-reviewed-toxin",
        accession=accession,
        query_id=record.query_id,
        sequence_id=record.sequence_id,
        start=record.start,
        end=record.end,
        strand=record.strand,
        frame=record.frame,
        scores=scores,
        thresholds=band.to_dict(),
        source_path=str(database_record["path"]),
        source_sha256=str(database_record["sha256"]),
        tool_version=tool_pin.version,
        database_version=f"UniProt {manifest_section['uniprot_release']}",
        evidence_path=record.evidence_path,
        evidence_method="diamond-blastp",
        threshold_policy=policy.policy_id,
        threshold_policy_sha256=policy.sha256,
        tool_path=str(tool_pin.path),
        tool_sha256=tool_pin.sha256,
        profile=None,
    )


@_bind_result_policy(
    policy_id=TOXIN_HOMOLOGY_POLICY_V1.policy_id,
    policy_sha256=TOXIN_HOMOLOGY_POLICY_V1.sha256,
    policy_kwarg=True,
)
def parse_toxin_diamond_output(
    output_tsv: Path,
    *,
    artifacts: ORFArtifacts,
    manifest_section: Mapping[str, object],
    tool_pin: ToolPin,
    required: bool,
    policy: HomologyPolicy = TOXIN_HOMOLOGY_POLICY_V1,
) -> AdapterResult:
    """Parse normalized DIAMOND output using joint high and review bands."""
    output_tsv = Path(output_tsv)
    if not output_tsv.is_file():
        return _indeterminate_adapter_result(
            "toxin", required=required, reason_code="TOXIN_DIAMOND_OUTPUT_MISSING", raw_output_path=output_tsv
        )
    if output_tsv.stat().st_size == 0:
        return _indeterminate_adapter_result(
            "toxin", required=required, reason_code="TOXIN_DIAMOND_OUTPUT_EMPTY", raw_output_path=output_tsv
        )
    try:
        output_text = output_tsv.read_text()
    except (OSError, UnicodeError):
        return _indeterminate_adapter_result(
            "toxin",
            required=required,
            reason_code="TOXIN_DIAMOND_PARSER_SCHEMA_MISMATCH",
            raw_output_path=output_tsv,
        )
    noncomment_lines = [line for line in output_text.splitlines() if line.strip() and not line.startswith("#")]
    if not noncomment_lines or tuple(noncomment_lines[0].split("\t")) != _DIAMOND_COLUMNS:
        return _indeterminate_adapter_result(
            "toxin",
            required=required,
            reason_code="TOXIN_DIAMOND_PARSER_SCHEMA_MISMATCH",
            raw_output_path=output_tsv,
        )
    try:
        query_records = _query_record_index(artifacts)
        accessions = _read_uniprot_accessions(Path(str(_toxin_file_records(manifest_section)["annotations"]["path"])))
    except (AssetProvenanceError, KeyError, TypeError, ValueError):
        return _indeterminate_adapter_result(
            "toxin",
            required=required,
            reason_code="TOXIN_ASSET_PROVENANCE_MISMATCH",
            raw_output_path=output_tsv,
        )

    high_findings: list[NormalizedSafetyFinding] = []
    review_findings: list[NormalizedSafetyFinding] = []
    seen_hits: set[tuple[str, str]] = set()
    for line in noncomment_lines[1:]:
        fields = tuple(line.split("\t"))
        if fields == _DIAMOND_COLUMNS:
            return _indeterminate_adapter_result(
                "toxin",
                required=required,
                reason_code="TOXIN_DIAMOND_DUPLICATE_HEADER",
                raw_output_path=output_tsv,
            )
        if len(fields) != len(_DIAMOND_COLUMNS):
            return _indeterminate_adapter_result(
                "toxin",
                required=required,
                reason_code="TOXIN_DIAMOND_PARSER_SCHEMA_MISMATCH",
                raw_output_path=output_tsv,
            )
        hit_key = (fields[0], fields[1])
        if hit_key in seen_hits:
            return _indeterminate_adapter_result(
                "toxin",
                required=required,
                reason_code="TOXIN_DIAMOND_DUPLICATE_HIT",
                raw_output_path=output_tsv,
            )
        seen_hits.add(hit_key)
        row = dict(zip(_DIAMOND_COLUMNS, fields, strict=True))
        record = query_records.get(row["qseqid"])
        if record is None:
            return _indeterminate_adapter_result(
                "toxin",
                required=required,
                reason_code="TOXIN_DIAMOND_UNKNOWN_QUERY_ID",
                raw_output_path=output_tsv,
            )
        try:
            accession = _canonical_uniprot_accession(row["sseqid"])
        except ValueError:
            return _indeterminate_adapter_result(
                "toxin",
                required=required,
                reason_code="TOXIN_DIAMOND_MALFORMED_TARGET_ID",
                raw_output_path=output_tsv,
            )
        if accession not in accessions:
            return _indeterminate_adapter_result(
                "toxin",
                required=required,
                reason_code="TOXIN_DIAMOND_UNKNOWN_ACCESSION",
                raw_output_path=output_tsv,
            )
        try:
            scores = _parse_diamond_scores(row)
        except (TypeError, ValueError):
            return _indeterminate_adapter_result(
                "toxin",
                required=required,
                reason_code="TOXIN_DIAMOND_INVALID_NUMERIC_VALUE",
                raw_output_path=output_tsv,
            )
        if policy.high.matches(scores):
            high_findings.append(
                _toxin_finding(
                    record=record,
                    accession=accession,
                    scores=scores,
                    state=SafetyState.FAIL,
                    reason_code="TOXIN_HIGH_CONFIDENCE_HOMOLOGY",
                    band=policy.high,
                    policy=policy,
                    manifest_section=manifest_section,
                    tool_pin=tool_pin,
                )
            )
        elif policy.review.matches(scores):
            review_findings.append(
                _toxin_finding(
                    record=record,
                    accession=accession,
                    scores=scores,
                    state=SafetyState.INDETERMINATE,
                    reason_code="TOXIN_REVIEW_HOMOLOGY",
                    band=policy.review,
                    policy=policy,
                    manifest_section=manifest_section,
                    tool_pin=tool_pin,
                )
            )

    if high_findings:
        state = SafetyState.FAIL
        reason_codes = ("TOXIN_HIGH_CONFIDENCE_HOMOLOGY",)
    elif review_findings:
        state = SafetyState.INDETERMINATE
        reason_codes = ("TOXIN_REVIEW_HOMOLOGY",)
    else:
        state = SafetyState.PASS
        reason_codes = ("TOXIN_DIAMOND_MEASURED_NO_REVIEW_HIT",)
    findings = (*high_findings, *review_findings)
    return AdapterResult(
        class_result=_class_result("toxin", state, required=required, findings=findings, reason_codes=reason_codes),
        raw_output_path=str(output_tsv),
        raw_output_sha256=_sha256_file(output_tsv),
    )


_parse_toxin_diamond_output_validated = parse_toxin_diamond_output


@_bind_result_policy(
    policy_id=TOXIN_HOMOLOGY_POLICY_V1.policy_id,
    policy_sha256=TOXIN_HOMOLOGY_POLICY_V1.sha256,
    policy_kwarg=True,
)
def parse_toxin_diamond_output(
    output_tsv: Path,
    *,
    artifacts: ORFArtifacts,
    manifest_section: Mapping[str, object],
    tool_pin: ToolPin,
    required: bool,
    policy: HomologyPolicy = TOXIN_HOMOLOGY_POLICY_V1,
) -> AdapterResult:
    """Parse caller-supplied output conservatively; only :func:`run_toxin_diamond` may emit measured PASS."""
    parsed = _parse_toxin_diamond_output_validated(
        output_tsv,
        artifacts=artifacts,
        manifest_section=manifest_section,
        tool_pin=tool_pin,
        required=required,
        policy=policy,
    )
    return _without_validated_search_pass(
        parsed,
        safety_class="toxin",
        reason_code="TOXIN_DIAMOND_SEARCH_EVIDENCE_UNVALIDATED",
    )


def _write_normalized_header(output_path: Path, columns: tuple[str, ...], raw_path: Path) -> None:
    raw_text = raw_path.read_text()
    if raw_text and not raw_text.endswith("\n"):
        raw_text += "\n"
    output_path.write_text("\t".join(columns) + "\n" + raw_text)


@_bind_result_policy(
    policy_id=TOXIN_HOMOLOGY_POLICY_V1.policy_id,
    policy_sha256=TOXIN_HOMOLOGY_POLICY_V1.sha256,
    policy_kwarg=True,
)
def run_toxin_diamond(
    artifacts: ORFArtifacts,
    *,
    manifest_section: Mapping[str, object],
    tool_pin: ToolPin,
    work_dir: Path,
    threads: int = 1,
    required: bool = True,
    runner: CommandRunner = subprocess.run,
    timeout: float = 300.0,
    policy: HomologyPolicy = TOXIN_HOMOLOGY_POLICY_V1,
) -> AdapterResult:
    """Validate toxin assets, search primary/fallback proteins, and parse two evidence bands."""
    try:
        database_path, _ = _validate_toxin_assets(manifest_section)
        validate_tool_pin(tool_pin, runner=runner, timeout=timeout)
    except (AssetProvenanceError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        return _indeterminate_adapter_result("toxin", required=required, reason_code="TOXIN_ASSET_PROVENANCE_MISMATCH")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_output = work_dir / "toxin_diamond.raw.tsv"
    normalized_output = work_dir / "toxin_diamond.tsv"
    raw_output.unlink(missing_ok=True)
    normalized_output.unlink(missing_ok=True)
    command = tuple(
        build_diamond_command(
            diamond=tool_pin.path,
            queries_faa=artifacts.all_queries_faa,
            database=database_path,
            output_tsv=raw_output,
            threads=threads,
        )
    )
    try:
        runner(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _indeterminate_adapter_result(
            "toxin",
            required=required,
            reason_code="TOXIN_DIAMOND_EXECUTION_TIMEOUT",
            command=command,
            raw_output_path=raw_output,
        )
    except (subprocess.CalledProcessError, OSError):
        return _indeterminate_adapter_result(
            "toxin",
            required=required,
            reason_code="TOXIN_DIAMOND_EXECUTION_FAILED",
            command=command,
            raw_output_path=raw_output,
        )
    if not raw_output.is_file():
        return _indeterminate_adapter_result(
            "toxin",
            required=required,
            reason_code="TOXIN_DIAMOND_OUTPUT_MISSING",
            command=command,
            raw_output_path=raw_output,
        )
    try:
        _write_normalized_header(normalized_output, _DIAMOND_COLUMNS, raw_output)
    except (OSError, UnicodeError):
        return _indeterminate_adapter_result(
            "toxin",
            required=required,
            reason_code="TOXIN_DIAMOND_PARSER_SCHEMA_MISMATCH",
            command=command,
            raw_output_path=raw_output,
        )
    parsed = _parse_toxin_diamond_output_validated(
        normalized_output,
        artifacts=artifacts,
        manifest_section=manifest_section,
        tool_pin=tool_pin,
        required=required,
        policy=policy,
    )
    return AdapterResult(
        class_result=parsed.class_result,
        command=command,
        raw_output_path=parsed.raw_output_path,
        raw_output_sha256=parsed.raw_output_sha256,
    )


PHROGS_HOMOLOGY_POLICY_V1 = HomologyPolicy(
    policy_id="phrogs-homology-v1",
    high=HomologyBand(identity=30.0, query_coverage=0.70, reference_coverage=0.70, evalue=1e-10),
    review=HomologyBand(identity=20.0, query_coverage=0.50, reference_coverage=0.50, evalue=1e-5),
)
_PHROGS_COLUMNS = (
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
_PHROGS_LOOKUP_COLUMNS = ("phrog", "annot", "category", "confidence", "matched_term")
_PHROGS_SEARCH_ORIENTATION = "phrog_profile_query_vs_orf_target"
_PHROGS_SEARCH_PROFILE_SCOPE = "full_phrogs_v4_profile_database"
_PHROGS_LOOKUP_JOIN_POLICY = "classify_only_profile_ids_present_in_pinned_lookup"
_PHROGS_QUERY_ID_PATTERN = r"^phrog_[1-9][0-9]*$"
_PHROGS_UNITS = {"pident": "percent", "qcov": "fraction", "tcov": "fraction"}
_PHROGS_PROFILE_DATABASE_NAME = "phrogs_profile_db"
_PHROGS_PROFILE_RELEASE_MARKER = "VERSION_1_8_0"
_PHROGS_PROFILE_ROLE = "complete PHROGs v4 MMseqs profile database for identity-bearing lysogeny search"
_PHROGS_PROFILE_SOURCE_URL = "https://zenodo.org/record/17110353/files/pharokka_v1.8.0_databases.tar.gz"
_PHROGS_PROFILE_ARCHIVE_MD5 = "a63c485241b900a11989bd1821bfbb09"
_PHROGS_PROFILE_ARCHIVE_SIZE = 656_171_247
_PHROGS_PROFILE_RELEASE = "Pharokka database v1.8.0"
_PHROGS_DATASET_RELEASE = "PHROGs v4"
_PHROGS_PROFILE_DOI = "10.5281/zenodo.17110353"
_PHROGS_PROFILE_LICENSE = "CC BY 4.0"
_PHROGS_PROFILE_CITATION = "Pharokka database v1.8.0 (DOI: 10.5281/zenodo.17110353)."
_PHROGS_MINIMUM_MMSEQS_VERSION = "14"
_PHROGS_BUILT_WITH_MMSEQS_VERSION = "18.8cc5c"
_PHROGS_ANNOTATION_URL = DEFAULT_PHROGS_ANNOTATION_URL
_PHROGS_ANNOTATION_SHA256 = DEFAULT_PHROGS_ANNOTATION_SHA256
_PHROGS_V4_SAFETY_LOOKUP_COUNTS = MappingProxyType(
    {
        "total": 109,
        "high_confidence": 57,
        "review": 52,
    }
)


class VerifiedIdentityMappingMissingError(AssetProvenanceError):
    """The PHROGs asset cannot map search identifiers to reviewed profile metadata."""


def build_phrogs_command(
    *,
    mmseqs: Path,
    profile_database: Path,
    proteins_faa: Path,
    output_tsv: Path,
    temporary_dir: Path,
    threads: int,
) -> list[str]:
    """Build the identity-preserving PHROG-profile-query versus predicted-ORF-target search."""
    return [
        str(mmseqs),
        "easy-search",
        str(profile_database),
        str(proteins_faa),
        str(output_tsv),
        str(temporary_dir),
        "--threads",
        str(threads),
        "--format-output",
        ",".join(_PHROGS_COLUMNS),
    ]


def _digest_file_inventory(files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    names: set[str] = set()
    for path in sorted(files, key=lambda candidate: candidate.name):
        if path.name in names:
            raise AssetProvenanceError(f"duplicate asset basename in file inventory: {path.name}")
        names.add(path.name)
        if not path.is_file() or path.stat().st_size == 0:
            raise AssetProvenanceError(f"asset inventory path is missing or empty: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _digest_relative_file_inventory(root: Path, files: tuple[Path, ...]) -> str:
    """Match Task 2's digest for the selected profile tree, excluding bundled databases."""
    digest = hashlib.sha256()
    for path in sorted(files):
        if not path.is_file() or path.stat().st_size == 0:
            raise AssetProvenanceError(f"profile tree path is missing or empty: {path}")
        try:
            relative_path = path.relative_to(root)
        except ValueError as error:
            raise AssetProvenanceError(f"profile tree path escapes its root: {path}") from error
        digest.update(str(relative_path).encode())
        digest.update(b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _read_phrogs_profile_ids(profile_path: Path) -> frozenset[str]:
    """Read the MMseqs lookup that proves each runtime profile's canonical PHROG identity."""
    lookup_path = Path(f"{profile_path}.lookup")
    profile_ids: set[str] = set()
    internal_keys: set[str] = set()
    try:
        lines = lookup_path.read_text().splitlines()
    except (OSError, UnicodeError) as error:
        raise VerifiedIdentityMappingMissingError("PHROGs profile lookup is unreadable") from error
    if not lines:
        raise VerifiedIdentityMappingMissingError("PHROGs profile lookup is empty")
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0].isdecimal() or not fields[2].isdecimal():
            raise VerifiedIdentityMappingMissingError(
                f"PHROGs profile lookup row {line_number} does not have its exact identity schema"
            )
        internal_key, profile_id, _file_index = fields
        if (
            not re.fullmatch(_PHROGS_QUERY_ID_PATTERN, profile_id)
            or internal_key in internal_keys
            or profile_id in profile_ids
        ):
            raise VerifiedIdentityMappingMissingError(
                f"PHROGs profile lookup row {line_number} has an invalid or duplicate identity"
            )
        internal_keys.add(internal_key)
        profile_ids.add(profile_id)
    return frozenset(profile_ids)


def _phrogs_profile_id_inventory(profile_ids: frozenset[str]) -> dict[str, int | str]:
    digest = hashlib.sha256()
    for profile_id in sorted(profile_ids):
        digest.update(profile_id.encode())
        digest.update(b"\n")
    return {"count": len(profile_ids), "sha256": digest.hexdigest()}


def _read_phrogs_annotation_lookup(path: Path) -> dict[str, Mapping[str, str]]:
    """Regenerate the safety lookup exactly from the digest-pinned PHROGs v4 annotation snapshot."""
    profiles: dict[str, Mapping[str, str]] = {}
    try:
        with path.open(newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            required_columns = {"phrog", "color", "annot", "category"}
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise AssetProvenanceError("PHROGs annotation source schema mismatch")
            for source_row in reader:
                category = source_row["category"].strip()
                if category.casefold() != PHROGS_INTEGRATION_EXCISION_CATEGORY.casefold():
                    continue
                profile = source_row["phrog"].strip()
                annotation = source_row["annot"].strip()
                normalized_annotation = annotation.casefold()
                high_term = next(
                    (term for term in PHROGS_HIGH_CONFIDENCE_TERMS if term in normalized_annotation),
                    None,
                )
                review_term = next(
                    (term for term in PHROGS_REVIEW_TERMS if term in normalized_annotation),
                    None,
                )
                confidence = "high_confidence" if high_term is not None else "review"
                matched_term = high_term or review_term or "integration and excision category"
                if not re.fullmatch(_PHROGS_QUERY_ID_PATTERN, profile) or profile in profiles or not annotation:
                    raise AssetProvenanceError("PHROGs annotation source contains invalid safety metadata")
                profiles[profile] = MappingProxyType(
                    {
                        "phrog": profile,
                        "annot": annotation,
                        "category": category,
                        "confidence": confidence,
                        "matched_term": matched_term,
                    }
                )
    except (OSError, UnicodeError) as error:
        raise AssetProvenanceError("PHROGs annotation source is unreadable") from error
    if not profiles:
        raise AssetProvenanceError("PHROGs annotation source has no integration/excision profiles")
    return profiles


def _read_phrogs_lookup(path: Path) -> dict[str, Mapping[str, str]]:
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError) as error:
        raise AssetProvenanceError("PHROGs safety lookup is unreadable") from error
    if not lines or tuple(lines[0].split("\t")) != _PHROGS_LOOKUP_COLUMNS:
        raise AssetProvenanceError("PHROGs lookup schema mismatch")
    profiles: dict[str, Mapping[str, str]] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(_PHROGS_LOOKUP_COLUMNS):
            raise AssetProvenanceError("PHROGs lookup row does not match its header")
        row = dict(zip(_PHROGS_LOOKUP_COLUMNS, fields, strict=True))
        profile = row["phrog"]
        if (
            not re.fullmatch(_PHROGS_QUERY_ID_PATTERN, profile)
            or profile in profiles
            or row["category"] != "integration and excision"
            or row["confidence"] not in {"high_confidence", "review"}
            or not row["annot"]
            or not row["matched_term"]
        ):
            raise AssetProvenanceError("PHROGs lookup contains invalid or duplicate profile metadata")
        profiles[profile] = MappingProxyType(row)
    if not profiles:
        raise AssetProvenanceError("PHROGs lookup is empty")
    return profiles


def _validate_phrogs_assets(
    section: Mapping[str, object],
) -> tuple[
    Path,
    str,
    dict[str, Mapping[str, str]],
    frozenset[str],
    str,
    int,
]:
    profile_record = section.get("profile_database")
    if not isinstance(profile_record, Mapping):
        raise VerifiedIdentityMappingMissingError("PHROGs manifest lacks a verified identity-bearing profile database")
    expected_contract = {
        "search_orientation": _PHROGS_SEARCH_ORIENTATION,
        "search_profile_scope": _PHROGS_SEARCH_PROFILE_SCOPE,
        "lookup_join_policy": _PHROGS_LOOKUP_JOIN_POLICY,
        "output_fields": list(_PHROGS_COLUMNS),
        "units": _PHROGS_UNITS,
        "query_id_pattern": _PHROGS_QUERY_ID_PATTERN,
        "query_ids_join_lookup": True,
    }
    for field, expected in expected_contract.items():
        if profile_record.get(field) != expected:
            raise VerifiedIdentityMappingMissingError(f"PHROGs profile identity contract mismatch for {field}")
    inventory_record = profile_record.get("profile_id_inventory")
    if (
        not isinstance(inventory_record, Mapping)
        or frozenset(inventory_record) != {"count", "sha256"}
        or type(inventory_record.get("count")) is not int
        or int(inventory_record["count"]) <= 0
        or not isinstance(inventory_record.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(inventory_record["sha256"]))
    ):
        raise VerifiedIdentityMappingMissingError("PHROGs manifest lacks a valid profile ID inventory")
    expected_profile_record_keys = frozenset(
        {
            "path",
            "role",
            "sha256",
            "files",
            "extracted_tree",
            "search_orientation",
            "search_profile_scope",
            "lookup_join_policy",
            "output_fields",
            "units",
            "query_id_pattern",
            "query_ids_join_lookup",
            "profile_id_inventory",
            "provenance",
        }
    )
    if frozenset(profile_record) != expected_profile_record_keys:
        raise AssetProvenanceError("PHROGs profile database manifest keys do not match its schema")
    if profile_record.get("role") != _PHROGS_PROFILE_ROLE:
        raise AssetProvenanceError("PHROGs profile database role does not match its pinned purpose")
    path_value = profile_record.get("path")
    digest_value = profile_record.get("sha256")
    files_value = profile_record.get("files")
    if (
        not isinstance(path_value, str)
        or not path_value
        or not isinstance(digest_value, str)
        or not digest_value
        or not isinstance(files_value, list)
        or not files_value
        or not all(isinstance(value, str) and value for value in files_value)
    ):
        raise AssetProvenanceError("PHROGs profile database path/digest/inventory is incomplete")
    profile_path = Path(path_value)
    if profile_path.name != _PHROGS_PROFILE_DATABASE_NAME:
        raise AssetProvenanceError("PHROGs profile database prefix name is not pinned")
    required_paths = (
        profile_path,
        Path(f"{profile_path}.dbtype"),
        Path(f"{profile_path}.index"),
        Path(f"{profile_path}.lookup"),
        Path(f"{profile_path}.source"),
        Path(f"{profile_path}_h"),
        Path(f"{profile_path}_h.dbtype"),
        Path(f"{profile_path}_h.index"),
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_paths):
        raise AssetProvenanceError("PHROGs profile database is missing a required nonempty sidecar")
    observed_files = tuple(
        sorted(path.resolve() for path in profile_path.parent.glob(f"{profile_path.name}*") if path.is_file())
    )
    recorded_files = [str(Path(value).resolve()) for value in files_value]
    if recorded_files != [str(path) for path in observed_files]:
        raise AssetProvenanceError("PHROGs profile database file inventory drift")
    observed_digest = _digest_file_inventory(observed_files)
    if observed_digest != digest_value:
        raise AssetProvenanceError(
            f"PHROGs profile database digest drift: expected {digest_value}, observed {observed_digest}"
        )

    release_marker = profile_path.parent / _PHROGS_PROFILE_RELEASE_MARKER
    if not release_marker.is_file() or release_marker.stat().st_size == 0:
        raise AssetProvenanceError("PHROGs profile release marker is missing or empty")
    extracted_tree = profile_record.get("extracted_tree")
    if not isinstance(extracted_tree, Mapping) or frozenset(extracted_tree) != {
        "path",
        "sha256",
        "files",
    }:
        raise AssetProvenanceError("PHROGs profile extracted-tree record is invalid")
    profile_root = profile_path.parent.resolve()
    expected_tree_files = tuple(sorted((*observed_files, release_marker.resolve())))
    if extracted_tree.get("path") != str(profile_root) or extracted_tree.get("files") != [
        str(path) for path in expected_tree_files
    ]:
        raise AssetProvenanceError("PHROGs profile extracted-tree path or file inventory drift")
    observed_tree_digest = _digest_relative_file_inventory(profile_root, expected_tree_files)
    if extracted_tree.get("sha256") != observed_tree_digest:
        raise AssetProvenanceError("PHROGs profile extracted-tree digest drift")

    profile_ids = _read_phrogs_profile_ids(profile_path)
    if dict(inventory_record) != _phrogs_profile_id_inventory(profile_ids):
        raise AssetProvenanceError("PHROGs profile ID inventory drift")

    provenance = profile_record.get("provenance")
    expected_provenance_keys = frozenset(
        {
            "source_url",
            "archive_observed_sha256",
            "archive_published_sha256",
            "archive_published_md5",
            "archive_published_size",
            "retrieved_at",
            "release",
            "dataset_release",
            "doi",
            "license",
            "citation",
            "minimum_mmseqs_version",
            "built_with_mmseqs_version",
            "verified_archive",
        }
    )
    if not isinstance(provenance, Mapping) or frozenset(provenance) != expected_provenance_keys:
        raise AssetProvenanceError("PHROGs profile provenance keys do not match the pinned schema")
    expected_provenance = {
        "source_url": _PHROGS_PROFILE_SOURCE_URL,
        "archive_published_sha256": None,
        "archive_published_md5": _PHROGS_PROFILE_ARCHIVE_MD5,
        "archive_published_size": _PHROGS_PROFILE_ARCHIVE_SIZE,
        "release": _PHROGS_PROFILE_RELEASE,
        "dataset_release": _PHROGS_DATASET_RELEASE,
        "doi": _PHROGS_PROFILE_DOI,
        "license": _PHROGS_PROFILE_LICENSE,
        "citation": _PHROGS_PROFILE_CITATION,
        "minimum_mmseqs_version": _PHROGS_MINIMUM_MMSEQS_VERSION,
        "built_with_mmseqs_version": _PHROGS_BUILT_WITH_MMSEQS_VERSION,
    }
    if any(provenance.get(field) != expected for field, expected in expected_provenance.items()):
        raise AssetProvenanceError("PHROGs profile provenance does not match the pinned Pharokka release")
    if (
        not isinstance(provenance.get("archive_observed_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(provenance["archive_observed_sha256"]))
        or not isinstance(provenance.get("retrieved_at"), str)
        or not provenance["retrieved_at"]
    ):
        raise AssetProvenanceError("PHROGs profile retrieval provenance is invalid")
    verified_archive = provenance.get("verified_archive")
    if not isinstance(verified_archive, Mapping) or frozenset(verified_archive) != {"path", "sha256"}:
        raise AssetProvenanceError("PHROGs verified profile archive provenance is incomplete")
    archive_path_value = verified_archive.get("path")
    archive_sha256 = verified_archive.get("sha256")
    if (
        not isinstance(archive_path_value, str)
        or not isinstance(archive_sha256, str)
        or archive_sha256 != provenance["archive_observed_sha256"]
    ):
        raise AssetProvenanceError("PHROGs verified profile archive digest does not match observed provenance")
    archive_path = Path(archive_path_value)
    if (
        not archive_path.is_absolute()
        or archive_path.parent.name != "phrogs_safety_profile_archives"
        or archive_path.parent.parent.name != "downloads"
        or archive_path.name != f"{archive_sha256}.tar.gz"
        or not archive_path.is_file()
        or _sha256_file(archive_path) != archive_sha256
    ):
        raise AssetProvenanceError("PHROGs verified profile archive path or digest drift")

    if (
        section.get("annotation_url") != _PHROGS_ANNOTATION_URL
        or section.get("annotation_sha256") != _PHROGS_ANNOTATION_SHA256
        or section.get("source_sha256") != _PHROGS_ANNOTATION_SHA256
        or section.get("category") != PHROGS_INTEGRATION_EXCISION_CATEGORY
        or section.get("high_confidence_terms") != list(PHROGS_HIGH_CONFIDENCE_TERMS)
        or section.get("review_terms") != list(PHROGS_REVIEW_TERMS)
    ):
        raise AssetProvenanceError("PHROGs annotation provenance does not match the pinned v4 source")
    source_path_value = section.get("source_path")
    if not isinstance(source_path_value, str) or not source_path_value:
        raise AssetProvenanceError("PHROGs annotation source path is missing")
    source_path = Path(source_path_value)
    if not source_path.is_file() or _sha256_file(source_path) != _PHROGS_ANNOTATION_SHA256:
        raise AssetProvenanceError("PHROGs annotation source path or digest drift")
    expected_profiles = _read_phrogs_annotation_lookup(source_path)

    lookup_counts = section.get("lookup_counts")
    if not isinstance(lookup_counts, Mapping) or dict(lookup_counts) != dict(_PHROGS_V4_SAFETY_LOOKUP_COUNTS):
        raise AssetProvenanceError(
            "PHROGs safety lookup cardinality manifest does not match the pinned PHROGs v4 policy"
        )

    lookup_path_value = section.get("lookup_path")
    lookup_digest = section.get("lookup_sha256")
    if not isinstance(lookup_path_value, str) or not isinstance(lookup_digest, str):
        raise AssetProvenanceError("PHROGs lookup path/digest is incomplete")
    lookup_path = Path(lookup_path_value)
    if not lookup_path.is_file() or _sha256_file(lookup_path) != lookup_digest:
        raise AssetProvenanceError("PHROGs lookup path or digest drift")
    profiles = _read_phrogs_lookup(lookup_path)
    if not set(profiles).issubset(profile_ids):
        raise VerifiedIdentityMappingMissingError(
            "PHROGs safety lookup contains profiles absent from the pinned full profile database"
        )
    if profiles != expected_profiles:
        raise AssetProvenanceError("PHROGs safety lookup does not match the pinned annotation source")
    observed_lookup_counts = {
        "total": len(profiles),
        "high_confidence": sum(row["confidence"] == "high_confidence" for row in profiles.values()),
        "review": sum(row["confidence"] == "review" for row in profiles.values()),
    }
    if observed_lookup_counts != dict(_PHROGS_V4_SAFETY_LOOKUP_COUNTS):
        raise AssetProvenanceError("PHROGs safety lookup rows violate the pinned PHROGs v4 cardinality policy")
    database_version = f"{_PHROGS_DATASET_RELEASE} / {_PHROGS_PROFILE_RELEASE}"
    return (
        profile_path,
        str(digest_value),
        profiles,
        profile_ids,
        database_version,
        int(_PHROGS_MINIMUM_MMSEQS_VERSION),
    )


def _mmseqs_major(version: str) -> int:
    candidates = re.findall(r"(?<![A-Za-z0-9])(\d+)(?=[.-][A-Za-z0-9])", version)
    if len(candidates) != 1:
        raise AssetProvenanceError(f"cannot parse one MMseqs major version from {version!r}")
    return int(candidates[0])


def _validate_mmseqs_compatibility(*, expected_version: str, observed_version: str, minimum_major: int) -> None:
    expected_major = _mmseqs_major(expected_version)
    observed_major = _mmseqs_major(observed_version)
    if expected_major != observed_major or expected_major < minimum_major:
        raise AssetProvenanceError(
            f"MMseqs major {observed_major} is incompatible with required major >= {minimum_major}"
        )


def _parse_phrogs_scores(row: Mapping[str, str]) -> dict[str, float]:
    scores = {
        "identity": _finite_number(row["pident"]),
        "alignment_length": _finite_number(row["alnlen"]),
        "query_length": _finite_number(row["qlen"]),
        "reference_length": _finite_number(row["tlen"]),
        "query_coverage": _finite_number(row["qcov"]),
        "reference_coverage": _finite_number(row["tcov"]),
        "evalue": _finite_number(row["evalue"]),
        "bitscore": _finite_number(row["bits"]),
    }
    if not 0 <= scores["identity"] <= 100:
        raise ValueError("PHROGs identity is outside percent range")
    if not 0 <= scores["query_coverage"] <= 1 or not 0 <= scores["reference_coverage"] <= 1:
        raise ValueError("PHROGs coverage is outside fraction range")
    if any(scores[name] <= 0 for name in ("alignment_length", "query_length", "reference_length")):
        raise ValueError("PHROGs lengths must be positive")
    if scores["evalue"] < 0 or scores["bitscore"] < 0:
        raise ValueError("PHROGs E-value and bitscore must be nonnegative")
    return scores


def _lysogeny_required(host_domain: HostDomain, *, strict_lysis: bool) -> bool:
    if host_domain in {HostDomain.BACTERIA, HostDomain.BACTERIA_AND_ARCHAEA}:
        return True
    if host_domain is HostDomain.ARCHAEA:
        return strict_lysis
    raise ValueError(f"unsupported replication host for lysogeny profile: {host_domain}")


def _phrogs_finding(
    *,
    profile: str,
    record: ORFQueryRecord,
    scores: Mapping[str, float],
    state: SafetyState,
    reason_code: str,
    band: HomologyBand,
    policy: HomologyPolicy,
    profile_path: Path,
    profile_digest: str,
    database_version: str,
    tool_pin: ToolPin,
) -> NormalizedSafetyFinding:
    return NormalizedSafetyFinding(
        safety_class="lysogeny",
        state=state,
        reason_codes=(reason_code,),
        finding_id=f"lysogeny:{record.query_id}:{profile}",
        detector="mmseqs-phrogs-v4",
        accession=profile,
        query_id=record.query_id,
        sequence_id=record.sequence_id,
        start=record.start,
        end=record.end,
        strand=record.strand,
        frame=record.frame,
        scores=scores,
        thresholds=band.to_dict(),
        source_path=str(profile_path),
        source_sha256=profile_digest,
        tool_version=tool_pin.version,
        database_version=database_version,
        evidence_path=record.evidence_path,
        evidence_method="mmseqs-profile-search",
        threshold_policy=policy.policy_id,
        threshold_policy_sha256=policy.sha256,
        tool_path=str(tool_pin.path),
        tool_sha256=tool_pin.sha256,
        profile=profile,
    )


@_bind_result_policy(
    policy_id=PHROGS_HOMOLOGY_POLICY_V1.policy_id,
    policy_sha256=PHROGS_HOMOLOGY_POLICY_V1.sha256,
    policy_kwarg=True,
)
def parse_phrogs_output(
    output_tsv: Path,
    *,
    artifacts: ORFArtifacts,
    manifest_section: Mapping[str, object],
    tool_pin: ToolPin,
    host_domain: HostDomain,
    strict_lysis: bool = False,
    policy: HomologyPolicy = PHROGS_HOMOLOGY_POLICY_V1,
) -> AdapterResult:
    """Parse verified PHROGs profile hits with bacterial/archaeal applicability semantics."""
    try:
        required = _lysogeny_required(host_domain, strict_lysis=strict_lysis)
    except ValueError:
        return _indeterminate_adapter_result("lysogeny", required=True, reason_code="PHROGS_UNSUPPORTED_HOST_PROFILE")
    try:
        profile_path, profile_digest, profiles, profile_ids, database_version, _ = _validate_phrogs_assets(
            manifest_section
        )
    except VerifiedIdentityMappingMissingError:
        return _indeterminate_adapter_result(
            "lysogeny", required=required, reason_code="PHROGS_VERIFIED_IDENTITY_MAPPING_MISSING"
        )
    except (AssetProvenanceError, KeyError, TypeError, ValueError):
        return _indeterminate_adapter_result(
            "lysogeny", required=required, reason_code="PHROGS_ASSET_PROVENANCE_MISMATCH"
        )
    output_tsv = Path(output_tsv)
    if not output_tsv.is_file():
        return _indeterminate_adapter_result(
            "lysogeny", required=required, reason_code="PHROGS_OUTPUT_MISSING", raw_output_path=output_tsv
        )
    if output_tsv.stat().st_size == 0:
        return _indeterminate_adapter_result(
            "lysogeny", required=required, reason_code="PHROGS_OUTPUT_EMPTY", raw_output_path=output_tsv
        )
    try:
        output_text = output_tsv.read_text()
    except (OSError, UnicodeError):
        return _indeterminate_adapter_result(
            "lysogeny",
            required=required,
            reason_code="PHROGS_PARSER_SCHEMA_MISMATCH",
            raw_output_path=output_tsv,
        )
    noncomment_lines = [line for line in output_text.splitlines() if line.strip() and not line.startswith("#")]
    if not noncomment_lines or tuple(noncomment_lines[0].split("\t")) != _PHROGS_COLUMNS:
        return _indeterminate_adapter_result(
            "lysogeny",
            required=required,
            reason_code="PHROGS_PARSER_SCHEMA_MISMATCH",
            raw_output_path=output_tsv,
        )
    try:
        query_records = _query_record_index(artifacts, primary_only=True)
    except ValueError:
        return _indeterminate_adapter_result(
            "lysogeny", required=required, reason_code="PHROGS_DUPLICATE_QUERY_ID", raw_output_path=output_tsv
        )

    high_findings: list[NormalizedSafetyFinding] = []
    review_findings: list[NormalizedSafetyFinding] = []
    seen_hits: set[tuple[str, str]] = set()
    for line in noncomment_lines[1:]:
        fields = tuple(line.split("\t"))
        if fields == _PHROGS_COLUMNS:
            return _indeterminate_adapter_result(
                "lysogeny", required=required, reason_code="PHROGS_DUPLICATE_HEADER", raw_output_path=output_tsv
            )
        if len(fields) != len(_PHROGS_COLUMNS):
            return _indeterminate_adapter_result(
                "lysogeny",
                required=required,
                reason_code="PHROGS_PARSER_SCHEMA_MISMATCH",
                raw_output_path=output_tsv,
            )
        hit_key = (fields[0], fields[1])
        if hit_key in seen_hits:
            return _indeterminate_adapter_result(
                "lysogeny", required=required, reason_code="PHROGS_DUPLICATE_HIT", raw_output_path=output_tsv
            )
        seen_hits.add(hit_key)
        row = dict(zip(_PHROGS_COLUMNS, fields, strict=True))
        profile = row["query"]
        if not re.fullmatch(_PHROGS_QUERY_ID_PATTERN, profile) or profile not in profile_ids:
            return _indeterminate_adapter_result(
                "lysogeny", required=required, reason_code="PHROGS_UNKNOWN_PROFILE_ID", raw_output_path=output_tsv
            )
        record = query_records.get(row["target"])
        if record is None:
            return _indeterminate_adapter_result(
                "lysogeny", required=required, reason_code="PHROGS_UNKNOWN_QUERY_ID", raw_output_path=output_tsv
            )
        try:
            scores = _parse_phrogs_scores(row)
        except (TypeError, ValueError):
            return _indeterminate_adapter_result(
                "lysogeny", required=required, reason_code="PHROGS_INVALID_NUMERIC_VALUE", raw_output_path=output_tsv
            )
        metadata = profiles.get(profile)
        if metadata is None:
            continue
        if metadata["confidence"] == "high_confidence" and policy.high.matches(scores):
            high_findings.append(
                _phrogs_finding(
                    profile=profile,
                    record=record,
                    scores=scores,
                    state=SafetyState.FAIL,
                    reason_code="LYSOGENY_HIGH_CONFIDENCE_PROFILE",
                    band=policy.high,
                    policy=policy,
                    profile_path=profile_path,
                    profile_digest=profile_digest,
                    database_version=database_version,
                    tool_pin=tool_pin,
                )
            )
        elif policy.review.matches(scores):
            review_findings.append(
                _phrogs_finding(
                    profile=profile,
                    record=record,
                    scores=scores,
                    state=SafetyState.INDETERMINATE,
                    reason_code="LYSOGENY_REVIEW_PROFILE",
                    band=policy.review,
                    policy=policy,
                    profile_path=profile_path,
                    profile_digest=profile_digest,
                    database_version=database_version,
                    tool_pin=tool_pin,
                )
            )

    if high_findings:
        state = SafetyState.FAIL
        reason_codes = ("LYSOGENY_HIGH_CONFIDENCE_PROFILE",)
    elif review_findings:
        state = SafetyState.INDETERMINATE
        reason_codes = ("LYSOGENY_REVIEW_PROFILE",)
    else:
        state = SafetyState.PASS
        reason_codes = ("PHROGS_MEASURED_NO_REVIEW_HIT",)
    findings = (*high_findings, *review_findings)
    if host_domain is HostDomain.ARCHAEA and not strict_lysis and findings:
        reason_codes = ("LYSOGENY_INFORMATIONAL_ARCHAEAL_PROFILE",)
    return AdapterResult(
        class_result=_class_result("lysogeny", state, required=required, findings=findings, reason_codes=reason_codes),
        raw_output_path=str(output_tsv),
        raw_output_sha256=_sha256_file(output_tsv),
    )


_parse_phrogs_output_validated = parse_phrogs_output


@_bind_result_policy(
    policy_id=PHROGS_HOMOLOGY_POLICY_V1.policy_id,
    policy_sha256=PHROGS_HOMOLOGY_POLICY_V1.sha256,
    policy_kwarg=True,
)
def parse_phrogs_output(
    output_tsv: Path,
    *,
    artifacts: ORFArtifacts,
    manifest_section: Mapping[str, object],
    tool_pin: ToolPin,
    host_domain: HostDomain,
    strict_lysis: bool = False,
    policy: HomologyPolicy = PHROGS_HOMOLOGY_POLICY_V1,
) -> AdapterResult:
    """Parse caller-supplied output conservatively; only :func:`run_phrogs` may emit measured PASS."""
    parsed = _parse_phrogs_output_validated(
        output_tsv,
        artifacts=artifacts,
        manifest_section=manifest_section,
        tool_pin=tool_pin,
        host_domain=host_domain,
        strict_lysis=strict_lysis,
        policy=policy,
    )
    return _without_validated_search_pass(
        parsed,
        safety_class="lysogeny",
        reason_code="PHROGS_SEARCH_EVIDENCE_UNVALIDATED",
    )


@_bind_result_policy(
    policy_id=PHROGS_HOMOLOGY_POLICY_V1.policy_id,
    policy_sha256=PHROGS_HOMOLOGY_POLICY_V1.sha256,
    policy_kwarg=True,
)
def run_phrogs(
    artifacts: ORFArtifacts,
    *,
    manifest_section: Mapping[str, object],
    tool_pin: ToolPin,
    host_domain: HostDomain,
    work_dir: Path,
    threads: int = 1,
    strict_lysis: bool = False,
    runner: CommandRunner = subprocess.run,
    timeout: float = 300.0,
    policy: HomologyPolicy = PHROGS_HOMOLOGY_POLICY_V1,
) -> AdapterResult:
    """Validate a PHROG-identity contract, execute profile search, and parse fail closed."""
    try:
        required = _lysogeny_required(host_domain, strict_lysis=strict_lysis)
    except ValueError:
        return _indeterminate_adapter_result("lysogeny", required=True, reason_code="PHROGS_UNSUPPORTED_HOST_PROFILE")
    try:
        profile_path, _, _, _, _, minimum_mmseqs_major = _validate_phrogs_assets(manifest_section)
        observed_version = validate_tool_pin(tool_pin, runner=runner, timeout=timeout)
        _validate_mmseqs_compatibility(
            expected_version=tool_pin.version,
            observed_version=observed_version,
            minimum_major=minimum_mmseqs_major,
        )
    except VerifiedIdentityMappingMissingError:
        return _indeterminate_adapter_result(
            "lysogeny", required=required, reason_code="PHROGS_VERIFIED_IDENTITY_MAPPING_MISSING"
        )
    except (AssetProvenanceError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        return _indeterminate_adapter_result(
            "lysogeny", required=required, reason_code="PHROGS_ASSET_PROVENANCE_MISMATCH"
        )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_output = work_dir / "phrogs.raw.tsv"
    normalized_output = work_dir / "phrogs.tsv"
    temporary_dir = work_dir / "tmp"
    raw_output.unlink(missing_ok=True)
    normalized_output.unlink(missing_ok=True)
    command = tuple(
        build_phrogs_command(
            mmseqs=tool_pin.path,
            profile_database=profile_path,
            proteins_faa=artifacts.proteins_faa,
            output_tsv=raw_output,
            temporary_dir=temporary_dir,
            threads=threads,
        )
    )
    try:
        runner(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _indeterminate_adapter_result(
            "lysogeny",
            required=required,
            reason_code="PHROGS_EXECUTION_TIMEOUT",
            command=command,
            raw_output_path=raw_output,
        )
    except (subprocess.CalledProcessError, OSError):
        return _indeterminate_adapter_result(
            "lysogeny",
            required=required,
            reason_code="PHROGS_EXECUTION_FAILED",
            command=command,
            raw_output_path=raw_output,
        )
    if not raw_output.is_file():
        return _indeterminate_adapter_result(
            "lysogeny",
            required=required,
            reason_code="PHROGS_OUTPUT_MISSING",
            command=command,
            raw_output_path=raw_output,
        )
    try:
        _write_normalized_header(normalized_output, _PHROGS_COLUMNS, raw_output)
    except (OSError, UnicodeError):
        return _indeterminate_adapter_result(
            "lysogeny",
            required=required,
            reason_code="PHROGS_PARSER_SCHEMA_MISMATCH",
            command=command,
            raw_output_path=raw_output,
        )
    parsed = _parse_phrogs_output_validated(
        normalized_output,
        artifacts=artifacts,
        manifest_section=manifest_section,
        tool_pin=tool_pin,
        host_domain=host_domain,
        strict_lysis=strict_lysis,
        policy=policy,
    )
    return AdapterResult(
        class_result=parsed.class_result,
        command=command,
        raw_output_path=parsed.raw_output_path,
        raw_output_sha256=parsed.raw_output_sha256,
    )
