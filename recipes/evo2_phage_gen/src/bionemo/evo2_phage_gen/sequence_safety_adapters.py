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

"""Adapters for AMR, toxin, and lysogeny sequence screens."""

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import local
from typing import Protocol
from urllib.parse import quote

from bionemo.evo2_phage_gen.design_scope import HostDomain
from bionemo.evo2_phage_gen.sequence_safety import SafetyClassResult, SafetyFinding, SafetyState


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_SEQUENCE_ID = re.compile(r"^[A-Za-z0-9_.|-]+$")


def _external_tool_environment() -> dict[str, str]:
    """Preserve the caller environment without interactive shell startup hooks."""
    environment = dict(os.environ)
    environment.pop("BASH_ENV", None)
    environment.pop("ENV", None)
    return environment


def _run_external(
    command: Sequence[str], *, runner: CommandRunner, timeout: float
) -> subprocess.CompletedProcess[str]:
    return runner(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_external_tool_environment(),
    )


@dataclass(frozen=True)
class GenomeInput:
    """Represent one genome submitted to the safety adapters."""

    sequence_id: str
    sequence: str
    circular: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize the genome."""
        if not _SEQUENCE_ID.fullmatch(self.sequence_id):
            raise ValueError(f"invalid genome sequence ID: {self.sequence_id!r}")
        sequence = "".join(self.sequence.split()).upper()
        if not sequence or set(sequence) - set("ACGTN"):
            raise ValueError(f"{self.sequence_id!r} must contain only A, C, G, T, or N")
        object.__setattr__(self, "sequence", sequence)


@dataclass(frozen=True)
class PredictedGene:
    """Represent a predicted coding sequence and translation."""

    start: int
    end: int
    strand: str
    nucleotide: str
    protein: str

    def __post_init__(self) -> None:
        """Validate gene coordinates, strand, and sequences."""
        if self.start < 1 or self.end < self.start or self.strand not in {"+", "-"}:
            raise ValueError("predicted gene coordinates or strand are invalid")
        if not self.nucleotide or not self.protein:
            raise ValueError("predicted genes require nucleotide and protein sequences")


class GenePredictor(Protocol):
    """Define the gene-prediction interface used by the adapters."""

    def predict(self, sequence: str, *, circular: bool) -> tuple[PredictedGene, ...]:
        """Predict genes in one genome."""
        ...


@dataclass(frozen=True)
class ORFQueryRecord:
    """Map one protein query back to its genome coordinates."""

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
    """Collect the files and query mappings produced by gene prediction."""

    genomes_fna: Path
    proteins_faa: Path
    proteins_fna: Path
    proteins_gff: Path
    all_queries_faa: Path
    query_records: tuple[ORFQueryRecord, ...]


@dataclass(frozen=True)
class ToolRuntime:
    """An executable selected for this run; its observed version is logged."""

    path: Path
    version_args: tuple[str, ...] = ("--version",)

    def __post_init__(self) -> None:
        """Normalize a tool path and its version arguments."""
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "version_args", tuple(self.version_args))


@dataclass(frozen=True, kw_only=True)
class NormalizedSafetyFinding(SafetyFinding):
    """Store normalized coordinates and detector evidence for a finding."""

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
    tool_version: str
    database_version: str
    evidence_path: str
    evidence_method: str
    policy_id: str
    profile: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the normalized finding."""
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
            "tool_version": self.tool_version,
            "database_version": self.database_version,
            "evidence_path": self.evidence_path,
            "evidence_method": self.evidence_method,
            "policy_id": self.policy_id,
            "profile": self.profile,
        }


@dataclass(frozen=True)
class AdapterResult:
    """Return one detector result together with execution evidence."""

    class_result: SafetyClassResult
    execution_status: str
    command: tuple[str, ...] = ()
    supplemental_findings: tuple[NormalizedSafetyFinding, ...] = ()


@dataclass(frozen=True)
class HomologyBand:
    """Define a homology threshold band."""

    identity: float
    query_coverage: float
    reference_coverage: float
    evalue: float
    alignment_length: float = 0.0

    def matches(self, scores: Mapping[str, float]) -> bool:
        """Return whether observed scores meet this band."""
        return (
            scores["identity"] >= self.identity
            and scores["query_coverage"] >= self.query_coverage
            and scores["reference_coverage"] >= self.reference_coverage
            and scores["alignment_length"] >= self.alignment_length
            and scores["evalue"] <= self.evalue
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize the threshold band."""
        return {
            "identity": self.identity,
            "query_coverage": self.query_coverage,
            "reference_coverage": self.reference_coverage,
            "alignment_length": self.alignment_length,
            "evalue": self.evalue,
        }


TOXIN_HIGH = HomologyBand(80.0, 80.0, 80.0, 1e-10)
TOXIN_REVIEW = HomologyBand(40.0, 60.0, 60.0, 1e-5)
TOXIN_FRAGMENT_REVIEW = HomologyBand(50.0, 70.0, 10.0, 1e-5, 50.0)
TOXIN_DOMAIN_REVIEW = HomologyBand(50.0, 0.0, 70.0, 1e-5, 50.0)
PHROGS_HIGH = HomologyBand(30.0, 0.70, 0.70, 1e-10)
PHROGS_REVIEW = HomologyBand(20.0, 0.50, 0.50, 1e-5)


class _PyrodigalPredictor:
    def __init__(self, module: object) -> None:
        self.module = module

    def predict(self, sequence: str, *, circular: bool) -> tuple[PredictedGene, ...]:
        length = len(sequence)
        search = sequence * 3 if circular else sequence
        finder = self.module.ViralGeneFinder(meta=True, viral_only=False, closed=not circular)
        calls: list[PredictedGene] = []
        seen: set[tuple[int, int, str, str]] = set()
        for gene in finder.find_genes(search):
            start, end = int(gene.begin), int(gene.end)
            if circular:
                if not length < start <= 2 * length:
                    continue
                start -= length
                end -= length
            strand = "+" if int(gene.strand) == 1 else "-"
            protein = str(gene.translate(include_stop=False)).rstrip("*")
            call = PredictedGene(start, end, strand, str(gene.sequence()), protein)
            key = (start, end, strand, protein)
            if key not in seen:
                seen.add(key)
                calls.append(call)
        return tuple(sorted(calls, key=lambda item: (item.start, item.end, item.strand)))


def _new_predictor() -> GenePredictor:
    import pyrodigal_gv

    return _PyrodigalPredictor(pyrodigal_gv)


def _primary_frame(gene: PredictedGene, length: int) -> int:
    if gene.strand == "+":
        return (gene.start - 1) % 3 + 1
    return -((length - ((gene.end - 1) % length + 1)) % 3 + 1)


def _six_frame_records(genome: GenomeInput, minimum_amino_acids: int) -> list[ORFQueryRecord]:
    from Bio.Seq import Seq

    records: list[ORFQueryRecord] = []
    sequence_length = len(genome.sequence)
    orientations = (("+", genome.sequence), ("-", str(Seq(genome.sequence).reverse_complement())))
    ordinal = 0
    for strand, oriented in orientations:
        for offset in range(3):
            usable = ((len(oriented) - offset) // 3) * 3
            if not usable:
                continue
            coding = oriented[offset : offset + usable]
            translated = str(Seq(coding).translate())
            peptide_start = 0
            for peptide_end in range(len(translated) + 1):
                if peptide_end < len(translated) and translated[peptide_end] != "*":
                    continue
                protein = translated[peptide_start:peptide_end]
                if len(protein) >= minimum_amino_acids:
                    ordinal += 1
                    if strand == "+":
                        start = offset + peptide_start * 3 + 1
                        end = offset + peptide_end * 3
                        frame = offset + 1
                    else:
                        start = sequence_length - (offset + peptide_end * 3) + 1
                        end = sequence_length - (offset + peptide_start * 3)
                        frame = -(offset + 1)
                    records.append(
                        ORFQueryRecord(
                            query_id=f"{genome.sequence_id}__sixframe_{ordinal:04d}",
                            sequence_id=genome.sequence_id,
                            start=start,
                            end=end,
                            strand=strand,
                            frame=frame,
                            nucleotide=coding[peptide_start * 3 : peptide_end * 3],
                            protein=protein,
                            evidence_path="six-frame-fallback",
                        )
                    )
                peptide_start = peptide_end + 1
    return records


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{identifier}\n{sequence}\n" for identifier, sequence in records))


def prepare_orf_artifacts(
    genomes: tuple[GenomeInput, ...],
    work_dir: Path,
    *,
    predictor: GenePredictor | None = None,
    minimum_fallback_amino_acids: int = 8,
    workers: int = 1,
) -> ORFArtifacts:
    """Predict ORFs and write the shared detector query files."""
    if not genomes or len({genome.sequence_id for genome in genomes}) != len(genomes):
        raise ValueError("ORF preparation requires unique, nonempty genome IDs")
    if type(workers) is not int or workers < 1:
        raise ValueError("ORF workers must be a positive integer")
    resolved_workers = min(workers, len(genomes))
    if predictor is not None and resolved_workers > 1:
        raise ValueError("a custom ORF predictor may only be used with one worker")
    thread_state = local()

    def predict(genome: GenomeInput) -> tuple[tuple[PredictedGene, ...], tuple[ORFQueryRecord, ...]]:
        selected = predictor
        if selected is None:
            selected = getattr(thread_state, "predictor", None)
            if selected is None:
                selected = _new_predictor()
                thread_state.predictor = selected
        calls = selected.predict(genome.sequence, circular=genome.circular)
        primary_keys = {(gene.start, gene.end, gene.strand, gene.protein) for gene in calls}
        fallback = tuple(
            record
            for record in _six_frame_records(genome, minimum_fallback_amino_acids)
            if (record.start, record.end, record.strand, record.protein) not in primary_keys
        )
        return calls, fallback

    if resolved_workers == 1:
        predictions = tuple(predict(genome) for genome in genomes)
    else:
        with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="phage-orf") as executor:
            predictions = tuple(executor.map(predict, genomes))

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    primary: list[ORFQueryRecord] = []
    fallback: list[ORFQueryRecord] = []
    gff = ["##gff-version 3"]
    for genome, (calls, fallback_records) in zip(genomes, predictions, strict=True):
        gff.append(f"##sequence-region {genome.sequence_id} 1 {len(genome.sequence)}")
        for index, gene in enumerate(calls, start=1):
            query_id = f"{genome.sequence_id}__orf{index:04d}"
            record = ORFQueryRecord(
                query_id,
                genome.sequence_id,
                gene.start,
                gene.end,
                gene.strand,
                _primary_frame(gene, len(genome.sequence)),
                gene.nucleotide,
                gene.protein,
                "pyrodigal-gv",
            )
            primary.append(record)
            escaped = quote(query_id, safe="._-")
            gff.append(
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
                        f"ID={escaped};Name={escaped}",
                    )
                )
            )
        fallback.extend(fallback_records)

    genomes_fna = work_dir / "genomes.fna"
    proteins_faa = work_dir / "proteins.faa"
    proteins_fna = work_dir / "proteins.fna"
    proteins_gff = work_dir / "proteins.gff"
    all_queries_faa = work_dir / "all_queries.faa"
    _write_fasta(genomes_fna, [(genome.sequence_id, genome.sequence) for genome in genomes])
    _write_fasta(proteins_faa, [(record.query_id, record.protein) for record in primary])
    _write_fasta(proteins_fna, [(record.query_id, record.nucleotide) for record in primary])
    proteins_gff.write_text("\n".join(gff) + "\n")
    query_records = (*primary, *fallback)
    _write_fasta(all_queries_faa, [(record.query_id, record.protein) for record in query_records])
    return ORFArtifacts(
        genomes_fna,
        proteins_faa,
        proteins_fna,
        proteins_gff,
        all_queries_faa,
        query_records,
    )


def observe_tool_version(
    runtime: ToolRuntime,
    *,
    runner: CommandRunner = subprocess.run,
    timeout: float = 300.0,
) -> str:
    """Read a tool version without interpreting its format."""
    if not runtime.path.is_file():
        raise FileNotFoundError(runtime.path)
    completed = _run_external((str(runtime.path), *runtime.version_args), runner=runner, timeout=timeout)
    return (completed.stdout.strip() or completed.stderr.strip()).splitlines()[0]


def _class_result(
    safety_class: str,
    state: SafetyState,
    required: bool,
    findings: tuple[NormalizedSafetyFinding, ...] = (),
    reasons: tuple[str, ...] = (),
) -> SafetyClassResult:
    return SafetyClassResult(safety_class, state, required, findings, reasons)


def _result(
    safety_class: str,
    state: SafetyState,
    required: bool,
    reason: str,
    *,
    findings: tuple[NormalizedSafetyFinding, ...] = (),
    status: str = "COMPLETED_AND_PARSED",
    command: tuple[str, ...] = (),
    supplemental: tuple[NormalizedSafetyFinding, ...] = (),
) -> AdapterResult:
    return AdapterResult(
        _class_result(safety_class, state, required, findings, (reason,)),
        status,
        command,
        supplemental,
    )


def _all_results(
    sequence_ids: tuple[str, ...],
    safety_class: str,
    state: SafetyState,
    required: bool,
    reason: str,
    *,
    status: str,
    command: tuple[str, ...] = (),
) -> dict[str, AdapterResult]:
    return {
        sequence_id: _result(safety_class, state, required, reason, status=status, command=command)
        for sequence_id in sequence_ids
    }


def _finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(value)
    return parsed


def _query_index(artifacts: ORFArtifacts) -> dict[str, ORFQueryRecord]:
    return {record.query_id: record for record in artifacts.query_records}


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

_AMRFINDER_REQUIRED_COLUMNS = frozenset(
    {
        "Protein id",
        "Contig id",
        "Start",
        "Stop",
        "Strand",
        "Element symbol",
        "Scope",
        "Type",
        "Method",
        "% Coverage of reference",
        "% Identity to reference",
        "Alignment length",
        "Closest reference accession",
        "HMM accession",
    }
)


def _amrfinder_command(
    artifacts: ORFArtifacts,
    runtime: ToolRuntime,
    database: Path,
    bin_dir: Path,
    output: Path,
    threads: int,
) -> tuple[str, ...]:
    command = [
        str(runtime.path),
        "-n",
        str(artifacts.genomes_fna),
    ]
    if artifacts.proteins_faa.stat().st_size:
        command.extend(("-p", str(artifacts.proteins_faa), "-g", str(artifacts.proteins_gff)))
    command.extend(
        (
            "--annotation_format",
            "standard",
            "--plus",
            "--database",
            str(database),
            "--blast_bin",
            str(bin_dir),
            "--hmmer_bin",
            str(bin_dir),
            "--threads",
            str(threads),
            "-o",
            str(output),
        )
    )
    return tuple(command)


def run_amrfinder_batch(
    genomes: tuple[GenomeInput, ...],
    artifacts: ORFArtifacts,
    *,
    runtime: ToolRuntime,
    database: Path,
    database_version: str,
    work_dir: Path,
    threads: int = 1,
    runner: CommandRunner = subprocess.run,
    timeout: float = 300.0,
) -> dict[str, AdapterResult]:
    """Run AMRFinderPlus for a batch of genomes."""
    sequence_ids = tuple(genome.sequence_id for genome in genomes)
    output = Path(work_dir) / "amrfinder.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = _amrfinder_command(artifacts, runtime, Path(database), runtime.path.parent, output, threads)
    try:
        version = observe_tool_version(runtime, runner=runner, timeout=timeout)
        _run_external(command, runner=runner, timeout=timeout)
        if not output.is_file():
            raise ValueError("missing output")
        with output.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = frozenset(reader.fieldnames or ())
            rows = list(reader)
        if not _AMRFINDER_REQUIRED_COLUMNS <= fieldnames:
            raise ValueError("missing required columns")
    except subprocess.TimeoutExpired:
        return _all_results(
            sequence_ids,
            "amr",
            SafetyState.INDETERMINATE,
            True,
            "AMRFINDER_EXECUTION_TIMEOUT",
            status="FAILED",
            command=command,
        )
    except subprocess.CalledProcessError as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        (Path(work_dir) / "amrfinder.log").write_text(f"stdout:\n{stdout.rstrip()}\n\nstderr:\n{stderr.rstrip()}\n")
        return _all_results(
            sequence_ids,
            "amr",
            SafetyState.INDETERMINATE,
            True,
            "AMRFINDER_EXECUTION_FAILED",
            status="FAILED",
            command=command,
        )
    except (OSError, subprocess.SubprocessError):
        return _all_results(
            sequence_ids,
            "amr",
            SafetyState.INDETERMINATE,
            True,
            "AMRFINDER_EXECUTION_FAILED",
            status="FAILED",
            command=command,
        )
    except (TypeError, ValueError):
        return _all_results(
            sequence_ids,
            "amr",
            SafetyState.INDETERMINATE,
            True,
            "AMRFINDER_PARSER_ERROR",
            status="PARSER_ERROR",
            command=command,
        )

    queries = _query_index(artifacts)
    genome_lengths = {genome.sequence_id: len(genome.sequence) for genome in genomes}
    amr: dict[str, list[NormalizedSafetyFinding]] = {sequence_id: [] for sequence_id in sequence_ids}
    virulence: dict[str, list[NormalizedSafetyFinding]] = {sequence_id: [] for sequence_id in sequence_ids}
    try:
        for row in rows:
            protein_id = row["Protein id"]
            contig_id = row["Contig id"]
            if contig_id not in genome_lengths:
                raise ValueError("unknown contig")
            if protein_id == "NA":
                start, end = int(row["Start"]), int(row["Stop"])
                strand = row["Strand"]
                if start < 1 or end < start or end > genome_lengths[contig_id] or strand not in {"+", "-"}:
                    raise ValueError("invalid nucleotide coordinates")
                frame = (start - 1) % 3 + 1 if strand == "+" else -((genome_lengths[contig_id] - end) % 3 + 1)
                query_id = f"{contig_id}__amrfinder_nt_{start}_{end}"
                evidence_path = "amrfinder-nucleotide"
            else:
                record = queries.get(protein_id)
                if record is None or record.sequence_id != contig_id:
                    raise ValueError("unknown protein or contig")
                start, end, strand, frame = record.start, record.end, record.strand, record.frame
                query_id, evidence_path = record.query_id, record.evidence_path
            for key in ("% Coverage of reference", "% Identity to reference"):
                if row[key] != "NA" and not 0 <= _finite(row[key]) <= 100:
                    raise ValueError("percent outside range")
            accession = row["Closest reference accession"]
            if accession in {"", "NA"}:
                accession = row["HMM accession"] or row["Element symbol"]
            scores = {
                name: _finite(row[column])
                for name, column in (
                    ("identity", "% Identity to reference"),
                    ("reference_coverage", "% Coverage of reference"),
                    ("alignment_length", "Alignment length"),
                )
                if row[column] != "NA"
            }
            finding_args = {
                "detector": "amrfinder-plus",
                "accession": accession,
                "query_id": query_id,
                "sequence_id": contig_id,
                "start": start,
                "end": end,
                "strand": strand,
                "frame": frame,
                "scores": scores,
                "thresholds": {},
                "tool_version": version,
                "database_version": database_version,
                "evidence_path": evidence_path,
                "evidence_method": row["Method"],
                "policy_id": "amrfinder-curated-calls",
            }
            if row["Type"] == "AMR":
                amr[contig_id].append(
                    NormalizedSafetyFinding(
                        safety_class="amr",
                        state=SafetyState.FAIL,
                        reason_codes=("AMR_DETERMINANT_DETECTED",),
                        finding_id=f"amr:{query_id}:{accession}",
                        **finding_args,
                    )
                )
            elif row["Scope"] == "plus" and row["Type"] == "VIRULENCE":
                virulence[contig_id].append(
                    NormalizedSafetyFinding(
                        safety_class="toxin",
                        state=SafetyState.INDETERMINATE,
                        reason_codes=("AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL",),
                        finding_id=f"toxin:{query_id}:{accession}",
                        **finding_args,
                    )
                )
    except (KeyError, TypeError, ValueError):
        return _all_results(
            sequence_ids,
            "amr",
            SafetyState.INDETERMINATE,
            True,
            "AMRFINDER_PARSER_ERROR",
            status="PARSER_ERROR",
            command=command,
        )

    return {
        sequence_id: _result(
            "amr",
            SafetyState.FAIL if amr[sequence_id] else SafetyState.PASS,
            True,
            "AMR_DETERMINANT_DETECTED" if amr[sequence_id] else "AMRFINDER_MEASURED_NO_AMR_HIT",
            findings=tuple(amr[sequence_id]),
            command=command,
            supplemental=tuple(virulence[sequence_id]),
        )
        for sequence_id in sequence_ids
    }


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


def run_toxin_batch(
    genomes: tuple[GenomeInput, ...],
    artifacts: ORFArtifacts,
    *,
    runtime: ToolRuntime,
    database: Path,
    database_version: str,
    work_dir: Path,
    threads: int = 1,
    runner: CommandRunner = subprocess.run,
    timeout: float = 300.0,
) -> dict[str, AdapterResult]:
    """Run toxin homology screening for a batch of genomes."""
    sequence_ids = tuple(genome.sequence_id for genome in genomes)
    queries = _query_index(artifacts)
    by_sequence = {
        sequence_id: [record for record in artifacts.query_records if record.sequence_id == sequence_id]
        for sequence_id in sequence_ids
    }
    output = Path(work_dir) / "toxins.tsv"
    command = (
        str(runtime.path),
        "blastp",
        "--query",
        str(artifacts.all_queries_faa),
        "--db",
        str(database),
        "--out",
        str(output),
        "--outfmt",
        "6",
        *_DIAMOND_COLUMNS,
        "--threads",
        str(threads),
        "--max-target-seqs",
        "0",
        "--sensitive",
    )
    if not queries:
        return _all_results(
            sequence_ids,
            "toxin",
            SafetyState.INDETERMINATE,
            True,
            "TOXIN_NO_PROTEIN_QUERIES",
            status="NOT_RUN",
            command=command,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        version = observe_tool_version(runtime, runner=runner, timeout=timeout)
        _run_external(command, runner=runner, timeout=timeout)
        if not output.is_file():
            raise ValueError("missing output")
        rows = [line.split("\t") for line in output.read_text().splitlines() if line.strip()]
    except subprocess.TimeoutExpired:
        return _all_results(
            sequence_ids,
            "toxin",
            SafetyState.INDETERMINATE,
            True,
            "TOXIN_EXECUTION_TIMEOUT",
            status="FAILED",
            command=command,
        )
    except (OSError, subprocess.SubprocessError):
        return _all_results(
            sequence_ids,
            "toxin",
            SafetyState.INDETERMINATE,
            True,
            "TOXIN_EXECUTION_FAILED",
            status="FAILED",
            command=command,
        )
    except ValueError:
        return _all_results(
            sequence_ids,
            "toxin",
            SafetyState.INDETERMINATE,
            True,
            "TOXIN_PARSER_ERROR",
            status="PARSER_ERROR",
            command=command,
        )

    findings: dict[str, list[NormalizedSafetyFinding]] = {sequence_id: [] for sequence_id in sequence_ids}
    try:
        for fields in rows:
            if len(fields) != len(_DIAMOND_COLUMNS):
                raise ValueError("wrong column count")
            row = dict(zip(_DIAMOND_COLUMNS, fields, strict=True))
            record = queries.get(row["qseqid"])
            if record is None:
                raise ValueError("unknown query")
            scores = {
                "identity": _finite(row["pident"]),
                "alignment_length": _finite(row["length"]),
                "query_length": _finite(row["qlen"]),
                "reference_length": _finite(row["slen"]),
                "query_coverage": _finite(row["qcovhsp"]),
                "reference_coverage": _finite(row["scovhsp"]),
                "evalue": _finite(row["evalue"]),
                "bitscore": _finite(row["bitscore"]),
            }
            if not 0 <= scores["identity"] <= 100 or not 0 <= scores["query_coverage"] <= 100:
                raise ValueError("invalid score")
            target = row["sseqid"]
            curated = target.startswith("domain|")
            band: HomologyBand | None = None
            state: SafetyState | None = None
            reason = ""
            if curated and TOXIN_DOMAIN_REVIEW.matches(scores):
                band, state, reason = TOXIN_DOMAIN_REVIEW, SafetyState.INDETERMINATE, "TOXIN_CURATED_DOMAIN_REVIEW"
            elif TOXIN_HIGH.matches(scores):
                band, state, reason = TOXIN_HIGH, SafetyState.FAIL, "TOXIN_HIGH_CONFIDENCE_HOMOLOGY"
            elif TOXIN_REVIEW.matches(scores) or TOXIN_FRAGMENT_REVIEW.matches(scores):
                band, state, reason = TOXIN_REVIEW, SafetyState.INDETERMINATE, "TOXIN_HOMOLOGY_REVIEW"
            if band is None or state is None:
                continue
            parts = target.split("|")
            accession = parts[1] if len(parts) >= 2 else target
            findings[record.sequence_id].append(
                NormalizedSafetyFinding(
                    safety_class="toxin",
                    state=state,
                    reason_codes=(reason,),
                    finding_id=f"toxin:{record.query_id}:{accession}",
                    detector="diamond-toxin-reference",
                    accession=accession,
                    query_id=record.query_id,
                    sequence_id=record.sequence_id,
                    start=record.start,
                    end=record.end,
                    strand=record.strand,
                    frame=record.frame,
                    scores=scores,
                    thresholds=band.to_dict(),
                    tool_version=version,
                    database_version=database_version,
                    evidence_path=record.evidence_path,
                    evidence_method="diamond-blastp",
                    policy_id="toxin-homology-v2",
                    profile=accession if curated else None,
                )
            )
    except (KeyError, TypeError, ValueError):
        return _all_results(
            sequence_ids,
            "toxin",
            SafetyState.INDETERMINATE,
            True,
            "TOXIN_PARSER_ERROR",
            status="PARSER_ERROR",
            command=command,
        )

    results: dict[str, AdapterResult] = {}
    for sequence_id in sequence_ids:
        selected = findings[sequence_id]
        if not by_sequence[sequence_id]:
            results[sequence_id] = _result(
                "toxin", SafetyState.INDETERMINATE, True, "TOXIN_NO_PROTEIN_QUERIES", status="NOT_RUN"
            )
        elif any(finding.state is SafetyState.FAIL for finding in selected):
            results[sequence_id] = _result(
                "toxin",
                SafetyState.FAIL,
                True,
                "TOXIN_HIGH_CONFIDENCE_HOMOLOGY",
                findings=tuple(selected),
                command=command,
            )
        elif selected:
            results[sequence_id] = _result(
                "toxin",
                SafetyState.INDETERMINATE,
                True,
                "TOXIN_HOMOLOGY_REVIEW",
                findings=tuple(selected),
                command=command,
            )
        else:
            results[sequence_id] = _result(
                "toxin", SafetyState.PASS, True, "TOXIN_MEASURED_NO_REVIEW_HIT", command=command
            )
    return results


_PHROGS_COLUMNS = ("query", "target", "pident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits")


def _phrogs_profiles(path: Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    profiles = {row["phrog"]: row for row in rows}
    if not profiles:
        raise ValueError("PHROGs lysogeny lookup is empty")
    return profiles


def _phrogs_commands(
    runtime: ToolRuntime,
    database: Path,
    proteins: Path,
    output: Path,
    temporary: Path,
    threads: int,
) -> tuple[tuple[str, ...], ...]:
    target = temporary / "orf-target"
    hits = temporary / "profile-hits"
    return (
        (str(runtime.path), "createdb", str(proteins), str(target)),
        (
            str(runtime.path),
            "search",
            str(database),
            str(target),
            str(hits),
            str(temporary / "search"),
            "--threads",
            str(threads),
            "--alignment-mode",
            "3",
        ),
        (
            str(runtime.path),
            "convertalis",
            str(database),
            str(target),
            str(hits),
            str(output),
            "--format-output",
            ",".join(_PHROGS_COLUMNS),
        ),
    )


def run_phrogs_batch(
    genomes: tuple[GenomeInput, ...],
    artifacts: ORFArtifacts,
    *,
    runtime: ToolRuntime,
    database: Path,
    lookup: Path,
    database_version: str,
    host_domain: HostDomain,
    strict_lysis: bool,
    work_dir: Path,
    threads: int = 1,
    runner: CommandRunner = subprocess.run,
    timeout: float = 300.0,
) -> dict[str, AdapterResult]:
    """Run PHROGs lysogeny screening for a batch of genomes."""
    sequence_ids = tuple(genome.sequence_id for genome in genomes)
    required = host_domain in {HostDomain.BACTERIA, HostDomain.BACTERIA_AND_ARCHAEA} or strict_lysis
    primary = {record.query_id: record for record in artifacts.query_records if record.evidence_path == "pyrodigal-gv"}
    by_sequence = {
        sequence_id: [record for record in primary.values() if record.sequence_id == sequence_id]
        for sequence_id in sequence_ids
    }
    output = Path(work_dir) / "phrogs.tsv"
    temporary = Path(work_dir) / "phrogs-tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    commands = _phrogs_commands(runtime, Path(database), artifacts.proteins_faa, output, temporary, threads)
    if not primary:
        return _all_results(
            sequence_ids,
            "lysogeny",
            SafetyState.INDETERMINATE,
            required,
            "PHROGS_NO_PREDICTED_GENES",
            status="NOT_RUN",
            command=commands[-1],
        )
    try:
        profiles = _phrogs_profiles(lookup)
        version = observe_tool_version(runtime, runner=runner, timeout=timeout)
        for command in commands:
            _run_external(command, runner=runner, timeout=timeout)
        if not output.is_file():
            raise ValueError("missing output")
        rows = [line.split("\t") for line in output.read_text().splitlines() if line.strip()]
    except subprocess.TimeoutExpired:
        return _all_results(
            sequence_ids,
            "lysogeny",
            SafetyState.INDETERMINATE,
            required,
            "PHROGS_EXECUTION_TIMEOUT",
            status="FAILED",
            command=commands[-1],
        )
    except (OSError, subprocess.SubprocessError):
        return _all_results(
            sequence_ids,
            "lysogeny",
            SafetyState.INDETERMINATE,
            required,
            "PHROGS_EXECUTION_FAILED",
            status="FAILED",
            command=commands[-1],
        )
    except (KeyError, TypeError, ValueError):
        return _all_results(
            sequence_ids,
            "lysogeny",
            SafetyState.INDETERMINATE,
            required,
            "PHROGS_PARSER_ERROR",
            status="PARSER_ERROR",
            command=commands[-1],
        )

    findings: dict[str, list[NormalizedSafetyFinding]] = {sequence_id: [] for sequence_id in sequence_ids}
    try:
        for fields in rows:
            if len(fields) != len(_PHROGS_COLUMNS):
                raise ValueError("wrong column count")
            row = dict(zip(_PHROGS_COLUMNS, fields, strict=True))
            metadata = profiles.get(row["query"])
            if metadata is None:
                continue
            record = primary.get(row["target"])
            if record is None:
                raise ValueError("unknown ORF")
            scores = {
                "identity": _finite(row["pident"]),
                "alignment_length": _finite(row["alnlen"]),
                "query_length": _finite(row["qlen"]),
                "reference_length": _finite(row["tlen"]),
                "query_coverage": _finite(row["qcov"]),
                "reference_coverage": _finite(row["tcov"]),
                "evalue": _finite(row["evalue"]),
                "bitscore": _finite(row["bits"]),
            }
            if not 0 <= scores["identity"] <= 100 or not 0 <= scores["query_coverage"] <= 1:
                raise ValueError("invalid score")
            high = metadata["confidence"] == "high_confidence" and PHROGS_HIGH.matches(scores)
            review = PHROGS_REVIEW.matches(scores)
            if not high and not review:
                continue
            state = SafetyState.FAIL if high else SafetyState.INDETERMINATE
            reason = "LYSOGENY_HIGH_CONFIDENCE_PROFILE" if high else "LYSOGENY_REVIEW_PROFILE"
            findings[record.sequence_id].append(
                NormalizedSafetyFinding(
                    safety_class="lysogeny",
                    state=state,
                    reason_codes=(reason,),
                    finding_id=f"lysogeny:{record.query_id}:{row['query']}",
                    detector="mmseqs-phrogs-v4",
                    accession=row["query"],
                    query_id=record.query_id,
                    sequence_id=record.sequence_id,
                    start=record.start,
                    end=record.end,
                    strand=record.strand,
                    frame=record.frame,
                    scores=scores,
                    thresholds=(PHROGS_HIGH if high else PHROGS_REVIEW).to_dict(),
                    tool_version=version,
                    database_version=database_version,
                    evidence_path=record.evidence_path,
                    evidence_method="mmseqs-profile-search",
                    policy_id="phrogs-homology-v1",
                    profile=row["query"],
                )
            )
    except (KeyError, TypeError, ValueError):
        return _all_results(
            sequence_ids,
            "lysogeny",
            SafetyState.INDETERMINATE,
            required,
            "PHROGS_PARSER_ERROR",
            status="PARSER_ERROR",
            command=commands[-1],
        )

    results: dict[str, AdapterResult] = {}
    for sequence_id in sequence_ids:
        selected = findings[sequence_id]
        if not by_sequence[sequence_id]:
            results[sequence_id] = _result(
                "lysogeny", SafetyState.INDETERMINATE, required, "PHROGS_NO_PREDICTED_GENES", status="NOT_RUN"
            )
        elif any(finding.state is SafetyState.FAIL for finding in selected):
            results[sequence_id] = _result(
                "lysogeny",
                SafetyState.FAIL,
                required,
                "LYSOGENY_HIGH_CONFIDENCE_PROFILE",
                findings=tuple(selected),
                command=commands[-1],
            )
        elif selected:
            results[sequence_id] = _result(
                "lysogeny",
                SafetyState.INDETERMINATE,
                required,
                "LYSOGENY_REVIEW_PROFILE",
                findings=tuple(selected),
                command=commands[-1],
            )
        else:
            results[sequence_id] = _result(
                "lysogeny", SafetyState.PASS, required, "PHROGS_MEASURED_NO_REVIEW_HIT", command=commands[-1]
            )
    return results
