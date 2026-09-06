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

"""Coverage-aware protein evidence shared by online rewards and Arc hard QC."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.optimize import linear_sum_assignment


ORFIPY_INTERVAL_RE = re.compile(r"\[(\d+)-(\d+)\]")


@dataclass(frozen=True)
class SmoothReferenceArchitecture:
    """Continuous content, order, and excess-copy evidence for reference loci."""

    reward: float
    content_score: float
    ordered_score: float
    duplicate_score: float
    content_integrity_sum: float
    ordered_integrity_sum: float
    duplicate_integrity_sum: float
    assignment: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GeneAOriginScore:
    """Gene-A replication-origin reward and its interpretable components."""

    reward: float
    motif_score: float
    position_score: float
    exact_functional_site: bool
    strong_site_count: int


def _validate_smooth_protein_match_config(
    *,
    identity_full_credit: float,
    reference_coverage_full_credit: float,
    candidate_coverage_full_credit: float,
    gamma: float,
    raw_integrity_min: float,
    min_credit: float,
    significance_zero_evalue: float = 1.0,
    significance_full_evalue: float = 1e-5,
) -> None:
    if not (
        0.0 < identity_full_credit <= 1.0
        and 0.0 < reference_coverage_full_credit <= 1.0
        and 0.0 < candidate_coverage_full_credit <= 1.0
        and gamma > 0.0
        and 0.0 <= raw_integrity_min < 1.0
        and 0.0 <= min_credit < 1.0
        and 0.0 < significance_full_evalue < significance_zero_evalue
    ):
        raise ValueError("Invalid smooth protein-match configuration")


def smooth_protein_match_integrity(
    percent_identity: object,
    e_value: object,
    alignment_length: object,
    reference_length: object,
    candidate_length: object,
    *,
    identity_full_credit: float,
    reference_coverage_full_credit: float,
    candidate_coverage_full_credit: float,
    gamma: float,
    raw_integrity_min: float,
    min_credit: float,
    significance_zero_evalue: float = 1.0,
    significance_full_evalue: float = 1e-5,
) -> float:
    """Grade a complete-ORF alignment while suppressing shuffled-sequence evidence."""
    _validate_smooth_protein_match_config(
        identity_full_credit=identity_full_credit,
        reference_coverage_full_credit=reference_coverage_full_credit,
        candidate_coverage_full_credit=candidate_coverage_full_credit,
        gamma=gamma,
        raw_integrity_min=raw_integrity_min,
        min_credit=min_credit,
        significance_zero_evalue=significance_zero_evalue,
        significance_full_evalue=significance_full_evalue,
    )
    try:
        identity, evalue, aligned, reference, candidate = map(
            float,
            (percent_identity, e_value, alignment_length, reference_length, candidate_length),
        )
    except (TypeError, ValueError):
        return 0.0
    values = (identity, evalue, aligned, reference, candidate)
    if not all(math.isfinite(value) for value in values):
        return 0.0
    if not 0.0 <= identity <= 100.0 or evalue < 0.0 or min(aligned, reference, candidate) <= 0.0:
        return 0.0
    if evalue >= significance_zero_evalue:
        significance = 0.0
    elif evalue <= significance_full_evalue:
        significance = 1.0
    else:
        zero_log = -math.log10(significance_zero_evalue)
        full_log = -math.log10(significance_full_evalue)
        significance = (-math.log10(evalue) - zero_log) / (full_log - zero_log)

    identity_progress = min((identity / 100.0) / identity_full_credit, 1.0)
    reference_progress = min((aligned / reference) / reference_coverage_full_credit, 1.0)
    candidate_progress = min((aligned / candidate) / candidate_coverage_full_credit, 1.0)
    raw_integrity = significance * (identity_progress * reference_progress * candidate_progress) ** gamma
    if raw_integrity <= raw_integrity_min:
        return 0.0
    rescaled = (raw_integrity - raw_integrity_min) / (1.0 - raw_integrity_min)
    return min(1.0, min_credit + (1.0 - min_credit) * rescaled)


def _maximum_weight_reference_assignment(
    edge_weights: dict[tuple[str, str], float],
    reference_order: tuple[str, ...],
    candidate_order: tuple[str, ...],
) -> tuple[float, tuple[tuple[str, str], ...]]:
    """Return an exact maximum-weight one-to-one assignment at genome scale."""
    if len(set(reference_order)) != len(reference_order) or len(set(candidate_order)) != len(candidate_order):
        raise ValueError("Reference and candidate orders must contain unique locus identifiers")
    if not reference_order or not candidate_order:
        return 0.0, ()
    matrix = [
        [float(edge_weights.get((reference, candidate), 0.0)) for candidate in candidate_order]
        for reference in reference_order
    ]
    row_indices, column_indices = linear_sum_assignment(matrix, maximize=True)
    assignment = tuple(
        sorted(
            (
                (reference_order[row], candidate_order[column])
                for row, column in zip(row_indices, column_indices, strict=True)
                if matrix[row][column] > 0.0
            ),
            key=lambda pair: (reference_order.index(pair[0]), candidate_order.index(pair[1])),
        )
    )
    return sum(float(edge_weights[pair]) for pair in assignment), assignment


def _linear_ordered_integrity(
    edge_weights: dict[tuple[str, str], float],
    reference_order: tuple[str, ...],
    candidate_order: tuple[str, ...],
) -> float:
    """Return a maximum-weight order-preserving one-to-one alignment."""
    previous = [0.0] * (len(candidate_order) + 1)
    for reference in reference_order:
        current = [0.0]
        for candidate_index, candidate in enumerate(candidate_order, start=1):
            current.append(
                max(
                    previous[candidate_index],
                    current[candidate_index - 1],
                    previous[candidate_index - 1] + float(edge_weights.get((reference, candidate), 0.0)),
                )
            )
        previous = current
    return previous[-1]


def score_smooth_reference_architecture(
    edge_weights: dict[tuple[str, str], float],
    *,
    reference_order: tuple[str, ...],
    candidate_order: tuple[str, ...],
    order_weight: float,
    duplicate_penalty_weight: float,
) -> SmoothReferenceArchitecture:
    """Score ORF content and circular order without rewarding deletion or duplicate repair."""
    if not reference_order:
        return SmoothReferenceArchitecture(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ())
    if len(set(reference_order)) != len(reference_order) or len(set(candidate_order)) != len(candidate_order):
        raise ValueError("Reference and candidate orders must contain unique locus identifiers")
    if not 0.0 <= order_weight <= 1.0 or duplicate_penalty_weight < 0.0:
        raise ValueError("Invalid smooth architecture weights")
    if any(not math.isfinite(float(weight)) or not 0.0 <= float(weight) <= 1.0 for weight in edge_weights.values()):
        raise ValueError("Smooth architecture edges must be finite values in [0, 1]")

    edge_candidates = {candidate for _reference, candidate in edge_weights}
    all_candidates = candidate_order + tuple(sorted(edge_candidates - set(candidate_order)))
    content_sum, assignment = _maximum_weight_reference_assignment(edge_weights, reference_order, all_candidates)
    ordered_sum = max(
        (
            _linear_ordered_integrity(
                edge_weights,
                reference_order,
                candidate_order[offset:] + candidate_order[:offset],
            )
            for offset in range(len(candidate_order))
        ),
        default=0.0,
    )
    homolog_mass = sum(
        max((float(edge_weights.get((reference, candidate), 0.0)) for reference in reference_order), default=0.0)
        for candidate in all_candidates
    )
    duplicate_sum = max(0.0, homolog_mass - content_sum)
    denominator = float(len(reference_order))
    content_score = content_sum / denominator
    ordered_score = ordered_sum / denominator
    duplicate_score = duplicate_sum / denominator
    reward = (
        (1.0 - order_weight) * content_score
        + order_weight * ordered_score
        - duplicate_penalty_weight * duplicate_score
    )
    return SmoothReferenceArchitecture(
        reward=max(0.0, min(1.0, reward)),
        content_score=content_score,
        ordered_score=ordered_score,
        duplicate_score=duplicate_score,
        content_integrity_sum=content_sum,
        ordered_integrity_sum=ordered_sum,
        duplicate_integrity_sum=duplicate_sum,
        assignment=assignment,
    )


def _origin_site_components(observed: str, functional_motif: str) -> tuple[float, float, float]:
    recognition = sum(a == b for a, b in zip(observed[:10], functional_motif[:10], strict=False)) / 10.0
    binding = sum(a == b for a, b in zip(observed[10:28], functional_motif[10:28], strict=False)) / 18.0
    nicking = sum(a == b for a, b in zip(observed[3:7], functional_motif[3:7], strict=False)) / 4.0
    return recognition, binding, nicking


def _circular_strong_origin_count(sequence: str, functional_motif: str) -> int:
    extended = sequence + sequence[: len(functional_motif) - 1]
    count = 0
    for index in range(len(sequence)):
        recognition, binding, nicking = _origin_site_components(
            extended[index : index + len(functional_motif)],
            functional_motif,
        )
        count += recognition >= 0.8 and binding >= 14.0 / 18.0 and nicking == 1.0
    return count


def score_gene_a_origin(
    *,
    candidate_a_orf_nt: str,
    candidate_genome_nt: str,
    a_match_integrity: float,
    motif: str,
    expected_offset_nt: int,
    offset_tolerance_nt: int,
) -> GeneAOriginScore:
    """Score the PhiX replication origin only in its expected frame and gene-A context."""
    candidate_a_orf_nt = str(candidate_a_orf_nt).upper()
    candidate_genome_nt = str(candidate_genome_nt).upper()
    motif = str(motif).upper()
    if len(motif) < 28 or expected_offset_nt < 0 or offset_tolerance_nt <= 0:
        raise ValueError("Gene-A origin scoring requires a 28-base motif and positive offset tolerance")
    if not math.isfinite(float(a_match_integrity)) or not 0.0 <= float(a_match_integrity) <= 1.0:
        raise ValueError("Gene-A match integrity must be in [0, 1]")

    functional_motif = motif[:28]
    best_motif_score = 0.0
    best_position_score = 0.0
    best_exact = False
    first = max(0, expected_offset_nt - offset_tolerance_nt)
    last = min(len(candidate_a_orf_nt) - len(motif), expected_offset_nt + offset_tolerance_nt)
    for offset in range(first, last + 1):
        if (offset - expected_offset_nt) % 3 != 0:
            continue
        observed = candidate_a_orf_nt[offset : offset + len(motif)]
        recognition, binding, nicking = _origin_site_components(observed, functional_motif)
        if recognition < 0.8 or binding < 14.0 / 18.0:
            continue
        motif_score = nicking**2 * recognition * binding
        position_score = 1.0 - abs(offset - expected_offset_nt) / float(offset_tolerance_nt)
        combined = motif_score * max(0.0, position_score)
        if combined > best_motif_score * best_position_score:
            best_motif_score = motif_score
            best_position_score = max(0.0, position_score)
            best_exact = observed[:28] == functional_motif

    strong_site_count = _circular_strong_origin_count(candidate_genome_nt, functional_motif)
    uniqueness = 1.0 / max(1, strong_site_count)
    reward = float(a_match_integrity) * best_motif_score * best_position_score * uniqueness
    return GeneAOriginScore(
        reward=max(0.0, min(1.0, reward)),
        motif_score=best_motif_score,
        position_score=best_position_score,
        exact_functional_site=best_exact,
        strong_site_count=strong_site_count,
    )


def write_reference_protein_fasta(reference_gff: str | Path, output_fasta: str | Path) -> tuple[str, ...]:
    """Translate a coordinate-normalized embedded-FASTA GFF in circular locus order."""
    lines = Path(reference_gff).read_text().splitlines()
    if "##FASTA" not in lines:
        raise ValueError(f"Reference GFF must contain an embedded FASTA: {reference_gff}")
    fasta_index = lines.index("##FASTA")
    sequences: dict[str, str] = {}
    sequence_id: str | None = None
    for line in lines[fasta_index + 1 :]:
        if line.startswith(">"):
            sequence_id = line[1:].split()[0]
            sequences[sequence_id] = ""
        elif sequence_id is not None:
            sequences[sequence_id] += line.strip()

    features = []
    for line in lines[:fasta_index]:
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 9 or fields[2] != "CDS":
            continue
        locus = re.search(r"(?:^|;)ID=([^;]+)", fields[8])
        if locus is None or fields[0] not in sequences:
            raise ValueError(f"Cannot resolve reference CDS sequence or ID: {line}")
        features.append((int(fields[3]), int(fields[4]), locus.group(1), fields[0], fields[6]))
    features.sort(key=lambda feature: (feature[0], feature[1], feature[2]))
    if not features or len({feature[2] for feature in features}) != len(features):
        raise ValueError("Reference GFF must contain uniquely named CDS features")

    records = []
    for start, end, locus, seq_id, strand in features:
        coding_sequence = Seq(sequences[seq_id][start - 1 : end])
        if strand == "-":
            coding_sequence = coding_sequence.reverse_complement()
        if len(coding_sequence) % 3:
            raise ValueError(f"Reference CDS {locus} is not codon aligned")
        protein = str(coding_sequence.translate())
        if "*" in protein[:-1]:
            raise ValueError(f"Reference CDS {locus} contains an internal stop")
        records.append(SeqRecord(Seq(protein.removesuffix("*")), id=locus, description=""))
    SeqIO.write(records, output_fasta, "fasta")
    return tuple(record.id for record in records)


def load_candidate_orf_context(
    nucleotide_orfs_fasta: str | Path,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Load complete called-ORF sequences and their coordinate order by genome."""
    sequences: dict[str, str] = {}
    ordered: dict[str, list[tuple[int, int, str]]] = {}
    for record in SeqIO.parse(nucleotide_orfs_fasta, "fasta"):
        genome_id, separator, _ = record.id.rpartition("_ORF.")
        interval = ORFIPY_INTERVAL_RE.search(record.description)
        if not separator or interval is None:
            raise ValueError(f"Cannot resolve ORFipy source interval for {record.id}")
        start, end = int(interval.group(1)), int(interval.group(2))
        sequences[record.id] = str(record.seq).upper()
        ordered.setdefault(genome_id, []).append((start, end, record.id))
    orders = {
        genome_id: tuple(record_id for _start, _end, record_id in sorted(features))
        for genome_id, features in ordered.items()
    }
    return sequences, orders


def summarize_smooth_reference_evidence(
    hits_df: pd.DataFrame,
    *,
    genome_sequences: dict[str, str],
    candidate_orf_sequences: dict[str, str],
    candidate_orders: dict[str, tuple[str, ...]],
    reference_order: tuple[str, ...],
    synteny_match_parameters: dict[str, float],
    tropism_match_parameters: dict[str, float],
    synteny_order_weight: float,
    synteny_duplicate_penalty_weight: float,
    gene_a_reference_locus: str,
    tropism_reference_locus: str,
    gene_a_origin_motif: str,
    gene_a_origin_offset_nt: int,
    gene_a_origin_offset_tolerance_nt: int,
) -> pd.DataFrame:
    """Summarize one permissive reference-to-called-ORF search as graded objectives."""
    _validate_smooth_protein_match_config(**synteny_match_parameters)
    _validate_smooth_protein_match_config(**tropism_match_parameters)
    required_columns = {"query", "target", "evalue", "pident", "alnlen", "qlen", "tlen"}
    if not hits_df.empty and not required_columns.issubset(hits_df.columns):
        raise ValueError(f"Smooth reference hits are missing columns: {sorted(required_columns - set(hits_df))}")

    synteny_edges: dict[str, dict[tuple[str, str], float]] = {}
    tropism_edges: dict[str, dict[str, float]] = {}
    for hit in hits_df.itertuples(index=False):
        reference, candidate = str(hit.query), str(hit.target)
        genome_id, separator, _ = candidate.rpartition("_ORF.")
        if (
            not separator
            or genome_id not in genome_sequences
            or candidate not in candidate_orf_sequences
            or reference not in reference_order
        ):
            continue
        evidence = {
            "percent_identity": hit.pident,
            "e_value": hit.evalue,
            "alignment_length": hit.alnlen,
            "reference_length": hit.qlen,
            "candidate_length": hit.tlen,
        }
        synteny_integrity = smooth_protein_match_integrity(**evidence, **synteny_match_parameters)
        edge = (reference, candidate)
        synteny_edges.setdefault(genome_id, {})[edge] = max(
            synteny_integrity,
            synteny_edges.get(genome_id, {}).get(edge, 0.0),
        )
        if reference == tropism_reference_locus:
            tropism_integrity = smooth_protein_match_integrity(**evidence, **tropism_match_parameters)
            tropism_edges.setdefault(genome_id, {})[candidate] = max(
                tropism_integrity,
                tropism_edges.get(genome_id, {}).get(candidate, 0.0),
            )

    output_columns = [
        "id_prompt",
        "reward_external_synteny",
        "synteny_smooth_content_score",
        "synteny_smooth_ordered_score",
        "synteny_smooth_duplicate_score",
        "smooth_reference_matched_loci",
        "smooth_reference_best_integrity",
        "reward_external_tropism",
        "reward_gene_a_origin",
        "gene_a_origin_motif_score",
        "gene_a_origin_position_score",
        "gene_a_origin_exact_functional_site",
        "gene_a_origin_strong_site_count",
    ]
    rows = []
    for genome_id, genome_sequence in genome_sequences.items():
        edges = synteny_edges.get(genome_id, {})
        architecture = score_smooth_reference_architecture(
            edges,
            reference_order=reference_order,
            candidate_order=candidate_orders.get(genome_id, ()),
            order_weight=synteny_order_weight,
            duplicate_penalty_weight=synteny_duplicate_penalty_weight,
        )
        assignment = dict(architecture.assignment)
        a_candidate = assignment.get(gene_a_reference_locus)
        a_integrity = edges.get((gene_a_reference_locus, a_candidate), 0.0) if a_candidate else 0.0
        origin = score_gene_a_origin(
            candidate_a_orf_nt=candidate_orf_sequences.get(a_candidate, "") if a_candidate else "",
            candidate_genome_nt=genome_sequence,
            a_match_integrity=a_integrity,
            motif=gene_a_origin_motif,
            expected_offset_nt=gene_a_origin_offset_nt,
            offset_tolerance_nt=gene_a_origin_offset_tolerance_nt,
        )
        rows.append(
            {
                "id_prompt": genome_id,
                "reward_external_synteny": architecture.reward,
                "synteny_smooth_content_score": architecture.content_score,
                "synteny_smooth_ordered_score": architecture.ordered_score,
                "synteny_smooth_duplicate_score": architecture.duplicate_score,
                "smooth_reference_matched_loci": len(architecture.assignment),
                "smooth_reference_best_integrity": max(edges.values(), default=0.0),
                "reward_external_tropism": max(tropism_edges.get(genome_id, {}).values(), default=0.0),
                "reward_gene_a_origin": origin.reward,
                "gene_a_origin_motif_score": origin.motif_score,
                "gene_a_origin_position_score": origin.position_score,
                "gene_a_origin_exact_functional_site": float(origin.exact_functional_site),
                "gene_a_origin_strong_site_count": origin.strong_site_count,
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def remove_pseudocircular_extension_orfs(
    source_fasta: str | Path,
    nucleotide_orfs_fasta: str | Path,
    protein_orfs_fasta: str | Path,
) -> None:
    """Remove extension-only calls and native-prefix tails repeated by circular ORFs."""
    source_sequences = {record.id: str(record.seq).upper() for record in SeqIO.parse(source_fasta, "fasta")}
    if not source_sequences:
        raise ValueError(f"No source genomes found in {source_fasta}")
    source_lengths = {record_id: len(sequence) for record_id, sequence in source_sequences.items()}

    nucleotide_orfs_fasta = Path(nucleotide_orfs_fasta)
    protein_orfs_fasta = Path(protein_orfs_fasta)
    nucleotide_records = list(SeqIO.parse(nucleotide_orfs_fasta, "fasta"))
    protein_records = list(SeqIO.parse(protein_orfs_fasta, "fasta"))
    if [record.id for record in nucleotide_records] != [record.id for record in protein_records]:
        raise ValueError("ORFipy nucleotide and protein FASTAs contain different record IDs")

    intervals: dict[str, tuple[str, int, int]] = {}
    keep_ids: set[str] = set()
    for record in nucleotide_records:
        genome_id, separator, _ = record.id.rpartition("_ORF.")
        interval = ORFIPY_INTERVAL_RE.search(record.description)
        if not separator or genome_id not in source_lengths or interval is None:
            raise ValueError(f"Cannot resolve ORFipy source interval for {record.id}")
        start = int(interval.group(1))
        end = int(interval.group(2))
        intervals[record.id] = (genome_id, start, end)
        if start < source_lengths[genome_id]:
            keep_ids.add(record.id)

    protein_by_id = {record.id: str(record.seq) for record in protein_records}
    extension_lengths: dict[str, int] = {}
    for genome_id, sequence in source_sequences.items():
        first_stops = []
        for frame in range(3):
            framed = sequence[frame:]
            for offset in range(0, len(framed) - 2, 3):
                if framed[offset : offset + 3] in {"TAA", "TAG", "TGA"}:
                    first_stops.append(offset + frame + 3)
                    break
        extension_lengths[genome_id] = max(first_stops) if first_stops else len(sequence)

    for record_id in tuple(keep_ids):
        genome_id, _start, end = intervals[record_id]
        if end > extension_lengths[genome_id]:
            continue
        prefix_protein = protein_by_id[record_id]
        if any(
            other_id != record_id
            and other_genome == genome_id
            and other_end > source_lengths[genome_id]
            and len(protein_by_id[other_id]) > len(prefix_protein)
            and protein_by_id[other_id].endswith(prefix_protein)
            for other_id, (other_genome, _other_start, other_end) in intervals.items()
        ):
            keep_ids.remove(record_id)

    for path, records in (
        (nucleotide_orfs_fasta, nucleotide_records),
        (protein_orfs_fasta, protein_records),
    ):
        temporary = path.with_name(f"{path.name}.tmp")
        SeqIO.write((record for record in records if record.id in keep_ids), temporary, "fasta")
        temporary.replace(path)


def _gff_cds_annotations(gff_path: Path) -> dict[str, str]:
    annotations = {}
    if not gff_path.exists():
        return annotations
    for line in gff_path.read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 9 or fields[2] != "CDS":
            continue
        gene_id = re.search(r"(?:^|;)ID=([^;]+)", fields[8])
        product = re.search(r"(?:^|;)product=([^;]+)", fields[8])
        if gene_id:
            annotations[gene_id.group(1)] = product.group(1) if product else "Unknown gene"
    return annotations


def _gff_cds_order(gff_path: Path) -> list[str]:
    features = []
    if not gff_path.exists():
        return []
    for line in gff_path.read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 9 or fields[2] != "CDS":
            continue
        gene_id = re.search(r"(?:^|;)ID=([^;]+)", fields[8])
        if gene_id:
            features.append((int(fields[3]), int(fields[4]), gene_id.group(1)))
    return [gene_id for _, _, gene_id in sorted(features)]


def stage_coordinate_normalized_reference_gff(
    source: str | Path,
    destination: str | Path,
    circular_genome_length: int | None = None,
) -> None:
    """Stage intact 1-based CDSs without redundant pseudocircular prefix ORFs."""
    source = Path(source)
    destination = Path(destination)
    lines = source.read_text().splitlines()
    staged_lines: list[str] = []
    cds_features: list[tuple[int, str, int, int, str]] = []
    in_fasta = False
    for line in lines:
        staged_line = line
        if line == "##FASTA":
            in_fasta = True
        if not in_fasta and line and not line.startswith("#"):
            fields = line.split("\t")
            if len(fields) >= 9 and fields[2] == "CDS":
                start, end = int(fields[3]), int(fields[4])
                if (end - start + 1) % 3 != 0:
                    if (end - start) % 3 != 0:
                        raise ValueError(
                            f"Reference CDS is not codon-aligned before or after +1 normalization: {line}"
                        )
                    start += 1
                    fields[3] = str(start)
                    staged_line = "\t".join(fields)
                cds_features.append((len(staged_lines), fields[0], start, end, fields[6]))
        staged_lines.append(staged_line)

    sequences: dict[str, str] = {}
    sequence_id: str | None = None
    for line in lines[lines.index("##FASTA") + 1 :] if "##FASTA" in lines else []:
        if line.startswith(">"):
            sequence_id = line[1:].split()[0]
            sequences[sequence_id] = ""
        elif sequence_id is not None:
            sequences[sequence_id] += line.strip()
    if not cds_features or not sequences:
        raise ValueError(f"Reference GFF must contain CDS features and embedded FASTA: {source}")

    proteins: dict[int, str] = {}
    for line_index, seq_id, start, end, strand in cds_features:
        coding_sequence = Seq(sequences[seq_id][start - 1 : end])
        if strand == "-":
            coding_sequence = coding_sequence.reverse_complement()
        protein = str(coding_sequence.translate())
        if len(coding_sequence) % 3 != 0 or "*" in protein[:-1]:
            raise ValueError(f"Reference CDS {seq_id}:{start}-{end} does not encode an intact protein")
        proteins[line_index] = protein

    redundant_prefix_features: set[int] = set()
    if circular_genome_length is not None:
        for line_index, seq_id, _start, end, strand in cds_features:
            extension_length = len(sequences[seq_id]) - int(circular_genome_length)
            if extension_length <= 0 or end > extension_length:
                continue
            prefix_protein = proteins[line_index]
            for other_index, other_seq_id, other_start, other_end, other_strand in cds_features:
                if (
                    other_seq_id == seq_id
                    and other_strand == strand
                    and other_end > int(circular_genome_length)
                    and other_start > extension_length
                    and len(proteins[other_index]) > len(prefix_protein)
                    and (
                        proteins[other_index].startswith(prefix_protein)
                        or proteins[other_index].endswith(prefix_protein)
                    )
                ):
                    redundant_prefix_features.add(line_index)
                    break

    destination.write_text(
        "\n".join(line for index, line in enumerate(staged_lines) if index not in redundant_prefix_features) + "\n"
    )


def _circular_order_violation_count(observed: list[str], reference_order: list[str]) -> int:
    """Count directed circular adjacencies that disagree with reference order."""
    if len(observed) < 3:
        return 0
    observed_set = set(observed)
    expected = [reference for reference in reference_order if reference in observed_set]
    if len(expected) != len(observed):
        return len(observed)
    expected_edges = {(expected[index], expected[(index + 1) % len(expected)]) for index in range(len(expected))}
    return sum(
        (observed[index], observed[(index + 1) % len(observed)]) not in expected_edges
        for index in range(len(observed))
    )


def _maximum_reference_matching(candidate_to_references: dict[str, set[str]]) -> dict[str, str]:
    """Return a deterministic one-candidate-per-reference matching."""
    reference_to_candidate: dict[str, str] = {}

    def assign(candidate: str, visited: set[str]) -> bool:
        for reference in sorted(candidate_to_references[candidate]):
            if reference in visited:
                continue
            visited.add(reference)
            incumbent = reference_to_candidate.get(reference)
            if incumbent is None or assign(incumbent, visited):
                reference_to_candidate[reference] = candidate
                return True
        return False

    for candidate in sorted(candidate_to_references):
        assign(candidate, set())
    return reference_to_candidate


def measure_reference_cluster_architecture(
    root_dir: str | Path,
    gff_dir: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
    reference_gff_path: str | Path | None = None,
) -> None:
    """Measure distinct reference loci and excess homolog copies in LoVis4u clusters."""
    root_dir = Path(root_dir)
    gff_dir = Path(gff_dir)
    input_df = pd.read_csv(input_csv)
    metric_columns = [
        "num_syntenic_genes",
        "reference_num_genes",
        "duplicate_reference_gene_count",
        "reference_order_violation_count",
        "non_syntenic_genes",
        "non_syntenic_annotations",
        "missing_synteny_output",
    ]
    input_df = input_df.drop(columns=[column for column in metric_columns if column in input_df])
    if input_df.empty:
        for column in metric_columns:
            input_df[column] = pd.Series(dtype="bool" if column == "missing_synteny_output" else "object")
        input_df.to_csv(output_csv, index=False)
        return
    if "genome_id" not in input_df:
        raise KeyError("Reference-cluster input must contain genome_id")
    reference_annotations = _gff_cds_annotations(Path(reference_gff_path)) if reference_gff_path is not None else {}
    reference_order = _gff_cds_order(Path(reference_gff_path)) if reference_gff_path is not None else []
    reference_gene_count = len(reference_annotations) if reference_gff_path is not None else None
    rows = []
    for sequence in input_df.itertuples(index=False):
        genome_id = str(sequence.genome_id)
        cluster_path = root_dir / genome_id / "mmseqs" / "mmseqs_clustering.tsv"
        gff_path = gff_dir / f"{genome_id}.gff"
        if not cluster_path.exists() or cluster_path.stat().st_size == 0 or not gff_path.exists():
            rows.append(
                {
                    "genome_id": genome_id,
                    "num_syntenic_genes": 0,
                    "reference_num_genes": reference_gene_count or 0,
                    "duplicate_reference_gene_count": 0,
                    "reference_order_violation_count": 0,
                    "non_syntenic_genes": "",
                    "non_syntenic_annotations": "",
                    "missing_synteny_output": True,
                }
            )
            continue

        clusters = pd.read_csv(cluster_path, sep="\t", header=None, names=["representative", "member"])
        adjacency: dict[str, set[str]] = {}
        for raw_left, raw_right in clusters.itertuples(index=False, name=None):
            left, right = str(raw_left), str(raw_right)
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)

        candidate_prefix = f"{genome_id}-"
        candidate_to_references: dict[str, set[str]] = {}
        reference_ids: set[str] = set()
        visited: set[str] = set()
        for node in adjacency:
            if node in visited:
                continue
            component = set()
            stack = [node]
            while stack:
                member = stack.pop()
                if member in component:
                    continue
                component.add(member)
                stack.extend(adjacency.get(member, ()))
            visited.update(component)
            candidates = {member for member in component if member.startswith(candidate_prefix)}
            references = component - candidates
            if reference_gff_path is not None:
                references.intersection_update(reference_annotations)
            reference_ids.update(references)
            for candidate in candidates:
                candidate_to_references.setdefault(candidate, set()).update(references)

        reference_to_candidate = _maximum_reference_matching(candidate_to_references)
        matched_candidates = set(reference_to_candidate.values())
        homologous_candidates = {candidate for candidate, refs in candidate_to_references.items() if refs}
        annotations = _gff_cds_annotations(gff_path)
        matched_gene_ids = {
            match.group(0)
            for candidate in matched_candidates
            if (match := re.search(r"ORF\.\d+", candidate)) is not None
        }
        candidate_order = {gene_id: index for index, gene_id in enumerate(_gff_cds_order(gff_path))}
        ordered_reference_ids = []
        unplaced_matches = 0
        for reference, candidate in reference_to_candidate.items():
            candidate_gene_id = candidate.removeprefix(candidate_prefix)
            if candidate_gene_id not in candidate_order:
                unplaced_matches += 1
                continue
            ordered_reference_ids.append((candidate_order[candidate_gene_id], reference))
        observed_reference_order = [reference for _, reference in sorted(ordered_reference_ids)]
        order_violation_count = (
            unplaced_matches + _circular_order_violation_count(observed_reference_order, reference_order)
            if reference_order
            else 0
        )
        non_syntenic_genes = sorted(set(annotations) - matched_gene_ids)
        rows.append(
            {
                "genome_id": genome_id,
                "num_syntenic_genes": len(reference_to_candidate),
                "reference_num_genes": reference_gene_count or len(reference_ids),
                "duplicate_reference_gene_count": max(0, len(homologous_candidates) - len(matched_candidates)),
                "reference_order_violation_count": order_violation_count,
                "non_syntenic_genes": ",".join(non_syntenic_genes),
                "non_syntenic_annotations": ",".join(annotations[gene] for gene in non_syntenic_genes),
                "missing_synteny_output": False,
            }
        )

    metrics_df = pd.DataFrame(rows)
    input_df.merge(metrics_df, on="genome_id", how="left").to_csv(output_csv, index=False)


def protein_alignment_integrity(
    percent_identity: object,
    alignment_length: object,
    query_length: object,
    target_length: object,
) -> float:
    """Return minimum reciprocal coverage for a measured protein hit."""
    try:
        identity, aligned, query, target = map(
            float,
            (percent_identity, alignment_length, query_length, target_length),
        )
    except (TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(value) for value in (identity, aligned, query, target)):
        return 0.0
    if aligned <= 0.0 or query <= 0.0 or target <= 0.0:
        return 0.0
    if not 0.0 <= identity <= 100.0:
        return 0.0
    return max(0.0, min(1.0, aligned / query, aligned / target))


def add_protein_alignment_evidence(hits_df: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, bool]:
    """Add reciprocal coverage and alignment-integrity columns to MMseqs protein hits."""
    hits_df = hits_df.copy()
    identity = f"{prefix}_mmseqs_percent_identity"
    aligned = f"{prefix}_mmseqs_alignment_length"
    query = f"{prefix}_mmseqs_query_length"
    target = f"{prefix}_mmseqs_target_length"
    query_coverage = f"{prefix}_mmseqs_query_coverage"
    target_coverage = f"{prefix}_mmseqs_target_coverage"
    reciprocal_coverage = f"{prefix}_min_reciprocal_coverage"
    integrity = f"{prefix}_alignment_integrity"
    if not {identity, aligned, query, target}.issubset(hits_df.columns):
        for column in (query_coverage, target_coverage, reciprocal_coverage, integrity):
            hits_df[column] = 0.0
        return hits_df, False

    for column in (identity, aligned, query, target):
        hits_df[column] = pd.to_numeric(hits_df[column], errors="coerce")
    numeric_evidence = hits_df[[identity, aligned, query, target]]
    finite = numeric_evidence.notna().all(axis=1) & numeric_evidence.abs().lt(math.inf).all(axis=1)
    valid = (
        finite
        & hits_df[identity].between(0.0, 100.0)
        & (hits_df[aligned] > 0)
        & (hits_df[query] > 0)
        & (hits_df[target] > 0)
    )
    hits_df[query_coverage] = (hits_df[aligned] / hits_df[query]).where(valid, 0.0).clip(0.0, 1.0)
    hits_df[target_coverage] = (hits_df[aligned] / hits_df[target]).where(valid, 0.0).clip(0.0, 1.0)
    hits_df[reciprocal_coverage] = hits_df[[query_coverage, target_coverage]].min(axis=1)
    hits_df[integrity] = [
        protein_alignment_integrity(*values)
        for values in zip(hits_df[identity], hits_df[aligned], hits_df[query], hits_df[target], strict=False)
    ]
    return hits_df, True


def _full_length_hits(hits_df: pd.DataFrame, prefix: str, minimum_coverage: float) -> pd.DataFrame:
    if not 0.0 <= float(minimum_coverage) <= 1.0:
        raise ValueError("minimum_coverage must be between zero and one")
    hits_df, available = add_protein_alignment_evidence(hits_df, prefix)
    if not available:
        return hits_df.iloc[0:0].copy()
    return hits_df.loc[
        (hits_df[f"{prefix}_mmseqs_query_coverage"] >= minimum_coverage)
        & (hits_df[f"{prefix}_mmseqs_target_coverage"] >= minimum_coverage)
    ].copy()


def valid_coverage_aware_protein_database_hit_count(
    hits_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    id_column: str = "id_prompt",
    min_hits: int = 7,
    minimum_reciprocal_coverage: float = 0.75,
) -> pd.DataFrame:
    """Keep genomes with enough unique, reciprocally full-length target families."""
    hits_df = _full_length_hits(hits_df, "protein_database", minimum_reciprocal_coverage)
    target = "protein_database_mmseqs_target"
    if id_column not in hits_df or target not in hits_df:
        result = sequences_df.iloc[0:0].copy()
        result["protein_database_hit_count"] = pd.Series(dtype="int64")
        return result
    hits_df["_genome_id"] = hits_df[id_column].astype(str).str.rsplit("_", n=1).str[0]
    counts = hits_df.drop_duplicates(["_genome_id", target])["_genome_id"].value_counts()
    result = sequences_df.loc[sequences_df["id_prompt"].astype(str).isin(counts[counts >= min_hits].index)].copy()
    result["protein_database_hit_count"] = result["id_prompt"].astype(str).map(counts).fillna(0).astype(int)
    return result


def valid_coverage_aware_mmseqs_pident(
    hits_df: pd.DataFrame,
    prefix: str,
    pident_range: tuple,
    sequences_df: pd.DataFrame,
    minimum_reciprocal_coverage: float = 0.95,
) -> pd.DataFrame:
    """Apply a protein-identity gate only to reciprocally full-length hits."""
    hits_df = _full_length_hits(hits_df, prefix, minimum_reciprocal_coverage)
    pident = f"{prefix}_mmseqs_percent_identity"
    result = sequences_df.copy()
    if pident not in hits_df:
        result[pident] = 0.0
        return result.iloc[0:0].copy()
    hits_df[pident] = pd.to_numeric(hits_df[pident], errors="coerce").fillna(0.0)
    hits_df["_genome_id"] = hits_df["id_prompt"].astype(str).str.rsplit("_", n=1).str[0]
    result[pident] = result["id_prompt"].astype(str).map(hits_df.groupby("_genome_id")[pident].max()).fillna(0.0)
    return result.loc[result[pident].between(min(pident_range), max(pident_range))].copy()


def summarize_full_length_aai(hits_df: pd.DataFrame, minimum_reciprocal_coverage: float = 0.75) -> pd.DataFrame:
    """Average identity over one best reciprocally full-length hit per target family."""
    hits_df = _full_length_hits(hits_df, "protein_database", minimum_reciprocal_coverage)
    target = "protein_database_mmseqs_target"
    pident = "protein_database_mmseqs_percent_identity"
    output_columns = ["id_prompt", "average_protein_percent_identity", "average_protein_identity_gene_count"]
    if hits_df.empty or not {"id_prompt", target, pident}.issubset(hits_df.columns):
        return pd.DataFrame(columns=output_columns)
    hits_df["_genome_id"] = hits_df["id_prompt"].astype(str).str.rsplit("_", n=1).str[0]
    best_hits = hits_df.sort_values(
        ["_genome_id", target, pident, "protein_database_alignment_integrity", "id_prompt"],
        ascending=[True, True, False, False, True],
    ).drop_duplicates(["_genome_id", target])
    return (
        best_hits.groupby("_genome_id")[pident].agg(["mean", "count"]).reset_index().set_axis(output_columns, axis=1)
    )


def summarize_required_gene_evidence(
    hits_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    required_products: tuple,
    minimum_reciprocal_coverage: float = 0.75,
) -> pd.DataFrame:
    """Assign required label copies one-to-one to distinct ORFs and target families."""
    required_products = tuple(map(str, required_products))
    hits_df, available = add_protein_alignment_evidence(hits_df, "protein_database")
    target_column = "protein_database_mmseqs_target"
    available = available and {"id_prompt", "annot", target_column}.issubset(hits_df.columns)
    if available:
        hits_df["_genome_id"] = hits_df["id_prompt"].astype(str).str.rsplit("_", n=1).str[0]
        hits_df["annot"] = hits_df["annot"].astype(str)
        hits_df[target_column] = hits_df[target_column].astype(str)

    rows = []
    for sequence in sequences_df.itertuples(index=False):
        genome_hits = hits_df.loc[hits_df["_genome_id"] == str(sequence.id_prompt)] if available else pd.DataFrame()
        integrities = []
        full_length_count = 0
        for product, required_copy_count in Counter(required_products).items():
            family = genome_hits.loc[genome_hits["annot"] == product] if not genome_hits.empty else pd.DataFrame()
            assigned_integrities: list[float] = []
            if len(family):
                candidate_to_targets = family.groupby("id_prompt")[target_column].agg(set).map(set).to_dict()
                target_to_candidate = _maximum_reference_matching(candidate_to_targets)
                assigned_integrities = [
                    float(
                        family.loc[
                            (family["id_prompt"] == candidate) & (family[target_column] == target),
                            "protein_database_alignment_integrity",
                        ].max()
                    )
                    for target, candidate in target_to_candidate.items()
                ]
                integrities.extend(sorted(assigned_integrities, reverse=True)[:required_copy_count])

                full_family = family.loc[
                    (family["protein_database_mmseqs_query_coverage"] >= minimum_reciprocal_coverage)
                    & (family["protein_database_mmseqs_target_coverage"] >= minimum_reciprocal_coverage)
                ]
                full_candidate_to_targets = full_family.groupby("id_prompt")[target_column].agg(set).map(set).to_dict()
                full_length_count += min(
                    required_copy_count,
                    len(_maximum_reference_matching(full_candidate_to_targets)),
                )
            assigned_copy_count = min(required_copy_count, len(assigned_integrities))
            missing_copy_count = required_copy_count - assigned_copy_count
            integrities.extend([0.0] * missing_copy_count)
        rows.append(
            {
                "id_prompt": str(sequence.id_prompt),
                "genome_id": str(sequence.genome_id),
                "required_genes_matched_count": sum(value > 0.0 for value in integrities),
                "required_genes_total_count": len(required_products),
                "required_genes_integrity_sum": sum(integrities),
                "required_genes_full_length_count": full_length_count,
                "required_genes_alignment_evidence_available": available,
            }
        )
    return pd.DataFrame(rows)
