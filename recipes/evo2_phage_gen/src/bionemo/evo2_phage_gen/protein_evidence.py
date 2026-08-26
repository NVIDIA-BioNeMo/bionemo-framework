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
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


ORFIPY_INTERVAL_RE = re.compile(r"\[(\d+)-(\d+)\]")


def remove_pseudocircular_extension_orfs(
    source_fasta: str | Path,
    nucleotide_orfs_fasta: str | Path,
    protein_orfs_fasta: str | Path,
) -> None:
    """Remove ORFipy calls that start wholly inside a pseudocircular extension."""
    source_lengths = {record.id: len(record.seq) for record in SeqIO.parse(source_fasta, "fasta")}
    if not source_lengths:
        raise ValueError(f"No source genomes found in {source_fasta}")

    nucleotide_orfs_fasta = Path(nucleotide_orfs_fasta)
    protein_orfs_fasta = Path(protein_orfs_fasta)
    nucleotide_records = list(SeqIO.parse(nucleotide_orfs_fasta, "fasta"))
    protein_records = list(SeqIO.parse(protein_orfs_fasta, "fasta"))
    if [record.id for record in nucleotide_records] != [record.id for record in protein_records]:
        raise ValueError("ORFipy nucleotide and protein FASTAs contain different record IDs")

    keep_ids: set[str] = set()
    for record in nucleotide_records:
        genome_id, separator, _ = record.id.rpartition("_ORF.")
        interval = ORFIPY_INTERVAL_RE.search(record.description)
        if not separator or genome_id not in source_lengths or interval is None:
            raise ValueError(f"Cannot resolve ORFipy source interval for {record.id}")
        start = int(interval.group(1))
        if start < source_lengths[genome_id]:
            keep_ids.add(record.id)

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
