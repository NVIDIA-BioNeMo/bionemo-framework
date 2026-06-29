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

"""UniProt annotation parsing utilities.

Parses UniProt TSV exports to extract position-level annotation labels
for SAE interpretability evaluation.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Mapping from normalized UniProt TSV column names to annotation type codes.
# Column names are normalized via: str.lower().str.replace(' ', '_')
# e.g. "Active site" -> "active_site", "Domain [FT]" -> "domain_[ft]"
FEATURE_COLUMNS = {
    "active_site": "ACT_SITE",
    "binding_site": "BINDING",
    "disulfide_bond": "DISULFID",
    "glycosylation": "CARBOHYD",
    "lipidation": "LIPID",
    "modified_residue": "MOD_RES",
    "signal_peptide": "SIGNAL",
    "transit_peptide": "TRANSIT",
    "helix": "HELIX",
    "turn": "TURN",
    "beta_strand": "STRAND",
    "coiled_coil": "COILED",
    "compositional_bias": "COMPBIAS",
    "domain_[ft]": "DOMAIN",
    "motif": "MOTIF",
    "region": "REGION",
    "zinc_finger": "ZN_FING",
}


@dataclass
class AnnotatedProtein:
    """Protein with parsed annotations."""

    accession: str
    sequence: str
    annotations: Dict[str, np.ndarray]  # concept_name -> binary array


def download_annotated_proteins(
    output_path: Union[str, Path],
    organism: Optional[int] = None,
    max_length: int = 512,
    reviewed_only: bool = True,
    annotation_score: Optional[int] = None,
    max_results: Optional[int] = None,
) -> Path:
    """Download annotated proteins from UniProt REST API.

    Args:
        output_path: Path to save the TSV file.
        organism: NCBI taxonomy ID (e.g., 9606 for human, 10090 for mouse).
        max_length: Maximum sequence length.
        reviewed_only: Only include Swiss-Prot (reviewed) entries.
        annotation_score: Minimum annotation score (1-5, 5 is best).
        max_results: Limit number of results (None for all).

    Returns:
        Path to downloaded file.
    """
    import requests

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build query
    query_parts = []
    if reviewed_only:
        query_parts.append("(reviewed:true)")
    if organism:
        query_parts.append(f"(model_organism:{organism})")
    if annotation_score:
        query_parts.append(f"(annotation_score:{annotation_score})")
    query_parts.append(f"(length:[1 TO {max_length}])")

    query = " AND ".join(query_parts)

    # Fields to request
    fields = [
        "accession",
        "sequence",
        "length",
        "ft_act_site",
        "ft_binding",
        "ft_disulfid",
        "ft_carbohyd",
        "ft_lipid",
        "ft_mod_res",
        "ft_signal",
        "ft_transit",
        "ft_helix",
        "ft_turn",
        "ft_strand",
        "ft_coiled",
        "ft_compbias",
        "ft_domain",
        "ft_motif",
        "ft_region",
        "ft_zn_fing",
    ]

    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "fields": ",".join(fields),
        "format": "tsv",
        "compressed": "true",
    }
    if max_results:
        params["size"] = max_results

    print(f"Downloading from UniProt: {query}")
    response = requests.get(url, params=params, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded: {output_path}")
    return output_path


def parse_position(pos_str: str) -> Optional[Tuple[int, int]]:
    """Parse a UniProt position string to (start, end) tuple (0-indexed).

    Handles formats:
        - "45" -> (44, 45)
        - "45..120" -> (44, 120)
        - "<45..120" -> (44, 120)  (uncertain start)
        - "45..>120" -> (44, 120)  (uncertain end)

    Returns None for ambiguous positions (containing ? or :).
    """
    if not pos_str or "?" in pos_str or ":" in pos_str:
        return None

    pos_str = pos_str.strip()

    if ".." in pos_str:
        parts = pos_str.split("..")
        start = parts[0].strip("<").strip()
        end = parts[1].strip(">").strip()
        try:
            return (int(start) - 1, int(end))  # 0-indexed start, exclusive end
        except ValueError:
            return None
    else:
        # Single position
        pos_str = pos_str.strip("<>").strip()
        try:
            pos = int(pos_str)
            return (pos - 1, pos)
        except ValueError:
            return None


def parse_annotation_field(
    field_value: str,
    seq_length: int,
    domain_counter: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Parse a UniProt annotation field into position-level arrays.

    UniProt REST API TSV format separates annotations with '; ' and puts
    qualifiers (/note, /evidence) as separate '; '-delimited entries:
        DOMAIN 133..266; /note="TIR"; /evidence="ECO:..."; DOMAIN 31..759; /note="GH81"

    This parser groups qualifiers with their parent TYPE+position entry.

    Args:
        field_value: Raw annotation string from TSV (may contain multiple annotations).
        seq_length: Length of the protein sequence.
        domain_counter: If provided, each annotation span gets the next integer
            for its concept (globally unique domain IDs). If None, binary 1.0 labels.

    Returns:
        Dict mapping concept names to arrays of shape (seq_length,).
        Values are binary (1.0) when domain_counter is None, or incrementing
        domain-instance IDs when domain_counter is provided.
    """
    if pd.isna(field_value) or not field_value.strip():
        return {}

    results = {}

    # Split on '; ' and group: a TYPE+position entry starts a new annotation,
    # subsequent /qualifier entries belong to the previous annotation.
    parts = field_value.split("; ")

    # Group into annotations: each starts with TYPE+position, followed by qualifiers
    annotations = []  # list of (type_pos_str, [qualifier_strs])
    for part in parts:
        part = part.strip().rstrip(";")
        if not part:
            continue
        if part.startswith("/"):
            # Qualifier — attach to previous annotation
            if annotations:
                annotations[-1][1].append(part)
        else:
            # New annotation (TYPE + position)
            annotations.append((part, []))

    for type_pos, qualifiers in annotations:
        # Split into type and position
        tokens = type_pos.split(None, 1)
        if len(tokens) < 2:
            continue

        ann_type = tokens[0]
        pos_str = tokens[1]

        # Parse position
        pos = parse_position(pos_str)
        if pos is None:
            continue

        start, end = pos
        if start < 0 or end > seq_length:
            continue

        # Extract note from qualifiers
        note = None
        for q in qualifiers:
            note_match = re.search(r'/note="([^"]*)"', q)
            if note_match:
                note = note_match.group(1)
                break

        # Create concept name
        if note:
            concept = f"{ann_type}:{note}"
        else:
            concept = ann_type

        # Create or update array
        if concept not in results:
            results[concept] = np.zeros(seq_length, dtype=np.float32)

        if domain_counter is not None:
            domain_counter[concept] = domain_counter.get(concept, 0) + 1
            results[concept][start:end] = domain_counter[concept]
        else:
            results[concept][start:end] = 1.0

    return results


def load_annotations_tsv(
    tsv_path: Union[str, Path],
    min_positives: int = 10,
    max_proteins: Optional[int] = None,
    use_domain_ids: bool = False,
) -> Tuple[List[AnnotatedProtein], Dict[str, int]]:
    """Load and parse UniProt annotations from TSV file.

    Args:
        tsv_path: Path to TSV file (can be gzipped).
        min_positives: Minimum total positive positions for a concept to be kept.
        max_proteins: Maximum number of proteins to load.
        use_domain_ids: If True, each annotation span gets a globally unique
            incrementing integer ID instead of binary 1.0. This enables
            domain-level recall computation.

    Returns:
        Tuple of (list of AnnotatedProtein, dict of concept -> total_positives).
    """
    tsv_path = Path(tsv_path)

    # Read TSV
    if str(tsv_path).endswith(".gz"):
        df = pd.read_csv(tsv_path, sep="\t", compression="gzip")
    else:
        df = pd.read_csv(tsv_path, sep="\t")

    if max_proteins:
        df = df.head(max_proteins)

    # Normalize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    proteins = []
    concept_counts = {}
    domain_counter = {} if use_domain_ids else None

    for _, row in df.iterrows():
        accession = row.get("accession", row.get("entry", ""))
        sequence = row.get("sequence", "")

        if not sequence:
            continue

        seq_len = len(sequence)
        all_annotations = {}

        # Parse each feature column
        for col, ann_type in FEATURE_COLUMNS.items():
            if col not in df.columns:
                continue

            field_value = row.get(col, "")
            parsed = parse_annotation_field(str(field_value), seq_len, domain_counter)

            for concept, arr in parsed.items():
                all_annotations[concept] = arr
                # Use (arr > 0).sum() so counting works for both binary and domain-ID arrays
                concept_counts[concept] = concept_counts.get(concept, 0) + int((arr > 0).sum())

        if all_annotations:
            proteins.append(
                AnnotatedProtein(
                    accession=accession,
                    sequence=sequence,
                    annotations=all_annotations,
                )
            )

    # Filter concepts by min_positives
    valid_concepts = {c for c, count in concept_counts.items() if count >= min_positives}

    for protein in proteins:
        protein.annotations = {c: arr for c, arr in protein.annotations.items() if c in valid_concepts}

    filtered_counts = {c: count for c, count in concept_counts.items() if c in valid_concepts}

    print(f"Loaded {len(proteins)} proteins with {len(filtered_counts)} concepts (min_positives={min_positives})")

    return proteins, filtered_counts


def proteins_to_concept_labels(proteins: List[AnnotatedProtein]) -> Tuple[List[str], List[Dict[str, np.ndarray]]]:
    """Convert AnnotatedProtein list to format expected by compute_f1_scores.

    Returns:
        Tuple of (sequences, concept_labels) where concept_labels[i] is a dict
        mapping concept names to binary arrays for sequence i.
    """
    sequences = [p.sequence for p in proteins]
    concept_labels = [p.annotations for p in proteins]
    return sequences, concept_labels
