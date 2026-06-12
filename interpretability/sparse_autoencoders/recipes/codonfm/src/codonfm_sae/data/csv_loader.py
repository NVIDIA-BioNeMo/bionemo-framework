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

"""CSV data loading for CodonFM codon sequences."""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .types import CodonRecord


SEQ_COLUMN_CANDIDATES = ["seq", "cds", "sequence", "dna_seq"]
ID_COLUMN_CANDIDATES = ["id", "seq_id", "gene", "name"]


def read_codon_csv(
    filepath: Union[str, Path],
    seq_column: Optional[str] = None,
    id_column: Optional[str] = None,
    max_sequences: Optional[int] = None,
    max_codons: Optional[int] = None,
    min_codons: Optional[int] = None,
) -> List[CodonRecord]:
    """Read codon sequences from a CSV file.

    Auto-detects column names: tries 'seq', 'cds', 'sequence' for the
    sequence column and 'id', 'seq_id', 'gene' for the ID column.

    Args:
        filepath: Path to CSV file.
        seq_column: Column name containing DNA sequences (auto-detect if None).
        id_column: Column name for sequence IDs (auto-detect if None).
        max_sequences: Maximum number of sequences to return.
        max_codons: Filter out sequences with more codons than this.
        min_codons: Filter out sequences with fewer codons than this.

    Returns:
        List of CodonRecord objects.
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath)

    # Auto-detect sequence column
    if seq_column is None:
        for candidate in SEQ_COLUMN_CANDIDATES:
            if candidate in df.columns:
                seq_column = candidate
                break
        if seq_column is None:
            raise ValueError(
                f"Cannot auto-detect sequence column in {filepath}. "
                f"Columns: {list(df.columns)}. Pass seq_column= explicitly."
            )

    # Auto-detect ID column
    if id_column is None:
        for candidate in ID_COLUMN_CANDIDATES:
            if candidate in df.columns:
                id_column = candidate
                break

    # Drop rows with missing sequences
    df = df.dropna(subset=[seq_column])

    # Filter by codon count
    codon_counts = df[seq_column].str.len() // 3
    if max_codons is not None:
        df = df[codon_counts <= max_codons]
        codon_counts = codon_counts[df.index]
    if min_codons is not None:
        df = df[codon_counts >= min_codons]

    if max_sequences is not None:
        df = df.head(max_sequences)

    # Metadata columns to carry through (if present)
    metadata_columns = [
        "var_pos_offset",
        "ref_codon",
        "alt_codon",
        "source",
        "ROLE_IN_CANCER",
        "MUTATION_DESCRIPTION",
        "is_pathogenic",
        "in_splice_junction",
        "phylop",
        "gene",
        "5b_cdwt",
        "5b",
        "1b_cdwt",
        "1b",
        "600m",
        "80m",
        "5b_avg",
        "trinuc_context",
        "gc_content",
    ]
    available_meta = [c for c in metadata_columns if c in df.columns]

    records = []
    for idx, row in df.iterrows():
        record_id = str(row[id_column]) if id_column else f"seq_{idx}"
        meta = {}
        for col in available_meta:
            val = row[col]
            if pd.isna(val):
                meta[col] = None
            else:
                meta[col] = val
        records.append(CodonRecord(id=record_id, sequence=str(row[seq_column]), metadata=meta))

    return records
