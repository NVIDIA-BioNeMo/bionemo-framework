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

"""Data utilities for biological sequence data.

Provides utilities for:
- Reading FASTA files
- Downloading from UniProt/SwissProt
- Creating datasets for protein sequences
"""

import gzip
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TextIO, Tuple, Union

from sae.utils import get_file_limit
from tqdm import tqdm

from .types import ProteinRecord


def shard_fasta(
    input_file: Path, output_dir: Path, proteins_per_shard: int = 1000, max_open_files: Optional[int] = None
) -> int:
    """Split a large FASTA file into smaller shards with a specified number of proteins per shard.

    This function processes large FASTA files in batches to respect system file handle
    limits. It creates numbered shard files in the specified output directory.

    Args:
        input_file: Path to the input FASTA file
        output_dir: Directory where shard files will be created
        proteins_per_shard: Number of proteins to include in each shard
        max_open_files: Maximum number of files to keep open simultaneously.
            If None, calculated based on system limits

    Returns:
        int: Total number of shards created
    """
    print(f"Proteins per shard: {proteins_per_shard}")

    # Configure file handling limits
    system_limit = get_file_limit()
    if max_open_files is None:
        max_open_files = max(100, min(system_limit - 100, 1000))

    print(f"System file limit: {system_limit}")
    print(f"Using max_open_files: {max_open_files}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Count proteins and calculate shards
    total_proteins = count_proteins_in_fasta(input_file)
    num_shards = (total_proteins + proteins_per_shard - 1) // proteins_per_shard
    print(f"Total proteins: {total_proteins}")
    print(f"Number of shards: {num_shards}")

    # Process shards in batches
    print("Sharding proteins...")
    with tqdm(total=total_proteins, unit=" proteins") as pbar:
        for start_shard in range(0, num_shards, max_open_files):
            end_shard = min(start_shard + max_open_files, num_shards)

            # Open current batch of shard files
            current_shard_files: Dict[int, TextIO] = {
                i: open(output_dir / f"shard_{i}.fasta", "w") for i in range(start_shard, end_shard)
            }

            try:
                with open(input_file, "r") as infile:
                    current_protein = 0
                    current_content: list[str] = []

                    # Process each line in the input file
                    for line in infile:
                        if line.startswith(">"):
                            # Write previous protein if it belongs to current batch
                            if current_content:
                                shard = current_protein // proteins_per_shard
                                if start_shard <= shard < end_shard:
                                    current_shard_files[shard].write("".join(current_content))
                                    pbar.update(1)
                                current_content = []
                            current_protein += 1

                        # Collect lines for current protein if it belongs to current batch
                        shard = (current_protein - 1) // proteins_per_shard
                        if start_shard <= shard < end_shard:
                            current_content.append(line)

                    # Handle the last protein in the file
                    if current_content:
                        shard = (current_protein - 1) // proteins_per_shard
                        if start_shard <= shard < end_shard:
                            current_shard_files[shard].write("".join(current_content))
                            pbar.update(1)

            finally:
                # Ensure all shard files are properly closed
                for file in current_shard_files.values():
                    file.close()

    return num_shards


def count_proteins_in_fasta(file_path: Union[str, Path]) -> int:
    """Count number of proteins in a FASTA file."""
    count = 0
    with gzip.open(file_path, "rt") if str(file_path).endswith(".gz") else open(file_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def sample_proteins_from_fasta(
    input_file: str, output_file: str, num_proteins: int, max_length: Optional[int] = 1022
) -> None:
    """Filter protein sequences by length and randomly select a subset.

    Uses reservoir sampling for O(k) memory complexity where k = num_proteins.

    Note - The reservoir sampling version writes them in the order they were added to the reservoir,
    which is partially random but not uniformly shuffled. If you need the output order to be fully randomized:

    ```
    random.shuffle(reservoir)

    with open_output(output_file, "wt") as outfile:
        for header, sequence in reservoir:
            outfile.write(header)
            outfile.write(sequence)
    ```

    """
    open_input = gzip.open if str(input_file).endswith(".gz") else open
    open_output = gzip.open if str(output_file).endswith(".gz") else open

    print(f"Filtering proteins by length (max_length: {max_length})...")

    # Reservoir to hold our sample
    reservoir: List[Tuple[str, str]] = []
    filtered_count = 0

    with open_input(input_file, "rt") as infile:
        current_header = ""
        current_sequence = ""

        for line in tqdm(infile, desc="Sampling proteins"):
            if line.startswith(">"):
                # Process previous sequence
                if current_header:
                    seq_clean = current_sequence.replace("\n", "")
                    if len(seq_clean) <= max_length:
                        # Reservoir sampling logic
                        if filtered_count < num_proteins:
                            reservoir.append((current_header, current_sequence))
                        else:
                            # Replace with decreasing probability
                            j = random.randint(0, filtered_count)
                            if j < num_proteins:
                                reservoir[j] = (current_header, current_sequence)
                        filtered_count += 1

                current_header = line
                current_sequence = ""
            else:
                current_sequence += line

        # Handle last sequence
        if current_header:
            seq_clean = current_sequence.replace("\n", "")
            if len(seq_clean) <= max_length:
                if filtered_count < num_proteins:
                    reservoir.append((current_header, current_sequence))
                else:
                    j = random.randint(0, filtered_count)
                    if j < num_proteins:
                        reservoir[j] = (current_header, current_sequence)
                filtered_count += 1

    print(f"Found {filtered_count} proteins meeting length criteria")
    print(f"Writing {len(reservoir)} randomly selected proteins...")

    with open_output(output_file, "wt") as outfile:
        for header, sequence in reservoir:
            outfile.write(header)
            outfile.write(sequence)

    print(f"Successfully wrote {len(reservoir)} proteins to {output_file}")


def read_fasta(
    filepath: Union[str, Path],
    max_sequences: Optional[int] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
) -> List[ProteinRecord]:
    """Read sequences from a FASTA file.

    Args:
        filepath: Path to FASTA file (supports .fasta, .fa, .fasta.gz, .fa.gz)
        max_sequences: Maximum number of sequences to read (None for all)
        max_length: Filter out sequences longer than this
        min_length: Filter out sequences shorter than this

    Returns:
        List of ProteinRecord objects
    """
    records = []
    filepath = Path(filepath)

    # Handle gzipped files
    def open_fn(p):
        """Open a file, handling gzip compression."""
        if str(p).endswith(".gz"):
            return gzip.open(p, "rt")
        return open(p, "r")

    with open_fn(filepath) as f:
        current_id = None
        current_desc = ""
        current_seq = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous record if exists
                if current_id is not None:
                    seq = "".join(current_seq)
                    if _passes_length_filter(seq, min_length, max_length):
                        records.append(ProteinRecord(id=current_id, sequence=seq, description=current_desc))

                    if max_sequences and len(records) >= max_sequences:
                        return records

                # Parse new header
                header = line[1:]  # Remove '>'
                parts = header.split(None, 1)  # Split on first whitespace
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget last record
        if current_id is not None:
            seq = "".join(current_seq)
            if _passes_length_filter(seq, min_length, max_length):
                records.append(ProteinRecord(id=current_id, sequence=seq, description=current_desc))

    return records


def _passes_length_filter(seq: str, min_length: Optional[int], max_length: Optional[int]) -> bool:
    """Check if sequence passes length filters."""
    if min_length and len(seq) < min_length:
        return False
    if max_length and len(seq) > max_length:
        return False
    return True


def stream_fasta(
    filepath: Union[str, Path], max_length: Optional[int] = None, min_length: Optional[int] = None
) -> Iterator[ProteinRecord]:
    """Stream sequences from a FASTA file one at a time.

    Memory-efficient for large files.

    Args:
        filepath: Path to FASTA file
        max_length: Filter out sequences longer than this
        min_length: Filter out sequences shorter than this

    Yields:
        ProteinRecord objects
    """
    filepath = Path(filepath)

    def open_fn(p):
        """Open a file, handling gzip compression."""
        if str(p).endswith(".gz"):
            return gzip.open(p, "rt")
        return open(p, "r")

    with open_fn(filepath) as f:
        current_id = None
        current_desc = ""
        current_seq = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    seq = "".join(current_seq)
                    if _passes_length_filter(seq, min_length, max_length):
                        yield ProteinRecord(id=current_id, sequence=seq, description=current_desc)

                header = line[1:]
                parts = header.split(None, 1)
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None:
            seq = "".join(current_seq)
            if _passes_length_filter(seq, min_length, max_length):
                yield ProteinRecord(id=current_id, sequence=seq, description=current_desc)


def write_fasta(records: List[ProteinRecord], filepath: Union[str, Path], line_width: int = 80) -> None:
    """Write sequences to a FASTA file.

    Args:
        records: List of ProteinRecord objects
        filepath: Output path
        line_width: Number of characters per line for sequences
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        for record in records:
            # Write header
            if record.description:
                f.write(f">{record.id} {record.description}\n")
            else:
                f.write(f">{record.id}\n")

            # Write sequence with line wrapping
            seq = record.sequence
            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")


def sample_sequences(
    records: List[ProteinRecord], n: int, seed: Optional[int] = None, stratify_by_length: bool = False
) -> List[ProteinRecord]:
    """Randomly sample sequences from a list.

    Args:
        records: List of ProteinRecord objects
        n: Number of sequences to sample
        seed: Random seed for reproducibility
        stratify_by_length: If True, sample evenly across length bins

    Returns:
        Sampled list of records
    """
    import random

    if seed is not None:
        random.seed(seed)

    if n >= len(records):
        return records

    if stratify_by_length:
        # Create length bins and sample from each
        lengths = [r.length for r in records]
        min_len, max_len = min(lengths), max(lengths)
        n_bins = min(10, n)
        bin_size = (max_len - min_len) / n_bins

        bins = [[] for _ in range(n_bins)]
        for record in records:
            bin_idx = min(int((record.length - min_len) / bin_size), n_bins - 1)
            bins[bin_idx].append(record)

        # Sample from each bin
        samples_per_bin = n // n_bins
        remainder = n % n_bins
        sampled = []

        for i, bin_records in enumerate(bins):
            n_sample = samples_per_bin + (1 if i < remainder else 0)
            n_sample = min(n_sample, len(bin_records))
            sampled.extend(random.sample(bin_records, n_sample))

        return sampled
    else:
        return random.sample(records, n)
