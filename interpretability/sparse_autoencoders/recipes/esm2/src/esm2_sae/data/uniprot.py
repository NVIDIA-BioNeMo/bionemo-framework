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

import gzip
import io
from pathlib import Path
from typing import Optional, Union

import requests


SWISSPROT_FASTA_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
)
UNIREF50_FASTA_URL = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"


def download_swissprot(output_path: Union[str, Path]) -> Path:
    """Download the SwissProt FASTA database to the specified directory."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = "uniprot_sprot.fasta.gz"
    filepath = output_path / filename

    print(f"Downloading Swiss-Prot to {filepath}...")
    _download_file(SWISSPROT_FASTA_URL, filepath)
    print(f"Downloaded: {filepath}")

    return filepath


def download_uniref50(
    output_path: Union[str, Path],
    max_proteins: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Path:
    """Download UniRef50 FASTA data.

    Args:
        output_path: Directory where the downloaded file should be written.
        max_proteins: If provided, write only the first N proteins into a
            smaller FASTA file. If None, download the full compressed archive.
        max_length: If provided, only keep proteins with sequence length
            <= max_length. Streams through the database until max_proteins
            sequences passing the filter have been collected.

    Returns:
        Path to downloaded FASTA file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if max_proteins is None:
        filepath = output_path / "uniref50.fasta.gz"
        print(f"Downloading full UniRef50 to {filepath}...")
        _download_file(UNIREF50_FASTA_URL, filepath)
        print(f"Downloaded: {filepath}")
        return filepath

    if max_proteins <= 0:
        raise ValueError(f"max_proteins must be > 0, got {max_proteins}")

    if max_length:
        filepath = output_path / f"uniref50_{max_proteins}_maxlen{max_length}.fasta"
    else:
        filepath = output_path / f"uniref50_first_{max_proteins}.fasta"
    print(
        f"Downloading {max_proteins:,} UniRef50 proteins to {filepath}..."
        + (f" (max_length={max_length})" if max_length else "")
    )
    written, scanned = _download_gzipped_fasta_subset(
        UNIREF50_FASTA_URL,
        filepath,
        max_proteins=max_proteins,
        max_length=max_length,
    )
    print(
        f"Downloaded subset: {filepath} ({written:,} proteins" + (f", scanned {scanned:,}" if max_length else "") + ")"
    )
    return filepath


def _download_file(url: str, filepath: Path) -> None:
    """Download a file with progress indication."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(filepath, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rDownloading: {percent:.1f}%", end="", flush=True)
    print()


def _download_gzipped_fasta_subset(
    url: str,
    filepath: Path,
    max_proteins: int,
    max_length: Optional[int] = None,
) -> tuple:
    """Download a gzipped FASTA URL and write the first N proteins.

    If max_length is provided, only keeps proteins with sequence length
    <= max_length, continuing to stream until max_proteins are collected.

    Returns:
        Tuple of (written_count, scanned_count).
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    written = 0
    scanned = 0
    open_output = gzip.open if str(filepath).endswith(".gz") else open

    with response:
        with gzip.GzipFile(fileobj=response.raw, mode="rb") as gz_file:
            with io.TextIOWrapper(gz_file, encoding="utf-8") as infile:
                with open_output(filepath, "wt") as outfile:
                    if max_length is None:
                        # Fast path: no length filtering
                        for line in infile:
                            if line.startswith(">"):
                                if written >= max_proteins:
                                    break
                                written += 1
                            if written > 0:
                                outfile.write(line)
                        scanned = written
                    else:
                        # Buffer each protein, check length, then write
                        header = None
                        seq_lines = []
                        for line in infile:
                            if line.startswith(">"):
                                # Process previous protein
                                if header is not None:
                                    scanned += 1
                                    seq = "".join(l.strip() for l in seq_lines)
                                    if len(seq) <= max_length:
                                        outfile.write(header)
                                        for sl in seq_lines:
                                            outfile.write(sl)
                                        written += 1
                                        if written % 1000 == 0:
                                            print(
                                                f"\r  Collected {written:,}/{max_proteins:,} (scanned {scanned:,})",
                                                end="",
                                                flush=True,
                                            )
                                        if written >= max_proteins:
                                            break
                                header = line
                                seq_lines = []
                            else:
                                seq_lines.append(line)

                        # Handle last protein
                        if header is not None and written < max_proteins:
                            scanned += 1
                            seq = "".join(l.strip() for l in seq_lines)
                            if len(seq) <= max_length:
                                outfile.write(header)
                                for sl in seq_lines:
                                    outfile.write(sl)
                                written += 1

                        if max_length:
                            print()  # newline after progress

    if written == 0:
        raise RuntimeError("No proteins were downloaded from UniRef50 stream.")

    return written, scanned
