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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: recipes/evo2_megatron/src/bionemo/common/fasta/nvfaidx.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

# ruff: noqa: D102,D105,D107

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence


__all__: Sequence[str] = (
    "NvFaidx",
    "PyFaidxRecord",
    "PyIndexedMmapFastaReader",
    "SequenceAccessor",
    "back_transcribe_sequence",
    "complement_sequence",
    "reverse_sequence",
    "transcribe_sequence",
)


_DNA_COMPLEMENT = str.maketrans("ACGTRYMKBDHVNacgtrymkbdhvn", "TGCAYRKMVHDBNtgcayrkmvhdbn")


def complement_sequence(sequence: str) -> str:
    """Return the DNA complement of *sequence*."""
    return sequence.translate(_DNA_COMPLEMENT)


def reverse_sequence(sequence: str) -> str:
    """Return *sequence* in reverse order."""
    return sequence[::-1]


def transcribe_sequence(sequence: str) -> str:
    """Transcribe DNA to RNA by replacing thymine with uracil."""
    return sequence.replace("T", "U").replace("t", "u")


def back_transcribe_sequence(sequence: str) -> str:
    """Back-transcribe RNA to DNA by replacing uracil with thymine."""
    return sequence.replace("U", "T").replace("u", "t")


@dataclass(frozen=True)
class PyFaidxRecord:
    """Small Python equivalent of the record metadata exposed by bionemo.common.fasta."""

    name: str
    length: int


class PyIndexedMmapFastaReader:
    """Indexed FASTA reader with the subset of the bionemo.common.fasta API used by recipes."""

    def __init__(self, fasta_path: str | Path, ignore_existing_fai: bool = True) -> None:
        del ignore_existing_fai
        self.fasta_path = Path(fasta_path)
        self._sequences = _read_fasta(self.fasta_path)
        self._records = [PyFaidxRecord(name=name, length=len(seq)) for name, seq in self._sequences.items()]

    @classmethod
    def from_fasta_and_faidx(cls, fasta_path: str | Path, faidx_path: str | Path) -> "PyIndexedMmapFastaReader":
        del faidx_path
        return cls(fasta_path)

    def records(self) -> list[PyFaidxRecord]:
        return list(self._records)

    def read_sequence_mmap(self, region: str) -> str:
        seqid, bounds = region.rsplit(":", 1)
        start_s, end_s = bounds.split("-", 1)
        start = int(start_s) - 1
        end = int(end_s)
        return self._sequences[seqid][start:end]

    @staticmethod
    def create_faidx(fasta_filename: str | Path, force: bool = False) -> str:
        fasta_path = Path(fasta_filename)
        faidx_path = fasta_path.with_suffix(fasta_path.suffix + ".fai")
        if faidx_path.exists() and not force:
            return str(faidx_path)

        records = _scan_fasta_for_fai(fasta_path)
        with faidx_path.open("w") as f:
            for name, length, offset, line_bases, line_width in records:
                f.write(f"{name}\t{length}\t{offset}\t{line_bases}\t{line_width}\n")
        return str(faidx_path)


def _read_fasta(fasta_path: Path) -> dict[str, str]:
    sequences: dict[str, list[str]] = {}
    current_name: str | None = None

    with fasta_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_name = line[1:].split()[0]
                sequences.setdefault(current_name, [])
            elif current_name is None:
                raise ValueError(f"Found sequence data before a FASTA header in {fasta_path}")
            else:
                sequences[current_name].append(line)

    return {name: "".join(parts) for name, parts in sequences.items()}


def _scan_fasta_for_fai(fasta_path: Path) -> list[tuple[str, int, int, int, int]]:
    records: list[tuple[str, int, int, int, int]] = []
    current_name: str | None = None
    current_length = 0
    sequence_offset = 0
    line_bases = 0
    line_width = 0

    with fasta_path.open("rb") as f:
        while True:
            line_offset = f.tell()
            line = f.readline()
            if not line:
                break
            stripped = line.rstrip(b"\r\n")
            if not stripped:
                continue
            if stripped.startswith(b">"):
                if current_name is not None:
                    records.append((current_name, current_length, sequence_offset, line_bases, line_width))
                current_name = stripped[1:].split(None, 1)[0].decode()
                current_length = 0
                sequence_offset = f.tell()
                line_bases = 0
                line_width = 0
            else:
                if current_name is None:
                    raise ValueError(f"Found sequence data before a FASTA header in {fasta_path}")
                current_length += len(stripped)
                if line_bases == 0:
                    line_bases = len(stripped)
                    line_width = len(line)
                if sequence_offset == 0:
                    sequence_offset = line_offset

    if current_name is not None:
        records.append((current_name, current_length, sequence_offset, line_bases, line_width))
    return records


class SequenceAccessor:
    """Dictionary-like sequence accessor for integer and slice lookups."""

    def __init__(self, reader: PyIndexedMmapFastaReader, seqid: str, length: int) -> None:
        self.reader = reader
        self.seqid = seqid
        self.length = length

    def __getitem__(self, key: int | slice) -> str:
        if isinstance(key, slice):
            if key.step not in (None, 1):
                return self.sequence()[key]

            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.length
            if start < 0:
                start += self.length
            if stop < 0:
                stop += self.length
            start = max(0, min(self.length, start))
            stop = max(0, min(self.length, stop))
            if start > stop:
                return ""
            return self.reader.read_sequence_mmap(f"{self.seqid}:{start + 1}-{stop}")

        if isinstance(key, int):
            if key < 0:
                key += self.length
            if key < 0 or key >= self.length:
                raise IndexError(f"Position {key} is out of bounds for '{self.seqid}' with length {self.length}.")
            return self.reader.read_sequence_mmap(f"{self.seqid}:{key + 1}-{key + 1}")

        raise TypeError("Index must be an integer or a slice.")

    def __len__(self) -> int:
        return self.length

    def sequence_id(self) -> str:
        return self.seqid

    def sequence(self) -> str:
        return self[:]


class NvFaidx:
    """Dictionary-like FASTA index compatible with the recipe subset of bionemo.common.fasta."""

    def __init__(
        self,
        fasta_path: str | Path,
        faidx_path: Optional[str | Path] = None,
        ignore_existing_fai: bool = True,
        allow_duplicate_seqids: bool = False,
    ) -> None:
        fasta_path = str(fasta_path)
        if faidx_path is not None:
            faidx_path = str(faidx_path)

        if faidx_path is not None and not ignore_existing_fai:
            self.reader = PyIndexedMmapFastaReader.from_fasta_and_faidx(fasta_path, faidx_path)
        else:
            self.reader = PyIndexedMmapFastaReader(fasta_path, ignore_existing_fai=ignore_existing_fai)

        records = self.reader.records()
        self.records: Dict[str | int, PyFaidxRecord] = {record.name: record for record in records}
        if len(self.records) != len(records):
            if not allow_duplicate_seqids:
                raise ValueError(
                    "Non-unique sequence-id detected in FASTA. Correct headers and try again or pass "
                    "allow_duplicate_seqids=True."
                )
            self.records = dict(enumerate(records))

    def __getitem__(self, seqid: str | int) -> SequenceAccessor:
        if seqid not in self.records:
            raise KeyError(f"Sequence '{seqid}' not found in index.")
        record = self.records[seqid]
        return SequenceAccessor(self.reader, record.name, record.length)

    def __contains__(self, seqid: str | int) -> bool:
        return seqid in self.records

    def __len__(self) -> int:
        return len(self.records)

    def keys(self) -> set[str | int]:
        return set(self.records.keys())

    def __iter__(self) -> Iterable[str | int]:
        return iter(self.keys())

    def items(self):
        for key in self.keys():
            yield key, self[key][:]

    def values(self):
        for key in self.keys():
            yield self[key][:]

    @staticmethod
    def create_faidx(fasta_filename: str | Path, force: bool = False) -> str:
        return PyIndexedMmapFastaReader.create_faidx(fasta_filename, force)
