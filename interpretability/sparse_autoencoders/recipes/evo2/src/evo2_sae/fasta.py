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

"""Shared FASTA reader for the evo2 SAE recipe (stdlib-only; no torch import).

One streaming parser reused by the batch CLI (`cli.py`) and the FASTA chunker
(`scripts/chunk_fasta.py`) so the header/`.gz`/concat logic lives in one place.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield ``(seq_id, sequence)`` for each record in a FASTA file.

    Args:
        path: Path to a FASTA file; a ``.gz`` suffix is decompressed transparently.

    Yields:
        ``(seq_id, sequence)``: the first whitespace-delimited token of the header,
        or a generated ``seq_<n>`` when the header carries no token (e.g. ``">"`` or
        ``"> "``), paired with the record's concatenated sequence lines.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    seq_id: str | None = None
    parts: list[str] = []
    n = 0  # records yielded so far — used to name token-less headers
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(parts)
                    n += 1
                header = line[1:].strip().split()
                seq_id, parts = (header[0] if header else f"seq_{n}"), []
            else:
                parts.append(line)
    if seq_id is not None:
        yield seq_id, "".join(parts)
