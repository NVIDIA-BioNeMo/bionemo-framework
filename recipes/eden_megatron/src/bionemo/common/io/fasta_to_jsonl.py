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
# This file is copied from: recipes/evo2_megatron/src/bionemo/common/io/fasta_to_jsonl.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

"""Convert FASTA files to JSONL format for use with inference ``--prompt-file``.

Each FASTA record becomes one JSONL line::

    {"id": "sequence_header", "prompt": "ATCGATCG..."}

Usage::

    bionemo_fasta_to_jsonl input.fasta output.jsonl
    bionemo_fasta_to_jsonl input.fa output.jsonl --upper

This module is used by multiple recipes via ``bionemo.common``.
**It must not import megatron-core, megatron-bridge, or NeMo.**
"""

import argparse
import json
import sys
from pathlib import Path


def fasta_to_jsonl(input_path: Path, output_path: Path, *, uppercase: bool = False) -> int:
    """Convert a FASTA file to JSONL.

    Args:
        input_path: Path to input FASTA file (.fasta, .fa, .fna, etc.).
        output_path: Path to output JSONL file.
        uppercase: If True, convert sequences to uppercase.

    Returns:
        Number of records written.
    """
    count = 0
    current_id: str | None = None
    sequence_parts: list[str] = []

    def _flush(f):
        nonlocal count, current_id, sequence_parts
        if current_id is not None:
            seq = "".join(sequence_parts)
            if uppercase:
                seq = seq.upper()
            f.write(json.dumps({"id": current_id, "prompt": seq}) + "\n")
            count += 1
        current_id = None
        sequence_parts = []

    with open(input_path) as fin, open(output_path, "w") as fout:
        for raw_line in fin:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                _flush(fout)
                current_id = stripped[1:].split()[0]
            else:
                sequence_parts.append(stripped)
        _flush(fout)

    return count


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(
        description="Convert a FASTA file to JSONL for use with inference --prompt-file",
    )
    ap.add_argument("input", type=Path, help="Input FASTA file")
    ap.add_argument("output", type=Path, help="Output JSONL file")
    ap.add_argument(
        "--upper",
        action="store_true",
        default=False,
        help="Convert sequences to uppercase",
    )
    return ap.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    count = fasta_to_jsonl(args.input, args.output, uppercase=args.upper)
    print(f"Wrote {count} record(s) to {args.output}")


if __name__ == "__main__":
    main()
