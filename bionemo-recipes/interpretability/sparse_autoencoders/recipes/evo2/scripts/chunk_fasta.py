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

"""Chunk a FASTA into <=N-bp windows so predict_evo2 stays inside the model's trained context.

Evo2 1B was trained with seq_length=8192; longer inputs OOM in the Hyena
fftconv path (intermediates scale super-linearly with L). For 7B/40B raise
--window to whatever those checkpoints were context-extended to.

Non-overlapping windows by default. Each chunk gets a header of the form
">{orig_id}:{start}-{end}" so downstream parquet can be back-mapped.
"""

import argparse
from pathlib import Path

from evo2_sae.fasta import read_fasta


def main():
    """Read input FASTA, write non-overlapping <=window-bp chunks to output FASTA."""
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=int, default=8192)
    args = p.parse_args()
    if args.window <= 0:
        p.error("--window must be a positive integer")
    if args.input.resolve() == args.output.resolve():
        p.error("--input and --output must be different files")

    n_in = n_out = bp_out = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for seq_id, seq in read_fasta(args.input):
            n_in += 1
            for start in range(0, len(seq), args.window):
                end = min(start + args.window, len(seq))
                chunk = seq[start:end]
                out.write(f">{seq_id}:{start}-{end}\n{chunk}\n")
                n_out += 1
                bp_out += len(chunk)

    print(f"Chunked {n_in} sequences -> {n_out} chunks ({bp_out:,} bp) at window={args.window}")


if __name__ == "__main__":
    main()
