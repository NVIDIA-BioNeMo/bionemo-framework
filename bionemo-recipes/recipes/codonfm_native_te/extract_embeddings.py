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

"""Extract CLS-token embeddings from a pretrained CodonFM model.

Usage:
    python extract_embeddings.py \
        --model-name-or-path nvidia/NV-CodonFM-Encodon-TE-Cdwt-1B-v1 \
        --input seqs.fasta \
        --output embeddings.npz
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from modeling_codonfm_te import CodonFMForMaskedLM
from tokenizer import CodonTokenizer


@dataclass
class EmbeddingOutput:
    """Container for extracted embeddings and the corresponding record ids."""

    embeddings: np.ndarray
    ids: np.ndarray | None = None


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into a list of (id, sequence) tuples."""
    records: list[tuple[str, str]] = []
    seq_id: str | None = None
    seq_parts: list[str] = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    records.append((seq_id, "".join(seq_parts)))
                seq_id = line[1:].split()[0] if len(line) > 1 else ""
                seq_parts = []
            else:
                seq_parts.append(line)
    if seq_id is not None:
        records.append((seq_id, "".join(seq_parts)))
    return records


def _tokenize_batch(
    tokenizer: CodonTokenizer,
    sequences: list[str],
    max_seq_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode and right-pad a batch of DNA sequences to the longest seq in the batch."""
    encoded: list[list[int]] = []
    for s in sequences:
        ids = tokenizer.encode(s, add_special_tokens=True)
        if len(ids) > max_seq_length:
            ids = [*ids[: max_seq_length - 1], tokenizer.sep_token_id]
        encoded.append(ids)

    pad_to = max(len(ids) for ids in encoded)
    input_ids = np.full((len(encoded), pad_to), tokenizer.pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((len(encoded), pad_to), dtype=np.int64)
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1

    return (
        torch.from_numpy(input_ids).to(device),
        torch.from_numpy(attention_mask).to(device),
    )


def extract_embeddings(
    model: CodonFMForMaskedLM,
    tokenizer: CodonTokenizer,
    records: list[tuple[str, str]],
    batch_size: int,
    max_seq_length: int,
    device: torch.device | str = "cuda",
) -> EmbeddingOutput:
    """Return CLS-token embeddings from the final hidden layer for each record."""
    device = torch.device(device)
    ids = [r[0] for r in records]
    seqs = [r[1] for r in records]

    all_embeds: list[np.ndarray] = []
    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i : i + batch_size]
        input_ids, attention_mask = _tokenize_batch(tokenizer, batch_seqs, max_seq_length, device)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        cls = output.hidden_states[-1][:, 0, :]
        if cls.dtype != torch.float32:
            cls = cls.float()
        all_embeds.append(cls.cpu().numpy())

    embeddings = np.concatenate(all_embeds, axis=0) if all_embeds else np.zeros((0, 0), dtype=np.float32)
    return EmbeddingOutput(embeddings=embeddings, ids=np.array(ids))


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Hugging Face Hub tag or local directory with a CodonFM checkpoint.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to a FASTA file with DNA sequences.")
    parser.add_argument("--output", type=Path, default=None, help="Optional .npz path to save embeddings and ids.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = CodonFMForMaskedLM.from_pretrained(args.model_name_or_path).to(args.device).eval()
    tokenizer = CodonTokenizer()

    records = read_fasta(args.input)
    if not records:
        raise ValueError(f"No FASTA records found in {args.input}")

    out = extract_embeddings(
        model,
        tokenizer,
        records,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        device=args.device,
    )

    if args.output is not None:
        np.savez(args.output, embeddings=out.embeddings, ids=out.ids)
        print(f"Saved {out.embeddings.shape[0]} embeddings of dim {out.embeddings.shape[1]} to {args.output}")
    else:
        print(f"Extracted {out.embeddings.shape[0]} embeddings of dim {out.embeddings.shape[1]}")


if __name__ == "__main__":
    main()
