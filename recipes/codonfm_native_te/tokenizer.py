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

from itertools import product


class CodonTokenizer:
    """Simple codon tokenizer for DNA sequences.

    Splits raw coding sequences into 3-mer codon tokens and maps them to integer IDs.
    Vocabulary: 5 special tokens + 64 DNA codons = 69 total.
    """

    SPECIAL_TOKENS: list[str] = ["<CLS>", "<SEP>", "<UNK>", "<PAD>", "<MASK>"]  # noqa: RUF012

    def __init__(self, seq_type="dna"):
        """Initialize the tokenizer.

        Args:
            seq_type: Sequence type, either "dna" or "rna".
        """
        seq_type = seq_type.lower()
        if seq_type not in ("dna", "rna"):
            raise ValueError(f"seq_type must be 'dna' or 'rna', got {seq_type}")

        bases = "ACGT" if seq_type == "dna" else "ACGU"
        self.codons = ["".join(c) for c in product(bases, repeat=3)]
        self.seq_type = seq_type

        all_tokens = self.SPECIAL_TOKENS + self.codons
        self.encoder = {tok: i for i, tok in enumerate(all_tokens)}
        self.decoder = {i: tok for tok, i in self.encoder.items()}

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.encoder)

    @property
    def cls_token_id(self) -> int:
        """Return the CLS token ID."""
        return self.encoder["<CLS>"]

    @property
    def sep_token_id(self) -> int:
        """Return the SEP token ID."""
        return self.encoder["<SEP>"]

    @property
    def unk_token_id(self) -> int:
        """Return the UNK token ID."""
        return self.encoder["<UNK>"]

    @property
    def pad_token_id(self) -> int:
        """Return the PAD token ID."""
        return self.encoder["<PAD>"]

    @property
    def mask_token_id(self) -> int:
        """Return the MASK token ID."""
        return self.encoder["<MASK>"]

    def tokenize(self, sequence: str) -> list[str]:
        """Split a DNA/RNA sequence into codon tokens.

        Args:
            sequence: Raw DNA/RNA string (length must be divisible by 3).

        Returns:
            List of codon token strings.
        """
        sequence = sequence.upper()
        tokens = []
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i : i + 3]
            if len(codon) == 3:
                tokens.append(codon)
        return tokens

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]:
        """Encode a DNA/RNA sequence into token IDs.

        Args:
            sequence: Raw DNA/RNA string.
            add_special_tokens: Whether to add CLS and SEP tokens.

        Returns:
            List of token IDs.
        """
        tokens = self.tokenize(sequence)
        ids = [self.encoder.get(tok, self.unk_token_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.cls_token_id, *ids, self.sep_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a sequence string.

        Args:
            ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens in the output.

        Returns:
            Decoded sequence string.
        """
        special_ids = set(range(len(self.SPECIAL_TOKENS)))
        tokens = []
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            tokens.append(self.decoder.get(i, "<UNK>"))
        return "".join(tokens)
