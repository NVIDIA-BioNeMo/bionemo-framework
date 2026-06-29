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

from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """Dataset for protein sequences.

    Args:
        sequences: List of protein sequences (amino acid strings)
        tokenizer: ESM2 tokenizer
        max_length: Maximum sequence length
        ids: Optional list of sequence identifiers
    """

    def __init__(self, sequences: List[str], tokenizer, max_length: int = 1024, ids: Optional[List[str]] = None):
        """Initialize the dataset with sequences, tokenizer, and optional IDs."""
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ids = ids if ids is not None else [str(i) for i in range(len(sequences))]

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def preprocess_sequence(self, sequence: str) -> str:
        """Clean protein sequence."""
        return "".join(sequence.split()).upper()

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Return tokenized sequence and metadata for the given index."""
        sequence = self.preprocess_sequence(self.sequences[index])

        encoding = self.tokenizer(
            sequence, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sequence": sequence,
            "id": self.ids[index],
        }
