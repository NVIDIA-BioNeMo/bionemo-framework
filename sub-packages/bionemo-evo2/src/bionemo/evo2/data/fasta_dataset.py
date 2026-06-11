# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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


import json
from pathlib import Path
from typing import Callable

import torch

from bionemo.noodles.nvfaidx import NvFaidx


class SimpleFastaDataset(torch.utils.data.Dataset):
    """A simple dataset for Evo2 prediction.

    Currently, this will not work for pre-training or fine-tuning, as that would require:
    1) including "labels" in the input and 2) offsetting/rolling either the labels or
    input_ids to handle the off-by-one token prediction alignment.
    """

    def __init__(
        self, fasta_path: Path, tokenizer, prepend_bos: bool = True, custom_loss_masker: Callable | None = None
    ):
        """Initialize the dataset."""
        super().__init__()
        self.fasta = NvFaidx(fasta_path)
        self.seqids = sorted(self.fasta.keys())
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos  # needed for getting predictions for the requested set of tokens.
        self.custom_loss_masker = custom_loss_masker

    def write_idx_map(self, output_dir: Path):
        """Write the index map to the output directory."""
        with open(output_dir / "seq_idx_map.json", "w") as f:
            json.dump({seqid: idx for idx, seqid in enumerate(self.seqids)}, f)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.seqids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get an item from the dataset."""
        sequence = self.fasta[self.seqids[idx]].sequence().upper()
        tokenized_seq = self.tokenizer.text_to_ids(sequence)
        if self.prepend_bos:  # in pretraining we use EOS to start new sequences.
            tokens: list[int] = [self.tokenizer.eod] + tokenized_seq
        else:
            tokens: list[int] = tokenized_seq
        loss_mask = torch.ones_like(torch.tensor(tokens, dtype=torch.long), dtype=torch.long)
        if self.custom_loss_masker is not None:
            custom_loss_mask = self.custom_loss_masker(tokens)
            loss_mask &= custom_loss_mask
        if self.prepend_bos:
            loss_mask[0] = (
                0  # mask the eos token which we use for causal offsetting. Later in predict we take the output
            )
            #  for the first [:-1] tokens which align with the sequence starting after the EOS.
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "position_ids": torch.arange(len(tokens), dtype=torch.long),
            "seq_idx": torch.tensor(idx, dtype=torch.long),
            "loss_mask": loss_mask,
        }
