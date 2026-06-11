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

"""Mock dataset provider for Eden smoke tests and debugging."""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from torch.utils.data import Dataset


class MockEdenDataset(Dataset):
    """Synthetic dataset producing random token sequences for Eden model testing."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int = 256, seed: int = 42) -> None:
        """Initialize the mock dataset."""
        super().__init__()
        self._length = num_samples
        self._sl = seq_length
        self._vocab_size = vocab_size
        self._seed = seed
        self.loss_mask = torch.ones(seq_length, dtype=torch.float)
        self.position_ids = torch.arange(seq_length, dtype=torch.int64)

    def __len__(self) -> int:
        """Return number of samples."""
        return self._length

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Return a single sample with deterministic random tokens."""
        np_gen = np.random.default_rng(seed=(self._seed + int(idx)))
        tokens = torch.from_numpy(np_gen.integers(self._vocab_size, size=[self._sl + 1], dtype=np.int64))
        return {
            "tokens": tokens[:-1],
            "labels": tokens[1:],
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
        }


@dataclass
class MockEdenDatasetProvider(DatasetProvider):
    """Mock dataset provider for Eden training smoke tests."""

    random_seed: int = 42
    seq_length: int = 8192
    vocab_size: int = 256
    dataloader_type: str = "single"

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
        """Build synthetic train/val/test datasets."""
        train_ds = MockEdenDataset(
            num_samples=context.train_samples,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            seed=self.random_seed,
        )
        val_ds = MockEdenDataset(
            num_samples=context.valid_samples,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            seed=self.random_seed + 1,
        )
        test_ds = MockEdenDataset(
            num_samples=context.test_samples,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            seed=self.random_seed + 2,
        )
        return train_ds, val_ds, test_ds
