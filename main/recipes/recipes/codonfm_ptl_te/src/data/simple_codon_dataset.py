# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Simple synthetic codon dataset for testing and demo purposes.

This dataset generates random sequences on-the-fly without requiring any data files.
Useful for quick testing, debugging, and development without setting up real data.
"""

from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from src.data.metadata import MetadataFields


class SimpleCodonDataset(Dataset):
    """Simple synthetic dataset that generates random codon sequences.

    This dataset is useful for:
    - Quick testing without setting up data files
    - Debugging model training loops
    - Development and prototyping
    - FSDP/distributed training tests

    Args:
        num_samples (int): Number of samples in the dataset. Defaults to 1000.
        seq_length (int): Length of each sequence. Defaults to 2048.
        vocab_size (int): Size of the vocabulary. Defaults to 69 (codon vocabulary size).
        split_name (str): Split name ('train', 'val', 'test', or 'all'). Defaults to 'all'.
        train_ratio (float): Ratio of training samples. Defaults to 0.8.
        val_ratio (float): Ratio of validation samples. Defaults to 0.1.
        process_item (Callable, optional): Function to process items. Not used in this dataset.
        seed (int, optional): Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 2048,
        vocab_size: int = 69,
        split_name: str = "all",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        process_item: Optional[Callable] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the SimpleCodonDataset."""
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.split_name = split_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.process_item = process_item
        self.seed = seed

        # Calculate split boundaries
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)

        # Set the actual samples for this split
        if split_name == "train":
            self.start_idx = 0
            self.end_idx = train_end
        elif split_name == "val":
            self.start_idx = train_end
            self.end_idx = val_end
        elif split_name == "test":
            self.start_idx = val_end
            self.end_idx = num_samples
        else:  # 'all'
            self.start_idx = 0
            self.end_idx = num_samples

        self.actual_num_samples = self.end_idx - self.start_idx

    def __len__(self):
        """Return the number of samples in this split."""
        return self.actual_num_samples

    def __getitem__(self, idx):
        """Generate a random codon sequence sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - INPUT_IDS: Random token IDs (numpy array)
                - LABELS: Random labels for MLM (numpy array)
                - ATTENTION_MASK: All ones (no padding) (numpy array)
                - INPUT_MASK: All ones (no masking) (numpy array)
        """
        # Use deterministic random generation based on seed and index
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + self.start_idx + idx)
        else:
            rng = np.random.default_rng()

        return {
            MetadataFields.INPUT_IDS: rng.integers(0, self.vocab_size, size=self.seq_length, dtype=np.int64),
            MetadataFields.LABELS: rng.integers(0, self.vocab_size, size=self.seq_length, dtype=np.int64),
            MetadataFields.ATTENTION_MASK: np.ones(self.seq_length, dtype=bool),
            MetadataFields.INPUT_MASK: np.ones(self.seq_length, dtype=bool),
        }

    def get_train(self, process_item: Optional[Callable] = None) -> "SimpleCodonDataset":
        """Return the training split of the dataset.

        Args:
            process_item: Optional processing function (not used in this dataset).

        Returns:
            SimpleCodonDataset instance for the training split.
        """
        return SimpleCodonDataset(
            num_samples=self.num_samples,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            split_name="train",
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            process_item=process_item or self.process_item,
            seed=self.seed,
        )

    def get_validation(self, process_item: Optional[Callable] = None) -> "SimpleCodonDataset":
        """Return the validation split of the dataset.

        Args:
            process_item: Optional processing function (not used in this dataset).

        Returns:
            SimpleCodonDataset instance for the validation split.
        """
        return SimpleCodonDataset(
            num_samples=self.num_samples,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            split_name="val",
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            process_item=process_item or self.process_item,
            seed=self.seed,
        )

    def get_test(self, process_item: Optional[Callable] = None) -> "SimpleCodonDataset":
        """Return the test split of the dataset.

        Args:
            process_item: Optional processing function (not used in this dataset).

        Returns:
            SimpleCodonDataset instance for the test split.
        """
        return SimpleCodonDataset(
            num_samples=self.num_samples,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            split_name="test",
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            process_item=process_item or self.process_item,
            seed=self.seed,
        )
