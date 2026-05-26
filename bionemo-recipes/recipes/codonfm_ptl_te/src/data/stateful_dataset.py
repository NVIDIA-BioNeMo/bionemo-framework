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


from typing import Any

import numpy as np
from torch.utils.data import Dataset


class StatefulDataset(Dataset):
    """A stateful dataset that allows for to resume training from a given epoch.

    Args:
        dataset (Dataset): The underlying dataset.
        total_samples (int): The total number of samples in the dataset.
        consumed_samples (int, optional): The number of samples already consumed. Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): The random seed for shuffling. Defaults to 123.
    """

    def __init__(self, dataset, total_samples, global_batch_size=None, consumed_samples=0, shuffle=False, seed=123):  # noqa: D107
        self.dataset = dataset
        self.total_samples = total_samples
        self.global_batch_size = global_batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.consumed_samples = consumed_samples
        self.epoch_size = (
            len(dataset) // global_batch_size * global_batch_size if global_batch_size is not None else len(dataset)
        )
        if self.epoch_size == 0:
            self.epoch_size = len(dataset)
        self.dataset_size = len(dataset)
        self.idx_container = np.full((3, self.dataset_size), -1, dtype=np.int64)
        self.epoch_container = np.full(3, -1, dtype=np.int64)
        _ = self.get_cur_idxs(self.consumed_samples)

    def get_cur_idxs(self, idx: int) -> np.ndarray:
        """Get the current indices based on the consumed samples.

        Args:
            idx (int): The number of samples already consumed.

        Returns:
            np.ndarray: The current indices.
        """
        epoch = idx // self.epoch_size
        local_epoch = epoch % 3  # - map to one of 3 sets of indices
        if self.idx_container[local_epoch, 0] == -1 or self.epoch_container[local_epoch] != epoch:
            self.idx_container[local_epoch] = self._get_cur_idxs(epoch)
            self.epoch_container[local_epoch] = epoch
        return self.idx_container[local_epoch]

    def _get_cur_idxs(self, epoch: int) -> np.ndarray:
        """Get the current indices for a specific epoch.

        Args:
            epoch (int): The current epoch.

        Returns:
            np.ndarray: The current indices.

        """
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        idxs = np.arange(self.dataset_size)

        if self.shuffle:
            rng.shuffle(idxs)
        return idxs

    def _resolve_idx(self, idx: int) -> int:
        """Map an external index to the underlying dataset index."""
        idx = (self.consumed_samples + idx) % self.total_samples
        idxs = self.get_cur_idxs(idx)
        return idxs[idx % self.epoch_size]

    def get_sequence_length(self, idx: int) -> int:
        """Return the unpadded token count, delegating to the underlying dataset."""
        return self.dataset.get_sequence_length(self._resolve_idx(idx))

    def __getitem__(self, idx: int) -> Any:
        """Get the item at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            Any: The item at the specified index.

        """
        return self.dataset[self._resolve_idx(idx)]

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: The total number of samples.

        """
        return self.epoch_size
