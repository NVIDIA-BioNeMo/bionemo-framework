# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, NamedTuple, Protocol, Sequence, TypeVar

import numpy as np
from torch.utils.data import Dataset

from bionemo.common.data.permute import permute


__all__: Sequence[str] = (
    "EpochIndex",
    "IdentityMultiEpochDatasetWrapper",
    "MultiEpochDataset",
    "MultiEpochDatasetResampler",
    "MultiEpochDatasetWrapper",
    "SizedDataset",
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
U_co = TypeVar("U_co", covariant=True)


class EpochIndex(NamedTuple):
    """A tuple that contains both the current epoch and index for multi-epoch training."""

    epoch: int
    """An integer representing the current epoch."""

    idx: int
    """An integer representing the index within the current epoch."""


class SizedDataset(Protocol[T_co]):
    """A protocol for integer-indexed datasets that have a fixed length."""

    def __getitem__(self, index: int) -> T_co:  # noqa: D105
        ...

    def __len__(self) -> int:  # noqa: D105
        ...


class MultiEpochDataset(Protocol[T_co]):
    """A protocol for datasets for multi-epoch training in Megatron-LM.

    !!! important "Dataset determinism in Megatron-LM"
        In megatron training, the sampler and dataset objects are used to ensure consistent data loading across
        model-parallel ranks. For datasets to work with megatron training, they must return exactly the same data for
        every call to `__getitem__` with the same index.
    """

    def __getitem__(self, index: EpochIndex) -> T_co:  # noqa: D105
        ...

    def __len__(self) -> int:  # noqa: D105
        ...


@dataclass
class MultiEpochDatasetResampler(Dataset[T_co]):
    """A dataset wrapper class that converts the sequential sampling from Megatron-LM to epoch-based sampling.

    Either `num_epochs` or `num_samples` should be provided. If neither are provided, the dataset will use a single
    epoch. If `num_epochs` is given, the resampled dataset will have `len(dataset) * num_epochs` samples. If
    `num_samples` the resampled dataset will have `num_samples` samples. For `num_samples`, the dataset will be repeated
    for multiple epochs until the desired number of samples is reached (with the final epoch being truncated).
    """

    dataset: MultiEpochDataset[T_co]
    """The dataset to resample. Must support indexing with an `EpochIndex`."""

    num_epochs: int | None = None
    """The total number of epochs. The length of the resampled dataset will be len(dataset) * num_epochs."""

    num_samples: int | None = None
    """The total number of samples to draw.

    The number of epochs will be determined by the number of samples and the length of the dataset.
    """

    shuffle: bool = True
    """Whether to shuffle the samples in the dataset each epoch."""

    seed: int = 42  # type: ignore
    """A random seed for reproducibility."""

    def __post_init__(self):
        """Pre-shuffle each epoch's samples."""
        if self.num_epochs is None and self.num_samples is None:
            self.num_epochs = 1
        elif self.num_epochs is not None and self.num_samples is not None:
            raise ValueError("Only one of num_epochs and num_samples should be provided.")

        if self.num_epochs is None and self.num_samples is not None:
            self.num_epochs = math.ceil(self.num_samples / len(self.dataset))

        elif self.num_samples is None and self.num_epochs is not None:
            self.num_samples = len(self.dataset) * self.num_epochs

        # Type guard statements, the above if/elif block should ensure these are not None.
        assert self.num_epochs is not None
        assert self.num_samples is not None

        if self.num_epochs < 1:
            raise ValueError("num_epochs must be at least 1.")

        rng = np.random.default_rng(self.seed)

        # Initialize a vector of random seeds so that each epoch is shuffled differently.
        self.epoch_seeds = rng.integers(0, np.iinfo(np.int32).max, size=self.num_epochs)

    def __getitem__(self, index: int) -> T_co:
        """Get the sample at the given index."""
        if index not in range(len(self)):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}.")
        return self.dataset[self._global_index_to_permuted_local_index(index)]

    def __len__(self) -> int:
        """Return the length of the resampled dataset."""
        return self.num_samples  # type: ignore

    def _global_index_to_permuted_local_index(self, index: int) -> EpochIndex:
        """Convert a global index to an epoch index."""
        epoch = index // len(self.dataset)
        idx = index % len(self.dataset)
        if self.shuffle:
            idx = permute(idx, len(self.dataset), self.epoch_seeds[epoch])
        return EpochIndex(epoch, idx)


@dataclass
class MultiEpochDatasetWrapper(Dataset[U_co], Generic[T, U_co], ABC):
    """A wrapper to convert a standard pytorch dataset into one that supports multi-epoch megatron training.

    The underlying dataset's __getitem__ method must be deterministic, i.e. it must return the same data for the same
    index every time it is called. If there are any non-deterministic operations, they should be moved to the
    `apply_transform` method. This method must also be deterministic for every (epoch, index) pair, but it can use
    the epoch to implement data augmentation each epoch.
    """

    dataset: SizedDataset[T]
    """A deterministic dataset that supports indexing with an integer index."""

    @abstractmethod
    def apply_transform(self, sample: T, index: EpochIndex) -> U_co:
        """Apply any transformations to the sample for the given epoch."""
        raise NotImplementedError

    def __getitem__(self, index: EpochIndex) -> U_co:
        """Get the sample at the given epoch and index."""
        return self.apply_transform(self.dataset[index.idx], index)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)


class IdentityMultiEpochDatasetWrapper(MultiEpochDatasetWrapper[T, T]):
    """An implementation of the `MultiEpochDatasetWrapper` that does not apply any transformations."""

    def apply_transform(self, sample: T, index: EpochIndex) -> T:
        """Return the sample as is."""
        del index  # Unused.
        return sample
