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
import random
from typing import Tuple

import pytest
import torch
import torch.utils.data

from bionemo.testing.megatron_dataset_compatibility import (
    DatasetDistributedNondeterministic,
    DatasetLocallyNondeterministic,
    assert_dataset_compatible_with_megatron,
)


class DistributedBadDataset(torch.utils.data.Dataset):
    def __init__(self, seed: int = 3, len: int = 2, shape: Tuple[int, ...] = (3,)):
        self.seed = seed
        self.len = len
        self.shape = shape

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(self.seed + index)  # fails because torch.manual_seed is impacted by distributed parallel
        return {"tensor": torch.rand(self.shape)}


class LocallyBadDataset(torch.utils.data.Dataset):
    def __init__(self, len: int = 2, shape: Tuple[int, ...] = (3,)):
        self.len = len
        self.shape = shape

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Generate a totally random seed, simulate not setting a seed
        random_seed = random.randint(0, 2**32 - 1)
        # Set the random seed for PyTorch
        torch.manual_seed(random_seed + index)
        return {"tensor": torch.rand(self.shape)}


class OKDataset(torch.utils.data.Dataset):
    def __init__(self, seed: int = 3, len: int = 2, shape: Tuple[int, ...] = (3,)):
        self.seed = seed
        self.len = len
        self.shape = shape

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + index)
        return {"tensor": torch.rand(self.shape, generator=generator)}


def test_ok_dataset_passes():
    ok_ds = OKDataset()
    assert_dataset_compatible_with_megatron(ok_ds)


def test_locally_bad_dataset_fails():
    locally_bad_ds = LocallyBadDataset()
    with pytest.raises(DatasetLocallyNondeterministic):
        assert_dataset_compatible_with_megatron(locally_bad_ds)


def test_distributed_bad_dataset_fails():
    distributed_bad_ds = DistributedBadDataset()
    with pytest.raises(DatasetDistributedNondeterministic):
        assert_dataset_compatible_with_megatron(distributed_bad_ds)
