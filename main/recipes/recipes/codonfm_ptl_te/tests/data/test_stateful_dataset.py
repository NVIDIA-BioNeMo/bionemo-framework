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


import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.data.stateful_dataset import StatefulDataset


@pytest.fixture
def base_dataset():
    return TensorDataset(torch.arange(100))


@pytest.fixture
def stateful_dataset(base_dataset):
    return StatefulDataset(
        dataset=base_dataset, total_samples=1000, global_batch_size=10, consumed_samples=0, shuffle=False
    )


@pytest.fixture
def stateful_dataset_shuffled(base_dataset):
    return StatefulDataset(
        dataset=base_dataset, total_samples=1000, global_batch_size=10, consumed_samples=0, shuffle=True, seed=42
    )


class TestStatefulDataset:
    def test_len(self, stateful_dataset):
        # epoch_size = 100 // 10 * 10 = 100
        assert len(stateful_dataset) == 100

    def test_getitem_no_shuffle(self, stateful_dataset):
        # With no consumed samples, it should be sequential
        assert stateful_dataset[0][0] == 0
        assert stateful_dataset[10][0] == 10

    def test_getitem_with_consumed_samples(self, base_dataset):
        ds = StatefulDataset(
            dataset=base_dataset, total_samples=1000, global_batch_size=10, consumed_samples=50, shuffle=False
        )
        # Should start from sample 50
        assert ds[0][0] == 50
        assert ds[10][0] == 60

    def test_getitem_shuffled(self, stateful_dataset_shuffled):
        # Should be shuffled, but deterministic
        item0 = stateful_dataset_shuffled[0][0]
        item1 = stateful_dataset_shuffled[1][0]

        # Re-create and check for same sequence
        ds2 = StatefulDataset(
            dataset=stateful_dataset_shuffled.dataset,
            total_samples=1000,
            global_batch_size=10,
            consumed_samples=0,
            shuffle=True,
            seed=42,
        )
        assert ds2[0][0] == item0
        assert ds2[1][0] == item1

        # Should not be sequential
        assert not (item0 == 0 and item1 == 1)

    def test_epoch_boundary(self, base_dataset):
        ds = StatefulDataset(
            dataset=base_dataset,
            total_samples=1000,
            global_batch_size=10,
            consumed_samples=95,  # near the end of epoch 0
            shuffle=False,
        )
        # last 5 samples of epoch 0
        assert ds[0][0] == 95
        assert ds[4][0] == 99

        # first sample of epoch 1
        assert ds[5][0] == 0

    def test_shuffled_epoch_boundary(self, stateful_dataset_shuffled):
        # Get indices for epoch 0 and 1
        indices_epoch0 = stateful_dataset_shuffled.get_cur_idxs(0)
        indices_epoch1 = stateful_dataset_shuffled.get_cur_idxs(100)  # 100 samples is one epoch

        assert not np.array_equal(indices_epoch0, indices_epoch1)

        # Check that we get elements from the correct shuffled epoch
        # consumed_samples = 99
        ds = StatefulDataset(
            dataset=stateful_dataset_shuffled.dataset,
            total_samples=1000,
            global_batch_size=10,
            consumed_samples=99,
            shuffle=True,
            seed=42,
        )
        # last item of epoch 0
        assert ds[0][0] == indices_epoch0[99]
        # first item of epoch 1
        assert ds[1][0] == indices_epoch1[0]

    def test_index_caching(self, stateful_dataset_shuffled):
        # Access first epoch
        stateful_dataset_shuffled[0]
        assert stateful_dataset_shuffled.epoch_container[0] == 0
        assert np.all(stateful_dataset_shuffled.idx_container[0] != -1)
        assert stateful_dataset_shuffled.epoch_container[1] == -1

        # Access second epoch
        stateful_dataset_shuffled[100]
        assert stateful_dataset_shuffled.epoch_container[1] == 1
        assert np.all(stateful_dataset_shuffled.idx_container[1] != -1)

        # Access first epoch again, should be cached
        epoch0_indices_before = stateful_dataset_shuffled.idx_container[0].copy()
        stateful_dataset_shuffled[1]
        epoch0_indices_after = stateful_dataset_shuffled.idx_container[0].copy()
        assert np.array_equal(epoch0_indices_before, epoch0_indices_after)
