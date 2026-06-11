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

from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from bionemo.llm.data.datamodule import MockDataModule


class SimpleTokenizedDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"text": torch.tensor(self.sequences[idx])}


@pytest.fixture
def example_sequences():
    return [
        [1, 2, 3],  # Short sequence
        [4, 5, 6, 7, 8],  # Medium sequence
        [9, 10, 11, 12, 13, 14, 15],  # Long sequence
    ]


@pytest.fixture
def tokenized_dataset(example_sequences):
    return SimpleTokenizedDataset(example_sequences)


def test_train_dataloader_creation(tokenized_dataset):
    datamodule = MockDataModule(
        train_dataset=tokenized_dataset,
        micro_batch_size=2,
        global_batch_size=2,
    )

    datamodule.trainer = mock.Mock()
    datamodule.trainer.global_step = 0

    train_dataloader = datamodule.train_dataloader()

    with (
        mock.patch("megatron.core.parallel_state.get_data_parallel_rank", return_value=0),
        mock.patch("megatron.core.parallel_state.get_data_parallel_world_size", return_value=1),
    ):
        train_dataloader = datamodule.data_sampler.transform_dataloader(train_dataloader)

    assert isinstance(train_dataloader, DataLoader)

    # The training dataloader will drop the final batch with uneven sizes
    batches = list(train_dataloader)
    assert len(batches) == 1
    assert len(batches[0]["text"]) == 2


def test_padding_and_truncation(example_sequences):
    dataset = SimpleTokenizedDataset(example_sequences)
    datamodule = MockDataModule(
        train_dataset=dataset,
        pad_token_id=0,
        min_seq_length=4,
        max_seq_length=6,
        micro_batch_size=3,  # Process all sequences in one batch to observe padding
    )

    datamodule.trainer = mock.Mock()
    datamodule.trainer.global_step = 0

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Get the padded sequences tensor
    padded_sequences = batch["text"]

    # Check tensor shape and properties
    assert padded_sequences.shape[0] == 3  # Batch size
    assert padded_sequences.shape[1] >= 4  # At least min_length
    assert padded_sequences.shape[1] <= 6  # At most max_length


def test_validation_dataloader(tokenized_dataset):
    datamodule = MockDataModule(
        valid_dataset=tokenized_dataset,
        micro_batch_size=2,
        global_batch_size=2,
    )

    datamodule.trainer = mock.Mock()
    datamodule.trainer.global_step = 0

    val_loader = datamodule.val_dataloader()

    with (
        mock.patch("megatron.core.parallel_state.get_data_parallel_rank", return_value=0),
        mock.patch("megatron.core.parallel_state.get_data_parallel_world_size", return_value=1),
    ):
        val_loader = datamodule.data_sampler.transform_dataloader(val_loader)

    assert isinstance(val_loader, DataLoader)

    # The validation dataloader will drop the final batch with uneven sizes
    batches = list(val_loader)
    assert len(batches) == 1
    assert len(batches[0]["text"]) == 2


def test_test_dataloader(tokenized_dataset):
    datamodule = MockDataModule(
        test_dataset=tokenized_dataset,
        micro_batch_size=2,
        global_batch_size=2,
    )
    test_loader = datamodule.test_dataloader()

    with (
        mock.patch("megatron.core.parallel_state.get_data_parallel_rank", return_value=0),
        mock.patch("megatron.core.parallel_state.get_data_parallel_world_size", return_value=1),
    ):
        test_loader = datamodule.data_sampler.transform_dataloader(test_loader)

    assert isinstance(test_loader, DataLoader)

    # Validate that all samples are seen with uneven batch sizes
    batches = list(test_loader)
    assert len(batches) == 2
    assert len(batches[0]["text"]) == 2
    assert len(batches[1]["text"]) == 1


def test_predict_dataloader(tokenized_dataset):
    datamodule = MockDataModule(
        predict_dataset=tokenized_dataset,
        micro_batch_size=2,
        global_batch_size=2,
    )
    predict_loader = datamodule.predict_dataloader()

    with (
        mock.patch("megatron.core.parallel_state.get_data_parallel_rank", return_value=0),
        mock.patch("megatron.core.parallel_state.get_data_parallel_world_size", return_value=1),
    ):
        predict_loader = datamodule.data_sampler.transform_dataloader(predict_loader)

    assert isinstance(predict_loader, DataLoader)

    # Validate that all samples are seen with uneven batch sizes
    batches = list(predict_loader)
    assert len(batches) == 2
    assert len(batches[0]["text"]) == 2
    assert len(batches[1]["text"]) == 1


def test_missing_datasets():
    datamodule = MockDataModule()

    with pytest.raises(ValueError, match="No train_dataset was provided"):
        datamodule.train_dataloader()

    with pytest.raises(ValueError, match="No valid_dataset was provided"):
        datamodule.val_dataloader()

    with pytest.raises(ValueError, match="No test_dataset was provided"):
        datamodule.test_dataloader()


def test_batch_collation(example_sequences):
    dataset = SimpleTokenizedDataset(example_sequences)
    datamodule = MockDataModule(
        train_dataset=dataset,
        micro_batch_size=3,  # Process all sequences in one batch
    )

    datamodule.trainer = mock.Mock()
    datamodule.trainer.global_step = 0

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Check that we get a proper batch tensor
    assert isinstance(batch["text"], torch.Tensor)
    # Check batch size
    assert batch["text"].shape[0] == 3
    # Check that sequences are padded to the length of the longest sequence
    assert batch["text"].shape[1] == 7  # Length of longest sequence
