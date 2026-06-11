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
import torch.utils.data
from nemo.lightning.data import MegatronPretrainingRandomSampler, MegatronPretrainingSampler

from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.testing.torch import recursive_assert_approx_equal


def test_create_esm_datamodule_raises_without_trainer(dummy_protein_dataset, dummy_parquet_train_val_inputs):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
    )
    assert data_module is not None

    with pytest.raises(RuntimeError, match="Setup should be completed when trainer and config are attached."):
        data_module.setup()


def test_esm_datamodule_sets_min_seq_len_to_max_seq_len(dummy_protein_dataset, dummy_parquet_train_val_inputs):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        max_seq_length=36,
    )

    assert data_module._min_seq_length == 36


def test_create_esm_datamodule_raises_without_trainer_max_steps(dummy_protein_dataset, dummy_parquet_train_val_inputs):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 0

    with pytest.raises(RuntimeError, match="Please specify trainer.max_steps"):
        data_module.setup()


def test_create_esm_datamodule_creates_valid_dataloaders(dummy_protein_dataset, dummy_parquet_train_val_inputs):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=2,
        micro_batch_size=4,
        min_seq_length=36,
        max_seq_length=36,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 10
    data_module.trainer.val_check_interval = 2
    data_module.trainer.limit_val_batches = 1

    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)

    val_dataloader = data_module.val_dataloader()
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)

    assert len(train_dataloader) == 10 * 2  # max steps * global batch size
    assert len(val_dataloader) == 2  # global batch size; index reset every val epoch

    for batch in train_dataloader:
        assert isinstance(batch, dict)
        assert isinstance(batch["text"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)
        assert isinstance(batch["loss_mask"], torch.Tensor)
        assert isinstance(batch["is_random"], torch.Tensor)

    for batch in val_dataloader:
        assert isinstance(batch, dict)
        assert isinstance(batch["text"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)
        assert isinstance(batch["loss_mask"], torch.Tensor)
        assert isinstance(batch["is_random"], torch.Tensor)


def test_create_esm_datamodule_creates_valid_dataloaders_with_fractional_limit_val_batches(
    dummy_protein_dataset, dummy_parquet_train_val_inputs
):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=2,
        micro_batch_size=1,
        min_seq_length=36,
        max_seq_length=36,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 10
    data_module.trainer.val_check_interval = 2
    data_module.trainer.limit_val_batches = 0.5  # fractional value

    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)

    val_dataloader = data_module.val_dataloader()
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)

    assert len(train_dataloader) == 10 * 2  # max steps * global batch size
    assert len(val_dataloader) == int(4 * 0.5) // 1  # number of validation clusters // global batch size


def test_create_esm_datamodule_creates_valid_dataloaders_fractional_limit_val_batches_smaller_than_global_batch_size(
    dummy_protein_dataset, dummy_parquet_train_val_inputs
):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=8,
        micro_batch_size=4,
        min_seq_length=36,
        max_seq_length=36,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 10
    data_module.trainer.val_check_interval = 2
    data_module.trainer.limit_val_batches = 0.5  # fractional value

    # num_val_cluster * limit_val_batches = 4 * 0.5 = 1 < global_batch_size
    with pytest.raises(ValueError, match="The limited number of val samples 2 is less than the global batch size 8"):
        data_module.setup()


def test_create_esm_datamodule_creates_valid_dataloaders_fractional_limit_val_batches_not_multiple_of_global_batch_size(
    dummy_protein_dataset, dummy_parquet_train_val_inputs
):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=1,
        micro_batch_size=1,
        min_seq_length=36,
        max_seq_length=36,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 10
    data_module.trainer.val_check_interval = 2
    data_module.trainer.limit_val_batches = 0.7  # fractional value

    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)

    val_dataloader = data_module.val_dataloader()
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)

    assert len(train_dataloader) == 10 * 1  # max steps * global batch size
    assert len(val_dataloader) == int(4 * 0.7) // 1  # number of validation clusters // global batch size


def test_create_esm_datamodule_creates_valid_dataloaders_fractional_limit_val_batches_1p0(
    dummy_protein_dataset, dummy_parquet_train_val_inputs
):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=1,
        micro_batch_size=1,
        min_seq_length=36,
        max_seq_length=36,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 10
    data_module.trainer.val_check_interval = 2
    data_module.trainer.limit_val_batches = 1.0  # fractional value to use the whole dataset

    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)

    val_dataloader = data_module.val_dataloader()
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)

    assert len(train_dataloader) == 10 * 1  # max steps * global batch size
    assert len(val_dataloader) == 4 // 1  # number of validation clusters // global batch size


def test_create_esm_datamodule_limit_val_batches_none_equals_limit_val_batches_1p0(
    dummy_protein_dataset, dummy_parquet_train_val_inputs
):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    # Initialize the data module with limit_val_batches = 1.0
    data_module_one = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=1,
        micro_batch_size=1,
        min_seq_length=36,
        max_seq_length=36,
    )
    assert data_module_one is not None

    data_module_one.trainer = mock.Mock()
    data_module_one.trainer.max_epochs = 1
    data_module_one.trainer.max_steps = 10
    data_module_one.trainer.val_check_interval = 2
    data_module_one.trainer.limit_val_batches = 1.0  # fractional value to use the whole dataset

    data_module_one.setup()

    # Initialize the data module with limit_val_batches = None
    data_module_none = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=1,
        micro_batch_size=1,
        min_seq_length=36,
        max_seq_length=36,
    )
    assert data_module_none is not None

    data_module_none.trainer = mock.Mock()
    data_module_none.trainer.max_epochs = 1
    data_module_none.trainer.max_steps = 10
    data_module_none.trainer.val_check_interval = 2
    data_module_none.trainer.limit_val_batches = None  # None to use the whole dataset

    data_module_none.setup()

    # Check that the two dataloaders have the same number of samples.
    assert len(data_module_one.val_dataloader()) == len(data_module_none.val_dataloader())


def test_create_esm_datamodule_valid_dataloaders_has_consistent_samples_per_epoch(
    dummy_protein_dataset, dummy_parquet_train_val_inputs
):
    """
    Test that the ESMDataModule dataloaders produce consistent samples per epoch.

    This test ensures that the ESMDataModule creates dataloaders that produce consistent
    samples across epochs, even if the data is reshuffled (controlled by `is_ordered`).

    Parameters:
    - dummy_protein_dataset: A dummy protein dataset used for testing.
    - dummy_parquet_train_val_inputs: A tuple containing paths to dummy parquet files
      for training and validation clusters.
    """
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs
    micro_batch_size = 2

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=1,
        micro_batch_size=micro_batch_size,
        min_seq_length=36,
        max_seq_length=36,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 1
    data_module.trainer.val_check_interval = 1
    data_module.trainer.limit_val_batches = 1.0  # use the whole validation dataset

    data_module.setup()

    # Make sure two passes through the val_dataloader are identical.
    batches_1 = list(data_module.val_dataloader())
    recursive_assert_approx_equal(batches_1, data_module.val_dataloader())


def test_create_esm_datamodule_train_dataloaders_with_sequential_epoch_sampling(
    dummy_protein_dataset, dummy_parquet_train_val_inputs
):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs
    micro_batch_size = 2

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=2,
        micro_batch_size=micro_batch_size,
        min_seq_length=36,
        max_seq_length=36,
        dataloader_type="single",
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 10
    data_module.trainer.val_check_interval = 2
    data_module.trainer.limit_val_batches = None  # None to use the whole dataset
    data_module.setup()

    with (
        mock.patch("megatron.core.parallel_state.get_data_parallel_rank", return_value=0),
        mock.patch("megatron.core.parallel_state.get_data_parallel_world_size", return_value=1),
    ):
        train_dataloader = data_module.data_sampler.transform_dataloader(data_module.train_dataloader())

    assert isinstance(train_dataloader.batch_sampler, MegatronPretrainingSampler)

    # Make sure two passes through the train_dataloader are identical.
    batches_1 = list(train_dataloader)
    recursive_assert_approx_equal(batches_1, train_dataloader)


def test_create_esm_datamodule_train_dataloaders_with_random_epoch_sampling(
    dummy_protein_dataset, dummy_parquet_train_val_inputs
):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs
    micro_batch_size = 2

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=2,
        micro_batch_size=micro_batch_size,
        min_seq_length=36,
        max_seq_length=36,
        dataloader_type="cyclic",
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 10
    data_module.trainer.val_check_interval = 2
    data_module.trainer.limit_val_batches = None  # None to use the whole dataset
    data_module.setup()

    with (
        mock.patch("megatron.core.parallel_state.get_data_parallel_rank", return_value=0),
        mock.patch("megatron.core.parallel_state.get_data_parallel_world_size", return_value=1),
    ):
        train_dataloader = data_module.data_sampler.transform_dataloader(data_module.train_dataloader())

    assert isinstance(train_dataloader.batch_sampler, MegatronPretrainingRandomSampler)

    batches_1 = list(train_dataloader)

    for batch1, batch2 in zip(batches_1, train_dataloader, strict=True):
        with pytest.raises(AssertionError):
            recursive_assert_approx_equal(batch1, batch2)
