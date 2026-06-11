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
from datasets import Dataset

from bionemo.amplify import tokenizer
from bionemo.amplify.datamodule import AMPLIFYDataModule
from bionemo.esm2.data.dataset import RandomMaskStrategy


@pytest.fixture
def dummy_train_hf_dataset():
    """Create a mock HuggingFace training dataset with protein sequences."""
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "DEFGHIKLMNPQRSTVWYAC",
        "MGHIKLMNPQRSTVWYACDE",
        "MKTVRQERLKSIVRI",
        "MRILERSKEPVSGAQLA",
    ]
    return Dataset.from_dict({"sequence": sequences})


@pytest.fixture
def dummy_valid_hf_dataset():
    """Create a mock HuggingFace validation dataset with protein sequences."""
    sequences = [
        "KSTVRQERLKSIVRIM",
        "RPILERSKEPASGAQLA",
        "ADKLPQRSTVWY",
        "FCGHIKLMNPST",
    ]
    return Dataset.from_dict({"sequence": sequences})


@pytest.fixture
def amplify_tokenizer():
    """Return the AMPLIFY tokenizer."""
    return tokenizer.BioNeMoAMPLIFYTokenizer()


def test_amplify_datamodule_raises_without_trainer(dummy_train_hf_dataset, dummy_valid_hf_dataset, amplify_tokenizer):
    """Test that AMPLIFYDataModule raises an error when setup is called without a trainer."""
    # Initialize the data module
    data_module = AMPLIFYDataModule(
        train_hf_dataset=dummy_train_hf_dataset,
        valid_hf_dataset=dummy_valid_hf_dataset,
        tokenizer=amplify_tokenizer,
    )
    assert data_module is not None

    with pytest.raises(RuntimeError, match="Setup should be completed when trainer and config are attached."):
        data_module.setup()


def test_amplify_datamodule_raises_without_trainer_max_steps(
    dummy_train_hf_dataset, dummy_valid_hf_dataset, amplify_tokenizer
):
    """Test that AMPLIFYDataModule raises an error when trainer.max_steps is not set."""
    # Initialize the data module
    data_module = AMPLIFYDataModule(
        train_hf_dataset=dummy_train_hf_dataset,
        valid_hf_dataset=dummy_valid_hf_dataset,
        tokenizer=amplify_tokenizer,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 0

    with pytest.raises(RuntimeError, match="Please specify trainer.max_steps"):
        data_module.setup()


def test_amplify_datamodule_creates_valid_dataloaders(
    dummy_train_hf_dataset, dummy_valid_hf_dataset, amplify_tokenizer
):
    """Test that AMPLIFYDataModule creates valid dataloaders."""
    # Initialize the data module
    data_module = AMPLIFYDataModule(
        train_hf_dataset=dummy_train_hf_dataset,
        valid_hf_dataset=dummy_valid_hf_dataset,
        global_batch_size=2,
        micro_batch_size=1,
        min_seq_length=None,
        max_seq_length=36,
        tokenizer=amplify_tokenizer,
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

    # Check batch structure
    for batch in train_dataloader:
        assert isinstance(batch, dict)
        assert isinstance(batch["text"], torch.Tensor)
        assert isinstance(batch["types"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)
        assert isinstance(batch["loss_mask"], torch.Tensor)
        assert isinstance(batch["is_random"], torch.Tensor)

    for batch in val_dataloader:
        assert isinstance(batch, dict)
        assert isinstance(batch["text"], torch.Tensor)
        assert isinstance(batch["types"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)
        assert isinstance(batch["loss_mask"], torch.Tensor)
        assert isinstance(batch["is_random"], torch.Tensor)


def test_amplify_datamodule_with_different_mask_prob(
    dummy_train_hf_dataset, dummy_valid_hf_dataset, amplify_tokenizer
):
    """Test that AMPLIFYDataModule works with different mask probabilities."""
    mask_prob = 0.2  # Different mask probability

    # Initialize the data module
    data_module = AMPLIFYDataModule(
        train_hf_dataset=dummy_train_hf_dataset,
        valid_hf_dataset=dummy_valid_hf_dataset,
        global_batch_size=2,
        micro_batch_size=1,
        min_seq_length=None,
        max_seq_length=36,
        tokenizer=amplify_tokenizer,
        mask_prob=mask_prob,
        mask_token_prob=0.8,
        mask_random_prob=0.1,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 10
    data_module.trainer.val_check_interval = 2
    data_module.trainer.limit_val_batches = 1

    data_module.setup()

    # Verify the mask_prob was set correctly in the datamodule
    assert data_module._mask_prob == mask_prob


def test_amplify_datamodule_with_different_random_mask_strategy(
    dummy_train_hf_dataset, dummy_valid_hf_dataset, amplify_tokenizer
):
    """Test that AMPLIFYDataModule works with different random mask strategies."""
    custom_strategy = RandomMaskStrategy.ALL_TOKENS  # Different random mask strategy

    # Initialize the data module
    data_module = AMPLIFYDataModule(
        train_hf_dataset=dummy_train_hf_dataset,
        valid_hf_dataset=dummy_valid_hf_dataset,
        global_batch_size=2,
        micro_batch_size=1,
        min_seq_length=None,
        max_seq_length=36,
        tokenizer=amplify_tokenizer,
        random_mask_strategy=custom_strategy,
    )
    assert data_module is not None

    data_module.trainer = mock.Mock()
    data_module.trainer.max_epochs = 1
    data_module.trainer.max_steps = 10
    data_module.trainer.val_check_interval = 2
    data_module.trainer.limit_val_batches = 1

    data_module.setup()

    # Verify the random_mask_strategy was set correctly in the datamodule
    assert data_module._random_mask_strategy == custom_strategy


def test_amplify_datamodule_with_min_seq_length(dummy_train_hf_dataset, dummy_valid_hf_dataset, amplify_tokenizer):
    """Test that AMPLIFYDataModule works with min_seq_length."""
    min_seq_length = 20

    # Initialize the data module
    data_module = AMPLIFYDataModule(
        train_hf_dataset=dummy_train_hf_dataset,
        valid_hf_dataset=dummy_valid_hf_dataset,
        global_batch_size=2,
        micro_batch_size=1,
        min_seq_length=min_seq_length,
        max_seq_length=36,
        tokenizer=amplify_tokenizer,
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

    # Check that min_seq_length is set correctly
    assert data_module._min_seq_length == min_seq_length

    # Get a batch and check its size
    for batch in train_dataloader:
        # Check that padding was applied to meet min_seq_length
        assert batch["text"].size(1) >= min_seq_length
        break


def test_amplify_datamodule_tokenizer_property(dummy_train_hf_dataset, dummy_valid_hf_dataset, amplify_tokenizer):
    """Test that AMPLIFYDataModule.tokenizer property returns the correct tokenizer."""
    # Initialize the data module
    data_module = AMPLIFYDataModule(
        train_hf_dataset=dummy_train_hf_dataset,
        valid_hf_dataset=dummy_valid_hf_dataset,
        tokenizer=amplify_tokenizer,
    )

    assert data_module.tokenizer is amplify_tokenizer


def test_amplify_datamodule_test_dataloader_raises_not_implemented(
    dummy_train_hf_dataset, dummy_valid_hf_dataset, amplify_tokenizer
):
    """Test that AMPLIFYDataModule.test_dataloader raises NotImplementedError."""
    # Initialize the data module
    data_module = AMPLIFYDataModule(
        train_hf_dataset=dummy_train_hf_dataset,
        valid_hf_dataset=dummy_valid_hf_dataset,
        tokenizer=amplify_tokenizer,
    )

    with pytest.raises(NotImplementedError, match="No test dataset provided for AMPLIFY"):
        data_module.test_dataloader()
