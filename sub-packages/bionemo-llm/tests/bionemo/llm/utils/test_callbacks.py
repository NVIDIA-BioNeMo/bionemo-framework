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


import os
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch

from bionemo.llm.lightning import batch_collator
from bionemo.llm.utils.callbacks import PredictionWriter
from bionemo.testing import megatron_parallel_state_utils


# Fixture for temporary directory
@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def mock_trainer():
    trainer = MagicMock(spec=pl.Trainer)
    trainer.global_rank = 0
    return trainer


@pytest.fixture
def mock_module():
    return MagicMock(spec=pl.LightningModule)


@pytest.fixture
def sample_predictions():
    return [{"temp": torch.tensor([1, 2, 3])}, {"temp": torch.tensor([4, 5, 6])}, None]


@pytest.fixture
def collated_prediction(sample_predictions):
    return batch_collator([item for item in sample_predictions if item is not None])


@pytest.mark.parametrize("write_interval", ["batch", "epoch"])
def test_prediction_writer_init(temp_dir, write_interval):
    writer = PredictionWriter(output_dir=temp_dir, write_interval=write_interval)
    assert writer.output_dir == temp_dir
    if write_interval == "batch":
        assert writer.interval.on_batch
    if write_interval == "epoch":
        assert writer.interval.on_epoch


@patch("torch.save")
def test_write_on_batch_end(mock_torch_save, temp_dir, mock_trainer, mock_module, collated_prediction):
    writer = PredictionWriter(output_dir=temp_dir, write_interval="batch")

    batch_idx = 1
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        writer.write_on_batch_end(
            trainer=mock_trainer,
            pl_module=mock_module,
            prediction=collated_prediction,
            batch_indices=[],
            batch=None,
            batch_idx=batch_idx,
            dataloader_idx=0,
        )

    expected_path = os.path.join(
        temp_dir, f"predictions__rank_{mock_trainer.global_rank}__dp_rank_0__batch_{batch_idx}.pt"
    )
    mock_torch_save.assert_called_once_with(collated_prediction, expected_path)


@patch("torch.save")
def test_write_on_epoch_end(
    mock_torch_save, temp_dir, mock_trainer, mock_module, sample_predictions, collated_prediction
):
    writer = PredictionWriter(output_dir=temp_dir, write_interval="epoch")

    with megatron_parallel_state_utils.distributed_model_parallel_state():
        writer.write_on_epoch_end(
            trainer=mock_trainer,
            pl_module=mock_module,
            predictions=sample_predictions,
            batch_indices=[],
        )

    expected_path = os.path.join(temp_dir, f"predictions__rank_{mock_trainer.global_rank}__dp_rank_0.pt")

    mock_torch_save.assert_called_once()  # Ensure it's called exactly once

    # Extract the actual call arguments
    actual_args, actual_kwargs = mock_torch_save.call_args
    prediction = actual_args[0]
    assert actual_args[1] == expected_path, "Paths do not match"

    # Compare tensors manually
    assert isinstance(prediction, dict)
    for key in prediction:
        assert torch.equal(prediction[key], collated_prediction[key]), "Tensors do not match"
