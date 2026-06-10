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


import os
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.utils.pred_writer import PredWriter


@dataclass
class MockPrediction:
    logits: torch.Tensor
    labels: torch.Tensor


@pytest.fixture
def pred_writer(tmpdir):
    return PredWriter(output_dir=str(tmpdir), write_interval="batch", caching_interval=2)


@pytest.fixture
def pred_writer_for_merge(tmpdir):
    return PredWriter(
        output_dir=str(tmpdir), write_interval="epoch", merge_on_epoch_end=True, delete_after_merge=False
    )


class TestPredWriter:
    def test_convert_predictions(self, pred_writer):
        prediction = MockPrediction(logits=torch.randn(4, 10), labels=torch.randint(0, 10, (4,)))
        converted = pred_writer._convert_predictions(prediction)

        assert isinstance(converted, dict)
        assert "logits" in converted
        assert "labels" in converted
        assert isinstance(converted["logits"], np.ndarray)
        assert converted["logits"].shape == (4, 10)

    def test_write_on_batch_end_caching(self, pred_writer, tmpdir):
        trainer = MagicMock()
        trainer.global_rank = 0
        trainer.datamodule.init_consumed_samples = 0
        trainer.datamodule.global_batch_size = 4

        prediction = MockPrediction(logits=torch.ones(2, 5), labels=torch.ones(2))

        # First call, should be buffered
        pred_writer.write_on_batch_end(trainer, None, prediction, None, None, 0, 0)
        assert len(pred_writer.predictions_buffer) == 1
        assert len(os.listdir(str(tmpdir))) == 0

        # Second call, should trigger save
        pred_writer.write_on_batch_end(trainer, None, prediction, None, None, 1, 0)
        assert len(pred_writer.predictions_buffer) == 0
        assert len(os.listdir(str(tmpdir))) == 2  # logits and labels files

        # Check file contents
        logits_file = os.path.join(str(tmpdir), "logits_rank_0_batch_1.npy")
        labels_file = os.path.join(str(tmpdir), "labels_rank_0_batch_1.npy")
        assert os.path.exists(logits_file)
        assert os.path.exists(labels_file)

        logits_data = np.load(logits_file)
        assert logits_data.shape == (4, 5)  # 2 batches of size 2
        assert np.all(logits_data == 1)

    def test_merge_predictions_single_rank(self, pred_writer_for_merge, tmpdir):
        # Create dummy files
        logits_b0 = np.random.rand(10, 5)  # noqa: NPY002
        labels_b0 = np.random.rand(10)  # noqa: NPY002
        np.save(os.path.join(str(tmpdir), "logits_rank_0_batch_0.npy"), logits_b0)
        np.save(os.path.join(str(tmpdir), "labels_rank_0_batch_0.npy"), labels_b0)

        logits_b1 = np.random.rand(8, 5)  # noqa: NPY002
        labels_b1 = np.random.rand(8)  # noqa: NPY002
        np.save(os.path.join(str(tmpdir), "logits_rank_0_batch_1.npy"), logits_b1)
        np.save(os.path.join(str(tmpdir), "labels_rank_0_batch_1.npy"), labels_b1)

        pred_writer_for_merge._merge_predictions()

        merged_logits_file = os.path.join(str(tmpdir), "logits_merged.npy")
        merged_labels_file = os.path.join(str(tmpdir), "labels_merged.npy")
        assert os.path.exists(merged_logits_file)
        assert os.path.exists(merged_labels_file)

        merged_logits = np.load(merged_logits_file)
        merged_labels = np.load(merged_labels_file)

        assert merged_logits.shape == (18, 5)
        assert merged_labels.shape == (18,)
        assert np.allclose(merged_logits, np.concatenate([logits_b0, logits_b1]))
        assert np.allclose(merged_labels, np.concatenate([labels_b0, labels_b1]))

    def test_merge_predictions_multi_rank(self, pred_writer_for_merge, tmpdir):
        # Simulate a DDP run with 2 ranks and one batch
        # Total batch size 10. rank 0 gets 5, rank 1 gets 5
        # samples for rank 0: 0, 2, 4, 6, 8
        # samples for rank 1: 1, 3, 5, 7, 9
        original_data = np.arange(10).reshape(10, 1)

        data_r0 = original_data[[0, 2, 4, 6, 8]]
        data_r1 = original_data[[1, 3, 5, 7, 9]]

        np.save(os.path.join(str(tmpdir), "data_rank_0_batch_0.npy"), data_r0)
        np.save(os.path.join(str(tmpdir), "data_rank_1_batch_0.npy"), data_r1)

        pred_writer_for_merge._merge_predictions()

        merged_file = os.path.join(str(tmpdir), "data_merged.npy")
        assert os.path.exists(merged_file)

        merged_data = np.load(merged_file)
        assert merged_data.shape == (10, 1)
        assert np.all(merged_data == original_data)

    def test_delete_after_merge(self, pred_writer_for_merge, tmpdir):
        pred_writer_for_merge.delete_after_merge = True
        np.save(os.path.join(str(tmpdir), "data_rank_0_batch_0.npy"), np.array([1]))

        pred_writer_for_merge._merge_predictions()

        assert not os.path.exists(os.path.join(str(tmpdir), "data_rank_0_batch_0.npy"))
        assert os.path.exists(os.path.join(str(tmpdir), "data_merged.npy"))

    def test_on_predict_epoch_end(self, pred_writer_for_merge, tmpdir):
        trainer = MagicMock()
        trainer.is_global_zero = True
        trainer.strategy.barrier = MagicMock()
        trainer.global_rank = 0

        # Add some data to buffer
        pred_writer_for_merge.predictions_buffer.append({"logits": np.array([1])})

        # Mock datamodule for consumed samples calculation
        trainer.datamodule.calc_consumed_samples.return_value = 10
        trainer.datamodule.global_batch_size = 4

        pred_writer_for_merge.on_predict_epoch_end(trainer, None)

        # Buffer should be flushed and file saved
        assert os.path.exists(os.path.join(str(tmpdir), "logits_rank_0_batch_3.npy"))

        # Merge should be called
        assert os.path.exists(os.path.join(str(tmpdir), "logits_merged.npy"))
        trainer.strategy.barrier.assert_called_once()
