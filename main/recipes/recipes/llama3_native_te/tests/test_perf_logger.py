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

"""Tests for PerfLogger loss calculation correctness."""

from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf
from transformers.modeling_outputs import CausalLMOutputWithPast

from distributed_config import DistributedConfig
from perf_logger import PerfLogger


def _make_args(logging_frequency=1, num_train_steps=100):
    """Create a minimal args config for PerfLogger."""
    return OmegaConf.create(
        {
            "logger": {"frequency": logging_frequency},
            "wandb": {"project": "test", "mode": "disabled"},
            "num_train_steps": num_train_steps,
            "profiler": {"enabled": False},
            "quant_stats_config": {"enabled": False},
        }
    )


def _make_batch(seq_len=128, device="cuda:0"):
    """Create a minimal batch dict."""
    return {
        "input_ids": torch.ones(1, seq_len, dtype=torch.long, device=device),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long, device=device),
    }


def _make_outputs(loss_value, device="cuda:0"):
    """Create CausalLMOutputWithPast with a given loss."""
    return CausalLMOutputWithPast(loss=torch.tensor(loss_value, device=device))


@pytest.fixture
def mock_wandb():
    """Mock wandb to prevent actual logging."""
    with mock.patch("perf_logger.wandb") as mocked:
        mocked.init.return_value = mock.MagicMock()
        yield mocked


@pytest.fixture
def mock_tqdm():
    """Mock tqdm to prevent progress bar output."""
    with mock.patch("perf_logger.tqdm") as mocked:
        yield mocked


def _create_perf_logger(logging_frequency, mock_wandb, mock_tqdm):
    """Create a PerfLogger with the given logging_frequency."""
    dist_config = DistributedConfig()
    args = _make_args(logging_frequency=logging_frequency)
    return PerfLogger(dist_config, args, start_step=0)


def _run_steps(perf_logger, losses, grad_acc_steps=1):
    """Simulate training steps with given per-optimizer-step losses.

    Args:
        perf_logger: The PerfLogger instance.
        losses: List of loss values, one per optimizer step. With grad_acc_steps>1,
            each value is used for all micro steps in that optimizer step.
        grad_acc_steps: Number of micro steps per optimizer step.
    """
    device = perf_logger._device
    for step_idx, loss_val in enumerate(losses):
        step = step_idx + 1
        batch = _make_batch(device=device)
        outputs = _make_outputs(loss_val, device=device)
        for _ in range(grad_acc_steps):
            perf_logger.log_micro_step(step, batch, outputs)
        perf_logger.log_step(step, torch.tensor(1.0, device=device), 1e-4)


def _get_logged_losses(mock_wandb):
    """Extract reported loss values from wandb.log calls."""
    return [call[0][0]["train/loss"] for call in mock_wandb.log.call_args_list]


class TestPerfLoggerLoss:
    """Test that PerfLogger computes average loss correctly."""

    def test_logging_frequency_1_reports_each_loss(self, mock_wandb, mock_tqdm):
        """With logging_frequency=1, each step's loss should be reported as-is."""
        perf_logger = _create_perf_logger(1, mock_wandb, mock_tqdm)
        losses = [1.0, 2.0, 3.0, 4.0, 5.0]
        _run_steps(perf_logger, losses)

        reported = _get_logged_losses(mock_wandb)
        assert len(reported) == len(losses)
        for i, (got, expected) in enumerate(zip(reported, losses)):
            assert got == pytest.approx(expected), f"Step {i + 1}: expected {expected}, got {got}"

    def test_logging_frequency_5_matches_averaged_frequency_1(self, mock_wandb, mock_tqdm):
        """logging_frequency=5 should report the same average as manually averaging 5 frequency-1 losses."""
        losses = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        logging_freq = 5

        # Run with logging_frequency=1
        perf_logger_1 = _create_perf_logger(1, mock_wandb, mock_tqdm)
        _run_steps(perf_logger_1, losses)
        freq1_losses = _get_logged_losses(mock_wandb)
        assert len(freq1_losses) == 10

        # Compute expected averages over windows of size logging_freq
        expected = []
        for i in range(0, len(freq1_losses), logging_freq):
            window = freq1_losses[i : i + logging_freq]
            expected.append(sum(window) / len(window))

        # Run with logging_frequency=5
        mock_wandb.log.reset_mock()
        perf_logger_5 = _create_perf_logger(logging_freq, mock_wandb, mock_tqdm)
        _run_steps(perf_logger_5, losses)
        freq5_losses = _get_logged_losses(mock_wandb)

        assert len(freq5_losses) == len(expected), f"Expected {len(expected)} log events, got {len(freq5_losses)}"
        for i, (got, exp) in enumerate(zip(freq5_losses, expected)):
            assert got == pytest.approx(exp), f"Window {i}: expected {exp}, got {got}"

    def test_logging_frequency_with_grad_accumulation(self, mock_wandb, mock_tqdm):
        """Loss should be correct when combining gradient accumulation with logging_frequency > 1."""
        grad_acc_steps = 4
        logging_freq = 3
        # Each value is used for all micro steps in that optimizer step
        losses = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        # Run with logging_frequency=1 to get per-step losses
        perf_logger_1 = _create_perf_logger(1, mock_wandb, mock_tqdm)
        _run_steps(perf_logger_1, losses, grad_acc_steps=grad_acc_steps)
        freq1_losses = _get_logged_losses(mock_wandb)
        assert len(freq1_losses) == 6

        # Each step's loss should equal the input loss (all micro steps have same value)
        for i, (got, expected) in enumerate(zip(freq1_losses, losses)):
            assert got == pytest.approx(expected), f"Step {i + 1}: expected {expected}, got {got}"

        # Compute expected averages
        expected = []
        for i in range(0, len(freq1_losses), logging_freq):
            window = freq1_losses[i : i + logging_freq]
            expected.append(sum(window) / len(window))

        # Run with logging_frequency=logging_freq
        mock_wandb.log.reset_mock()
        perf_logger_n = _create_perf_logger(logging_freq, mock_wandb, mock_tqdm)
        _run_steps(perf_logger_n, losses, grad_acc_steps=grad_acc_steps)
        freqn_losses = _get_logged_losses(mock_wandb)

        assert len(freqn_losses) == len(expected)
        for i, (got, exp) in enumerate(zip(freqn_losses, expected)):
            assert got == pytest.approx(exp), f"Window {i}: expected {exp}, got {got}"

    def test_logging_frequency_with_varying_micro_losses(self, mock_wandb, mock_tqdm):
        """Test with different loss values across micro steps within a single optimizer step."""
        logging_freq = 2
        device = torch.device("cuda:0")

        perf_logger = _create_perf_logger(logging_freq, mock_wandb, mock_tqdm)

        # Step 1: micro losses [1.0, 3.0] → avg micro loss = 2.0
        for loss_val in [1.0, 3.0]:
            batch = _make_batch(device=device)
            outputs = _make_outputs(loss_val, device=device)
            perf_logger.log_micro_step(1, batch, outputs)
        perf_logger.log_step(1, torch.tensor(1.0, device=device), 1e-4)

        # Step 2: micro losses [5.0, 7.0] → avg micro loss = 6.0
        # Window of 2 steps: avg = (2.0 + 6.0) / 2 = 4.0
        for loss_val in [5.0, 7.0]:
            batch = _make_batch(device=device)
            outputs = _make_outputs(loss_val, device=device)
            perf_logger.log_micro_step(2, batch, outputs)
        perf_logger.log_step(2, torch.tensor(1.0, device=device), 1e-4)

        reported = _get_logged_losses(mock_wandb)
        assert len(reported) == 1
        # Total running_loss = 1.0 + 3.0 + 5.0 + 7.0 = 16.0
        # grad_acc_step_count = 4 (2 micro steps * 2 optimizer steps)
        # avg = 16.0 / 4 = 4.0
        assert reported[0] == pytest.approx(4.0), f"Expected 4.0, got {reported[0]}"

    def test_min_loss_tracked_correctly(self, mock_wandb, mock_tqdm):
        """min_loss should track the true minimum average loss across windows."""
        perf_logger = _create_perf_logger(1, mock_wandb, mock_tqdm)
        losses = [5.0, 2.0, 8.0, 1.0, 4.0]
        _run_steps(perf_logger, losses)

        assert perf_logger.min_loss.item() == pytest.approx(1.0)
