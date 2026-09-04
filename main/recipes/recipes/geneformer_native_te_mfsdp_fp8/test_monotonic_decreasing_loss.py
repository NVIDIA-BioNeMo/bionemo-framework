#!/usr/bin/env python3

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

"""Test for monotonic decreasing loss during training.

This test runs actual training steps and verifies that the loss shows a decreasing trend,
which is a key indicator that the model is learning correctly.

To run. Remove once this is in CI.
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=0 python -m pytest test_monotonic_decreasing_loss.py -v

"""

import logging
import os
import re
import subprocess
import sys
import unittest

import pytest
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_loss_values(log_output):
    """Parse loss values from training log output, ensuring correct step order."""
    step_loss_pairs = []
    # Look for pattern like "Step 0 loss: 10.5, grad_norm: 1.2, lr: 0.0001"
    pattern = r"Step (\d+) loss: ([\d.]+)"
    matches = re.findall(pattern, log_output)

    for step_str, loss_str in matches:
        try:
            step = int(step_str)
            loss = float(loss_str)
            step_loss_pairs.append((step, loss))
        except ValueError:
            continue

    # Sort by step number to ensure correct order
    step_loss_pairs.sort(key=lambda x: x[0])

    # Return just the loss values in correct step order
    return [loss for step, loss in step_loss_pairs]


def check_loss_decreasing_trend(loss_values):
    """Check if loss values show a general decreasing trend."""
    if len(loss_values) < 3:
        return False

    # Compare first third with last third of values
    n = len(loss_values)
    first_third = loss_values[: n // 3] if n // 3 > 0 else loss_values[:1]
    last_third = loss_values[-n // 3 :] if n // 3 > 0 else loss_values[-1:]

    avg_first = sum(first_third) / len(first_third)
    avg_last = sum(last_third) / len(last_third)

    # Check if average of last third is lower than first third
    return avg_last < avg_first


class TestMonotonicDecreasingLoss(unittest.TestCase):
    """Test that training loss shows a decreasing trend over time."""

    def test_loss_parsing_functions(self):
        """Test that the loss parsing and trend checking functions work correctly."""
        # Test with ordered, decreasing loss pattern
        mock_log_output = """
        [2025-07-15 20:43:05,144][__main__][INFO] - Step 0 loss: 10.5, grad_norm: 0.72, lr: 0.0001
        [2025-07-15 20:43:05,193][__main__][INFO] - Step 1 loss: 10.2, grad_norm: 0.74, lr: 0.0001
        [2025-07-15 20:43:05,243][__main__][INFO] - Step 2 loss: 9.8, grad_norm: 0.70, lr: 0.0001
        [2025-07-15 20:43:05,295][__main__][INFO] - Step 3 loss: 9.5, grad_norm: 0.66, lr: 0.0001
        [2025-07-15 20:43:05,344][__main__][INFO] - Step 4 loss: 9.1, grad_norm: 0.68, lr: 0.0001
        [2025-07-15 20:43:05,393][__main__][INFO] - Step 5 loss: 8.8, grad_norm: 0.63, lr: 0.0001
        """

        loss_values = parse_loss_values(mock_log_output)
        expected_losses = [10.5, 10.2, 9.8, 9.5, 9.1, 8.8]

        self.assertEqual(len(loss_values), 6)
        self.assertEqual(loss_values, expected_losses)

        # Test that it detects decreasing trend
        is_decreasing = check_loss_decreasing_trend(loss_values)
        self.assertTrue(is_decreasing)

        # Test with non-decreasing pattern
        non_decreasing_losses = [8.0, 9.0, 10.0, 11.0, 12.0]
        is_not_decreasing = check_loss_decreasing_trend(non_decreasing_losses)
        self.assertFalse(is_not_decreasing)

        logger.info("✅ Loss parsing function tests passed")

    def _run_training_with_config(self, config_name, kwargs=None):
        """Helper method to run training with a specific config and return results."""
        logger.info(f"🚀 Starting training test with config: {config_name}")

        # Set environment variables for single GPU training
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",  # Force single GPU
                "LOCAL_RANK": "0",
                "RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12355",
                "WANDB_MODE": "disabled",
            }
        )

        # Run the training script with the specified config and override num_train_steps
        cmd = [
            sys.executable,
            "train.py",
            "--config-name",
            config_name,
            *kwargs,
            "training.num_train_steps=50",  # Override to 50 steps regardless of config
            "training.resume_from_checkpoint=false",
        ]

        try:
            result = subprocess.run(
                cmd,
                check=False,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            # Check if training completed successfully
            if result.returncode != 0:
                logger.error(f"Training failed for config {config_name}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return None, f"Training failed with stderr: {result.stderr}"

            # Parse loss values from combined output
            combined_output = result.stdout + result.stderr
            loss_values = parse_loss_values(combined_output)

            logger.info(f"📊 Found {len(loss_values)} loss values from {config_name} training")

            return loss_values, None

        except subprocess.TimeoutExpired:
            return None, "Training timed out after 10 minutes"
        except Exception as e:
            return None, f"Training test failed with error: {e}"

    @pytest.mark.slow
    def test_sanity_config_loss_decreases(self):
        """Test that sanity config shows decreasing loss trend."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available - skipping real training test")

        loss_values, error = self._run_training_with_config(
            "l0_sanity",
            ["model.use_te_layers=false", "training.use_mfsdp=false", "training.resume_from_checkpoint=false"],
        )

        if error:
            self.fail(f"sanity config: {error}")

        # Ensure loss_values is not None before proceeding
        self.assertIsNotNone(loss_values, "Loss values should not be None")
        # Type narrowing for linter - we know loss_values is not None after the assert
        assert loss_values is not None

        # Verify we have enough loss values for meaningful analysis
        self.assertGreaterEqual(
            len(loss_values), 50, f"Expected at least 50 loss values for trend analysis, got {len(loss_values)}"
        )

        # Check if loss shows decreasing trend
        is_decreasing = check_loss_decreasing_trend(loss_values)

        # Log the results
        if is_decreasing:
            logger.info("✅ sanity: Loss shows decreasing trend - model is learning!")
            logger.info(f"📊 sanity: First loss: {loss_values[0]:.4f}, Last loss: {loss_values[-1]:.4f}")
        else:
            logger.warning("⚠️ sanity: Loss does not show clear decreasing trend")
            logger.info(f"📊 sanity: First 10 values: {loss_values[:10]}")
            logger.info(f"📊 sanity: Last 10 values: {loss_values[-10:]}")

        # Assert that the loss is decreasing
        self.assertTrue(is_decreasing, f"sanity config: Loss should show decreasing trend. Loss values: {loss_values}")

    @pytest.mark.slow
    def test_sanity_te_config_loss_decreases(self):
        """Test that sanity_te config shows decreasing loss trend."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available - skipping real training test")

        loss_values, error = self._run_training_with_config(
            "l0_sanity", ["model.use_te_layers=true", "training.use_mfsdp=false"]
        )

        if error:
            self.fail(f"sanity_te config: {error}")

        # Ensure loss_values is not None before proceeding
        self.assertIsNotNone(loss_values, "Loss values should not be None")
        # Type narrowing for linter - we know loss_values is not None after the assert
        assert loss_values is not None

        # Verify we have enough loss values for meaningful analysis
        self.assertGreaterEqual(
            len(loss_values), 50, f"Expected at least 50 loss values for trend analysis, got {len(loss_values)}"
        )

        # Check if loss shows decreasing trend
        is_decreasing = check_loss_decreasing_trend(loss_values)

        # Log the results
        if is_decreasing:
            logger.info("✅ sanity_te: Loss shows decreasing trend - model is learning!")
            logger.info(f"📊 sanity_te: First loss: {loss_values[0]:.4f}, Last loss: {loss_values[-1]:.4f}")
        else:
            logger.warning("⚠️ sanity_te: Loss does not show clear decreasing trend")
            logger.info(f"📊 sanity_te: First 10 values: {loss_values[:10]}")
            logger.info(f"📊 sanity_te: Last 10 values: {loss_values[-10:]}")

        # Assert that the loss is decreasing
        self.assertTrue(
            is_decreasing, f"sanity_te config: Loss should show decreasing trend. Loss values: {loss_values}"
        )

    @pytest.mark.slow
    def test_sanity_te_mfsdp_config_loss_decreases(self):
        """Test that sanity_te_mfsdp config shows decreasing loss trend."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available - skipping real training test")

        loss_values, error = self._run_training_with_config(
            "l0_sanity", ["model.use_te_layers=true", "training.use_mfsdp=true", "training.save_final_model=false"]
        )

        if error:
            self.fail(f"sanity_te_mfsdp config: {error}")

        # Ensure loss_values is not None before proceeding
        self.assertIsNotNone(loss_values, "Loss values should not be None")

        # Verify we have enough loss values for meaningful analysis
        self.assertGreaterEqual(
            len(loss_values), 50, f"Expected at least 50 loss values for trend analysis, got {len(loss_values)}"
        )

        # Check if loss shows decreasing trend
        is_decreasing = check_loss_decreasing_trend(loss_values)

        # Log the results
        if is_decreasing:
            logger.info("✅ sanity_te_mfsdp: Loss shows decreasing trend - model is learning!")
            logger.info(f"📊 sanity_te_mfsdp: First loss: {loss_values[0]:.4f}, Last loss: {loss_values[-1]:.4f}")
        else:
            logger.warning("⚠️ sanity_te_mfsdp: Loss does not show clear decreasing trend")
            logger.info(f"📊 sanity_te_mfsdp: First 10 values: {loss_values[:10]}")
            logger.info(f"📊 sanity_te_mfsdp: Last 10 values: {loss_values[-10:]}")

        # Assert that the loss is decreasing
        self.assertTrue(
            is_decreasing, f"sanity_te_mfsdp config: Loss should show decreasing trend. Loss values: {loss_values}"
        )

    @pytest.mark.slow
    def test_sanity_mfsdp_config_loss_decreases(self):
        """Test that sanity_mfsdp config shows decreasing loss trend."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available - skipping real training test")

        loss_values, error = self._run_training_with_config(
            "l0_sanity", ["training.use_mfsdp=true", "model.use_te_layers=false", "training.save_final_model=false"]
        )

        if error:
            self.fail(f"sanity_mfsdp config: {error}")

        # Ensure loss_values is not None before proceeding
        self.assertIsNotNone(loss_values, "Loss values should not be None")

        # Verify we have enough loss values for meaningful analysis
        self.assertGreaterEqual(
            len(loss_values), 50, f"Expected at least 50 loss values for trend analysis, got {len(loss_values)}"
        )

        # Check if loss shows decreasing trend
        is_decreasing = check_loss_decreasing_trend(loss_values)

        # Log the results
        if is_decreasing:
            logger.info("✅ sanity_mfsdp: Loss shows decreasing trend - model is learning!")
            logger.info(f"📊 sanity_mfsdp: First loss: {loss_values[0]:.4f}, Last loss: {loss_values[-1]:.4f}")
        else:
            logger.warning("⚠️ sanity_mfsdp: Loss does not show clear decreasing trend")
            logger.info(f"📊 sanity_mfsdp: First 10 values: {loss_values[:10]}")
            logger.info(f"📊 sanity_mfsdp: Last 10 values: {loss_values[-10:]}")

        # Assert that the loss is decreasing
        self.assertTrue(
            is_decreasing, f"sanity_mfsdp config: Loss should show decreasing trend. Loss values: {loss_values}"
        )


if __name__ == "__main__":
    print("🧪 Testing Monotonic Decreasing Loss Across All Configurations")
    print("=" * 60)

    # Run the tests
    unittest.main(verbosity=2)
