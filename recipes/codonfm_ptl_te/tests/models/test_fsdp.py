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

"""FSDP tests for EncodonPL model.

IMPORTANT: Multi-GPU distributed tests run the example training script as a subprocess.
This tests the full training infrastructure with FSDP on real hardware.
"""

import subprocess
from pathlib import Path

import pytest
import torch


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


@requires_multi_gpu
def test_encodon_pl_fsdp_training_2gpus(tmp_path):
    """Test EncodonPL training with FSDP on 2 GPUs using SimpleCodonDataset.

    This test runs the full training infrastructure to verify FSDP works correctly
    with SimpleCodonDataset and the runner.py infrastructure.
    """
    # Get path to the recipe root
    test_dir = Path(__file__).parent.parent.parent  # Go up to recipe root

    # Build the training command
    cmd = [
        "python",
        "src/runner.py",
        "pretrain",
        "--exp_name",
        "fsdp_test",
        "--dataset_name",
        "SimpleCodonDataset",
        "--process_item",
        "mlm_memmap",
        "--model_name",
        "encodon_200k",
        "--use_transformer_engine",
        "--enable_fsdp",
        "--num_nodes",
        "1",
        "--num_gpus",
        "2",
        "--context_length",
        "64",
        "--train_batch_size",
        "1",
        "--val_batch_size",
        "1",
        "--max_steps",
        "100",
        "--lr",
        "1e-4",
        "--warmup_iterations",
        "10",
        "--log_every_n_steps",
        "10",
        "--val_check_interval",
        "50",
        "--limit_val_batches",
        "5",
        "--mlm_probability",
        "0.15",
        "--mask_replace_prob",
        "0.8",
        "--random_replace_prob",
        "0.1",
        "--num_workers",
        "2",
    ]

    # Run the training command
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,  # 5 minutes should be enough for 100 steps
        cwd=str(test_dir),
    )
    # Verify training completed successfully
    assert result.returncode == 0, "Training should complete successfully"


@requires_multi_gpu
def test_encodon_pl_fsdp_training_2gpus_no_te(tmp_path):
    """Test EncodonPL training with FSDP on 2 GPUs without transformer engine.

    This test runs the full training infrastructure to verify FSDP works correctly
    with SimpleCodonDataset and the runner.py infrastructure without transformer engine.
    """
    # Get path to the recipe root
    test_dir = Path(__file__).parent.parent.parent  # Go up to recipe root

    # Build the training command
    cmd = [
        "python",
        "src/runner.py",
        "pretrain",
        "--exp_name",
        "fsdp_test_no_te",
        "--dataset_name",
        "SimpleCodonDataset",
        "--process_item",
        "mlm_memmap",
        "--model_name",
        "encodon_200k",
        "--enable_fsdp",
        "--num_nodes",
        "1",
        "--num_gpus",
        "2",
        "--context_length",
        "64",
        "--train_batch_size",
        "1",
        "--val_batch_size",
        "1",
        "--max_steps",
        "100",
        "--lr",
        "1e-4",
        "--warmup_iterations",
        "10",
        "--log_every_n_steps",
        "10",
        "--val_check_interval",
        "50",
        "--limit_val_batches",
        "5",
        "--mlm_probability",
        "0.15",
        "--mask_replace_prob",
        "0.8",
        "--random_replace_prob",
        "0.1",
        "--num_workers",
        "2",
    ]

    # Run the training command
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,  # 5 minutes should be enough for 100 steps
        cwd=str(test_dir),
    )
    # Verify training completed successfully
    assert result.returncode == 0, "Training should complete successfully"
