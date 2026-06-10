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

"""Multi-GPU training tests for OpenGenome2.

These tests validate that FSDP2 training works correctly with multiple GPUs.
They require at least 2 GPUs to run and will be skipped on single-GPU machines.
"""

import subprocess

import pytest
import torch


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

# TODO(@jomitchell): Delete once https://nvbugspro.nvidia.com/bug/5458694 is fixed.
requires_datacenter_hardware = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not any(
        gpu_name in torch.cuda.get_device_name(0).upper() for gpu_name in ["H100", "H200", "B100", "B200", "B300"]
    ),
    reason="Test requires datacenter hardware (H100, H200, B100, B200, B300)",
)


def run_train_cmd(cmd, recipe_path):
    """Run a training command and check for errors."""
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
        cwd=str(recipe_path),
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command:\n{' '.join(cmd)}\nfailed with exit code {result.returncode}")


@requires_multi_gpu
def test_multi_gpu_train_fsdp2(tmp_path, recipe_path):
    """Test FSDP2 training on 2 GPUs.

    Validates that FSDP2 launches, shards the model, and completes training
    without errors on multiple GPUs.
    """
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node",
            "2",
            "--standalone",
            "train_fsdp2.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",
        ],
        recipe_path,
    )


@requires_multi_gpu
def test_multi_gpu_train_fsdp2_with_checkpointing(tmp_path, recipe_path):
    """Test FSDP2 training on 2 GPUs with checkpoint saving.

    Validates that sharded checkpoints are created correctly across
    multiple processes without race conditions.
    """
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node",
            "2",
            "--standalone",
            "train_fsdp2.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=10",
            f"checkpoint.ckpt_dir={tmp_path}",
            "checkpoint.save_every_n_steps=5",
            "dataset.use_stateful_dataloader=true",
        ],
        recipe_path,
    )

    ckpt_dir = tmp_path / "train_fsdp2"
    assert ckpt_dir.exists(), f"Checkpoint directory not created: {ckpt_dir}"
    assert (ckpt_dir / "step_5").exists(), "Checkpoint at step 5 not found"


@requires_multi_gpu
def test_multi_gpu_train_te_fsdp2_cp_bshd(tmp_path, recipe_path):
    """Test FSDP2 with context parallelism on 2 GPUs using BSHD input format."""
    run_train_cmd(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "train_fsdp2_cp.py",
            "--config-name",
            "L0_sanity_cp",
            "num_train_steps=10",
            f"checkpoint.ckpt_dir={tmp_path}",
            "checkpoint.save_every_n_steps=5",
            "cp_size=2",
            "use_sequence_packing=false",
            "config_kwargs.attn_input_format=bshd",
            "config_kwargs.self_attn_mask_type=causal",
        ],
        recipe_path,
    )


@requires_multi_gpu
@requires_datacenter_hardware
def test_multi_gpu_train_te_fsdp2_cp_thd(tmp_path, recipe_path):
    """Test FSDP2 with context parallelism on 2 GPUs using THD input format."""
    run_train_cmd(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "train_fsdp2_cp.py",
            "--config-name",
            "L0_sanity_cp",
            "num_train_steps=10",
            f"checkpoint.ckpt_dir={tmp_path}",
            "checkpoint.save_every_n_steps=5",
            "cp_size=2",
            "use_sequence_packing=true",
            "config_kwargs.attn_input_format=thd",
            "config_kwargs.self_attn_mask_type=padding_causal",
        ],
        recipe_path,
    )
