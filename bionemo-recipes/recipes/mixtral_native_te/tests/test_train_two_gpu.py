# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess

import pytest
import torch


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
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
def test_multi_gpu_train_ddp(recipe_path, local_tokenizer_path):
    """Test DDP training on 2 GPUs."""
    run_train_cmd(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            "2",
            "train_ddp.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",
            "expert_parallel_size=1",
            f"dataset.tokenizer_name_or_path={local_tokenizer_path}",
        ],
        recipe_path,
    )


@requires_multi_gpu
def test_multi_gpu_train_fsdp2(recipe_path, local_tokenizer_path):
    run_train_cmd(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            "2",
            "train_fsdp2.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",
            f"dataset.tokenizer_name_or_path={local_tokenizer_path}",
        ],
        recipe_path,
    )


@requires_multi_gpu
def test_multi_gpu_train_fsdp2_with_checkpointing(tmp_path, recipe_path, local_tokenizer_path):
    """Test FSDP2 training on 2 GPUs with checkpoint saving."""
    run_train_cmd(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            "2",
            "train_fsdp2.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=10",
            f"checkpoint.ckpt_dir={tmp_path}",
            "checkpoint.save_every_n_steps=5",
            "dataset.use_stateful_dataloader=true",
            "expert_parallel_size=1",
            f"dataset.tokenizer_name_or_path={local_tokenizer_path}",
        ],
        recipe_path,
    )

    # Verify checkpoint was created
    ckpt_dir = tmp_path / "train_fsdp2"
    assert ckpt_dir.exists(), f"Checkpoint directory not created: {ckpt_dir}"
    assert (ckpt_dir / "step_5").exists(), "Checkpoint at step 5 not found"


@requires_multi_gpu
def test_multi_gpu_train_fsdp2_ep2(recipe_path, local_tokenizer_path):
    """Test FSDP2 training with expert parallelism (EP=2) on 2 GPUs."""
    run_train_cmd(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            "2",
            "train_fsdp2.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",
            "expert_parallel_size=2",
            f"dataset.tokenizer_name_or_path={local_tokenizer_path}",
        ],
        recipe_path,
    )
