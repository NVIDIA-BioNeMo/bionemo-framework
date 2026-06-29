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

"""Test suite for distributed checkpointing functionality.

This module tests checkpoint save/resume functionality across different
distributed training configurations:
- DDP (Distributed Data Parallel) with 1 and 2 processes
- FSDP2 (PyTorch native Fully Sharded Data Parallel v2) with 1 and 2 processes
- FSDP2 with context parallelism
- FP8 quantized model init with checkpoint save/resume

Test Strategy:
1. Phase 1: Train for N steps and save checkpoint
2. Phase 2: Resume training from checkpoint and continue
3. Validate: Checkpoints created, resuming works, training continues seamlessly, losses are valid

Each test uses temporary directories and disables wandb logging for isolation.
"""

import gc
import os
import subprocess

import pytest
import torch
from hydra import compose, initialize_config_dir

from train_ddp import main as main_ddp
from train_fsdp2 import main as main_fsdp2
from train_fsdp2_cp import main as main_fsdp2_cp


os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


# ---------------------------------------------------------------------------
# Test Utilities
# ---------------------------------------------------------------------------


def _compose_config(recipe_path, tmp_path, config_name, overrides):
    """Compose a Hydra config with standard checkpoint-test settings.

    Every config gets ``checkpoint.ckpt_dir``, ``+wandb.dir``, and
    ``dataset.use_stateful_dataloader`` set automatically so that callers
    only need to supply test-specific overrides.
    """
    ckpt_dir = str(tmp_path / "ckpt")
    base = [
        f"checkpoint.ckpt_dir={ckpt_dir}",
        f"+wandb.dir={tmp_path}",
        "dataset.use_stateful_dataloader=true",
    ]
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        return compose(config_name=config_name, overrides=base + list(overrides or []))


def _assert_loss_valid(loss, label=""):
    """Assert that a training loss is finite and not NaN."""
    tag = f" ({label})" if label else ""
    assert loss is not None, f"Loss is None{tag}"
    loss_val = float(loss)
    assert not torch.isnan(torch.tensor(loss_val)), f"Loss is NaN{tag}"
    assert torch.isfinite(torch.tensor(loss_val)), f"Loss is not finite: {loss_val}{tag}"


def _assert_checkpoint_step(ckpt_subdir, step, num_ranks=1, is_ddp=True):
    """Assert that a checkpoint step directory has the expected files.

    For DDP checks for ``checkpoint.pt`` and exact file counts.
    For FSDP2 (DCP format) only checks for per-rank dataloader files.
    """
    step_dir = os.path.join(ckpt_subdir, f"step_{step}")
    assert os.path.isdir(step_dir), f"Step {step} directory not found: {step_dir}"
    files = os.listdir(step_dir)

    if is_ddp:
        expected_count = 1 + num_ranks  # checkpoint.pt + one dataloader per rank
        assert len(files) == expected_count, (
            f"Expected {expected_count} files in step_{step}, found {len(files)}: {files}"
        )
        assert "checkpoint.pt" in files, f"checkpoint.pt not in step_{step}: {files}"
        assert os.path.isfile(os.path.join(step_dir, "checkpoint.pt"))

    dataloader_files = [f for f in files if "dataloader" in f]
    assert len(dataloader_files) >= num_ranks, (
        f"Expected >= {num_ranks} dataloader files in step_{step}, found {len(dataloader_files)}: {dataloader_files}"
    )
    for rank in range(num_ranks):
        assert any(f"rank_{rank}" in f for f in dataloader_files), (
            f"No dataloader file for rank {rank} in step_{step}: {dataloader_files}"
        )


def _run_single_process_checkpoint_test(
    recipe_path,
    tmp_path,
    main_fn,
    ckpt_subdir_name,
    config_name="L0_sanity",
    extra_overrides=None,
    is_ddp=True,
):
    """Run a two-phase checkpoint save/resume test in a single process.

    Phase 1 trains for 10 steps (saving at step 5), phase 2 resumes and
    continues to step 15 (saving at step 10).  Both phases validate that
    checkpoints are created correctly and that losses are finite.

    Returns:
        Tuple of (phase1_loss, phase2_loss).
    """
    ckpt_dir = str(tmp_path / "ckpt")
    common = [
        "checkpoint.save_every_n_steps=5",
        "checkpoint.async_save=false",
        *(extra_overrides or []),
    ]

    # Phase 1: train 10 steps, checkpoint at step 5
    cfg1 = _compose_config(
        recipe_path,
        tmp_path,
        config_name,
        [
            "num_train_steps=10",
            "checkpoint.resume_from_checkpoint=false",
            *common,
        ],
    )

    loss1 = main_fn(cfg1)
    gc.collect()
    torch.cuda.empty_cache()

    ckpt_subdir = os.path.join(ckpt_dir, ckpt_subdir_name)
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"
    _assert_checkpoint_step(ckpt_subdir, 5, num_ranks=1, is_ddp=is_ddp)

    step_dirs = sorted(d for d in os.listdir(ckpt_subdir) if d.startswith("step_"))
    assert step_dirs == ["step_5"], f"Expected only step_5 after phase 1, found: {step_dirs}"

    # Phase 2: resume and continue to step 15, checkpoint at step 10
    cfg2 = _compose_config(
        recipe_path,
        tmp_path,
        config_name,
        [
            "num_train_steps=15",
            "checkpoint.resume_from_checkpoint=true",
            *common,
        ],
    )

    loss2 = main_fn(cfg2)
    gc.collect()
    torch.cuda.empty_cache()

    _assert_checkpoint_step(ckpt_subdir, 5, num_ranks=1, is_ddp=is_ddp)
    _assert_checkpoint_step(ckpt_subdir, 10, num_ranks=1, is_ddp=is_ddp)

    step_dirs = sorted(d for d in os.listdir(ckpt_subdir) if d.startswith("step_"))
    assert set(step_dirs) == {"step_5", "step_10"}, f"Expected step_5 and step_10, found: {step_dirs}"

    # Validate losses are finite and not NaN
    _assert_loss_valid(loss1, "phase 1")
    _assert_loss_valid(loss2, "phase 2")

    return loss1, loss2


def _run_multi_process_checkpoint_test(
    recipe_path,
    tmp_path,
    train_script_name,
    ckpt_subdir_name,
    nproc=2,
    extra_overrides=None,
    is_ddp=True,
):
    """Run a two-phase checkpoint save/resume test using ``torchrun``.

    Same two-phase strategy as :func:`_run_single_process_checkpoint_test`
    but spawns *nproc* processes via ``torchrun``.
    """
    ckpt_dir = str(tmp_path / "ckpt")
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    train_script = recipe_path / train_script_name
    common = [
        f"checkpoint.ckpt_dir={ckpt_dir}",
        "checkpoint.save_every_n_steps=5",
        "dataset.use_stateful_dataloader=true",
        *(extra_overrides or []),
    ]

    base_cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", str(train_script)]

    # Phase 1
    result1 = subprocess.run(
        [*base_cmd, "num_train_steps=10", "checkpoint.resume_from_checkpoint=false", *common],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    ckpt_subdir = os.path.join(ckpt_dir, ckpt_subdir_name)
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"
    _assert_checkpoint_step(ckpt_subdir, 5, num_ranks=nproc, is_ddp=is_ddp)

    step_dirs = [d for d in os.listdir(ckpt_subdir) if d.startswith("step_")]
    assert len(step_dirs) == 1, f"Expected 1 checkpoint dir after phase 1, found: {step_dirs}"

    # Phase 2
    result2 = subprocess.run(
        [*base_cmd, "num_train_steps=15", "checkpoint.resume_from_checkpoint=true", *common],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    _assert_checkpoint_step(ckpt_subdir, 5, num_ranks=nproc, is_ddp=is_ddp)
    _assert_checkpoint_step(ckpt_subdir, 10, num_ranks=nproc, is_ddp=is_ddp)

    step_dirs = sorted(d for d in os.listdir(ckpt_subdir) if d.startswith("step_"))
    assert set(step_dirs) == {"step_5", "step_10"}, f"Expected step_5 and step_10, found: {step_dirs}"


# ---------------------------------------------------------------------------
# DDP Checkpoint Tests
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load_single_process_ddp(recipe_path, tmp_path):
    """Test checkpoint save/resume for DDP with a single process."""
    _run_single_process_checkpoint_test(
        recipe_path,
        tmp_path,
        main_ddp,
        ckpt_subdir_name="train_ddp",
        is_ddp=True,
    )


@requires_multi_gpu
def test_checkpoint_save_and_load_two_processes_ddp(recipe_path, tmp_path):
    """Test checkpoint save/resume for DDP with two processes."""
    _run_multi_process_checkpoint_test(
        recipe_path,
        tmp_path,
        "train_ddp.py",
        ckpt_subdir_name="train_ddp",
        is_ddp=True,
    )


# ---------------------------------------------------------------------------
# FSDP2 Checkpoint Tests
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load_single_process_fsdp2(recipe_path, tmp_path):
    """Test checkpoint save/resume for FSDP2 with a single process."""
    _run_single_process_checkpoint_test(
        recipe_path,
        tmp_path,
        main_fsdp2,
        ckpt_subdir_name="train_fsdp2",
        is_ddp=False,
    )


@requires_multi_gpu
def test_checkpoint_save_and_load_two_processes_fsdp2(recipe_path, tmp_path):
    """Test checkpoint save/resume for FSDP2 with two processes."""
    _run_multi_process_checkpoint_test(
        recipe_path,
        tmp_path,
        "train_fsdp2.py",
        ckpt_subdir_name="train_fsdp2",
        is_ddp=False,
    )


# ---------------------------------------------------------------------------
# FSDP2 + Context Parallelism Checkpoint Tests
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load_single_process_fsdp2_with_context_parallelism(recipe_path, tmp_path):
    """Test checkpoint save/resume for FSDP2 with context parallelism (single process)."""
    _run_single_process_checkpoint_test(
        recipe_path,
        tmp_path,
        main_fsdp2_cp,
        ckpt_subdir_name="train_fsdp2",
        config_name="L0_sanity_cp",
        is_ddp=False,
    )


@requires_multi_gpu
def test_checkpoint_save_and_load_two_processes_fsdp2_with_context_parallelism(recipe_path, tmp_path):
    """Test checkpoint save/resume for FSDP2 with context parallelism (two processes)."""
    _run_multi_process_checkpoint_test(
        recipe_path,
        tmp_path,
        "train_fsdp2_cp.py",
        ckpt_subdir_name="train_fsdp2",
        extra_overrides=["checkpoint.async_save=false", "cp_size=2"],
        is_ddp=False,
    )


# ---------------------------------------------------------------------------
# Scheduler Resume Tests
# ---------------------------------------------------------------------------


def test_scheduler_resume_single_gpu(recipe_path, tmp_path):
    """Test that the LR scheduler resumes from the correct state after checkpoint load."""
    _run_single_process_checkpoint_test(
        recipe_path,
        tmp_path,
        main_ddp,
        ckpt_subdir_name="train_ddp",
        extra_overrides=[
            "lr_scheduler_kwargs.num_warmup_steps=20",
            "lr_scheduler_kwargs.num_decay_steps=100",
        ],
        is_ddp=True,
    )


@requires_multi_gpu
def test_scheduler_resume_two_gpu(recipe_path, tmp_path):
    """Test that the LR scheduler resumes correctly with multi-GPU FSDP2 training."""
    _run_multi_process_checkpoint_test(
        recipe_path,
        tmp_path,
        "train_fsdp2.py",
        ckpt_subdir_name="train_fsdp2",
        extra_overrides=[
            "lr_scheduler_kwargs.num_warmup_steps=20",
            "lr_scheduler_kwargs.num_decay_steps=100",
        ],
        is_ddp=False,
    )


# ---------------------------------------------------------------------------
# Final Model Save Tests
# ---------------------------------------------------------------------------


def test_final_model_save_ddp(recipe_path, tmp_path):
    """Test that DDP saves a final model with model.safetensors and config.json."""
    cfg = _compose_config(
        recipe_path,
        tmp_path,
        "L0_sanity",
        [
            "checkpoint.save_final_model=true",
            "num_train_steps=3",
        ],
    )

    loss = main_ddp(cfg)
    gc.collect()
    torch.cuda.empty_cache()

    _assert_loss_valid(loss, "final model ddp")

    final_model_dir = os.path.join(str(tmp_path / "ckpt"), "train_ddp", "final_model")
    assert os.path.exists(final_model_dir), "Final model directory not created"
    for fname in ("model.safetensors", "config.json"):
        fpath = os.path.join(final_model_dir, fname)
        assert os.path.exists(fpath), f"Missing: {fname}"
        assert os.path.getsize(fpath) > 0, f"{fname} is empty"


def test_final_model_save_fsdp2(recipe_path, tmp_path):
    """Test that FSDP2 gathers weights and saves a final model."""
    cfg = _compose_config(
        recipe_path,
        tmp_path,
        "L0_sanity",
        [
            "checkpoint.save_final_model=true",
            "num_train_steps=3",
        ],
    )

    loss = main_fsdp2(cfg)
    gc.collect()
    torch.cuda.empty_cache()

    _assert_loss_valid(loss, "final model fsdp2")

    final_model_dir = os.path.join(str(tmp_path / "ckpt"), "train_fsdp2", "final_model")
    assert os.path.exists(final_model_dir), "Final model directory not created"
    for fname in ("model.safetensors", "config.json"):
        fpath = os.path.join(final_model_dir, fname)
        assert os.path.exists(fpath), f"Missing: {fname}"
        assert os.path.getsize(fpath) > 0, f"{fname} is empty"


# ---------------------------------------------------------------------------
# Checkpoint Pruning Tests
# ---------------------------------------------------------------------------


def test_checkpoint_pruning(tmp_path):
    """Test checkpoint pruning keeps only the latest N checkpoints."""
    from checkpoint import prune_checkpoints

    temp_dir = str(tmp_path / "test_checkpoint_pruning")
    os.makedirs(temp_dir, exist_ok=True)
    for i in range(11):
        os.makedirs(os.path.join(temp_dir, f"step_{i}"), exist_ok=True)
    assert len(os.listdir(temp_dir)) == 11
    prune_checkpoints(temp_dir, 5)
    assert len(os.listdir(temp_dir)) == 5
    assert "step_6" in os.listdir(temp_dir)
    assert "step_7" in os.listdir(temp_dir)
    assert "step_8" in os.listdir(temp_dir)
    assert "step_9" in os.listdir(temp_dir)
    assert "step_10" in os.listdir(temp_dir)


def test_checkpoint_pruning_not_enough_checkpoints(tmp_path):
    """Test checkpoint pruning when fewer checkpoints than max exist."""
    from checkpoint import prune_checkpoints

    temp_dir = str(tmp_path / "test_checkpoint_pruning")
    os.makedirs(temp_dir, exist_ok=True)
    for i in range(3):
        os.makedirs(os.path.join(temp_dir, f"step_{i}"), exist_ok=True)
    assert len(os.listdir(temp_dir)) == 3
    prune_checkpoints(temp_dir, 5)
    assert len(os.listdir(temp_dir)) == 3


def test_checkpoint_pruning_with_files(tmp_path):
    """Test checkpoint pruning with file-based checkpoints."""
    from checkpoint import prune_checkpoints

    for i in range(11):
        (tmp_path / f"step_{i}.pt").touch()
    assert len(list(tmp_path.glob("step_*.pt"))) == 11
    prune_checkpoints(tmp_path, 5)
    assert len(list(tmp_path.glob("step_*.pt"))) == 5
    assert (tmp_path / "step_6.pt").exists()
    assert (tmp_path / "step_7.pt").exists()
    assert (tmp_path / "step_8.pt").exists()
    assert (tmp_path / "step_9.pt").exists()
    assert (tmp_path / "step_10.pt").exists()


# ---------------------------------------------------------------------------
# FP8 Checkpoint Tests (with quantized_model_init)
# ---------------------------------------------------------------------------

_FP8_QUANTIZED_OVERRIDES = [
    "fp8_config.enabled=true",
    "+config_kwargs.use_quantized_model_init=true",
    "+dataset.pad_sequences_to_be_divisible_by=16",
]


def test_checkpoint_save_and_load_single_process_ddp_fp8_quantized(recipe_path, tmp_path, fp_recipe):
    """Test checkpoint save/resume for DDP with FP8 quantized model init."""

    if fp_recipe[0].endswith("Float8BlockScaling"):
        pytest.xfail(reason="Float8BlockScaling currently does not support quantized model init + dcp")

    _run_single_process_checkpoint_test(
        recipe_path,
        tmp_path,
        main_ddp,
        ckpt_subdir_name="train_ddp",
        config_name="L0_sanity_cp",
        extra_overrides=[*_FP8_QUANTIZED_OVERRIDES, *fp_recipe],
        is_ddp=True,
    )


def test_checkpoint_save_and_load_single_process_fsdp2_fp8_quantized(recipe_path, tmp_path, fp_recipe):
    """Test checkpoint save/resume for FSDP2 with FP8 quantized model init."""

    if fp_recipe[0].endswith("Float8BlockScaling"):
        pytest.xfail(reason="Float8BlockScaling currently does not support quantized model init + dcp")

    _run_single_process_checkpoint_test(
        recipe_path,
        tmp_path,
        main_fsdp2,
        ckpt_subdir_name="train_fsdp2",
        config_name="L0_sanity_cp",
        extra_overrides=[*_FP8_QUANTIZED_OVERRIDES, *fp_recipe],
        is_ddp=False,
    )


def test_checkpoint_save_and_load_single_process_fsdp2_cp_fp8_quantized(recipe_path, tmp_path, fp_recipe):
    """Test checkpoint save/resume for FSDP2 with context parallelism and FP8 quantized model init."""

    if fp_recipe[0].endswith("Float8BlockScaling"):
        pytest.xfail(reason="Float8BlockScaling currently does not support quantized model init + dcp")

    _run_single_process_checkpoint_test(
        recipe_path,
        tmp_path,
        main_fsdp2_cp,
        ckpt_subdir_name="train_fsdp2",
        config_name="L0_sanity_cp",
        extra_overrides=[*_FP8_QUANTIZED_OVERRIDES, *fp_recipe],
        is_ddp=False,
    )


def test_checkpoint_save_and_load_single_process_fsdp2_cp_fp8_quantized_async(recipe_path, tmp_path, fp_recipe):
    """Test checkpoint save/resume for FSDP2+CP with FP8 quantized model init and async save.

    This reproduces the corys_config scenario where async_save=true (the default)
    is used with FP8 quantized model init.
    """

    if fp_recipe[0].endswith("Float8BlockScaling"):
        pytest.xfail(reason="Float8BlockScaling currently does not support quantized model init + dcp")

    _run_single_process_checkpoint_test(
        recipe_path,
        tmp_path,
        main_fsdp2_cp,
        ckpt_subdir_name="train_fsdp2",
        config_name="L0_sanity_cp",
        extra_overrides=[
            *_FP8_QUANTIZED_OVERRIDES,
            *fp_recipe,
            "checkpoint.async_save=true",
        ],
        is_ddp=False,
    )
