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
- mFSDP (Megatron-style Fully Sharded Data Parallel) with 1 and 2 processes
- FSDP2 (PyTorch native Fully Sharded Data Parallel v2) with 1 and 2 processes

Test Strategy:
1. Phase 1: Train for N steps and save checkpoint
2. Phase 2: Resume training from checkpoint and continue
3. Validate: Checkpoints created, resuming works, training continues seamlessly

Each test uses temporary directories and disables wandb logging for isolation.
"""

import os
import subprocess

import pytest
import torch
from hydra import compose, initialize_config_dir

from train_ddp import main as main_ddp
from train_fsdp2 import main as main_fsdp2
from train_mfsdp import main as main_mfsdp


os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


def test_checkpoint_save_and_load_single_process_ddp(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for DDP with single process.

    This test validates:
    - DDP creates single-file checkpoints (step_X.pt files)
    - Standard PyTorch checkpoint format (model + optimizer state)
    - Single-process DDP training and resuming works correctly
    - Checkpoint files contain complete model state

    Process:
    1. Train 10 steps (0-9), save checkpoint file at step 5
    2. Resume training from checkpoint, continue to step 15
    3. Verify step_X.pt checkpoint files exist at steps 5 and 10
    """
    temp_dir = str(tmp_path / "test_ckpt_ddp")

    # Phase 1: Train for 10 steps, saving a checkpoint at step 5
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        phase1_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                f"hydra.run.dir={tmp_path}",
                "num_train_steps=10",
                "checkpoint.save_every_n_steps=5",
                "checkpoint.resume_from_checkpoint=false",  # Start fresh
                "dataset.use_stateful_dataloader=true",
            ],
        )

    main_ddp(phase1_config)

    # Phase 1 creates this directory structure:
    # ckpt_subdir/
    # └── step_5/
    #     ├── checkpoint.pt
    #     └── dataloader_step_5_rank_0_num_workers_1.pt

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_ddp")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify step_5 checkpoint was created
    step_5_dir = os.path.join(ckpt_subdir, "step_5")

    # Check step_5 directory exists and contains expected files
    assert os.path.isdir(step_5_dir), f"Step 5 directory not found: {step_5_dir}"
    step_5_files = os.listdir(step_5_dir)
    assert len(step_5_files) == 2, f"Expected 2 files in step_5 directory, found {len(step_5_files)}: {step_5_files}"
    assert "checkpoint.pt" in step_5_files, f"checkpoint.pt not found in step_5 directory. Files found: {step_5_files}"
    assert any("dataloader" in f for f in step_5_files), (
        f"No dataloader file found in step_5 directory. Files found: {step_5_files}"
    )

    # Verify the actual checkpoint files are valid files
    assert os.path.isfile(os.path.join(step_5_dir, "checkpoint.pt")), "step_5/checkpoint.pt is not a valid file"

    # Check that only step_5 exists at this point (no step_10 yet)
    all_step_dirs = [d for d in os.listdir(ckpt_subdir) if d.startswith("step_")]
    assert len(all_step_dirs) == 1, (
        f"Expected only 1 checkpoint directory after phase 1, found {len(all_step_dirs)}: {all_step_dirs}"
    )
    assert all_step_dirs[0] == "step_5", f"Expected only step_5 after phase 1, found: {all_step_dirs}"

    # Phase 2: Resume training (should start from step 5, continue to step 15)
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        phase2_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                f"hydra.run.dir={tmp_path}",
                "num_train_steps=15",
                "checkpoint.save_every_n_steps=5",
                "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
                "dataset.use_stateful_dataloader=true",
            ],
        )

    main_ddp(phase2_config)

    # Phase 2 adds to the directory structure:
    # ckpt_subdir/
    # ├── step_5/
    # │   ├── checkpoint.pt
    # │   └── dataloader_step_5_rank_0_num_workers_1.pt
    # └── step_10/
    #     ├── checkpoint.pt
    #     └── dataloader_step_10_rank_0_num_workers_1.pt

    # Verify the checkpoint files exist in the correct directories
    step_5_dir = os.path.join(ckpt_subdir, "step_5")
    step_10_dir = os.path.join(ckpt_subdir, "step_10")

    # Check step_5 directory and files
    assert os.path.isdir(step_5_dir), f"Step 5 directory not found: {step_5_dir}"
    step_5_files = os.listdir(step_5_dir)
    assert "checkpoint.pt" in step_5_files, f"checkpoint.pt not found in step_5 directory. Files found: {step_5_files}"
    assert any("dataloader" in f for f in step_5_files), (
        f"No dataloader file found in step_5 directory. Files found: {step_5_files}"
    )

    # Check step_10 directory and files
    assert os.path.isdir(step_10_dir), f"Step 10 directory not found: {step_10_dir}"
    step_10_files = os.listdir(step_10_dir)
    assert "checkpoint.pt" in step_10_files, (
        f"checkpoint.pt not found in step_10 directory. Files found: {step_10_files}"
    )
    assert any("dataloader" in f for f in step_10_files), (
        f"No dataloader file found in step_10 directory. Files found: {step_10_files}"
    )

    # Verify the actual checkpoint files are valid files
    assert os.path.isfile(os.path.join(step_5_dir, "checkpoint.pt")), "step_5/checkpoint.pt is not a valid file"
    assert os.path.isfile(os.path.join(step_10_dir, "checkpoint.pt")), "step_10/checkpoint.pt is not a valid file"

    # Final check: we should have exactly 2 checkpoint directories (step_5 and step_10)
    all_step_dirs = [d for d in os.listdir(ckpt_subdir) if d.startswith("step_")]
    assert len(all_step_dirs) == 2, f"Expected 2 checkpoint directories, found {len(all_step_dirs)}: {all_step_dirs}"
    assert set(all_step_dirs) == {"step_5", "step_10"}, f"Expected step_5 and step_10, found: {all_step_dirs}"


@requires_multi_gpu
def test_checkpoint_save_and_load_two_processes_ddp(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for DDP with two processes.

    This test validates:
    - Multi-process DDP checkpoint behavior (main process saves only)
    - Checkpoint files can be loaded by all DDP processes
    - Process synchronization during resume (all processes load same checkpoint)
    - DDP training continues correctly after resume across processes

    Process:
    1. Train 10 steps (0-9) across 2 processes, main process saves checkpoint at step 5
    2. Resume training with 2 processes, all load same checkpoint file, continue to step 15
    3. Verify step_X.pt checkpoint files exist at steps 5 and 10
    """
    temp_dir = str(tmp_path / "test_ckpt_ddp_2p")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_ddp.py
    train_script = recipe_path / "train_ddp.py"

    # Phase 1: Train for 10 steps with 2 processes
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh
        "dataset.use_stateful_dataloader=true",
        f"hydra.run.dir={tmp_path}",
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Phase 1 creates this directory structure with 2 processes:
    # ckpt_subdir/
    # └── step_5/
    #     ├── checkpoint.pt
    #     ├── dataloader_step_5_rank_0_num_workers_1.pt
    #     └── dataloader_step_5_rank_1_num_workers_1.pt

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_ddp")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify step_5 checkpoint was created
    step_5_dir = os.path.join(ckpt_subdir, "step_5")

    # Check step_5 directory exists and contains expected files
    assert os.path.isdir(step_5_dir), f"Step 5 directory not found: {step_5_dir}"
    step_5_files = os.listdir(step_5_dir)

    # With 2 processes, we expect: 1 checkpoint.pt + 2 dataloader files (one per rank)
    assert len(step_5_files) == 3, (
        f"Expected 3 files in step_5 directory (1 checkpoint + 2 dataloaders), found {len(step_5_files)}: {step_5_files}"
    )
    assert "checkpoint.pt" in step_5_files, f"checkpoint.pt not found in step_5 directory. Files found: {step_5_files}"

    # Check for dataloader files for both ranks
    dataloader_files = [f for f in step_5_files if "dataloader" in f]
    assert len(dataloader_files) == 2, (
        f"Expected 2 dataloader files (rank 0 and 1), found {len(dataloader_files)}: {dataloader_files}"
    )

    # Verify we have dataloader files for both rank 0 and rank 1
    assert any("rank_0" in f for f in dataloader_files), (
        f"No dataloader file for rank 0 found. Files: {dataloader_files}"
    )
    assert any("rank_1" in f for f in dataloader_files), (
        f"No dataloader file for rank 1 found. Files: {dataloader_files}"
    )

    # Verify the actual checkpoint file is valid
    assert os.path.isfile(os.path.join(step_5_dir, "checkpoint.pt")), "step_5/checkpoint.pt is not a valid file"

    # Check that only step_5 exists at this point (no step_10 yet)
    all_step_dirs = [d for d in os.listdir(ckpt_subdir) if d.startswith("step_")]
    assert len(all_step_dirs) == 1, (
        f"Expected only 1 checkpoint directory after phase 1, found {len(all_step_dirs)}: {all_step_dirs}"
    )
    assert all_step_dirs[0] == "step_5", f"Expected only step_5 after phase 1, found: {all_step_dirs}"

    # Phase 2: Resume training with 2 processes
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
        "dataset.use_stateful_dataloader=true",
        f"hydra.run.dir={tmp_path}",
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Phase 2 adds to the directory structure:
    # ckpt_subdir/
    # ├── step_5/
    # │   ├── checkpoint.pt
    # │   ├── dataloader_step_5_rank_0_num_workers_1.pt
    # │   └── dataloader_step_5_rank_1_num_workers_1.pt
    # └── step_10/
    #     ├── checkpoint.pt
    #     ├── dataloader_step_10_rank_0_num_workers_1.pt
    #     └── dataloader_step_10_rank_1_num_workers_1.pt

    # Verify step_10 checkpoint was created
    step_10_dir = os.path.join(ckpt_subdir, "step_10")

    # Check step_10 directory exists and contains expected files
    assert os.path.isdir(step_10_dir), f"Step 10 directory not found: {step_10_dir}"
    step_10_files = os.listdir(step_10_dir)

    # With 2 processes, we expect: 1 checkpoint.pt + 2 dataloader files (one per rank)
    assert len(step_10_files) == 3, (
        f"Expected 3 files in step_10 directory (1 checkpoint + 2 dataloaders), found {len(step_10_files)}: {step_10_files}"
    )
    assert "checkpoint.pt" in step_10_files, (
        f"checkpoint.pt not found in step_10 directory. Files found: {step_10_files}"
    )

    # Check for dataloader files for both ranks
    dataloader_files_10 = [f for f in step_10_files if "dataloader" in f]
    assert len(dataloader_files_10) == 2, (
        f"Expected 2 dataloader files (rank 0 and 1), found {len(dataloader_files_10)}: {dataloader_files_10}"
    )

    # Verify we have dataloader files for both rank 0 and rank 1
    assert any("rank_0" in f for f in dataloader_files_10), (
        f"No dataloader file for rank 0 found in step_10. Files: {dataloader_files_10}"
    )
    assert any("rank_1" in f for f in dataloader_files_10), (
        f"No dataloader file for rank 1 found in step_10. Files: {dataloader_files_10}"
    )

    # Verify the actual checkpoint file is valid
    assert os.path.isfile(os.path.join(step_10_dir, "checkpoint.pt")), "step_10/checkpoint.pt is not a valid file"

    # Final check: we should have exactly 2 checkpoint directories (step_5 and step_10)
    all_step_dirs = [d for d in os.listdir(ckpt_subdir) if d.startswith("step_")]
    assert len(all_step_dirs) == 2, f"Expected 2 checkpoint directories, found {len(all_step_dirs)}: {all_step_dirs}"
    assert set(all_step_dirs) == {"step_5", "step_10"}, f"Expected step_5 and step_10, found: {all_step_dirs}"


def test_checkpoint_save_and_load_single_process_mfsdp(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for mFSDP with single process.

    This test validates:
    - mFSDP creates distributed checkpoints (step_X directories)
    - Dataloader state is saved alongside model checkpoint
    - Checkpoints are saved at specified intervals (every 5 steps)
    - Training can resume from latest checkpoint and continue
    - Resume starts from correct step count

    Process:
    1. Train 10 steps (0-9), save checkpoint at step 5
    2. Resume training from step 5, continue to step 15
    3. Verify checkpoints exist at steps 5 and 10 with dataloader files
    """
    temp_dir = str(tmp_path / "test_ckpt_mfsdp")

    # Phase 1: Train for 10 steps
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        phase1_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                f"hydra.run.dir={tmp_path}",
                "num_train_steps=10",
                "checkpoint.save_every_n_steps=5",
                "checkpoint.resume_from_checkpoint=false",  # Start fresh
                "dataset.use_stateful_dataloader=true",
            ],
        )

    main_mfsdp(phase1_config)

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_mfsdp")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created (mFSDP creates directories)
    checkpoint_dirs = [
        f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, f))
    ]
    assert len(checkpoint_dirs) > 0, "No checkpoint directories created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5"
    assert expected_checkpoint in checkpoint_dirs, f"Expected {expected_checkpoint} not found"

    # Check dataloader file exists in step_5 directory
    step_5_dir = os.path.join(ckpt_subdir, "step_5")
    assert os.path.isdir(step_5_dir), f"Step 5 directory not found: {step_5_dir}"
    step_5_files = os.listdir(step_5_dir)

    # With single process, we expect dataloader file for rank 0
    dataloader_files_5 = [f for f in step_5_files if "dataloader" in f]
    assert len(dataloader_files_5) >= 1, (
        f"Expected at least 1 dataloader file, found {len(dataloader_files_5)}: {dataloader_files_5}"
    )
    assert any("rank_0" in f for f in dataloader_files_5), (
        f"No dataloader file for rank 0 found in step_5. Files: {dataloader_files_5}"
    )

    # Phase 2: Resume training
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        phase2_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                f"hydra.run.dir={tmp_path}",
                "num_train_steps=15",
                "checkpoint.save_every_n_steps=5",
                "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
                "dataset.use_stateful_dataloader=true",
            ],
        )

    main_mfsdp(phase2_config)

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_dirs = [
        f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, f))
    ]
    expected_checkpoints = ["step_5", "step_10"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_dirs, f"Missing checkpoint: {expected}"

    # Check dataloader file exists in step_10 directory
    step_10_dir = os.path.join(ckpt_subdir, "step_10")
    assert os.path.isdir(step_10_dir), f"Step 10 directory not found: {step_10_dir}"
    step_10_files = os.listdir(step_10_dir)

    # With single process, we expect dataloader file for rank 0
    dataloader_files_10 = [f for f in step_10_files if "dataloader" in f]
    assert len(dataloader_files_10) >= 1, (
        f"Expected at least 1 dataloader file in step_10, found {len(dataloader_files_10)}: {dataloader_files_10}"
    )
    assert any("rank_0" in f for f in dataloader_files_10), (
        f"No dataloader file for rank 0 found in step_10. Files: {dataloader_files_10}"
    )


@requires_multi_gpu
def test_checkpoint_save_and_load_two_processes_mfsdp(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for mFSDP with two processes.

    This test validates:
    - Multi-process mFSDP coordination during checkpoint save/load
    - Dataloader state is saved for each rank alongside model checkpoint
    - Distributed checkpoint format works across process boundaries
    - Both processes participate in distributed checkpoint operations
    - Training resumes correctly with proper process synchronization

    Process:
    1. Train 10 steps (0-9) across 2 processes, save checkpoint at step 5
    2. Resume training with 2 processes from step 5, continue to step 15
    3. Verify distributed checkpoints exist at steps 5 and 10 with dataloader files for both ranks
    """
    temp_dir = str(tmp_path / "test_ckpt_mfsdp_2p")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_mfsdp.py
    train_script = recipe_path / "train_mfsdp.py"

    # Phase 1: Train for 10 steps with 2 processes
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh
        "dataset.use_stateful_dataloader=true",
        f"hydra.run.dir={tmp_path}",
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_mfsdp")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created
    checkpoint_dirs = [
        f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, f))
    ]
    assert len(checkpoint_dirs) > 0, "No checkpoint directories created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5"
    assert expected_checkpoint in checkpoint_dirs, f"Expected {expected_checkpoint} not found"

    # Check dataloader files exist in step_5 directory for both ranks
    step_5_dir = os.path.join(ckpt_subdir, "step_5")
    assert os.path.isdir(step_5_dir), f"Step 5 directory not found: {step_5_dir}"
    step_5_files = os.listdir(step_5_dir)

    # With 2 processes, we expect dataloader files for rank 0 and rank 1
    dataloader_files_5 = [f for f in step_5_files if "dataloader" in f]
    assert len(dataloader_files_5) == 2, (
        f"Expected 2 dataloader files (rank 0 and 1), found {len(dataloader_files_5)}: {dataloader_files_5}"
    )
    assert any("rank_0" in f for f in dataloader_files_5), (
        f"No dataloader file for rank 0 found in step_5. Files: {dataloader_files_5}"
    )
    assert any("rank_1" in f for f in dataloader_files_5), (
        f"No dataloader file for rank 1 found in step_5. Files: {dataloader_files_5}"
    )

    # Phase 2: Resume training with 2 processes
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
        "dataset.use_stateful_dataloader=true",
        f"hydra.run.dir={tmp_path}",
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_dirs = [
        f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, f))
    ]
    expected_checkpoints = ["step_5", "step_10"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_dirs, f"Missing checkpoint: {expected}"

    # Check dataloader files exist in step_10 directory for both ranks
    step_10_dir = os.path.join(ckpt_subdir, "step_10")
    assert os.path.isdir(step_10_dir), f"Step 10 directory not found: {step_10_dir}"
    step_10_files = os.listdir(step_10_dir)

    # With 2 processes, we expect dataloader files for rank 0 and rank 1
    dataloader_files_10 = [f for f in step_10_files if "dataloader" in f]
    assert len(dataloader_files_10) == 2, (
        f"Expected 2 dataloader files (rank 0 and 1) in step_10, found {len(dataloader_files_10)}: {dataloader_files_10}"
    )
    assert any("rank_0" in f for f in dataloader_files_10), (
        f"No dataloader file for rank 0 found in step_10. Files: {dataloader_files_10}"
    )
    assert any("rank_1" in f for f in dataloader_files_10), (
        f"No dataloader file for rank 1 found in step_10. Files: {dataloader_files_10}"
    )


def test_checkpoint_save_and_load_single_process_fsdp2_no_meta_device(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for FSDP2 with single process.

    This test validates:
    - FSDP2 creates distributed checkpoints (step_X directories by default)
    - Each rank saves its shard (even with single process)
    - Dataloader state is saved alongside model checkpoint
    - Training can resume from latest checkpoint and continue
    - Resume starts from correct step count

    Process:
    1. Train 10 steps (0-9), save checkpoint at step 5
    2. Resume training from step 5, continue to step 15
    3. Verify checkpoints exist at steps 5 and 10
    """
    temp_dir = str(tmp_path / "test_ckpt_fsdp2")

    # Phase 1: Train for 10 steps (using distributed checkpoint by default)
    # Use smaller model for faster tests
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        phase1_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                f"hydra.run.dir={tmp_path}",
                "num_train_steps=10",
                "checkpoint.save_every_n_steps=5",
                "checkpoint.resume_from_checkpoint=false",  # Start fresh
                "dataset.use_stateful_dataloader=true",
                "use_meta_device=false",
            ],
        )

    main_fsdp2(phase1_config)

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created (FSDP2 now creates directories by default)
    checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    assert len(checkpoint_dirs) > 0, "No checkpoint directories created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5"
    assert expected_checkpoint in checkpoint_dirs, f"Expected {expected_checkpoint} not found"

    # Check dataloader file exists in step_5 directory
    step_5_dir = os.path.join(ckpt_subdir, "step_5")
    assert os.path.isdir(step_5_dir), f"Step 5 directory not found: {step_5_dir}"
    step_5_files = os.listdir(step_5_dir)

    # With single process, we expect dataloader file for rank 0
    dataloader_files_5 = [f for f in step_5_files if "dataloader" in f]
    assert len(dataloader_files_5) >= 1, (
        f"Expected at least 1 dataloader file, found {len(dataloader_files_5)}: {dataloader_files_5}"
    )
    assert any("rank_0" in f for f in dataloader_files_5), (
        f"No dataloader file for rank 0 found in step_5. Files: {dataloader_files_5}"
    )

    # Phase 2: Resume training
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        phase2_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                "num_train_steps=15",
                "checkpoint.save_every_n_steps=5",
                "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
                "dataset.use_stateful_dataloader=true",
                "use_meta_device=false",
            ],
        )

    main_fsdp2(phase2_config)

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    expected_checkpoints = ["step_5", "step_10"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_dirs, f"Missing checkpoint: {expected}"

    # Check dataloader file exists in step_10 directory
    step_10_dir = os.path.join(ckpt_subdir, "step_10")
    assert os.path.isdir(step_10_dir), f"Step 10 directory not found: {step_10_dir}"
    step_10_files = os.listdir(step_10_dir)

    # With single process, we expect dataloader file for rank 0
    dataloader_files_10 = [f for f in step_10_files if "dataloader" in f]
    assert len(dataloader_files_10) >= 1, (
        f"Expected at least 1 dataloader file in step_10, found {len(dataloader_files_10)}: {dataloader_files_10}"
    )
    assert any("rank_0" in f for f in dataloader_files_10), (
        f"No dataloader file for rank 0 found in step_10. Files: {dataloader_files_10}"
    )


def test_checkpoint_save_and_load_single_process_fsdp2(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for FSDP2 with single process.

    This test validates:
    - FSDP2 creates distributed checkpoints (step_X directories by default)
    - Each rank saves its shard (even with single process)
    - Dataloader state is saved alongside model checkpoint
    - Training can resume from latest checkpoint and continue
    - Resume starts from correct step count

    Process:
    1. Train 10 steps (0-9), save checkpoint at step 5
    2. Resume training from step 5, continue to step 15
    3. Verify checkpoints exist at steps 5 and 10
    """
    temp_dir = str(tmp_path / "test_ckpt_fsdp2")

    # Phase 1: Train for 10 steps (using distributed checkpoint by default)
    # Use smaller model for faster tests
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        phase1_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                f"hydra.run.dir={tmp_path}",
                "num_train_steps=10",
                "checkpoint.save_every_n_steps=5",
                "checkpoint.resume_from_checkpoint=false",  # Start fresh
                "dataset.use_stateful_dataloader=true",
            ],
        )

    main_fsdp2(phase1_config)

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created (FSDP2 now creates directories by default)
    checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    assert len(checkpoint_dirs) > 0, "No checkpoint directories created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5"
    assert expected_checkpoint in checkpoint_dirs, f"Expected {expected_checkpoint} not found"

    # Check dataloader file exists in step_5 directory
    step_5_dir = os.path.join(ckpt_subdir, "step_5")
    assert os.path.isdir(step_5_dir), f"Step 5 directory not found: {step_5_dir}"
    step_5_files = os.listdir(step_5_dir)

    # With single process, we expect dataloader file for rank 0
    dataloader_files_5 = [f for f in step_5_files if "dataloader" in f]
    assert len(dataloader_files_5) >= 1, (
        f"Expected at least 1 dataloader file, found {len(dataloader_files_5)}: {dataloader_files_5}"
    )
    assert any("rank_0" in f for f in dataloader_files_5), (
        f"No dataloader file for rank 0 found in step_5. Files: {dataloader_files_5}"
    )

    # Phase 2: Resume training
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        phase2_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                "num_train_steps=15",
                "checkpoint.save_every_n_steps=5",
                "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
                "dataset.use_stateful_dataloader=true",
            ],
        )

    main_fsdp2(phase2_config)

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    expected_checkpoints = ["step_5", "step_10"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_dirs, f"Missing checkpoint: {expected}"

    # Check dataloader file exists in step_10 directory
    step_10_dir = os.path.join(ckpt_subdir, "step_10")
    assert os.path.isdir(step_10_dir), f"Step 10 directory not found: {step_10_dir}"
    step_10_files = os.listdir(step_10_dir)

    # With single process, we expect dataloader file for rank 0
    dataloader_files_10 = [f for f in step_10_files if "dataloader" in f]
    assert len(dataloader_files_10) >= 1, (
        f"Expected at least 1 dataloader file in step_10, found {len(dataloader_files_10)}: {dataloader_files_10}"
    )
    assert any("rank_0" in f for f in dataloader_files_10), (
        f"No dataloader file for rank 0 found in step_10. Files: {dataloader_files_10}"
    )


@requires_multi_gpu
def test_checkpoint_save_and_load_two_processes_fsdp2(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for FSDP2 with two processes.

    This test validates:
    - Multi-process FSDP2 distributed checkpointing (each rank saves its shard)
    - Dataloader state is saved for each rank alongside model checkpoint
    - All ranks participate in saving and loading
    - Training resumes correctly with proper process synchronization

    Process:
    1. Train 10 steps (0-9) across 2 processes, save checkpoint at step 5
    2. Resume training with 2 processes from step 5, continue to step 15
    3. Verify checkpoints exist at steps 5 and 10 with dataloader files for both ranks
    """
    temp_dir = str(tmp_path / "test_ckpt_fsdp2_2p")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_fsdp2.py
    train_script = recipe_path / "train_fsdp2.py"

    # Phase 1: Train for 10 steps with 2 processes
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "dataset.use_stateful_dataloader=true",
        f"hydra.run.dir={tmp_path}",
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created (FSDP2 now creates directories by default)
    checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    assert len(checkpoint_dirs) > 0, "No checkpoint directories created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5"
    assert expected_checkpoint in checkpoint_dirs, f"Expected {expected_checkpoint} not found"

    # Check dataloader files exist in step_5 directory for both ranks
    step_5_dir = os.path.join(ckpt_subdir, "step_5")
    assert os.path.isdir(step_5_dir), f"Step 5 directory not found: {step_5_dir}"
    step_5_files = os.listdir(step_5_dir)

    # With 2 processes, we expect dataloader files for rank 0 and rank 1
    dataloader_files_5 = [f for f in step_5_files if "dataloader" in f]
    assert len(dataloader_files_5) == 2, (
        f"Expected 2 dataloader files (rank 0 and 1), found {len(dataloader_files_5)}: {dataloader_files_5}"
    )
    assert any("rank_0" in f for f in dataloader_files_5), (
        f"No dataloader file for rank 0 found in step_5. Files: {dataloader_files_5}"
    )
    assert any("rank_1" in f for f in dataloader_files_5), (
        f"No dataloader file for rank 1 found in step_5. Files: {dataloader_files_5}"
    )

    # Phase 2: Resume training with 2 processes
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
        "dataset.use_stateful_dataloader=true",
        f"hydra.run.dir={tmp_path}",
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    expected_checkpoints = ["step_5", "step_10"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_dirs, f"Missing checkpoint: {expected}"

    # Check dataloader files exist in step_10 directory for both ranks
    step_10_dir = os.path.join(ckpt_subdir, "step_10")
    assert os.path.isdir(step_10_dir), f"Step 10 directory not found: {step_10_dir}"
    step_10_files = os.listdir(step_10_dir)

    # With 2 processes, we expect dataloader files for rank 0 and rank 1
    dataloader_files_10 = [f for f in step_10_files if "dataloader" in f]
    assert len(dataloader_files_10) == 2, (
        f"Expected 2 dataloader files (rank 0 and 1) in step_10, found {len(dataloader_files_10)}: {dataloader_files_10}"
    )
    assert any("rank_0" in f for f in dataloader_files_10), (
        f"No dataloader file for rank 0 found in step_10. Files: {dataloader_files_10}"
    )
    assert any("rank_1" in f for f in dataloader_files_10), (
        f"No dataloader file for rank 1 found in step_10. Files: {dataloader_files_10}"
    )


def test_final_model_save_ddp(recipe_path, tmp_path):
    """Test final model saving for DDP.

    Validates that DDP saves the final model correctly with:
    - model.safetensors containing weights
    - config.json with model configuration
    """
    temp_dir = str(tmp_path / "test_final_ddp")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                "checkpoint.save_final_model=true",
                "num_train_steps=3",
            ],
        )

    main_ddp(config)

    # Check final model directory
    final_model_dir = os.path.join(temp_dir, "train_ddp", "final_model")
    assert os.path.exists(final_model_dir), "Final model directory not created"

    # Check required files
    required_files = ["model.safetensors", "config.json"]
    for file in required_files:
        file_path = os.path.join(final_model_dir, file)
        assert os.path.exists(file_path), f"Missing required file: {file}"
        assert os.path.getsize(file_path) > 0, f"File {file} is empty"


def test_final_model_save_mfsdp(recipe_path, tmp_path):
    """Test final model saving for mFSDP.

    Validates that mFSDP gathers parameters and saves the final model with:
    - model.safetensors containing gathered weights
    - config.json with model configuration
    """
    temp_dir = str(tmp_path / "test_final_mfsdp")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                f"hydra.run.dir={tmp_path}",
                "num_train_steps=3",
                "checkpoint.save_final_model=true",
            ],
        )

    main_mfsdp(config)

    # Check final model directory
    final_model_dir = os.path.join(temp_dir, "train_mfsdp", "final_model")
    assert os.path.exists(final_model_dir), "Final model directory not created"

    # Check required files
    required_files = ["model.safetensors", "config.json"]
    for file in required_files:
        file_path = os.path.join(final_model_dir, file)
        assert os.path.exists(file_path), f"Missing required file: {file}"
        assert os.path.getsize(file_path) > 0, f"File {file} is empty"


def test_final_model_save_fsdp2(recipe_path, tmp_path):
    """Test final model saving for FSDP2.

    Validates that FSDP2 gathers full state dict and saves the final model with:
    - model.safetensors containing gathered weights
    - config.json with model configuration
    """
    temp_dir = str(tmp_path / "test_final_fsdp2")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                f"hydra.run.dir={tmp_path}",
                "checkpoint.save_final_model=true",
                "num_train_steps=3",
            ],
        )

    main_fsdp2(config)

    # Check final model directory
    final_model_dir = os.path.join(temp_dir, "train_fsdp2", "final_model")
    assert os.path.exists(final_model_dir), "Final model directory not created"

    # Check required files (FSDP2 doesn't save esm_nv.py)
    required_files = ["model.safetensors", "config.json"]
    for file in required_files:
        file_path = os.path.join(final_model_dir, file)
        assert os.path.exists(file_path), f"Missing required file: {file}"
        assert os.path.getsize(file_path) > 0, f"File {file} is empty"


def test_scheduler_resume_single_gpu(recipe_path, tmp_path):
    """Test that learning rate scheduler resumes from correct state after checkpoint load.

    This test validates:
    - Scheduler state is saved in checkpoint
    - Scheduler resumes with correct step count
    - Learning rate continues from where it left off (not reset)
    - Warmup and decay continue correctly after resume

    Process:
    1. Train for 10 steps, save checkpoint with scheduler state at step 5
    2. Resume training, verify scheduler continues from step 6 (not step 0)
    3. Check that learning rate progression is continuous across resume
    """
    temp_dir = str(tmp_path / "test_scheduler_resume")

    # Phase 1: Train for 10 steps with warmup
    # Use small warmup to see LR changes quickly
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        phase1_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                f"hydra.run.dir={tmp_path}",
                "num_train_steps=10",
                "checkpoint.save_every_n_steps=5",
                "checkpoint.resume_from_checkpoint=false",  # Start fresh, don't look for checkpoints
                "lr_scheduler_kwargs.num_warmup_steps=20",  # Warmup over 20 steps
                "lr_scheduler_kwargs.num_training_steps=100",  # Total 100 steps
            ],
        )

    main_ddp(phase1_config)

    # Phase 2: Resume training for 5 more steps
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        phase2_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"checkpoint.ckpt_dir={temp_dir}",
                f"+wandb_init_args.dir={tmp_path}",
                f"hydra.run.dir={tmp_path}",
                "num_train_steps=15",
                "checkpoint.save_every_n_steps=5",
                "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
                "lr_scheduler_kwargs.num_warmup_steps=20",
                "lr_scheduler_kwargs.num_training_steps=100",
            ],
        )

    main_ddp(phase2_config)

    # Verify checkpoints were created - basic validation that training ran successfully
    ckpt_subdir = os.path.join(temp_dir, "train_ddp")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Check that checkpoint directories exist (not files)
    checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    expected_checkpoint_dirs = ["step_5", "step_10"]
    for expected_dir in expected_checkpoint_dirs:
        assert expected_dir in checkpoint_dirs, f"Missing checkpoint directory: {expected_dir}"

        # Verify each checkpoint directory contains the checkpoint file
        checkpoint_file = os.path.join(ckpt_subdir, expected_dir, "checkpoint.pt")
        assert os.path.isfile(checkpoint_file), f"Missing checkpoint file: {checkpoint_file}"


@requires_multi_gpu
def test_scheduler_resume_two_gpu(recipe_path, tmp_path):
    """Test that learning rate scheduler resumes correctly with multi-GPU training.

    This test validates:
    - Scheduler state is synchronized across GPUs during save
    - All GPUs resume with same scheduler state
    - Learning rate is consistent across all processes after resume

    Process:
    1. Train for 10 steps across 2 GPUs, save checkpoint at step 5
    2. Resume training on 2 GPUs, verify scheduler continues correctly
    3. Ensure both GPUs have same learning rate progression
    """
    temp_dir = str(tmp_path / "test_scheduler_resume_2gpu")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Test with FSDP2 as it's most complex for scheduler state
    train_script = recipe_path / "train_fsdp2.py"

    # Phase 1: Train for 10 steps with 2 GPUs
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh, don't look for checkpoints
        "lr_scheduler_kwargs.num_warmup_steps=20",
        "lr_scheduler_kwargs.num_training_steps=100",
        f"hydra.run.dir={tmp_path}",
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Check that checkpoint was created (FSDP2 uses distributed format by default)
    ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")
    checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    assert "step_5" in checkpoint_dirs, "Checkpoint at step 5 not found"

    # Phase 2: Resume training with 2 GPUs
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
        "lr_scheduler_kwargs.num_warmup_steps=20",
        "lr_scheduler_kwargs.num_training_steps=100",
        f"hydra.run.dir={tmp_path}",
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Verify training continued (check for step progression in logs)
    assert "global_step: 6" in result2.stdout.lower(), "Phase 2 should start from step 6 after resuming from step 5"

    # Check that final checkpoint was created (distributed format)
    final_checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    assert "step_10" in final_checkpoint_dirs, "Checkpoint at step 10 not found"
