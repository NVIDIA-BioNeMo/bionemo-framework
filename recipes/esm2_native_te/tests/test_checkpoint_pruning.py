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

"""Tests for checkpoint helper functions: prune_checkpoints, get_latest_checkpoint, should_save_checkpoint."""

from pathlib import Path

from checkpoint import get_latest_checkpoint, prune_checkpoints, should_save_checkpoint


def _sort_by_step(paths: list[Path]) -> list[Path]:
    """Sort checkpoint paths by step number."""
    return sorted(paths, key=lambda x: int(Path(x).stem.split("_")[1]))


def test_checkpoint_pruning(tmp_path):
    """Create 11 checkpoint dirs, prune to 5, verify latest 5 remain."""
    for i in range(11):
        (tmp_path / f"step_{i * 100}").mkdir()

    prune_checkpoints(tmp_path, max_checkpoints=5)

    remaining = _sort_by_step(list(tmp_path.iterdir()))
    assert len(remaining) == 5
    expected_names = [f"step_{i * 100}" for i in range(6, 11)]
    assert [d.name for d in remaining] == expected_names


def test_checkpoint_pruning_not_enough_checkpoints(tmp_path):
    """Create 3 dirs, prune to 5, all should remain."""
    for i in range(3):
        (tmp_path / f"step_{i * 100}").mkdir()

    prune_checkpoints(tmp_path, max_checkpoints=5)

    remaining = list(tmp_path.iterdir())
    assert len(remaining) == 3


def test_checkpoint_pruning_with_files(tmp_path):
    """Create .pt files instead of dirs, verify pruning removes files."""
    for i in range(6):
        (tmp_path / f"step_{i * 100}.pt").touch()

    prune_checkpoints(tmp_path, max_checkpoints=3)

    remaining = _sort_by_step(list(tmp_path.iterdir()))
    assert len(remaining) == 3
    expected_names = [f"step_{i * 100}.pt" for i in range(3, 6)]
    assert [f.name for f in remaining] == expected_names


def test_get_latest_checkpoint(tmp_path):
    """Verify returns the checkpoint with the highest step number."""
    for step in [100, 500, 200, 1000, 300]:
        (tmp_path / f"step_{step}").mkdir()

    latest, step = get_latest_checkpoint(tmp_path)
    assert step == 1000
    assert latest.name == "step_1000"


def test_get_latest_checkpoint_empty(tmp_path):
    """Verify returns (None, 0) when no checkpoints exist."""
    latest, step = get_latest_checkpoint(tmp_path)
    assert latest is None
    assert step == 0


def test_get_latest_checkpoint_nonexistent(tmp_path):
    """Verify returns (None, 0) when directory does not exist."""
    latest, step = get_latest_checkpoint(tmp_path / "nonexistent")
    assert latest is None
    assert step == 0


def test_should_save_checkpoint():
    """Verify step 0 never saves, divisible steps save."""
    # Step 0 should never save
    assert should_save_checkpoint(step=0, save_every_n_steps=10) is False

    # Exact divisible steps should save
    assert should_save_checkpoint(step=10, save_every_n_steps=10) is True
    assert should_save_checkpoint(step=20, save_every_n_steps=10) is True

    # Non-divisible steps should not save
    assert should_save_checkpoint(step=5, save_every_n_steps=10) is False
    assert should_save_checkpoint(step=11, save_every_n_steps=10) is False

    # save_every_n_steps=0 should never save
    assert should_save_checkpoint(step=10, save_every_n_steps=0) is False

    # save_every_n_steps < 0 should never save
    assert should_save_checkpoint(step=10, save_every_n_steps=-1) is False
