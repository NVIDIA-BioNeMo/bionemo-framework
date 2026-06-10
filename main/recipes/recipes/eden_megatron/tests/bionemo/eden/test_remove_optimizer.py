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

"""Tests for optimizer removal from MBridge checkpoints."""

import subprocess
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

from bionemo.recipeutils.checkpoint.remove_optimizer import remove_optimizer


def _get_dcp_keys(iter_dir: Path) -> set[str]:
    """Return all DCP state-dict keys in an iter directory."""
    reader = FileSystemReader(str(iter_dir))
    metadata = reader.read_metadata()
    return set(metadata.state_dict_metadata.keys())


def _get_tensor_keys(iter_dir: Path) -> set[str]:
    """Return only tensor (non-bytes) DCP keys."""
    reader = FileSystemReader(str(iter_dir))
    metadata = reader.read_metadata()
    return {k for k, v in metadata.state_dict_metadata.items() if not isinstance(v, BytesStorageMetadata)}


def _dir_size_bytes(path: Path) -> int:
    """Recursively compute total size of all files under *path*."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@pytest.mark.slow
def test_remove_optimizer(mbridge_eden_checkpoint, tmp_path):
    """Train checkpoint -> remove optimizer -> verify weights remain, optimizer gone, size smaller."""
    src_iter_dir = Path(mbridge_eden_checkpoint)
    src_ckpt_dir = src_iter_dir.parent
    dst_ckpt_dir = tmp_path / "weights_only"

    # Verify source has optimizer keys (tensor keys only for comparison)
    src_keys = _get_dcp_keys(src_iter_dir)
    src_tensor_keys = _get_tensor_keys(src_iter_dir)
    src_optimizer_tensor_keys = {k for k in src_tensor_keys if k.startswith("optimizer.")}
    src_model_tensor_keys = {k for k in src_tensor_keys if not k.startswith("optimizer.")}
    assert len(src_model_tensor_keys) > 0, f"Source checkpoint should have model keys, got: {sorted(src_keys)[:20]}"
    assert len(src_optimizer_tensor_keys) > 0, (
        f"Source checkpoint should have optimizer keys, got: {sorted(src_keys)[:20]}"
    )

    src_size = _dir_size_bytes(src_ckpt_dir)

    # Run optimizer removal
    remove_optimizer(src_ckpt_dir, dst_ckpt_dir)

    # Verify destination structure
    dst_iter_dir = dst_ckpt_dir / src_iter_dir.name
    assert dst_iter_dir.exists(), f"Destination iter dir not found: {dst_iter_dir}"
    assert (dst_iter_dir / "run_config.yaml").exists(), "run_config.yaml should be copied"
    assert (dst_iter_dir / "train_state.pt").exists(), "train_state.pt should be copied"

    # Verify no optimizer keys in destination (compare tensor keys only - bytes/extra_state are TE metadata)
    dst_tensor_keys = _get_tensor_keys(dst_iter_dir)
    dst_optimizer_tensor_keys = {k for k in dst_tensor_keys if k.startswith("optimizer.")}
    dst_model_tensor_keys = {k for k in dst_tensor_keys if not k.startswith("optimizer.")}
    assert len(dst_optimizer_tensor_keys) == 0, (
        f"Destination should have no optimizer keys, got: {sorted(dst_optimizer_tensor_keys)[:10]}"
    )
    assert dst_model_tensor_keys == src_model_tensor_keys, "Model tensor keys should be identical"

    # Verify model weights match
    src_tensor_keys = _get_tensor_keys(src_iter_dir)
    src_weight_keys = {k for k in src_tensor_keys if not k.startswith("optimizer.")}

    src_sd = {}
    reader = FileSystemReader(str(src_iter_dir))
    metadata = reader.read_metadata()
    for key, item_meta in metadata.state_dict_metadata.items():
        if isinstance(item_meta, BytesStorageMetadata):
            continue
        if not key.startswith("optimizer."):
            src_sd[key] = torch.empty(item_meta.size, dtype=item_meta.properties.dtype, device="cpu")
    dcp.load(state_dict=src_sd, storage_reader=reader, no_dist=True)

    dst_sd = {}
    reader = FileSystemReader(str(dst_iter_dir))
    metadata = reader.read_metadata()
    for key, item_meta in metadata.state_dict_metadata.items():
        if isinstance(item_meta, BytesStorageMetadata):
            continue
        dst_sd[key] = torch.empty(item_meta.size, dtype=item_meta.properties.dtype, device="cpu")
    dcp.load(state_dict=dst_sd, storage_reader=reader, no_dist=True)

    for key in sorted(src_weight_keys):
        assert key in dst_sd, f"Missing model key in destination: {key}"
        torch.testing.assert_close(src_sd[key], dst_sd[key], msg=f"Weight mismatch for {key}")

    # Verify destination is smaller
    dst_size = _dir_size_bytes(dst_ckpt_dir)
    assert dst_size < src_size, f"Destination ({dst_size}) should be smaller than source ({src_size})"

    # Verify common.pt has no optimizer metadata
    common_path = dst_iter_dir / "common.pt"
    if common_path.exists():
        common = torch.load(common_path, map_location="cpu", weights_only=False)
        assert "optimizer" not in common, "common.pt should not have 'optimizer' key"
        assert "opt_param_scheduler" not in common, "common.pt should not have 'opt_param_scheduler' key"


@pytest.mark.slow
def test_remove_optimizer_cli(mbridge_eden_checkpoint, tmp_path):
    """Test the eden_remove_optimizer CLI entry point."""
    src_iter_dir = Path(mbridge_eden_checkpoint)
    src_ckpt_dir = src_iter_dir.parent
    dst_ckpt_dir = tmp_path / "cli_weights_only"

    result = subprocess.run(
        [
            "eden_remove_optimizer",
            "--src-ckpt-dir",
            str(src_ckpt_dir),
            "--dst-ckpt-dir",
            str(dst_ckpt_dir),
            "--verbose",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    dst_iter_dir = dst_ckpt_dir / src_iter_dir.name
    assert dst_iter_dir.exists(), f"Destination iter dir not found: {dst_iter_dir}"

    dst_tensor_keys = _get_tensor_keys(dst_iter_dir)
    dst_optimizer_keys = {k for k in dst_tensor_keys if k.startswith("optimizer.")}
    dst_model_keys = {k for k in dst_tensor_keys if not k.startswith("optimizer.")}
    assert len(dst_optimizer_keys) == 0, f"CLI output should have no optimizer keys, got: {sorted(dst_optimizer_keys)}"
    assert len(dst_model_keys) > 0, "CLI output should have model keys"
