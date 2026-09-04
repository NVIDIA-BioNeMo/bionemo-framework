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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Tests for producing model-only Megatron Bridge checkpoints."""

from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultSavePlanner, FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

from bionemo.common.checkpoint.remove_optimizer import remove_optimizer


def _write_checkpoint_with_model_object(tmp_path: Path) -> tuple[Path, str]:
    source_iteration = tmp_path / "source" / "iter_0000001"
    source_iteration.mkdir(parents=True)
    extra_state_key = "decoder.final_norm._extra_state/shard_0_1"
    dcp.save(
        state_dict={
            "decoder.final_norm.weight": torch.arange(4, dtype=torch.float32),
            # Transformer Engine object state can contain tensors. DCP must retain
            # the containing object under this exact metadata key rather than
            # flattening its contents into a ``.0`` child key.
            extra_state_key: [torch.empty(0, dtype=torch.uint8)],
            "optimizer.state.exp_avg": torch.ones(4),
        },
        storage_writer=FileSystemWriter(str(source_iteration)),
        planner=DefaultSavePlanner(flatten_state_dict=False),
        no_dist=True,
    )
    source_metadata = FileSystemReader(str(source_iteration)).read_metadata()
    assert isinstance(source_metadata.state_dict_metadata[extra_state_key], BytesStorageMetadata)
    return source_iteration, extra_state_key


def test_remove_optimizer_omits_model_object_state_by_default(tmp_path: Path) -> None:
    """The existing reducer contract continues to emit tensor weights only by default."""
    source_iteration, extra_state_key = _write_checkpoint_with_model_object(tmp_path)

    destination = tmp_path / "model-only"
    remove_optimizer(source_iteration, destination)

    prepared_metadata = FileSystemReader(str(destination / source_iteration.name)).read_metadata()
    assert "decoder.final_norm.weight" in prepared_metadata.state_dict_metadata
    assert extra_state_key not in prepared_metadata.state_dict_metadata
    assert "optimizer.state.exp_avg" not in prepared_metadata.state_dict_metadata


def test_remove_optimizer_can_preserve_model_object_state(tmp_path: Path) -> None:
    """An explicit opt-in retains Transformer Engine object shards for strict loading."""
    source_iteration, extra_state_key = _write_checkpoint_with_model_object(tmp_path)

    destination = tmp_path / "model-only"
    remove_optimizer(source_iteration, destination, preserve_model_object_state=True)

    prepared_iteration = destination / source_iteration.name
    prepared_metadata = FileSystemReader(str(prepared_iteration)).read_metadata()
    assert extra_state_key in prepared_metadata.state_dict_metadata
    assert isinstance(prepared_metadata.state_dict_metadata[extra_state_key], BytesStorageMetadata)
    assert "optimizer.state.exp_avg" not in prepared_metadata.state_dict_metadata

    loaded_extra_state = {extra_state_key: BytesStorageMetadata()}
    dcp.load(
        state_dict=loaded_extra_state,
        storage_reader=FileSystemReader(str(prepared_iteration)),
        no_dist=True,
    )
    assert isinstance(loaded_extra_state[extra_state_key], list)
    assert len(loaded_extra_state[extra_state_key]) == 1
    torch.testing.assert_close(loaded_extra_state[extra_state_key][0], torch.empty(0, dtype=torch.uint8))
