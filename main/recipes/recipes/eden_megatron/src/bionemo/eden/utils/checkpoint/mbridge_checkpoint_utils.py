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

"""MBridge checkpoint utilities for Eden.

Provides ``load_mbridge_state_dict`` for reading checkpoint tensors from
Megatron Bridge DCP format. Used by roundtrip tests and HF export.
"""

import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata


def load_mbridge_state_dict(mbridge_ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Load state dict from an mbridge DCP checkpoint directory.

    Args:
        mbridge_ckpt_dir: Path to the mbridge checkpoint root (containing iter_XXXXXXX/),
            or directly to an iter_XXXXXXX directory.

    Returns:
        Flat state dict with all tensor parameters.
    """
    if re.match(r"^iter_\d+$", mbridge_ckpt_dir.name):
        iter_dir = mbridge_ckpt_dir
    elif (latest_file := mbridge_ckpt_dir / "latest_checkpointed_iteration.txt").exists():
        iteration = latest_file.read_text().strip()
        iter_dir = mbridge_ckpt_dir / f"iter_{int(iteration):07d}"
    else:
        iter_dirs = sorted(mbridge_ckpt_dir.glob("iter_*"))
        if not iter_dirs:
            raise FileNotFoundError(f"No iter_* directories in {mbridge_ckpt_dir}")
        iter_dir = iter_dirs[-1]

    reader = FileSystemReader(str(iter_dir))
    metadata = reader.read_metadata()

    state_dict = {}
    for key, item_meta in metadata.state_dict_metadata.items():
        if isinstance(item_meta, BytesStorageMetadata):
            continue
        state_dict[key] = torch.empty(item_meta.size, dtype=item_meta.properties.dtype, device="cpu")

    dcp.load(state_dict=state_dict, storage_reader=reader, no_dist=True)
    return state_dict
