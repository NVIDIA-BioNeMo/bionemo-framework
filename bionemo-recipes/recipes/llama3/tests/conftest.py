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

import sys
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.distributed.device_mesh import _mesh_resources, init_device_mesh


sys.path.append(Path(__file__).parent.parent.as_posix())
sys.path.append(Path(__file__).parent.as_posix())

from distributed_config import DistributedConfig


@pytest.fixture
def recipe_path() -> Path:
    """Return the root directory of the recipe."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def mock_genomic_parquet(tmp_path_factory) -> Path:
    """Create a mock genomic sequences parquet file for testing.
    
    This fixture creates a small parquet file with synthetic genomic sequences
    that can be used for training tests without relying on external data files.
    
    Returns:
        Path to the generated parquet file
    """
    tmp_dir = tmp_path_factory.mktemp("data")
    parquet_path = tmp_dir / "test_genomic_sequences.parquet"
    
    # Create mock genomic sequences with simple repeating patterns
    # These are easy for the model to overfit to, which is perfect for sanity tests
    sequences = [
        "ATCG" * 300,  # 1200 bp - simple ATCG repeat
        "AAAA" * 250 + "TTTT" * 250,  # 2000 bp - alternating A and T blocks
        "GCGC" * 200,  # 800 bp - GC repeat
        "ACGT" * 400,  # 1600 bp - all 4 nucleotides
        "TGCA" * 350,  # 1400 bp - reverse pattern
    ]
    
    # Create parquet table with 'sequence' column
    table = pa.table({
        "sequence": sequences,
    })
    
    pq.write_table(table, parquet_path)
    return parquet_path


@pytest.fixture(scope="session", autouse=True)
def device_mesh():
    """Create a re-usable device mesh for testing.

    This is a "auto-use", session-scope fixture so that a single device mesh is created and used in all tests.

    Megatron-FSDP throws issues when re-creating the torch device mesh in the same process, starting in the 25.09 NGC
    pytorch container release. To work around this, we create a re-usable device mesh that use in all single-process
    tests.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    device_mesh = init_device_mesh("cuda", mesh_shape=(1, 1), mesh_dim_names=("dp", "tp"))

    # Mock these torch.distributed functions so that we re-use the same device mesh, and don't re-create or destroy the
    # global process group.
    with (
        mock.patch("torch.distributed.device_mesh.init_device_mesh", return_value=device_mesh),
        mock.patch("torch.distributed.init_process_group", return_value=None),
        mock.patch("torch.distributed.destroy_process_group", return_value=None),
    ):
        yield

    # At the end of all tests, destroy the process group and clear the device mesh resources.
    torch.distributed.destroy_process_group()
    _mesh_resources.mesh_stack.clear()
    _mesh_resources.child_to_root_mapping.clear()
    _mesh_resources.root_to_flatten_mapping.clear()
    _mesh_resources.flatten_name_to_root_dims.clear()
    _mesh_resources.mesh_dim_group_options.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


