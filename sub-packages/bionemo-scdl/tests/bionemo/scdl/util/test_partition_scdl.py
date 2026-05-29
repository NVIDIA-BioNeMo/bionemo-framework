# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the SCDL partition script."""

import numpy as np
import pytest

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.schema.header import SCDLHeader
from bionemo.scdl.util.partition_scdl import partition_scdl
from bionemo.scdl.util.scdl_constants import Backend, FileNames


@pytest.fixture
def partitioned_scdl(tmp_path, make_h5ad_with_raw):
    """Create an SCDL dataset and partition it into chunks."""
    CHUNK_SIZE = 50
    h5ad_path = make_h5ad_with_raw(tmp_path)
    scdl_path = tmp_path / "scdl"
    original = SingleCellMemMapDataset(scdl_path, h5ad_path=h5ad_path)
    chunked_path = tmp_path / "chunked"
    header = partition_scdl(scdl_path, chunked_path, chunk_size=CHUNK_SIZE)
    return original, header, chunked_path, CHUNK_SIZE


def test_header_backend_is_chunked(partitioned_scdl):
    """Header has CHUNKED_MEMMAP_V0 backend type."""
    _, header, *_ = partitioned_scdl
    assert header.backend == Backend.CHUNKED_MEMMAP_V0


def test_header_has_chunked_info(partitioned_scdl):
    """Header has chunked_info with correct values."""
    original, header, _, CHUNK_SIZE = partitioned_scdl
    assert header.chunked_info is not None
    assert header.chunked_info.total_rows == len(original)
    assert header.chunked_info.chunk_size == CHUNK_SIZE
    assert header.chunked_info.num_chunks == (len(original) + CHUNK_SIZE - 1) // CHUNK_SIZE


def test_partition_row_data_correctness(partitioned_scdl):
    """Partitioned chunk's row data matches original for all rows."""
    original, header, chunked_path, _ = partitioned_scdl
    chunked_info = header.chunked_info
    dtype_map = {arr.name: arr.dtype.numpy_dtype_string for arr in header.arrays}

    for global_idx in range(chunked_info.total_rows):
        chunk_id, local_idx = chunked_info.get_chunk_for_row(global_idx)
        chunk_dir = chunked_path / f"chunk_{chunk_id:05d}"

        rowptr = np.memmap(chunk_dir / FileNames.ROWPTR.value, dtype=dtype_map["ROWPTR"], mode="r")
        data = np.memmap(chunk_dir / FileNames.DATA.value, dtype=dtype_map["DATA"], mode="r")
        colptr = np.memmap(chunk_dir / FileNames.COLPTR.value, dtype=dtype_map["COLPTR"], mode="r")

        (orig_vals, orig_cols), _, _ = original.get_row(global_idx)
        chunk_vals = data[rowptr[local_idx] : rowptr[local_idx + 1]]
        chunk_cols = colptr[rowptr[local_idx] : rowptr[local_idx + 1]]

        np.testing.assert_array_equal(orig_vals, chunk_vals)
        np.testing.assert_array_equal(orig_cols, chunk_cols)


def test_header_save_load_roundtrip(partitioned_scdl):
    """Header with ChunkedInfo saves and loads correctly from disk."""
    _, header, chunked_path, _ = partitioned_scdl
    loaded = SCDLHeader.load(str(chunked_path / FileNames.HEADER.value))

    assert loaded.backend == header.backend
    assert loaded.chunked_info is not None
    assert loaded.chunked_info.total_rows == header.chunked_info.total_rows
    assert loaded.chunked_info.chunk_size == header.chunked_info.chunk_size
    assert loaded.chunked_info.num_chunks == header.chunked_info.num_chunks
