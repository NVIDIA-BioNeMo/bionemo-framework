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

"""Tests for chunked SingleCellMemMapDataset functionality."""

import numpy as np
import pytest

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.scdl_constants import Backend


def test_to_chunked(tmp_path, make_h5ad_with_raw):
    """Convert to chunked, verify data and features match."""
    h5ad_path = make_h5ad_with_raw(tmp_path)
    original = SingleCellMemMapDataset(tmp_path / "orig", h5ad_path=h5ad_path)
    chunked = original.to_chunked(str(tmp_path / "chunked"), chunk_size=30)

    # Basic properties
    assert chunked._is_chunked
    assert chunked.header.backend == Backend.CHUNKED_MEMMAP_V0
    assert len(chunked) == len(original)

    # Data matches
    for idx in range(len(original)):
        (orig_vals, orig_cols), _, _ = original.get_row(idx)
        (chunk_vals, chunk_cols), _, _ = chunked.get_row(idx)
        np.testing.assert_array_equal(orig_vals, chunk_vals)
        np.testing.assert_array_equal(orig_cols, chunk_cols)

    # Features preserved
    assert len(chunked._var_feature_index) == len(original._var_feature_index)
    assert chunked._obs_feature_index.number_of_rows() == original._obs_feature_index.number_of_rows()


def test_to_chunked_inplace(tmp_path, make_h5ad_with_raw):
    """In-place conversion replaces original with chunked."""
    h5ad_path = make_h5ad_with_raw(tmp_path)
    scdl_path = tmp_path / "scdl"
    SingleCellMemMapDataset(scdl_path, h5ad_path=h5ad_path)

    chunked = SingleCellMemMapDataset(scdl_path).to_chunked(chunk_size=30)

    assert chunked._is_chunked
    assert chunked.data_path == str(scdl_path)


def test_to_chunked_already_chunked_raises(tmp_path, make_h5ad_with_raw):
    """Cannot chunk an already chunked dataset."""
    h5ad_path = make_h5ad_with_raw(tmp_path)
    original = SingleCellMemMapDataset(tmp_path / "orig", h5ad_path=h5ad_path)
    chunked = original.to_chunked(str(tmp_path / "chunked"), chunk_size=30)

    with pytest.raises(ValueError, match="already chunked"):
        chunked.to_chunked()
