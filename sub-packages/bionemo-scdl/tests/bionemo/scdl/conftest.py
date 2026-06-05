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


import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from bionemo.scdl.data.load import load
from bionemo.scdl.index.row_feature_index import ObservedFeatureIndex, VariableFeatureIndex
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset


@pytest.fixture
def test_directory() -> Path:
    """Gets the path to the directory with test data.

    Returns:
        A Path object that is the directory with test data.
    """
    return load("scdl/sample_scdl_feature_ids") / "scdl_data_with_feature_ids"


@pytest.fixture
def test_neighbor_directory() -> Path:
    """Gets the path to the directory with neighbor test data.

    Returns:
        A Path object that is the directory with neighbor test data.
    """
    return load("scdl/sample_scdl_neighbor")


@pytest.fixture
def create_cellx_val_data(tmpdir) -> Path:
    """Gets the path to the directory with test data.

    Returns:
        A Path object that is the directory with test data.
    """
    cellx_input_val_path = (
        load("scdl/testdata-20240506") / "cellxgene_2023-12-15_small" / "input_data" / "val" / "assay__10x_3_v2/"
    )
    file1 = (
        cellx_input_val_path
        / "sex__female/development_stage__74-year-old_human_stage/self_reported_ethnicity__Asian/tissue_general__lung/dataset_id__f64e1be1-de15-4d27-8da4-82225cd4c035/sidx_40575621_2_0.h5ad"
    )
    file2 = (
        cellx_input_val_path
        / "sex__male/development_stage__82-year-old_human_stage/self_reported_ethnicity__European/tissue_general__lung/dataset_id__f64e1be1-de15-4d27-8da4-82225cd4c035/sidx_40596188_1_0.h5ad"
    )
    collated_dir = tmpdir / "collated_val"
    collated_dir.mkdir()
    shutil.copy(file1, collated_dir)
    shutil.copy(file2, collated_dir)
    return collated_dir


# ==== Fixtures for VariableFeatureIndex and ObservedFeatureIndex ======
@pytest.fixture
def make_feat_dictionary():
    """Create a simple dictionary with num_cols columns of identical length num_rows. This will be used to create a
    VariableFeatureIndex or ObservedFeatureIndex. num_cols is the number of columns in the dictionary, width is the length of the columns, and
    key_prefix is the prefix of the keys in the dictionary."""

    def _make(num_cols: int, width: int, *, key_prefix: str = "f") -> dict[str, np.ndarray]:
        feats: dict[str, np.ndarray] = {}
        for c in range(num_cols):
            # some random values here
            feats[f"{key_prefix}{c}"] = np.random.randint(0, 100, size=width)
        return feats

    return _make


@pytest.fixture
def assert_index_state():
    """Assert properties of a VariableFeatureIndex or ObservedFeatureIndex are what is expected."""

    def _assert(
        idx: VariableFeatureIndex | ObservedFeatureIndex,
        *,
        length: int | None = None,
        rows: int | None = None,
        col_widths: list[int] | None = None,
        values: list[int] | None = None,
    ) -> None:
        if length is not None:
            assert len(idx) == length
        if rows is not None:
            assert idx.number_of_rows() == rows
        if col_widths is not None:
            assert idx.column_dims() == col_widths
        if values is not None:
            assert idx.number_of_values() == values

    return _assert


# ==== Creating H5ad files to check downcasting performance ======
@pytest.fixture
def make_random_csr():
    def _make_random_csr(total_nnz: int, n_cols: int, seed: int = 42, fn_prefix: str = "random_csr"):
        rng = np.random.default_rng(seed)
        indptr = np.arange(total_nnz + 1, dtype=np.int64)
        indices = rng.integers(0, n_cols, total_nnz)
        data = np.ones(total_nnz, dtype=np.float64)
        X = sp.csr_matrix((data, indices, indptr), shape=(total_nnz, n_cols))

        return X

    return _make_random_csr


@pytest.fixture
def make_two_datasets(make_random_csr):
    """Factory to create two datasets and expected arrays for concatenation tests."""

    def _make(tmp_path, dtype1: str, dtype2: str):
        X1 = make_random_csr(total_nnz=224, n_cols=200)
        X2 = make_random_csr(total_nnz=224, n_cols=200)

        h1 = tmp_path / "var1.h5ad"
        h2 = tmp_path / "var2.h5ad"
        ad.AnnData(
            obs=pd.DataFrame(index=range(X1.shape[0])), var=pd.DataFrame(index=np.arange(X1.shape[1])), raw={"X": X1}
        ).write_h5ad(h1)
        ad.AnnData(
            obs=pd.DataFrame(index=range(X2.shape[0])), var=pd.DataFrame(index=np.arange(X2.shape[1])), raw={"X": X2}
        ).write_h5ad(h2)

        ds1 = SingleCellMemMapDataset(tmp_path / "var_ds1", h5ad_path=h1, data_dtype=dtype1)
        ds2 = SingleCellMemMapDataset(tmp_path / "var_ds2", h5ad_path=h2, data_dtype=dtype2)

        expected_row_ptr = np.concatenate([X1.indptr, X2.indptr[1:] + int(X1.nnz)])
        expected_cols = np.concatenate([X1.indices, X2.indices])
        expected_data = np.concatenate([X1.data, X2.data])
        return ds1, ds2, expected_row_ptr, expected_cols, expected_data

    return _make


@pytest.fixture
def make_small_and_large_h5ads():
    """Factory to create small/large h5ads and expected arrays for collection tests."""

    def _make(tmp_path):
        # Small (float32 non-integers), 4 rows with empty first and third rows
        n_rows_small, n_cols_small = 4, 12
        small_data_vals = np.array([0.5, -1.25, 3.75, 2.0], dtype=np.float32)
        indices_small_vals = np.array([0, 11, 5, 7], dtype=np.int64)
        indptr_small_vals = np.array([0, 0, 2, 2, 4], dtype=np.int64)
        X_small = ad.AnnData(
            obs=pd.DataFrame(index=range(n_rows_small)),
            var=pd.DataFrame(index=np.arange(n_cols_small)),
            raw={
                "X": sp.csr_matrix(
                    (small_data_vals, indices_small_vals, indptr_small_vals), shape=(n_rows_small, n_cols_small)
                )
            },
        )
        small_path = tmp_path / "small.h5ad"
        X_small.write_h5ad(small_path)

        # Large (uint32 via integer-valued with large magnitude), 3 rows with empty middle row
        n_rows_large, n_cols_large = 3, 70_000
        large_data_vals = np.array([70_000.0, 1.0], dtype=np.uint32)
        indices_large_vals = np.array([10, 65_537], dtype=np.int64)
        indptr_large_vals = np.array([0, 1, 1, 2], dtype=np.int64)
        X_large = ad.AnnData(
            obs=pd.DataFrame(index=range(n_rows_large)),
            var=pd.DataFrame(index=np.arange(n_cols_large)),
            raw={
                "X": sp.csr_matrix(
                    (large_data_vals, indices_large_vals, indptr_large_vals), shape=(n_rows_large, n_cols_large)
                )
            },
        )
        large_path = tmp_path / "large.h5ad"
        X_large.write_h5ad(large_path)

        small = {"data_vals": small_data_vals, "indices_vals": indices_small_vals, "indptr_vals": indptr_small_vals}
        large = {"data_vals": large_data_vals, "indices_vals": indices_large_vals, "indptr_vals": indptr_large_vals}
        return small_path, large_path, small, large

    return _make


@pytest.fixture
def make_h5ad_with_raw(make_random_csr):
    """Factory to create an h5ad with uniquely randomized data for the fields .raw.X and .X"""

    def _make(tmp_path):
        X = make_random_csr(total_nnz=100, n_cols=50, seed=42)
        X_raw = make_random_csr(total_nnz=100, n_cols=50, seed=43)
        h = tmp_path / "var.h5ad"
        ad.AnnData(X=X, var=pd.DataFrame(index=np.arange(X.shape[1])), raw={"X": X_raw}).write_h5ad(h)
        return h

    return _make
