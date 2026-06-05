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

from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset


first_array_values = [1, 2, 3, 4, 5]
second_array_values = [10, 9, 8, 7, 6, 5, 4, 3]


@pytest.fixture
def generate_dataset(tmp_path, test_directory) -> SingleCellMemMapDataset:
    """
    Create a SingleCellMemMapDataset, save and reload it

    Args:
        tmp_path: temporary directory fixture
    Returns:
        A SingleCellMemMapDataset
    """
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad", use_X_not_raw=True)
    ds.save()
    del ds
    reloaded = SingleCellMemMapDataset(tmp_path / "scy")
    assert np.array_equal(np.array(reloaded.row_index), ad.read_h5ad(test_directory / "adata_sample0.h5ad").X.indptr)
    assert np.array_equal(np.array(reloaded.col_index), ad.read_h5ad(test_directory / "adata_sample0.h5ad").X.indices)
    assert np.array_equal(np.array(reloaded.data), ad.read_h5ad(test_directory / "adata_sample0.h5ad").X.data)
    return reloaded


@pytest.fixture
def create_and_fill_mmap_arrays(tmp_path) -> Tuple[np.memmap, np.memmap]:
    """
    Instantiate and fill two np.memmap arrays.

    Args:
        tmp_path: temporary directory fixture
    Returns:
        Two instantiated np.memmap arrays.
    """
    arr1 = np.memmap(tmp_path / "x.npy", dtype="uint32", shape=(len(first_array_values),), mode="w+")
    arr1[:] = np.array(first_array_values, dtype="uint32")

    arr2 = np.memmap(tmp_path / "y.npy", dtype="uint32", shape=(len(second_array_values),), mode="w+")
    arr2[:] = np.array(second_array_values, dtype="uint32")
    return arr1, arr2


@pytest.fixture
def compare_fn():
    def _compare(dns: SingleCellMemMapDataset, dt: SingleCellMemMapDataset) -> bool:
        """
        Returns whether two SingleCellMemMapDatasets are equal

        Args:
            dns: SingleCellMemMapDataset
            dnt: SingleCellMemMapDataset
        Returns:
            True if these datasets are equal.
        """

        assert dns.number_of_rows() == dt.number_of_rows()
        assert dns.number_of_values() == dt.number_of_values()
        assert dns.number_nonzero_values() == dt.number_nonzero_values()
        assert dns.number_of_variables() == dt.number_of_variables()
        assert dns.number_of_rows() == dt.number_of_rows()
        assert np.array_equal(np.array(dns.row_index), np.array(dt.row_index))
        assert np.array_equal(np.array(dns.col_index), np.array(dt.col_index))
        assert np.array_equal(np.array(dns.data), np.array(dt.data))
        assert dns.dtypes == dt.dtypes, f"Dtype mismatch: {dns.dtypes} != {dt.dtypes}"

    return _compare


@pytest.fixture
def big_h5ad_data():
    """Fixture providing large CSR matrix test data for dtype promotion tests (integer-approximate)."""
    n_rows, n_cols = 2, 70_000
    data = np.array([1.5, 70_000.5345, 10.0], dtype="float32")
    indices = np.array([0, 10, 65_537], dtype=np.int64)
    indptr = np.array([0, 1, 3], dtype=np.int64)
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "data": data,
        "indices": indices,
        "indptr": indptr,
    }


@pytest.fixture
def big_int_h5ad(tmp_path, big_h5ad_data):
    """Create and return the path to an h5ad with large values/columns for dtype promotion tests."""
    d = big_h5ad_data
    X = sp.csr_matrix((d["data"].astype(np.uint32), d["indices"], d["indptr"]), shape=(d["n_rows"], d["n_cols"]))
    a = ad.AnnData(obs=pd.DataFrame(index=range(d["n_rows"])), raw={"X": X})
    h5ad_path = tmp_path / "big_dtype.h5ad"
    a.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.fixture
def big_float_h5ad(tmp_path, big_h5ad_data):
    """Create and return the path to an h5ad with large values/columns for dtype promotion tests."""
    d = big_h5ad_data
    X = sp.csr_matrix((d["data"].astype("float32"), d["indices"], d["indptr"]), shape=(d["n_rows"], d["n_cols"]))
    a = ad.AnnData(obs=pd.DataFrame(index=range(d["n_rows"])), raw={"X": X})
    h5ad_path = tmp_path / "big_dtype.h5ad"
    a.write_h5ad(h5ad_path)
    return h5ad_path


def test_empty_dataset_save_and_reload(tmp_path):
    ds = SingleCellMemMapDataset(data_path=tmp_path / "scy", num_rows=2, num_elements=10)
    ds.save()
    del ds
    reloaded = SingleCellMemMapDataset(tmp_path / "scy")
    assert reloaded.number_of_rows() == 0
    assert reloaded.number_of_variables() == [0]
    assert reloaded.number_of_values() == 0
    assert len(reloaded) == 0
    assert len(reloaded[1][0]) == 0


def test_wrong_arguments_for_dataset(tmp_path):
    with pytest.raises(
        ValueError, match=r"An np.memmap path, an h5ad path, or the number of elements and rows is required"
    ):
        SingleCellMemMapDataset(data_path=tmp_path / "scy")


def test_load_h5ad(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad", use_X_not_raw=True)
    assert ds.number_of_rows() == 8
    assert ds.number_of_variables() == [10]
    assert len(ds) == 8
    assert ds.number_of_values() == 80
    assert ds.number_nonzero_values() == 5
    assert np.isclose(ds.sparsity(), 0.9375, rtol=1e-6)
    assert np.array_equal(ds.data, [6.0, 19.0, 12.0, 16.0, 1.0])
    assert len(ds) == 8
    assert ds.dtypes["data.npy"] == "float32"
    assert ds.dtypes["col_ptr.npy"] == "uint8"
    assert ds.dtypes["row_ptr.npy"] == "uint8"


def test_load_h5ad_with_raw(tmp_path, test_directory):
    # Create a dataset with no X data, but with raw X data
    adata = ad.read_h5ad(test_directory / "adata_sample0.h5ad")
    adata_new = ad.AnnData(X=None, raw={"X": adata.X}, var=adata.var, obs=adata.obs)
    adata_new.write_h5ad(tmp_path / "adata_sample0_raw.h5ad")

    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=tmp_path / "adata_sample0_raw.h5ad")
    assert ds.number_of_rows() == 8
    assert ds.number_of_variables() == [10]
    assert len(ds) == 8
    assert ds.number_of_values() == 80
    assert ds.number_nonzero_values() == 5
    assert np.isclose(ds.sparsity(), 0.9375, rtol=1e-6)
    assert np.array_equal(ds.data, [6.0, 19.0, 12.0, 16.0, 1.0])
    assert len(ds) == 8
    assert ds.dtypes["data.npy"] == "float32"
    assert ds.dtypes["col_ptr.npy"] == "uint8"
    assert ds.dtypes["row_ptr.npy"] == "uint8"


def test_h5ad_no_file(tmp_path):
    ds = SingleCellMemMapDataset(data_path=tmp_path / "scy", num_rows=2, num_elements=10)
    with pytest.raises(FileNotFoundError, match=rf"Error: could not find h5ad path {tmp_path}/a"):
        ds.load_h5ad(anndata_path=tmp_path / "a")


def test_SingleCellMemMapDataset_constructor(generate_dataset):
    assert generate_dataset.number_of_rows() == 8
    assert generate_dataset.number_of_variables() == [10]
    assert generate_dataset.number_of_values() == 80
    assert generate_dataset.number_nonzero_values() == 5
    assert np.isclose(generate_dataset.sparsity(), 0.9375, rtol=1e-6)
    assert len(generate_dataset) == 8

    assert generate_dataset.shape() == (8, [10])
    # Dtype expectations: integer-valued counts in 0-255, 10 columns, 5 nnz
    assert generate_dataset.dtypes["data.npy"] == "float32"
    assert generate_dataset.dtypes["col_ptr.npy"] == "uint8"
    assert generate_dataset.dtypes["row_ptr.npy"] == "uint8"


def test_SingleCellMemMapDataset_get_row(generate_dataset):
    assert len(generate_dataset[0][0]) == 1
    vals, cols = generate_dataset[0]
    assert vals[0] == 6.0
    assert cols[0] == 2
    assert len(generate_dataset[1][1]) == 0
    assert len(generate_dataset[1][0]) == 0
    vals, cols = generate_dataset[2]
    assert vals[0] == 19.0
    assert cols[0] == 2
    vals, cols = generate_dataset[7]
    assert vals[0] == 1.0
    assert cols[0] == 8


def test_SingleCellMemMapDataset_get_row_colum(generate_dataset):
    assert generate_dataset.get_row_column(0, 0, impute_missing_zeros=True) == 0.0
    assert generate_dataset.get_row_column(0, 0, impute_missing_zeros=False) is None
    assert generate_dataset.get_row_column(0, 2) == 6.0
    assert generate_dataset.get_row_column(6, 3) == 16.0
    assert generate_dataset.get_row_column(3, 2) == 12.0


def test_SingleCellMemMapDataset_get_row_padded(generate_dataset):
    padded_row, var_feats, _ = generate_dataset.get_row_padded(
        0, return_var_features=True, var_feature_names=["feature_name"]
    )
    assert len(padded_row) == 10
    assert padded_row[2] == 6.0
    assert len(var_feats[0]) == 10
    assert generate_dataset.get_row_padded(0)[0][0] == 0.0
    assert generate_dataset.data[0] == 6.0
    assert generate_dataset.data[1] == 19.0
    assert len(generate_dataset.get_row_padded(2)[0]) == 10


def test_concat_SingleCellMemMapDatasets_same(tmp_path, big_int_h5ad):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=big_int_h5ad)
    dt = SingleCellMemMapDataset(tmp_path / "sct", h5ad_path=big_int_h5ad)
    dt.concat(ds)

    assert dt.number_of_rows() == 2 * ds.number_of_rows()
    assert dt.number_of_values() == 2 * ds.number_of_values()
    assert dt.number_nonzero_values() == 2 * ds.number_nonzero_values()


# TODO: Addtest that raises error
def test_concat_SingleCellMemMapDatasets_empty(tmp_path, big_float_h5ad):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=big_float_h5ad)
    exp_rows = np.array(ds.row_index)
    exp_cols = np.array(ds.col_index)
    exp_data = np.array(ds.data)

    ds.concat([])
    assert (np.array(ds.row_index) == exp_rows).all()
    assert (np.array(ds.col_index) == exp_cols).all()
    assert (np.array(ds.data) == exp_data).all()


@pytest.mark.parametrize("extend_copy_size", [1, 10 * 1_024 * 1_024])
def test_concat_SingleCellMemMapDatasets_underlying_memmaps_regular_X(tmp_path, test_directory, extend_copy_size):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad", use_X_not_raw=True)
    dt = SingleCellMemMapDataset(tmp_path / "sct", h5ad_path=test_directory / "adata_sample1.h5ad", use_X_not_raw=True)
    exp_rows = np.append(dt.row_index, ds.row_index[1:] + len(dt.col_index))
    exp_cols = np.append(dt.col_index, ds.col_index)
    exp_data = np.append(dt.data, ds.data)

    dt.concat(ds, extend_copy_size)
    assert (np.array(dt.row_index) == exp_rows).all()
    assert (np.array(dt.col_index) == exp_cols).all()
    assert (np.array(dt.data) == exp_data).all()
    # Dtypes should remain minimal and consistent
    assert dt.dtypes["data.npy"] == "float32"
    assert dt.dtypes["col_ptr.npy"] == "uint8"
    assert dt.dtypes["row_ptr.npy"] == "uint8"


@pytest.mark.parametrize("extend_copy_size", [1, 10 * 1_024 * 1_024])
def test_concat_SingleCellMemMapDatasets_underlying_memmaps_raw_X(
    tmp_path, big_float_h5ad, big_int_h5ad, extend_copy_size
):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=big_float_h5ad)
    dt = SingleCellMemMapDataset(tmp_path / "sct", h5ad_path=big_int_h5ad)
    exp_rows = np.append(dt.row_index, ds.row_index[1:] + len(dt.col_index))
    exp_cols = np.append(dt.col_index, ds.col_index)
    exp_data = np.append(dt.data, ds.data)

    dt.concat(ds, extend_copy_size)
    assert (np.array(dt.row_index) == exp_rows).all()
    assert (np.array(dt.col_index) == exp_cols).all()
    assert (np.array(dt.data) == exp_data).all()
    # Dtypes should remain minimal and consistent
    assert dt.dtypes["data.npy"] == "float32"
    assert dt.dtypes["col_ptr.npy"] == "uint32"
    assert dt.dtypes["row_ptr.npy"] == "uint8"


def test_concat_SingleCellMemMapDatasets_diff(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad", use_X_not_raw=True)
    dt = SingleCellMemMapDataset(tmp_path / "sct", h5ad_path=test_directory / "adata_sample1.h5ad", use_X_not_raw=True)

    exp_number_of_rows = ds.number_of_rows() + dt.number_of_rows()
    exp_n_val = ds.number_of_values() + dt.number_of_values()
    exp_nnz = ds.number_nonzero_values() + dt.number_nonzero_values()
    dt.concat(ds)
    assert dt.number_of_rows() == exp_number_of_rows
    assert dt.number_of_values() == exp_n_val
    assert dt.number_nonzero_values() == exp_nnz
    # Dtypes should promote safely; for sample inputs they remain uint8
    assert dt.dtypes["data.npy"] == "float32"
    assert dt.dtypes["col_ptr.npy"] == "uint8"
    assert dt.dtypes["row_ptr.npy"] == "uint8"


def test_concat_SingleCellMemMapDatasets_multi(tmp_path, compare_fn, test_directory):
    ds = SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad", use_X_not_raw=True)
    dt = SingleCellMemMapDataset(tmp_path / "sct", h5ad_path=test_directory / "adata_sample1.h5ad", use_X_not_raw=True)
    dx = SingleCellMemMapDataset(
        tmp_path / "sccx", h5ad_path=test_directory / "adata_sample2.h5ad", use_X_not_raw=True
    )
    exp_n_obs = ds.number_of_rows() + dt.number_of_rows() + dx.number_of_rows()
    dt.concat(ds)
    dt.concat(dx)
    assert dt.number_of_rows() == exp_n_obs
    dns = SingleCellMemMapDataset(
        tmp_path / "scdns", h5ad_path=test_directory / "adata_sample1.h5ad", use_X_not_raw=True
    )
    dns.concat([ds, dx])
    compare_fn(dns, dt)


def test_lazy_load_SingleCellMemMapDatasets_one_dataset(tmp_path, compare_fn, test_directory):
    ds_regular = SingleCellMemMapDataset(
        tmp_path / "sc1", h5ad_path=test_directory / "adata_sample1.h5ad", use_X_not_raw=True
    )
    ds_lazy = SingleCellMemMapDataset(
        tmp_path / "sc2",
        h5ad_path=test_directory / "adata_sample1.h5ad",
        paginated_load_cutoff=0,
        load_block_row_size=2,
        use_X_not_raw=True,
    )
    compare_fn(ds_regular, ds_lazy)


def test_lazy_load_SingleCellMemMapDatasets_another_dataset(tmp_path, compare_fn, test_directory):
    ds_regular = SingleCellMemMapDataset(
        tmp_path / "sc1", h5ad_path=test_directory / "adata_sample0.h5ad", use_X_not_raw=True
    )
    ds_lazy = SingleCellMemMapDataset(
        tmp_path / "sc2",
        h5ad_path=test_directory / "adata_sample0.h5ad",
        paginated_load_cutoff=0,
        load_block_row_size=3,
        use_X_not_raw=True,
    )
    compare_fn(ds_regular, ds_lazy)


def test_lazy_load_SingleCellMemMapDatasets_with_raw(tmp_path, compare_fn, make_h5ad_with_raw):
    h5ad_path = make_h5ad_with_raw(tmp_path)

    # Assert that the anndata file has different data in .raw.X and .X
    adata = ad.read_h5ad(h5ad_path)
    # but these are sparse
    assert not (adata.raw.X != adata.X).nnz == 0

    ds_regular = SingleCellMemMapDataset(tmp_path / "regular_load", h5ad_path=h5ad_path)
    ds_lazy = SingleCellMemMapDataset(
        tmp_path / "lazy_load",
        h5ad_path=h5ad_path,
        paginated_load_cutoff=0,
        load_block_row_size=2,
    )
    compare_fn(ds_regular, ds_lazy)


def test_lazy_load_SingleCellMemMapDatasets_with_raw_load_regular(tmp_path, compare_fn, make_h5ad_with_raw):
    h5ad_path = make_h5ad_with_raw(tmp_path)

    # Assert that the anndata file has different data in .raw.X and .X
    adata = ad.read_h5ad(h5ad_path)
    assert not (adata.raw.X != adata.X).nnz == 0

    ds_regular = SingleCellMemMapDataset(tmp_path / "regular_load", h5ad_path=h5ad_path, use_X_not_raw=True)
    ds_lazy = SingleCellMemMapDataset(
        tmp_path / "lazy_load",
        h5ad_path=h5ad_path,
        paginated_load_cutoff=0,
        load_block_row_size=2,
        use_X_not_raw=True,
    )
    compare_fn(ds_regular, ds_lazy)


@pytest.mark.parametrize("dtype", [None, "uint32", "uint64", "float32", "float64"])
def test_load_h5ad_properly_converted_dtypes_int(tmp_path, test_directory, big_int_h5ad, big_h5ad_data, dtype):
    """Use shared big-dtype h5ad to force dtype promotion and verify results."""
    ds = SingleCellMemMapDataset(tmp_path / "scy_big", h5ad_path=big_int_h5ad, data_dtype=dtype)
    assert ds.dtypes["data.npy"] == dtype if dtype is not None else "float32"
    assert ds.dtypes["col_ptr.npy"] == "uint32"
    assert ds.dtypes["row_ptr.npy"] == "uint8"
    assert np.array_equal(ds.data, big_h5ad_data["data"].astype(int))
    assert np.array_equal(ds.col_index, big_h5ad_data["indices"])
    assert np.array_equal(ds.row_index, big_h5ad_data["indptr"])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_load_h5ad_properly_converted_dtypes_float(tmp_path, test_directory, big_int_h5ad, big_h5ad_data, dtype):
    """Use shared big-dtype h5ad to force dtype promotion and verify results."""
    ds = SingleCellMemMapDataset(tmp_path / "scy_big", h5ad_path=big_int_h5ad, data_dtype=dtype)

    assert ds.dtypes["data.npy"] == dtype if dtype is not None else "float32"
    assert ds.dtypes["col_ptr.npy"] == "uint32"
    assert ds.dtypes["row_ptr.npy"] == "uint8"
    assert np.array_equal(ds.data, big_h5ad_data["data"].astype(int))
    assert np.array_equal(ds.col_index, big_h5ad_data["indices"])
    assert np.array_equal(ds.row_index, big_h5ad_data["indptr"])


@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
    ],
)
def test_load_h5ad_overflows_on_loading_dtypes_int(tmp_path, test_directory, big_int_h5ad, dtype):
    """Verify that requesting too-small data dtype leads to overflow and loss of precision."""
    with pytest.warns(UserWarning, match="Downcasted data values for 'data.npy' are not close to original values."):
        SingleCellMemMapDataset(tmp_path / "scy_big", h5ad_path=big_int_h5ad, data_dtype=dtype)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_load_h5ad_overflows_on_loading_dtypes_paginated_int(tmp_path, test_directory, big_int_h5ad, dtype):
    """Verify that requesting too-small data dtype leads to overflow and loss of precision."""
    with pytest.warns(UserWarning, match="Downcasted data values for 'data.npy' are not close to original values."):
        SingleCellMemMapDataset(
            tmp_path / "scy_big",
            h5ad_path=big_int_h5ad,
            data_dtype=dtype,
            paginated_load_cutoff=0,
            load_block_row_size=2,
        )


@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
    ],
)
def test_load_h5ad_overflows_on_loading_dtypes_float(tmp_path, test_directory, big_float_h5ad, dtype):
    """Verify that requesting too-small data dtype leads to overflow and loss of precision."""
    with pytest.warns(UserWarning, match="Downcasted data values for 'data.npy' are not close to original values."):
        SingleCellMemMapDataset(tmp_path / "scy_big", h5ad_path=big_float_h5ad, data_dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
    ],
)
def test_load_h5ad_overflows_on_loading_dtypes_paginated_float(tmp_path, test_directory, big_float_h5ad, dtype):
    """Verify that requesting too-small data dtype leads to overflow and loss of precision."""
    with pytest.warns(UserWarning, match="Downcasted data values for 'data.npy' are not close to original values."):
        SingleCellMemMapDataset(
            tmp_path / "scy_big",
            h5ad_path=big_float_h5ad,
            data_dtype=dtype,
            paginated_load_cutoff=0,
            load_block_row_size=2,
        )


@pytest.mark.parametrize(
    "dtype",
    [
        "uint32",
        "uint64",
    ],
)
def test_load_h5ad_does_not_overflow_on_high_tolerance(tmp_path, test_directory, big_float_h5ad, big_h5ad_data, dtype):
    """Verify that high tolerance does not lead to overflow."""
    ds = SingleCellMemMapDataset(
        tmp_path / "scy_big", h5ad_path=big_float_h5ad, data_dtype=dtype, data_dtype_tolerance=1
    )
    assert np.array_equal(ds.data, big_h5ad_data["data"].astype(dtype))


def test_concat_SingleCellMemMapDatasets_raises_diff_dtypes(tmp_path, test_directory):
    ds_float = SingleCellMemMapDataset(
        tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad", data_dtype="float32", use_X_not_raw=True
    )
    dt_int = SingleCellMemMapDataset(
        tmp_path / "sct", h5ad_path=test_directory / "adata_sample1.h5ad", data_dtype="uint8", use_X_not_raw=True
    )
    with pytest.raises(ValueError, match="mix of int and float dtypes"):
        ds_float.concat(dt_int)

    with pytest.raises(ValueError, match="mix of int and float dtypes"):
        dt_int.concat(ds_float)


def test_cast_data_dtype_updates_dtype_and_preserves_values(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(
        tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad", data_dtype="float32", use_X_not_raw=True
    )
    original = np.array(ds.data, copy=True)

    ds.cast_data_to_dtype("float64")

    reloaded = SingleCellMemMapDataset(tmp_path / "scy")
    assert reloaded.dtypes["data.npy"] == "float64"
    np.testing.assert_allclose(np.array(reloaded.data, dtype=np.float64), original.astype(np.float64), rtol=0, atol=0)


def test_cast_data_dtype_downscales_dtype_and_preserves_values(tmp_path, test_directory):
    ds = SingleCellMemMapDataset(
        tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad", data_dtype="float32", use_X_not_raw=True
    )
    original = np.array(ds.data, copy=True)

    ds.cast_data_to_dtype("uint16")

    reloaded = SingleCellMemMapDataset(tmp_path / "scy")
    assert reloaded.dtypes["data.npy"] == "uint16"
    np.testing.assert_allclose(np.array(reloaded.data, dtype=np.float64), original.astype(np.float64), rtol=0, atol=0)


def test_concat_rowptr_dtype_changes_on_concatenation(tmp_path, make_two_datasets):
    """Each nnz < 225 individually; combined > 255 → row_ptr switches to uint16."""
    ds1, ds2, expected_row_ptr, expected_cols, expected_data = make_two_datasets(tmp_path, "float32", "float32")

    ds1.concat(ds2)

    assert ds1.dtypes["row_ptr.npy"] == "uint16"
    assert ds1.dtypes["data.npy"] == "float32"
    assert ds1.dtypes["col_ptr.npy"] == "uint8"
    np.testing.assert_array_equal(np.array(ds1.row_index), expected_row_ptr)
    np.testing.assert_array_equal(np.array(ds1.col_index), expected_cols)
    np.testing.assert_allclose(np.array(ds1.data), expected_data, rtol=0, atol=0)


def test_concat_rowptr_dtype_error_on_data_mismatch_on_concatenation(tmp_path, make_two_datasets):
    """Each nnz < 225 individually; combined > 255 → row_ptr switches to uint16."""
    ds1, ds2, _, _, _ = make_two_datasets(tmp_path, "float32", "uint8")
    with pytest.raises(ValueError, match="Cannot merge datasets with a mix of int and float dtypes for data: "):
        ds1.concat(ds2)


def test_SingleCellMemMapDataset_obs_features_identical_to_anndata_source(
    tmp_path, create_cellx_val_data, assert_index_state
):
    memmap_data = tmp_path / "out"
    adata = ad.read_h5ad(create_cellx_val_data / "sidx_40575621_2_0.h5ad")
    ds = SingleCellMemMapDataset(
        memmap_data, h5ad_path=create_cellx_val_data / "sidx_40575621_2_0.h5ad", use_X_not_raw=True
    )

    assert_index_state(ds.obs_features(), length=1, rows=adata.obs.shape[0], col_widths=[adata.obs.shape[1]])
    obs_feats0 = ds.get_row(index=0, return_obs_features=True)[2]
    obs_feats1 = ds.get_row(index=1, return_obs_features=True)[2]
    assert np.array_equal(obs_feats0, adata.obs.iloc[0].tolist())
    assert np.array_equal(obs_feats1, adata.obs.iloc[1].tolist())


def test_SingleCellMemMapDataset_var_features_identical_to_anndata_source(
    tmp_path, create_cellx_val_data, assert_index_state
):
    memmap_data = tmp_path / "out"
    ds = SingleCellMemMapDataset(
        memmap_data, h5ad_path=create_cellx_val_data / "sidx_40575621_2_0.h5ad", use_X_not_raw=True
    )
    adata = ad.read_h5ad(create_cellx_val_data / "sidx_40575621_2_0.h5ad")
    assert_index_state(ds.var_features(), length=1, rows=adata.shape[0], col_widths=[adata.var.shape[0]])
    var_feats0 = ds.get_row(index=0, return_var_features=True)[1]
    assert np.array_equal(np.stack([adata.var[c].to_numpy() for c in adata.var.columns]), np.stack(var_feats0))


def test_SingleCellMemMapDataset_raises_when_trying_to_load_raw_X_without_raw(tmp_path, test_directory):
    with pytest.raises(
        ValueError,
        match="This file does not have raw count data; set use_X_not_raw=True to use normalized counts instead.",
    ):
        SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=test_directory / "adata_sample0.h5ad")


def test_SingleCellMemMapDataset_raises_when_trying_to_access_X_with_empty_X(tmp_path, big_float_h5ad):
    with pytest.raises(
        ValueError, match="This file does not have count data; set use_X_not_raw=False to use raw counts instead."
    ):
        SingleCellMemMapDataset(tmp_path / "scy", h5ad_path=big_float_h5ad, use_X_not_raw=True)
