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

import importlib.metadata
import json
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy
import torch

from bionemo.scdl.api.single_cell_row_dataset import SingleCellRowDataset
from bionemo.scdl.index.row_feature_index import ObservedFeatureIndex, VariableFeatureIndex
from bionemo.scdl.schema.header import ArrayDType, ArrayInfo, Backend, FeatureIndexInfo, SCDLHeader
from bionemo.scdl.schema.version import CurrentSCDLVersion
from bionemo.scdl.util.filecopyutil import extend_files
from bionemo.scdl.util.memmap_utils import (
    _create_compressed_sparse_row_memmaps,
    _create_row_memmaps,
    _extract_features,
    _pad_sparse_array,
    determine_dtype,
    smallest_uint_dtype,
)
from bionemo.scdl.util.partition_scdl import partition_scdl
from bionemo.scdl.util.scdl_constants import FLOAT_ORDER, INT_ORDER, FileNames, Mode, NeighborSamplingStrategy


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SingleCellMemMapDataset(SingleCellRowDataset):
    """Represents one or more AnnData matrices.

    Data is stored in large, memory-mapped arrays that enables fast access of
    datasets larger than the available amount of RAM on a system. SCMMAP
    implements a consistent API defined in SingleCellRowDataset.

    Attributes:
        data_path: Location of np.memmap files to be loaded from or that will be
        created.
        mode: Whether the dataset will be read in (r+) from np.memmap files or
        written to np.memmap files (w+).
        data: A numpy array of the data
        row_index: A numpy array of row pointers
        col_index: A numpy array of column values
        metadata: Various metadata about the dataset.
        _feature_index: The corresponding VariableFeatureIndex where features are
        stored
        dtypes: A dictionary containing the datatypes of the data, row_index,
        and col_index arrays.
        _version: The version of the dataset
        load_neighbors (bool, optional): Whether to load and utilize neighbor information
            from the 'neighbor_key' in AnnData's .obsp. Defaults to False.
        neighbor_key (str, optional): The key in AnnData's .obsp containing the
            sparse adjacency matrix for neighbors. Defaults to 'next_cell_ids'.
        neighbor_sampling_strategy (str, optional): Strategy for sampling neighbors ('random').
            Defaults to 'random'.
        fallback_to_identity (bool, optional): If a cell has no neighbors, whether
            to use the cell itself as its neighbor. Defaults to True.
    """

    def __init__(
        self,
        data_path: str,
        h5ad_path: Optional[str] = None,
        num_elements: Optional[int] = None,
        num_rows: Optional[int] = None,
        mode: Mode = Mode.READ_APPEND,
        paginated_load_cutoff: int = 10_000,
        load_block_row_size: int = 1_000_000,
        var_feature_index_name="var_feature_id",
        obs_feature_index_name="obs_feature_id",
        # --- Neighbor Args ---
        load_neighbors: bool = False,
        neighbor_key: str = "next_cell_ids",
        neighbor_sampling_strategy: str = NeighborSamplingStrategy.RANDOM,
        fallback_to_identity: bool = True,
        data_dtype: Optional[str] = None,  # Must be one of INT_ORDER or FLOAT_ORDER in scdl_constants
        data_dtype_tolerance: float = 1e-08,
        use_X_not_raw: bool = False,  # If True, use .X instead of .raw.X for the data
    ) -> None:
        """Instantiate the class.

        Args:
            data_path: The location where the data np.memmap files are read from
            or stored.
            h5ad_path: Optional, the location of the h5_ad path.
            num_elements: The total number of elements in the array.
            num_rows: The number of rows in the data frame.
            mode: Whether to read or write from the data_path.
            paginated_load_cutoff: MB size on disk at which to load the h5ad structure with paginated load.
            load_block_row_size: Number of rows to load into memory with paginated load
            var_feature_index_name: The name of the features if the features are only stored in features_df.index.values
            obs_feature_index_name: The name of the obs features if the features are only stored in features_df.index.values
            # --- New Neighbor Args ---
            load_neighbors (bool, optional): Boolean to control to control whether to load and utilize neighbor information
            neighbor_key (str, optional): The key in AnnData's .obsp containing neighbor information.
            neighbor_sampling_strategy (str, optional): Sampling strategy for neighbors.
            fallback_to_identity (bool, optional): If a cell has no neighbors, whether to use the cell itself as its neighbor.
            data_dtype (str | None, optional): Desired dtype for `data.npy` when creating
                new datasets; if None, defaults to 'float32'. Must be one of
                'uint8','uint16','uint32','uint64','float16','float32','float64'.
            data_dtype_tolerance (float, optional): Tolerance for data type conversion. Defaults to 1e-08.
            use_X_not_raw (bool, optional): If True, use .X instead of .raw.X for the data.
        """
        self._version: str = importlib.metadata.version("bionemo.scdl")
        self.data_path: str = data_path
        self.header: SCDLHeader = None
        self.mode: Mode = mode
        self._is_chunked: bool = False
        self._chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.paginated_load_cutoff = paginated_load_cutoff
        self.load_block_row_size = load_block_row_size
        self.var_feature_index_name = var_feature_index_name
        self.obs_feature_index_name = obs_feature_index_name
        self.use_X_not_raw = use_X_not_raw
        # Backing arrays
        self.data: Optional[np.ndarray] = None
        self.row_index: Optional[np.ndarray] = None
        # Metadata and attributes
        self.metadata: Dict[str, int] = {}

        # Stores the Feature Index, which tracks
        # the original AnnData features (e.g., gene names)
        # and allows us to store ragged arrays in our SCMMAP structure.
        self._var_feature_index: VariableFeatureIndex = VariableFeatureIndex()
        self._obs_feature_index: ObservedFeatureIndex = ObservedFeatureIndex()
        allowed_dtypes = list(INT_ORDER + FLOAT_ORDER)
        if data_dtype is not None and data_dtype not in allowed_dtypes:
            raise ValueError(f"Invalid data_dtype '{data_dtype}'. Must be one of: {', '.join(allowed_dtypes)}")
        # Variables for int packing / reduced precision
        self.dtypes: Dict[FileNames, str] = {
            f"{FileNames.DATA.value}": "float32" if data_dtype is None else data_dtype,
            f"{FileNames.COLPTR.value}": "uint32",
            f"{FileNames.ROWPTR.value}": "uint64",
            f"{FileNames.NEIGHBOR_INDICES.value}": "uint32",
            f"{FileNames.NEIGHBOR_INDICES_PTR.value}": "uint64",
            f"{FileNames.NEIGHBOR_VALUES.value}": "float32",
        }
        self.data_dtype_tolerance = data_dtype_tolerance
        # Neighbor configuration
        self.load_neighbors = load_neighbors
        self._has_neighbors = False
        if load_neighbors:
            self._init_neighbor_args(neighbor_key, neighbor_sampling_strategy, fallback_to_identity)

        if mode == Mode.CREATE_APPEND and os.path.exists(data_path):
            raise FileExistsError(f"Output directory already exists: {data_path}")

        if h5ad_path is not None and (data_path is not None and os.path.exists(data_path)):
            raise FileExistsError(
                "Invalid input; both an existing SCMMAP and an h5ad file were passed. "
                "Please pass either an existing SCMMAP or an h5ad file."
            )

        # If there is only a data path, and it exists already, load SCMMAP data.
        elif data_path is not None and os.path.exists(data_path):
            self.__init__obj()
            self.load(data_path)

        # If there is only an h5ad path, load the HDF5 data
        elif h5ad_path is not None:
            self.__init__obj()
            self.load_h5ad(h5ad_path)
        else:
            match num_rows, num_elements:
                case (int(), int()):
                    self.__init__obj()
                    self._init_arrs(num_elements=num_elements, num_rows=num_rows)
                case _:
                    raise ValueError("An np.memmap path, an h5ad path, or the number of elements and rows is required")

    def _path_in_archive(self, filename: str | Path) -> str:
        """Returns the full path to a file within the archive, joining self.data_path and the filename.

        Args:
            filename: The filename or Path object to resolve within the archive.

        Returns:
            The full path as a string.
        """
        if isinstance(filename, Path):
            filename = str(filename)
        return os.path.join(self.data_path, filename)

    @property
    def header_path(self) -> str:
        """Returns the full path to the header file in the archive.

        Example:
            >>> ds = SingleCellMemMapDataset(data_path="my_data")
            >>> ds.header_path
            'my_data/scdl_header.json'
        """
        return self._path_in_archive(FileNames.HEADER.value)

    def _init_neighbor_args(self, neighbor_key, neighbor_sampling_strategy, fallback_to_identity):
        # Neighbor tracking
        self._has_neighbors = False  # Track if neighbor data was successfully loaded/found

        self.neighbor_key = neighbor_key
        try:
            # Convert string to enum if a string was passed
            if isinstance(neighbor_sampling_strategy, str):
                neighbor_sampling_strategy = NeighborSamplingStrategy(neighbor_sampling_strategy)
            # Validate that it's a valid enum value
            if not isinstance(neighbor_sampling_strategy, NeighborSamplingStrategy):
                raise ValueError(f"Unsupported neighbor_sampling_strategy: {neighbor_sampling_strategy}")
        except ValueError:
            raise ValueError(f"Unsupported neighbor_sampling_strategy: {neighbor_sampling_strategy}")

        self.neighbor_sampling_strategy = neighbor_sampling_strategy
        self.fallback_to_identity = fallback_to_identity

    def __init__obj(self):
        """Initializes the data path and writes the version."""
        os.makedirs(self.data_path, exist_ok=True)

        # Write the version
        if not os.path.exists(f"{self.data_path}/{FileNames.VERSION.value}"):
            with open(f"{self.data_path}/{FileNames.VERSION.value}", "w") as vfi:
                json.dump(self.version(), vfi)

    def _init_arrs(self, num_elements: int, num_rows: int) -> None:
        self.mode = Mode.CREATE_APPEND
        data_arr, col_arr, row_arr = _create_compressed_sparse_row_memmaps(
            num_elements=num_elements,
            num_rows=num_rows,
            memmap_dir_path=Path(self.data_path),
            mode=self.mode,
            dtypes=self.dtypes,
        )
        self.data = data_arr
        self.col_index = col_arr
        self.row_index = row_arr

    def version(self) -> str:
        """Returns a version number.

        (following <major>.<minor>.<point> convention).
        """
        return self._version

    def _extract_neighbor_data(self, adata) -> bool:
        """Extracts neighbor data from AnnData.obsp object and saves to memmap files.

        Args:
            adata: AnnData object containing neighbor information
        Returns:
            bool: True if neighbor data was successfully loaded/found, False otherwise.
        """
        # Check if neighbor key exists in AnnData.obsp
        if self.neighbor_key not in adata.obsp:
            warnings.warn(f"Neighbor key '{self.neighbor_key}' not found in AnnData.obsp. Neighbor loading skipped.")
            return False

        logger.info(f"Extracting neighbor data from {self.neighbor_key} in AnnData.obsp")

        # Get the neighbor matrix from obsp
        neighbor_matrix = adata.obsp[self.neighbor_key]

        # Check if the neighbor matrix is a sparse matrix
        if not scipy.sparse.issparse(neighbor_matrix):
            raise ValueError(f"Neighbor matrix for key '{self.neighbor_key}' is not a sparse matrix.")

        # Initialize memory-mapped arrays for neighbor data with proper sizes
        indptr_len = len(neighbor_matrix.indptr)
        nnz = len(neighbor_matrix.indices)  # number of non-zero elements
        # No need to calculate data_len separately since it equals nnz

        # Create memory-mapped arrays for neighbor data
        self._neighbor_indptr = np.memmap(
            f"{self.data_path}/{FileNames.NEIGHBOR_INDICES_PTR.value}",
            dtype=self.dtypes[f"{FileNames.NEIGHBOR_INDICES_PTR.value}"],
            mode=Mode.CREATE_APPEND.value,
            shape=(indptr_len,),
        )

        self._neighbor_indices = np.memmap(
            f"{self.data_path}/{FileNames.NEIGHBOR_INDICES.value}",
            dtype=self.dtypes[f"{FileNames.NEIGHBOR_INDICES.value}"],
            mode=Mode.CREATE_APPEND.value,
            shape=(nnz,),
        )

        self._neighbor_data = np.memmap(
            f"{self.data_path}/{FileNames.NEIGHBOR_VALUES.value}",
            dtype=self.dtypes[f"{FileNames.NEIGHBOR_VALUES.value}"],
            mode=Mode.CREATE_APPEND.value,
            shape=(nnz,),
        )

        # Copy data into memory-mapped arrays (with dtype conversion)
        self._neighbor_indptr[:] = neighbor_matrix.indptr.astype(
            self.dtypes[f"{FileNames.NEIGHBOR_INDICES_PTR.value}"]
        )
        self._neighbor_indices[:] = neighbor_matrix.indices.astype(self.dtypes[f"{FileNames.NEIGHBOR_INDICES.value}"])
        self._neighbor_data[:] = neighbor_matrix.data.astype(self.dtypes[f"{FileNames.NEIGHBOR_VALUES.value}"])

        logger.info("Neighbor data extracted to memory-mapped arrays")
        return True

    def cast_data_to_dtype(self, dtype: str) -> None:
        """Casts the data dtype of the dataset to the given dtype. This will convert the data memory map in-place on the disk.

        Args:
            dtype: The dtype to cast the data to. Must be one of INT_ORDER + FLOAT_ORDER.
        """
        allowed_dtypes = list(INT_ORDER + FLOAT_ORDER)

        if dtype is None or dtype not in allowed_dtypes:
            raise ValueError(f"Invalid data_dtype '{dtype}'. Must be one of: {', '.join(allowed_dtypes)}")

        # writes the new dtype to the disk
        self._convert_dataset_to_new_dtypes(new_dtypes={FileNames.DATA.value: dtype}, allow_downscaling=True)
        # Save the updated header (with a new data dtype to the disk)
        self._write_header()

    def _extract_neighbor_data_paginated(self, adata) -> bool:
        """Extracts neighbor data using paginated approach for large datasets.

        Uses the same pattern as paginated_load_h5ad with binary file I/O and chunking
        to efficiently handle large neighbor matrices without loading everything at once.

        Args:
            adata: AnnData object containing neighbor information
        Returns:
            bool: True if neighbor data was successfully loaded/found, False otherwise.
        """
        # Check if neighbor key exists in AnnData.obsp
        if self.neighbor_key not in adata.obsp:
            warnings.warn(f"Neighbor key '{self.neighbor_key}' not found in AnnData.obsp. Neighbor loading skipped.")
            return False

        logger.info(f"Extracting neighbor data from {self.neighbor_key} in AnnData.obsp using chunked approach")

        # Get the neighbor matrix from obsp
        neighbor_matrix = adata.obsp[self.neighbor_key]

        # Check if the neighbor matrix is a sparse matrix
        if not scipy.sparse.issparse(neighbor_matrix):
            raise ValueError(f"Neighbor matrix for key '{self.neighbor_key}' is not a sparse matrix.")

        # First write indptr which gives us the structure - this is usually small enough to handle in one go
        memmap_dir_path = Path(self.data_path)
        with open(f"{memmap_dir_path}/{FileNames.NEIGHBOR_INDICES_PTR.value}", "wb") as indptr_file:
            # Convert to hardcoded dtype before writing
            indptr_converted = neighbor_matrix.indptr.astype(self.dtypes[f"{FileNames.NEIGHBOR_INDICES_PTR.value}"])
            indptr_file.write(indptr_converted.tobytes())

        # Get dimensions from indptr
        num_rows = len(neighbor_matrix.indptr) - 1
        # Process indices and data in chunks based on rows
        with (
            open(f"{memmap_dir_path}/{FileNames.NEIGHBOR_INDICES.value}", "wb") as indices_file,
            open(f"{memmap_dir_path}/{FileNames.NEIGHBOR_VALUES.value}", "wb") as data_file,
        ):
            for row_start in range(0, num_rows, self.load_block_row_size):
                row_end = min(row_start + self.load_block_row_size, num_rows)

                # Get slice of the matrix for this chunk of rows
                chunk = neighbor_matrix[row_start:row_end]

                # Convert to hardcoded dtypes before writing
                indices_converted = chunk.indices.astype(self.dtypes[f"{FileNames.NEIGHBOR_INDICES.value}"])
                data_converted = chunk.data.astype(self.dtypes[f"{FileNames.NEIGHBOR_VALUES.value}"])

                # Write chunk data to files
                indices_file.write(indices_converted.tobytes())
                data_file.write(data_converted.tobytes())

                logger.info(f"Processed neighbor data rows {row_start} to {row_end - 1}")

        # Then re-open as memory-mapped arrays with the final shapes
        self._neighbor_indptr = np.memmap(
            f"{self.data_path}/{FileNames.NEIGHBOR_INDICES_PTR.value}",
            dtype=self.dtypes[f"{FileNames.NEIGHBOR_INDICES_PTR.value}"],
            mode=Mode.READ_APPEND.value,
            shape=(len(neighbor_matrix.indptr),),
        )

        self._neighbor_indices = np.memmap(
            f"{self.data_path}/{FileNames.NEIGHBOR_INDICES.value}",
            dtype=self.dtypes[f"{FileNames.NEIGHBOR_INDICES.value}"],
            mode=Mode.READ_APPEND.value,
            shape=(len(neighbor_matrix.indices),),
        )

        self._neighbor_data = np.memmap(
            f"{self.data_path}/{FileNames.NEIGHBOR_VALUES.value}",
            dtype=self.dtypes[f"{FileNames.NEIGHBOR_VALUES.value}"],
            mode=Mode.READ_APPEND.value,
            shape=(len(neighbor_matrix.data),),
        )

        logger.info("Neighbor data extracted to memory-mapped arrays using chunked approach")
        return True

    def get_row(
        self,
        index: int,
        return_var_features: bool = False,
        var_feature_names: Optional[List[str]] = None,
        return_obs_features: bool = False,
        obs_feature_names: Optional[List[str]] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Returns a given row in the dataset along with optional features.

        Args:
            index: The row to be returned. This is in the range of [0, num_rows)
            return_var_features: boolean that indicates whether to return features
            var_feature_names: Optional, variable feature names to extract
            return_obs_features: boolean indicating whether to return observed (row) features
            obs_feature_names: Optional, observed feature variables to extract
        Return:
            [Tuple[np.ndarray, np.ndarray]: data values and column pointes
            List[np.ndarray]: optional, corresponding variable (column) features.
            List[np.ndarray]: optional, corresponding observed (row) features.
        """
        if self._is_chunked:
            chunk_id, local_idx = self.header.chunked_info.get_chunk_for_row(index)
            data, rowptr, colptr = self._chunks[chunk_id]
            start, end = rowptr[local_idx], rowptr[local_idx + 1]
            values, columns = data[start:end], colptr[start:end]
        else:
            start = self.row_index[index]
            end = self.row_index[index + 1]
            values = self.data[start:end]
            columns = self.col_index[start:end]
        ret = (values, columns)
        var_features = (
            self._var_feature_index.lookup(index, select_features=var_feature_names)[0]
            if return_var_features
            else None
        )
        obs_features = (
            self._obs_feature_index.lookup(index, select_features=obs_feature_names)[0]
            if return_obs_features
            else None
        )
        return ret, var_features, obs_features

    def get_row_with_neighbor(
        self,
        index: int,
        return_var_features: bool = False,
        var_feature_names: Optional[List[str]] = None,
        return_obs_features: bool = False,
        obs_feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Union[Tuple[np.ndarray, np.ndarray], int, Optional[List[np.ndarray]]]]:
        """Returns a given row in the dataset along with optional features and neighbor data.

        Args:
            index: The row to be returned. This is in the range of [0, num_rows)
            return_var_features: Boolean that indicates whether to return variable features
            var_feature_names: Optional, variable feature names to extract
            return_obs_features: Boolean that indicates whether to return observed features
            obs_feature_names: Optional, observed feature variables to extract

        Returns:
            Dict with keys:
            - 'current_cell': Tuple[np.ndarray, np.ndarray] - (values, columns) for current cell
            - 'next_cell': Tuple[np.ndarray, np.ndarray] - (values, columns) for neighbor cell
            - 'current_cell_index': int - Index of current cell
            - 'next_cell_index': int - Index of neighbor cell
            - 'var_features': List[np.ndarray] - Variable features if return_features is True, else None
            - 'obs_features': List[np.ndarray] - Observed features if return_obs_features is True, else None

        Raises:
            ValueError: If neighbor functionality is disabled or no neighbor data is available
        """
        # Validate neighbor availability since this function requires neighbors
        if not (self.load_neighbors and self._has_neighbors):
            raise ValueError(
                "Cannot include neighbor data: neighbor functionality is disabled or no neighbor data available"
            )

        # Get current cell data using the existing get_row function
        current_cell_data, var_features, obs_features = self.get_row(
            index, return_var_features, var_feature_names, return_obs_features, obs_feature_names
        )

        # Sample neighbor and get its data
        neighbor_index = self.sample_neighbor_index(index)

        # Case where neighbor is the same as current cell
        if neighbor_index == index:
            next_cell_data = current_cell_data
        else:
            # Get neighbor cell data using the get_row function
            next_cell_data, _, _ = self.get_row(neighbor_index, False, None)

        # Return all data in a dictionary format
        return {
            "current_cell": current_cell_data,
            "next_cell": next_cell_data,
            "current_cell_index": index,
            "next_cell_index": neighbor_index,
            "var_features": var_features,
            "obs_features": obs_features,
        }

    def get_row_padded(
        self,
        index: int,
        return_var_features: bool = False,
        var_feature_names: Optional[List[str]] = None,
        return_obs_features: bool = False,
        obs_feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Returns a padded version of a row in the dataset.

        A padded version is one where the a sparse array representation is
        converted to a conventional represenentation. Optionally, features are
        returned.

        Args:
            index: The row to be returned
            return_var_features: boolean that indicates whether to return variable features
            var_feature_names: Optional, variable feature names to extract
            return_obs_features: Boolean that indicates whether to return observed features
            obs_feature_names: Optional, observed feature variables to extract
        Return:
            np.ndarray: conventional row representation
            List[np.ndarray]: optional, corresponding variable (column) features.
            List[np.ndarray]: optional, corresponding observed (row) features.
        """
        (row_values, row_column_pointer), var_features, obs_features = self.get_row(
            index, return_var_features, var_feature_names, return_obs_features, obs_feature_names
        )
        return (
            _pad_sparse_array(row_values, row_column_pointer, self._var_feature_index.number_vars_at_row(index)),
            var_features,
            obs_features,
        )

    def get_row_padded_with_neighbor(
        self,
        index: int,
        return_var_features: bool = False,
        var_feature_names: Optional[List[str]] = None,
        return_obs_features: bool = False,
        obs_feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Union[np.ndarray, int, List[np.ndarray]]]:
        """Returns a padded version of a row with optional neighbor data.

        A padded version converts sparse representation to a dense array where
        missing values are filled with zeros.

        Args:
            index: The row to be returned
            return_var_features: Boolean that indicates whether to return variable features
            var_feature_names: Optional, variable feature names to extract
            return_obs_features: Boolean that indicates whether to return observed features
            obs_feature_names: Optional, observed feature variables to extract

        Returns:
            Dict with keys:
            - 'current_cell': np.ndarray - Padded array for current cell
            - 'next_cell': np.ndarray - Padded array for neighbor cell
            - 'current_cell_index': int - Index of current cell
            - 'next_cell_index': int - Index of neighbor cell
            - 'features': List[np.ndarray] - Variable features if return_features is True, else None

        Raises:
            ValueError: If neighbor functionality is disabled or no neighbor data is available
        """
        # Validate neighbor availability since this function requires neighbors
        if not (self.load_neighbors and self._has_neighbors):
            raise ValueError(
                "Cannot include neighbor data: neighbor functionality is disabled or no neighbor data available"
            )

        # Get both current cell and neighbor data
        result = self.get_row_with_neighbor(
            index, return_var_features, var_feature_names, return_obs_features, obs_feature_names
        )

        # Get current cell padded array using get_row_padded
        curr_padded, _, _ = self.get_row_padded(index, False, None, False, None)

        # For neighbor, get the padded array
        next_idx = result["next_cell_index"]
        if next_idx == index:
            # If neighbor is the same as current cell, reuse the current padded array
            next_padded = curr_padded
        else:
            # Otherwise get the neighbor's padded array
            next_padded, _, _ = self.get_row_padded(next_idx, False, None)

        # Return in dictionary format
        return {
            "current_cell": curr_padded,
            "next_cell": next_padded,
            "current_cell_index": result["current_cell_index"],
            "next_cell_index": result["next_cell_index"],
            "var_features": result["var_features"],
            "obs_features": result["obs_features"],
        }

    def get_row_column(self, index: int, column: int, impute_missing_zeros: bool = True) -> Optional[float]:
        """Returns the value at a given index and the corresponding column.

        Args:
            index: The index to be returned
            column: The column to be returned
            impute_missing_zeros: boolean that indicates whether to set missing
            data to 0
        Return:
            A float that is the value in the array or None.
        """
        (row_values, row_column_pointer), _, _ = self.get_row(index)
        if column is not None:
            for col_index, col in enumerate(row_column_pointer):
                if col == column:
                    # return the value at this position
                    return row_values[col_index]
                elif col > column:
                    try:
                        raise ValueError(f"Column pointer {col} is larger than the column {column}.")
                    except ValueError:
                        break
            return 0.0 if impute_missing_zeros else None

    def var_features(self) -> Optional[VariableFeatureIndex]:
        """Return the corresponding VariableFeatureIndex."""
        return self._var_feature_index

    def obs_features(self) -> Optional[ObservedFeatureIndex]:
        """Return the corresponding ObservedFeatureIndex."""
        return self._obs_feature_index

    def _load_mmap_file_if_exists(self, file_path, dtype):
        if os.path.exists(file_path):
            return np.memmap(file_path, dtype=dtype, mode=self.mode.value)
        else:
            raise FileNotFoundError(f"The mmap file at {file_path} is missing")

    def load(self, stored_path: str) -> None:
        """Loads the data at store_path that is an np.memmap format.

        Args:
            stored_path: directory with np.memmap files
        Raises:
            FileNotFoundError if the corresponding directory or files are not
            found, or if the metadata file is not present.
        """
        if not os.path.exists(stored_path):
            raise FileNotFoundError(
                f"""Error: the specified data path to the mmap files {stored_path} does not exist.
                                    Specify an updated filepath or provide an h5ad path to the dataset. The data can
                                    be loaded with SingleCellMemMapDataset.load_h5ad. Alternatively, the class can be instantiated
                                    with  SingleCellMemMapDataset(<path to data that will be created>, h5ad_path=<path to h5ad file>"""
            )
        self.data_path = stored_path
        self.mode = Mode.READ_APPEND
        # Load header if present; keep None if missing or unreadable
        if os.path.exists(self.header_path):
            try:
                self.header = SCDLHeader.load(str(self.header_path))
            except Exception as e:
                warnings.warn(f"Failed to load SCDL header at {self.header_path}: {e}")
                self.header = None
        else:
            warnings.warn(f"SCDL header missing at {self.header_path}; continuing without header.")
            self.header = None
        # If header is loaded, extract dtypes from header and set self.dtypes accordingly
        if self.header is not None and hasattr(self.header, "arrays"):
            # Map from FileNames.value to dtype string
            for array_info in self.header.arrays:
                if FileNames[array_info.name].value not in self.dtypes:
                    raise ValueError(f"Array name {FileNames[array_info.name].value} not found in dtypes")
                self.dtypes[FileNames[array_info.name].value] = array_info.dtype.numpy_dtype_string

        # Load metadata if exists
        metadata_path = f"{self.data_path}/{FileNames.METADATA.value}"
        if os.path.exists(metadata_path):
            with open(metadata_path, Mode.READ_APPEND.value) as mfi:
                self.metadata = json.load(mfi)

        # Load feature indices
        if os.path.exists(f"{self.data_path}/{FileNames.VAR_FEATURES.value}"):
            self._var_feature_index = VariableFeatureIndex.load(f"{self.data_path}/{FileNames.VAR_FEATURES.value}")
        elif os.path.exists(f"{self.data_path}/{FileNames.FEATURES.value}"):
            self._var_feature_index = VariableFeatureIndex.load(f"{self.data_path}/{FileNames.FEATURES.value}")
        if os.path.exists(f"{self.data_path}/{FileNames.OBS_FEATURES.value}"):
            self._obs_feature_index = ObservedFeatureIndex.load(f"{self.data_path}/{FileNames.OBS_FEATURES.value}")

        # Load data arrays - chunked vs monolithic
        if self.header is not None and self.header.backend == Backend.CHUNKED_MEMMAP_V0:
            self._is_chunked = True
            self._load_chunk_memmaps()
        else:
            self.data = self._load_mmap_file_if_exists(
                f"{self.data_path}/{FileNames.DATA.value}", self.dtypes[f"{FileNames.DATA.value}"]
            )
            self.row_index = self._load_mmap_file_if_exists(
                f"{self.data_path}/{FileNames.ROWPTR.value}", dtype=self.dtypes[f"{FileNames.ROWPTR.value}"]
            )
            self.col_index = self._load_mmap_file_if_exists(
                f"{self.data_path}/{FileNames.COLPTR.value}", dtype=self.dtypes[f"{FileNames.COLPTR.value}"]
            )
            if self.load_neighbors:
                self._load_neighbor_memmaps()

    def _load_chunk_memmaps(self) -> None:
        """Preload all chunk memmaps (lazy - just file handles, no RAM)."""
        for chunk_id in range(self.header.chunked_info.num_chunks):
            chunk_path = Path(self.data_path) / f"chunk_{chunk_id:05d}"
            self._chunks.append(
                (
                    np.memmap(chunk_path / FileNames.DATA.value, dtype=self.dtypes[FileNames.DATA.value], mode="r"),
                    np.memmap(
                        chunk_path / FileNames.ROWPTR.value, dtype=self.dtypes[FileNames.ROWPTR.value], mode="r"
                    ),
                    np.memmap(
                        chunk_path / FileNames.COLPTR.value, dtype=self.dtypes[FileNames.COLPTR.value], mode="r"
                    ),
                )
            )

    def _write_metadata(self) -> None:
        with open(f"{self.data_path}/{FileNames.METADATA.value}", f"{Mode.CREATE.value}") as mfi:
            json.dump(self.metadata, mfi)

    def _check_data_downcast(self, count_data: scipy.sparse.spmatrix, warning_prefix: str = "Warning") -> None:
        count_data_downcast = count_data.data.astype(self.dtypes[f"{FileNames.DATA.value}"])
        if not np.allclose(count_data_downcast, count_data.data, rtol=0, atol=self.data_dtype_tolerance):
            warnings.warn(
                f"{warning_prefix}: Downcasted data values for '{FileNames.DATA.value}' are not close to original values. "
                f"Max abs diff: {np.max(np.abs(count_data_downcast - count_data.data))}"
            )
        return count_data_downcast

    def regular_load_h5ad(
        self,
        anndata_path: str,
    ) -> Tuple[pd.DataFrame, int]:
        """Method for loading an h5ad file into memorySu and converting it to the SCDL format.

        Args:
            anndata_path: location of data to load
        Raises:
            NotImplementedError if the data is not in scipy.sparse.spmatrix format
            ValueError it there is not count data
        Returns:
            pd.DataFrame: var variables for features
            int: number of rows in the dataframe.

        """
        adata = ad.read_h5ad(anndata_path)  # slow
        count_data = self._get_matrix_X(adata)
        if not isinstance(count_data, scipy.sparse.spmatrix):
            raise NotImplementedError("Error: dense matrix loading not yet implemented.")

        self._check_data_downcast(count_data, "First 1000 rows of the dataset")

        # Check and load neighbor data
        # NOTE: More clear to have a check here and not call _extract_neighbor_data() if there no neighbors
        if self.load_neighbors:
            self._has_neighbors = self._extract_neighbor_data(adata)

        num_rows, num_cols = count_data.shape

        num_elements_stored = count_data.nnz
        # Currently, anndata is assumed to be sparse
        self.dtypes[f"{FileNames.ROWPTR.value}"] = smallest_uint_dtype(num_elements_stored)
        self.dtypes[f"{FileNames.COLPTR.value}"] = smallest_uint_dtype(num_cols - 1)
        # Create the arrays.
        self._init_arrs(num_elements_stored, num_rows)
        # Store data
        count_data_downcast = self._check_data_downcast(count_data, "Full Dataset")
        self.data[0:num_elements_stored] = count_data_downcast
        # Store the col idx array
        self.col_index[0:num_elements_stored] = count_data.indices.astype(self.dtypes[f"{FileNames.COLPTR.value}"])

        # Store the row idx array
        self.row_index[0 : num_rows + 1] = count_data.indptr.astype(self.dtypes[f"{FileNames.ROWPTR.value}"])
        vars = adata.var
        obs = adata.obs
        file_handle = getattr(adata, "file", None)
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass
        return vars, obs, num_rows

    def _get_matrix_X(self, adata_obj: Optional[ad.AnnData] = None):
        if self.use_X_not_raw:
            if adata_obj.X is None:
                raise ValueError(
                    "This file does not have count data; set use_X_not_raw=False to use raw counts instead."
                )
            return adata_obj.X
        else:
            if getattr(getattr(adata_obj, "raw", None), "X", None) is None:
                raise ValueError(
                    "This file does not have raw count data; set use_X_not_raw=True to use normalized counts instead."
                )
            return adata_obj.raw.X

    def paginated_load_h5ad(
        self,
        anndata_path: str,
    ) -> Tuple[pd.DataFrame, int]:
        """Method for block loading a larger h5ad file and converting it to the SCDL format.

        This should be used in the case when the entire anndata file cannot be loaded into memory.
        The anndata is loaded into memory load_block_row_size number of rows at a time. Each chunk
        is converted into numpy memory maps which are then concatenated together.

        Raises:
            NotImplementedError if the data is not loaded in the CSRDataset format.

        Returns:
            pd.DataFrame: var variables for features
            int: number of rows in the dataframe.
        """
        adata = ad.read_h5ad(anndata_path, backed=True)
        if self.load_neighbors:
            self._has_neighbors = self._extract_neighbor_data_paginated(adata)
        X_full = self._get_matrix_X(adata)
        if not isinstance(X_full, ad.experimental.CSRDataset):
            raise NotImplementedError("Error: dense matrix loading not yet implemented.")

        # Use slice-then-raw when sampling rows
        count_data = self._get_matrix_X(adata[:1_000])

        # Use full matrix for pointers and shapes
        n_elements = X_full._indptr[-1]
        row_index = X_full._indptr.astype(self.dtypes[f"{FileNames.ROWPTR.value}"])

        self._check_data_downcast(count_data, "First 1000 rows of the dataset")
        num_rows, num_cols = X_full.shape
        self.dtypes[f"{FileNames.COLPTR.value}"] = smallest_uint_dtype(num_cols - 1)
        self.dtypes[f"{FileNames.ROWPTR.value}"] = smallest_uint_dtype(n_elements)
        # Read the row indices into a memory map.
        mode = Mode.CREATE_APPEND
        self.row_index = _create_row_memmaps(num_rows, Path(self.data_path), mode, self.dtypes)
        self.row_index[:] = row_index

        # The data from each column and data chunk of the original anndata file is read in. This is saved into the final
        # location of the memmap file. In this step, it is saved in the binary file format.
        memmap_dir_path = Path(self.data_path)
        with (
            open(f"{memmap_dir_path}/{FileNames.COLPTR.value}", "wb") as col_file,
            open(f"{memmap_dir_path}/{FileNames.DATA.value}", "wb") as data_file,
        ):
            for row_start in range(0, num_rows, self.load_block_row_size):
                adata_block = adata[row_start : row_start + self.load_block_row_size]
                adata_block_X = self._get_matrix_X(adata_block)
                # Write each array's data to the file in binary format
                col_block = adata_block_X.indices.astype(self.dtypes[f"{FileNames.COLPTR.value}"])
                col_file.write(col_block.tobytes())
                count_data_downcast = self._check_data_downcast(
                    adata_block_X, f"Rows {row_start} to {row_start + self.load_block_row_size - 1} of the dataset"
                )

                data_file.write(count_data_downcast.tobytes())

        # The column and data files are re-opened as memory-mapped arrays with the final shape
        mode = Mode.READ_APPEND
        self.col_index = np.memmap(
            f"{memmap_dir_path}/{FileNames.COLPTR.value}",
            self.dtypes[f"{FileNames.COLPTR.value}"],
            mode=mode,
            shape=(n_elements,),
        )
        self.data = np.memmap(
            f"{memmap_dir_path}/{FileNames.DATA.value}",
            dtype=self.dtypes[f"{FileNames.DATA.value}"],
            mode=mode,
            shape=(n_elements,),
        )
        vars = adata.var
        obs = adata.obs
        file_handle = getattr(adata, "file", None)
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass

        return vars, obs, num_rows

    def _load_neighbor_memmaps(self):
        try:
            # mmap the existing arrays
            self._neighbor_indices = self._load_mmap_file_if_exists(
                f"{self.data_path}/{FileNames.NEIGHBOR_INDICES.value}",
                self.dtypes[f"{FileNames.NEIGHBOR_INDICES.value}"],
            )
            self._neighbor_indptr = self._load_mmap_file_if_exists(
                f"{self.data_path}/{FileNames.NEIGHBOR_INDICES_PTR.value}",
                self.dtypes[f"{FileNames.NEIGHBOR_INDICES_PTR.value}"],
            )
            self._neighbor_data = self._load_mmap_file_if_exists(
                f"{self.data_path}/{FileNames.NEIGHBOR_VALUES.value}",
                self.dtypes[f"{FileNames.NEIGHBOR_VALUES.value}"],
            )

            self._has_neighbors = True

        except FileNotFoundError:
            # Neighbor files don't exist - this is OK if load_neighbors=False
            # or if dataset was created without neighbors
            self._has_neighbors = False
            if self.load_neighbors:
                warnings.warn("Neighbor loading was requested but neighbor files are missing")

    def load_h5ad(
        self,
        anndata_path: str,
    ) -> None:
        """Loads an existing AnnData archive from disk.

        This creates a new backing data structure which is saved.
        Note: the storage utilized will roughly double. Currently, the data must
        be in a scipy.sparse.spmatrix format.

        Args:
            anndata_path: location of data to load
        Raises:
            FileNotFoundError if the data path does not exist.
            NotImplementedError if the data is not in scipy.sparse.spmatrix
            format
            ValueError it there is not count data
        """
        if not os.path.exists(anndata_path):
            raise FileNotFoundError(f"Error: could not find h5ad path {anndata_path}")
        file_size_MB = os.path.getsize(anndata_path) / (1_024**2)

        if file_size_MB < self.paginated_load_cutoff:
            var_features_df, obs_features_df, num_rows = self.regular_load_h5ad(anndata_path)
        else:
            var_features_df, obs_features_df, num_rows = self.paginated_load_h5ad(anndata_path)

        var_features = _extract_features(var_features_df, self.var_feature_index_name)
        obs_features = _extract_features(obs_features_df, self.obs_feature_index_name)
        self._var_feature_index.append_features(n_obs=num_rows, features=var_features, label=anndata_path)
        self._obs_feature_index.append_features(features=obs_features, label=anndata_path)
        self.save()

    def _write_header(self):
        ## Write the SCDL header.
        arrays: List[ArrayInfo] = []
        # Use FileNames enums directly to ensure correct dtype lookup
        for fname, matrix in [
            (FileNames.DATA, self.data),
            (FileNames.ROWPTR, self.row_index),
            (FileNames.COLPTR, self.col_index),
        ]:
            # Convert numpy dtype to ArrayDType enum, defaulting reasonably on failures
            dtype_value = self.dtypes.get(fname.value, self.dtypes[FileNames.DATA.value])
            try:
                array_dtype = ArrayDType.from_numpy_dtype(dtype_value)
            except ValueError:
                array_dtype = ArrayDType.FLOAT32_ARRAY

            info = ArrayInfo(
                fname.name,
                len(matrix),
                array_dtype,
                None,
            )
            arrays.append(info)

        # Populate FeatureIndexInfo entries for the feature index directory
        indices: List[FeatureIndexInfo] = []
        for feature_index, feature_index_path in [
            (self._var_feature_index, FileNames.VAR_FEATURES.value),
            (self._obs_feature_index, FileNames.OBS_FEATURES.value),
        ]:
            # If feature index is None, it is not recorded in the header
            if feature_index is None:
                continue

            try:
                num_frames = len(feature_index)
                num_rows = feature_index.number_of_rows()
            except Exception as e:
                warnings.warn(f"Unable to determine length or number_of_rows of feature index: {e}")
                continue

            feature_array_dtype = ArrayDType.STRING_ARRAY
            features_rel_path = f"{feature_index_path}"
            index_files: List[str] = [
                f"{features_rel_path}/cumulative_sum_index.npy",
                f"{features_rel_path}/labels.json",
                f"{features_rel_path}/version.npy",
            ]
            if num_frames > 0:
                num_digits = len(str(num_frames))
                for i in range(num_frames):
                    index_files.append(f"{features_rel_path}/dataframe_{i:0{num_digits}d}.parquet")

            fi_info = FeatureIndexInfo(
                name=feature_index_path,
                length=num_rows,
                dtype=feature_array_dtype,
                index_files=index_files,
                shape=None,
            )
            indices.append(fi_info)

        header = (
            self.header
            if self.header is not None
            else SCDLHeader(
                CurrentSCDLVersion(),
                Backend.MEMMAP_V0,
                arrays,
                indices,
            )
        )
        header.save(self.header_path)

    def save(self, output_path: Optional[str] = None) -> None:
        """Saves the class to a given output path.

        Args:
            output_path: The location to save - not yet implemented and should
            be self.data_path

        Raises:
           NotImplementedError if output_path is not None.
        """
        self._write_header()
        if "num_rows" not in self.metadata:
            self.metadata["num_rows"] = self.number_of_rows()

        self._write_metadata()
        # Write the var and obs feature index. This may not exist.
        self._var_feature_index.save(f"{self.data_path}/{FileNames.VAR_FEATURES.value}")
        self._obs_feature_index.save(f"{self.data_path}/{FileNames.OBS_FEATURES.value}")
        # Ensure the object is in a valid state. These are saved at creation!
        for postfix in [
            f"{FileNames.VERSION.value}",
            f"{FileNames.DATA.value}",
            f"{FileNames.COLPTR.value}",
            f"{FileNames.ROWPTR.value}",
            f"{FileNames.VAR_FEATURES.value}",
            f"{FileNames.OBS_FEATURES.value}",
        ]:
            if not os.path.exists(f"{self.data_path}/{postfix}"):
                raise FileNotFoundError(f"This file should exist from object creation: {self.data_path}/{postfix}")

        self.data.flush()  # NOTE: saves the data to disk, do the approach for neighbor data
        self.row_index.flush()
        self.col_index.flush()

        # Flush neighbor data to disk if it exists
        if self._has_neighbors and self._neighbor_indptr is not None:
            self._neighbor_indptr.flush()
            self._neighbor_indices.flush()
            self._neighbor_data.flush()

        if output_path is not None:
            raise NotImplementedError("Saving to separate path is not yet implemented.")

        return True

    def get_neighbor_indices_for_cell(self, cell_index: int) -> np.ndarray:
        """Returns the array of neighbor indices for a given cell.

        Args:
            cell_index: Index of the cell to get neighbors for

        Returns:
            np.ndarray: Array of neighbor indices, empty if no neighbors or neighbor data unavailable

        Raises:
            IndexError: If cell_index is out of bounds
            ValueError: If neighbor functionality was explicitly enabled but data is unavailable
        """
        if not (0 <= cell_index < self.number_of_rows()):
            raise IndexError(f"Cell index {cell_index} out of bounds for dataset with {self.number_of_rows()} cells")

        # Check if neighbor functionality was requested but is unavailable
        if self.load_neighbors and not self._has_neighbors:
            raise ValueError("Neighbor functionality was enabled but no neighbor data is available")

        if not self.load_neighbors or not self._has_neighbors or self._neighbor_indptr is None:
            return np.array([], dtype=int)  # Return empty array if neighbor data not available

        # Get neighbor indices using CSR format indptr and indices
        start = self._neighbor_indptr[cell_index]
        end = self._neighbor_indptr[cell_index + 1]
        return self._neighbor_indices[start:end]

    def get_neighbor_weights_for_cell(self, cell_index: int) -> np.ndarray:
        """Returns the array of neighbor weights (e.g., pseudotime differences) for a given cell.

        Args:
            cell_index: Index of the cell to get neighbor weights for

        Returns:
            np.ndarray: Array of weights corresponding to neighbors, empty if no neighbors

        Raises:
            IndexError: If cell_index is out of bounds
        """
        # Check if neighbor functionality was requested but is unavailable
        if self.load_neighbors and not self._has_neighbors:
            raise ValueError("Neighbor functionality was enabled but no neighbor data is available")

        if (
            not self.load_neighbors
            or not self._has_neighbors
            or self._neighbor_indptr is None
            or self._neighbor_data is None
        ):
            return np.array([], dtype=float)

        if not (0 <= cell_index < self.number_of_rows()):
            raise IndexError(f"Cell index {cell_index} out of bounds for dataset with {self.number_of_rows()} cells")

        # Get neighbor weights using CSR format indptr and data
        start = self._neighbor_indptr[cell_index]
        end = self._neighbor_indptr[cell_index + 1]
        return self._neighbor_data[start:end]

    def sample_neighbor_index(self, cell_index: int) -> int:
        """Samples a neighbor index for the given cell based on the configured sampling strategy.

        Args:
            cell_index: Index of the cell to sample a neighbor for

        Returns:
            int: Index of the sampled neighbor
                 If no neighbors exist and fallback_to_identity is True, returns cell_index

        Raises:
            ValueError: If an unsupported sampling strategy is specified
            IndexError: If cell_index is out of bounds
        """
        # Basic validation
        if not (0 <= cell_index < self.number_of_rows()):
            raise IndexError(f"Cell index {cell_index} out of bounds for dataset with {self.number_of_rows()} cells")

        # Check if neighbor functionality was requested but is unavailable
        if self.load_neighbors and not self._has_neighbors:
            raise ValueError("Neighbor functionality was enabled but no neighbor data is available")

        # Skip sampling if neighbor functionality is disabled
        if not self.load_neighbors:
            return cell_index  # Always return self as neighbor when neighbors disabled

        # Get the neighbor indices for this cell
        neighbor_indices = self.get_neighbor_indices_for_cell(cell_index)

        # If no neighbors found, handle according to fallback policy
        if len(neighbor_indices) == 0:
            if self.fallback_to_identity:
                return cell_index  # Return the cell itself
            else:
                # NOTE: implement fallback policy here if needed
                warnings.warn(
                    f"Cell {cell_index} has no neighbors and fallback_to_identity=False. "
                    f"Returning cell index itself anyway."
                )
                return cell_index  # Currently always return self if no neighbors

        # Sample neighbor based on strategy
        if self.neighbor_sampling_strategy == NeighborSamplingStrategy.RANDOM:
            # Simple random sampling with equal probability
            chosen_index = np.random.choice(neighbor_indices)
            return chosen_index
        elif self.neighbor_sampling_strategy == NeighborSamplingStrategy.FIRST:
            # First neighbor sampling
            return neighbor_indices[0]
        # NOTE: Future - Add weighted sampling strategy
        else:
            raise ValueError(f"Unsupported neighbor sampling strategy: {self.neighbor_sampling_strategy}")

    def get_neighbor_stats(self) -> dict:
        """Returns statistics about the neighbors in the dataset.

        Returns:
            dict: Dictionary with neighbor statistics:
                - has_neighbors: Whether dataset has neighbor data
                - total_connections: Total number of neighbor relationships
                - min_neighbors_per_cell: Minimum number of neighbors any cell has
                - max_neighbors_per_cell: Maximum number of neighbors any cell has
                - avg_neighbors_per_cell: Average number of neighbors per cell
                - cells_with_no_neighbors: Count of cells that have no neighbors
        """
        if not self._has_neighbors or self._neighbor_indptr is None or self._neighbor_indices is None:
            return {"has_neighbors": False}

        # Calculate stats based on CSR indptr (difference between consecutive elements)
        neighbor_counts = np.diff(self._neighbor_indptr)

        return {
            "has_neighbors": True,
            "total_connections": len(self._neighbor_indices),
            "min_neighbors_per_cell": int(np.min(neighbor_counts)),
            "max_neighbors_per_cell": int(np.max(neighbor_counts)),
            "avg_neighbors_per_cell": float(np.mean(neighbor_counts)),
            "cells_with_no_neighbors": int(np.sum(neighbor_counts == 0)),
        }

    def number_of_values(self) -> int:
        """Get the total number of values in the array.

        For each index, the length of the corresponding np.ndarray of features is counted.

        Returns:
            The sum of lengths of the features in every row
        """
        return sum(self._var_feature_index.number_of_values())

    def number_of_rows(self) -> int:
        """The number of rows in the dataset.

        Returns:
            The number of rows in the dataset
        Raises:
            ValueError if the length of the number of rows in the feature
            index does not correspond to the number of stored rows.
        """
        if self._is_chunked:
            return self.header.chunked_info.total_rows
        if len(self._var_feature_index) > 0 and self._var_feature_index.number_of_rows() != self.row_index.size - 1:
            raise ValueError(
                f"""The number of rows in the feature index {self._var_feature_index.number_of_rows()}
                             does not correspond to the number of rows in the row_index {self.row_index.size - 1}"""
            )
        return self._var_feature_index.number_of_rows()

    def number_nonzero_values(self) -> int:
        """Number of non zero entries in the dataset."""
        return self.data.size

    def __len__(self):
        """Return the number of rows."""
        return self.number_of_rows()

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get the row values located and index idx."""
        return torch.from_numpy(np.stack(self.get_row(idx)[0]))

    def number_of_variables(self) -> List[int]:
        """Get the number of features in every entry in the dataset.

        Returns:
            A list containing the lengths of the features in every row
        """
        feats = self._var_feature_index
        if len(feats) == 0:
            return [0]
        num_vars = feats.column_dims()
        return num_vars

    def shape(self) -> Tuple[int, List[int]]:
        """Get the shape of the dataset.

        This is the number of entries by the the length of the feature index
        corresponding to that variable.

        Returns:
            The number of elements in the dataset
            A list containing the number of variables for each row.
        """
        return self.number_of_rows(), self.number_of_variables()

    def _verify_concat_compatibility_and_types(
        self, other_dataset: Union[list["SingleCellMemMapDataset"], "SingleCellMemMapDataset"]
    ) -> None:
        cumulative_elements = self.number_nonzero_values()
        column_dtypes = [self.dtypes[FileNames.COLPTR.value]]
        data_dtypes = [self.dtypes[FileNames.DATA.value]]

        for dataset in other_dataset:
            if self.version() != dataset.version():
                raise ValueError(
                    f"""Incompatable versions: input version: {dataset.version()},
            this version:  {self.version}"""
                )
            column_dtypes.append(str(dataset.dtypes[FileNames.COLPTR.value]))
            data_dtypes.append(str(dataset.dtypes[FileNames.DATA.value]))
            cumulative_elements += dataset.number_nonzero_values()

        if not (
            all(np.dtype(dt).name in FLOAT_ORDER for dt in data_dtypes)
            or all(np.dtype(dt).name in INT_ORDER for dt in data_dtypes)
        ):
            float_file_names = []
            int_file_names = []
            for dt, name in zip(
                data_dtypes, [self.data_path.name] + [dataset.data_path.name for dataset in other_dataset]
            ):
                dtype_name = np.dtype(dt).name
                if dtype_name in FLOAT_ORDER:
                    float_file_names.append(name)
                elif dtype_name in INT_ORDER:
                    int_file_names.append(name)

            raise ValueError(f"""Cannot merge datasets with a mix of int and float dtypes for data: {data_dtypes};
            Float Data datasets: {", ".join(float_file_names)};
            Int Data datasets: {", ".join(int_file_names)}
            Cast all of the datasets to either int dtypes or to float dtypes before concatenation.
            For example for a dataset with data_dtype "uint8", you can cast it to "float32" with:
            ds = load(data_set_path)
            ds.cast_data_to_dtype("float32")
            This will allow downscaling of the data dtype so it is advisable to examine the data if downcasting.
            """)

        new_dtypes = {
            FileNames.COLPTR.value: determine_dtype(column_dtypes),
            FileNames.DATA.value: determine_dtype(data_dtypes),
            FileNames.ROWPTR.value: smallest_uint_dtype(cumulative_elements),
        }
        return new_dtypes

    def _convert_dataset_to_new_dtypes(
        self, new_dtypes: Dict[str, str], extend_copy_size: int = 10 * 1_024 * 1_024, allow_downscaling: bool = False
    ) -> None:
        # If any dtype is changing, convert the file in-place to the new dtype using extend_files.
        # This ensures that after this block, self.dtypes and the on-disk files are updated to the new dtype.
        for key in [FileNames.COLPTR.value, FileNames.DATA.value, FileNames.ROWPTR.value]:
            if key not in new_dtypes:
                continue
            current_dtype = self.dtypes[key]
            target_dtype = new_dtypes[key]
            if current_dtype != target_dtype:
                # Convert the file in-place to the new dtype using a temporary file
                src_file = f"{self.data_path}/{key}"
                tmp_file = f"{self.data_path}/{key}.tmp"
                # Move the original file to tmp_file
                os.rename(src_file, tmp_file)
                # Create an empty destination file for extend_files to append into
                open(src_file, "wb").close()
                # Use extend_files to convert from tmp_file to src_file (now as the new dtype)
                extend_files(
                    src_file,
                    tmp_file,
                    elements_per_chunk=extend_copy_size,
                    delete_file2_on_complete=True,
                    source_dtype=current_dtype,
                    dest_dtype=target_dtype,
                    allow_downscaling=allow_downscaling,
                )
                # Update dtype in self.dtypes
                self.dtypes[key] = target_dtype

    def concat(
        self,
        other_dataset: Union[list["SingleCellMemMapDataset"], "SingleCellMemMapDataset"],
        extend_copy_size: int = 10 * 1_024 * 1_024,
        output_path: str | None = None,
        destroy_on_copy: bool = False,
    ) -> None:
        """Concatenates one or a list of SingleCellMemMapDatasest to the existing one.

        The data is stored in the same place as for the original data set or at output_path
        if it is set. Then, at output_path or at self.data_path, there would be a saved
        SingleCellMemmpDataset, which can be read in with SingleCellMemmpDataset(output_path).

        Args:
            other_dataset: A SingleCellMemMapDataset or a list of
            SingleCellMemMapDatasets
            extend_copy_size: how much to copy in memory at once
            output_path: location to store new dataset
            destroy_on_copy: Whether to remove the current data_path

        Raises:
           ValueError if the other dataset(s) are not of the same version or
           something of another type is passed in.
        """
        match other_dataset:
            case self.__class__():
                other_dataset = [other_dataset]
            case list():
                pass
            case _:
                raise ValueError(
                    f"Expecting either a {SingleCellMemMapDataset} or a list thereof. Actually got: {type(other_dataset)}"
                )

        # Verify the other dataset or datasets are of the same type.
        new_dtypes = self._verify_concat_compatibility_and_types(other_dataset)

        # Set our mode:
        self.mode: Mode = Mode.READ_APPEND
        if output_path is not None:
            if destroy_on_copy:
                shutil.move(self.data_path, output_path)
            else:
                shutil.copytree(self.data_path, output_path)
            self.data_path = output_path

        self._convert_dataset_to_new_dtypes(new_dtypes)
        # Copy the data from self and other into the new arrays.
        element_counter = self.number_nonzero_values()
        row_counter = self.number_of_rows()
        for mmap in other_dataset:
            extend_files(
                f"{self.data_path}/{FileNames.ROWPTR.value}",
                f"{mmap.data_path}/{FileNames.ROWPTR.value}",
                elements_per_chunk=extend_copy_size,
                delete_file2_on_complete=destroy_on_copy,
                offset=np.dtype(mmap.dtypes[f"{FileNames.ROWPTR.value}"]).itemsize,
                source_dtype=mmap.dtypes[f"{FileNames.ROWPTR.value}"],
                dest_dtype=new_dtypes[f"{FileNames.ROWPTR.value}"],
                add_value=element_counter,
            )

            extend_files(
                f"{self.data_path}/{FileNames.DATA.value}",
                f"{mmap.data_path}/{FileNames.DATA.value}",
                elements_per_chunk=extend_copy_size,
                delete_file2_on_complete=destroy_on_copy,
                source_dtype=mmap.dtypes[f"{FileNames.DATA.value}"],
                dest_dtype=new_dtypes[f"{FileNames.DATA.value}"],
            )
            extend_files(
                f"{self.data_path}/{FileNames.COLPTR.value}",
                f"{mmap.data_path}/{FileNames.COLPTR.value}",
                elements_per_chunk=extend_copy_size,
                delete_file2_on_complete=destroy_on_copy,
                source_dtype=mmap.dtypes[f"{FileNames.COLPTR.value}"],
                dest_dtype=new_dtypes[f"{FileNames.COLPTR.value}"],
            )
            self._var_feature_index.concat(mmap._var_feature_index)
            self._obs_feature_index.concat(mmap._obs_feature_index)
            # Update counters
            element_counter += mmap.number_nonzero_values()
            row_counter += mmap.number_of_rows()

        # Reopen the data, colptr, and rowptr arrays
        self.data = np.memmap(
            f"{self.data_path}/{FileNames.DATA.value}",
            dtype=self.dtypes[f"{FileNames.DATA.value}"],
            shape=(element_counter,),
            mode=Mode.READ_APPEND.value,
        )
        self.row_index = np.memmap(
            f"{self.data_path}/{FileNames.ROWPTR.value}",
            dtype=self.dtypes[f"{FileNames.ROWPTR.value}"],
            shape=(row_counter + 1,),
            mode=Mode.READ_APPEND.value,
        )
        self.col_index = np.memmap(
            f"{self.data_path}/{FileNames.COLPTR.value}",
            dtype=self.dtypes[f"{FileNames.COLPTR.value}"],
            shape=(element_counter,),
            mode=Mode.READ_APPEND.value,
        )
        self.save()

    def to_chunked(
        self, output_path: Optional[str] = None, chunk_size: int = 100_000, delete_original: bool = False
    ) -> "SingleCellMemMapDataset":
        """Convert this dataset to a chunked format for efficient remote access.

        Args:
            output_path: Path where the chunked dataset will be created. If None, replaces in-place.
            chunk_size: Number of rows per chunk (default: 100,000).
            delete_original: If True and output_path is set, delete the original after conversion.

        Returns:
            A new SingleCellMemMapDataset instance pointing to the chunked data.
        """
        if self._is_chunked:
            raise ValueError("Dataset is already chunked")

        src = Path(self.data_path)
        if output_path is None:
            # In-place: partition to temp, then swap
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / "chunked"
                partition_scdl(src, tmp_path, chunk_size=chunk_size)
                shutil.rmtree(src)
                shutil.move(str(tmp_path), str(src))
            return SingleCellMemMapDataset(str(src))

        partition_scdl(src, Path(output_path), chunk_size=chunk_size, delete_original=delete_original)
        return SingleCellMemMapDataset(output_path)
