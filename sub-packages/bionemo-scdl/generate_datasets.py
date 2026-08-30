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


import os
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def generate_random_csr(n_rows, n_cols, sparsity, max_data=500):
    """Generate a random csr_matrix with a given sparsity level.

    Parameters:
    - n_rows: Number of rows in the matrix
    - n_cols: Number of columns in the matrix
    - sparsity: Fraction of matrix that should be zero (e.g., 0.8 means 80% zeros)

    Returns:
        csr sprase matrix
    """
    # Calculate the number of non-zero elements based on sparsity
    total_elements = n_rows * n_cols
    n_nonzero = round(total_elements * (1.0 - sparsity))
    # Randomly generate `data` (non-zero values)
    data = np.random.rand(n_nonzero) * max_data

    # Randomly generate `indices` (column indices for non-zero elements)
    indices = np.random.choice(n_cols, size=n_nonzero)

    # Randomly distribute non-zero values across rows
    indptr = np.zeros(n_rows + 1, dtype=int)

    # Ensure each row can have non-zero elements, distribute them randomly
    row_distribution = np.random.choice(range(n_rows), size=n_nonzero)

    for row in row_distribution:
        indptr[row + 1] += 1

    # Cumulative sum to get the row pointers
    indptr = np.cumsum(indptr)
    sparse_matrix = csr_matrix((data.astype(np.float32), indices, indptr), shape=(n_rows, n_cols))
    return sparse_matrix


def _create_anndata(fn, n_rows, n_cols, sparsity):
    X = generate_random_csr(n_rows, n_cols, sparsity)
    adata = ad.AnnData(X=X)
    adata.var = pd.DataFrame(
        {
            "gene_name": [f"gene_{i}" for i in range(n_cols)],  # Example variable names
            "is_highly_variable": [False] * n_cols,  # Example metadata
        },
        index=[f"gene_{i}" for i in range(n_cols)],
    )  # Ensure index matches column labels in X

    # Save the synthetic AnnData to a file
    adata.write(fn)


def _get_disk_size(directory):
    """Size of directory on disk."""
    result = subprocess.run(["du", "-sb", directory], stdout=subprocess.PIPE, text=True)
    size_in_bytes = int(result.stdout.split()[0])
    return size_in_bytes


row_sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

genes = [100, 1_000, 10_000]
sparsity_vals = [0.7, 0.8, 0.9, 0.95, 0.99]
if not os.path.exists("samples"):
    os.makedirs("samples")

for r in row_sizes:
    for g in genes:
        for s in sparsity_vals:
            fn = f"samples/sample_{r}_{g}_{s}.h5ad"
            if not os.path.isfile(fn):
                try:
                    _create_anndata(fn, r, g, s)
                    print(r, g, s, _get_disk_size(fn) / (1_024**2))
                except Exception as e:
                    # Print the error message
                    print(f"An error occurred: {e}")
