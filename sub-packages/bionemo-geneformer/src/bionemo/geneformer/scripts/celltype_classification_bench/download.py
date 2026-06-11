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


# NBVAL_CHECK_OUTPUT
import argparse
import random
from contextlib import contextmanager
from pathlib import Path

import cellxgene_census
import numpy as np


@contextmanager
def random_seed(seed: int):
    """Context manager to set the random seed for reproducibility."""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser(description="Download and prepare cell type benchmark dataset")

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/ubuntu/data/20250501-bench/notebook_tutorials/geneformer_celltype_classification/celltype-bench-dataset-input",
        help="Base directory for data downloads",
    )

    parser.add_argument("--census-version", type=str, default="2023-12-15", help="Cellxgene census version to use")

    parser.add_argument(
        "--dataset-id",
        type=str,
        default="8e47ed12-c658-4252-b126-381df8d52a3d",
        help="Dataset ID to download from census. Don't change this unless you have a good reason.",
    )

    parser.add_argument("--random-seed", type=int, default=32, help="Random seed for reproducibility")

    return parser.parse_args()


# Replace signature with actual parameters
def download(census_version: str, dataset_id: str, base_dir: Path):
    """Download cell type dataset from cellxgene census.

    Args:
        census_version: Version of the cellxgene census to use
        dataset_id: ID of the dataset to download
        base_dir: Base directory for data downloads
    """
    # Ensure base_dir exists
    base_dir.mkdir(parents=True, exist_ok=True)

    # Setup paths
    h5ad_outfile = base_dir / "hs-celltype-bench.h5ad"

    # Download data from census
    print(f"Downloading data from census version {census_version}")
    with cellxgene_census.open_soma(census_version=census_version) as census:
        adata = cellxgene_census.get_anndata(
            census,
            "Homo sapiens",
            obs_value_filter=f'dataset_id=="{dataset_id}"',
        )

    # Print unique cell types
    uq_cells = sorted(adata.obs["cell_type"].unique().tolist())
    print(f"Found {len(uq_cells)} unique cell types")

    # Handle subsampling
    selection = list(range(len(adata)))

    print(f"Selected {len(selection)} cells")

    # Subset and save data - Fix: Convert list to numpy array
    adata = adata[np.array(selection)].copy()
    adata.write_h5ad(h5ad_outfile)
    print(f"Saved data to {h5ad_outfile}")


def main():  # noqa: D103
    args = parse_args()
    download(args.census_version, args.dataset_id, Path(args.base_dir))


if __name__ == "__main__":
    main()
