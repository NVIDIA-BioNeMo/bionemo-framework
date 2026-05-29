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

r"""This script benchmarks the performance of the scDataset wrapper for the SCDL and the AnnLoader with different numbers of workers.

Usage:
    python scdataset_multiworker.py \
        --adata-path /path/to/data.h5ad \
        --scdl-path /path/to/scdl/
The results will be saved to scdataset_multiworker_benchmark_{timestamp}_detailed_breakdown.csv
"""

import argparse
from datetime import datetime
from typing import Sequence, Union

import numpy as np
import torch
from comprehensive_benchmarking import (
    create_anndata_dataset_factory,
    create_annloader_factory,
    create_scdl_dataset_and_loader_factory,
    custom_collate,
)

# Optional import for scDataset
from scdataset import scDataset
from torch.utils.data import DataLoader

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch

# Import the benchmarking framework
from bionemo.scspeedtest import benchmark_dataloaders_with_configs


def fetch_transform_adata(batch):
    """Transform batch to AnnData format."""
    return torch.tensor(batch.X.toarray())


def create_scdataset_annloader_factory(batch_size=64, shuffle=True, block_size=1, fetch_factor=2, num_workers=0):
    """Create a dataloader factory that wraps a pre-loaded AnnData dataset.

    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        block_size: Block size for scDataset
        fetch_factor: Fetch factor for scDataset
        num_workers: Number of worker processes

    Returns:
        Factory function that creates AnnLoader from pre-loaded dataset
    """

    def factory(dataset):
        dataset = scDataset(
            data_collection=dataset,
            batch_size=batch_size,
            block_size=block_size,
            fetch_factor=fetch_factor,
            fetch_transform=fetch_transform_adata,
        )
        prefetch_factor = fetch_factor + 1 if num_workers > 0 else None
        dataset.set_mode("train")
        if num_workers > 0:
            collate_fn = custom_collate
        else:
            collate_fn = None
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn,
        )
        return loader

    return factory


def fetch_callback_bionemo(self, idx: Union[int, slice, Sequence[int], np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Fetch callback for bionemo dataset when used with scDataset."""
    if isinstance(idx, int):
        # Single index
        return collate_sparse_matrix_batch([self.__getitem__(idx)]).to_dense()
    elif isinstance(idx, slice):
        # Slice: convert to a list of indices
        indices = list(range(*idx.indices(len(self))))
        batch_tensors = [self.__getitem__(i) for i in indices]
        return collate_sparse_matrix_batch(batch_tensors).to_dense()
    elif isinstance(idx, (list, np.ndarray, torch.Tensor)):
        # Batch indexing
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        batch_tensors = [self.__getitem__(int(i)) for i in idx]
        return collate_sparse_matrix_batch(batch_tensors).to_dense()
    else:
        raise TypeError(f"Unsupported index type: {type(idx)}")


def create_scdl_scdataset_factory(
    batch_size=64, block_size=1, shuffle=True, adata_path=None, data_path=None, num_workers=0, fetch_factor=1
):
    """Create a factory function for SCDL with scDataset wrapper.

    Args:
        batch_size: Number of samples per batch
        block_size: Block size for scDataset
        shuffle: Whether to shuffle the data
        adata_path: Path to the AnnData file (unused but kept for compatibility)
        data_path: Path to the data files
        num_workers: Number of worker processes for data loading
        fetch_factor: Fetch factor for scDataset

    Returns:
        Factory function that creates an SCDL dataloader with scDataset wrapper
    """

    def factory():
        dataset = SingleCellMemMapDataset(data_path)
        wrapped_dataset = scDataset(
            data_collection=dataset,  # Use the created dataset as data_collection
            batch_size=batch_size,
            block_size=block_size,
            fetch_factor=fetch_factor,
            fetch_transform=None,
            **{"fetch_callback": fetch_callback_bionemo},
        )
        wrapped_dataset.set_mode("train")

        prefetch_factor = fetch_factor + 1 if num_workers > 0 else None
        return DataLoader(
            wrapped_dataset,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    return factory


def multiworker_example(
    num_epochs=1,
    num_runs=1,
    adata_path=None,
    scdl_path=None,
    fetch_factors=None,
    block_sizes=None,
    max_time_seconds=120.0,
    warmup_time_seconds=30.0,
    num_workers_list=None,
):
    """Comprehensive benchmarking example demonstrating various dataloader configurations.

    Args:
        num_epochs: Number of epochs to run for each configuration
        num_runs: Number of runs to perform for each configuration
        adata_path: Path to the AnnData file (.h5ad)
        scdl_path: Path to the SCDL directory
        fetch_factors: List of fetch factors to test (default: [1])
        block_sizes: List of block sizes to test (default: [1, 2, 4, 8, 16, 32, 64])
        max_time_seconds: Maximum time to run each configuration (default: 120.0)
        warmup_time_seconds: Time to warmup before benchmarking (default: 30.0)
        num_workers_list: List of number of workers to test for multiprocessing
    """
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARKING EXAMPLE")
    print("=" * 80)
    print()

    print(f"Using AnnData path: {adata_path}")
    print(f"Using SCDL path: {scdl_path}")
    print(f"Fetch factors: {fetch_factors}")
    print(f"Block sizes: {block_sizes}")
    print()

    # Create timestamped prefix for all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"All results will be saved to: scdataset_benchmark_{timestamp}_detailed_breakdown.csv")
    print()

    print(f"Benchmarking {num_runs} run(s) each")
    print()
    # =============================================================================
    # Part 1: SCDL Dataset with Multiple DataLoader Configurations
    # =============================================================================

    # First run SCDL Regular as baseline
    print("Running SCDL Regular baseline...")
    scdl_configurations = []
    for num_workers in num_workers_list:
        scdl_configurations.append(
            {
                "name": f"Baseline SCDL_{num_workers}_workers",
                "dataloader_factory": create_scdl_dataset_and_loader_factory(
                    batch_size=64, shuffle=True, data_path=scdl_path, num_workers=num_workers
                ),
                "num_epochs": num_epochs,
                "max_time_seconds": max_time_seconds,
                "warmup_time_seconds": warmup_time_seconds,
                "data_path": scdl_path,
                "num_runs": num_runs,
                "num_workers": num_workers,
            }
        )
        for fetch_factor in fetch_factors:
            for block_size in block_sizes:
                scdl_configurations.append(
                    {
                        "name": f"ScDataset_{block_size}_block_size_{fetch_factor}_fetch_factor_{num_workers}_workers",
                        "dataloader_factory": create_scdl_scdataset_factory(
                            batch_size=64,
                            shuffle=True,
                            data_path=scdl_path,
                            num_workers=num_workers,
                            block_size=block_size,
                            fetch_factor=fetch_factor,
                        ),
                        "num_epochs": num_epochs,
                        "max_time_seconds": max_time_seconds,
                        "warmup_time_seconds": warmup_time_seconds,
                        "data_path": scdl_path,
                        "num_runs": 1,
                        "num_workers": num_workers,
                    }
                )
    # =============================================================================
    # Part 2: AnnData Dataset with ScDataset Configurations
    # =============================================================================
    anndata_configurations = []
    for num_workers in num_workers_list:
        if num_workers == 32:
            continue
        anndata_configurations.append(
            {
                "name": f"AnnLoader_Baseline_{num_workers}_workers",
                "dataloader_factory": create_annloader_factory(batch_size=64, shuffle=True, num_workers=num_workers),
                "num_epochs": num_epochs,
                "max_time_seconds": max_time_seconds,
                "warmup_time_seconds": warmup_time_seconds,
                "data_path": adata_path,
                "num_runs": num_runs,
                "num_workers": num_workers,
            }
        )
        for fetch_factor in fetch_factors:
            for block_size in block_sizes:
                anndata_configurations.append(
                    {
                        "name": f"ScDataset_AnnData_{block_size}_block_size_{fetch_factor}_fetch_factor_{num_workers}_workers",
                        "dataloader_factory": create_scdataset_annloader_factory(
                            batch_size=64,
                            shuffle=True,
                            block_size=block_size,
                            fetch_factor=fetch_factor,
                            num_workers=num_workers,
                        ),
                        "num_epochs": num_epochs,
                        "max_time_seconds": max_time_seconds,
                        "warmup_time_seconds": warmup_time_seconds,
                        "data_path": adata_path,
                        "num_runs": num_runs,
                        "num_workers": num_workers,
                    }
                )
    benchmark_dataloaders_with_configs(
        dataloader_configs=scdl_configurations,
        shared_dataset_factory=None,  # Each config creates its own dataset
        output_prefix=f"scdataset_multiworker_benchmark_{timestamp}",
    )
    benchmark_dataloaders_with_configs(
        dataloader_configs=anndata_configurations,
        shared_dataset_factory=create_anndata_dataset_factory(adata_path),
        output_prefix=f"scdataset_multiworker_benchmark_{timestamp}",  # Same file as SCDL results
    )

    print("Benchmarking completed!")
    print(f"All results saved to: scdataset_multiworker_benchmark_{timestamp}_detailed_breakdown.csv")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioNeMo Benchmarking Framework - ScDataset Test")
    parser.add_argument(
        "--adata-path",
        type=str,
        default="/home/pbinder/bionemo-framework/tahoe_data",
        help="Path to the AnnData file (.h5ad). Default: %(default)s",
    )
    parser.add_argument(
        "--scdl-path",
        type=str,
        default="/home/pbinder/bionemo-framework/all_tahoe_memmap/",
        help="Path to the SCDL directory. Default: %(default)s",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of epochs to run for each configuration. Default: %(default)s",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs to perform for each configuration. Default: %(default)s",
    )
    parser.add_argument(
        "--fetch-factors",
        nargs="+",
        type=int,
        default=[2],
        help="List of fetch factors to test. Default: %(default)s",
    )
    parser.add_argument(
        "--block-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="List of block sizes to test. Default: %(default)s",
    )
    parser.add_argument(
        "--num-workers",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32, 64],
        help="List of block sizes to test. Default: %(default)s",
    )

    parser.add_argument(
        "--max-time",
        type=float,
        default=120.0,
        help="Maximum time to run each configuration in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--warmup-time",
        type=float,
        default=30.0,
        help="Time to warmup before benchmarking in seconds. Default: %(default)s",
    )

    args = parser.parse_args()

    print("BioNeMo Benchmarking Framework - ScDataset Test")
    print("=" * 80)
    multiworker_example(
        num_epochs=args.num_epochs,
        num_runs=args.num_runs,
        adata_path=args.adata_path,
        scdl_path=args.scdl_path,
        fetch_factors=args.fetch_factors,
        block_sizes=args.block_sizes,
        max_time_seconds=args.max_time,
        warmup_time_seconds=args.warmup_time,
        num_workers_list=args.num_workers,
    )
