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


#!/usr/bin/env python3

r"""This script benchmarks the performance of the scDataset wrapper for the AnnLoader when only using the .X attribute of the AnnData object.

Usage:
    python anndata_compare.py \
        --adata-path /path/to/data.h5ad

The results will be saved to scdataset_anndata_X_benchmark_{timestamp}_detailed_breakdown.csv
"""

import argparse
from datetime import datetime

import torch
from comprehensive_benchmarking import (
    create_anndata_dataset_factory,
)

# Optional import for scDataset
from scdataset import scDataset
from torch.utils.data import DataLoader

# Import the benchmarking framework
from bionemo.scspeedtest import benchmark_dataloaders_with_configs


def fetch_transform_adata_X(batch):
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
            fetch_transform=fetch_transform_adata_X,
        )
        prefetch_factor = fetch_factor + 1 if num_workers > 0 else None
        dataset.set_mode("train")
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        return loader

    return factory


def anndataX_benchmarking_example(
    num_epochs=1,
    num_runs=1,
    adata_path=None,
    fetch_factors=None,
    block_sizes=None,
    max_time_seconds=120.0,
    warmup_time_seconds=30.0,
):
    """Comprehensive benchmarking example demonstrating various dataloader configurations.

    Args:
        num_epochs: Number of epochs to run for each configuration
        num_runs: Number of runs to perform for each configuration
        adata_path: Path to the AnnData file (.h5ad)
        fetch_factors: List of fetch factors to test (default: [1])
        block_sizes: List of block sizes to test (default: [1, 2, 4, 8, 16, 32, 64])
        max_time_seconds: Maximum time to run each configuration (default: 120.0)
        warmup_time_seconds: Time to warmup before benchmarking (default: 30.0)
    """
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARKING EXAMPLE")
    print("=" * 80)
    print()

    print(f"Using AnnData path: {adata_path}")
    print(f"Fetch factors: {fetch_factors}")
    print(f"Block sizes: {block_sizes}")
    print()

    # Create timestamped prefix for all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"All results will be saved to: scdataset_anndata_X_benchmark_{timestamp}_detailed_breakdown.csv")
    print()

    print(f"Benchmarking {num_runs} run(s) each")
    print()
    # =============================================================================
    # Part1: AnnData Dataset with .X backing with ScDataset Configurations
    # =============================================================================
    anndata_configurations = []
    for fetch_factor in fetch_factors:
        for block_size in block_sizes:
            anndata_configurations.append(
                {
                    "name": f"ScDataset_AnnData_X_{block_size}_block_size_{fetch_factor}_fetch_factor",
                    "dataloader_factory": create_scdataset_annloader_factory(
                        batch_size=64, shuffle=True, block_size=block_size, fetch_factor=fetch_factor, num_workers=0
                    ),
                    "num_epochs": num_epochs,
                    "max_time_seconds": max_time_seconds,
                    "warmup_time_seconds": warmup_time_seconds,
                    "data_path": adata_path,
                    "num_runs": num_runs,
                }
            )
    benchmark_dataloaders_with_configs(
        dataloader_configs=anndata_configurations,
        shared_dataset_factory=create_anndata_dataset_factory(adata_path),
        output_prefix=f"scdataset_anndata_X_benchmark_{timestamp}",  # Same file as SCDL results
    )
    print("Benchmarking completed!")
    print(f"All results saved to: scdataset_anndata_X_benchmark_{timestamp}_detailed_breakdown.csv")
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
        default=[1, 2, 4, 8, 16, 32, 64],
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
    anndataX_benchmarking_example(
        num_epochs=args.num_epochs,
        num_runs=args.num_runs,
        adata_path=args.adata_path,
        fetch_factors=args.fetch_factors,
        block_sizes=args.block_sizes,
        max_time_seconds=args.max_time,
        warmup_time_seconds=args.warmup_time,
    )
