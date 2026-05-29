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

r"""Comprehensive Benchmarking Example.

Usage:
    python comprehensive_benchmarking.py \\
        --adata-path /path/to/data.h5ad \\
        --scdl-path /path/to/scdl/ \\
        --num-epochs 2 \\
        --num-runs 3

Example:
    python comprehensive_benchmarking.py \\
        --adata-path "/data/sample_data.h5ad" \\
        --scdl-path "/data/scdl_data/"

This example demonstrates BOTH approaches:
1. Dataset Reuse (AnnData): Dataset loaded ONCE, multiple configs tested on SAME dataset
2. Dataset Reload (SCDL): Each config loads its own dataset independently

This mixed approach shows:
- When to use dataset reuse (expensive loading, config comparison)
- When to use reload (full isolation, different datasets)
- Flexibility to choose per use case

Key Benefits of Dataset Reuse:
- Faster benchmarking (dataset loaded once, not N times)
- Memory efficient
- Fair comparison (all configs use identical data)
- Separate tracking of dataset vs dataloader instantiation times

Key Benefits of Dataset Reload:
- Full isolation between configs
- Fresh dataset state per config
- Individual dataset loading performance measurement
"""

import os
from datetime import datetime

import anndata
import numpy as np
import torch
from anndata.experimental import AnnCollection, AnnLoader
from torch.utils.data import DataLoader

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch
from bionemo.scspeedtest import benchmark_dataloaders_with_configs, print_comparison


# =============================================================================
# DATASET FACTORY FUNCTIONS (Load data once)
# =============================================================================


def create_anndata_dataset_factory(data_path, backed="r"):
    """Create a dataset factory that loads AnnData once.

    Args:
        data_path: Path to h5ad file or directory containing h5ad files
        backed: Backing mode for AnnData loading (default: "r" for read-only)

    Returns:
        Factory function that loads the dataset once
    """

    def factory():
        print(f"Loading AnnData dataset from: {data_path}")
        if data_path.endswith(".h5ad"):
            h5ad_files = [data_path]
        elif os.path.isdir(data_path) and any(f.endswith(".h5ad") for f in os.listdir(data_path)):
            h5ad_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".h5ad")]
        else:
            raise ValueError("AnnData requires a .h5ad file or directory with .h5ad files")
        dataset = AnnCollection([anndata.read_h5ad(f, backed=backed) for f in h5ad_files])
        print(f"Dataset loaded: {dataset.shape[0]:,} cells x {dataset.shape[1]:,} genes")
        return dataset

    return factory


# =============================================================================
# DATALOADER FACTORY FUNCTIONS (Receive pre-loaded dataset)
# =============================================================================
def custom_collate(batch):
    """Custom collate function for handling batch data.

    Converts batch elements to tensors, handling scipy sparse matrices
    and AnnData view-style wrappers.

    Args:
        batch: Batch of data to collate

    Returns:
        Collated tensor data
    """

    def to_tensor(x):
        if hasattr(x, "toarray"):
            # Handle scipy sparse slices
            return torch.tensor(x.toarray().ravel(), dtype=torch.float32)
        elif hasattr(x, "X"):  # Catch AnnData view-style wrapper
            return torch.tensor(np.asarray(x.X).ravel(), dtype=torch.float32)
        else:
            return torch.tensor(np.asarray(x).ravel(), dtype=torch.float32)

    batch_tensor = [to_tensor(x) for x in batch]
    return torch.stack(batch_tensor)


def create_annloader_factory(batch_size=64, shuffle=True, num_workers=0):
    """Create a dataloader factory that wraps a pre-loaded AnnData dataset.

    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        collate_fn: Function to collate data samples into batches

    Returns:
        Factory function that creates AnnLoader from pre-loaded dataset
    """

    def factory(dataset):
        if num_workers > 0:
            collate_fn = custom_collate
        else:
            collate_fn = None
        return AnnLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )

    return factory


def create_scdl_dataset_and_loader_factory(batch_size=64, shuffle=True, data_path=None, num_workers=0):
    """Create a SCDL dataloader factory that loads dataset each time.

    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        data_path: Path to the memmap data files
        num_workers: Number of worker processes

    Returns:
        Factory function that creates SCDL dataset and DataLoader (reload approach)
    """

    def factory():
        print(f"Loading SCDL dataset from: {data_path}")
        dataset = SingleCellMemMapDataset(data_path)
        print(f"Dataset loaded: {len(dataset):,} samples")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=collate_sparse_matrix_batch,
            num_workers=num_workers,
        )

    return factory


# =============================================================================
# BENCHMARKING EXAMPLES
# =============================================================================


def dataset_reuse_benchmarking_example(num_epochs=1, num_runs=1, adata_path=None, scdl_path=None):
    """Demonstrate dataset reuse functionality.

    This example shows how to:
    1. Load a dataset ONCE
    2. Test multiple dataloader configurations on the SAME dataset
    3. Get separate instantiation times for dataset vs dataloader creation

    Args:
        num_epochs: Number of epochs to run per configuration
        num_runs: Number of runs per configuration for statistical analysis
        adata_path: Path to the AnnData file (.h5ad)
        scdl_path: Path to the SCDL directory
    """
    # Create timestamped prefix for all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"All results will be saved to: comprehensive_benchmark_{timestamp}_detailed_breakdown.csv")
    print()

    print("Mixed Benchmarking Example: Dataset Reuse vs Reload")
    print("=" * 60)
    print(f"Testing {num_runs} run(s) each across different configs")
    print("AnnData: Dataset loaded ONCE (reuse)")
    print("SCDL: Dataset loaded PER CONFIG (reload)")
    print()

    # =============================================================================
    # EXAMPLE 1: AnnData Dataset with Multiple DataLoader Configurations
    # =============================================================================

    print("EXAMPLE 1: AnnData with Multiple DataLoader Configs")
    print("-" * 60)

    anndata_configurations = [
        {
            "name": "AnnLoader_Single_Worker",
            "dataloader_factory": create_annloader_factory(batch_size=64, shuffle=True, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 5.0,
            "warmup_time_seconds": 1.0,
            "data_path": adata_path,
            "num_runs": num_runs,
        },
        {
            "name": "AnnLoader_Small_Batch",
            "dataloader_factory": create_annloader_factory(batch_size=32, shuffle=True, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 5.0,
            "warmup_time_seconds": 1.0,
            "data_path": adata_path,
            "num_runs": num_runs,
        },
        {
            "name": "AnnLoader_Large_Batch",
            "dataloader_factory": create_annloader_factory(batch_size=128, shuffle=True, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 5.0,
            "warmup_time_seconds": 1.0,
            "data_path": adata_path,
            "num_runs": num_runs,
        },
        {
            "name": "AnnLoader_No_Shuffle",
            "dataloader_factory": create_annloader_factory(batch_size=64, shuffle=False, num_workers=0),
            "num_epochs": num_epochs,
            "max_time_seconds": 5.0,
            "warmup_time_seconds": 1.0,
            "data_path": adata_path,
            "num_runs": num_runs,
        },
    ]

    anndata_results = benchmark_dataloaders_with_configs(
        dataloader_configs=anndata_configurations,
        shared_dataset_factory=create_anndata_dataset_factory(adata_path),
        output_prefix=f"comprehensive_benchmark_{timestamp}",
    )

    print()
    print("=" * 60)

    # =============================================================================
    # EXAMPLE 2: SCDL Dataset with Multiple DataLoader Configurations
    # =============================================================================

    print("EXAMPLE 2: SCDL with Multiple DataLoader Configs (Reload Each Time)")
    print("-" * 60)

    scdl_configurations = [
        {
            "name": "SCDL_Batch32_Shuffle",
            "dataloader_factory": create_scdl_dataset_and_loader_factory(
                batch_size=64, shuffle=True, data_path=scdl_path, num_workers=0
            ),
            "num_epochs": num_epochs,
            "max_time_seconds": 1.0,
            "warmup_time_seconds": 0.0,
            "data_path": scdl_path,
            "num_runs": num_runs,
        },
        {
            "name": "SCDL_Batch128_Shuffle",
            "dataloader_factory": create_scdl_dataset_and_loader_factory(
                batch_size=128, shuffle=True, data_path=scdl_path, num_workers=0
            ),
            "num_epochs": num_epochs,
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 0.5,
            "data_path": scdl_path,
            "num_runs": num_runs,
        },
        {
            "name": "SCDL_Batch64_Multi_Worker",
            "dataloader_factory": create_scdl_dataset_and_loader_factory(
                batch_size=64, shuffle=True, data_path=scdl_path, num_workers=2
            ),
            "num_epochs": num_epochs,
            "max_time_seconds": 3.0,
            "warmup_time_seconds": 0.5,
            "data_path": scdl_path,
            "num_runs": num_runs,
        },
    ]

    # Each config loads its own dataset

    scdl_results = benchmark_dataloaders_with_configs(
        dataloader_configs=scdl_configurations,
        shared_dataset_factory=None,  # Each config loads its own dataset
        output_prefix=f"comprehensive_benchmark_{timestamp}",
    )

    print()
    print("=" * 60)

    # =============================================================================
    # ANALYSIS AND SUMMARY
    # =============================================================================

    print("ANALYSIS & COMPARISON")
    print("-" * 60)

    print_comparison(anndata_results + scdl_results)
    print()
    print("COMPREHENSIVE BENCHMARKING COMPLETED!")
    print(f"All results saved to: comprehensive_benchmark_{timestamp}_detailed_breakdown.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BioNeMo Comprehensive Benchmarking Example",
        epilog="""
Examples:
  %(prog)s --adata-path /data/sample_data.h5ad --scdl-path /data/scdl_data/
  %(prog)s --adata-path /data/large.h5ad --scdl-path /data/scdl/ --num-epochs 3 --num-runs 5
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--adata-path", type=str, required=True, help="Path to the AnnData file (.h5ad)")
    parser.add_argument("--scdl-path", type=str, required=True, help="Path to the SCDL directory")
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of epochs to run for each configuration (default: 1)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=1, help="Number of runs to perform for each configuration (default: 1)"
    )

    args = parser.parse_args()

    # Validate paths exist
    if not os.path.exists(args.adata_path):
        print(f"Error: AnnData file not found: {args.adata_path}")
        exit(1)
    if not os.path.exists(args.scdl_path):
        print(f"Error: SCDL directory not found: {args.scdl_path}")
        exit(1)

    print("BioNeMo Benchmarking Framework - Dataset Reuse Example")
    print("=" * 80)
    print(f"AnnData path: {args.adata_path}")
    print(f"SCDL path: {args.scdl_path}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Runs: {args.num_runs}")
    print()

    # Run the actual dataset reuse benchmarking
    dataset_reuse_benchmarking_example(
        num_epochs=args.num_epochs, num_runs=args.num_runs, adata_path=args.adata_path, scdl_path=args.scdl_path
    )
