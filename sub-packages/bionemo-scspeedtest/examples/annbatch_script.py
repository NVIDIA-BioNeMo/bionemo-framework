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

import argparse
import os
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

import anndata as ad
import zarr
from annbatch import ZarrSparseDataset, create_anndata_collection
from torch.utils.data import DataLoader

from bionemo.scspeedtest import benchmark_dataloaders_with_configs


zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


def _get_default_collection_path(adata_path: Path) -> Path:
    return Path(adata_path).parent / "collection"


def create_on_disk_collection(adata_path: Path, *, output_path: Path | None = None) -> Path:
    """Creates a collection at either the output_path or at the adata_path.parent / "collection" if output_path is None."""
    output_zarr_collection = _get_default_collection_path(adata_path) if output_path is None else output_path

    def load_adata(adata_path):
        adata = ad.experimental.read_lazy(adata_path)
        # TODO: File anndata issue
        import dask.array as da

        adata.uns = {k: v.compute() if isinstance(v, da.Array) else v for k, v in adata.uns.items()}
        return adata

    create_anndata_collection(
        [adata_path],
        output_zarr_collection,
        zarr_sparse_chunk_size=32768 // 2,
        load_adata=load_adata,
        zarr_compressor=None,
    )
    return Path(output_zarr_collection)


def create_adata(collection_path: Path, num_workers: int, *, use_direct_io: bool = False) -> list[ad.AnnData]:
    """Create a list of AnnData objects loaded from collection_path with the specified parameters, loading only X."""
    # Allocate each worker an even number of threads plus a little
    # TODO: How big is the data? Presumably if it is not big enough to fit in memory, we would want O_DIRECT reading to be on.
    # TODO: There are probably faster ways to get a directory size? If you're using this data loader, presumably your data doesn't fit in memory?
    # TODO: Does X always contain the genes of interest? Layers? Raw?
    # TODO: Should num_workers control threading? If scdataset were to wrap zarr, it would still be multithreaded under the processes, so my inclination is "no".
    with zarr.config.set(
        {
            "threading.max_workers": (os.cpu_count() // max(num_workers, 1)) + 1,
            "codec_pipeline.direct_io": use_direct_io,
        }
    ):
        return [ad.AnnData(X=ad.io.sparse_dataset(zarr.open(p)["X"])) for p in collection_path.iterdir()]


def create_dataset_factory(
    collection_path: Path, num_workers: int, *, use_direct_io: bool = False
) -> Callable[[], list[ad.AnnData]]:
    """Generate an `anndata.AnnData` object for use in `annbatch` i.e., zarr v3 on disk, loaded using `anndata.io.sparse_dataset`."""
    return lambda: create_adata(collection_path=collection_path, num_workers=num_workers, use_direct_io=use_direct_io)


def to_annbatch(
    adatas: list[ad.AnnData],
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    block_size: int = 1,
    fetch_factor: int = 2,
    num_workers: int = 0,
) -> Iterable:
    """Generate an `annbatch.ZarrSparseDataset` based on configuration and a list of input `anndata.AnnData` objects backed by zarr v3 on disk."""
    if num_workers > 0:
        ds = ZarrSparseDataset(
            batch_size=batch_size,
            chunk_size=block_size,
            preload_nchunks=((fetch_factor * batch_size) // block_size),
            preload_to_gpu=False,
            shuffle=shuffle,
            to_torch=True,
        ).add_anndatas(adatas)
        loader = DataLoader(
            ds,
            batch_size=None,
            num_workers=num_workers,
            multiprocessing_context="spawn",
        )
        return (v[0] for v in iter(loader))
    ds = ZarrSparseDataset(
        batch_size=batch_size,
        chunk_size=block_size,
        preload_nchunks=((fetch_factor * batch_size) // block_size),
        preload_to_gpu=False,
    ).add_anndatas(adatas)
    # Internally `get_batch_size` in the benchmarking will just take `len(v)` for benchmarking, but we return a tuple (for now) whose first element is the matrix
    return (v[0] for v in iter(ds))


def create_annbatch_from_preloaded_anndata_factory(
    batch_size=64, shuffle=True, block_size=1, fetch_factor=2, num_workers=0
) -> Callable[[list[ad.AnnData]], Iterable]:
    """Factory creator for an `annbatch.ZarrSparseDataset` given a list of anndatas (backed by zarr v3 on-disk) and based on configuration in the arguments to this function."""

    def factory(adatas: list[ad.AnnData]) -> Iterable:
        return to_annbatch(
            adatas,
            batch_size=batch_size,
            shuffle=shuffle,
            block_size=block_size,
            fetch_factor=fetch_factor,
            num_workers=num_workers,
        )

    return factory


def create_annbatch_factory(
    batch_size: int = 64,
    block_size: int = 1,
    shuffle: bool = True,
    collection_path: Path | None = None,
    num_workers: int = 0,
    fetch_factor: int = 1,
    *,
    use_direct_io: bool = False,
) -> Callable[[], Iterable]:
    """Factory creator for on-disk zarr v3 sharded anndata file __and__ a `annbatch.ZarrSparseDataset` based on configuration in the arguments to this function as well as that on-disk."""

    def factory():
        adatas = create_dataset_factory(
            collection_path=collection_path, num_workers=num_workers, use_direct_io=use_direct_io
        )()
        return to_annbatch(
            adatas,
            batch_size=batch_size,
            shuffle=shuffle,
            block_size=block_size,
            fetch_factor=fetch_factor,
            num_workers=num_workers,
        )

    return factory


def comprehensive_benchmarking_example(
    num_epochs: int = 1,
    num_runs: int = 1,
    collection_path: Path | None = None,
    adata_path: Path | None = None,
    fetch_factors: Path | None = None,
    block_sizes: Path | None = None,
    max_time_seconds: float = 120.0,
    warmup_time_seconds: float = 30.0,
    measure_collection_creation_time: bool = False,
):
    """Benchmarking exampe for `annbatch.ZarrSparseDataset`.

    Args:
        num_epochs: Number of epochs to run for each configuration
        num_runs: Number of runs to perform for each configuration
        collection_path: Path to the shuffled AnnData collection (.zarr v3)
        adata_path: Path to the AnnData file (.h5ad)
        fetch_factors: List of fetch factors to test (default: [1])
        block_sizes: List of block sizes to test (default: [1, 2, 4, 8, 16, 32, 64])
        max_time_seconds: Maximum time to run each configuration (default: 120.0)
        warmup_time_seconds: Time to warmup before benchmarking (default: 30.0)
        measure_collection_creation_time: Whether or not to measure the time to make the on-disk collection (default: False)
    """
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARKING EXAMPLE")
    print("=" * 80)
    print()

    print(f"Using AnnData path: {collection_path}")
    print(f"Fetch factors: {fetch_factors}")
    print(f"Block sizes: {block_sizes}")
    print()

    # Create timestamped prefix for all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"All results will be saved to: scdataset_benchmark_{timestamp}_detailed_breakdown.csv")
    print()

    print(f"Benchmarking {num_runs} run(s) each")
    print()
    print("Running annbatch...")
    # TODO: should we make this by default or?
    if collection_path is None and not measure_collection_creation_time:
        if adata_path is None:
            raise ValueError("Cannot create collection from adata_path None.")
        collection_path = create_on_disk_collection(adata_path)
    if measure_collection_creation_time:
        if adata_path is None:
            raise ValueError("Cannot measure collection creation time from adata_path None.")
        if collection_path is None:
            collection_path = _get_default_collection_path(adata_path)
    num_workers = 0  # TODO: why 0? the other scripts seem to have this hardcoded though
    use_direct_io = False  # TODO: grid search this parameter as well
    annbatch_configurations = []
    for fetch_factor in fetch_factors:
        for block_size in block_sizes:
            annbatch_configurations.append(
                {
                    "name": f"annbatch_{block_size}_{fetch_factor}{'_measure_collection_creation' if measure_collection_creation_time else ''}",
                    "dataloader_factory": create_annbatch_from_preloaded_anndata_factory(
                        batch_size=64,
                        shuffle=True,
                        num_workers=num_workers,
                        block_size=block_size,
                        fetch_factor=fetch_factor,
                    ),
                    "num_epochs": num_epochs,
                    "max_time_seconds": max_time_seconds,
                    "warmup_time_seconds": warmup_time_seconds,
                    "data_path": collection_path,
                    "num_runs": 1,
                }
            )

    benchmark_dataloaders_with_configs(
        dataloader_configs=annbatch_configurations,
        shared_dataset_factory=create_dataset_factory(
            collection_path=collection_path, num_workers=num_workers, use_direct_io=use_direct_io
        )
        if not measure_collection_creation_time
        else lambda: create_adata(
            collection_path=create_on_disk_collection(adata_path=adata_path, output_path=collection_path),
            num_workers=num_workers,
            use_direct_io=use_direct_io,
        ),
        output_prefix=f"annbatch_benchmark_{timestamp}",
    )

    print("Benchmarking completed!")
    print(f"All results saved to: annbatch_benchmark_{timestamp}_detailed_breakdown.csv")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioNeMo Benchmarking Framework - annbatch Test")
    parser.add_argument(
        "--collection-path",
        type=Path,
        default=None,
        help="Path to the AnnData zarr v3 collection. Default: %(default)s",
    )
    parser.add_argument(
        "--adata-path",
        type=Path,
        default=None,
        help="Path to the AnnData H5AD file. Default: %(default)s",
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
        default=1.0,
        help="Time to warmup before benchmarking in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--measure-collection-creation-time",
        type=bool,
        default=False,
        help="Whether to benchmark dataset creation time. Default: %(default)s",
    )

    args = parser.parse_args()

    print("BioNeMo Benchmarking Framework - annbatch Test")
    print("=" * 80)
    comprehensive_benchmarking_example(
        num_epochs=args.num_epochs,
        num_runs=args.num_runs,
        collection_path=args.collection_path,
        adata_path=args.adata_path,
        fetch_factors=args.fetch_factors,
        block_sizes=args.block_sizes,
        max_time_seconds=args.max_time,
        warmup_time_seconds=args.warmup_time,
        measure_collection_creation_time=args.measure_collection_creation_time,
    )
