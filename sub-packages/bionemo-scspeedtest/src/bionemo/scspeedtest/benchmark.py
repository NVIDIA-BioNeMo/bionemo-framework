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

import gc
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tqdm import tqdm

from bionemo.scspeedtest.common import (
    BenchmarkResult,
    export_benchmark_results,
    get_batch_size,
    get_disk_size,
    measure_peak_memory_full,
)


"""Benchmarking framework for any dataloader.

This module provides a comprehensive framework for benchmarking any dataloader
implementation. It supports both simple direct benchmarking and factory-based
benchmarking, with features like time-based limits, warmup phases, instantiation
measurement, and comprehensive performance metrics.
"""

__all__ = [
    "BenchmarkConfig",
    "benchmark_dataloaders_with_configs",
    "benchmark_single_dataloader",
    "print_comparison",
    "print_results",
]


def _drop_caches():
    try:
        if platform.system() == "Darwin":
            subprocess.call(["sync", "&&", "sudo", "purge"])
        elif platform.system() == "Linux":
            subprocess.call(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
        else:
            raise Exception("Unsupported platform")
    except subprocess.CalledProcessError:
        print("⚠️ Warning: failed to drop caches — are you running with sudo?")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking.

    This dataclass contains all the configuration parameters needed
    to run a benchmark. It supports both time-based and batch-based
    limits, as well as warmup phases.

    Attributes:
        name: Name of the benchmark
        num_epochs: Number of epochs to run
        max_batches: Maximum number of batches to process (None for all)
        max_time_seconds: Maximum time to run benchmark (None for no limit)
        warmup_batches: Number of warmup batches
        warmup_time_seconds: Time to warmup in seconds (overrides warmup_batches if set)
        data_path: Path to data files (for disk size measurement)
    """

    name: str = "UnnamedBenchmark"
    num_epochs: int = 1
    max_batches: Optional[int] = None
    max_time_seconds: Optional[float] = None
    warmup_batches: Optional[int] = None
    warmup_time_seconds: Optional[float] = None
    data_path: Optional[Union[str, Path]] = None
    shuffle: bool = True
    num_runs: int = 1


def run_benchmark(
    dataloader: Any,
    config: BenchmarkConfig,
    run_name: Optional[str] = None,
    **instantiation_kwargs,
) -> BenchmarkResult:
    """Run the actual benchmark and collect metrics.

    Args:
        dataloader: The dataloader to benchmark
        config: Configuration for the benchmark run
        run_name: Optional name for this run
        **instantiation_kwargs: Instantiation metrics (dataset_instantiation_time_seconds,
                               dataloader_instantiation_time_seconds, peak_memory_during_instantiation_mb,
                               memory_before_instantiation_mb, memory_after_instantiation_mb)

    Returns:
        BenchmarkResult containing all collected data and calculated metrics
    """
    # Use measure_peak_memory_full to get memory info during benchmark
    gc.collect()

    def benchmark_iteration_single_epoch(epoch_num, do_warmup):
        """Run a single epoch of benchmarking, with optional warmup."""
        gc.collect()

        update_interval = 10
        epoch_samples = 0
        epoch_batches = 0
        warmup_samples = 0
        warmup_batches = 0
        warmup_time = 0.0
        elapsed = 0.0
        start_time = None

        pbar = tqdm(desc=f"{config.name} - Epoch {epoch_num + 1}/{config.num_epochs}")
        warm_up_start = time.perf_counter()
        if not do_warmup or not config.warmup_time_seconds:
            config.warmup_time_seconds = 0
        warm_up_end = warm_up_start + config.warmup_time_seconds
        is_warming_up = True

        for num, batch in enumerate(dataloader):
            batch_size = get_batch_size(batch)

            current_time = time.perf_counter()

            if is_warming_up:
                # We're in warm-up period - count samples and batches
                warmup_samples += batch_size
                warmup_batches += 1

                if current_time >= warm_up_end:
                    # Warm-up complete and start the actual timing
                    warmup_time = current_time - warm_up_start

                    print(f"Warmup completed: {warmup_samples:,} samples, {warmup_batches:,} batches")

                    is_warming_up = False
                    start_time = time.perf_counter()
                    end_time = start_time + config.max_time_seconds if config.max_time_seconds is not None else None
                    pbar.set_description(f"{config.name} - Epoch {epoch_num + 1} (warmup complete)")
                else:
                    if warmup_batches % update_interval == 0:
                        elapsed_warmup = current_time - warm_up_start
                        current_warmup_speed = warmup_samples / elapsed_warmup if elapsed_warmup > 0 else 0
                        pbar.set_description(
                            f"{config.name} - Warmup: {elapsed_warmup:.1f}/{config.warmup_time_seconds}s, {current_warmup_speed:.1f} samples/sec"
                        )
                        pbar.update(update_interval)
                continue

            # Now we're past the warm-up period (or no warmup)
            epoch_samples += batch_size
            epoch_batches += 1
            elapsed = current_time - start_time if start_time else 0
            if epoch_batches % update_interval == 0:
                postfix_dict = {
                    "epoch": f"{epoch_num + 1}/{config.num_epochs}",
                    "samples": epoch_samples,
                    "elapsed": f"{elapsed:.2f}s",
                }

                pbar.set_postfix(**postfix_dict, refresh=False)
                pbar.update(update_interval)
            # Check max_batches limit
            if (config.max_batches and epoch_batches >= config.max_batches) or (end_time and current_time >= end_time):
                break

        # If no samples were processed in the epoch, likely because warmup consumed the entire dataset
        if epoch_samples == 0:
            import warnings

            warnings.warn(
                f"Epoch {epoch_num + 1}: No samples processed after warmup. "
                "Warmup may have consumed the entire dataset. "
                "Consider reducing warmup_batches or warmup_time_seconds.",
                RuntimeWarning,
            )

        # Final progress bar update
        if epoch_samples > 0 and elapsed > 0:
            postfix_dict = {
                "epoch": f"{epoch_num + 1}/{config.num_epochs}",
                "samples": epoch_samples,
                "elapsed": f"{elapsed:.2f}s",
                "samples_per_sec": f"{epoch_samples / elapsed:.2f}",
            }
            pbar.set_postfix(**postfix_dict, refresh=False)

        pbar.close()

        return epoch_samples, epoch_batches, elapsed, warmup_samples, warmup_batches, warmup_time

    epoch_results = []
    for epoch in range(config.num_epochs):
        # Create a modified benchmark_iteration for this epoch

        result_tuple = measure_peak_memory_full(
            lambda: benchmark_iteration_single_epoch(epoch, epoch == 0),
            multi_worker=hasattr(dataloader, "num_workers") and dataloader.num_workers > 0,
        )
        (
            (epoch_samples, epoch_batches, elapsed, warmup_samples, warmup_batches, warmup_time),
            _,
            peak,
            avg,
            _,
            _,
            iteration_time,
        ) = result_tuple

        epoch_results.append(
            {
                "epoch": epoch + 1,
                "samples": epoch_samples,
                "batches": epoch_batches,
                "warmup_samples": warmup_samples,
                "warmup_batches": warmup_batches,
                "peak_memory": peak,
                "avg_memory": avg,
                "iteration_time": iteration_time,
                "elapsed": elapsed,
                "warmup_time": warmup_time,
            }
        )

        print(f"Epoch {epoch + 1} completed: {epoch_samples:,} samples, {epoch_batches:,} batches")

    result = BenchmarkResult(
        name=config.name,
        data_path=str(config.data_path) if config.data_path else None,
        max_time_seconds=config.max_time_seconds,
        shuffle=config.shuffle,
        num_workers=getattr(dataloader, "num_workers", -1),
        # Instantiation metrics passed as kwargs
        **instantiation_kwargs,
        epoch_results=epoch_results,
    )
    return result


def benchmark_dataloaders_with_configs(
    dataloader_configs: List[Dict[str, Any]],
    shared_dataset_factory: Optional[Callable[[], Any]] = None,
    output_prefix: str = "consolidated_benchmark_results",
) -> List[BenchmarkResult]:
    """Benchmark multiple dataloader configs with optional shared dataset.

    Each config can have its own dataset_factory, use the shared_dataset_factory, or have none (dataloader creates everything).

    Args:
        dataloader_configs: List of dicts with keys: name, dataloader_factory, dataset_factory (optional), data_path, etc.
        shared_dataset_factory: Optional function that creates a dataset once, then reused across multiple dataloaders
        output_prefix: Prefix for the output CSV filename

    Returns:
        List[BenchmarkResult] for multiple dataloader configs.
    """
    results = []

    # Ensure every config has a dataloader_factory
    for idx, config in enumerate(dataloader_configs):
        if "dataloader_factory" not in config or config["dataloader_factory"] is None:
            raise ValueError(
                f"Config at index {idx} ('{config.get('name', 'UnnamedBenchmark')}') is missing a 'dataloader_factory'."
            )

    _drop_caches()

    # Create shared dataset if factory is provided
    shared_dataset = None
    shared_dataset_baseline = None
    shared_dataset_time = None
    if shared_dataset_factory is not None:
        shared_dataset, shared_dataset_baseline, _, _, _, _, shared_dataset_time = measure_peak_memory_full(
            shared_dataset_factory
        )
    for dl_config in dataloader_configs:
        # Determine which dataset factory to use
        if "dataset_factory" in dl_config:
            # Config has its own dataset factory
            config_dataset_factory = dl_config["dataset_factory"]
        else:
            # No dataset factory - dataloader factory creates everything
            config_dataset_factory = None

        config_dataloader_factory = dl_config["dataloader_factory"]
        if shared_dataset is not None:

            def config_dataloader_from_dataset():
                return config_dataloader_factory(shared_dataset)

            dataloader_factory = config_dataloader_from_dataset
        else:
            dataloader_factory = config_dataloader_factory

        result = benchmark_single_dataloader(
            dataloader_factory=dataloader_factory,
            data_path=dl_config.get("data_path", None),
            name=dl_config.get("name", "UnnamedBenchmark"),
            dataset_factory=config_dataset_factory,
            num_epochs=dl_config.get("num_epochs", 1),
            max_batches=dl_config.get("max_batches", None),
            max_time_seconds=dl_config.get("max_time_seconds", None),
            warmup_batches=dl_config.get("warmup_batches", 5),
            warmup_time_seconds=dl_config.get("warmup_time_seconds", None),
            shuffle=dl_config.get("shuffle", True),
            num_runs=dl_config.get("num_runs", 1),
            dataset_baseline=shared_dataset_baseline,
            output_prefix=output_prefix,
            dataset_instantiation_time=shared_dataset_time,
        )
        # If this hasn't been set, set it to the minimum in the first dataloader
        if not shared_dataset_baseline:
            shared_dataset_baseline = result.memory_before_instantiation_mb

        print_results(result)
        if isinstance(result, list):
            for r in result:
                r.dataset_instantiation_time_seconds = shared_dataset_time
            results.extend(result)
        else:
            result.dataset_instantiation_time_seconds = shared_dataset_time
            results.append(result)
        _drop_caches()
    return results


def benchmark_single_dataloader(
    dataloader_factory: Callable[..., Any],
    data_path: Union[str, Path],
    name: str = "UnnamedBenchmark",
    dataset_factory: Optional[Callable[[], Any]] = None,
    num_epochs: int = 1,
    max_batches: Optional[int] = None,
    max_time_seconds: Optional[float] = None,
    warmup_batches: int = 5,
    warmup_time_seconds: Optional[float] = None,
    shuffle: bool = False,
    num_runs: int = 1,
    dataset_baseline: Optional[float] = None,
    output_prefix: str = "consolidated_benchmark_results",
    dataset_instantiation_time: Optional[float] = None,
) -> Union[BenchmarkResult, List[BenchmarkResult]]:
    """Benchmark a single dataloader with optional separate dataset factory.

    Args:
        dataloader_factory: Factory function that creates a dataloader. If dataset_factory is provided,
                           this should accept a dataset parameter. Otherwise, it should create everything internally.
        data_path: Path to the data file
        name: Name of the benchmark
        dataset_factory: Optional factory function that creates the dataset separately
        num_epochs: Number of epochs to run
        max_batches: Maximum number of batches per epoch (None for unlimited)
        max_time_seconds: Maximum time to run in seconds (None for unlimited)
        warmup_batches: Number of batches for warmup
        warmup_time_seconds: Time in seconds for warmup
        shuffle: Whether to shuffle the data
        num_runs: Number of runs to perform
        dataset_baseline: Optional baseline memory usage for the dataset (for dataset reuse with multiple dataloaders)
        output_prefix: Prefix for the output CSV filename
        dataset_instantiation_time: Optional time taken to instantiate the datasets

    Returns:
        Single BenchmarkResult for num_runs=1, or List[BenchmarkResult] for multiple runs
    """
    if dataset_factory is not None:
        # Separate dataset and dataloader creation
        dataset, dataset_baseline_measured, dataset_peak, _, _, dataset_final, dataset_time = measure_peak_memory_full(
            dataset_factory
        )

        def dataloader_from_dataset():
            return dataloader_factory(dataset)

        dataloader, dl_baseline, dl_peak, _, _, dl_final, dl_time = measure_peak_memory_full(dataloader_from_dataset)

        instantiation_metrics = {
            "peak_memory_during_instantiation_mb": max(dl_peak, dataset_peak),
            "memory_after_instantiation_mb": dl_final,
            "memory_before_instantiation_mb": dataset_baseline_measured,
            "dataset_instantiation_time_seconds": dataset_time,
            "dataloader_instantiation_time_seconds": dl_time,
        }

    else:
        # Dataloader factory creates everything internally
        dataloader, dataloader_baseline_measured, peak, _, _, final_mib, setup_time = measure_peak_memory_full(
            dataloader_factory
        )
        instantiation_metrics = {
            "peak_memory_during_instantiation_mb": peak,
            "memory_after_instantiation_mb": final_mib,
            "memory_before_instantiation_mb": dataset_baseline
            if dataset_baseline is not None
            else dataloader_baseline_measured,
            "dataset_instantiation_time_seconds": dataset_instantiation_time
            if dataset_instantiation_time is not None
            else 0,  # Combined time when no separate dataset factory
            "dataloader_instantiation_time_seconds": setup_time,
        }
    disk_size_mb = get_disk_size(data_path)

    results = []
    for run_idx in range(num_runs):
        # For single run, use the provided dataloader; for multiple runs, re-instantiate as needed
        if run_idx == 0:
            current_dataloader = dataloader
            run_name_str = name if num_runs == 1 else f"{name}_run_{run_idx + 1}"
        else:
            if dataset_factory is not None:

                def dataloader_from_dataset():
                    return dataloader_factory(dataset)

                current_dataloader = dataloader_from_dataset()
            else:
                current_dataloader = dataloader_factory()
            run_name_str = f"{name}_run_{run_idx + 1}"

        run_config = BenchmarkConfig(
            name=run_name_str,
            num_epochs=num_epochs,
            max_batches=max_batches,
            max_time_seconds=max_time_seconds,
            warmup_batches=warmup_batches,
            warmup_time_seconds=warmup_time_seconds,
            data_path=data_path,
            shuffle=shuffle,
        )
        run_result = run_benchmark(current_dataloader, run_config, run_name_str, **instantiation_metrics)
        del current_dataloader
        gc.collect()
        run_result.disk_size_mb = disk_size_mb
        results.append(run_result)

        export_benchmark_results(run_result, output_prefix=output_prefix)
        _drop_caches()

    if num_runs == 1:
        return results[0]
    else:
        return results


def print_results(result_or_results: Union[BenchmarkResult, List[BenchmarkResult]]) -> None:
    """Print benchmark results in a formatted way. Accepts a single result or a list of results."""
    results = result_or_results if isinstance(result_or_results, list) else [result_or_results]
    for result in results:
        print("=" * 60)
        print(f"Benchmark: {result.name}")
        print(f"Samples/sec: {result.samples_per_second:.2f}")
        print(f"Total samples: {result.total_samples}")
        print(f"Total time: {result.total_time_seconds:.3f}s")
        print(f"Dataset instantiation: {result.dataset_instantiation_time_seconds:.3f}s")
        print(f"Dataloader instantiation: {result.dataloader_instantiation_time_seconds:.3f}s")
        print(f"Peak memory durint iteration: {result.peak_memory_mb:.1f} MB")
        print(f"Peak memory during instantiation: {result.peak_memory_during_instantiation_mb:.1f} MB")
        print(f"Disk size: {result.disk_size_mb:.1f} MB")
        print("=" * 60 + "\n")


def print_comparison(results: List[BenchmarkResult]) -> None:
    """Print comparison of multiple benchmark results."""
    if not results or len(results) < 2:
        return

    print(f"\nComparison ({len(results)} configurations)")

    # Show individual results
    for result in results:
        print(f"\nResult for {result.name}: {result.samples_per_second:.2f} samples/sec")
        print(f"   Memory: {result.peak_memory_mb:.1f} MB")

    # Find best performers
    best_samples_per_sec = max(results, key=lambda r: r.samples_per_second)
    lowest_memory = min(results, key=lambda r: r.peak_memory_mb)

    print("\nBest Performers:")
    print(f"Best speed: {best_samples_per_sec.name} ({best_samples_per_sec.samples_per_second:.2f} samples/sec)")
    print(f"Lowest memory: {lowest_memory.name} ({lowest_memory.peak_memory_mb:.2f} MB)")

    fastest_instantiation = min(
        results,
        key=lambda r: (r.dataset_instantiation_time_seconds or 0) + (r.dataloader_instantiation_time_seconds or 0),
    )
    fastest_time = (fastest_instantiation.dataset_instantiation_time_seconds or 0) + (
        fastest_instantiation.dataloader_instantiation_time_seconds or 0
    )
    print(f"Fastest instantiation: {fastest_instantiation.name} ({fastest_time:.3f} s)")
