# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""End-to-end optimized vLLM benchmark and proof runner for Evo2."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bionemo.evo2.vllm.benchmark import (
    BenchmarkSample,
    WorkloadManifest,
    benchmark_sample_from_vllm_outputs,
    sampling_params_kwargs,
    summarize_vllm_outputs,
)


_SEED_ROUND_STRIDE = 1_000_003


class CUDAGraphProofRecorder:
    """Persist scheduler CUDA-graph observations without periodic logger resets."""

    def __init__(self) -> None:
        """Create an empty phase-aware observation stream."""
        self._phase = "unlabeled"
        self.observations: list[dict[str, Any]] = []

    def start_phase(self, phase: str) -> None:
        """Label subsequent scheduler observations."""
        if not phase:
            raise ValueError("CUDA graph proof phase cannot be empty")
        self._phase = phase

    def record(
        self,
        scheduler_stats: Any | None,
        iteration_stats: Any | None,
        mm_cache_stats: Any | None = None,
        engine_idx: int | None = None,
    ) -> None:
        """Record one scheduler dispatch when CUDA graph metrics are present."""
        del iteration_stats, mm_cache_stats
        if scheduler_stats is None or scheduler_stats.cudagraph_stats is None:
            return
        stats = scheduler_stats.cudagraph_stats
        self.observations.append(
            {
                "phase": self._phase,
                "engine_index": 0 if engine_idx is None else engine_idx,
                "num_unpadded_tokens": int(stats.num_unpadded_tokens),
                "num_padded_tokens": int(stats.num_padded_tokens),
                "num_paddings": int(stats.num_paddings),
                "runtime_mode": str(stats.runtime_mode),
            }
        )

    def log(self) -> None:
        """Retain observations when vLLM requests a periodic log flush."""

    def log_engine_initialized(self) -> None:
        """Satisfy the vLLM stat logger protocol."""

    def record_sleep_state(self, sleep: int = 0, level: int = 0) -> None:
        """Satisfy the vLLM stat logger protocol."""
        del sleep, level


def validate_full_decode_proof(
    observations: list[dict[str, Any]],
    *,
    phase: str,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    """Require PIECEWISE/FULL execution and exact-batch FULL steady decode."""
    phase_observations = [item for item in observations if item["phase"] == phase]
    if not phase_observations:
        raise AssertionError(f"no CUDA graph observations were recorded for {phase}")
    if any(item["runtime_mode"].endswith("NONE") for item in phase_observations):
        raise AssertionError(f"{phase} used forbidden CUDAGraphMode.NONE fallback execution")

    full_decode = [
        item
        for item in phase_observations
        if item["runtime_mode"].endswith("FULL") and item["num_unpadded_tokens"] == batch_size
    ]
    required_replays = max_new_tokens - 1
    if len(full_decode) < required_replays:
        raise AssertionError(
            f"{phase} recorded {len(full_decode)} exact-batch FULL decodes; "
            f"at least {required_replays} are required"
        )
    if any(
        item["num_padded_tokens"] != batch_size or item["num_paddings"] != 0
        for item in full_decode
    ):
        raise AssertionError(f"{phase} used semantic or scheduler padding during FULL decode")


def request_seed(
    base_seed: int,
    *,
    generation_round: int,
    global_request_index: int,
) -> int:
    """Return a topology-invariant seed unique to one request and generation round."""
    if base_seed < 0 or generation_round < 0 or global_request_index < 0:
        raise ValueError("seed coordinates must be nonnegative")
    seed = base_seed + generation_round * _SEED_ROUND_STRIDE + global_request_index
    if seed >= 2**63:
        raise ValueError("derived request seed exceeds signed int64")
    return seed


def build_request_sampling_params(
    manifest: WorkloadManifest,
    *,
    sampling_params_factory: Callable[..., Any],
    generation_round: int,
    global_request_offset: int,
) -> list[Any]:
    """Build exact-length per-request sampling params with stable global seeds."""
    common_kwargs = sampling_params_kwargs(manifest)
    return [
        sampling_params_factory(
            **common_kwargs,
            seed=request_seed(
                manifest.seed,
                generation_round=generation_round,
                global_request_index=global_request_offset + local_index,
            ),
        )
        for local_index in range(len(manifest.requests))
    ]


@dataclass(frozen=True)
class GenerationPhaseResult:
    """One timed generation phase plus its unreset CUDA graph observations."""

    phase: str
    sample: BenchmarkSample
    observations: tuple[dict[str, Any], ...]
    output_summaries: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe phase record."""
        return {
            "phase": self.phase,
            "sample": self.sample.to_dict(),
            "cudagraph_observations": list(self.observations),
            "outputs": list(self.output_summaries),
        }


def run_generation_phase(
    *,
    llm: Any,
    manifest: WorkloadManifest,
    sampling_params: list[Any],
    phase: str,
    sample_index: int,
    recorder: CUDAGraphProofRecorder,
    memory_monitor_factory: Callable[[], PeakMemoryMonitor],
    clock: Callable[[], float] = time.perf_counter,
    barrier: Any | None = None,
) -> GenerationPhaseResult:
    """Time one complete offline vLLM batch with optional DP synchronization."""
    if len(sampling_params) != len(manifest.requests):
        raise ValueError("sampling params must align with every request")
    prompts = [
        {"prompt_token_ids": list(request.prompt_token_ids)}
        for request in manifest.requests
    ]
    recorder.start_phase(phase)
    observation_start = len(recorder.observations)
    with memory_monitor_factory() as monitor:
        if barrier is not None:
            barrier.wait()
        begin = clock()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        if barrier is not None:
            barrier.wait()
        generation_s = clock() - begin

    output_summaries = summarize_vllm_outputs(manifest, outputs)
    sample = benchmark_sample_from_vllm_outputs(
        manifest,
        outputs,
        sample_index=sample_index,
        generation_s=generation_s,
        peak_device_memory_bytes=monitor.peak_device_memory_bytes,
        validated_summaries=output_summaries,
    )
    return GenerationPhaseResult(
        phase=phase,
        sample=sample,
        observations=tuple(recorder.observations[observation_start:]),
        output_summaries=output_summaries,
    )


class PeakMemoryMonitor:
    """Poll low-overhead per-device used memory and retain phase-local peaks."""

    def __init__(
        self,
        read_device_memory_bytes: Callable[[], tuple[int, ...]],
        *,
        interval_s: float = 0.02,
    ) -> None:
        """Configure a monitor around one stable per-device memory reader."""
        if interval_s <= 0:
            raise ValueError("memory polling interval must be positive")
        self._read = read_device_memory_bytes
        self._interval_s = interval_s
        self._peaks: tuple[int, ...] = ()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    @property
    def peak_device_memory_bytes(self) -> tuple[int, ...]:
        """Return the maximum used bytes observed for every device."""
        with self._lock:
            return self._peaks

    def sample_now(self) -> None:
        """Read and merge one synchronous device-memory sample."""
        values = tuple(int(value) for value in self._read())
        if not values:
            raise RuntimeError("memory reader returned no devices")
        with self._lock:
            if self._peaks and len(values) != len(self._peaks):
                raise RuntimeError("memory reader device count changed during a phase")
            self._peaks = values if not self._peaks else tuple(map(max, self._peaks, values))

    def _poll(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self.sample_now()
            except BaseException as error:
                self._error = error
                self._stop_event.set()

    def __enter__(self) -> PeakMemoryMonitor:
        """Start phase-local polling."""
        self.sample_now()
        self._thread = threading.Thread(target=self._poll, name="evo2-nvml-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Stop polling and surface monitor failures."""
        del exc_type, exc_value, traceback
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.sample_now()
        if self._error is not None:
            raise RuntimeError("peak-memory polling failed") from self._error


__all__ = [
    "CUDAGraphProofRecorder",
    "GenerationPhaseResult",
    "PeakMemoryMonitor",
    "build_request_sampling_params",
    "request_seed",
    "run_generation_phase",
    "validate_full_decode_proof",
]
