# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""End-to-end optimized vLLM benchmark and proof runner for Evo2."""

from __future__ import annotations

import json
import platform
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from bionemo.evo2.vllm.benchmark import (
    BenchmarkSample,
    WorkloadManifest,
    aggregate_samples,
    benchmark_sample_from_vllm_outputs,
    build_parser,
    build_request_waves,
    sampling_params_kwargs,
    summarize_vllm_outputs,
    validate_compilation_proof,
)
from bionemo.evo2.vllm.profile import (
    Evo2VllmProfile,
    resolved_config_snapshot,
    validate_resolved_profile,
)


_SEED_ROUND_STRIDE = 1_000_003


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def runtime_versions() -> dict[str, Any]:
    """Return exact runtime versions needed to reproduce an artifact."""
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm": _package_version("vllm"),
        "triton": _package_version("triton"),
        "transformers": _package_version("transformers"),
        "nemo_rl": _package_version("nemo-rl"),
    }


def prepare_workload(
    manifest: WorkloadManifest,
    *,
    request_count: int | None,
    uniform_prompt_length: int | None,
    request_id_prefix: str,
    max_new_tokens: int | None,
) -> WorkloadManifest:
    """Return an immutable exact-shape workload derived from a pinned manifest."""
    result = manifest
    if uniform_prompt_length is not None:
        result = result.with_uniform_prompt_length(
            uniform_prompt_length,
            request_count=len(result.requests) if request_count is None else request_count,
            request_id_prefix=request_id_prefix,
        )
    elif request_count is not None:
        result = result.with_request_count(request_count, request_id_prefix=request_id_prefix)
    if max_new_tokens is not None:
        result = result.with_max_new_tokens(max_new_tokens)
    return result


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
    """Allow mixed prefill while requiring dense, exact FULL decode replay."""
    phase_observations = [item for item in observations if item["phase"] == phase]
    if not phase_observations:
        raise AssertionError(f"no CUDA graph observations were recorded for {phase}")
    eager_decode = [
        item
        for item in phase_observations
        if item["runtime_mode"].endswith("NONE") and item["num_unpadded_tokens"] <= batch_size
    ]
    if eager_decode:
        raise AssertionError(f"{phase} used forbidden CUDAGraphMode.NONE decode fallback execution")

    full_decode = [item for item in phase_observations if item["runtime_mode"].endswith("FULL")]
    if any(
        item["num_padded_tokens"] != item["num_unpadded_tokens"] or item["num_paddings"] != 0 for item in full_decode
    ):
        raise AssertionError(f"{phase} used semantic or scheduler padding during FULL decode")
    if max_new_tokens > 1 and not any(item["num_unpadded_tokens"] == batch_size for item in full_decode):
        raise AssertionError(f"{phase} did not execute a FULL global batch")

    # Fresh offline requests can be admitted over multiple scheduler iterations.
    # Decode work performed alongside those prefills is PIECEWISE and cannot be
    # separated from prompt tokens in CUDAGraphStat. Long runs therefore allow
    # one global batch of mixed work, then require dense FULL replay thereafter.
    if max_new_tokens >= 32:
        expected_decode_tokens = batch_size * (max_new_tokens - 1)
        full_decode_tokens = sum(item["num_unpadded_tokens"] for item in full_decode)
        minimum_full_decode_tokens = expected_decode_tokens - batch_size
        if full_decode_tokens < minimum_full_decode_tokens:
            raise AssertionError(
                f"{phase} FULL decode coverage was {full_decode_tokens}/{expected_decode_tokens} tokens; "
                f"at least {minimum_full_decode_tokens} are required"
            )
        average_batch_occupancy = full_decode_tokens / len(full_decode)
        minimum_average_occupancy = batch_size * 0.9
        if average_batch_occupancy < minimum_average_occupancy:
            raise AssertionError(
                f"{phase} FULL decode occupancy averaged {average_batch_occupancy:.3f}/{batch_size}; "
                f"at least {minimum_average_occupancy:.3f} is required"
            )


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
    worker_proof: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe phase record."""
        observation_limit = 256
        if len(self.observations) <= observation_limit:
            retained_observations = list(self.observations)
        else:
            retained_observations = [*self.observations[:128], *self.observations[-128:]]
        return {
            "phase": self.phase,
            "sample": self.sample.to_dict(),
            "cudagraph_observation_count": len(self.observations),
            "cudagraph_observations_retained": retained_observations,
            "cudagraph_summary": summarize_cudagraph_observations(self.observations),
            "outputs": list(self.output_summaries),
            "worker_proof": list(self.worker_proof),
        }


def summarize_cudagraph_observations(observations: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Aggregate graph-mode observations without emitting one row per decode token."""
    counts = Counter(
        (
            item["engine_index"],
            item["runtime_mode"],
            item["num_unpadded_tokens"],
            item["num_padded_tokens"],
            item["num_paddings"],
        )
        for item in observations
    )
    return [
        {
            "engine_index": engine_index,
            "runtime_mode": runtime_mode,
            "num_unpadded_tokens": unpadded,
            "num_padded_tokens": padded,
            "num_paddings": paddings,
            "count": count,
        }
        for (engine_index, runtime_mode, unpadded, padded, paddings), count in sorted(counts.items())
    ]


def run_generation_phase(
    *,
    llm: Any,
    manifest: WorkloadManifest,
    sampling_params: list[Any],
    phase: str,
    sample_index: int,
    recorder: CUDAGraphProofRecorder,
    memory_monitor_factory: Callable[[], PeakMemoryMonitor],
    reset_worker_proof: Callable[[], Any] | None = None,
    snapshot_worker_proof: Callable[[], tuple[dict[str, Any], ...]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    barrier: Any | None = None,
) -> GenerationPhaseResult:
    """Time one complete offline vLLM batch with optional DP synchronization."""
    if len(sampling_params) != len(manifest.requests):
        raise ValueError("sampling params must align with every request")
    prompts = [{"prompt_token_ids": list(request.prompt_token_ids)} for request in manifest.requests]
    recorder.start_phase(phase)
    observation_start = len(recorder.observations)
    if reset_worker_proof is not None:
        reset_worker_proof()
    with memory_monitor_factory() as monitor:
        if barrier is not None:
            barrier.wait()
        begin = clock()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        if barrier is not None:
            barrier.wait()
        generation_s = clock() - begin
    worker_proof = () if snapshot_worker_proof is None else snapshot_worker_proof()

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
        worker_proof=worker_proof,
    )


def reset_vllm_worker_proof_state(worker: Any) -> dict[str, int]:
    """Reset phase-local FIR telemetry and CUDA allocator peaks on one vLLM worker."""
    del worker
    import torch

    from bionemo.evo2.vllm.packed_fir import reset_fir_route_stats

    reset_fir_route_stats()
    torch.cuda.reset_peak_memory_stats()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    return {"rank": int(rank), "device": int(torch.cuda.current_device())}


def snapshot_vllm_worker_proof_state(worker: Any) -> dict[str, Any]:
    """Collect route, compile, and CUDA-memory evidence from one vLLM worker."""
    del worker
    import torch

    from bionemo.evo2.vllm.packed_fir import get_fir_route_stats
    from bionemo.evo2.vllm.profile import compilation_counter_snapshot

    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    return {
        "rank": int(rank),
        "device": int(device),
        "device_name": torch.cuda.get_device_name(device),
        "fir_routes": get_fir_route_stats(),
        "compilation": compilation_counter_snapshot(),
        "cuda_memory": {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }


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


def make_nvml_memory_reader() -> Callable[[], tuple[int, ...]]:
    """Create a stable reader for used memory on every visible physical GPU."""
    import pynvml

    pynvml.nvmlInit()
    handles = tuple(pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount()))
    return lambda: tuple(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used) for handle in handles)


def _attach_cudagraph_recorder(llm: Any, recorder: CUDAGraphProofRecorder) -> None:
    manager = llm.llm_engine.logger_manager
    if manager is None:
        raise RuntimeError("vLLM stat logger manager is disabled; CUDA graph proof is unavailable")
    manager.stat_loggers.append(recorder)


def _reset_worker_proof(llm: Any) -> tuple[dict[str, Any], ...]:
    return tuple(llm.collective_rpc("reset_evo2_proof_state"))


def _snapshot_worker_proof(llm: Any) -> tuple[dict[str, Any], ...]:
    return tuple(llm.collective_rpc("snapshot_evo2_proof_state"))


def _phase_specs(warmups: int, repetitions: int) -> tuple[tuple[str, int], ...]:
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be nonnegative and repetitions must be positive")
    return (
        ("cold-generation", 0),
        *((f"warmup-{index}", index + 1) for index in range(warmups)),
        *((f"steady-{index}", warmups + index + 1) for index in range(repetitions)),
    )


def run_tp2_benchmark(args: Any, manifest: WorkloadManifest) -> dict[str, Any]:
    """Run one TP2 Ray engine through cold, warm, and measured exact phases."""
    if args.topology != "tp2":
        raise ValueError("run_tp2_benchmark requires topology=tp2")
    from vllm import LLM, SamplingParams

    profile = Evo2VllmProfile(
        topology="tp2",
        max_model_len=args.max_model_len or manifest.max_total_tokens,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        async_scheduling=args.async_scheduling,
        proof=args.proof,
        max_concurrent_partial_prefills=args.max_concurrent_partial_prefills,
        long_prefill_chunk_tokens=args.long_prefill_chunk_tokens,
        optimization_level=args.optimization_level,
        performance_mode=args.performance_mode,
    )
    engine_kwargs = profile.engine_kwargs(
        model=str(args.checkpoint),
        seed=manifest.seed,
        load_format=args.load_format,
    )
    memory_reader = make_nvml_memory_reader()
    recorder = CUDAGraphProofRecorder()

    init_begin = time.perf_counter()
    with PeakMemoryMonitor(memory_reader) as init_memory:
        llm = LLM(**engine_kwargs)
    engine_init_s = time.perf_counter() - init_begin
    _attach_cudagraph_recorder(llm, recorder)
    resolved = resolved_config_snapshot(llm.llm_engine.vllm_config)
    validate_resolved_profile(profile, resolved)
    initialized_worker_proof = _snapshot_worker_proof(llm)

    phase_results = []
    for sample_index, (phase, round_offset) in enumerate(_phase_specs(args.warmups, args.repetitions)):
        sampling_params = build_request_sampling_params(
            manifest,
            sampling_params_factory=SamplingParams,
            generation_round=args.generation_round + round_offset,
            global_request_offset=0,
        )
        result = run_generation_phase(
            llm=llm,
            manifest=manifest,
            sampling_params=sampling_params,
            phase=phase,
            sample_index=sample_index,
            recorder=recorder,
            memory_monitor_factory=lambda: PeakMemoryMonitor(memory_reader),
            reset_worker_proof=lambda: _reset_worker_proof(llm),
            snapshot_worker_proof=lambda: _snapshot_worker_proof(llm),
        )
        if args.proof:
            validate_full_decode_proof(
                list(result.observations),
                phase=phase,
                batch_size=min(profile.per_engine_batch_size, len(manifest.requests)),
                max_new_tokens=manifest.max_new_tokens,
            )
        phase_results.append(result)

    final_worker_proof = phase_results[-1].worker_proof
    for initialized, final in zip(initialized_worker_proof, final_worker_proof, strict=True):
        validate_compilation_proof(initialized["compilation"], final["compilation"])

    steady_results = [result for result in phase_results if result.phase.startswith("steady-")]
    waves = build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=profile.global_batch_size,
        replica_count=profile.replica_count,
    )
    return {
        "schema_version": 1,
        "backend": "vllm",
        "topology": "tp2",
        "versions": runtime_versions(),
        "checkpoint": str(args.checkpoint),
        "manifest": manifest.to_dict(),
        "manifest_sha256": manifest.sha256,
        "profile": asdict(profile),
        "engine_kwargs": engine_kwargs,
        "resolved_config": resolved,
        "execution_contract": {
            "outer_model": "torch.compile Inductor",
            "prefill": "optimized eager no_compile custom op; packed route proven per worker",
            "decode": "FULL CUDA graph replay required",
            "prefix_caching": False,
            "shared_prefix_state_reuse": False,
            "semantic_padding": False,
        },
        "request_waves": [
            {
                "wave_index": wave.wave_index,
                "start": wave.start,
                "stop": wave.stop,
                "request_count": wave.request_count,
                "shards": [asdict(shard) | {"request_count": shard.request_count} for shard in wave.shards],
            }
            for wave in waves
        ],
        "timing": {
            "engine_init_s": engine_init_s,
            "engine_init_peak_device_memory_bytes": list(init_memory.peak_device_memory_bytes),
        },
        "initialized_worker_proof": list(initialized_worker_proof),
        "phases": [result.to_dict() for result in phase_results],
        "steady_aggregate": aggregate_samples([result.sample for result in steady_results]),
    }


def write_json_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    """Write one durable, deterministic benchmark artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)


def main(argv: list[str] | None = None) -> int:
    """Run the optimized exact-length vLLM benchmark CLI."""
    args = build_parser().parse_args(argv)
    if args.backend != "vllm":
        raise NotImplementedError("the MCore baseline uses its pinned backend adapter")
    source_manifest = WorkloadManifest.from_path(args.manifest)
    manifest = prepare_workload(
        source_manifest,
        request_count=args.request_count,
        uniform_prompt_length=args.uniform_prompt_length,
        request_id_prefix=args.request_id_prefix,
        max_new_tokens=args.max_new_tokens,
    )
    if args.topology != "tp2":
        raise NotImplementedError("DP2 execution is provided by the concurrent replica launcher")
    artifact = run_tp2_benchmark(args, manifest)
    write_json_artifact(args.output, artifact)
    return 0


__all__ = [
    "CUDAGraphProofRecorder",
    "GenerationPhaseResult",
    "PeakMemoryMonitor",
    "build_request_sampling_params",
    "prepare_workload",
    "request_seed",
    "reset_vllm_worker_proof_state",
    "run_generation_phase",
    "runtime_versions",
    "snapshot_vllm_worker_proof_state",
    "summarize_cudagraph_observations",
    "validate_full_decode_proof",
    "write_json_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
