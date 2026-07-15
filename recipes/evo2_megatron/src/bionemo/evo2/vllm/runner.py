# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""End-to-end optimized vLLM benchmark and proof runner for Evo2."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from bionemo.evo2.vllm.benchmark import (
    BenchmarkSample,
    GenerationRecord,
    WorkloadManifest,
    aggregate_samples,
    benchmark_sample_from_vllm_outputs,
    build_parser,
    build_request_waves,
    exact_length_evidence,
    records_from_vllm_outputs,
    sampling_params_kwargs,
    summarize_vllm_outputs,
    validate_compilation_proof,
)
from bionemo.evo2.vllm.profile import (
    Evo2VllmProfile,
    context_length_preflight,
    resolved_config_snapshot,
    validate_resolved_profile,
)


_SEED_ROUND_STRIDE = 1_000_003
_SEED_MODULUS = 2**31
_REQUIRED_GPU_HEADROOM_BYTES = 2 * 1024**3


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


def benchmark_mode_from_args(args: Any) -> str:
    """Resolve one fail-closed preflight, proof, or linked speed invocation."""
    linked_proof = getattr(args, "linked_proof_artifact", None)
    if args.context_preflight_only:
        if args.proof or linked_proof is not None:
            raise ValueError("context preflight cannot enable proof or link a proof artifact")
        return "preflight"
    if args.proof:
        if linked_proof is not None:
            raise ValueError("a proof run cannot link another proof artifact")
        return "proof"
    if linked_proof is None:
        raise ValueError("a low-overhead speed run requires a linked proof artifact")
    return "speed"


def benchmark_instrumentation_contract(mode: str) -> dict[str, bool]:
    """Describe instrumentation that is active inside generation measurements."""
    if mode not in {"proof", "speed"}:
        raise ValueError(f"generation instrumentation requires proof or speed mode, got {mode!r}")
    collect_proof = mode == "proof"
    return {
        "scheduler_callbacks_during_generation": collect_proof,
        "worker_proof_rpcs": collect_proof,
        "prefix_clone_instrumentation": collect_proof,
        "peak_memory_polling_during_generation": collect_proof,
        "post_generation_exact_output_validation": True,
    }


def build_benchmark_contract(
    args: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
) -> dict[str, Any]:
    """Build the optimized engine/workload identity shared by proof and speed lanes."""
    generation_round = int(args.generation_round)
    if generation_round < 0:
        raise ValueError("generation_round must be nonnegative")
    profile_contract = asdict(profile)
    profile_contract.pop("proof")
    return {
        "schema_version": 2,
        "backend": args.backend,
        "topology": args.topology,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "load_format": args.load_format,
        "manifest_sha256": manifest.sha256,
        "profile": profile_contract,
        "seed_stream": {
            "schema_version": 1,
            "base_seed": manifest.seed,
            "generation_round": generation_round,
            "round_stride": _SEED_ROUND_STRIDE,
            "modulus": _SEED_MODULUS,
        },
        "measurement": {
            "warmups": int(args.warmups),
            "repetitions": int(args.repetitions),
        },
    }


def benchmark_contract_sha256(contract: dict[str, Any]) -> str:
    """Return the canonical digest used to link proof and speed artifacts."""
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_linked_proof_artifact(
    path: str | Path,
    *,
    expected_contract: dict[str, Any],
    require_memory_headroom: bool = False,
) -> dict[str, Any]:
    """Require one successful proof artifact with the exact speed-run contract."""
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"linked proof artifact is missing: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("benchmark_mode") != "proof":
        raise AssertionError("linked artifact is not a proof benchmark")
    if artifact.get("invocation", {}).get("exit_status") != 0:
        raise AssertionError("linked proof artifact did not complete successfully")
    if artifact.get("proof_status", {}).get("passed") is not True:
        raise AssertionError("linked proof artifact did not pass its proof gates")
    phases = artifact.get("phases")
    if not isinstance(phases, list) or not phases or any(not isinstance(phase, dict) for phase in phases):
        raise AssertionError("linked proof artifact is missing concrete phase evidence")
    retained_contract = artifact.get("benchmark_contract")
    if not isinstance(retained_contract, dict):
        raise AssertionError("linked proof artifact is missing its benchmark contract")
    retained_sha256 = benchmark_contract_sha256(retained_contract)
    if artifact.get("benchmark_contract_sha256") != retained_sha256:
        raise AssertionError("linked proof artifact benchmark contract digest is invalid")
    expected_sha256 = benchmark_contract_sha256(expected_contract)
    if retained_sha256 != expected_sha256 or retained_contract != expected_contract:
        raise AssertionError("linked proof artifact benchmark contract does not match the speed run")
    recomputed = _validate_linked_proof_evidence(
        artifact,
        artifact_path=artifact_path,
        expected_contract=expected_contract,
        require_memory_headroom=require_memory_headroom,
    )
    memory_headroom = recomputed["gpu_memory_headroom"]
    return {
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256_file(artifact_path),
        "benchmark_contract_sha256": retained_sha256,
        "proof_status": dict(artifact["proof_status"]),
        "gpu_memory_headroom": memory_headroom,
        "validated_evidence": recomputed,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dict(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AssertionError(f"{label} must be a JSON array")
    return value


def _validate_full_decode_summary_values(
    summary: dict[str, Any],
    *,
    phase: str,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    expected_tokens = batch_size * max(0, max_new_tokens - 1)
    minimum_tokens = max(0, expected_tokens - batch_size)
    if summary.get("phase") != phase:
        raise AssertionError("FULL decode summary phase does not match its physical wave")
    if summary.get("batch_size") != batch_size or summary.get("max_new_tokens") != max_new_tokens:
        raise AssertionError("FULL decode summary workload dimensions drifted")
    if summary.get("expected_decode_tokens") != expected_tokens:
        raise AssertionError("FULL decode expected-token count is inconsistent")
    if summary.get("minimum_full_decode_tokens") != minimum_tokens:
        raise AssertionError("FULL decode minimum-token gate is inconsistent")
    observation_count = summary.get("observation_count")
    dispatch_count = summary.get("full_dispatch_count")
    full_tokens = summary.get("full_decode_tokens")
    maximum_full_batch = summary.get("maximum_full_batch")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (observation_count, dispatch_count, full_tokens, maximum_full_batch)
    ):
        raise AssertionError("FULL decode numeric evidence is malformed")
    if observation_count == 0 or (max_new_tokens > 1 and dispatch_count == 0):
        raise AssertionError("FULL decode proof retained no graph replay")
    if dispatch_count > observation_count:
        raise AssertionError("FULL decode dispatch count exceeds scheduler observations")
    if maximum_full_batch > batch_size or full_tokens > dispatch_count * batch_size:
        raise AssertionError("FULL decode occupancy exceeds the exact physical batch")
    if summary.get("eager_decode_dispatch_count") != 0:
        raise AssertionError("FULL decode proof retained eager decode dispatch")
    if summary.get("full_decode_unpadded") is not True:
        raise AssertionError("FULL decode proof retained semantic padding")
    global_batch_hit = maximum_full_batch == batch_size
    if summary.get("global_batch_hit") is not global_batch_hit:
        raise AssertionError("FULL decode global-batch hit is inconsistent")
    if max_new_tokens > 1 and not global_batch_hit:
        raise AssertionError("FULL decode proof did not hit the exact physical batch")
    coverage = full_tokens / expected_tokens if expected_tokens else 1.0
    if not math.isclose(float(summary.get("coverage_fraction", -1.0)), coverage):
        raise AssertionError("FULL decode coverage fraction is inconsistent")
    average_occupancy = full_tokens / dispatch_count if dispatch_count else 0.0
    if not math.isclose(float(summary.get("average_full_batch_occupancy", -1.0)), average_occupancy):
        raise AssertionError("FULL decode occupancy is inconsistent")
    minimum_occupancy = batch_size * 0.9
    if not math.isclose(float(summary.get("minimum_average_occupancy", -1.0)), minimum_occupancy):
        raise AssertionError("FULL decode minimum occupancy threshold is inconsistent")
    occupancy_fraction = average_occupancy / batch_size
    if not math.isclose(float(summary.get("occupancy_fraction", -1.0)), occupancy_fraction):
        raise AssertionError("FULL decode occupancy fraction is inconsistent")
    long_gate = max_new_tokens >= 32
    coverage_passed = not long_gate or full_tokens >= minimum_tokens
    occupancy_passed = not long_gate or average_occupancy >= batch_size * 0.9
    if summary.get("long_run_gates_applied") is not long_gate:
        raise AssertionError("FULL decode long-run gate selection is inconsistent")
    if summary.get("coverage_gate_passed") is not coverage_passed:
        raise AssertionError("FULL decode coverage gate result is inconsistent")
    if summary.get("occupancy_gate_passed") is not occupancy_passed:
        raise AssertionError("FULL decode occupancy gate result is inconsistent")
    if summary.get("passed") is not True or not coverage_passed or not occupancy_passed:
        raise AssertionError("FULL decode proof did not pass recomputed gates")


def _validate_cudagraph_phase_evidence(
    phase: dict[str, Any],
    *,
    maximum_wave_size: int,
) -> int:
    observation_count = phase.get("cudagraph_observation_count")
    if isinstance(observation_count, bool) or not isinstance(observation_count, int) or observation_count <= 0:
        raise AssertionError("proof phase is missing concrete CUDA graph observations")
    retained = _require_list(
        phase.get("cudagraph_observations_retained"),
        label="retained CUDA graph observations",
    )
    expected_retained = min(observation_count, 256)
    if len(retained) != expected_retained or any(not isinstance(item, dict) for item in retained):
        raise AssertionError("retained CUDA graph observation count is inconsistent")
    aggregate = _require_list(phase.get("cudagraph_summary"), label="CUDA graph aggregate")
    if not aggregate or any(not isinstance(item, dict) for item in aggregate):
        raise AssertionError("CUDA graph aggregate is empty or malformed")
    aggregate_counts = {}
    total_count = 0
    full_tokens = 0
    for item in aggregate:
        count = item.get("count")
        unpadded = item.get("num_unpadded_tokens")
        padded = item.get("num_padded_tokens")
        paddings = item.get("num_paddings")
        mode = item.get("runtime_mode")
        engine_index = item.get("engine_index")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (engine_index, unpadded, padded, paddings)
            )
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(mode, str)
        ):
            raise AssertionError("CUDA graph aggregate contains malformed numeric evidence")
        key = (engine_index, mode, unpadded, padded, paddings)
        if key in aggregate_counts:
            raise AssertionError("CUDA graph aggregate contains duplicate buckets")
        aggregate_counts[key] = count
        total_count += count
        if mode.endswith("FULL"):
            if padded != unpadded or paddings != 0:
                raise AssertionError("FULL CUDA graph aggregate contains scheduler padding")
            full_tokens += count * unpadded
        if mode.endswith("NONE") and unpadded <= maximum_wave_size:
            raise AssertionError("CUDA graph aggregate contains eager decode dispatch")
    if total_count != observation_count:
        raise AssertionError("CUDA graph aggregate count does not match retained observation count")
    if observation_count <= 256:
        if summarize_cudagraph_observations(tuple(retained)) != aggregate:
            raise AssertionError("CUDA graph aggregate does not match raw retained observations")
    else:
        retained_counts = Counter(
            (
                item.get("engine_index"),
                item.get("runtime_mode"),
                item.get("num_unpadded_tokens"),
                item.get("num_padded_tokens"),
                item.get("num_paddings"),
            )
            for item in retained
        )
        if any(aggregate_counts.get(key, 0) < count for key, count in retained_counts.items()):
            raise AssertionError("truncated CUDA graph observations do not match their aggregate")
    return full_tokens


def _validate_fir_route_evidence(
    worker_proof: Sequence[dict[str, Any]],
    *,
    manifest: WorkloadManifest,
) -> None:
    prompt_lengths = [len(request.prompt_token_ids) for request in manifest.requests]
    long_equal_prefill = min(prompt_lengths) >= 1_024 and len(set(prompt_lengths)) == 1
    for worker in worker_proof:
        routes = _require_dict(worker.get("fir_routes"), label="worker FIR route evidence")
        fallback_reasons = routes.get("fallback_reasons", {})
        if not isinstance(fallback_reasons, dict):
            raise AssertionError("FIR fallback reasons must be a JSON object")
        forbidden = set(fallback_reasons) - {"short_request"}
        if forbidden:
            raise AssertionError(f"FIR production dispatch used forbidden fallback reasons: {sorted(forbidden)}")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in fallback_reasons.values()
        ):
            raise AssertionError("FIR fallback counters must be positive integers")
        route_names = set(routes) - {"fallback_reasons"}
        if not route_names:
            raise AssertionError("worker retained no production FIR route hits")
        unknown_routes = route_names - {"direct", "equal_length_conv"}
        if unknown_routes:
            raise AssertionError(f"worker retained unknown FIR routes: {sorted(unknown_routes)}")
        for route_name in route_names:
            totals = routes[route_name]
            if not isinstance(totals, dict) or any(
                isinstance(totals.get(field), bool) or not isinstance(totals.get(field), int) or totals[field] <= 0
                for field in ("calls", "requests", "tokens")
            ):
                raise AssertionError(f"FIR route {route_name!r} has malformed counters")
        if "direct" in route_names:
            if sum(fallback_reasons.values()) != routes["direct"]["calls"]:
                raise AssertionError("direct FIR route calls do not match their retained reasons")
        elif fallback_reasons:
            raise AssertionError("FIR fallback reasons were retained without direct route calls")
        if long_equal_prefill and "equal_length_conv" not in route_names:
            raise AssertionError("long equal-length prefill did not hit equal_length_conv")
        if not long_equal_prefill and "equal_length_conv" in route_names:
            raise AssertionError("short or ragged workload unexpectedly hit equal_length_conv")
        if not long_equal_prefill and "direct" not in route_names:
            raise AssertionError("short or ragged FIR workload did not hit the direct production route")
        if manifest.max_new_tokens > 1 and "direct" not in route_names:
            raise AssertionError("autoregressive decode did not hit the direct production FIR route")


def _validate_worker_gpu_bindings(
    workers: Sequence[dict[str, Any]],
    *,
    hardware: dict[str, Any],
    expected_worker_count: int,
) -> None:
    devices = _require_list(hardware.get("devices"), label="GPU hardware devices")
    if len(workers) != expected_worker_count:
        raise AssertionError("worker proof count does not match the physical model topology")
    physical = {}
    physical_by_index = {}
    physical_by_uuid = {}
    for device in devices:
        if not isinstance(device, dict):
            raise AssertionError("GPU hardware device provenance is malformed")
        identity = (device.get("uuid"), device.get("pci_bus_id"))
        if not all(isinstance(value, str) and value for value in identity) or identity in physical:
            raise AssertionError("GPU UUID/PCI provenance must be complete and unique")
        physical[identity] = device
        physical_by_index[str(device.get("index"))] = device
        physical_by_uuid[device.get("uuid")] = device
    ranks = []
    observed = set()
    for worker in workers:
        rank = worker.get("rank")
        identity = (worker.get("device_uuid"), worker.get("pci_bus_id"))
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise AssertionError("worker rank is malformed")
        if identity not in physical or identity in observed:
            raise AssertionError("worker rank is not bound to one unique physical GPU UUID/PCI identity")
        if worker.get("device_name") != physical[identity].get("name"):
            raise AssertionError("worker device name does not match physical GPU provenance")
        visible = worker.get("cuda_visible_devices")
        logical_device = worker.get("logical_device")
        if (
            not isinstance(visible, str)
            or isinstance(logical_device, bool)
            or not isinstance(logical_device, int)
            or logical_device < 0
        ):
            raise AssertionError("worker proof omitted CUDA_VISIBLE_DEVICES")
        selectors = tuple(item.strip() for item in visible.split(","))
        if logical_device >= len(selectors) or not selectors[logical_device]:
            raise AssertionError("worker logical device is not present in CUDA_VISIBLE_DEVICES")
        selector = selectors[logical_device]
        if worker.get("visible_device_selector") != selector:
            raise AssertionError("worker retained an inconsistent visible-device selector")
        selected = physical_by_index.get(selector) if selector.isdecimal() else physical_by_uuid.get(selector)
        if selected is None or (selected.get("uuid"), selected.get("pci_bus_id")) != identity:
            raise AssertionError("worker CUDA_VISIBLE_DEVICES selector does not match its UUID/PCI binding")
        ranks.append(rank)
        observed.add(identity)
    if sorted(ranks) != list(range(expected_worker_count)):
        raise AssertionError("worker ranks are not exact and contiguous")


def _validate_full_output_sidecar(
    metadata: dict[str, Any],
    *,
    artifact_path: Path,
    phase: str,
    manifest: WorkloadManifest,
    expected_executions: Sequence[RequestExecutionRecord],
    output_summaries: Any,
) -> dict[str, int]:
    if metadata.get("schema_version") != 2 or metadata.get("format") != "jsonl":
        raise AssertionError("full-output sidecar schema is unsupported")
    if metadata.get("compression") != "gzip":
        raise AssertionError("full-output sidecar compression is not gzip")
    sidecar = Path(str(metadata.get("path", ""))).expanduser().resolve()
    expected_path = phase_output_artifact_path(artifact_path, phase=phase).resolve()
    if sidecar != expected_path or not sidecar.is_file():
        raise AssertionError("full-output sidecar path does not match its proof namespace")
    if metadata.get("sha256") != _sha256_file(sidecar):
        raise AssertionError("full-output sidecar SHA256 does not match retained bytes")
    if metadata.get("size_bytes") != sidecar.stat().st_size:
        raise AssertionError("full-output sidecar byte count is inconsistent")
    summaries = _require_list(output_summaries, label="phase output summaries")
    if len(expected_executions) != len(manifest.requests) or len(summaries) != len(manifest.requests):
        raise AssertionError("phase outputs do not cover the exact manifest")

    request_count = 0
    generated_count = 0
    seen_execution_uids = set()
    try:
        with gzip.open(sidecar, mode="rt", encoding="utf-8") as handle:
            for request_count, line in enumerate(handle, start=1):
                index = request_count - 1
                if index >= len(manifest.requests):
                    raise AssertionError("full-output sidecar contains extra requests")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise AssertionError("full-output sidecar row is not a JSON object")
                request = manifest.requests[index]
                execution = expected_executions[index]
                if any(row.get(key) != value for key, value in execution.to_dict().items()):
                    raise AssertionError("sidecar execution ownership or seed coordinates drifted")
                execution_uid = row.get("execution_uid")
                if execution_uid in seen_execution_uids:
                    raise AssertionError("full-output sidecar contains a duplicate execution UID")
                seen_execution_uids.add(execution_uid)
                if row.get("prompt_token_ids") != list(request.prompt_token_ids):
                    raise AssertionError("sidecar prompt tokens do not match the manifest")
                output_ids = row.get("output_token_ids")
                logprobs = row.get("chosen_token_logprobs")
                if (
                    not isinstance(output_ids, list)
                    or len(output_ids) != manifest.max_new_tokens
                    or any(isinstance(token, bool) or not isinstance(token, int) for token in output_ids)
                ):
                    raise AssertionError("sidecar output token IDs are malformed or not exact length")
                if (
                    not isinstance(logprobs, list)
                    or len(logprobs) != len(output_ids)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in logprobs
                    )
                ):
                    raise AssertionError("sidecar chosen-token logprobs are malformed or non-finite")
                expected_lengths = exact_length_evidence(
                    prompt_tokens=len(request.prompt_token_ids),
                    generated_tokens=len(output_ids),
                    requested_new_tokens=manifest.max_new_tokens,
                )
                if any(row.get(key) != value for key, value in expected_lengths.items()):
                    raise AssertionError("sidecar requested/observed exact-length evidence drifted")
                if (
                    row.get("finish_reason") != "length"
                    or row.get("stop_reason") is not None
                    or row.get("stopped_on_eos") is not False
                ):
                    raise AssertionError("sidecar request did not finish at the exact length boundary")
                record = GenerationRecord(
                    request_id=request.request_id,
                    prompt_token_ids=request.prompt_token_ids,
                    output_token_ids=tuple(output_ids),
                    output_logprobs=tuple(float(value) for value in logprobs),
                    requested_max_tokens=manifest.max_new_tokens,
                    finish_reason="length",
                    stop_reason=None,
                    stopped_on_eos=False,
                )
                if summaries[index] != record.summary_dict():
                    raise AssertionError("phase output summary does not match its full sidecar row")
                generated_count += len(output_ids)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssertionError("full-output sidecar could not be decoded") from error
    if request_count != len(manifest.requests):
        raise AssertionError("full-output sidecar omitted manifest requests")
    if metadata.get("request_count") != request_count:
        raise AssertionError("full-output sidecar request count is inconsistent")
    if metadata.get("generated_token_count") != generated_count:
        raise AssertionError("full-output sidecar generated-token count is inconsistent")
    return {
        "request_count": request_count,
        "generated_token_count": generated_count,
    }


def _validate_phase_sample(
    phase: dict[str, Any],
    *,
    manifest: WorkloadManifest,
    sample_index: int,
) -> tuple[int, ...]:
    sample = _require_dict(phase.get("sample"), label="phase benchmark sample")
    prompt_tokens = sum(len(request.prompt_token_ids) for request in manifest.requests)
    generated_tokens = len(manifest.requests) * manifest.max_new_tokens
    if (
        sample.get("sample_index") != sample_index
        or sample.get("request_count") != len(manifest.requests)
        or sample.get("prompt_tokens") != prompt_tokens
        or sample.get("generated_tokens") != generated_tokens
        or sample.get("output_lengths") != [manifest.max_new_tokens] * len(manifest.requests)
    ):
        raise AssertionError("phase benchmark sample does not match the exact workload")
    generation_calls = _require_list(phase.get("generation_call_s"), label="generation call timings")
    if (
        not generation_calls
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in generation_calls
        )
        or not math.isclose(float(sample.get("generation_s", -1.0)), sum(generation_calls))
    ):
        raise AssertionError("phase generation timing does not match its physical calls")
    peaks = sample.get("peak_device_memory_bytes")
    if not isinstance(peaks, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in peaks
    ):
        raise AssertionError("phase peak-memory evidence is malformed")
    return tuple(peaks)


def _validate_direct_phase_evidence(
    artifact: dict[str, Any],
    *,
    artifact_path: Path,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    generation_round: int,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    measurement = _require_dict(
        artifact["benchmark_contract"].get("measurement"),
        label="benchmark measurement contract",
    )
    warmups = measurement.get("warmups")
    repetitions = measurement.get("repetitions")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (warmups, repetitions))
        or repetitions == 0
    ):
        raise AssertionError("benchmark measurement counts are malformed")
    expected_names = [
        "cold-generation",
        *(f"warmup-{index}" for index in range(warmups)),
        *(f"steady-{index}" for index in range(repetitions)),
    ]
    phases = _require_list(artifact.get("phases"), label="proof phases")
    if [phase.get("phase") for phase in phases] != expected_names:
        raise AssertionError("proof phases do not match the benchmark measurement contract")
    call_index = generation_round
    memory_peaks = []
    final_workers = []
    for sample_index, phase in enumerate(phases):
        if not isinstance(phase, dict) or phase.get("proof_collected") is not True:
            raise AssertionError("proof phase lacks production proof collection")
        phase_name = expected_names[sample_index]
        waves = build_request_waves(
            request_count=len(manifest.requests),
            global_batch_size=profile.global_wave_size,
            replica_count=1,
        )
        retained_waves = _require_list(phase.get("wave_proofs"), label="physical wave proofs")
        if len(retained_waves) != len(waves):
            raise AssertionError("physical proof wave count does not match the exact workload")
        full_summaries = []
        for wave, retained in zip(waves, retained_waves, strict=True):
            expected_phase = f"{phase_name}.wave-{wave.wave_index:03d}"
            expected_fields = {
                "wave_index": wave.wave_index,
                "start": wave.start,
                "stop": wave.stop,
                "request_count": wave.request_count,
                "call_index": call_index + wave.wave_index,
            }
            if any(retained.get(key) != value for key, value in expected_fields.items()):
                raise AssertionError("physical wave boundaries or call indices drifted")
            full_decode = _require_dict(
                retained.get("full_decode_proof"),
                label="wave FULL decode proof",
            )
            _validate_full_decode_summary_values(
                full_decode,
                phase=expected_phase,
                batch_size=wave.request_count,
                max_new_tokens=manifest.max_new_tokens,
            )
            scheduler = _require_dict(
                retained.get("scheduler_capacity_proof"),
                label="wave scheduler proof",
            )
            if (
                scheduler.get("phase") != expected_phase
                or scheduler.get("global_wave_size") != wave.request_count
                or scheduler.get("engine_request_count") != wave.request_count
                or scheduler.get("max_num_seqs") != profile.resolved_max_num_seqs
            ):
                raise AssertionError("scheduler proof dimensions do not match the physical wave")
            validate_scheduler_capacity_proof(scheduler)
            full_summaries.append(full_decode)
        if phase.get("wave_execution") != wave_execution_summary(retained_waves):
            raise AssertionError("physical wave execution summary is inconsistent")
        phase_full = _require_dict(phase.get("full_decode_proof"), label="phase FULL decode proof")
        expected_tokens = sum(item["expected_decode_tokens"] for item in full_summaries)
        full_tokens = sum(item["full_decode_tokens"] for item in full_summaries)
        if (
            phase_full.get("phase") != phase_name
            or phase_full.get("wave_count") != len(waves)
            or phase_full.get("expected_decode_tokens") != expected_tokens
            or phase_full.get("full_decode_tokens") != full_tokens
            or phase_full.get("waves") != full_summaries
            or phase_full.get("passed") is not True
        ):
            raise AssertionError("phase FULL decode aggregate is inconsistent")
        coverage = full_tokens / expected_tokens if expected_tokens else 1.0
        if not math.isclose(float(phase_full.get("coverage_fraction", -1.0)), coverage):
            raise AssertionError("phase FULL decode aggregate coverage is inconsistent")
        graph_full_tokens = _validate_cudagraph_phase_evidence(
            phase,
            maximum_wave_size=profile.global_wave_size,
        )
        if graph_full_tokens != full_tokens:
            raise AssertionError("raw CUDA graph aggregate does not match wave FULL decode totals")

        executions = build_wave_execution_records(
            manifest,
            global_wave_size=profile.global_wave_size,
            call_index_start=call_index,
        )
        if phase.get("request_executions") != [record.to_dict() for record in executions]:
            raise AssertionError("phase request ownership or seed stream drifted")
        _validate_full_output_sidecar(
            _require_dict(phase.get("full_output_artifact"), label="full-output sidecar metadata"),
            artifact_path=artifact_path,
            phase=phase_name,
            manifest=manifest,
            expected_executions=executions,
            output_summaries=phase.get("outputs"),
        )
        memory_peaks.append(
            _validate_phase_sample(
                phase,
                manifest=manifest,
                sample_index=sample_index,
            )
        )
        workers = _require_list(phase.get("worker_proof"), label="phase worker proof")
        hardware = _require_dict(
            artifact.get("gpu_hardware_provenance"),
            label="GPU hardware provenance",
        )
        _validate_worker_gpu_bindings(
            workers,
            hardware=hardware,
            expected_worker_count=profile.tensor_parallel_size,
        )
        _validate_fir_route_evidence(workers, manifest=manifest)
        if profile.shared_prefix_state_reuse:
            retained_prefix = _require_dict(
                phase.get("shared_prefix_state_reuse"),
                label="shared-prefix state reuse evidence",
            )
            resolved = _require_dict(artifact.get("resolved_config"), label="resolved vLLM config")
            cache = _require_dict(resolved.get("cache"), label="resolved vLLM cache config")
            recomputed_prefix = shared_prefix_state_reuse_evidence(
                manifest,
                cached_tokens=_require_list(
                    retained_prefix.get("cached_tokens_by_request"),
                    label="cached-token prefix evidence",
                ),
                worker_proof=workers,
                expected_worker_clone_counts=tuple(len(manifest.requests) - 1 for _ in workers),
                cache_block_size=int(cache.get("block_size", 0)),
            )
            recomputed_prefix = {
                **recomputed_prefix,
                "phase_prefix_cache_reset": True,
            }
            if phase.get("prefix_cache_reset") is not True or retained_prefix != recomputed_prefix:
                raise AssertionError("shared-prefix physical reuse evidence failed recomputation")
        elif phase.get("shared_prefix_state_reuse") is not None or phase.get("prefix_cache_reset") is not False:
            raise AssertionError("non-prefix proof retained unexpected prefix-clone evidence")
        final_workers = workers
        call_index += len(waves)

    initialized = _require_list(
        artifact.get("initialized_worker_proof"),
        label="initialized worker proof",
    )
    hardware = _require_dict(
        artifact.get("gpu_hardware_provenance"),
        label="GPU hardware provenance",
    )
    _validate_worker_gpu_bindings(
        initialized,
        hardware=hardware,
        expected_worker_count=profile.tensor_parallel_size,
    )
    initialized_by_rank = {worker["rank"]: worker for worker in initialized}
    final_by_rank = {worker["rank"]: worker for worker in final_workers}
    if initialized_by_rank.keys() != final_by_rank.keys():
        raise AssertionError("initialized and final worker ranks differ")
    for rank in sorted(initialized_by_rank):
        initialized_worker = initialized_by_rank[rank]
        final_worker = final_by_rank[rank]
        if (
            initialized_worker.get("device_uuid"),
            initialized_worker.get("pci_bus_id"),
        ) != (
            final_worker.get("device_uuid"),
            final_worker.get("pci_bus_id"),
        ):
            raise AssertionError("worker rank moved to a different physical GPU")
        validate_compilation_proof(
            _require_dict(initialized_worker.get("compilation"), label="initialized compilation proof"),
            _require_dict(final_worker.get("compilation"), label="final compilation proof"),
        )
    return tuple(max(values) for values in zip(*memory_peaks, strict=True)), final_workers


def _expected_dp2_executions(
    manifest: WorkloadManifest,
    *,
    profile: Evo2VllmProfile,
    call_index_start: int,
    global_index_start: int,
) -> tuple[RequestExecutionRecord, ...]:
    records = []
    for wave in build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=profile.global_wave_size,
        replica_count=profile.replica_count,
    ):
        for shard in wave.shards:
            records.extend(
                build_request_execution_records(
                    manifest.request_slice(shard.start, shard.stop),
                    global_request_offset=global_index_start + shard.start,
                    dp_rank=shard.replica_index,
                    dp_size=profile.replica_count,
                    call_index=call_index_start + wave.wave_index,
                )
            )
    return tuple(records)


def _validate_dp2_phase_evidence(
    artifact: dict[str, Any],
    *,
    artifact_path: Path,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    generation_round: int,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    measurement = _require_dict(
        artifact["benchmark_contract"].get("measurement"),
        label="benchmark measurement contract",
    )
    warmups = measurement.get("warmups")
    repetitions = measurement.get("repetitions")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (warmups, repetitions))
        or repetitions == 0
    ):
        raise AssertionError("benchmark measurement counts are malformed")
    expected_names = [
        "cold-generation",
        *(f"warmup-{index}" for index in range(warmups)),
        *(f"steady-{index}" for index in range(repetitions)),
    ]
    phases = _require_list(artifact.get("phases"), label="proof phases")
    if [phase.get("phase") for phase in phases] != expected_names:
        raise AssertionError("DP2 proof phases do not match the measurement contract")

    call_index = generation_round
    global_index = 0
    memory_peaks = []
    final_engines = []
    hardware = _require_dict(
        artifact.get("gpu_hardware_provenance"),
        label="GPU hardware provenance",
    )
    for sample_index, phase in enumerate(phases):
        if not isinstance(phase, dict) or phase.get("proof_collected") is not True:
            raise AssertionError("DP2 proof phase lacks production proof collection")
        phase_name = expected_names[sample_index]
        waves = build_request_waves(
            request_count=len(manifest.requests),
            global_batch_size=profile.global_wave_size,
            replica_count=profile.replica_count,
        )
        retained_waves = _require_list(phase.get("waves"), label="DP2 physical wave proofs")
        if len(retained_waves) != len(waves):
            raise AssertionError("DP2 proof wave count does not match the exact workload")
        phase_workers = []
        for wave, retained in zip(waves, retained_waves, strict=True):
            wave_phase = f"{phase_name}.wave-{wave.wave_index:03d}"
            expected_wave = {
                "wave_index": wave.wave_index,
                "phase": wave_phase,
                "start": wave.start,
                "stop": wave.stop,
                "request_count": wave.request_count,
            }
            if any(retained.get(key) != value for key, value in expected_wave.items()):
                raise AssertionError("DP2 physical wave boundaries drifted")
            engines = _require_list(retained.get("engines"), label="DP2 engine wave proofs")
            if len(engines) != len(wave.shards):
                raise AssertionError("DP2 engine proof count does not match active replicas")
            for shard, engine in zip(wave.shards, engines, strict=True):
                if (
                    engine.get("dp_rank") != shard.replica_index
                    or engine.get("request_count") != shard.request_count
                    or engine.get("phase") != wave_phase
                ):
                    raise AssertionError("DP2 engine ownership does not match its exact shard")
                observations = _require_list(
                    engine.get("cudagraph_observations"),
                    label="DP2 raw CUDA graph observations",
                )
                if engine.get("cudagraph_summary") != summarize_cudagraph_observations(tuple(observations)):
                    raise AssertionError("DP2 CUDA graph aggregate does not match raw observations")
                recomputed_full = full_decode_proof_summary(
                    observations,
                    phase=wave_phase,
                    batch_size=shard.request_count,
                    max_new_tokens=manifest.max_new_tokens,
                )
                if engine.get("full_decode_proof") != recomputed_full:
                    raise AssertionError("DP2 FULL decode proof does not match raw observations")
                _validate_full_decode_summary_values(
                    recomputed_full,
                    phase=wave_phase,
                    batch_size=shard.request_count,
                    max_new_tokens=manifest.max_new_tokens,
                )
                scheduler_observations = _require_list(
                    engine.get("scheduler_observations"),
                    label="DP2 raw scheduler observations",
                )
                recomputed_scheduler = scheduler_capacity_proof_summary(
                    scheduler_observations,
                    phase=wave_phase,
                    global_wave_size=wave.request_count,
                    engine_request_count=shard.request_count,
                    max_num_seqs=profile.resolved_max_num_seqs,
                )
                if engine.get("scheduler_capacity_proof") != recomputed_scheduler:
                    raise AssertionError("DP2 scheduler proof does not match raw observations")
                validate_scheduler_capacity_proof(recomputed_scheduler)
                try:
                    validate_resolved_profile(
                        profile,
                        _require_dict(engine.get("resolved_config"), label="DP2 resolved config"),
                    )
                except (AssertionError, KeyError, TypeError, ValueError) as error:
                    raise AssertionError("DP2 engine resolved config drifted") from error
                workers = _require_list(engine.get("worker_proof"), label="DP2 inner worker proof")
                _validate_worker_gpu_bindings(
                    workers,
                    hardware=hardware,
                    expected_worker_count=profile.tensor_parallel_size,
                )
                _validate_fir_route_evidence(workers, manifest=manifest.request_slice(wave.start, wave.stop))
                phase_workers.extend(workers)
            identities = {
                (worker.get("device_uuid"), worker.get("pci_bus_id")) for worker in phase_workers[-len(wave.shards) :]
            }
            if len(identities) != len(wave.shards):
                raise AssertionError("DP2 replicas are not bound to distinct physical GPUs")
            if profile.shared_prefix_state_reuse:
                retained_prefix = _require_dict(
                    retained.get("shared_prefix_state_reuse"),
                    label="DP2 shared-prefix evidence",
                )
                workers = [
                    worker
                    for engine in engines
                    for worker in _require_list(engine.get("worker_proof"), label="DP2 worker proof")
                ]
                cache = _require_dict(engines[0]["resolved_config"].get("cache"), label="DP2 cache config")
                recomputed_prefix = shared_prefix_state_reuse_evidence(
                    manifest.request_slice(wave.start, wave.stop),
                    cached_tokens=_require_list(
                        retained_prefix.get("cached_tokens_by_request"),
                        label="DP2 cached-token evidence",
                    ),
                    worker_proof=workers,
                    expected_worker_clone_counts=tuple(
                        shard.request_count - int(wave.wave_index == 0) for shard in wave.shards
                    ),
                    cache_block_size=int(cache.get("block_size", 0)),
                    expected_cache_misses=len(wave.shards) if wave.wave_index == 0 else 0,
                )
                recomputed_prefix = {
                    **recomputed_prefix,
                    "phase_prefix_cache_reset_before_first_wave": True,
                }
                if retained_prefix != recomputed_prefix:
                    raise AssertionError("DP2 shared-prefix evidence failed physical recomputation")
            elif retained.get("shared_prefix_state_reuse") is not None:
                raise AssertionError("non-prefix DP2 wave retained unexpected prefix evidence")
            final_engines = engines
        if phase.get("wave_execution") != wave_execution_summary(retained_waves):
            raise AssertionError("DP2 physical wave execution summary is inconsistent")
        if phase.get("generation_call_s") != [wave["generation_s"] for wave in retained_waves]:
            raise AssertionError("DP2 generation calls do not match retained physical waves")
        executions = _expected_dp2_executions(
            manifest,
            profile=profile,
            call_index_start=call_index,
            global_index_start=global_index,
        )
        if phase.get("request_executions") != [record.to_dict() for record in executions]:
            raise AssertionError("DP2 request ownership or rank-local seed streams drifted")
        _validate_full_output_sidecar(
            _require_dict(phase.get("full_output_artifact"), label="DP2 full-output sidecar"),
            artifact_path=artifact_path,
            phase=phase_name,
            manifest=manifest,
            expected_executions=executions,
            output_summaries=phase.get("outputs"),
        )
        memory_peaks.append(_validate_phase_sample(phase, manifest=manifest, sample_index=sample_index))
        if phase.get("prefix_cache_reset") is not profile.shared_prefix_state_reuse:
            raise AssertionError("DP2 phase prefix-cache reset contract drifted")
        call_index += len(waves)
        global_index += len(manifest.requests)

    initialized_engines = _require_list(
        artifact.get("initialized_engine_proofs"),
        label="initialized DP2 engine proofs",
    )
    if len(initialized_engines) != profile.replica_count:
        raise AssertionError("initialized DP2 engine count does not match the topology")
    resolved_configs = _require_list(artifact.get("resolved_configs"), label="DP2 resolved configs")
    if len(resolved_configs) != profile.replica_count:
        raise AssertionError("DP2 resolved-config count does not match the topology")
    final_by_rank = {engine["dp_rank"]: engine for engine in final_engines}
    final_workers = []
    initialized_identities = set()
    for dp_rank, (initialized, resolved) in enumerate(zip(initialized_engines, resolved_configs, strict=True)):
        if initialized.get("resolved_config") != resolved:
            raise AssertionError("DP2 initialized and retained resolved configs differ")
        try:
            validate_resolved_profile(profile, _require_dict(resolved, label="DP2 resolved config"))
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            raise AssertionError("DP2 retained resolved config drifted") from error
        initialized_workers = _require_list(
            initialized.get("worker_proof"),
            label="initialized DP2 inner worker proof",
        )
        final = final_by_rank.get(dp_rank)
        if final is None:
            raise AssertionError("final DP2 wave omitted one replica")
        retained_final_workers = _require_list(final.get("worker_proof"), label="final DP2 worker proof")
        _validate_worker_gpu_bindings(
            initialized_workers,
            hardware=hardware,
            expected_worker_count=profile.tensor_parallel_size,
        )
        initial_identity = (
            initialized_workers[0].get("device_uuid"),
            initialized_workers[0].get("pci_bus_id"),
        )
        final_identity = (
            retained_final_workers[0].get("device_uuid"),
            retained_final_workers[0].get("pci_bus_id"),
        )
        if initial_identity != final_identity or initial_identity in initialized_identities:
            raise AssertionError("DP2 replica physical GPU binding changed or overlaps")
        initialized_identities.add(initial_identity)
        validate_compilation_proof(
            _require_dict(
                initialized_workers[0].get("compilation"),
                label="initialized DP2 compilation proof",
            ),
            _require_dict(
                retained_final_workers[0].get("compilation"),
                label="final DP2 compilation proof",
            ),
        )
        final_workers.extend(retained_final_workers)
    return tuple(max(values) for values in zip(*memory_peaks, strict=True)), final_workers


def _validate_linked_proof_evidence(
    artifact: dict[str, Any],
    *,
    artifact_path: Path,
    expected_contract: dict[str, Any],
    require_memory_headroom: bool,
) -> dict[str, Any]:
    try:
        manifest = WorkloadManifest.from_dict(_require_dict(artifact.get("manifest"), label="proof manifest"))
    except (KeyError, TypeError, ValueError) as error:
        raise AssertionError("proof manifest is malformed") from error
    if (
        artifact.get("manifest_sha256") != manifest.sha256
        or expected_contract.get("manifest_sha256") != manifest.sha256
    ):
        raise AssertionError("proof manifest SHA256 does not match the benchmark contract")
    profile_data = _require_dict(artifact.get("profile"), label="proof profile")
    try:
        profile = Evo2VllmProfile(**profile_data)
    except (TypeError, ValueError) as error:
        raise AssertionError("proof profile is malformed") from error
    if profile.proof is not True:
        raise AssertionError("linked proof profile did not enable proof instrumentation")
    profile_contract = asdict(profile)
    profile_contract.pop("proof")
    if expected_contract.get("profile") != profile_contract:
        raise AssertionError("proof profile does not match the linked benchmark contract")
    seed_stream = _require_dict(expected_contract.get("seed_stream"), label="seed-stream contract")
    expected_seed_stream = {
        "schema_version": 1,
        "base_seed": manifest.seed,
        "generation_round": seed_stream.get("generation_round"),
        "round_stride": _SEED_ROUND_STRIDE,
        "modulus": _SEED_MODULUS,
    }
    if seed_stream != expected_seed_stream:
        raise AssertionError("benchmark seed-stream contract is malformed")
    generation_round = seed_stream["generation_round"]
    if isinstance(generation_round, bool) or not isinstance(generation_round, int) or generation_round < 0:
        raise AssertionError("benchmark generation round is malformed")
    invocation = _require_dict(artifact.get("invocation"), label="proof invocation")
    output_path = Path(str(invocation.get("output_artifact_path", ""))).expanduser().resolve()
    if output_path != artifact_path:
        raise AssertionError("proof invocation output path does not match the linked artifact")
    parsed_args = _require_dict(invocation.get("parsed_args"), label="proof parsed arguments")
    if parsed_args.get("generation_round") != generation_round:
        raise AssertionError("proof parsed generation round does not match the seed-stream contract")
    if artifact.get("topology") != profile.topology or artifact.get("topology") != expected_contract.get("topology"):
        raise AssertionError("proof topology does not match its profile and benchmark contract")
    backend = artifact.get("backend")
    if backend == "vllm" and profile.topology == "tp2":
        try:
            validate_resolved_profile(
                profile,
                _require_dict(artifact.get("resolved_config"), label="resolved vLLM config"),
            )
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            raise AssertionError("resolved vLLM config does not match the proof profile") from error
        phase_peak, final_workers = _validate_direct_phase_evidence(
            artifact,
            artifact_path=artifact_path,
            manifest=manifest,
            profile=profile,
            generation_round=generation_round,
        )
    elif backend == "nemo-rl-vllm" and profile.topology == "dp2":
        phase_peak, final_workers = _validate_dp2_phase_evidence(
            artifact,
            artifact_path=artifact_path,
            manifest=manifest,
            profile=profile,
            generation_round=generation_round,
        )
    else:
        raise AssertionError("linked proof backend/topology schema is unsupported")
    hardware = _require_dict(
        artifact.get("gpu_hardware_provenance"),
        label="GPU hardware provenance",
    )
    expected_runtime = _require_dict(
        expected_contract.get("runtime_attestation"),
        label="benchmark runtime attestation",
    )
    retained_sources = _require_dict(
        artifact.get("source_provenance"),
        label="proof source provenance",
    )
    if backend == "vllm":
        sources = {"bionemo": retained_sources}
    else:
        expected_source_names = set(_require_dict(expected_runtime.get("sources"), label="runtime source identities"))
        if set(retained_sources) != expected_source_names:
            raise AssertionError("DP2 proof source provenance does not match the runtime contract")
        sources = {
            name: _require_dict(source, label=f"proof source provenance {name!r}")
            for name, source in retained_sources.items()
        }
    try:
        recomputed_runtime = runtime_attestation_contract(
            checkpoint=_require_dict(
                artifact.get("checkpoint_provenance"),
                label="proof checkpoint provenance",
            ),
            sources=sources,
            vllm_installation=_require_dict(
                artifact.get("vllm_installation_provenance"),
                label="proof vLLM installation provenance",
            ),
            gpu_hardware=hardware,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise AssertionError("proof runtime attestation could not be recomputed") from error
    if recomputed_runtime != expected_runtime:
        raise AssertionError("proof runtime provenance does not match the linked speed-run contract")
    init_peak = _require_dict(artifact.get("timing"), label="proof timing").get("engine_init_peak_device_memory_bytes")
    if (
        not isinstance(init_peak, list)
        or len(init_peak) != len(phase_peak)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in init_peak)
    ):
        raise AssertionError("engine initialization peak-memory evidence is malformed")
    peak = tuple(max(initialized, phase_value) for initialized, phase_value in zip(init_peak, phase_peak, strict=True))
    try:
        recomputed_memory = gpu_memory_headroom_evidence(
            hardware,
            peak_device_memory_bytes=peak,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise AssertionError("proof GPU memory headroom did not pass recomputation") from error
    if artifact.get("gpu_memory_headroom") != recomputed_memory:
        raise AssertionError("retained GPU memory headroom does not match recomputed peaks")
    if require_memory_headroom and recomputed_memory.get("passed") is not True:
        raise AssertionError("linked proof lacks passed GPU memory headroom")
    proof_status = _require_dict(artifact.get("proof_status"), label="proof status")
    if (
        proof_status.get("phase_count") != len(artifact["phases"])
        or proof_status.get("full_decode_passed") is not True
        or proof_status.get("compilation_stable") is not True
    ):
        raise AssertionError("top-level proof status does not match recomputed phase evidence")
    return {
        "manifest_sha256": manifest.sha256,
        "phase_count": len(artifact["phases"]),
        "final_worker_count": len(final_workers),
        "runtime_attestation": recomputed_runtime,
        "gpu_memory_headroom": recomputed_memory,
        "passed": True,
    }


def _file_records(root: Path, paths: Any) -> list[dict[str, Any]]:
    records = []
    for path in sorted({Path(path).resolve() for path in paths}):
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"provenance path escapes root {root}: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(f"provenance file is missing: {path}")
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def package_installation_provenance(
    package_root: str | Path,
    *,
    distribution_name: str,
    distribution_version: str,
    metadata_paths: Sequence[str | Path] = (),
    require_binary: bool = False,
) -> dict[str, Any]:
    """Hash one installed Python package, including compiled extensions and metadata."""
    root = Path(package_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"installed package root is missing: {root}")

    def is_durable(path: Path) -> bool:
        return "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}

    package_paths = tuple(path for path in root.rglob("*") if path.is_file() and is_durable(path))
    source_suffixes = {".py", ".pyi", ".pyx", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    binary_suffixes = {".so", ".pyd", ".dll", ".dylib", ".cubin", ".fatbin"}
    source_paths = tuple(path for path in package_paths if path.suffix.lower() in source_suffixes)
    binary_paths = tuple(path for path in package_paths if path.suffix.lower() in binary_suffixes)
    if not source_paths:
        raise RuntimeError(f"installed {distribution_name} package contains no source implementation files")
    if require_binary and not binary_paths:
        raise RuntimeError(f"installed {distribution_name} package contains no compiled binary files")

    package_records = _file_records(root, package_paths)
    source_records = _file_records(root, source_paths)
    binary_records = _file_records(root, binary_paths)
    metadata_records = []
    for metadata_path in sorted({Path(path).expanduser().resolve() for path in metadata_paths}):
        if not metadata_path.is_file():
            raise FileNotFoundError(f"installed package metadata is missing: {metadata_path}")
        metadata_records.append(
            {
                "path": str(metadata_path),
                "size_bytes": metadata_path.stat().st_size,
                "sha256": _sha256_file(metadata_path),
            }
        )
    installation_identity = {
        "distribution_name": distribution_name,
        "distribution_version": distribution_version,
        "package_root": str(root),
        "package_files": package_records,
        "metadata_files": metadata_records,
    }
    installation_sha256 = hashlib.sha256(
        json.dumps(installation_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **installation_identity,
        "installation_sha256": installation_sha256,
        "package_file_count": len(package_records),
        "package_bytes": sum(record["size_bytes"] for record in package_records),
        "source_file_count": len(source_records),
        "binary_file_count": len(binary_records),
        "metadata_file_count": len(metadata_records),
        "source_files": source_records,
        "binary_files": binary_records,
    }


def vllm_installation_provenance() -> dict[str, Any]:
    """Discover and hash the exact installed vLLM implementation and binaries."""
    spec = find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("installed vLLM package could not be resolved")
    package_locations = tuple(Path(path).expanduser().resolve() for path in spec.submodule_search_locations)
    if len(package_locations) != 1:
        raise RuntimeError(f"vLLM must resolve to one package root, got {package_locations}")

    installed_distribution = distribution("vllm")
    distribution_files = installed_distribution.files
    if distribution_files is None:
        raise RuntimeError("installed vLLM distribution exposes no file manifest")
    retained_metadata_names = {"INSTALLER", "METADATA", "RECORD", "WHEEL", "direct_url.json"}
    metadata_paths = []
    for entry in distribution_files:
        entry_path = Path(str(entry))
        if not any(part.endswith(".dist-info") for part in entry_path.parts):
            continue
        if entry_path.name in retained_metadata_names:
            located = Path(installed_distribution.locate_file(entry)).expanduser().resolve()
            if located.is_file():
                metadata_paths.append(located)
    if not metadata_paths:
        raise RuntimeError("installed vLLM distribution metadata could not be resolved")
    return package_installation_provenance(
        package_locations[0],
        distribution_name="vllm",
        distribution_version=installed_distribution.version,
        metadata_paths=tuple(metadata_paths),
        require_binary=True,
    )


def runtime_attestation_contract(
    *,
    checkpoint: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    vllm_installation: dict[str, Any],
    gpu_hardware: dict[str, Any],
) -> dict[str, Any]:
    """Reduce full provenance to the immutable identities linked across benchmark lanes."""
    source_contract = {}
    for name, source in sorted(sources.items()):
        if source.get("git_dirty") is not False:
            raise RuntimeError(f"runtime attestation source {name!r} is dirty")
        source_contract[name] = {
            "git_head": source["git_head"],
            "source_tree_sha256": source["source_tree_sha256"],
        }
    devices = gpu_hardware.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ValueError("runtime attestation requires exact GPU hardware provenance")
    return {
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "sources": source_contract,
        "vllm": {
            "distribution_version": vllm_installation["distribution_version"],
            "installation_sha256": vllm_installation["installation_sha256"],
        },
        "gpu": {
            "driver_version": gpu_hardware["driver_version"],
            "cuda_visible_devices": gpu_hardware.get("cuda_visible_devices"),
            "devices": [
                {
                    "index": device["index"],
                    "uuid": device["uuid"],
                    "pci_bus_id": device["pci_bus_id"],
                    "name": device["name"],
                    "total_memory_bytes": device["total_memory_bytes"],
                }
                for device in devices
            ],
        },
    }


def checkpoint_provenance(checkpoint: str | Path) -> dict[str, Any]:
    """Hash the actual indexed checkpoint shards and every durable checkpoint file."""
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {root}")
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    manifest_path = root / "manifest.json"
    for required in (config_path, index_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"checkpoint provenance requires {required}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index must contain a non-empty weight_map")
    shard_paths = []
    for shard_name in sorted(set(weight_map.values())):
        shard_path = (root / str(shard_name)).resolve()
        try:
            shard_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"checkpoint shard escapes checkpoint root: {shard_name}") from error
        shard_paths.append(shard_path)
    shard_records = _file_records(root, shard_paths)
    all_records = _file_records(root, (path for path in root.rglob("*") if path.is_file()))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest_verification = {
        "config": manifest.get("config_sha256") == _sha256_file(config_path),
        "index": manifest.get("index_sha256") == _sha256_file(index_path),
    }
    if not all(digest_verification.values()):
        raise AssertionError(f"checkpoint manifest digest verification failed: {digest_verification}")
    return {
        "path": str(root),
        "checkpoint_sha256": _records_sha256(all_records),
        "file_count": len(all_records),
        "total_file_bytes": sum(item["size_bytes"] for item in all_records),
        "indexed_weight_bytes": sum(item["size_bytes"] for item in shard_records),
        "indexed_weight_shards": shard_records,
        "files": all_records,
        "manifest_digest_verification": digest_verification,
    }


def _git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def source_provenance(
    *,
    repository: str | Path | None = None,
    source_roots: tuple[str | Path, ...] | None = None,
    require_clean: bool = False,
) -> dict[str, Any]:
    """Record git identity plus a content hash of the production vLLM source tree."""
    if repository is None:
        repository = _git_output(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel").strip()
    root = Path(repository).expanduser().resolve()
    roots = (Path(__file__).resolve().parent,) if source_roots is None else tuple(Path(path) for path in source_roots)
    source_paths = []

    def is_durable_source(path: Path) -> bool:
        return "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}

    for source_root in roots:
        resolved = source_root.expanduser().resolve()
        if resolved.is_file():
            if is_durable_source(resolved):
                source_paths.append(resolved)
        elif resolved.is_dir():
            source_paths.extend(path for path in resolved.rglob("*") if path.is_file() and is_durable_source(path))
        else:
            raise FileNotFoundError(f"source provenance root is missing: {resolved}")
    source_records = _file_records(root, source_paths)
    source_tree_sha256 = _records_sha256(source_records)
    git_head = _git_output(root, "rev-parse", "HEAD").strip()
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if require_clean and status:
        raise RuntimeError(f"dirty source repository is not benchmarkable: {root}; status={status.splitlines()}")
    tracked_diff = _git_output(root, "diff", "--binary", "HEAD", "--")
    dirty_payload = json.dumps(
        {
            "status": status,
            "tracked_diff_sha256": hashlib.sha256(tracked_diff.encode()).hexdigest(),
            "source_tree_sha256": source_tree_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "repository": str(root),
        "git_head": git_head,
        "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff.encode()).hexdigest(),
        "dirty_fingerprint_sha256": hashlib.sha256(dirty_payload).hexdigest(),
        "source_tree_sha256": source_tree_sha256,
        "source_file_count": len(source_records),
        "source_files": source_records,
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
    if manifest.prompt_source_path is not None and (request_count is not None or uniform_prompt_length is not None):
        raise ValueError("a frozen prompt source cannot be rewritten with synthetic request IDs or prompt lengths")
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


def load_source_manifest(args: Any) -> WorkloadManifest:
    """Load a base manifest and optionally overlay one hash-pinned prompt JSONL source."""
    manifest = WorkloadManifest.from_path(args.manifest)
    prompt_jsonl = getattr(args, "prompt_jsonl", None)
    prompt_jsonl_sha256 = getattr(args, "prompt_jsonl_sha256", None)
    prompt_tokenizer_json = getattr(args, "prompt_tokenizer_json", None)
    expected_prompt_tokens = getattr(args, "expected_prompt_tokens", None)
    if prompt_jsonl is None:
        if any(value is not None for value in (prompt_jsonl_sha256, prompt_tokenizer_json, expected_prompt_tokens)):
            raise ValueError("prompt JSONL provenance options require --prompt-jsonl")
        return manifest
    if prompt_jsonl_sha256 is None or prompt_tokenizer_json is None:
        raise ValueError("--prompt-jsonl requires --prompt-jsonl-sha256 and --prompt-tokenizer-json")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(prompt_tokenizer_json))
    return manifest.with_prompt_jsonl(
        prompt_jsonl,
        tokenize=lambda prompt: tokenizer.encode(prompt, add_special_tokens=False).ids,
        tokenizer_path=prompt_tokenizer_json,
        expected_sha256=prompt_jsonl_sha256,
        expected_prompt_tokens=expected_prompt_tokens,
    )


class CUDAGraphProofRecorder:
    """Persist scheduler CUDA-graph observations without periodic logger resets."""

    def __init__(self) -> None:
        """Create an empty phase-aware observation stream."""
        self._phase = "unlabeled"
        self.observations: list[dict[str, Any]] = []
        self.scheduler_observations: list[dict[str, Any]] = []

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
        del mm_cache_stats
        engine_index = 0 if engine_idx is None else int(engine_idx)
        if scheduler_stats is not None and iteration_stats is not None:
            prefix_stats = getattr(scheduler_stats, "prefix_cache_stats", None)
            prompt_stats = getattr(iteration_stats, "prompt_token_stats", None)
            if prefix_stats is not None and prompt_stats is not None:
                preempted_queries = int(prefix_stats.preempted_queries)
                preempted_hits = int(prefix_stats.preempted_hits)
                preemption_events = int(iteration_stats.num_preempted_reqs)
                self.scheduler_observations.append(
                    {
                        "phase": self._phase,
                        "engine_index": engine_index,
                        "preemption_events": preemption_events,
                        "recompute_events": preemption_events,
                        "prefix_preempted_requests": int(prefix_stats.preempted_requests),
                        "prefix_preempted_queries": preempted_queries,
                        "prefix_preempted_hits": preempted_hits,
                        "preempted_prompt_recomputed_tokens": preempted_queries - preempted_hits,
                        "prompt_tokens_computed": int(prompt_stats.computed),
                        "prompt_tokens_cached": int(prompt_stats.cached_tokens),
                        "prompt_tokens_total": int(prompt_stats.total),
                        "num_running_requests": int(scheduler_stats.num_running_reqs),
                        "num_waiting_requests": int(scheduler_stats.num_waiting_reqs),
                        "num_skipped_waiting_requests": int(scheduler_stats.num_skipped_waiting_reqs),
                    }
                )

        graph_stats = None if scheduler_stats is None else scheduler_stats.cudagraph_stats
        if graph_stats is None:
            return
        stats = graph_stats
        self.observations.append(
            {
                "phase": self._phase,
                "engine_index": engine_index,
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


def scheduler_capacity_proof_summary(
    observations: Sequence[dict[str, Any]],
    *,
    phase: str,
    global_wave_size: int,
    max_num_seqs: int,
    engine_request_count: int | None = None,
) -> dict[str, Any]:
    """Summarize phase-local scheduler fit without inferring absent telemetry."""
    if not phase:
        raise ValueError("scheduler proof phase cannot be empty")
    submitted_engine_requests = global_wave_size if engine_request_count is None else engine_request_count
    if global_wave_size <= 0 or max_num_seqs <= 0 or submitted_engine_requests <= 0:
        raise ValueError("global wave, engine request, and max_num_seqs counts must be positive")
    phase_observations = [item for item in observations if item.get("phase") == phase]

    def total(field: str) -> int:
        return sum(int(item[field]) for item in phase_observations)

    preemption_events = total("preemption_events")
    recompute_events = total("recompute_events")
    prefix_preempted_requests = total("prefix_preempted_requests")
    preempted_queries = total("prefix_preempted_queries")
    preempted_hits = total("prefix_preempted_hits")
    preempted_recomputed = total("preempted_prompt_recomputed_tokens")
    maximum_running = max((int(item["num_running_requests"]) for item in phase_observations), default=0)
    request_count_within_scheduler_ceiling = submitted_engine_requests <= max_num_seqs
    running_count_within_scheduler_ceiling = maximum_running <= max_num_seqs
    batch_fit_without_preemption = (
        bool(phase_observations)
        and request_count_within_scheduler_ceiling
        and running_count_within_scheduler_ceiling
        and preemption_events == 0
        and recompute_events == 0
        and prefix_preempted_requests == 0
        and preempted_queries == 0
        and preempted_hits == 0
        and preempted_recomputed == 0
    )
    return {
        "phase": phase,
        "global_wave_size": global_wave_size,
        "engine_request_count": submitted_engine_requests,
        "max_num_seqs": max_num_seqs,
        "scheduler_observation_count": len(phase_observations),
        "preemption_events": preemption_events,
        "recompute_events": recompute_events,
        "prefix_preempted_requests": prefix_preempted_requests,
        "prefix_preempted_queries": preempted_queries,
        "prefix_preempted_hits": preempted_hits,
        "preempted_prompt_recomputed_tokens": preempted_recomputed,
        "prompt_tokens_computed": total("prompt_tokens_computed"),
        "prompt_tokens_cached": total("prompt_tokens_cached"),
        "prompt_tokens_total": total("prompt_tokens_total"),
        "maximum_running_requests": maximum_running,
        "maximum_waiting_requests": max((int(item["num_waiting_requests"]) for item in phase_observations), default=0),
        "maximum_skipped_waiting_requests": max(
            (int(item["num_skipped_waiting_requests"]) for item in phase_observations), default=0
        ),
        "request_count_within_scheduler_ceiling": request_count_within_scheduler_ceiling,
        "running_count_within_scheduler_ceiling": running_count_within_scheduler_ceiling,
        "batch_fit_without_preemption": batch_fit_without_preemption,
    }


def validate_scheduler_capacity_proof(proof: dict[str, Any]) -> None:
    """Fail closed unless one submitted wave fits without preemption or recompute."""
    positive_fields = ("global_wave_size", "engine_request_count", "max_num_seqs")
    nonnegative_fields = (
        "scheduler_observation_count",
        "preemption_events",
        "recompute_events",
        "prefix_preempted_requests",
        "prefix_preempted_queries",
        "prefix_preempted_hits",
        "preempted_prompt_recomputed_tokens",
        "prompt_tokens_computed",
        "prompt_tokens_cached",
        "prompt_tokens_total",
        "maximum_running_requests",
        "maximum_waiting_requests",
        "maximum_skipped_waiting_requests",
    )
    if any(
        isinstance(proof.get(field), bool) or not isinstance(proof.get(field), int) or proof[field] <= 0
        for field in positive_fields
    ):
        raise AssertionError("scheduler wave dimensions must be positive integers")
    if any(
        isinstance(proof.get(field), bool) or not isinstance(proof.get(field), int) or proof[field] < 0
        for field in nonnegative_fields
    ):
        raise AssertionError("scheduler telemetry counters must be nonnegative integers")
    if proof["scheduler_observation_count"] <= 0:
        raise AssertionError("no scheduler telemetry was retained for the generation wave")
    request_within_ceiling = proof["engine_request_count"] <= proof["max_num_seqs"]
    running_within_ceiling = proof["maximum_running_requests"] <= proof["max_num_seqs"]
    if proof.get("request_count_within_scheduler_ceiling") is not request_within_ceiling:
        raise AssertionError("scheduler request-ceiling gate is inconsistent")
    if proof.get("running_count_within_scheduler_ceiling") is not running_within_ceiling:
        raise AssertionError("scheduler running-count gate is inconsistent")
    if not request_within_ceiling or not running_within_ceiling:
        raise AssertionError("generation wave exceeded the per-engine max_num_seqs scheduler ceiling")
    preemption_fields = (
        "preemption_events",
        "recompute_events",
        "prefix_preempted_requests",
        "prefix_preempted_queries",
        "prefix_preempted_hits",
        "preempted_prompt_recomputed_tokens",
    )
    if any(proof[field] != 0 for field in preemption_fields):
        raise AssertionError("scheduler preemption/recompute occurred during the generation wave")
    if proof["prefix_preempted_hits"] > proof["prefix_preempted_queries"]:
        raise AssertionError("scheduler preempted prefix-cache hit telemetry is inconsistent")
    expected_fit = (
        request_within_ceiling and running_within_ceiling and all(proof[field] == 0 for field in preemption_fields)
    )
    if proof.get("batch_fit_without_preemption") is not expected_fit or not expected_fit:
        raise AssertionError("generation wave did not prove scheduler fit without preemption")


def validate_full_decode_proof(
    observations: list[dict[str, Any]],
    *,
    phase: str,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    """Allow mixed prefill while requiring dense, exact FULL decode replay."""
    summary = full_decode_proof_summary(
        observations,
        phase=phase,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    if summary["observation_count"] == 0:
        raise AssertionError(f"no CUDA graph observations were recorded for {phase}")
    if summary["eager_decode_dispatch_count"]:
        raise AssertionError(f"{phase} used forbidden CUDAGraphMode.NONE decode fallback execution")
    if not summary["full_decode_unpadded"]:
        raise AssertionError(f"{phase} used semantic or scheduler padding during FULL decode")
    if max_new_tokens > 1 and not summary["global_batch_hit"]:
        raise AssertionError(f"{phase} did not execute a FULL global batch")
    if summary["long_run_gates_applied"] and not summary["coverage_gate_passed"]:
        raise AssertionError(
            f"{phase} FULL decode coverage was {summary['full_decode_tokens']}/"
            f"{summary['expected_decode_tokens']} tokens; at least "
            f"{summary['minimum_full_decode_tokens']} are required"
        )
    if summary["long_run_gates_applied"] and not summary["occupancy_gate_passed"]:
        raise AssertionError(
            f"{phase} FULL decode occupancy averaged {summary['average_full_batch_occupancy']:.3f}/"
            f"{batch_size}; at least {summary['minimum_average_occupancy']:.3f} is required"
        )


def full_decode_proof_summary(
    observations: list[dict[str, Any]],
    *,
    phase: str,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Return durable numeric evidence for exact, batched FULL decode replay."""
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    phase_observations = [item for item in observations if item["phase"] == phase]
    eager_decode = [
        item
        for item in phase_observations
        if item["runtime_mode"].endswith("NONE") and item["num_unpadded_tokens"] <= batch_size
    ]
    full_decode = [item for item in phase_observations if item["runtime_mode"].endswith("FULL")]
    full_decode_unpadded = not any(
        item["num_padded_tokens"] != item["num_unpadded_tokens"] or item["num_paddings"] != 0 for item in full_decode
    )
    full_decode_tokens = sum(item["num_unpadded_tokens"] for item in full_decode)
    expected_decode_tokens = batch_size * max(0, max_new_tokens - 1)
    minimum_full_decode_tokens = max(0, expected_decode_tokens - batch_size)
    average_batch_occupancy = full_decode_tokens / len(full_decode) if full_decode else 0.0
    minimum_average_occupancy = batch_size * 0.9
    global_batch_hit = any(item["num_unpadded_tokens"] == batch_size for item in full_decode)
    long_run_gates_applied = max_new_tokens >= 32
    coverage_gate_passed = not long_run_gates_applied or full_decode_tokens >= minimum_full_decode_tokens
    occupancy_gate_passed = not long_run_gates_applied or average_batch_occupancy >= minimum_average_occupancy
    passed = (
        bool(phase_observations)
        and not eager_decode
        and full_decode_unpadded
        and (max_new_tokens <= 1 or global_batch_hit)
        and coverage_gate_passed
        and occupancy_gate_passed
    )
    return {
        "phase": phase,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "observation_count": len(phase_observations),
        "eager_decode_dispatch_count": len(eager_decode),
        "full_dispatch_count": len(full_decode),
        "expected_decode_tokens": expected_decode_tokens,
        "full_decode_tokens": full_decode_tokens,
        "minimum_full_decode_tokens": minimum_full_decode_tokens,
        "coverage_fraction": full_decode_tokens / expected_decode_tokens if expected_decode_tokens else 1.0,
        "maximum_full_batch": max((item["num_unpadded_tokens"] for item in full_decode), default=0),
        "average_full_batch_occupancy": average_batch_occupancy,
        "minimum_average_occupancy": minimum_average_occupancy,
        "occupancy_fraction": average_batch_occupancy / batch_size,
        "global_batch_hit": global_batch_hit,
        "full_decode_unpadded": full_decode_unpadded,
        "long_run_gates_applied": long_run_gates_applied,
        "coverage_gate_passed": coverage_gate_passed,
        "occupancy_gate_passed": occupancy_gate_passed,
        "passed": passed,
    }


def request_seed(
    base_seed: int,
    *,
    call_index: int,
    dp_rank: int,
    dp_size: int,
    request_index_in_stream: int,
) -> int:
    """Return one deterministic request seed from physical call and DP stream coordinates."""
    if min(base_seed, call_index, dp_rank, request_index_in_stream) < 0 or dp_size <= 0:
        raise ValueError("seed coordinates must be nonnegative")
    if dp_rank >= dp_size:
        raise ValueError("dp_rank must be smaller than dp_size")
    if request_index_in_stream >= _SEED_ROUND_STRIDE:
        raise ValueError("request index exceeds the collision-free stream stride")
    stream_seed = (base_seed + (call_index * dp_size + dp_rank) * _SEED_ROUND_STRIDE) % _SEED_MODULUS
    return (stream_seed + request_index_in_stream) % _SEED_MODULUS


def build_request_sampling_params(
    manifest: WorkloadManifest,
    *,
    sampling_params_factory: Callable[..., Any],
    execution_records: Sequence[RequestExecutionRecord],
) -> list[Any]:
    """Build exact-length sampling params from the persisted production seed records."""
    if len(execution_records) != len(manifest.requests):
        raise ValueError("execution records must align with every manifest request")
    if any(
        request.request_id != record.request_id
        for request, record in zip(manifest.requests, execution_records, strict=True)
    ):
        raise ValueError("execution record request IDs must preserve manifest order")
    common_kwargs = sampling_params_kwargs(manifest)
    return [
        sampling_params_factory(
            **common_kwargs,
            seed=record.seed,
        )
        for record in execution_records
    ]


@dataclass(frozen=True)
class RequestExecutionRecord:
    """Persist deterministic ownership and RNG coordinates for one request."""

    request_id: str
    global_request_index: int
    generation_round: int
    dp_rank: int
    call_index: int
    seed: int

    @property
    def execution_uid(self) -> str:
        """Return a phase-stable composite identity for one execution."""
        return (
            f"round={self.generation_round}/call={self.call_index}/"
            f"global={self.global_request_index}/dp={self.dp_rank}/request={self.request_id}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe execution record."""
        return {"execution_uid": self.execution_uid, **asdict(self)}


def build_request_execution_records(
    manifest: WorkloadManifest,
    *,
    global_request_offset: int,
    dp_rank: int,
    dp_size: int,
    call_index: int,
) -> tuple[RequestExecutionRecord, ...]:
    """Build one ownership/seed record for each exact manifest request."""
    if min(global_request_offset, dp_rank, call_index) < 0 or dp_size <= 0:
        raise ValueError("execution coordinates must be nonnegative")
    if dp_rank >= dp_size:
        raise ValueError("dp_rank must be smaller than dp_size")
    return tuple(
        RequestExecutionRecord(
            request_id=request.request_id,
            global_request_index=global_request_offset + local_index,
            generation_round=call_index,
            dp_rank=dp_rank,
            call_index=call_index,
            seed=request_seed(
                manifest.seed,
                call_index=call_index,
                dp_rank=dp_rank,
                dp_size=dp_size,
                request_index_in_stream=local_index,
            ),
        )
        for local_index, request in enumerate(manifest.requests)
    )


def build_wave_execution_records(
    manifest: WorkloadManifest,
    *,
    global_wave_size: int,
    call_index_start: int,
) -> tuple[RequestExecutionRecord, ...]:
    """Build exact request records whose call indices match physical generation calls."""
    if call_index_start < 0:
        raise ValueError("call_index_start must be nonnegative")
    records = []
    for wave in build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=global_wave_size,
        replica_count=1,
    ):
        records.extend(
            build_request_execution_records(
                manifest.request_slice(wave.start, wave.stop),
                global_request_offset=wave.start,
                dp_rank=0,
                dp_size=1,
                call_index=call_index_start + wave.wave_index,
            )
        )
    return tuple(records)


def write_full_output_artifact(
    path: str | Path,
    *,
    manifest: WorkloadManifest,
    outputs: Any,
    execution_records: tuple[RequestExecutionRecord, ...],
) -> dict[str, Any]:
    """Stream every output token and chosen-token logprob to deterministic gzip JSONL."""
    records = records_from_vllm_outputs(manifest, outputs)
    return write_full_generation_records_artifact(
        path,
        records=records,
        execution_records=execution_records,
    )


def write_full_generation_records_artifact(
    path: str | Path,
    *,
    records: Sequence[GenerationRecord],
    execution_records: Sequence[RequestExecutionRecord],
) -> dict[str, Any]:
    """Persist backend-neutral exact generation records as deterministic gzip JSONL."""
    if len(execution_records) != len(records):
        raise AssertionError("execution records must align with generated outputs")
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    generated_token_count = 0
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for generation, execution in zip(
                    records,
                    execution_records,
                    strict=True,
                ):
                    if execution.request_id != generation.request_id:
                        raise AssertionError("execution and generation request IDs must align")
                    row = {
                        **execution.to_dict(),
                        "prompt_token_ids": list(generation.prompt_token_ids),
                        "output_token_ids": list(generation.output_token_ids),
                        "chosen_token_logprobs": list(generation.output_logprobs),
                        **exact_length_evidence(
                            prompt_tokens=len(generation.prompt_token_ids),
                            generated_tokens=len(generation.output_token_ids),
                            requested_new_tokens=generation.requested_max_tokens,
                        ),
                        "finish_reason": generation.finish_reason,
                        "stop_reason": generation.stop_reason,
                        "stopped_on_eos": generation.stopped_on_eos,
                    }
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                    handle.write("\n")
                    generated_token_count += len(generation.output_token_ids)
    temporary.replace(output)
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema_version": 2,
        "format": "jsonl",
        "compression": "gzip",
        "path": str(output),
        "sha256": digest.hexdigest(),
        "size_bytes": output.stat().st_size,
        "request_count": len(records),
        "generated_token_count": generated_token_count,
    }


def phase_output_artifact_path(
    root_artifact_path: str | Path,
    *,
    phase: str,
    dp_rank: int | None = None,
) -> Path:
    """Return a collision-free full-output sidecar path for one phase/replica."""
    if not phase:
        raise ValueError("phase cannot be empty")
    if dp_rank is not None and dp_rank < 0:
        raise ValueError("dp_rank must be nonnegative")
    root = Path(root_artifact_path)
    replica_suffix = "" if dp_rank is None else f".dp{dp_rank}"
    return root.with_name(f"{root.name}.{phase}{replica_suffix}.outputs.jsonl.gz")


def _output_namespace_marker_path(path: str | Path) -> Path:
    output = Path(path).resolve()
    base_name = output.name.removesuffix(output.suffix)
    return output.with_name(f"{base_name}.inprogress")


def reserve_output_namespace(path: str | Path) -> Path:
    """Atomically reserve a new artifact namespace and refuse any stale outputs."""
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = _output_namespace_marker_path(output)
    legacy_marker = output.with_name(f"{output.name}.inprogress")
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    sidecar_prefix = f"{output.name.removesuffix(output.suffix)}."
    collisions = [candidate for candidate in {output, temporary, marker, legacy_marker} if candidate.exists()]
    collisions.extend(
        candidate
        for candidate in output.parent.iterdir()
        if candidate.name.startswith(sidecar_prefix)
        and (candidate.name.endswith(".outputs.jsonl.gz") or candidate.name.endswith(".outputs.jsonl.gz.tmp"))
    )
    if collisions:
        names = ", ".join(sorted({candidate.name for candidate in collisions}))
        raise FileExistsError(f"output namespace already contains prior artifacts: {names}")
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "state": "in_progress",
                "output_artifact_path": str(output),
                "started_unix_s": time.time(),
                "argv": [sys.executable, *sys.argv],
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
    return marker


def require_output_namespace_reservation(path: str | Path) -> Path:
    """Fail unless the caller reserved this exact output namespace."""
    marker = _output_namespace_marker_path(path)
    if not marker.is_file():
        raise RuntimeError(f"output namespace is not reserved: {marker}")
    return marker


def complete_output_namespace(
    marker: str | Path,
    *,
    output_path: str | Path,
    require_final_artifact: bool = True,
) -> None:
    """Release one successful reservation without touching any other artifacts."""
    output = Path(output_path).resolve()
    reservation = Path(marker).resolve()
    if reservation != _output_namespace_marker_path(output):
        raise ValueError("output namespace marker does not match the requested artifact")
    if require_final_artifact and not output.is_file():
        raise RuntimeError("cannot complete an output namespace before its final artifact exists")
    reservation.unlink()


def shared_prefix_manifest_evidence(manifest: WorkloadManifest) -> dict[str, Any]:
    """Validate one physically reusable prompt and return its stable identity."""
    if len(manifest.requests) < 2:
        raise AssertionError("shared-prefix reuse requires at least two requests")
    prompt = manifest.requests[0].prompt_token_ids
    if not prompt:
        raise AssertionError("shared-prefix reuse requires a nonempty prompt")
    if any(request.prompt_token_ids != prompt for request in manifest.requests[1:]):
        raise AssertionError("shared-prefix reuse requires identical prompt token IDs")
    payload = json.dumps(list(prompt), separators=(",", ":")).encode()
    return {
        "identical_prompt_count": len(manifest.requests),
        "prompt_tokens_per_request": len(prompt),
        "prompt_token_ids_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validated_attention_kv_groups(
    groups: Any,
    *,
    expected_prefix_tokens: int,
    expected_block_size: int,
    kind: str,
) -> list[dict[str, Any]]:
    if not isinstance(groups, list) or not groups:
        raise AssertionError(f"{kind} must retain at least one attention KV cache group")
    retained = []
    group_ids = set()
    for group in groups:
        if not isinstance(group, dict):
            raise AssertionError(f"{kind} attention KV cache group telemetry is malformed")
        group_id = group.get("kv_cache_group_id")
        if isinstance(group_id, bool) or not isinstance(group_id, int) or group_id < 0:
            raise AssertionError(f"{kind} attention KV cache group ID is malformed")
        if group_id in group_ids:
            raise AssertionError(f"{kind} attention KV cache group IDs must be unique")
        group_ids.add(group_id)
        layer_names = group.get("layer_names")
        if (
            not isinstance(layer_names, list)
            or not layer_names
            or any(not isinstance(name, str) or not name for name in layer_names)
            or len(set(layer_names)) != len(layer_names)
        ):
            raise AssertionError(f"{kind} attention KV layer ownership is malformed")
        block_size = group.get("block_size_tokens")
        if block_size != expected_block_size:
            raise AssertionError(f"{kind} attention KV block size does not match the resolved cache")
        expected_block_count = expected_prefix_tokens // expected_block_size
        if group.get("physical_block_count") != expected_block_count:
            raise AssertionError(f"{kind} attention KV physical block count is not exact")
        block_ids = group.get("physical_block_ids")
        if (
            not isinstance(block_ids, list)
            or len(block_ids) != expected_block_count
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in block_ids)
            or len(set(block_ids)) != len(block_ids)
        ):
            raise AssertionError(f"{kind} attention KV physical block IDs are malformed")
        expected_hash = hashlib.sha256(json.dumps(block_ids, separators=(",", ":")).encode()).hexdigest()
        if group.get("physical_block_ids_sha256") != expected_hash:
            raise AssertionError(f"{kind} attention KV physical block ID hash does not match retained IDs")
        retained.append(dict(group))
    return retained


def _validated_prefix_source(
    stats: dict[str, Any],
    *,
    prompt_tokens: int,
    physically_reused_tokens: int,
    cache_block_size: int,
    expected_cache_misses: int,
) -> dict[str, Any]:
    expected_worker_misses = int(expected_cache_misses > 0)
    if stats.get("cache_miss_count") != expected_worker_misses:
        raise AssertionError("physical worker cache-miss count does not match the exact wave layout")
    miss_ids = stats.get("cache_miss_request_ids")
    if not isinstance(miss_ids, list) or len(miss_ids) != expected_worker_misses:
        raise AssertionError("physical worker cache-miss request IDs are not exact")
    sources = stats.get("prefix_sources")
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
        raise AssertionError("physical worker must retain exactly one direct cache-miss source")
    source = sources[0]
    source_request_id = source.get("request_id")
    if not isinstance(source_request_id, str) or not source_request_id:
        raise AssertionError("direct cache-miss source request ID is malformed")
    if miss_ids and miss_ids != [source_request_id]:
        raise AssertionError("phase cache-miss request ID does not match the retained direct source")
    if source.get("prompt_tokens") != prompt_tokens:
        raise AssertionError("direct cache-miss source prompt length drifted")
    snapshots = source.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise AssertionError("direct cache-miss source has no physical attention block snapshots")
    previous_groups = None
    previous_prefix_tokens = 0
    retained_snapshots = []
    for snapshot_index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, dict) or snapshot.get("snapshot_index") != snapshot_index:
            raise AssertionError("direct cache-miss source snapshot indices must be exact and contiguous")
        for key in ("num_computed_tokens_before_step", "num_scheduled_tokens"):
            value = snapshot.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AssertionError(f"direct cache-miss source {key} is malformed")
        prefix_tokens = snapshot.get("directly_observed_prefix_tokens")
        if (
            isinstance(prefix_tokens, bool)
            or not isinstance(prefix_tokens, int)
            or prefix_tokens <= previous_prefix_tokens
            or prefix_tokens > physically_reused_tokens
            or prefix_tokens % cache_block_size
        ):
            raise AssertionError("direct cache-miss source prefix coverage is malformed")
        if snapshot["num_computed_tokens_before_step"] + snapshot["num_scheduled_tokens"] < prefix_tokens:
            raise AssertionError("direct cache-miss source snapshot exceeds scheduled prefill work")
        groups = _validated_attention_kv_groups(
            snapshot.get("attention_kv_groups"),
            expected_prefix_tokens=prefix_tokens,
            expected_block_size=cache_block_size,
            kind="source",
        )
        if previous_groups is not None:
            if len(previous_groups) != len(groups):
                raise AssertionError("direct cache-miss source attention group count changed")
            for previous, current in zip(previous_groups, groups, strict=True):
                for key in ("kv_cache_group_id", "layer_names", "block_size_tokens"):
                    if previous[key] != current[key]:
                        raise AssertionError("direct cache-miss source attention ownership changed")
                previous_ids = previous["physical_block_ids"]
                if previous_ids != current["physical_block_ids"][: len(previous_ids)]:
                    raise AssertionError("direct cache-miss source physical block identity changed")
        previous_groups = groups
        previous_prefix_tokens = prefix_tokens
        retained_snapshots.append({**snapshot, "attention_kv_groups": groups})
    if previous_prefix_tokens != physically_reused_tokens:
        raise AssertionError("direct cache-miss source does not cover the exact reusable prompt prefix")
    return {**source, "snapshots": retained_snapshots}


def _validated_fp32_state_copies(record: dict[str, Any]) -> list[dict[str, Any]]:
    copies = record.get("state_copies")
    if not isinstance(copies, list) or len(copies) != record["copy_entries"]:
        raise AssertionError("physical prefix clone must retain every recurrent-state copy entry")
    retained = []
    identities = set()
    copied_elements = 0
    copied_bytes = 0
    for entry in copies:
        if not isinstance(entry, dict):
            raise AssertionError("recurrent-state copy entry is malformed")
        group_id = entry.get("kv_cache_group_id")
        layer_name = entry.get("layer_name")
        state_index = entry.get("state_index")
        if (
            isinstance(group_id, bool)
            or not isinstance(group_id, int)
            or group_id < 0
            or not isinstance(layer_name, str)
            or not layer_name
            or isinstance(state_index, bool)
            or not isinstance(state_index, int)
            or state_index < 0
        ):
            raise AssertionError("recurrent-state copy ownership is malformed")
        identity = (group_id, layer_name, state_index)
        if identity in identities:
            raise AssertionError("recurrent-state copy ownership must be unique per request")
        identities.add(identity)
        if entry.get("dtype") != "torch.float32":
            raise AssertionError("every physical Evo2 recurrent-state copy must be FP32")
        state_shape = entry.get("state_shape")
        block_shape = entry.get("block_shape")
        if (
            not isinstance(state_shape, list)
            or len(state_shape) < 2
            or not isinstance(block_shape, list)
            or not block_shape
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in state_shape)
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in block_shape)
            or state_shape[1:] != block_shape
        ):
            raise AssertionError("recurrent-state copy tensor shape evidence is malformed")
        integer_keys = (
            "source_logical_block_index",
            "destination_logical_block_index",
            "source_physical_block_id",
            "destination_physical_block_id",
            "source_data_ptr",
            "destination_data_ptr",
            "copied_elements",
            "copied_bytes",
        )
        if any(
            isinstance(entry.get(key), bool) or not isinstance(entry.get(key), int) or entry[key] < 0
            for key in integer_keys
        ):
            raise AssertionError("recurrent-state copy pointer, slot, or size evidence is malformed")
        if entry["source_logical_block_index"] == entry["destination_logical_block_index"]:
            raise AssertionError("recurrent-state copy logical source and destination must differ")
        if entry["source_physical_block_id"] == entry["destination_physical_block_id"]:
            raise AssertionError("recurrent-state copy physical source and destination must differ")
        if entry["source_data_ptr"] <= 0 or entry["destination_data_ptr"] <= 0:
            raise AssertionError("recurrent-state copy pointers must be concrete nonzero addresses")
        if entry["source_data_ptr"] == entry["destination_data_ptr"]:
            raise AssertionError("recurrent-state copy source and destination pointers must differ")
        if state_shape[0] <= max(entry["source_physical_block_id"], entry["destination_physical_block_id"]):
            raise AssertionError("recurrent-state physical block IDs exceed the retained tensor shape")
        if entry["copied_elements"] != math.prod(block_shape):
            raise AssertionError("recurrent-state copied elements do not match the retained block shape")
        if entry["copied_bytes"] != 4 * entry["copied_elements"]:
            raise AssertionError("recurrent-state copied bytes do not match exact FP32 storage")
        copied_elements += entry["copied_elements"]
        copied_bytes += entry["copied_bytes"]
        retained.append(dict(entry))
    if copied_elements != record["copied_elements"] or copied_bytes != record["copied_bytes"]:
        raise AssertionError("per-state copy entries do not sum to the physical clone totals")
    return retained


def shared_prefix_state_reuse_evidence(
    manifest: WorkloadManifest,
    *,
    cached_tokens: Sequence[int | None],
    worker_proof: Sequence[dict[str, Any]],
    expected_worker_clone_counts: Sequence[int],
    cache_block_size: int,
    expected_cache_misses: int = 1,
) -> dict[str, Any]:
    """Prove exact scheduler hits and request-scoped FP32 recurrent-state clones."""
    identity = shared_prefix_manifest_evidence(manifest)
    request_count = len(manifest.requests)
    if len(cached_tokens) != request_count:
        raise AssertionError("cached-token telemetry must cover every request")
    if isinstance(cache_block_size, bool) or not isinstance(cache_block_size, int) or cache_block_size <= 0:
        raise AssertionError("prefix cache block size must be a positive integer")
    if expected_cache_misses < 0 or expected_cache_misses >= request_count:
        raise AssertionError("expected cache misses must leave at least one physical prefix clone")
    if len(expected_worker_clone_counts) != len(worker_proof) or not worker_proof:
        raise AssertionError("expected worker clone counts must cover every physical worker")

    prompt_tokens = identity["prompt_tokens_per_request"]
    physically_reused_tokens = (prompt_tokens - 1) // cache_block_size * cache_block_size
    if physically_reused_tokens <= 0:
        raise AssertionError("the prompt is too short for one block-aligned prefix clone")

    normalized_counts = []
    for value in cached_tokens:
        if value is None or isinstance(value, bool) or not isinstance(value, int):
            raise AssertionError("cached-token telemetry must contain concrete integer counts")
        if not 0 <= value <= prompt_tokens:
            raise AssertionError("cached-token telemetry exceeds the exact prompt length")
        normalized_counts.append(value)

    miss_count = sum(value == 0 for value in normalized_counts)
    if miss_count != expected_cache_misses:
        qualifier = "one" if expected_cache_misses == 1 else str(expected_cache_misses)
        raise AssertionError(f"shared-prefix execution must have exactly {qualifier} cache miss")
    hit_counts = [value for value in normalized_counts if value > 0]
    if len(hit_counts) != request_count - expected_cache_misses:
        raise AssertionError("shared-prefix execution did not clone every request after each replica's miss")
    if any(value != physically_reused_tokens for value in hit_counts):
        raise AssertionError("every cache hit must reuse the exact block-aligned prefix")

    worker_clones = []
    expected_elements_per_request = set()
    expected_bytes_per_request = set()
    for proof, expected_clone_count in zip(worker_proof, expected_worker_clone_counts, strict=True):
        if isinstance(expected_clone_count, bool) or not isinstance(expected_clone_count, int):
            raise AssertionError("expected worker clone counts must be integers")
        stats = proof.get("mamba_prefix_clones")
        if not isinstance(stats, dict):
            raise AssertionError("shared-prefix proof is missing request-scoped physical state clones")
        clone_count = stats.get("clone_count")
        requests = stats.get("requests")
        if clone_count != expected_clone_count or not isinstance(requests, list) or len(requests) != clone_count:
            raise AssertionError("physical worker prefix clone count does not match the exact request layout")
        source = _validated_prefix_source(
            stats,
            prompt_tokens=prompt_tokens,
            physically_reused_tokens=physically_reused_tokens,
            cache_block_size=cache_block_size,
            expected_cache_misses=expected_cache_misses,
        )
        source_snapshot = source["snapshots"][-1]
        source_groups = source_snapshot["attention_kv_groups"]

        retained_requests = []
        request_ids = set()
        for record in requests:
            if not isinstance(record, dict):
                raise AssertionError("request-scoped physical clone telemetry is malformed")
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or not request_id or request_id in request_ids:
                raise AssertionError("physical prefix clone request IDs must be unique nonempty strings")
            request_ids.add(request_id)
            for key in (
                "num_computed_tokens",
                "prompt_tokens",
                "block_size",
                "copy_entries",
                "copied_elements",
                "copied_bytes",
                "expected_copy_entries",
                "expected_copied_elements",
                "expected_copied_bytes",
            ):
                value = record.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise AssertionError(f"physical prefix clone {key} telemetry is malformed")
            if record["num_computed_tokens"] != physically_reused_tokens:
                raise AssertionError("physical clone source position does not match the scheduler cache hit")
            if record["prompt_tokens"] != prompt_tokens or record["block_size"] != cache_block_size:
                raise AssertionError("physical clone prompt or block-size provenance drifted")
            if record.get("all_state_dtypes_fp32") is not True:
                raise AssertionError("every cloned Evo2 recurrent state must be FP32")
            if record["copy_entries"] != record["expected_copy_entries"]:
                raise AssertionError("physical prefix clone copy entry count does not match the Evo2 state layout")
            if record["copied_elements"] != record["expected_copied_elements"]:
                raise AssertionError("physical prefix clone copy elements do not match the Evo2 state layout")
            if record["copied_bytes"] != record["expected_copied_bytes"]:
                raise AssertionError("physical prefix clone copy bytes do not match the Evo2 FP32 state layout")
            if record.get("source_miss_request_id") != source["request_id"]:
                raise AssertionError("physical prefix clone does not name its direct cache-miss source")
            if record.get("source_snapshot_index") != source_snapshot["snapshot_index"]:
                raise AssertionError("physical prefix clone does not name the exact direct source snapshot")
            if record.get("attention_kv_identity_verified") is not True:
                raise AssertionError("physical prefix clone lacks attention KV identity verification")
            retained_source_groups = _validated_attention_kv_groups(
                record.get("source_attention_kv_groups"),
                expected_prefix_tokens=physically_reused_tokens,
                expected_block_size=cache_block_size,
                kind="clone source",
            )
            retained_reused_groups = _validated_attention_kv_groups(
                record.get("reused_attention_kv_groups"),
                expected_prefix_tokens=physically_reused_tokens,
                expected_block_size=cache_block_size,
                kind="clone hit",
            )
            if retained_source_groups != source_groups:
                raise AssertionError("clone source attention KV blocks do not match the direct miss snapshot")
            if retained_reused_groups != retained_source_groups:
                raise AssertionError("clone hit attention KV physical block IDs do not exactly match its source")
            state_copies = _validated_fp32_state_copies(record)
            expected_elements_per_request.add(record["expected_copied_elements"])
            expected_bytes_per_request.add(record["expected_copied_bytes"])
            retained_requests.append(
                {
                    **record,
                    "source_attention_kv_groups": retained_source_groups,
                    "reused_attention_kv_groups": retained_reused_groups,
                    "state_copies": state_copies,
                }
            )

        worker_clones.append(
            {
                "rank": int(proof.get("rank", 0)),
                "device": int(proof.get("device", 0)),
                "clone_count": clone_count,
                "prefix_source": source,
                "requests": retained_requests,
            }
        )

    if len(expected_elements_per_request) != 1 or len(expected_bytes_per_request) != 1:
        raise AssertionError("all physical prefix clones must retain one exact Evo2 state-copy layout")

    total_prompt_tokens = prompt_tokens * request_count
    return {
        **identity,
        "cache_block_size": cache_block_size,
        "cached_tokens_by_request": normalized_counts,
        "cache_hit_request_count": len(hit_counts),
        "cache_miss_request_count": miss_count,
        "logical_clone_request_count": len(hit_counts),
        "physically_reused_prompt_tokens_per_clone": physically_reused_tokens,
        "recomputed_prompt_tokens_per_clone": prompt_tokens - physically_reused_tokens,
        "total_cached_prompt_tokens": sum(normalized_counts),
        "scheduled_uncached_prompt_tokens": total_prompt_tokens - sum(normalized_counts),
        "worker_state_clones": worker_clones,
        "rank_local_physical_clone_count": sum(worker["clone_count"] for worker in worker_clones),
        "expected_fp32_state_copy_elements_per_request": next(iter(expected_elements_per_request)),
        "expected_fp32_state_copy_bytes_per_request": next(iter(expected_bytes_per_request)),
        "attention_kv_physical_reuse_proven": True,
        "physical_state_copy_proven": True,
    }


def wave_execution_summary(
    wave_proofs: Sequence[dict[str, Any]],
    *,
    target_request_count: int = 96,
) -> dict[str, Any]:
    """Retain actual physical calls and measured wall time needed to cover a target batch."""
    if target_request_count <= 0:
        raise ValueError("target_request_count must be positive")
    if not wave_proofs:
        raise AssertionError("wave execution summary requires at least one physical generation call")

    request_counts = []
    generation_s = []
    covered_requests = 0
    measured_waves_to_target = None
    measured_time_to_target_s = None
    requests_completed_at_target_boundary = None
    for expected_index, proof in enumerate(wave_proofs):
        if proof.get("wave_index") != expected_index:
            raise AssertionError("physical generation wave indices must be exact and contiguous")
        request_count = proof.get("request_count")
        elapsed = proof.get("generation_s")
        if isinstance(request_count, bool) or not isinstance(request_count, int) or request_count <= 0:
            raise AssertionError("physical generation wave request counts must be positive integers")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise AssertionError("physical generation wave timings must be finite and positive")
        request_counts.append(request_count)
        generation_s.append(float(elapsed))
        covered_requests += request_count
        if measured_waves_to_target is None and covered_requests >= target_request_count:
            measured_waves_to_target = expected_index + 1
            measured_time_to_target_s = sum(generation_s)
            requests_completed_at_target_boundary = covered_requests

    return {
        "target_request_count": target_request_count,
        "actual_call_count": len(wave_proofs),
        "actual_request_count": sum(request_counts),
        "call_request_counts": request_counts,
        "call_generation_s": generation_s,
        "measured_waves_to_target": measured_waves_to_target,
        "measured_time_to_target_s": measured_time_to_target_s,
        "requests_completed_at_target_boundary": requests_completed_at_target_boundary,
    }


@dataclass(frozen=True)
class GenerationPhaseResult:
    """One timed generation phase plus its unreset CUDA graph observations."""

    phase: str
    sample: BenchmarkSample
    generation_call_s: tuple[float, ...]
    wave_proofs: tuple[dict[str, Any], ...]
    observations: tuple[dict[str, Any], ...]
    output_summaries: tuple[dict[str, Any], ...]
    request_executions: tuple[RequestExecutionRecord, ...]
    full_output_artifact: dict[str, Any]
    full_decode_proof: dict[str, Any] | None
    worker_proof: tuple[dict[str, Any], ...]
    shared_prefix_state_reuse: dict[str, Any] | None
    proof_collected: bool
    prefix_cache_reset: bool

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
            "generation_call_s": list(self.generation_call_s),
            "wave_proofs": list(self.wave_proofs),
            "wave_execution": wave_execution_summary(self.wave_proofs),
            "cudagraph_observation_count": len(self.observations),
            "cudagraph_observations_retained": retained_observations,
            "cudagraph_summary": summarize_cudagraph_observations(self.observations),
            "outputs": list(self.output_summaries),
            "request_executions": [record.to_dict() for record in self.request_executions],
            "full_output_artifact": self.full_output_artifact,
            "full_decode_proof": self.full_decode_proof,
            "worker_proof": list(self.worker_proof),
            "shared_prefix_state_reuse": self.shared_prefix_state_reuse,
            "proof_collected": self.proof_collected,
            "prefix_cache_reset": self.prefix_cache_reset,
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
    execution_records: tuple[RequestExecutionRecord, ...],
    full_output_path: str | Path,
    collect_proof: bool = True,
    reset_worker_proof: Callable[[], Any] | None = None,
    snapshot_worker_proof: Callable[[], tuple[dict[str, Any], ...]] | None = None,
    prefix_cache_block_size: int | None = None,
    require_shared_prefix_state_reuse: bool = False,
    global_wave_size: int | None = None,
    scheduler_max_num_seqs: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
    barrier: Any | None = None,
) -> GenerationPhaseResult:
    """Time explicit offline vLLM calls while preserving one ordered phase artifact."""
    if collect_proof and recorder is None:
        raise ValueError("proof collection requires a CUDA graph recorder")
    if len(sampling_params) != len(manifest.requests):
        raise ValueError("sampling params must align with every request")
    if len(execution_records) != len(manifest.requests):
        raise ValueError("execution records must align with every request")
    for request, params, execution in zip(manifest.requests, sampling_params, execution_records, strict=True):
        if execution.request_id != request.request_id:
            raise ValueError("execution record request IDs must preserve manifest order")
        if params.seed != execution.seed:
            raise ValueError("sampling parameter seeds must match persisted execution records")
    wave_size = len(manifest.requests) if global_wave_size is None else global_wave_size
    waves = build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=wave_size,
        replica_count=1,
    )
    if scheduler_max_num_seqs is not None and scheduler_max_num_seqs <= 0:
        raise ValueError("scheduler_max_num_seqs must be positive")
    call_index_start = execution_records[0].call_index
    for wave in waves:
        call_indexes = {record.call_index for record in execution_records[wave.start : wave.stop]}
        expected_call_index = call_index_start + wave.wave_index
        if call_indexes != {expected_call_index}:
            raise ValueError("execution call indices must match explicit generation waves")

    observation_start = 0 if recorder is None else len(recorder.observations)
    if collect_proof and reset_worker_proof is not None:
        reset_worker_proof()
    prefix_cache_reset = False
    if require_shared_prefix_state_reuse:
        shared_prefix_manifest_evidence(manifest)
        reset_prefix_cache = getattr(llm, "reset_prefix_cache", None)
        if reset_prefix_cache is None or reset_prefix_cache() is not True:
            raise AssertionError("shared-prefix execution requires a successful phase-local prefix-cache reset")
        prefix_cache_reset = True

    outputs = []
    generation_call_s = []
    wave_proofs = []
    monitor_context = memory_monitor_factory() if collect_proof else _UnmonitoredMemory()
    with monitor_context as monitor:
        for wave in waves:
            wave_phase = f"{phase}.wave-{wave.wave_index:03d}"
            wave_manifest = manifest.request_slice(wave.start, wave.stop)
            wave_prompts = [{"prompt_token_ids": list(request.prompt_token_ids)} for request in wave_manifest.requests]
            if collect_proof:
                recorder.start_phase(wave_phase)
                wave_observation_start = len(recorder.observations)
                wave_scheduler_start = len(recorder.scheduler_observations)
            if barrier is not None:
                barrier.wait()
            begin = clock()
            wave_outputs = list(
                llm.generate(
                    wave_prompts,
                    sampling_params[wave.start : wave.stop],
                    use_tqdm=False,
                )
            )
            if barrier is not None:
                barrier.wait()
            elapsed = clock() - begin
            if len(wave_outputs) != wave.request_count:
                raise AssertionError("vLLM output count must match the explicit generation wave")
            generation_call_s.append(elapsed)
            outputs.extend(wave_outputs)

            full_decode = None
            scheduler_proof = None
            if collect_proof:
                full_decode = full_decode_proof_summary(
                    recorder.observations[wave_observation_start:],
                    phase=wave_phase,
                    batch_size=wave.request_count,
                    max_new_tokens=manifest.max_new_tokens,
                )
                scheduler_proof = scheduler_capacity_proof_summary(
                    recorder.scheduler_observations[wave_scheduler_start:],
                    phase=wave_phase,
                    global_wave_size=wave.request_count,
                    max_num_seqs=(wave.request_count if scheduler_max_num_seqs is None else scheduler_max_num_seqs),
                )
                if scheduler_max_num_seqs is not None:
                    validate_scheduler_capacity_proof(scheduler_proof)
            wave_proofs.append(
                {
                    "wave_index": wave.wave_index,
                    "start": wave.start,
                    "stop": wave.stop,
                    "request_count": wave.request_count,
                    "call_index": call_index_start + wave.wave_index,
                    "generation_s": elapsed,
                    "full_decode_proof": full_decode,
                    "scheduler_capacity_proof": scheduler_proof,
                }
            )
    worker_proof = () if not collect_proof or snapshot_worker_proof is None else snapshot_worker_proof()
    shared_prefix_reuse = None
    if collect_proof and require_shared_prefix_state_reuse:
        if prefix_cache_block_size is None:
            raise AssertionError("shared-prefix proof requires the resolved cache block size")
        shared_prefix_reuse = shared_prefix_state_reuse_evidence(
            manifest,
            cached_tokens=tuple(getattr(output, "num_cached_tokens", None) for output in outputs),
            worker_proof=worker_proof,
            expected_worker_clone_counts=tuple(len(manifest.requests) - 1 for _ in worker_proof),
            cache_block_size=prefix_cache_block_size,
        )
        shared_prefix_reuse = {
            **shared_prefix_reuse,
            "phase_prefix_cache_reset": prefix_cache_reset,
        }

    output_summaries = summarize_vllm_outputs(manifest, outputs)
    sample = benchmark_sample_from_vllm_outputs(
        manifest,
        outputs,
        sample_index=sample_index,
        generation_s=sum(generation_call_s),
        peak_device_memory_bytes=monitor.peak_device_memory_bytes,
        validated_summaries=output_summaries,
    )
    full_output_artifact = write_full_output_artifact(
        full_output_path,
        manifest=manifest,
        outputs=outputs,
        execution_records=execution_records,
    )
    full_decode_proof = None
    if collect_proof:
        expected_decode_tokens = sum(
            int(proof["full_decode_proof"]["expected_decode_tokens"]) for proof in wave_proofs
        )
        full_decode_tokens = sum(int(proof["full_decode_proof"]["full_decode_tokens"]) for proof in wave_proofs)
        full_decode_proof = {
            "phase": phase,
            "wave_count": len(wave_proofs),
            "expected_decode_tokens": expected_decode_tokens,
            "full_decode_tokens": full_decode_tokens,
            "coverage_fraction": full_decode_tokens / expected_decode_tokens if expected_decode_tokens else 1.0,
            "passed": all(proof["full_decode_proof"]["passed"] for proof in wave_proofs),
            "waves": [proof["full_decode_proof"] for proof in wave_proofs],
        }
    return GenerationPhaseResult(
        phase=phase,
        sample=sample,
        generation_call_s=tuple(generation_call_s),
        wave_proofs=tuple(wave_proofs),
        observations=(tuple(recorder.observations[observation_start:]) if collect_proof else ()),
        output_summaries=output_summaries,
        request_executions=execution_records,
        full_output_artifact=full_output_artifact,
        full_decode_proof=full_decode_proof,
        worker_proof=worker_proof,
        shared_prefix_state_reuse=shared_prefix_reuse,
        proof_collected=collect_proof,
        prefix_cache_reset=prefix_cache_reset,
    )


def reset_vllm_worker_proof_state(
    worker: Any,
    reset_prefix_sources: bool = True,
) -> dict[str, int | bool]:
    """Reset phase-local FIR telemetry and CUDA allocator peaks on one vLLM worker."""
    del worker
    import torch

    from bionemo.evo2.vllm.model import (
        install_mamba_prefix_clone_proof_hook,
        reset_mamba_prefix_clone_stats,
        reset_mamba_state_copy_stats,
    )
    from bionemo.evo2.vllm.packed_fir import reset_fir_route_stats

    reset_fir_route_stats()
    reset_mamba_state_copy_stats()
    torch.cuda.reset_peak_memory_stats()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    install_mamba_prefix_clone_proof_hook()
    reset_mamba_prefix_clone_stats(reset_prefix_sources=reset_prefix_sources)
    return {
        "rank": int(rank),
        "device": int(torch.cuda.current_device()),
        "reset_prefix_sources": reset_prefix_sources,
    }


def snapshot_vllm_worker_proof_state(worker: Any) -> dict[str, Any]:
    """Collect route, compile, and CUDA-memory evidence from one vLLM worker."""
    del worker
    import torch

    from bionemo.evo2.vllm.model import (
        get_mamba_prefix_clone_stats,
        get_mamba_state_copy_stats,
    )
    from bionemo.evo2.vllm.packed_fir import get_fir_route_stats
    from bionemo.evo2.vllm.profile import compilation_counter_snapshot

    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    gpu_identity = worker_gpu_identity(logical_device=int(device))
    return {
        "rank": int(rank),
        "device": int(device),
        **gpu_identity,
        "fir_routes": get_fir_route_stats(),
        "mamba_state_copies": get_mamba_state_copy_stats(),
        "mamba_prefix_clones": get_mamba_prefix_clone_stats(),
        "compilation": compilation_counter_snapshot(),
        "cuda_memory": {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }


class _UnmonitoredMemory:
    """Expose the sample interface without polling during speed measurements."""

    peak_device_memory_bytes: tuple[int, ...] = ()

    def __enter__(self) -> _UnmonitoredMemory:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback


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


def _nvml_text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def worker_gpu_identity(
    *,
    logical_device: int,
    nvml_module: Any | None = None,
) -> dict[str, Any]:
    """Resolve one CUDA logical device to an exact physical UUID and PCI address."""
    if isinstance(logical_device, bool) or not isinstance(logical_device, int) or logical_device < 0:
        raise ValueError("logical_device must be a nonnegative integer")
    if nvml_module is None:
        import pynvml as nvml_module

    nvml_module.nvmlInit()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = tuple(item.strip() for item in visible.split(",")) if visible is not None and visible.strip() else ()
    if selectors:
        if logical_device >= len(selectors) or not selectors[logical_device]:
            raise RuntimeError("CUDA logical device is not represented by CUDA_VISIBLE_DEVICES")
        selector = selectors[logical_device]
    else:
        selector = str(logical_device)
    if selector.isdecimal():
        handle = nvml_module.nvmlDeviceGetHandleByIndex(int(selector))
    elif selector.startswith("GPU-"):
        handle = nvml_module.nvmlDeviceGetHandleByUUID(selector)
    else:
        raise RuntimeError(f"unsupported CUDA_VISIBLE_DEVICES selector: {selector!r}")
    return {
        "logical_device": logical_device,
        "cuda_visible_devices": visible,
        "visible_device_selector": selector,
        "device_uuid": _nvml_text(nvml_module.nvmlDeviceGetUUID(handle)),
        "pci_bus_id": _nvml_text(nvml_module.nvmlDeviceGetPciInfo(handle).busId),
        "device_name": _nvml_text(nvml_module.nvmlDeviceGetName(handle)),
    }


def gpu_hardware_provenance(
    *,
    nvml_module: Any | None = None,
    expected_device_count: int = 2,
) -> dict[str, Any]:
    """Record exact driver, UUID, name, and physical memory for every GPU."""
    if expected_device_count <= 0:
        raise ValueError("expected_device_count must be positive")
    if nvml_module is None:
        import pynvml as nvml_module

    nvml_module.nvmlInit()
    device_count = int(nvml_module.nvmlDeviceGetCount())
    if device_count != expected_device_count:
        raise RuntimeError(f"benchmark requires exactly {expected_device_count} NVML devices, found {device_count}")
    devices = []
    for index in range(device_count):
        handle = nvml_module.nvmlDeviceGetHandleByIndex(index)
        memory = nvml_module.nvmlDeviceGetMemoryInfo(handle)
        devices.append(
            {
                "index": index,
                "uuid": _nvml_text(nvml_module.nvmlDeviceGetUUID(handle)),
                "pci_bus_id": _nvml_text(nvml_module.nvmlDeviceGetPciInfo(handle).busId),
                "name": _nvml_text(nvml_module.nvmlDeviceGetName(handle)),
                "total_memory_bytes": int(memory.total),
                "initial_used_memory_bytes": int(memory.used),
                "initial_free_memory_bytes": int(memory.free),
            }
        )
    return {
        "driver_version": _nvml_text(nvml_module.nvmlSystemGetDriverVersion()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": device_count,
        "devices": devices,
    }


def gpu_memory_headroom_evidence(
    hardware: dict[str, Any],
    *,
    peak_device_memory_bytes: Sequence[int],
    required_headroom_bytes: int = _REQUIRED_GPU_HEADROOM_BYTES,
) -> dict[str, Any]:
    """Require at least 2 GiB beyond the observed peak on every benchmark GPU."""
    if required_headroom_bytes <= 0:
        raise ValueError("required_headroom_bytes must be positive")
    devices = hardware.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ValueError("GPU hardware provenance must contain a nonempty device list")
    if len(devices) != len(peak_device_memory_bytes):
        raise ValueError("peak-memory samples must align with every provenance GPU")

    retained = []
    for device, peak in zip(devices, peak_device_memory_bytes, strict=True):
        total = device.get("total_memory_bytes")
        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError("GPU total memory must be a positive integer")
        if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
            raise ValueError("GPU peak memory must be a nonnegative integer")
        headroom = total - peak
        retained.append(
            {
                "index": int(device.get("index", len(retained))),
                "uuid": str(device.get("uuid", "")),
                "total_memory_bytes": total,
                "peak_used_memory_bytes": peak,
                "headroom_bytes": headroom,
            }
        )
        if headroom < required_headroom_bytes:
            raise RuntimeError(
                f"GPU {device.get('uuid', device.get('index'))} has {headroom} bytes headroom; "
                "at least 2 GiB is required"
            )
    return {
        "required_headroom_bytes": required_headroom_bytes,
        "devices": retained,
        "passed": True,
    }


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


def profile_from_args(args: Any, manifest: WorkloadManifest) -> Evo2VllmProfile:
    """Map benchmark CLI settings to one topology-local physical engine profile."""
    profile = Evo2VllmProfile(
        topology=args.topology,
        max_model_len=args.max_model_len or manifest.max_total_tokens,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        async_scheduling=args.async_scheduling,
        proof=args.proof,
        max_concurrent_partial_prefills=args.max_concurrent_partial_prefills,
        long_prefill_chunk_tokens=args.long_prefill_chunk_tokens,
        optimization_level=args.optimization_level,
        performance_mode=args.performance_mode,
        shared_prefix_state_reuse=getattr(args, "shared_prefix_state_reuse", False),
        global_wave_size=getattr(args, "global_wave_size", 96),
        max_num_seqs=getattr(args, "max_num_seqs", None),
    )
    if profile.shared_prefix_state_reuse:
        shared_prefix_manifest_evidence(manifest)
    return profile


def run_context_length_preflight(args: Any, manifest: WorkloadManifest) -> dict[str, Any]:
    """Persist the resolved long-context contract without constructing a GPU engine."""
    require_output_namespace_reservation(args.output)
    profile = profile_from_args(args, manifest)

    preflight_begin = time.perf_counter()
    preflight = context_length_preflight(
        profile,
        model=args.checkpoint,
        workload_max_total_tokens=manifest.max_total_tokens,
        load_format=args.load_format,
    )
    preflight_s = time.perf_counter() - preflight_begin

    provenance_begin = time.perf_counter()
    source_identity = source_provenance()
    provenance_s = time.perf_counter() - provenance_begin
    return {
        "schema_version": 1,
        "task": "evo2-vllm-context-length-preflight",
        "benchmark_mode": "preflight",
        "backend": "vllm",
        "topology": args.topology,
        "versions": runtime_versions(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_provenance": source_identity,
        "invocation": {
            "argv": [sys.executable, *sys.argv],
            "parsed_args": {
                name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
            },
            "output_artifact_path": str(Path(args.output).resolve()),
            "exit_status": 0,
        },
        "manifest": manifest.to_dict(),
        "manifest_sha256": manifest.sha256,
        "profile": asdict(profile),
        "context_length_preflight": preflight,
        "timing": {
            "context_length_preflight_s": preflight_s,
            "source_provenance_s": provenance_s,
        },
    }


def run_tp2_benchmark(args: Any, manifest: WorkloadManifest) -> dict[str, Any]:
    """Run one TP2 Ray engine through cold, warm, and measured exact phases."""
    if args.topology != "tp2":
        raise ValueError("run_tp2_benchmark requires topology=tp2")
    require_output_namespace_reservation(args.output)
    profile = profile_from_args(args, manifest)
    benchmark_mode = benchmark_mode_from_args(args)
    instrumentation = benchmark_instrumentation_contract(benchmark_mode)
    preflight_begin = time.perf_counter()
    preflight = context_length_preflight(
        profile,
        model=args.checkpoint,
        workload_max_total_tokens=manifest.max_total_tokens,
        load_format=args.load_format,
    )
    preflight_s = time.perf_counter() - preflight_begin

    vllm_import_begin = time.perf_counter()
    from vllm import LLM, SamplingParams

    vllm_import_s = time.perf_counter() - vllm_import_begin

    provenance_begin = time.perf_counter()
    checkpoint_identity = checkpoint_provenance(args.checkpoint)
    source_identity = source_provenance(require_clean=True)
    vllm_identity = vllm_installation_provenance()
    gpu_identity = gpu_hardware_provenance()
    runtime_attestation = runtime_attestation_contract(
        checkpoint=checkpoint_identity,
        sources={"bionemo": source_identity},
        vllm_installation=vllm_identity,
        gpu_hardware=gpu_identity,
    )
    benchmark_contract = {
        **build_benchmark_contract(args, manifest, profile),
        "runtime_attestation": runtime_attestation,
    }
    linked_proof = (
        None
        if benchmark_mode == "proof"
        else validate_linked_proof_artifact(
            args.linked_proof_artifact,
            expected_contract=benchmark_contract,
            require_memory_headroom=True,
        )
    )
    provenance_s = time.perf_counter() - provenance_begin

    engine_kwargs = profile.engine_kwargs(
        model=str(args.checkpoint),
        seed=manifest.seed,
        load_format=args.load_format,
    )
    memory_reader = make_nvml_memory_reader()
    recorder = CUDAGraphProofRecorder() if profile.proof else None

    init_begin = time.perf_counter()
    with PeakMemoryMonitor(memory_reader) as init_memory:
        llm = LLM(**engine_kwargs)
    engine_init_s = time.perf_counter() - init_begin
    if profile.proof:
        _attach_cudagraph_recorder(llm, recorder)
    resolved = resolved_config_snapshot(llm.llm_engine.vllm_config)
    validate_resolved_profile(profile, resolved)
    initialized_worker_proof = _snapshot_worker_proof(llm) if profile.proof else ()

    phase_results = []
    call_index_start = args.generation_round
    for sample_index, (phase, _) in enumerate(_phase_specs(args.warmups, args.repetitions)):
        execution_records = build_wave_execution_records(
            manifest,
            global_wave_size=profile.global_wave_size,
            call_index_start=call_index_start,
        )
        sampling_params = build_request_sampling_params(
            manifest,
            sampling_params_factory=SamplingParams,
            execution_records=execution_records,
        )
        result = run_generation_phase(
            llm=llm,
            manifest=manifest,
            sampling_params=sampling_params,
            phase=phase,
            sample_index=sample_index,
            recorder=recorder,
            memory_monitor_factory=lambda: PeakMemoryMonitor(memory_reader),
            execution_records=execution_records,
            full_output_path=phase_output_artifact_path(args.output, phase=phase),
            collect_proof=profile.proof,
            reset_worker_proof=(lambda: _reset_worker_proof(llm)) if profile.proof else None,
            snapshot_worker_proof=(lambda: _snapshot_worker_proof(llm)) if profile.proof else None,
            require_shared_prefix_state_reuse=profile.shared_prefix_state_reuse,
            prefix_cache_block_size=int(resolved["cache"]["block_size"]),
            global_wave_size=profile.global_wave_size,
            scheduler_max_num_seqs=profile.resolved_max_num_seqs,
        )
        if args.proof:
            for wave_proof in result.wave_proofs:
                validate_full_decode_proof(
                    list(result.observations),
                    phase=wave_proof["full_decode_proof"]["phase"],
                    batch_size=wave_proof["request_count"],
                    max_new_tokens=manifest.max_new_tokens,
                )
        phase_results.append(result)
        call_index_start += len(result.generation_call_s)

    if profile.proof:
        final_worker_proof = phase_results[-1].worker_proof
        for initialized, final in zip(initialized_worker_proof, final_worker_proof, strict=True):
            validate_compilation_proof(initialized["compilation"], final["compilation"])
        memory_samples = [
            init_memory.peak_device_memory_bytes,
            *(result.sample.peak_device_memory_bytes for result in phase_results),
        ]
        peak_device_memory = tuple(max(values) for values in zip(*memory_samples, strict=True))
        memory_headroom = gpu_memory_headroom_evidence(
            gpu_identity,
            peak_device_memory_bytes=peak_device_memory,
        )
    else:
        if linked_proof is None or not isinstance(linked_proof.get("gpu_memory_headroom"), dict):
            raise RuntimeError("speed lane is missing linked proof GPU memory headroom evidence")
        memory_headroom = linked_proof["gpu_memory_headroom"]

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
        "benchmark_mode": benchmark_mode,
        "benchmark_contract": benchmark_contract,
        "benchmark_contract_sha256": benchmark_contract_sha256(benchmark_contract),
        "instrumentation": instrumentation,
        "linked_proof_artifact": linked_proof,
        "proof_status": (
            {
                "passed": True,
                "phase_count": len(phase_results),
                "full_decode_passed": all(result.full_decode_proof["passed"] for result in phase_results),
                "compilation_stable": True,
            }
            if profile.proof
            else {
                "passed": None,
                "attested_by_linked_proof": linked_proof,
            }
        ),
        "versions": runtime_versions(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_provenance": checkpoint_identity,
        "source_provenance": source_identity,
        "vllm_installation_provenance": vllm_identity,
        "gpu_hardware_provenance": gpu_identity,
        "gpu_memory_headroom": memory_headroom,
        "invocation": {
            "argv": [sys.executable, *sys.argv],
            "parsed_args": {
                name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
            },
            "output_artifact_path": str(Path(args.output).resolve()),
            "exit_status": 0,
        },
        "manifest": manifest.to_dict(),
        "manifest_sha256": manifest.sha256,
        "profile": asdict(profile),
        "context_length_preflight": preflight,
        "engine_kwargs": engine_kwargs,
        "resolved_config": resolved,
        "execution_contract": {
            "outer_model": "torch.compile Inductor",
            "prefill": "optimized eager no_compile custom op; packed route proven per worker",
            "decode": "FULL CUDA graph replay required",
            "prefix_caching": profile.shared_prefix_state_reuse,
            "mamba_cache_mode": "align" if profile.shared_prefix_state_reuse else "none",
            "shared_prefix_state_reuse": profile.shared_prefix_state_reuse,
            "global_wave_size": profile.global_wave_size,
            "per_engine_max_num_seqs": profile.resolved_max_num_seqs,
            "gdpo_target_request_count": profile.gdpo_target_batch_size,
            "planned_waves_to_96": profile.gdpo_waves_to_96,
            "semantic_padding": False,
            "benchmark_mode": benchmark_mode,
            "timed_generation_instrumentation": instrumentation,
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
            "context_length_preflight_s": preflight_s,
            "vllm_import_s": vllm_import_s,
            "provenance_hashing_s": provenance_s,
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
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)


def main(argv: list[str] | None = None) -> int:
    """Run the optimized exact-length vLLM benchmark CLI."""
    args = build_parser().parse_args(argv)
    benchmark_mode_from_args(args)
    reservation = reserve_output_namespace(args.output)
    if args.backend != "vllm":
        raise NotImplementedError("the MCore baseline uses its pinned backend adapter")
    source_manifest = load_source_manifest(args)
    manifest = prepare_workload(
        source_manifest,
        request_count=args.request_count,
        uniform_prompt_length=args.uniform_prompt_length,
        request_id_prefix=args.request_id_prefix,
        max_new_tokens=args.max_new_tokens,
    )
    if args.context_preflight_only:
        artifact = run_context_length_preflight(args, manifest)
    elif args.topology == "tp2":
        artifact = run_tp2_benchmark(args, manifest)
    else:
        from bionemo.evo2.vllm.nemo_runner import run_nemo_dp2_benchmark

        artifact = run_nemo_dp2_benchmark(args, manifest)
    write_json_artifact(args.output, artifact)
    complete_output_namespace(reservation, output_path=args.output)
    return 0


__all__ = [
    "CUDAGraphProofRecorder",
    "GenerationPhaseResult",
    "PeakMemoryMonitor",
    "RequestExecutionRecord",
    "build_request_execution_records",
    "build_request_sampling_params",
    "checkpoint_provenance",
    "complete_output_namespace",
    "full_decode_proof_summary",
    "load_source_manifest",
    "phase_output_artifact_path",
    "prepare_workload",
    "request_seed",
    "require_output_namespace_reservation",
    "reserve_output_namespace",
    "reset_vllm_worker_proof_state",
    "run_context_length_preflight",
    "run_generation_phase",
    "runtime_versions",
    "snapshot_vllm_worker_proof_state",
    "source_provenance",
    "summarize_cudagraph_observations",
    "validate_full_decode_proof",
    "write_full_generation_records_artifact",
    "write_full_output_artifact",
    "write_json_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
