# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""End-to-end optimized vLLM benchmark and proof runner for Evo2."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            source_paths.extend(
                path for path in resolved.rglob("*") if path.is_file() and is_durable_source(path)
            )
        else:
            raise FileNotFoundError(f"source provenance root is missing: {resolved}")
    source_records = _file_records(root, source_paths)
    source_tree_sha256 = _records_sha256(source_records)
    git_head = _git_output(root, "rev-parse", "HEAD").strip()
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
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
        "maximum_waiting_requests": max(
            (int(item["num_waiting_requests"]) for item in phase_observations), default=0
        ),
        "maximum_skipped_waiting_requests": max(
            (int(item["num_skipped_waiting_requests"]) for item in phase_observations), default=0
        ),
        "request_count_within_scheduler_ceiling": request_count_within_scheduler_ceiling,
        "running_count_within_scheduler_ceiling": running_count_within_scheduler_ceiling,
        "batch_fit_without_preemption": batch_fit_without_preemption,
    }


def validate_scheduler_capacity_proof(proof: dict[str, Any]) -> None:
    """Fail closed unless one submitted wave fits without preemption or recompute."""
    if int(proof.get("scheduler_observation_count", 0)) <= 0:
        raise AssertionError("no scheduler telemetry was retained for the generation wave")
    if not proof.get("request_count_within_scheduler_ceiling") or not proof.get(
        "running_count_within_scheduler_ceiling"
    ):
        raise AssertionError("generation wave exceeded the per-engine max_num_seqs scheduler ceiling")
    if any(
        int(proof.get(field, 0)) != 0
        for field in ("preemption_events", "recompute_events", "prefix_preempted_requests")
    ):
        raise AssertionError("scheduler preemption/recompute occurred during the generation wave")
    if int(proof.get("prefix_preempted_hits", 0)) > int(proof.get("prefix_preempted_queries", 0)):
        raise AssertionError("scheduler preempted prefix-cache hit telemetry is inconsistent")
    if int(proof.get("preempted_prompt_recomputed_tokens", 0)) != 0:
        raise AssertionError("scheduler recomputed prompt tokens after preemption")
    if proof.get("batch_fit_without_preemption") is not True:
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
    generation_round: int,
    global_request_offset: int,
    dp_rank: int,
    call_index: int,
) -> tuple[RequestExecutionRecord, ...]:
    """Build one ownership/seed record for each exact manifest request."""
    if min(generation_round, global_request_offset, dp_rank, call_index) < 0:
        raise ValueError("execution coordinates must be nonnegative")
    return tuple(
        RequestExecutionRecord(
            request_id=request.request_id,
            global_request_index=global_request_offset + local_index,
            generation_round=generation_round,
            dp_rank=dp_rank,
            call_index=call_index,
            seed=request_seed(
                manifest.seed,
                generation_round=generation_round,
                global_request_index=global_request_offset + local_index,
            ),
        )
        for local_index, request in enumerate(manifest.requests)
    )


def build_wave_execution_records(
    manifest: WorkloadManifest,
    *,
    generation_round: int,
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
                generation_round=generation_round,
                global_request_offset=wave.start,
                dp_rank=0,
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
        and (
            candidate.name.endswith(".outputs.jsonl.gz")
            or candidate.name.endswith(".outputs.jsonl.gz.tmp")
        )
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
            retained_requests.append(dict(record))

        worker_clones.append(
            {
                "rank": int(proof.get("rank", 0)),
                "device": int(proof.get("device", 0)),
                "clone_count": clone_count,
                "requests": retained_requests,
            }
        )

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
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
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
    full_decode_proof: dict[str, Any]
    worker_proof: tuple[dict[str, Any], ...]
    shared_prefix_state_reuse: dict[str, Any] | None

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

    observation_start = len(recorder.observations)
    if reset_worker_proof is not None:
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
    with memory_monitor_factory() as monitor:
        for wave in waves:
            wave_phase = f"{phase}.wave-{wave.wave_index:03d}"
            wave_manifest = manifest.request_slice(wave.start, wave.stop)
            wave_prompts = [
                {"prompt_token_ids": list(request.prompt_token_ids)} for request in wave_manifest.requests
            ]
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
                max_num_seqs=(
                    wave.request_count if scheduler_max_num_seqs is None else scheduler_max_num_seqs
                ),
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
    worker_proof = () if snapshot_worker_proof is None else snapshot_worker_proof()
    shared_prefix_reuse = None
    if require_shared_prefix_state_reuse:
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
        observations=tuple(recorder.observations[observation_start:]),
        output_summaries=output_summaries,
        request_executions=execution_records,
        full_output_artifact=full_output_artifact,
        full_decode_proof=full_decode_proof,
        worker_proof=worker_proof,
        shared_prefix_state_reuse=shared_prefix_reuse,
    )


def reset_vllm_worker_proof_state(worker: Any) -> dict[str, int]:
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
    reset_mamba_prefix_clone_stats()
    return {"rank": int(rank), "device": int(torch.cuda.current_device())}


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
    return {
        "rank": int(rank),
        "device": int(device),
        "device_name": torch.cuda.get_device_name(device),
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
    source_identity = source_provenance()
    provenance_s = time.perf_counter() - provenance_begin

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
    call_index_start = 0
    for sample_index, (phase, round_offset) in enumerate(_phase_specs(args.warmups, args.repetitions)):
        generation_round = args.generation_round + round_offset
        sampling_params = build_request_sampling_params(
            manifest,
            sampling_params_factory=SamplingParams,
            generation_round=generation_round,
            global_request_offset=0,
        )
        execution_records = build_wave_execution_records(
            manifest,
            generation_round=generation_round,
            global_wave_size=profile.global_wave_size,
            call_index_start=call_index_start,
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
            reset_worker_proof=lambda: _reset_worker_proof(llm),
            snapshot_worker_proof=lambda: _snapshot_worker_proof(llm),
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
        "checkpoint_provenance": checkpoint_identity,
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
