# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Strict adapters between Evo2 benchmark manifests and production NeMo-RL generation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from bionemo.evo2.vllm.benchmark import (
    BenchmarkSample,
    GenerationRecord,
    WorkloadManifest,
    aggregate_samples,
    build_request_waves,
    validate_compilation_proof,
)
from bionemo.evo2.vllm.profile import Evo2VllmProfile, context_length_preflight, validate_resolved_profile
from bionemo.evo2.vllm.runner import (
    PeakMemoryMonitor,
    RequestExecutionRecord,
    checkpoint_provenance,
    full_decode_proof_summary,
    make_nvml_memory_reader,
    phase_output_artifact_path,
    request_seed,
    require_output_namespace_reservation,
    runtime_versions,
    source_provenance,
    validate_full_decode_proof,
    write_full_generation_records_artifact,
)


_OUTPUT_METADATA_KEYS = (
    "generation_request_seeds",
    "generation_global_request_indices",
    "generation_rounds",
    "generation_call_indices",
    "generation_dp_ranks",
)


def build_nemo_generation_config(
    profile: Evo2VllmProfile,
    manifest: WorkloadManifest,
    *,
    checkpoint: Path,
    load_format: str,
    num_logprobs: int = 0,
) -> dict[str, Any]:
    """Build a complete NeMo-RL vLLM config for one exact workload."""
    if num_logprobs < 0:
        raise ValueError("num_logprobs must be nonnegative")
    if profile.max_model_len < manifest.max_total_tokens:
        raise ValueError(
            f"profile max_model_len={profile.max_model_len} is smaller than "
            f"workload max_total_tokens={manifest.max_total_tokens}"
        )

    config = profile.nemo_rl_generation_config(
        load_format=load_format,
        request_seed=manifest.seed,
    )
    config.update(
        {
            "model_name": str(checkpoint.resolve()),
            "dtype": manifest.dtype,
            "max_new_tokens": manifest.max_new_tokens,
            "temperature": manifest.temperature,
            "top_p": manifest.top_p,
            "top_k": manifest.top_k,
            "stop_token_ids": list(manifest.stop_token_ids),
            "stop_strings": None,
            "ignore_eos": manifest.ignore_eos,
            "_pad_token_id": 0,
            "colocated": {
                "enabled": False,
                "resources": {"gpus_per_node": 2, "num_nodes": 1},
            },
        }
    )
    if num_logprobs:
        config["num_logprobs"] = num_logprobs
        config["vllm_kwargs"]["max_logprobs"] = num_logprobs
    return config


def build_nemo_generation_input(manifest: WorkloadManifest) -> BatchedDataDict:
    """Right-pad prompts for transport while preserving every semantic length."""
    prompt_lengths = torch.tensor(
        [len(request.prompt_token_ids) for request in manifest.requests],
        dtype=torch.long,
    )
    max_prompt_length = int(prompt_lengths.max().item())
    input_ids = torch.zeros(
        (len(manifest.requests), max_prompt_length),
        dtype=torch.long,
    )
    for row_index, request in enumerate(manifest.requests):
        prompt = torch.tensor(request.prompt_token_ids, dtype=torch.long)
        input_ids[row_index, : len(prompt)] = prompt
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": prompt_lengths,
        }
    )


def _require_aligned_tensor(
    outputs: BatchedDataDict,
    key: str,
    *,
    request_count: int,
) -> torch.Tensor:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"NeMo-RL generation output {key!r} must be a tensor")
    if value.ndim != 1 or value.shape[0] != request_count:
        raise ValueError(
            f"NeMo-RL generation output {key!r} must have shape ({request_count},), got {tuple(value.shape)}"
        )
    return value.detach().cpu()


def records_from_nemo_generation_output(
    manifest: WorkloadManifest,
    outputs: BatchedDataDict,
) -> tuple[
    tuple[GenerationRecord, ...],
    tuple[RequestExecutionRecord, ...],
    dict[str, tuple[float, ...]],
]:
    """Validate and retain exact outputs, RNG ownership, and timing from NeMo-RL."""
    request_count = len(manifest.requests)
    output_ids = outputs.get("output_ids")
    logprobs = outputs.get("logprobs")
    if not isinstance(output_ids, torch.Tensor) or output_ids.ndim != 2:
        raise TypeError("NeMo-RL generation output 'output_ids' must be a rank-2 tensor")
    if not isinstance(logprobs, torch.Tensor) or logprobs.shape != output_ids.shape:
        raise ValueError("NeMo-RL generation logprobs must match output_ids exactly")
    if output_ids.shape[0] != request_count:
        raise ValueError(f"NeMo-RL returned {output_ids.shape[0]} outputs for {request_count} requests")
    output_ids = output_ids.detach().cpu()
    logprobs = logprobs.detach().cpu()

    generation_lengths = _require_aligned_tensor(
        outputs,
        "generation_lengths",
        request_count=request_count,
    )
    expected_generation_lengths = torch.full_like(
        generation_lengths,
        manifest.max_new_tokens,
    )
    if not torch.equal(generation_lengths, expected_generation_lengths):
        raise ValueError(
            "NeMo-RL generation did not return the requested exact output length: "
            f"expected={manifest.max_new_tokens}, actual={generation_lengths.tolist()}"
        )

    unpadded_lengths = _require_aligned_tensor(
        outputs,
        "unpadded_sequence_lengths",
        request_count=request_count,
    )
    truncated = _require_aligned_tensor(outputs, "truncated", request_count=request_count)
    if not bool(torch.all(truncated)):
        raise ValueError("exact-length generation must finish every request by max-token truncation")

    metadata = {
        key: _require_aligned_tensor(outputs, key, request_count=request_count) for key in _OUTPUT_METADATA_KEYS
    }
    ttft = _require_aligned_tensor(
        outputs,
        "generation_first_token_latency_s",
        request_count=request_count,
    ).to(torch.float64)
    decode = _require_aligned_tensor(
        outputs,
        "generation_decode_s",
        request_count=request_count,
    ).to(torch.float64)
    if not bool(torch.all(torch.isfinite(ttft))) or not bool(torch.all(ttft >= 0)):
        raise ValueError("first-token latency must be finite and nonnegative")
    if not bool(torch.all(torch.isfinite(decode))) or not bool(torch.all(decode >= 0)):
        raise ValueError("decode timing must be finite and nonnegative")

    records = []
    executions = []
    for row_index, request in enumerate(manifest.requests):
        prompt_length = len(request.prompt_token_ids)
        total_length = prompt_length + manifest.max_new_tokens
        if int(unpadded_lengths[row_index]) != total_length:
            raise ValueError(
                f"request {request.request_id} has unpadded length "
                f"{int(unpadded_lengths[row_index])}, expected {total_length}"
            )
        if output_ids.shape[1] < total_length:
            raise ValueError(
                f"request {request.request_id} output width {output_ids.shape[1]} "
                f"is shorter than exact length {total_length}"
            )
        actual_prompt = tuple(int(token) for token in output_ids[row_index, :prompt_length])
        if actual_prompt != request.prompt_token_ids:
            raise ValueError(f"request {request.request_id} prompt tokens changed during generation")

        generated_ids = tuple(int(token) for token in output_ids[row_index, prompt_length:total_length])
        generated_logprobs = tuple(float(value) for value in logprobs[row_index, prompt_length:total_length])
        if not all(math.isfinite(value) for value in generated_logprobs):
            raise ValueError(f"request {request.request_id} has non-finite chosen-token logprobs")
        records.append(
            GenerationRecord(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                output_token_ids=generated_ids,
                output_logprobs=generated_logprobs,
                requested_max_tokens=manifest.max_new_tokens,
                finish_reason="length",
                stop_reason=None,
                stopped_on_eos=False,
            )
        )
        executions.append(
            RequestExecutionRecord(
                request_id=request.request_id,
                global_request_index=int(metadata["generation_global_request_indices"][row_index]),
                generation_round=int(metadata["generation_rounds"][row_index]),
                dp_rank=int(metadata["generation_dp_ranks"][row_index]),
                call_index=int(metadata["generation_call_indices"][row_index]),
                seed=int(metadata["generation_request_seeds"][row_index]),
            )
        )

    global_indices = [record.global_request_index for record in executions]
    seeds = [record.seed for record in executions]
    if len(global_indices) != len(set(global_indices)):
        raise ValueError("NeMo-RL returned duplicate global request ownership")
    if len(seeds) != len(set(seeds)):
        raise ValueError("NeMo-RL returned duplicate per-request seeds")

    return (
        tuple(records),
        tuple(executions),
        {
            "ttft_s": tuple(float(value) for value in ttft),
            "decode_s": tuple(float(value) for value in decode),
        },
    )


def full_vocab_logprob_evidence_from_nemo_output(
    outputs: BatchedDataDict,
    *,
    records: tuple[GenerationRecord, ...],
    require_full: bool,
    expected_finite_support: int | None = None,
) -> dict[str, Any]:
    """Validate and retain per-step processed-logprob distributions."""
    dense = outputs.get("generation_vocab_logprobs")
    counts = outputs.get("generation_logprob_counts")
    if not isinstance(dense, torch.Tensor) or dense.ndim != 3:
        raise TypeError("NeMo-RL full-vocabulary logprobs must be a rank-3 tensor")
    if not isinstance(counts, torch.Tensor) or counts.shape != dense.shape[:2]:
        raise ValueError("NeMo-RL logprob coverage counts must align with every generated position")
    if dense.shape[0] != len(records):
        raise ValueError("NeMo-RL full-vocabulary logprobs must align with every output record")

    dense = dense.detach().cpu().to(torch.float32)
    counts = counts.detach().cpu().to(torch.long)
    vocab_size = dense.shape[2]
    if expected_finite_support is None and require_full:
        expected_finite_support = vocab_size
    if expected_finite_support is not None and not 1 <= expected_finite_support <= vocab_size:
        raise ValueError("expected finite logprob support must be within the model vocabulary")

    generation_lengths = [len(record.output_token_ids) for record in records]
    if any(dense.shape[1] < length for length in generation_lengths):
        raise ValueError("NeMo-RL full-vocabulary evidence is shorter than its generated output")
    finite_count_tensor = torch.isfinite(dense).sum(dim=-1)
    valid_coordinates = [
        (row_index, step)
        for row_index, generation_length in enumerate(generation_lengths)
        for step in range(generation_length)
    ]
    reported_values = [int(counts[row_index, step]) for row_index, step in valid_coordinates]
    finite_values = [int(finite_count_tensor[row_index, step]) for row_index, step in valid_coordinates]

    if expected_finite_support == vocab_size and reported_values != finite_values:
        mismatch_coordinates = [
            (row_index, step, reported, finite)
            for (row_index, step), reported, finite in zip(
                valid_coordinates,
                reported_values,
                finite_values,
                strict=True,
            )
            if reported != finite
        ]
        row_index, step, reported, finite = mismatch_coordinates[0]
        raise AssertionError(
            f"shape={list(dense.shape)}, "
            f"mismatched_positions={len(mismatch_coordinates)}/{len(valid_coordinates)}, "
            f"reported_counts={dict(sorted(Counter(reported_values).items()))}, "
            f"finite_counts={dict(sorted(Counter(finite_values).items()))}, "
            "first_mismatch=("
            f"request={row_index}, step={step}, reported={reported}, finite={finite})"
        )

    retained: list[list[list[float | None]]] = []
    retained_counts: list[list[int]] = []
    retained_finite_counts: list[list[int]] = []
    retained_negative_infinity_counts: list[list[int]] = []
    chosen_token_logprobs: list[list[float]] = []
    for row_index, record in enumerate(records):
        generation_length = len(record.output_token_ids)
        row = dense[row_index, :generation_length]
        row_counts = counts[row_index, :generation_length]
        finite_counts = torch.isfinite(row).sum(dim=-1)
        if bool(torch.any(row_counts <= 0)) or bool(torch.any(row_counts > vocab_size)):
            raise AssertionError("logprob coverage count is outside the model vocabulary")
        if require_full and not bool(torch.all(row_counts == vocab_size)):
            raise AssertionError("full-vocabulary evidence omitted one or more token logprobs")
        if bool(torch.any(torch.isnan(row))) or bool(torch.any(torch.isposinf(row))):
            raise AssertionError("processed logprob evidence contains NaN or positive infinity")
        if expected_finite_support is not None and not bool(
            torch.all(finite_counts == expected_finite_support)
        ):
            raise AssertionError(
                "processed logprob finite support does not match the configured sampling policy: "
                f"expected={expected_finite_support}, actual={finite_counts.tolist()}"
            )

        row_chosen_logprobs = []
        for position, (token_id, chosen_logprob) in enumerate(
            zip(record.output_token_ids, record.output_logprobs, strict=True)
        ):
            if not 0 <= token_id < vocab_size:
                raise AssertionError(f"generated token {token_id} is outside vocabulary size {vocab_size}")
            distribution_logprob = float(row[position, token_id])
            if not math.isfinite(distribution_logprob) or not math.isfinite(chosen_logprob):
                raise AssertionError("chosen token is outside finite processed support")
            if not math.isclose(
                distribution_logprob,
                chosen_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise AssertionError(
                    "chosen-token logprob does not match its full-vocabulary distribution entry"
                )
            row_chosen_logprobs.append(chosen_logprob)

        retained.append(
            [
                [float(value) if math.isfinite(float(value)) else None for value in position]
                for position in row
            ]
        )
        retained_counts.append([int(value) for value in row_counts])
        retained_finite_counts.append([int(value) for value in finite_counts])
        retained_negative_infinity_counts.append(
            [int(value) for value in torch.isneginf(row).sum(dim=-1)]
        )
        chosen_token_logprobs.append(row_chosen_logprobs)

    return {
        "shape": [len(records), dense.shape[1], vocab_size],
        "coverage_counts": retained_counts,
        "finite_support_counts": retained_finite_counts,
        "negative_infinity_counts": retained_negative_infinity_counts,
        "expected_finite_support": expected_finite_support,
        "chosen_token_oracle_passed": True,
        "chosen_token_in_finite_support": True,
        "chosen_token_logprobs": chosen_token_logprobs,
        "logprobs": retained,
    }


@dataclass(frozen=True)
class NemoGenerationPhaseResult:
    """One exact production NeMo-RL generation phase across all request waves."""

    phase: str
    sample: BenchmarkSample
    generation_call_s: tuple[float, ...]
    output_summaries: tuple[dict[str, Any], ...]
    request_executions: tuple[RequestExecutionRecord, ...]
    full_output_artifact: dict[str, Any]
    wave_proofs: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe production phase proof."""
        return {
            "phase": self.phase,
            "sample": self.sample.to_dict(),
            "generation_call_s": list(self.generation_call_s),
            "outputs": list(self.output_summaries),
            "request_executions": [record.to_dict() for record in self.request_executions],
            "full_output_artifact": self.full_output_artifact,
            "waves": list(self.wave_proofs),
        }


def _proof_rpc(
    generation: Any,
    method_name: str,
    *,
    phase: str,
    ray_get: Any,
) -> list[dict[str, Any]]:
    futures = generation.worker_group.run_all_workers_single_data(
        method_name,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        phase=phase,
    )
    return list(ray_get(futures))


def _validate_wave_execution(
    manifest: WorkloadManifest,
    executions: tuple[RequestExecutionRecord, ...],
    *,
    expected_dp_ranks: tuple[int, ...],
    expected_global_start: int,
    expected_call_index: int | None,
) -> int:
    if len(executions) != len(manifest.requests):
        raise AssertionError("execution metadata must cover every wave request")
    actual_ids = tuple(record.request_id for record in executions)
    expected_ids = tuple(request.request_id for request in manifest.requests)
    if actual_ids != expected_ids:
        raise AssertionError("NeMo-RL output ownership changed request order")
    if tuple(record.dp_rank for record in executions) != expected_dp_ranks:
        raise AssertionError("NeMo-RL DP ownership does not match exact shard boundaries")
    if tuple(record.global_request_index for record in executions) != tuple(
        range(expected_global_start, expected_global_start + len(executions))
    ):
        raise AssertionError("NeMo-RL global request indices are not exact and contiguous")

    call_indices = {record.call_index for record in executions}
    generation_rounds = {record.generation_round for record in executions}
    if len(call_indices) != 1 or generation_rounds != call_indices:
        raise AssertionError("each production generation call must own one persisted generation round")
    call_index = next(iter(call_indices))
    if expected_call_index is not None and call_index != expected_call_index:
        raise AssertionError("successive production generation calls did not advance exactly once")
    for record in executions:
        expected_seed = request_seed(
            manifest.seed,
            generation_round=record.generation_round,
            global_request_index=record.global_request_index,
        )
        if record.seed != expected_seed:
            raise AssertionError("persisted NeMo-RL request seed does not match its RNG coordinates")
    return call_index


def run_nemo_generation_phase(
    *,
    generation: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    phase: str,
    sample_index: int,
    full_output_path: str | Path,
    memory_monitor_factory: Any,
    ray_get: Any | None = None,
    clock: Any | None = None,
    greedy: bool = False,
    require_full_vocab_logprobs: bool = False,
    expected_finite_logprob_support: int | None = None,
) -> NemoGenerationPhaseResult:
    """Run exact NeMo-RL waves with per-engine graph, route, seed, and ownership proof."""
    if not phase:
        raise ValueError("phase cannot be empty")
    if ray_get is None:
        import ray

        ray_get = ray.get
    if clock is None:
        import time

        clock = time.perf_counter

    waves = build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=profile.global_batch_size,
        replica_count=profile.replica_count,
    )
    all_records: list[GenerationRecord] = []
    all_executions: list[RequestExecutionRecord] = []
    all_ttft: list[float] = []
    all_decode: list[float] = []
    generation_call_s = []
    wave_proofs = []
    phase_global_start: int | None = None
    previous_call_index: int | None = None

    with memory_monitor_factory() as monitor:
        for wave in waves:
            wave_manifest = manifest.request_slice(wave.start, wave.stop)
            wave_phase = f"{phase}.wave-{wave.wave_index:03d}"
            reset_proof = _proof_rpc(
                generation,
                "reset_evo2_proof_phase",
                phase=wave_phase,
                ray_get=ray_get,
            )
            begin = clock()
            outputs = generation.generate(
                build_nemo_generation_input(wave_manifest),
                greedy=greedy,
            )
            generation_call_s.append(clock() - begin)
            engine_proofs = _proof_rpc(
                generation,
                "snapshot_evo2_proof_phase",
                phase=wave_phase,
                ray_get=ray_get,
            )
            records, executions, timings = records_from_nemo_generation_output(wave_manifest, outputs)
            full_vocab_logprobs = (
                full_vocab_logprob_evidence_from_nemo_output(
                    outputs,
                    records=records,
                    require_full=True,
                    expected_finite_support=expected_finite_logprob_support,
                )
                if require_full_vocab_logprobs
                else None
            )

            expected_dp_ranks = tuple(shard.replica_index for shard in wave.shards for _ in range(shard.request_count))
            if phase_global_start is None:
                phase_global_start = executions[0].global_request_index - wave.start
            call_index = _validate_wave_execution(
                wave_manifest,
                executions,
                expected_dp_ranks=expected_dp_ranks,
                expected_global_start=phase_global_start + wave.start,
                expected_call_index=None if previous_call_index is None else previous_call_index + 1,
            )
            previous_call_index = call_index

            if len(engine_proofs) != len(wave.shards):
                raise AssertionError(
                    f"wave {wave.wave_index} returned {len(engine_proofs)} engine proofs for "
                    f"{len(wave.shards)} active DP replicas"
                )
            validated_engine_proofs = []
            for shard, engine_proof in zip(wave.shards, engine_proofs, strict=True):
                if engine_proof.get("phase") != wave_phase:
                    raise AssertionError("engine proof phase does not match its request wave")
                observations = list(engine_proof.get("cudagraph_observations", ()))
                proof_summary = full_decode_proof_summary(
                    observations,
                    phase=wave_phase,
                    batch_size=shard.request_count,
                    max_new_tokens=manifest.max_new_tokens,
                )
                if profile.proof:
                    validate_full_decode_proof(
                        observations,
                        phase=wave_phase,
                        batch_size=shard.request_count,
                        max_new_tokens=manifest.max_new_tokens,
                    )
                validated_engine_proofs.append(
                    {
                        "dp_rank": shard.replica_index,
                        "request_count": shard.request_count,
                        "full_decode_proof": proof_summary,
                        **engine_proof,
                    }
                )

            wave_proofs.append(
                {
                    "wave_index": wave.wave_index,
                    "phase": wave_phase,
                    "start": wave.start,
                    "stop": wave.stop,
                    "request_count": wave.request_count,
                    "generation_call_s": generation_call_s[-1],
                    "reset_proof": reset_proof,
                    "engines": validated_engine_proofs,
                    "full_vocab_logprobs": full_vocab_logprobs,
                }
            )
            all_records.extend(records)
            all_executions.extend(executions)
            all_ttft.extend(timings["ttft_s"])
            all_decode.extend(timings["decode_s"])

    if tuple(record.request_id for record in all_records) != tuple(
        request.request_id for request in manifest.requests
    ):
        raise AssertionError("production waves did not return every manifest request exactly once")
    global_indices = [record.global_request_index for record in all_executions]
    seeds = [record.seed for record in all_executions]
    if len(global_indices) != len(set(global_indices)) or len(seeds) != len(set(seeds)):
        raise AssertionError("production phase has duplicate request ownership or RNG streams")

    output_lengths = tuple(len(record.output_token_ids) for record in all_records)
    sample = BenchmarkSample(
        sample_index=sample_index,
        generation_s=sum(generation_call_s),
        request_count=len(all_records),
        prompt_tokens=sum(len(record.prompt_token_ids) for record in all_records),
        generated_tokens=sum(output_lengths),
        ttft_s=tuple(all_ttft),
        inter_token_latency_s=(
            tuple(
                decode_s / (output_length - 1)
                for decode_s, output_length in zip(all_decode, output_lengths, strict=True)
            )
            if manifest.max_new_tokens > 1
            else ()
        ),
        output_lengths=output_lengths,
        peak_device_memory_bytes=monitor.peak_device_memory_bytes,
    )
    full_output_artifact = write_full_generation_records_artifact(
        full_output_path,
        records=all_records,
        execution_records=all_executions,
    )
    return NemoGenerationPhaseResult(
        phase=phase,
        sample=sample,
        generation_call_s=tuple(generation_call_s),
        output_summaries=tuple(record.summary_dict() for record in all_records),
        request_executions=tuple(all_executions),
        full_output_artifact=full_output_artifact,
        wave_proofs=tuple(wave_proofs),
    )


def _inner_worker_proofs(engine_proof: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    workers = engine_proof.get("worker_proof")
    if not isinstance(workers, list) or not workers:
        raise AssertionError("production engine proof is missing internal vLLM worker evidence")
    return tuple(workers)


def run_nemo_dp2_benchmark(args: Any, manifest: WorkloadManifest) -> dict[str, Any]:
    """Launch the actual NeMo-RL TP1/DP2 path and retain exact production evidence."""
    if args.topology != "dp2":
        raise ValueError("run_nemo_dp2_benchmark requires topology=dp2")
    if args.generation_round != 0:
        raise ValueError("production NeMo-RL generation rounds advance from zero; use --generation-round=0")

    clock = __import__("time").perf_counter
    output_path = Path(args.output).resolve()
    require_output_namespace_reservation(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = Evo2VllmProfile(
        topology="dp2",
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
    preflight_begin = clock()
    preflight = context_length_preflight(
        profile,
        model=args.checkpoint,
        workload_max_total_tokens=manifest.max_total_tokens,
        load_format=args.load_format,
    )
    preflight_s = clock() - preflight_begin

    import_begin = __import__("time").perf_counter()
    import nemo_rl
    import ray
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration

    import_s = __import__("time").perf_counter() - import_begin
    provenance_begin = clock()
    checkpoint_identity = checkpoint_provenance(args.checkpoint)
    bionemo_source_identity = source_provenance()
    nemo_package = Path(nemo_rl.__file__).resolve().parent
    nemo_source_identity = source_provenance(
        repository=nemo_package.parent,
        source_roots=(
            nemo_package / "models/generation/interfaces.py",
            nemo_package / "models/generation/vllm/config.py",
            nemo_package / "models/generation/vllm/vllm_generation.py",
            nemo_package / "models/generation/vllm/vllm_worker.py",
        ),
    )
    provenance_s = clock() - provenance_begin

    config = build_nemo_generation_config(
        profile,
        manifest,
        checkpoint=args.checkpoint,
        load_format=args.load_format,
    )
    memory_reader = make_nvml_memory_reader()

    ray_dir_suffix = __import__("hashlib").sha256(str(output_path).encode()).hexdigest()[:10]
    ray_log_dir = Path("/tmp") / f"e2ray-{ray_dir_suffix}"
    ray_log_dir.mkdir(parents=True, exist_ok=True)

    cluster = None
    generation = None
    try:
        ray_begin = clock()
        init_ray(log_dir=str(ray_log_dir))
        ray_init_s = clock() - ray_begin
        cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[2],
            use_gpus=True,
            max_colocated_worker_groups=1,
            num_gpus_per_node=2,
            name=f"evo2-vllm-dp2-{output_path.stem}",
        )
        engine_begin = clock()
        with PeakMemoryMonitor(memory_reader) as init_memory:
            generation = VllmGeneration(
                cluster,
                config,
                name_prefix=f"evo2_vllm_dp2_{output_path.stem}",
            )
        engine_init_s = clock() - engine_begin

        initialized_phase = "engine-initialized"
        initialized_reset = _proof_rpc(
            generation,
            "reset_evo2_proof_phase",
            phase=initialized_phase,
            ray_get=ray.get,
        )
        initialized_proofs = _proof_rpc(
            generation,
            "snapshot_evo2_proof_phase",
            phase=initialized_phase,
            ray_get=ray.get,
        )
        if len(initialized_proofs) != profile.replica_count:
            raise AssertionError("NeMo-RL did not launch exactly two independent DP engines")
        for proof in initialized_proofs:
            validate_resolved_profile(profile, proof["resolved_config"])

        phase_specs = (
            ("cold-generation", 0),
            *((f"warmup-{index}", index + 1) for index in range(args.warmups)),
            *((f"steady-{index}", args.warmups + index + 1) for index in range(args.repetitions)),
        )
        phase_results = []
        for sample_index, (phase, _) in enumerate(phase_specs):
            phase_results.append(
                run_nemo_generation_phase(
                    generation=generation,
                    manifest=manifest,
                    profile=profile,
                    phase=phase,
                    sample_index=sample_index,
                    full_output_path=phase_output_artifact_path(output_path, phase=phase),
                    memory_monitor_factory=lambda: PeakMemoryMonitor(memory_reader),
                    ray_get=ray.get,
                    clock=clock,
                )
            )

        final_engine_proofs = phase_results[-1].wave_proofs[-1]["engines"]
        for initialized_engine, final_engine in zip(
            initialized_proofs,
            final_engine_proofs,
            strict=True,
        ):
            initialized_workers = _inner_worker_proofs(initialized_engine)
            final_workers = _inner_worker_proofs(final_engine)
            if len(initialized_workers) != len(final_workers):
                raise AssertionError("internal vLLM worker topology changed during generation")
            for initialized_worker, final_worker in zip(
                initialized_workers,
                final_workers,
                strict=True,
            ):
                validate_compilation_proof(
                    initialized_worker["compilation"],
                    final_worker["compilation"],
                )

        steady_results = [result for result in phase_results if result.phase.startswith("steady-")]
        return {
            "schema_version": 1,
            "backend": "nemo-rl-vllm",
            "topology": "dp2",
            "versions": runtime_versions(),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_provenance": checkpoint_identity,
            "source_provenance": {
                "bionemo": bionemo_source_identity,
                "nemo_rl": nemo_source_identity,
            },
            "invocation": {
                "argv": [__import__("sys").executable, *__import__("sys").argv],
                "parsed_args": {
                    name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
                },
                "output_artifact_path": str(output_path),
                "ray_log_dir": str(ray_log_dir),
                "exit_status": 0,
            },
            "manifest": manifest.to_dict(),
            "manifest_sha256": manifest.sha256,
            "profile": asdict(profile),
            "context_length_preflight": preflight,
            "nemo_generation_config": config,
            "resolved_configs": [proof["resolved_config"] for proof in initialized_proofs],
            "execution_contract": {
                "production_path": "nemo_rl.models.generation.vllm.VllmGeneration",
                "replicas": 2,
                "tensor_parallel_size_per_replica": 1,
                "global_batch_size": profile.global_batch_size,
                "per_engine_batch_size": profile.per_engine_batch_size,
                "semantic_padding": False,
                "prefix_caching": False,
                "shared_prefix_state_reuse": False,
            },
            "timing": {
                "context_length_preflight_s": preflight_s,
                "imports_s": import_s,
                "provenance_hashing_s": provenance_s,
                "ray_init_s": ray_init_s,
                "engine_init_s": engine_init_s,
                "engine_init_peak_device_memory_bytes": list(init_memory.peak_device_memory_bytes),
            },
            "initialized_reset": initialized_reset,
            "initialized_engine_proofs": initialized_proofs,
            "phases": [result.to_dict() for result in phase_results],
            "steady_aggregate": aggregate_samples([result.sample for result in steady_results]),
        }
    finally:
        if generation is not None:
            generation.shutdown()
        if cluster is not None:
            cluster.shutdown()
        ray.shutdown()


__all__ = [
    "NemoGenerationPhaseResult",
    "build_nemo_generation_config",
    "build_nemo_generation_input",
    "full_vocab_logprob_evidence_from_nemo_output",
    "records_from_nemo_generation_output",
    "run_nemo_dp2_benchmark",
    "run_nemo_generation_phase",
]
