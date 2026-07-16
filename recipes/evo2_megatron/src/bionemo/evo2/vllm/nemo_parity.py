# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Production NeMo-RL TP2 parity and checkpoint-refit proof helpers."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import torch

from bionemo.evo2.vllm.benchmark import WorkloadManifest, WorkloadRequest, build_request_waves
from bionemo.evo2.vllm.refit import (
    IndexedSafetensorsLayout,
    IpcChunkPlan,
    stream_indexed_checkpoint_to_device,
    validate_refit_proof,
)


def build_duplicate_prompt_manifest(
    manifest: WorkloadManifest,
    *,
    request_count: int,
    request_id_prefix: str,
) -> WorkloadManifest:
    """Repeat one exact prompt under unique request IDs for placement parity."""
    if len(manifest.requests) != 1:
        raise ValueError("duplicate-prompt parity requires exactly one source request")
    if request_count < 1:
        raise ValueError("request_count must be positive")
    if not request_id_prefix:
        raise ValueError("request_id_prefix cannot be empty")
    prompt = manifest.requests[0].prompt_token_ids
    requests = tuple(
        WorkloadRequest(
            request_id=f"{request_id_prefix}-{index:03d}",
            prompt_token_ids=prompt,
        )
        for index in range(request_count)
    )
    return replace(
        manifest,
        name=f"{manifest.name}-{request_id_prefix}-n{request_count}",
        requests=requests,
    )


def compare_full_vocab_evidence(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare every retained request, continuation step, and vocabulary token."""
    if not reference.get("chosen_token_oracle_passed") or not candidate.get("chosen_token_oracle_passed"):
        raise AssertionError("full-vocabulary evidence did not pass its chosen-token oracle")
    if reference.get("shape") != candidate.get("shape"):
        raise AssertionError("full-vocabulary evidence shapes differ")
    if reference.get("coverage_counts") != candidate.get("coverage_counts"):
        raise AssertionError("full-vocabulary evidence coverage differs")

    reference_tensor = torch.tensor(reference.get("logprobs"), dtype=torch.float64)
    candidate_tensor = torch.tensor(candidate.get("logprobs"), dtype=torch.float64)
    expected_shape = tuple(int(value) for value in reference["shape"])
    if tuple(reference_tensor.shape) != expected_shape or tuple(candidate_tensor.shape) != expected_shape:
        raise AssertionError("retained logprob arrays do not match their declared shape")
    if not bool(torch.all(torch.isfinite(reference_tensor))) or not bool(torch.all(torch.isfinite(candidate_tensor))):
        raise AssertionError("full-vocabulary evidence contains non-finite values")

    errors = torch.abs(reference_tensor - candidate_tensor)
    worst_flat_index = int(torch.argmax(errors).item())
    request_index, step, token_id = (
        int(value) for value in torch.unravel_index(torch.tensor(worst_flat_index), errors.shape)
    )
    return {
        "shape": list(expected_shape),
        "max_abs_logprob_error": float(errors.max()),
        "mean_abs_logprob_error": float(errors.mean()),
        "p95_abs_logprob_error": float(torch.quantile(errors.flatten(), 0.95)),
        "worst_coordinate": {
            "request_index": request_index,
            "step": step,
            "token_id": token_id,
        },
        "top1_identity": bool(
            torch.equal(torch.argmax(reference_tensor, dim=-1), torch.argmax(candidate_tensor, dim=-1))
        ),
    }


def _flatten_tp_device_uuids(generation: Any) -> list[str]:
    nested = generation.device_uuids
    if not isinstance(nested, list) or len(nested) != 1 or not isinstance(nested[0], list):
        raise AssertionError("TP2 generation must expose one engine-local device UUID list")
    device_uuids = [str(value) for value in nested[0]]
    if len(device_uuids) != 2 or len(set(device_uuids)) != 2:
        raise AssertionError("TP2 generation must expose two distinct device UUIDs")
    return device_uuids


def run_production_refit(
    *,
    generation: Any,
    layout: IndexedSafetensorsLayout,
    plan: IpcChunkPlan,
    phase: str,
    device_index_by_uuid: dict[str, int],
    ray_get: Callable[[Any], Any],
    memory_monitor_factory: Callable[[], Any],
    stream_fn: Callable[..., dict[str, Any]] = stream_indexed_checkpoint_to_device,
    clock: Callable[[], float] = time.perf_counter,
    ready_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run one exact two-rank refit through NeMo-RL's production IPC contract."""
    if not phase:
        raise ValueError("refit phase cannot be empty")
    device_uuids = _flatten_tp_device_uuids(generation)
    if set(device_index_by_uuid) != set(device_uuids):
        raise AssertionError("local refit producer devices do not match generation TP devices")

    reset_futures = generation.worker_group.run_all_workers_single_data(
        "reset_evo2_refit_phase",
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        phase=phase,
    )
    reset_results = list(ray_get(reset_futures))

    prepare_begin = clock()
    generation.prepare_refit_info(layout.state_dict_info)
    prepare_refit_s = clock() - prepare_begin

    start_event = threading.Event()
    ready_events = [threading.Event() for _ in device_uuids]
    producer_results: list[dict[str, Any] | None] = [None] * len(device_uuids)
    producer_errors: list[BaseException | None] = [None] * len(device_uuids)

    def run_producer(index: int, device_uuid: str) -> None:
        try:
            producer_results[index] = stream_fn(
                layout,
                device_index=device_index_by_uuid[device_uuid],
                expected_device_uuid=device_uuid,
                buffer_size_bytes=plan.buffer_size_bytes,
                ready_event=ready_events[index],
                start_event=start_event,
            )
        except BaseException as error:
            producer_errors[index] = error
            ready_events[index].set()

    threads = [
        threading.Thread(
            target=run_producer,
            args=(index, device_uuid),
            name=f"evo2-refit-{phase}-tp{index}",
            daemon=False,
        )
        for index, device_uuid in enumerate(device_uuids)
    ]
    for thread in threads:
        thread.start()
    for event in ready_events:
        if not event.wait(ready_timeout_s):
            start_event.set()
            raise TimeoutError("timed out waiting for TP refit producer socket")
    if any(error is not None for error in producer_errors):
        start_event.set()
        raise RuntimeError("TP refit producer failed before receiver launch") from next(
            error for error in producer_errors if error is not None
        )

    with memory_monitor_factory() as monitor:
        transfer_begin = clock()
        consumer_futures = generation.update_weights_via_ipc_zmq()
        start_event.set()
        consumer_results = list(ray_get(consumer_futures))
        for thread in threads:
            thread.join(ready_timeout_s)
        transfer_wall_s = clock() - transfer_begin

    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("TP refit producer did not finish after receiver completion")
    if any(error is not None for error in producer_errors):
        raise RuntimeError("TP refit producer failed during streaming") from next(
            error for error in producer_errors if error is not None
        )
    if not consumer_results or not all(result is True for result in consumer_results):
        raise RuntimeError(f"one or more NeMo-RL refit consumers failed: {consumer_results}")
    if any(result is None for result in producer_results):
        raise RuntimeError("one or more TP refit producers returned no evidence")

    if not math.isfinite(prepare_refit_s) or prepare_refit_s < 0:
        raise AssertionError("refit preparation timing is invalid")
    return {
        "phase": phase,
        "device_uuids": device_uuids,
        "reset_results": reset_results,
        "consumer_results": consumer_results,
        "producer_results": list(producer_results),
        "timing": {
            "prepare_refit_s": prepare_refit_s,
            "transfer_wall_s": transfer_wall_s,
        },
        "peak_device_memory_bytes": list(monitor.peak_device_memory_bytes),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the exact real-checkpoint TP2 parity/refit CLI."""
    parser = argparse.ArgumentParser(description="Prove Evo2 TP2 NeMo-RL parity and two production refits")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--load-format", default="safetensors")
    parser.add_argument("--max-model-len", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16_384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--optimization-level", type=int, choices=(2, 3), required=True)
    parser.add_argument("--performance-mode", choices=("balanced", "throughput"), required=True)
    parser.add_argument("--num-logprobs", type=int, default=512)
    parser.add_argument("--refit-buffer-size-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--placement-max-error", type=float, default=0.05)
    parser.add_argument("--placement-p95-error", type=float, default=0.005)
    parser.add_argument("--refit-max-error", type=float, default=1e-6)
    return parser


def _phase_evidence(phase: Any) -> dict[str, Any]:
    if len(phase.wave_proofs) != 1:
        raise AssertionError("short TP2 parity phases must execute in one exact wave")
    evidence = phase.wave_proofs[0].get("full_vocab_logprobs")
    if not isinstance(evidence, dict):
        raise AssertionError("TP2 parity phase is missing full-vocabulary evidence")
    return evidence


def validate_final_compilation_stability(initialized_proof: dict[str, Any], final_phase: Any) -> dict[str, Any]:
    """Require compile and graph counters to remain stable through the last generation."""
    from bionemo.evo2.vllm.runner import validate_compilation_proof

    initial_workers = initialized_proof.get("worker_proof")
    if not isinstance(initial_workers, list) or len(initial_workers) != 2:
        raise AssertionError("TP2 compilation proof must cover both initialized workers")
    if not final_phase.wave_proofs:
        raise AssertionError("final generation phase is missing worker proof")
    engines = final_phase.wave_proofs[-1].get("engines")
    if not isinstance(engines, list) or len(engines) != 1:
        raise AssertionError("final TP2 phase must contain exactly one engine proof")
    final_workers = engines[0].get("worker_proof")
    if not isinstance(final_workers, list) or len(final_workers) != 2:
        raise AssertionError("TP2 compilation proof must cover both final workers")
    for initialized_worker, final_worker in zip(initial_workers, final_workers, strict=True):
        validate_compilation_proof(initialized_worker["compilation"], final_worker["compilation"])
    return {
        "passed": True,
        "final_phase": str(final_phase.phase),
        "tp_worker_count": len(final_workers),
    }


def _logprob_evidence_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    """Summarize one uniform short-run logprob evidence lane."""
    shape = evidence.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise AssertionError("logprob evidence summary requires [requests, steps, vocab] shape")
    request_count, steps, vocab_size = (int(value) for value in shape)

    def uniform_value(key: str) -> int:
        rows = evidence.get(key)
        if not isinstance(rows, list) or len(rows) != request_count:
            raise AssertionError(f"logprob evidence {key} does not cover every request")
        values = [int(value) for row in rows for value in row]
        if len(values) != request_count * steps or len(set(values)) != 1:
            raise AssertionError(f"logprob evidence {key} is not uniform across every step")
        return values[0]

    coverage = uniform_value("coverage_counts")
    finite_support = uniform_value("finite_support_counts")
    negative_infinity = uniform_value("negative_infinity_counts")
    if coverage != vocab_size or finite_support + negative_infinity != vocab_size:
        raise AssertionError("logprob evidence does not account for every vocabulary token")
    if evidence.get("expected_finite_support") != finite_support:
        raise AssertionError("logprob evidence support does not match its configured contract")
    chosen_token_ids = evidence.get("chosen_token_ids")
    chosen_logprobs = evidence.get("chosen_token_logprobs")
    full_logprobs = evidence.get("logprobs")
    if (
        not isinstance(chosen_token_ids, list)
        or len(chosen_token_ids) != request_count
        or not isinstance(chosen_logprobs, list)
        or len(chosen_logprobs) != request_count
        or not isinstance(full_logprobs, list)
        or len(full_logprobs) != request_count
    ):
        raise AssertionError("logprob evidence does not retain exact chosen-token gather inputs")
    gathered_logprobs = []
    for row_index in range(request_count):
        token_row = chosen_token_ids[row_index]
        chosen_row = chosen_logprobs[row_index]
        distribution_row = full_logprobs[row_index]
        if (
            not isinstance(token_row, list)
            or len(token_row) != steps
            or not isinstance(chosen_row, list)
            or len(chosen_row) != steps
            or not isinstance(distribution_row, list)
            or len(distribution_row) != steps
        ):
            raise AssertionError("logprob evidence chosen-token rows do not match the retained tensor shape")
        gathered_row = []
        for step in range(steps):
            token_id = token_row[step]
            distribution = distribution_row[step]
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < vocab_size
                or not isinstance(distribution, list)
                or len(distribution) != vocab_size
            ):
                raise AssertionError("logprob evidence contains an invalid chosen-token gather coordinate")
            gathered = distribution[token_id]
            reported = chosen_row[step]
            if (
                isinstance(gathered, bool)
                or not isinstance(gathered, (int, float))
                or not math.isfinite(float(gathered))
                or isinstance(reported, bool)
                or not isinstance(reported, (int, float))
                or not math.isfinite(float(reported))
            ):
                raise AssertionError("logprob evidence contains a non-finite chosen-token value")
            gathered_value = float(gathered)
            if struct.pack(">d", gathered_value) != struct.pack(">d", float(reported)):
                raise AssertionError("chosen-token summary does not bitwise match the retained full-vocabulary tensor")
            gathered_row.append(gathered_value)
        gathered_logprobs.append(gathered_row)
    if not evidence.get("chosen_token_in_finite_support") or not evidence.get("chosen_token_oracle_passed"):
        raise AssertionError("logprob evidence did not prove its chosen processed-policy token")

    return {
        "shape": [request_count, steps, vocab_size],
        "returned_candidate_ids_per_step": coverage,
        "finite_support_size_per_step": finite_support,
        "negative_infinity_exclusions_per_step": negative_infinity,
        "chosen_token_in_finite_support": True,
        "chosen_token_oracle_passed": True,
        "chosen_token_ids": chosen_token_ids,
        "chosen_token_logprobs": gathered_logprobs,
    }


def _evidence_row(evidence: dict[str, Any], row_index: int) -> dict[str, Any]:
    request_count, steps, vocab_size = evidence["shape"]
    if not 0 <= row_index < request_count:
        raise IndexError("full-vocabulary evidence row is out of range")
    return {
        "shape": [1, steps, vocab_size],
        "coverage_counts": [evidence["coverage_counts"][row_index]],
        "finite_support_counts": [evidence["finite_support_counts"][row_index]],
        "negative_infinity_counts": [evidence["negative_infinity_counts"][row_index]],
        "expected_finite_support": evidence["expected_finite_support"],
        "chosen_token_oracle_passed": evidence["chosen_token_oracle_passed"],
        "chosen_token_in_finite_support": evidence["chosen_token_in_finite_support"],
        "chosen_token_ids": [evidence["chosen_token_ids"][row_index]],
        "chosen_token_logprobs": [evidence["chosen_token_logprobs"][row_index]],
        "logprobs": [evidence["logprobs"][row_index]],
    }


def _phase_tokens(phase: Any) -> list[list[int]]:
    tokens = []
    for summary in phase.output_summaries:
        if summary["output_length"] > len(summary["first_output_tokens"]):
            raise AssertionError("short parity output summary does not retain every token")
        tokens.append([int(value) for value in summary["first_output_tokens"]])
    return tokens


def _gated_comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    label: str,
    max_error: float,
    p95_error: float,
) -> dict[str, Any]:
    comparison = compare_full_vocab_evidence(reference, candidate)
    passed = (
        comparison["top1_identity"]
        and comparison["max_abs_logprob_error"] <= max_error
        and comparison["p95_abs_logprob_error"] <= p95_error
    )
    comparison.update(
        {
            "label": label,
            "max_abs_error_threshold": max_error,
            "p95_abs_error_threshold": p95_error,
            "passed": passed,
        }
    )
    if not passed:
        raise AssertionError(f"{label} full-vocabulary parity failed: {comparison}")
    return comparison


def _outer_rpc(generation: Any, method_name: str, *, phase: str, ray_get: Callable[[Any], Any]) -> list[Any]:
    futures = generation.worker_group.run_all_workers_single_data(
        method_name,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        phase=phase,
    )
    return list(ray_get(futures))


def _validate_initial_load_proof(proof: dict[str, Any]) -> dict[str, Any]:
    workers = proof.get("worker_proof")
    if not isinstance(workers, list) or len(workers) != 2:
        raise AssertionError("initial load proof must cover both TP workers")
    for worker in workers:
        loader = worker.get("loader")
        if not isinstance(loader, dict):
            raise AssertionError("initial load proof is missing loader state")
        if loader.get("completed_transactions") != 1:
            raise AssertionError("initial checkpoint did not complete exactly one load transaction")
        if loader.get("loaded_parameter_count") != loader.get("required_parameter_count"):
            raise AssertionError("initial checkpoint did not load every mandatory parameter")
        if not loader.get("complete") or not loader.get("consumed"):
            raise AssertionError("initial checkpoint was not consumed by production inference")
        if loader.get("pending_fc1_layer_count") != 0:
            raise AssertionError("initial checkpoint retained incomplete fused MLP weights")
    return {"passed": True, "tp_worker_count": len(workers), "completed_transactions": [1, 1]}


def _refit_plan_artifact(plan: IpcChunkPlan) -> dict[str, Any]:
    return {
        "buffer_size_bytes": plan.buffer_size_bytes,
        "per_buffer_capacity_bytes": plan.per_buffer_capacity_bytes,
        "chunk_count": plan.chunk_count,
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "tensor_count": chunk.tensor_count,
                "tensor_bytes": chunk.tensor_bytes,
                "used_bytes": chunk.used_bytes,
                "first_name": chunk.tensor_names[0],
                "last_name": chunk.tensor_names[-1],
                "names_sha256": hashlib.sha256("\n".join(chunk.tensor_names).encode()).hexdigest(),
            }
            for chunk in plan.chunks
        ],
    }


def run_tp2_refit_parity(args: Any) -> dict[str, Any]:
    """Run real 7B TP2 full-vocabulary parity around two production refits."""
    if args.max_new_tokens != 4:
        raise ValueError("canonical multi-step parity requires exactly four generated tokens")
    if args.num_logprobs != 512:
        raise ValueError("canonical Evo2 parity requires all 512 processed logprobs")

    import nemo_rl
    import ray
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
    from nemo_rl.utils.nvml import get_device_uuid

    from bionemo.evo2.vllm.nemo_runner import (
        build_nemo_generation_caller_ledgers,
        build_nemo_generation_config,
        run_nemo_generation_phase,
    )
    from bionemo.evo2.vllm.profile import Evo2VllmProfile, context_length_preflight, validate_resolved_profile
    from bionemo.evo2.vllm.refit import indexed_safetensors_layout, plan_ipc_chunks
    from bionemo.evo2.vllm.runner import (
        PeakMemoryMonitor,
        checkpoint_provenance,
        make_nvml_memory_reader,
        phase_output_artifact_path,
        require_output_namespace_reservation,
        runtime_versions,
        source_provenance,
    )

    clock = time.perf_counter
    output_path = Path(args.output).resolve()
    require_output_namespace_reservation(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = WorkloadManifest.from_path(args.manifest)
    parity_manifest = source_manifest.request_slice(0, 1).with_max_new_tokens(args.max_new_tokens)
    placement_manifest = build_duplicate_prompt_manifest(
        parity_manifest,
        request_count=2,
        request_id_prefix="placement",
    )
    stochastic_manifest = build_duplicate_prompt_manifest(
        parity_manifest,
        request_count=2,
        request_id_prefix="stochastic",
    )

    profile = Evo2VllmProfile(
        topology="tp2",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        proof=True,
        optimization_level=args.optimization_level,
        performance_mode=args.performance_mode,
    )
    preflight_begin = clock()
    preflight = context_length_preflight(
        profile,
        model=args.checkpoint,
        workload_max_total_tokens=parity_manifest.max_total_tokens,
        load_format=args.load_format,
    )
    preflight_s = clock() - preflight_begin
    config = build_nemo_generation_config(
        profile,
        parity_manifest,
        checkpoint=args.checkpoint,
        load_format=args.load_format,
        num_logprobs=args.num_logprobs,
    )

    layout_begin = clock()
    layout = indexed_safetensors_layout(args.checkpoint)
    plan = plan_ipc_chunks(layout, buffer_size_bytes=args.refit_buffer_size_bytes)
    layout_s = clock() - layout_begin

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
            nemo_package / "models/policy/utils.py",
        ),
    )
    provenance_s = clock() - provenance_begin

    memory_reader = make_nvml_memory_reader()
    ray_dir_suffix = hashlib.sha256(str(output_path).encode()).hexdigest()[:10]
    ray_log_dir = Path("/tmp") / f"e2ray-tp2-parity-{ray_dir_suffix}"
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
            name=f"evo2-vllm-tp2-parity-{output_path.stem}",
        )
        engine_begin = clock()
        with PeakMemoryMonitor(memory_reader) as init_memory:
            generation = VllmGeneration(
                cluster,
                config,
                name_prefix=f"evo2_vllm_tp2_parity_{output_path.stem}",
            )
        engine_init_s = clock() - engine_begin

        initialized_phase = "engine-initialized"
        initialized_reset = _outer_rpc(
            generation,
            "reset_evo2_proof_phase",
            phase=initialized_phase,
            ray_get=ray.get,
        )
        initialized_proofs = _outer_rpc(
            generation,
            "snapshot_evo2_proof_phase",
            phase=initialized_phase,
            ray_get=ray.get,
        )
        if len(initialized_proofs) != 1:
            raise AssertionError("TP2 parity must launch exactly one outer generation engine")
        validate_resolved_profile(profile, initialized_proofs[0]["resolved_config"])

        phase_index = 0
        global_call_index = 0

        def generate_phase(
            phase: str,
            manifest: WorkloadManifest,
            *,
            greedy: bool,
            expected_finite_support: int,
        ):
            nonlocal global_call_index, phase_index
            phase_call_count = len(
                build_request_waves(
                    request_count=len(manifest.requests),
                    global_batch_size=profile.global_wave_size,
                    replica_count=profile.replica_count,
                )
            )
            request_envelope_namespace = f"tp2-parity/{output_path.stem}"
            caller_ledger = build_nemo_generation_caller_ledgers(
                manifest=manifest,
                profile=profile,
                phases=(phase,),
                generation_round=phase_index,
                global_request_index_start=0,
                request_envelope_namespace=request_envelope_namespace,
            )[phase]
            result = run_nemo_generation_phase(
                generation=generation,
                manifest=manifest,
                profile=profile,
                phase=phase,
                sample_index=phase_index,
                generation_round=phase_index,
                global_call_index_start=global_call_index,
                global_request_index_start=0,
                caller_ledger_admission=caller_ledger,
                full_output_path=phase_output_artifact_path(output_path, phase=phase),
                namespace_output_path=output_path,
                memory_monitor_factory=lambda: PeakMemoryMonitor(memory_reader),
                ray_get=ray.get,
                clock=clock,
                greedy=greedy,
                require_full_vocab_logprobs=True,
                expected_finite_logprob_support=expected_finite_support,
                request_envelope_namespace=request_envelope_namespace,
            )
            phase_index += 1
            global_call_index += phase_call_count
            return result

        baseline = generate_phase(
            "baseline-greedy",
            parity_manifest,
            greedy=True,
            expected_finite_support=args.num_logprobs,
        )
        _outer_rpc(
            generation,
            "reset_evo2_refit_phase",
            phase="initial-load",
            ray_get=ray.get,
        )
        initial_load_proofs = _outer_rpc(
            generation,
            "snapshot_evo2_refit_phase",
            phase="initial-load",
            ray_get=ray.get,
        )
        if len(initial_load_proofs) != 1:
            raise AssertionError("initial load proof must come from one TP2 outer engine")
        initial_load_validation = _validate_initial_load_proof(initial_load_proofs[0])

        placement = generate_phase(
            "placement-batch2-greedy",
            placement_manifest,
            greedy=True,
            expected_finite_support=args.num_logprobs,
        )
        device_index_by_uuid = {get_device_uuid(index): index for index in range(torch.cuda.device_count())}

        refit1_transfer = run_production_refit(
            generation=generation,
            layout=layout,
            plan=plan,
            phase="refit-1",
            device_index_by_uuid=device_index_by_uuid,
            ray_get=ray.get,
            memory_monitor_factory=lambda: PeakMemoryMonitor(memory_reader),
        )
        after_refit1 = generate_phase(
            "after-refit-1-greedy",
            parity_manifest,
            greedy=True,
            expected_finite_support=args.num_logprobs,
        )
        refit1_proofs = _outer_rpc(
            generation,
            "snapshot_evo2_refit_phase",
            phase="refit-1",
            ray_get=ray.get,
        )
        refit1_validation = validate_refit_proof(
            refit1_proofs[0],
            layout=layout,
            plan=plan,
            expected_phase="refit-1",
            expected_completed_transactions=2,
            expected_tp_size=2,
        )

        refit2_transfer = run_production_refit(
            generation=generation,
            layout=layout,
            plan=plan,
            phase="refit-2",
            device_index_by_uuid=device_index_by_uuid,
            ray_get=ray.get,
            memory_monitor_factory=lambda: PeakMemoryMonitor(memory_reader),
        )
        after_refit2 = generate_phase(
            "after-refit-2-greedy",
            parity_manifest,
            greedy=True,
            expected_finite_support=args.num_logprobs,
        )
        refit2_proofs = _outer_rpc(
            generation,
            "snapshot_evo2_refit_phase",
            phase="refit-2",
            ray_get=ray.get,
        )
        refit2_validation = validate_refit_proof(
            refit2_proofs[0],
            layout=layout,
            plan=plan,
            expected_phase="refit-2",
            expected_completed_transactions=3,
            expected_tp_size=2,
        )

        stochastic1 = generate_phase(
            "stochastic-1",
            stochastic_manifest,
            greedy=False,
            expected_finite_support=stochastic_manifest.top_k,
        )
        stochastic2 = generate_phase(
            "stochastic-2",
            stochastic_manifest,
            greedy=False,
            expected_finite_support=stochastic_manifest.top_k,
        )

        baseline_logprob_summary = _logprob_evidence_summary(_phase_evidence(baseline))
        placement_logprob_summary = _logprob_evidence_summary(_phase_evidence(placement))
        refit1_logprob_summary = _logprob_evidence_summary(_phase_evidence(after_refit1))
        refit2_logprob_summary = _logprob_evidence_summary(_phase_evidence(after_refit2))
        stochastic1_logprob_summary = _logprob_evidence_summary(_phase_evidence(stochastic1))
        stochastic2_logprob_summary = _logprob_evidence_summary(_phase_evidence(stochastic2))

        baseline_tokens = _phase_tokens(baseline)[0]
        placement_tokens = _phase_tokens(placement)
        refit_tokens = [_phase_tokens(after_refit1)[0], _phase_tokens(after_refit2)[0]]
        if any(tokens != baseline_tokens for tokens in [*placement_tokens, *refit_tokens]):
            raise AssertionError("greedy tokens changed across placement or identical checkpoint refit")

        baseline_evidence = _phase_evidence(baseline)
        placement_evidence = _phase_evidence(placement)
        placement_comparisons = [
            _gated_comparison(
                baseline_evidence,
                _evidence_row(placement_evidence, row_index),
                label=f"batch1-vs-batch2-row{row_index}",
                max_error=args.placement_max_error,
                p95_error=args.placement_p95_error,
            )
            for row_index in range(2)
        ]
        refit_comparisons = [
            _gated_comparison(
                baseline_evidence,
                _phase_evidence(phase),
                label=label,
                max_error=args.refit_max_error,
                p95_error=args.refit_max_error,
            )
            for phase, label in (
                (after_refit1, "baseline-vs-refit-1"),
                (after_refit2, "baseline-vs-refit-2"),
            )
        ]

        stochastic1_seeds = [record.seed for record in stochastic1.request_executions]
        stochastic2_seeds = [record.seed for record in stochastic2.request_executions]
        stochastic1_calls = {record.call_index for record in stochastic1.request_executions}
        stochastic2_calls = {record.call_index for record in stochastic2.request_executions}
        if len(stochastic1_calls) != 1 or stochastic2_calls != {next(iter(stochastic1_calls)) + 1}:
            raise AssertionError("successive stochastic TP2 calls did not advance exactly once")
        if set(stochastic1_seeds) & set(stochastic2_seeds):
            raise AssertionError("successive stochastic TP2 calls replayed request seeds")

        final_compilation_validation = validate_final_compilation_stability(
            initialized_proofs[0],
            stochastic2,
        )

        phase_results = [
            baseline,
            placement,
            after_refit1,
            after_refit2,
            stochastic1,
            stochastic2,
        ]
        return {
            "schema_version": 2,
            "task": "evo2-tp2-production-parity-two-refits",
            "backend": "nemo-rl-vllm",
            "topology": "tp2",
            "versions": runtime_versions(),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_provenance": checkpoint_identity,
            "source_provenance": {
                "bionemo": bionemo_source_identity,
                "nemo_rl": nemo_source_identity,
            },
            "invocation": {
                "argv": [sys.executable, *sys.argv],
                "parsed_args": {
                    name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
                },
                "output_artifact_path": str(output_path),
                "log_path": str(Path(args.log_path).resolve()),
                "ray_log_dir": str(ray_log_dir),
                "exit_status": 0,
                "nemo_rl_py_executables_system": os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM"),
            },
            "manifest": parity_manifest.to_dict(),
            "manifest_sha256": parity_manifest.sha256,
            "profile": asdict(profile),
            "context_length_preflight": preflight,
            "nemo_generation_config": config,
            "resolved_config": initialized_proofs[0]["resolved_config"],
            "execution_contract": {
                "production_path": "nemo_rl.models.generation.vllm.VllmGeneration",
                "tensor_parallel_size": 2,
                "data_parallel_size": 1,
                "engine_logprobs_mode": "processed_logprobs",
                "requested_logprob_entries_per_step": 512,
                "greedy_raw_equivalent_full_vocab_support": 512,
                "stochastic_processed_policy_support": stochastic_manifest.top_k,
                "greedy_continuation_steps": 4,
                "production_ipc_refits": 2,
                "semantic_padding": False,
                "enforce_eager": False,
                "prefix_caching": False,
            },
            "timing": {
                "context_length_preflight_s": preflight_s,
                "layout_and_refit_plan_s": layout_s,
                "provenance_hashing_s": provenance_s,
                "ray_init_s": ray_init_s,
                "engine_export_compile_capture_init_s": engine_init_s,
                "engine_init_peak_device_memory_bytes": list(init_memory.peak_device_memory_bytes),
            },
            "refit_source": {
                "tensor_count": layout.tensor_count,
                "tensor_bytes": layout.total_tensor_bytes,
                "largest_tensor_bytes": layout.largest_tensor_bytes,
                "plan": _refit_plan_artifact(plan),
            },
            "initialized_reset": initialized_reset,
            "initialized_engine_proof": initialized_proofs[0],
            "initial_load": {
                "proof": initial_load_proofs[0],
                "validation": initial_load_validation,
            },
            "phases": [phase.to_dict() for phase in phase_results],
            "parity": {
                "greedy_token_ids": baseline_tokens,
                "placement": placement_comparisons,
                "refit_identity": refit_comparisons,
                "all_four_steps_full_vocab_retained": True,
                "chosen_token_oracle_passed": True,
            },
            "logprob_evidence_lanes": {
                "greedy_raw_equivalent_full_vocab": {
                    "engine_logprobs_mode": "processed_logprobs",
                    "sampling_mode": "greedy",
                    "top_k_applied": False,
                    "raw_equivalent": True,
                    "raw_equivalent_reason": (
                        "vLLM's greedy sampler returns log_softmax before temperature/top-k; "
                        "this workload config has no other active logit processor"
                    ),
                    "baseline": baseline_logprob_summary,
                    "placement_batch2": placement_logprob_summary,
                    "after_refit_1": refit1_logprob_summary,
                    "after_refit_2": refit2_logprob_summary,
                },
                "stochastic_processed_policy": {
                    "engine_logprobs_mode": "processed_logprobs",
                    "sampling_mode": "stochastic",
                    "temperature": stochastic_manifest.temperature,
                    "top_k": stochastic_manifest.top_k,
                    "top_p": stochastic_manifest.top_p,
                    "first_call": stochastic1_logprob_summary,
                    "second_call": stochastic2_logprob_summary,
                },
            },
            "refits": [
                {
                    "transfer": refit1_transfer,
                    "proof": refit1_proofs[0],
                    "validation": refit1_validation,
                },
                {
                    "transfer": refit2_transfer,
                    "proof": refit2_proofs[0],
                    "validation": refit2_validation,
                },
            ],
            "stochastic_seed_audit": {
                "passed": True,
                "first_call_seeds": stochastic1_seeds,
                "second_call_seeds": stochastic2_seeds,
                "first_call_output_sha256": [item["output_sha256"] for item in stochastic1.output_summaries],
                "second_call_output_sha256": [item["output_sha256"] for item in stochastic2.output_summaries],
                "successive_calls_advanced": True,
                "streams_disjoint": True,
                "processed_policy_chosen_tokens_finite": True,
                "processed_policy_support_size": stochastic_manifest.top_k,
                "processed_policy_negative_infinity_exclusions": (args.num_logprobs - stochastic_manifest.top_k),
            },
            "compilation_and_cudagraph_gate_passed": True,
            "final_compilation_validation": final_compilation_validation,
        }
    finally:
        if generation is not None:
            generation.shutdown()
        if cluster is not None:
            cluster.shutdown()
        ray.shutdown()


def main(argv: list[str] | None = None) -> int:
    """Run and persist the canonical TP2 parity/refit proof."""
    from bionemo.evo2.vllm.runner import (
        complete_output_namespace,
        require_output_namespace_reservation,
        reserve_output_namespace,
        write_json_artifact,
    )

    args = build_parser().parse_args(argv)
    reservation = reserve_output_namespace(args.output)
    artifact = run_tp2_refit_parity(args)
    require_output_namespace_reservation(args.output)
    write_json_artifact(
        args.output,
        artifact,
        ownership_validator=lambda: require_output_namespace_reservation(args.output),
    )
    complete_output_namespace(reservation, output_path=args.output)
    return 0


__all__ = [
    "build_duplicate_prompt_manifest",
    "build_parser",
    "compare_full_vocab_evidence",
    "main",
    "run_production_refit",
    "run_tp2_refit_parity",
    "validate_final_compilation_stability",
]


if __name__ == "__main__":
    raise SystemExit(main())
