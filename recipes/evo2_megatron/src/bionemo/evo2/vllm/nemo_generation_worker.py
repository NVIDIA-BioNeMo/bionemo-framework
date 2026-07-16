# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""NeMo-RL outer generation worker with phase-local Evo2 execution proof."""

import gzip
import hashlib
import io
import json
import math
import os
import re
import socket
from typing import Any

import ray
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    generation_prompt_token_ids_sha256,
    validate_generation_request_metadata,
    validate_generation_selected_logprobs,
)
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

from bionemo.evo2.vllm.artifact_io import (
    parse_json_bytes,
    publish_reserved_bytes,
    validate_file_publication_plan,
)
from bionemo.evo2.vllm.profile import resolved_config_snapshot
from bionemo.evo2.vllm.nemo_publication_schema import (
    NEMO_RANK_EXECUTION_OCCURRENCE_SCHEMA_VERSION,
    NEMO_RANK_PUBLICATION_OUTCOME_SCHEMA_VERSION,
    NEMO_RANK_PUBLICATION_SCHEMA_VERSION,
    NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION,
)
from bionemo.evo2.vllm.runner import CUDAGraphProofRecorder, summarize_cudagraph_observations


_RANK_PUBLICATION_FIELDS = {
    "schema_version",
    "publication_key",
    "publication_plan",
    "external_contract_sha256",
    "manifest_sha256",
    "phase",
    "wave_index",
    "generation_round",
    "call_index",
    "dp_rank",
    "dp_size",
    "tensor_parallel_size",
    "global_wave_start",
    "global_wave_stop",
    "global_wave_request_count",
    "expected_request_count",
    "requested_max_new_tokens",
    "caller_prompt_anchor_sha256",
    "execution_occurrence_sha256",
    "generation_caller_ledger_sha256",
    "generation_schedule_admission_sha256",
    "generation_semantic_namespace",
    "generation_request_envelope_sha256",
    "generation_execution_occurrence_sha256",
    "requests",
    "envelope_sha256",
}
_RANK_REQUEST_FIELDS = {
    "request_id",
    "global_request_index",
    "request_index_in_dp_stream",
    "seed",
    "prompt_token_count",
    "prompt_token_ids_sha256",
    "semantic_request_sha256",
}


def _require_digest(value: Any, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _require_integer(value: Any, *, label: str, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be a {qualifier} built-in integer")
    return value


def _rank_execution_occurrence_sha256(envelope: dict[str, Any]) -> str:
    payload = {
        "schema_version": NEMO_RANK_EXECUTION_OCCURRENCE_SCHEMA_VERSION,
        "publication_key": envelope["publication_key"],
        "phase": envelope["phase"],
        "wave_index": envelope["wave_index"],
        "generation_round": envelope["generation_round"],
        "call_index": envelope["call_index"],
        "dp_rank": envelope["dp_rank"],
        "active": True,
        "semantic_request_sha256": [
            request["semantic_request_sha256"] for request in envelope["requests"]
        ],
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _validate_rank_publication_envelope(
    envelope: Any,
    *,
    expected_envelope_sha256: str,
) -> dict[str, Any]:
    expected_envelope_sha256 = _require_digest(
        expected_envelope_sha256,
        label="coordinator-owned envelope",
    )
    if type(envelope) is not dict or set(envelope) != _RANK_PUBLICATION_FIELDS:
        raise RuntimeError("rank publication envelope fields are not exact")
    if envelope.get("schema_version") != NEMO_RANK_PUBLICATION_SCHEMA_VERSION:
        raise RuntimeError("rank publication envelope schema is unsupported")
    for label in ("publication_key", "phase"):
        if type(envelope.get(label)) is not str or not envelope[label]:
            raise RuntimeError(f"rank publication {label} must be a nonempty string")
    for label in (
        "wave_index",
        "generation_round",
        "call_index",
        "dp_rank",
        "global_wave_start",
        "global_wave_stop",
    ):
        _require_integer(envelope.get(label), label=label)
    for label in (
        "dp_size",
        "tensor_parallel_size",
        "global_wave_request_count",
        "expected_request_count",
        "requested_max_new_tokens",
    ):
        _require_integer(envelope.get(label), label=label, positive=True)
    if envelope["dp_rank"] >= envelope["dp_size"]:
        raise RuntimeError("rank publication DP rank is outside its topology")
    _require_digest(envelope.get("external_contract_sha256"), label="external contract")
    _require_digest(envelope.get("manifest_sha256"), label="manifest")
    caller_prompt_anchor_sha256 = _require_digest(
        envelope.get("caller_prompt_anchor_sha256"),
        label="caller prompt anchor",
    )
    execution_occurrence_sha256 = _require_digest(
        envelope.get("execution_occurrence_sha256"),
        label="execution occurrence",
    )
    _require_digest(
        envelope.get("generation_caller_ledger_sha256"),
        label="generation caller ledger",
    )
    _require_digest(
        envelope.get("generation_schedule_admission_sha256"),
        label="generation schedule admission",
    )
    if (
        type(envelope.get("generation_semantic_namespace")) is not str
        or not envelope["generation_semantic_namespace"]
    ):
        raise RuntimeError("generation semantic namespace must be a nonempty string")
    _require_digest(
        envelope.get("generation_request_envelope_sha256"),
        label="generation request envelope",
    )
    _require_digest(
        envelope.get("generation_execution_occurrence_sha256"),
        label="generation execution occurrence",
    )
    envelope_sha256 = _require_digest(envelope.get("envelope_sha256"), label="worker envelope")
    unsigned = dict(envelope)
    unsigned.pop("envelope_sha256")
    unsigned.pop("publication_plan")
    observed_envelope_sha256 = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if observed_envelope_sha256 != envelope_sha256 or envelope_sha256 != expected_envelope_sha256:
        raise RuntimeError("rank publication envelope differs from the coordinator-owned envelope digest")
    plan = validate_file_publication_plan(envelope.get("publication_plan"))
    if (
        plan.publication_key != envelope["publication_key"]
        or plan.external_contract_sha256 != envelope["external_contract_sha256"]
        or plan.payload_contract_sha256 != expected_envelope_sha256
    ):
        raise RuntimeError("rank publication plan is not bound to the caller envelope")
    requests = envelope.get("requests")
    if type(requests) is not list or len(requests) != envelope["expected_request_count"]:
        raise RuntimeError("rank publication request rows do not match expected_request_count")
    request_ids = []
    global_indices = []
    for local_index, request in enumerate(requests):
        if type(request) is not dict or set(request) != _RANK_REQUEST_FIELDS:
            raise RuntimeError("rank publication request coordinate fields are not exact")
        if type(request.get("request_id")) is not str or not request["request_id"]:
            raise RuntimeError("rank publication request ID must be a nonempty string")
        for label in ("global_request_index", "request_index_in_dp_stream", "seed"):
            _require_integer(request.get(label), label=label)
        _require_integer(request.get("prompt_token_count"), label="prompt_token_count", positive=True)
        _require_digest(request.get("prompt_token_ids_sha256"), label="prompt token IDs")
        _require_digest(request.get("semantic_request_sha256"), label="semantic request")
        if request["request_index_in_dp_stream"] != local_index:
            raise RuntimeError("rank publication request DP stream order changed")
        request_ids.append(request["request_id"])
        global_indices.append(request["global_request_index"])
    if len(request_ids) != len(set(request_ids)) or len(global_indices) != len(set(global_indices)):
        raise RuntimeError("rank publication request ownership contains duplicates")
    if global_indices != list(range(global_indices[0], global_indices[0] + len(global_indices))):
        raise RuntimeError("rank publication global request indices are not contiguous")
    if not (
        envelope["global_wave_start"] <= global_indices[0]
        and global_indices[-1] < envelope["global_wave_stop"]
        and envelope["global_wave_stop"] - envelope["global_wave_start"]
        == envelope["global_wave_request_count"]
    ):
        raise RuntimeError("rank publication requests fall outside their global wave")
    if _rank_execution_occurrence_sha256(envelope) != execution_occurrence_sha256:
        raise RuntimeError("rank publication execution occurrence is internally inconsistent")
    return envelope


def _decode_rank_publication_envelope(
    envelope_payload: bytes,
    *,
    expected_envelope_sha256: str,
) -> dict[str, Any]:
    """Parse one immutable wire snapshot before validating caller-owned semantics."""
    if type(envelope_payload) is not bytes:
        raise TypeError("rank publication envelope wire payload must be exact bytes")
    envelope = parse_json_bytes(envelope_payload, label="rank publication envelope")
    return _validate_rank_publication_envelope(
        envelope,
        expected_envelope_sha256=expected_envelope_sha256,
    )


def _rank_publication_payload(result: Any, envelope: dict[str, Any]) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    if not isinstance(result, BatchedDataDict):
        raise TypeError("rank publication generation result must be a BatchedDataDict")
    request_count = envelope["expected_request_count"]
    validate_generation_selected_logprobs(result)
    validate_generation_request_metadata(
        result,
        request_count=request_count,
        require_explicit_envelope=True,
        expected_data_parallel_size=envelope["dp_size"],
    )
    required_tensors = {
        "output_ids",
        "logprobs",
        "generation_lengths",
        "unpadded_sequence_lengths",
        "truncated",
        "generation_request_seeds",
        "generation_global_request_indices",
        "generation_rounds",
        "generation_flattened_call_indices",
        "generation_dp_ranks",
        "generation_immutable_local_ordinals",
        "generation_selected_logprob_valid",
        "generation_selected_logprob_counts",
    }
    if any(not isinstance(result.get(key), torch.Tensor) for key in required_tensors):
        raise RuntimeError("rank publication generation result is missing required tensors")
    tensors = {key: result[key].detach().cpu() for key in required_tensors}
    output_ids = tensors["output_ids"]
    logprobs = tensors["logprobs"]
    if output_ids.ndim != 2 or logprobs.shape != output_ids.shape or output_ids.shape[0] != request_count:
        raise RuntimeError("rank publication token and logprob tensors are not aligned")
    for key, tensor in tensors.items():
        if key in {"output_ids", "logprobs", "generation_selected_logprob_valid"}:
            continue
        if tensor.ndim != 1 or tensor.shape[0] != request_count:
            raise RuntimeError(f"rank publication metadata tensor {key!r} is not row-aligned")

    engine_request_ids = result.get("generation_engine_request_ids")
    local_request_ids = result.get("generation_local_request_ids")
    prompt_digests = result.get("generation_prompt_token_ids_sha256")
    for label, values in (
        ("engine request IDs", engine_request_ids),
        ("local request IDs", local_request_ids),
        ("prompt digests", prompt_digests),
    ):
        if type(values) is not list or len(values) != request_count:
            raise RuntimeError(f"rank publication {label} are not row-aligned")
    if any(type(request_id) is not str or not request_id for request_id in engine_request_ids):
        raise RuntimeError("rank publication engine request IDs are malformed")
    if len(set(engine_request_ids)) != request_count:
        raise RuntimeError("rank publication engine request IDs are duplicated")

    result_index_by_semantic_key = {}
    for row_index in range(request_count):
        semantic_key = (
            local_request_ids[row_index],
            tensors["generation_global_request_indices"][row_index].item(),
            tensors["generation_immutable_local_ordinals"][row_index].item(),
            tensors["generation_rounds"][row_index].item(),
            tensors["generation_flattened_call_indices"][row_index].item(),
            tensors["generation_dp_ranks"][row_index].item(),
            tensors["generation_request_seeds"][row_index].item(),
            prompt_digests[row_index],
        )
        if semantic_key in result_index_by_semantic_key:
            raise RuntimeError("rank publication result contains a duplicate semantic key")
        result_index_by_semantic_key[semantic_key] = row_index

    rows = []
    expected_semantic_keys = []
    for expected in envelope["requests"]:
        expected_semantic_key = (
            expected["request_id"],
            expected["global_request_index"],
            expected["request_index_in_dp_stream"],
            envelope["generation_round"],
            envelope["call_index"],
            envelope["dp_rank"],
            expected["seed"],
            expected["prompt_token_ids_sha256"],
        )
        expected_semantic_keys.append(expected_semantic_key)
        if expected_semantic_key not in result_index_by_semantic_key:
            raise RuntimeError("rank publication result is foreign to the caller semantic inventory")
        row_index = result_index_by_semantic_key[expected_semantic_key]
        prompt_count = expected["prompt_token_count"]
        generated_count = int(tensors["generation_lengths"][row_index].item())
        total_count = int(tensors["unpadded_sequence_lengths"][row_index].item())
        if generated_count != envelope["requested_max_new_tokens"] or total_count != prompt_count + generated_count:
            raise RuntimeError("rank publication output lengths do not match the sealed request")
        if tensors["truncated"][row_index].item() is not True:
            raise RuntimeError("rank publication request did not terminate at its exact token limit")
        for tensor_key, expected_value in (
            ("generation_request_seeds", expected["seed"]),
            ("generation_global_request_indices", expected["global_request_index"]),
            ("generation_rounds", envelope["generation_round"]),
            ("generation_flattened_call_indices", envelope["call_index"]),
            ("generation_dp_ranks", envelope["dp_rank"]),
        ):
            if int(tensors[tensor_key][row_index].item()) != expected_value:
                raise RuntimeError(f"rank publication result drifted from {tensor_key}")
        prompt_token_ids = [int(value) for value in output_ids[row_index, :prompt_count].tolist()]
        prompt_sha256 = generation_prompt_token_ids_sha256(prompt_token_ids)
        if prompt_sha256 != expected["prompt_token_ids_sha256"]:
            raise RuntimeError("rank publication result prompt differs from the sealed caller prompt")
        output_token_ids = [
            int(value) for value in output_ids[row_index, prompt_count : prompt_count + generated_count].tolist()
        ]
        validity = tensors["generation_selected_logprob_valid"]
        if validity.dtype is not torch.bool or validity.shape != output_ids.shape:
            raise RuntimeError("rank publication selected-logprob validity mask is not token-aligned")
        valid_generated = validity[row_index, prompt_count : prompt_count + generated_count]
        valid_count = int(tensors["generation_selected_logprob_counts"][row_index].item())
        if (
            int(valid_generated.sum().item()) != generated_count
            or valid_count != generated_count
            or bool(validity[row_index, :prompt_count].any().item())
            or bool(validity[row_index, prompt_count + generated_count :].any().item())
        ):
            raise RuntimeError("rank publication selected-logprob validity evidence is incomplete")
        chosen_token_logprobs = [
            float(value) for value in logprobs[row_index, prompt_count : prompt_count + generated_count].tolist()
        ]
        if any(not math.isfinite(value) for value in chosen_token_logprobs):
            raise RuntimeError("rank publication chosen-token logprob is not finite")
        rows.append(
            {
                "schema_version": NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION,
                "request_id": expected["request_id"],
                "global_request_index": expected["global_request_index"],
                "request_index_in_dp_stream": expected["request_index_in_dp_stream"],
                "semantic_request_sha256": expected["semantic_request_sha256"],
                "engine_request_id": engine_request_ids[row_index],
                "seed": expected["seed"],
                "generation_round": envelope["generation_round"],
                "call_index": envelope["call_index"],
                "dp_rank": envelope["dp_rank"],
                "phase": envelope["phase"],
                "wave_index": envelope["wave_index"],
                "prompt_token_ids": prompt_token_ids,
                "output_token_ids": output_token_ids,
                "chosen_token_logprobs": chosen_token_logprobs,
                "selected_logprob_valid_count": valid_count,
                "requested_max_tokens": envelope["requested_max_new_tokens"],
                "observed_prompt_tokens": prompt_count,
                "observed_new_tokens": generated_count,
                "observed_total_tokens": total_count,
                "finish_reason": "length",
                "stopped_on_eos": False,
                "external_contract_sha256": envelope["external_contract_sha256"],
                "envelope_sha256": envelope["envelope_sha256"],
                "caller_prompt_anchor_sha256": envelope["caller_prompt_anchor_sha256"],
                "execution_occurrence_sha256": envelope["execution_occurrence_sha256"],
            }
        )
    if set(result_index_by_semantic_key) != set(expected_semantic_keys):
        raise RuntimeError("rank publication result semantic inventory is incomplete or foreign")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as compressed:
        for row in rows:
            compressed.write(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            compressed.write(b"\n")
    return buffer.getvalue(), tuple(rows)


class Evo2NemoRlGenerationWorkerImpl(VllmGenerationWorkerImpl):
    """Attach scheduler proof to each production NeMo-RL vLLM engine."""

    def __init__(self, *args, **kwargs) -> None:
        """Register Evo2 before NeMo-RL performs Transformers model probes."""
        from bionemo.evo2.vllm.plugin import register

        register()
        super().__init__(*args, **kwargs)

    def post_init(self):
        """Finish NeMo-RL initialization and attach the engine-local recorder."""
        result = super().post_init()
        self._evo2_rank_publication_available = False
        self._evo2_last_generation_result = None
        self._evo2_proof_enabled = bool(self.cfg.get("evo2_collect_proof", False))
        if self._evo2_proof_enabled:
            self._attach_evo2_proof_recorder()
        return result

    def generate(self, data, greedy: bool = False):
        """Retain one rank-local result for an explicit untimed publication RPC."""
        rank_publication_required = bool(getattr(self, "cfg", {}).get("evo2_rank_publication_required", False))
        if rank_publication_required and getattr(self, "_evo2_rank_publication_available", False):
            raise RuntimeError("cannot overwrite an unpublished rank-local generation result")
        result = super().generate(data, greedy=greedy)
        if rank_publication_required:
            input_ids = data.get("input_ids")
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("rank-local generation input_ids must be a tensor")
            if len(input_ids) == 0:
                self._evo2_last_generation_result = None
                self._evo2_rank_publication_available = False
            else:
                self._evo2_last_generation_result = result
                self._evo2_rank_publication_available = True
        return result

    def publish_evo2_generation_sidecar(
        self,
        *,
        envelope_payload: bytes,
        expected_envelope_sha256: str,
    ) -> dict[str, Any]:
        """Publish the retained local result under a caller-owned pre-launch plan."""
        if not getattr(self, "_evo2_rank_publication_available", False):
            raise RuntimeError("no unpublished rank-local generation result is available")
        normalized_envelope = _decode_rank_publication_envelope(
            envelope_payload,
            expected_envelope_sha256=expected_envelope_sha256,
        )
        result = self._evo2_last_generation_result
        self._evo2_rank_publication_available = False
        self._evo2_last_generation_result = None
        payload, rows = _rank_publication_payload(result, normalized_envelope)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        sibling_results = self.llm.collective_rpc(
            "attest_evo2_publication_payload",
            args=(payload, payload_sha256, normalized_envelope["envelope_sha256"]),
        )
        if type(sibling_results) is not list or len(sibling_results) != normalized_envelope["tensor_parallel_size"]:
            raise RuntimeError("rank publication TP sibling evidence does not match the topology")
        sibling_evidence = []
        for tp_rank, sibling in enumerate(sibling_results):
            if type(sibling) is not dict or set(sibling) != {
                "payload_sha256",
                "payload_size_bytes",
                "envelope_sha256",
            }:
                raise RuntimeError("rank publication TP sibling evidence fields are not exact")
            if (
                sibling["payload_sha256"] != payload_sha256
                or sibling["payload_size_bytes"] != len(payload)
                or sibling["envelope_sha256"] != normalized_envelope["envelope_sha256"]
            ):
                raise RuntimeError("rank publication TP siblings did not attest identical bytes and coordinates")
            sibling_evidence.append(
                {
                    "tp_rank": tp_rank,
                    "publisher": tp_rank == 0,
                    **sibling,
                }
            )
        provisional_receipt = publish_reserved_bytes(normalized_envelope["publication_plan"], payload)
        return {
            "schema_version": NEMO_RANK_PUBLICATION_OUTCOME_SCHEMA_VERSION,
            "publication_key": normalized_envelope["publication_key"],
            "external_contract_sha256": normalized_envelope["external_contract_sha256"],
            "envelope_sha256": normalized_envelope["envelope_sha256"],
            "caller_prompt_anchor_sha256": normalized_envelope["caller_prompt_anchor_sha256"],
            "execution_occurrence_sha256": normalized_envelope["execution_occurrence_sha256"],
            "phase": normalized_envelope["phase"],
            "wave_index": normalized_envelope["wave_index"],
            "generation_round": normalized_envelope["generation_round"],
            "call_index": normalized_envelope["call_index"],
            "dp_rank": normalized_envelope["dp_rank"],
            "publisher": True,
            "payload_sha256": payload_sha256,
            "payload_size_bytes": len(payload),
            "row_count": len(rows),
            "request_ids": [row["request_id"] for row in rows],
            "semantic_request_coordinates": [
                {
                    "request_id": row["request_id"],
                    "global_request_index": row["global_request_index"],
                    "request_index_in_dp_stream": row["request_index_in_dp_stream"],
                    "semantic_request_sha256": row["semantic_request_sha256"],
                    "engine_request_id": row["engine_request_id"],
                    "seed": row["seed"],
                    "generation_round": row["generation_round"],
                    "call_index": row["call_index"],
                    "dp_rank": row["dp_rank"],
                }
                for row in rows
            ],
            "tp_sibling_evidence": sibling_evidence,
            "provisional_receipt": provisional_receipt.to_dict(),
        }

    def snapshot_evo2_resolved_config(self) -> dict[str, Any]:
        """Return actual engine resolution without enabling timed proof instrumentation."""
        return resolved_config_snapshot(self.llm.llm_engine.vllm_config)

    def _attach_evo2_proof_recorder(self) -> None:
        """Attach one persistent scheduler recorder to the outer vLLM engine."""
        if not getattr(self, "_evo2_proof_enabled", False):
            raise RuntimeError("Evo2 proof collection is disabled for this generation worker")
        manager = self.llm.llm_engine.logger_manager
        if manager is None:
            raise RuntimeError("vLLM stat logger manager is disabled; Evo2 proof is unavailable")
        if hasattr(self, "_evo2_cudagraph_recorder"):
            return
        self._evo2_cudagraph_recorder = CUDAGraphProofRecorder()
        self._evo2_proof_phase = "unlabeled"
        self._evo2_observation_start = 0
        self._evo2_scheduler_observation_start = 0
        manager.stat_loggers.append(self._evo2_cudagraph_recorder)

    def reset_evo2_proof_phase(self, phase: str) -> dict[str, Any]:
        """Reset inner route/memory counters and label subsequent scheduler events."""
        if not phase:
            raise ValueError("proof phase cannot be empty")
        self._attach_evo2_proof_recorder()
        self._evo2_proof_phase = phase
        self._evo2_cudagraph_recorder.start_phase(phase)
        self._evo2_observation_start = len(self._evo2_cudagraph_recorder.observations)
        self._evo2_scheduler_observation_start = len(self._evo2_cudagraph_recorder.scheduler_observations)
        reset_prefix_sources = ".wave-" not in phase or phase.endswith(".wave-000")
        worker_reset = self.llm.collective_rpc(
            "reset_evo2_proof_state",
            args=(reset_prefix_sources,),
        )
        return {
            "phase": phase,
            "reset_prefix_sources": reset_prefix_sources,
            "worker_reset": worker_reset,
        }

    def snapshot_evo2_proof_phase(self, phase: str) -> dict[str, Any]:
        """Return untruncated outer graph events plus inner route/compile/memory state."""
        if not getattr(self, "_evo2_proof_enabled", False):
            raise RuntimeError("Evo2 proof collection is disabled for this generation worker")
        if phase != self._evo2_proof_phase:
            raise ValueError(f"requested proof phase {phase!r} does not match active {self._evo2_proof_phase!r}")
        observations = tuple(self._evo2_cudagraph_recorder.observations[self._evo2_observation_start :])
        scheduler_observations = tuple(
            self._evo2_cudagraph_recorder.scheduler_observations[self._evo2_scheduler_observation_start :]
        )
        worker_proof = self.llm.collective_rpc("snapshot_evo2_proof_state", args=())
        return {
            "phase": phase,
            "resolved_config": resolved_config_snapshot(self.llm.llm_engine.vllm_config),
            "cudagraph_observations": list(observations),
            "cudagraph_summary": summarize_cudagraph_observations(observations),
            "scheduler_observations": list(scheduler_observations),
            "worker_proof": worker_proof,
        }

    def reset_evo2_refit_phase(self, phase: str) -> dict[str, Any]:
        """Reset refit chunk telemetry on every internal TP worker."""
        worker_reset = self.llm.collective_rpc("reset_evo2_refit_proof_state", args=(phase,))
        return {"phase": phase, "worker_reset": worker_reset}

    def snapshot_evo2_refit_phase(self, phase: str) -> dict[str, Any]:
        """Return internal TP refit transactions plus stable actor/model identity."""
        worker_proof = self.llm.collective_rpc("snapshot_evo2_refit_proof_state", args=(phase,))
        device_uuids = self.llm.collective_rpc("report_device_id", args=())
        model_config = self.llm.llm_engine.model_config
        return {
            "phase": phase,
            "actor": {
                "ray_actor_id": ray.get_runtime_context().get_actor_id(),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "model": {
                "model": str(model_config.model),
                "architectures": list(model_config.architectures),
                "dtype": str(model_config.dtype),
                "max_model_len": int(model_config.max_model_len),
            },
            "device_uuids": device_uuids,
            "worker_proof": worker_proof,
        }


@ray.remote(runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")})
class Evo2NemoRlGenerationWorker(Evo2NemoRlGenerationWorkerImpl):
    """Ray actor used by the Evo2 NeMo-RL production generation path."""


__all__ = ["Evo2NemoRlGenerationWorker", "Evo2NemoRlGenerationWorkerImpl"]
