# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Named vLLM worker controls for Evo2 proof collection."""

import hashlib
import json
import math
import re
import struct
from typing import Any

import numpy as np


def selected_stream_sha256(
    vllm_request_id: str,
    output_token_ids: list[int],
    chosen_logprob_float32_bits: list[str],
) -> str:
    """Digest one exact sampled-token and processed-logprob bitstream."""
    if type(vllm_request_id) is not str or not vllm_request_id:
        raise TypeError("vLLM request ID must be a nonempty built-in string")
    if type(output_token_ids) is not list or any(type(token_id) is not int for token_id in output_token_ids):
        raise TypeError("selected stream token IDs must be built-in integer lists")
    if type(chosen_logprob_float32_bits) is not list or len(chosen_logprob_float32_bits) != len(
        output_token_ids
    ):
        raise TypeError("selected stream logprob bits must align with token IDs")
    if any(type(bits) is not str or re.fullmatch(r"[0-9a-f]{8}", bits) is None for bits in chosen_logprob_float32_bits):
        raise ValueError("selected stream logprob bits must be lowercase float32 hex")
    payload = json.dumps(
        {
            "schema_version": "evo2-selected-stream/v1",
            "vllm_request_id": vllm_request_id,
            "output_token_ids": output_token_ids,
            "chosen_logprob_float32_bits": chosen_logprob_float32_bits,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

class Evo2VllmWorkerExtension:
    """Expose trusted string-RPC proof controls without pickle serialization."""

    def reset_evo2_proof_state(self, reset_prefix_sources: bool = True) -> dict[str, Any]:
        """Reset phase-local FIR and CUDA-memory telemetry."""
        from bionemo.evo2.vllm.runner import reset_vllm_worker_proof_state

        if getattr(self, "_evo2_rank_local_generation_active", False):
            self.abort_evo2_rank_local_generation_evidence()
        return reset_vllm_worker_proof_state(self, reset_prefix_sources)

    def snapshot_evo2_proof_state(self) -> dict[str, Any]:
        """Return route, compile, and CUDA-memory evidence for this worker."""
        from bionemo.evo2.vllm.runner import snapshot_vllm_worker_proof_state

        return snapshot_vllm_worker_proof_state(self)

    def begin_evo2_rank_local_generation_evidence(
        self,
        phase: str,
        expected_envelope_sha256: str,
        expected_request_count: int,
        expected_max_new_tokens: int,
    ) -> dict[str, Any]:
        """Observe the model-runner boundary used by every Ray compiled-DAG TP rank."""
        if type(phase) is not str or not phase:
            raise TypeError("rank-local evidence phase must be a nonempty built-in string")
        if type(expected_envelope_sha256) is not str or re.fullmatch(
            r"[0-9a-f]{64}", expected_envelope_sha256
        ) is None:
            raise ValueError("rank-local evidence envelope SHA256 must be a lowercase digest")
        for label, value in (
            ("expected_request_count", expected_request_count),
            ("expected_max_new_tokens", expected_max_new_tokens),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive built-in integer")
        if getattr(self, "_evo2_rank_local_generation_active", False):
            raise RuntimeError("rank-local generation evidence is already active")
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None or not callable(
            getattr(model_runner, "execute_model", None)
        ):
            raise RuntimeError("rank-local generation evidence requires model_runner.execute_model")
        wrapped_runner = getattr(self, "_evo2_rank_local_wrapped_model_runner", None)
        if wrapped_runner is not None and wrapped_runner is not model_runner:
            raise RuntimeError("rank-local generation evidence model_runner identity changed")
        if wrapped_runner is None:
            original_execute_model = model_runner.execute_model
            original_sample_tokens = getattr(model_runner, "sample_tokens", None)

            def observe_result(result: Any) -> None:
                if not getattr(self, "_evo2_rank_local_generation_active", False):
                    return
                if result is None:
                    return
                if hasattr(result, "req_ids") or callable(
                    getattr(result, "get_output", None)
                ):
                    self._observe_evo2_rank_local_model_runner_output(result)

            def observed_execute_model(*args: Any, **kwargs: Any):
                result = original_execute_model(*args, **kwargs)
                observe_result(result)
                return result

            model_runner.execute_model = observed_execute_model
            self._evo2_rank_local_original_execute_model = original_execute_model
            if original_sample_tokens is not None:

                def observed_sample_tokens(*args: Any, **kwargs: Any):
                    result = original_sample_tokens(*args, **kwargs)
                    observe_result(result)
                    return result

                model_runner.sample_tokens = observed_sample_tokens
                self._evo2_rank_local_original_sample_tokens = original_sample_tokens
            self._evo2_rank_local_wrapped_model_runner = model_runner
        self._evo2_rank_local_generation_active = True
        self._evo2_rank_local_generation_state = {
            "phase": phase,
            "expected_envelope_sha256": expected_envelope_sha256,
            "expected_request_count": expected_request_count,
            "expected_max_new_tokens": expected_max_new_tokens,
            "execution_call_count": 0,
            "request_order": [],
            "requests": {},
        }
        tp_rank = self._evo2_tensor_parallel_rank()
        return {
            "tp_rank": int(tp_rank),
            "phase": phase,
            "expected_envelope_sha256": expected_envelope_sha256,
            "expected_request_count": expected_request_count,
            "expected_max_new_tokens": expected_max_new_tokens,
            "source": "rank_local_model_runner_execute_or_sample",
        }

    @staticmethod
    def _evo2_tensor_parallel_rank() -> int:
        """Return TP-group rank, with rank zero only for non-distributed CPU tests."""
        import torch

        if not torch.distributed.is_initialized():
            return 0
        from vllm.distributed.parallel_state import get_tp_group

        rank = get_tp_group().rank_in_group
        if type(rank) is not int or rank < 0:
            raise RuntimeError("vLLM tensor-parallel rank is unavailable or malformed")
        return rank

    def _observe_evo2_rank_local_model_runner_output(self, result: Any) -> None:
        if result is None:
            raise RuntimeError("rank-local TP observer received no ModelRunnerOutput")
        if callable(getattr(result, "get_output", None)):
            raise RuntimeError("rank-local TP observer does not admit asynchronous ModelRunnerOutput")
        req_ids = getattr(result, "req_ids", None)
        req_id_to_index = getattr(result, "req_id_to_index", None)
        sampled_token_ids = getattr(result, "sampled_token_ids", None)
        if (
            type(req_ids) is not list
            or any(type(req_id) is not str or not req_id for req_id in req_ids)
            or len(req_ids) != len(set(req_ids))
            or type(req_id_to_index) is not dict
            or req_id_to_index != {req_id: index for index, req_id in enumerate(req_ids)}
            or type(sampled_token_ids) is not list
            or len(sampled_token_ids) != len(req_ids)
        ):
            raise RuntimeError("rank-local ModelRunnerOutput request coordinates are malformed")
        state = self._evo2_rank_local_generation_state
        state["execution_call_count"] += 1
        logprobs = getattr(result, "logprobs", None)
        total_sampled_tokens = sum(len(tokens) for tokens in sampled_token_ids)
        if total_sampled_tokens:
            if logprobs is None:
                raise RuntimeError("rank-local ModelRunnerOutput omitted chosen-token logprobs")
            token_rows = getattr(logprobs, "logprob_token_ids", None)
            logprob_rows = getattr(logprobs, "logprobs", None)
            starts = getattr(logprobs, "cu_num_generated_tokens", None)
            if not isinstance(token_rows, np.ndarray) or not isinstance(logprob_rows, np.ndarray):
                raise RuntimeError("rank-local ModelRunnerOutput logprobs are not concrete CPU arrays")
            if token_rows.shape != logprob_rows.shape or token_rows.ndim != 2:
                raise RuntimeError("rank-local ModelRunnerOutput logprob arrays are not aligned")
            if starts is None:
                offsets = []
                offset = 0
                for tokens in sampled_token_ids:
                    offsets.append(offset)
                    offset += len(tokens)
            elif type(starts) is list and len(starts) == len(req_ids) and all(
                type(offset) is int and offset >= 0 for offset in starts
            ):
                offsets = starts
            else:
                raise RuntimeError("rank-local ModelRunnerOutput logprob offsets are malformed")
        else:
            offsets = [0] * len(req_ids)

        for request_index, (req_id, tokens) in enumerate(zip(req_ids, sampled_token_ids, strict=True)):
            if type(tokens) is not list or any(type(token_id) is not int for token_id in tokens):
                raise RuntimeError("rank-local sampled token IDs are malformed")
            if req_id not in state["requests"]:
                state["request_order"].append(req_id)
                state["requests"][req_id] = {"tokens": [], "logprob_bits": []}
            request_state = state["requests"][req_id]
            offset = offsets[request_index]
            for position, token_id in enumerate(tokens):
                row_index = offset + position
                if row_index >= len(token_rows):
                    raise RuntimeError("rank-local chosen-logprob row is out of range")
                matches = np.flatnonzero(token_rows[row_index] == token_id)
                if len(matches) != 1:
                    raise RuntimeError("rank-local chosen token is absent or duplicated in processed logprobs")
                value = np.float32(logprob_rows[row_index, int(matches[0])])
                if not math.isfinite(float(value)):
                    raise RuntimeError("rank-local chosen-token processed logprob is not finite")
                request_state["tokens"].append(token_id)
                request_state["logprob_bits"].append(struct.pack("<f", float(value)).hex())

    def snapshot_evo2_rank_local_generation_evidence(self) -> dict[str, Any]:
        """Consume independently reconstructed TP-rank output stream evidence."""
        if not getattr(self, "_evo2_rank_local_generation_active", False):
            raise RuntimeError("rank-local generation evidence is not active")
        state = self._evo2_rank_local_generation_state
        if len(state["requests"]) != state["expected_request_count"]:
            raise RuntimeError("rank-local TP evidence request count is incomplete")
        requests = []
        if (
            len(state["request_order"]) != len(state["requests"])
            or len(set(state["request_order"])) != len(state["request_order"])
            or set(state["request_order"]) != set(state["requests"])
        ):
            raise RuntimeError("rank-local TP evidence request order is malformed")
        for req_id in state["request_order"]:
            request_state = state["requests"][req_id]
            if len(request_state["tokens"]) != state["expected_max_new_tokens"]:
                raise RuntimeError("rank-local TP evidence generated-token count is incomplete")
            requests.append(
                {
                    "vllm_request_id": req_id,
                    "token_count": len(request_state["tokens"]),
                    "selected_stream_sha256": selected_stream_sha256(
                        req_id,
                        request_state["tokens"],
                        request_state["logprob_bits"],
                    ),
                }
            )
        aggregate_payload = json.dumps(requests, sort_keys=True, separators=(",", ":")).encode("utf-8")
        evidence = {
            "schema_version": "evo2-rank-local-generation-evidence/v1",
            "source": "rank_local_model_runner_execute_or_sample",
            "tp_rank": self._evo2_tensor_parallel_rank(),
            "phase": state["phase"],
            "expected_envelope_sha256": state["expected_envelope_sha256"],
            "request_count": len(requests),
            "generated_token_count": sum(request["token_count"] for request in requests),
            "execution_call_count": state["execution_call_count"],
            "request_order": list(state["request_order"]),
            "requests": requests,
            "aggregate_selected_stream_sha256": hashlib.sha256(aggregate_payload).hexdigest(),
        }
        self._evo2_rank_local_generation_active = False
        self._evo2_rank_local_generation_state = None
        return evidence

    def abort_evo2_rank_local_generation_evidence(self) -> dict[str, Any]:
        """Terminally discard one active witness epoch without publishing evidence."""
        active = bool(getattr(self, "_evo2_rank_local_generation_active", False))
        state = getattr(self, "_evo2_rank_local_generation_state", None)
        self._evo2_rank_local_generation_active = False
        self._evo2_rank_local_generation_state = None
        return {
            "tp_rank": self._evo2_tensor_parallel_rank(),
            "aborted": active,
            "phase": state.get("phase") if type(state) is dict else None,
        }


__all__ = ["Evo2VllmWorkerExtension", "selected_stream_sha256"]
