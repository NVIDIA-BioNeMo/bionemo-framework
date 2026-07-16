# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import gzip
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import ray.cloudpickle as ray_cloudpickle
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationRequestEnvelope,
    GenerationRequestIdentity,
    GenerationRequestLedgerAdmission,
    GenerationRequestScheduleAdmission,
    generation_prompt_token_ids_sha256,
)
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

from bionemo.evo2.vllm.artifact_io import (
    DuplicateJsonKeyError,
    cancel_file_publication_reservation,
    finalize_reserved_publication,
)
from bionemo.evo2.vllm.benchmark import WorkloadManifest
from bionemo.evo2.vllm.nemo_generation_worker import (
    Evo2NemoRlGenerationWorkerImpl,
    _rank_publication_payload,
)
from bionemo.evo2.vllm.nemo_publication_schema import (
    NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION,
)
from bionemo.evo2.vllm.nemo_runner import (
    build_nemo_generation_caller_ledgers,
    rank_publication_contract_sha256,
    rank_publication_envelope_payload,
    reserve_nemo_rank_publications,
)
from bionemo.evo2.vllm.profile import Evo2VllmProfile


DATA = Path(__file__).with_name("data") / "gdpo_mixed_96.json"


def _rank_publication_fixture(tmp_path):
    manifest = replace(
        WorkloadManifest.from_path(DATA).request_slice(0, 2),
        max_new_tokens=2,
    )
    profile = Evo2VllmProfile(
        topology="tp2",
        max_model_len=32,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=False,
        proof=True,
    )
    reservations, _plans, envelopes, _prompt_anchors = reserve_nemo_rank_publications(
        tmp_path / "proof.json",
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=0,
        global_call_index_start=0,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=build_nemo_generation_caller_ledgers(
            manifest=manifest,
            profile=profile,
            phases=("cold-generation",),
            generation_round=0,
            global_request_index_start=0,
            request_envelope_namespace="a" * 64,
        ),
    )
    key, envelope = next(iter(envelopes.items()))
    return manifest, reservations[key], envelope


def _local_generation_output(manifest, envelope) -> BatchedDataDict:
    max_total = max(len(request.prompt_token_ids) + manifest.max_new_tokens for request in manifest.requests)
    output_ids = torch.zeros((len(manifest.requests), max_total), dtype=torch.long)
    logprobs = torch.zeros((len(manifest.requests), max_total), dtype=torch.float32)
    selected_logprob_valid = torch.zeros((len(manifest.requests), max_total), dtype=torch.bool)
    for row, request in enumerate(manifest.requests):
        prompt_length = len(request.prompt_token_ids)
        output_ids[row, :prompt_length] = torch.tensor(request.prompt_token_ids)
        output_ids[row, prompt_length : prompt_length + 2] = torch.tensor([65 + row, 67 + row])
        logprobs[row, prompt_length : prompt_length + 2] = torch.tensor([-0.125, -0.25])
        selected_logprob_valid[row, prompt_length : prompt_length + 2] = True
    namespace_root = envelope["external_contract_sha256"]
    schedule = GenerationRequestScheduleAdmission(
        semantic_namespace_root=namespace_root,
        base_seed=manifest.seed,
        data_parallel_size=envelope["dp_size"],
        batch_call_slots_per_round=1,
        turns_per_call=1,
    )
    ledger = GenerationRequestLedgerAdmission(
        schedule_admission=schedule,
        generation_round_start=envelope["generation_round"],
        generation_round_count=1,
        global_request_index_start=envelope["requests"][0][
            "global_request_index"
        ],
        request_counts_by_batch_call_slot=(len(manifest.requests),),
    )
    request_envelope = GenerationRequestEnvelope(
        schedule_admission=schedule,
        caller_ledger_sha256=ledger.sha256,
        semantic_namespace=ledger.semantic_namespace_for(
            envelope["generation_round"],
            envelope["wave_index"],
        ),
        generation_round=envelope["generation_round"],
        batch_call_slot=envelope["wave_index"],
        turn_index=0,
        execution_attempt_epoch=0,
        requests=tuple(
            GenerationRequestIdentity(
                local_request_id=request.request_id,
                global_request_index=envelope["requests"][row]["global_request_index"],
                target_data_parallel_rank=envelope["dp_rank"],
                original_batch_ordinal=row,
                immutable_local_ordinal=row,
                prompt_token_ids_sha256=generation_prompt_token_ids_sha256(
                    request.prompt_token_ids
                ),
            )
            for row, request in enumerate(manifest.requests)
        ),
    )
    ledger.require_envelope(request_envelope)
    return BatchedDataDict(
        {
            "output_ids": output_ids,
            "logprobs": logprobs,
            "generation_lengths": torch.tensor([2, 2], dtype=torch.long),
            "unpadded_sequence_lengths": torch.tensor(
                [len(request.prompt_token_ids) + 2 for request in manifest.requests],
                dtype=torch.long,
            ),
            "truncated": torch.tensor([True, True]),
            "generation_request_seeds": torch.tensor(
                [request["seed"] for request in envelope["requests"]], dtype=torch.long
            ),
            "generation_global_request_indices": torch.tensor(
                [request["global_request_index"] for request in envelope["requests"]],
                dtype=torch.long,
            ),
            "generation_rounds": torch.tensor([0, 0], dtype=torch.long),
            "generation_batch_call_indices": torch.tensor(
                [0, 0], dtype=torch.long
            ),
            "generation_flattened_call_indices": torch.tensor(
                [0, 0], dtype=torch.long
            ),
            "generation_dp_ranks": torch.tensor([0, 0], dtype=torch.long),
            "generation_original_batch_ordinals": torch.tensor(
                [0, 1], dtype=torch.long
            ),
            "generation_immutable_local_ordinals": torch.tensor(
                [0, 1], dtype=torch.long
            ),
            "generation_base_seeds": torch.full(
                (2,), manifest.seed, dtype=torch.long
            ),
            "generation_data_parallel_sizes": torch.full(
                (2,), envelope["dp_size"], dtype=torch.long
            ),
            "generation_batch_call_slots_per_round": torch.ones(
                2, dtype=torch.long
            ),
            "generation_total_physical_calls_per_round": torch.ones(
                2, dtype=torch.long
            ),
            "generation_batch_call_slots": torch.full(
                (2,), envelope["wave_index"], dtype=torch.long
            ),
            "generation_turn_indices": torch.zeros(2, dtype=torch.long),
            "generation_turns_per_call": torch.ones(2, dtype=torch.long),
            "generation_execution_attempt_epochs": torch.zeros(
                2, dtype=torch.long
            ),
            "generation_local_request_ids": [
                request.request_id for request in manifest.requests
            ],
            "generation_engine_request_ids": [
                f"engine:{request.request_id}" for request in manifest.requests
            ],
            "generation_semantic_namespaces": [
                request_envelope.semantic_namespace
            ]
            * 2,
            "generation_schedule_semantic_namespace_roots": [namespace_root] * 2,
            "generation_prompt_token_ids_sha256": [
                request.prompt_token_ids_sha256
                for request in request_envelope.requests
            ],
            "generation_schedule_admission_sha256": [schedule.sha256] * 2,
            "generation_caller_ledger_sha256": [ledger.sha256] * 2,
            "generation_request_envelope_sha256": [request_envelope.sha256] * 2,
            "generation_execution_occurrence_sha256": [
                request_envelope.execution_occurrence_sha256
            ]
            * 2,
            "generation_request_envelope_schema_versions": [
                request_envelope.schema_version
            ]
            * 2,
            "generation_selected_logprob_valid": selected_logprob_valid,
            "generation_selected_logprob_counts": torch.tensor([2, 2], dtype=torch.long),
        }
    )


def _reverse_generation_output_rows(output: BatchedDataDict) -> BatchedDataDict:
    row_count = len(output["generation_lengths"])
    order = torch.arange(row_count - 1, -1, -1)
    reordered = {}
    for key, value in output.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == row_count
        ):
            reordered[key] = value[order]
        elif type(value) is list and len(value) == row_count:
            reordered[key] = [value[index] for index in order.tolist()]
        else:
            reordered[key] = value
    return BatchedDataDict(reordered)


@pytest.mark.parametrize(
    "field",
    ("generation_request_seeds", "generation_selected_logprob_counts"),
)
def test_rank_publication_rejects_integral_float_evidence_before_conversion(
    tmp_path,
    field,
) -> None:
    manifest, reservation, envelope = _rank_publication_fixture(tmp_path)
    output = _local_generation_output(manifest, envelope)
    output[field] = output[field].to(torch.float64)
    try:
        with pytest.raises(TypeError, match=field):
            _rank_publication_payload(output, envelope)
    finally:
        cancel_file_publication_reservation(reservation)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("mask_position", "validity|incomplete"),
        ("count", "counts|incomplete"),
        ("value_alignment", "logprobs|shape|aligned"),
    ),
)
def test_rank_publication_rejects_selected_logprob_alignment_mutations(
    tmp_path,
    mutation,
    message,
) -> None:
    manifest, reservation, envelope = _rank_publication_fixture(tmp_path)
    output = _local_generation_output(manifest, envelope)
    prompt_length = len(manifest.requests[0].prompt_token_ids)
    if mutation == "mask_position":
        output["generation_selected_logprob_valid"][0, prompt_length] = False
        output["generation_selected_logprob_valid"][0, prompt_length - 1] = True
    elif mutation == "count":
        output["generation_selected_logprob_counts"][0] = 1
    else:
        output["logprobs"] = output["logprobs"][:, :-1]
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError), match=message):
            _rank_publication_payload(output, envelope)
    finally:
        cancel_file_publication_reservation(reservation)


def test_rank_publication_ray_transport_preserves_exact_mixed_metadata(
    tmp_path,
) -> None:
    manifest, reservation, envelope = _rank_publication_fixture(tmp_path)
    output = _local_generation_output(manifest, envelope)
    transported = ray_cloudpickle.loads(ray_cloudpickle.dumps(output))
    try:
        assert isinstance(transported, BatchedDataDict)
        for key in (
            "generation_request_seeds",
            "generation_immutable_local_ordinals",
            "generation_selected_logprob_valid",
            "generation_selected_logprob_counts",
        ):
            assert isinstance(transported[key], torch.Tensor)
        for key in (
            "generation_local_request_ids",
            "generation_semantic_namespaces",
            "generation_prompt_token_ids_sha256",
            "generation_request_envelope_sha256",
            "generation_request_envelope_schema_versions",
        ):
            assert type(transported[key]) is list
            assert transported[key] == output[key]

        _payload, rows = _rank_publication_payload(transported, envelope)
        assert [row["request_id"] for row in rows] == [
            request.request_id for request in manifest.requests
        ]
    finally:
        cancel_file_publication_reservation(reservation)


def test_generation_worker_canonicalizes_reordered_backend_rows_before_sidecar_publish(
    tmp_path,
    monkeypatch,
) -> None:
    manifest, reservation, envelope = _rank_publication_fixture(tmp_path)
    output = _reverse_generation_output_rows(
        _local_generation_output(manifest, envelope)
    )
    assert output["generation_local_request_ids"] == [
        request.request_id for request in reversed(manifest.requests)
    ]
    monkeypatch.setattr(VllmGenerationWorkerImpl, "generate", lambda self, data, greedy=False: output)

    class FakeLLM:
        def collective_rpc(self, method, args=tuple()):
            assert method == "attest_evo2_publication_payload"
            payload, expected_sha256, envelope_sha256 = args
            assert hashlib.sha256(payload).hexdigest() == expected_sha256
            return [
                {
                    "payload_sha256": expected_sha256,
                    "payload_size_bytes": len(payload),
                    "envelope_sha256": envelope_sha256,
                }
                for _ in range(2)
            ]

    worker = object.__new__(Evo2NemoRlGenerationWorkerImpl)
    worker.llm = FakeLLM()
    worker.cfg = {"evo2_rank_publication_required": True}
    worker.generate(BatchedDataDict({"input_ids": torch.ones((2, 1), dtype=torch.long)}))

    evidence = worker.publish_evo2_generation_sidecar(
        envelope_payload=rank_publication_envelope_payload(envelope),
        expected_envelope_sha256=envelope["envelope_sha256"],
    )
    receipt = finalize_reserved_publication(reservation, evidence["provisional_receipt"])
    with gzip.open(receipt.final_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    assert evidence["publisher"] is True
    assert evidence["row_count"] == 2
    assert evidence["payload_sha256"] == receipt.sha256
    assert [item["publisher"] for item in evidence["tp_sibling_evidence"]] == [True, False]
    assert len({item["payload_sha256"] for item in evidence["tp_sibling_evidence"]}) == 1
    assert {row["schema_version"] for row in rows} == {
        NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION
    }
    assert [row["request_id"] for row in rows] == [request.request_id for request in manifest.requests]
    assert [row["semantic_request_sha256"] for row in rows] == [
        request["semantic_request_sha256"] for request in envelope["requests"]
    ]
    assert [row["engine_request_id"] for row in rows] == [
        f"engine:{request.request_id}" for request in manifest.requests
    ]
    assert rows[0]["prompt_token_ids"] == list(manifest.requests[0].prompt_token_ids)
    assert rows[0]["output_token_ids"] == [65, 67]
    assert rows[0]["chosen_token_logprobs"] == [-0.125, -0.25]
    assert rows[0]["selected_logprob_valid_count"] == 2
    assert rows[1]["seed"] == envelope["requests"][1]["seed"]


def test_generation_worker_refuses_to_overwrite_unpublished_result_or_replay_sidecar(tmp_path, monkeypatch) -> None:
    manifest, reservation, envelope = _rank_publication_fixture(tmp_path)
    output = _local_generation_output(manifest, envelope)
    monkeypatch.setattr(VllmGenerationWorkerImpl, "generate", lambda self, data, greedy=False: output)
    worker = object.__new__(Evo2NemoRlGenerationWorkerImpl)
    worker.cfg = {"evo2_rank_publication_required": True}
    worker.llm = type(
        "FakeLLM",
        (),
        {
            "collective_rpc": lambda self, method, args=tuple(): [
                {
                    "payload_sha256": args[1],
                    "payload_size_bytes": len(args[0]),
                    "envelope_sha256": args[2],
                }
                for _ in range(2)
            ]
        },
    )()
    batch = BatchedDataDict({"input_ids": torch.ones((2, 1), dtype=torch.long)})
    worker.generate(batch)
    with pytest.raises(RuntimeError, match="unpublished"):
        worker.generate(batch)

    evidence = worker.publish_evo2_generation_sidecar(
        envelope_payload=rank_publication_envelope_payload(envelope),
        expected_envelope_sha256=envelope["envelope_sha256"],
    )
    finalize_reserved_publication(reservation, evidence["provisional_receipt"])
    with pytest.raises(RuntimeError, match="available|replay|published"):
        worker.publish_evo2_generation_sidecar(
            envelope_payload=rank_publication_envelope_payload(envelope),
            expected_envelope_sha256=envelope["envelope_sha256"],
        )


def test_inactive_empty_shard_can_reactivate_without_stale_unpublished_state(
    tmp_path,
    monkeypatch,
) -> None:
    manifest, reservation, envelope = _rank_publication_fixture(tmp_path)
    active_output = _local_generation_output(manifest, envelope)

    def fake_generate(self, data, greedy=False):
        del self, greedy
        if len(data["input_ids"]) == 0:
            return BatchedDataDict(
                {
                    "output_ids": torch.zeros((0, 0), dtype=torch.long),
                    "logprobs": torch.zeros((0, 0), dtype=torch.float32),
                    "generation_lengths": torch.zeros(0, dtype=torch.long),
                    "unpadded_sequence_lengths": torch.zeros(0, dtype=torch.long),
                    "truncated": torch.zeros(0, dtype=torch.bool),
                    "generation_selected_logprob_valid": torch.zeros(
                        (0, 0), dtype=torch.bool
                    ),
                    "generation_selected_logprob_counts": torch.zeros(
                        0, dtype=torch.long
                    ),
                }
            )
        return active_output

    monkeypatch.setattr(VllmGenerationWorkerImpl, "generate", fake_generate)
    worker = object.__new__(Evo2NemoRlGenerationWorkerImpl)
    worker.cfg = {"evo2_rank_publication_required": True}
    worker.llm = type(
        "FakeLLM",
        (),
        {
            "collective_rpc": lambda self, method, args=tuple(): [
                {
                    "payload_sha256": args[1],
                    "payload_size_bytes": len(args[0]),
                    "envelope_sha256": args[2],
                }
                for _ in range(2)
            ]
        },
    )()

    worker.generate(BatchedDataDict({"input_ids": torch.zeros((0, 1), dtype=torch.long)}))
    worker.generate(BatchedDataDict({"input_ids": torch.ones((2, 1), dtype=torch.long)}))
    evidence = worker.publish_evo2_generation_sidecar(
        envelope_payload=rank_publication_envelope_payload(envelope),
        expected_envelope_sha256=envelope["envelope_sha256"],
    )
    finalize_reserved_publication(reservation, evidence["provisional_receipt"])

    assert evidence["row_count"] == 2
    assert worker._evo2_rank_publication_available is False
    assert worker._evo2_last_generation_result is None


def test_generation_worker_rejects_coherent_worker_envelope_rewrite_against_coordinator_anchor(
    tmp_path,
    monkeypatch,
) -> None:
    manifest, reservation, envelope = _rank_publication_fixture(tmp_path)
    output = _local_generation_output(manifest, envelope)
    monkeypatch.setattr(VllmGenerationWorkerImpl, "generate", lambda self, data, greedy=False: output)
    worker = object.__new__(Evo2NemoRlGenerationWorkerImpl)
    worker.cfg = {"evo2_rank_publication_required": True}
    worker.generate(BatchedDataDict({"input_ids": torch.ones((2, 1), dtype=torch.long)}))

    rewritten = deepcopy(envelope)
    rewritten["generation_round"] = 1
    rewritten["envelope_sha256"] = rank_publication_contract_sha256(rewritten)
    try:
        with pytest.raises(RuntimeError, match="coordinator-owned envelope digest"):
            worker.publish_evo2_generation_sidecar(
                envelope_payload=rank_publication_envelope_payload(rewritten),
                expected_envelope_sha256=envelope["envelope_sha256"],
            )
        assert not Path(envelope["publication_plan"]["final_path"]).exists()
    finally:
        cancel_file_publication_reservation(reservation)


def test_generation_worker_rejects_duplicate_wire_keys_before_envelope_materialization(
    tmp_path,
    monkeypatch,
) -> None:
    manifest, reservation, envelope = _rank_publication_fixture(tmp_path)
    output = _local_generation_output(manifest, envelope)
    monkeypatch.setattr(VllmGenerationWorkerImpl, "generate", lambda self, data, greedy=False: output)
    worker = object.__new__(Evo2NemoRlGenerationWorkerImpl)
    worker.cfg = {"evo2_rank_publication_required": True}
    worker.generate(BatchedDataDict({"input_ids": torch.ones((2, 1), dtype=torch.long)}))
    payload = rank_publication_envelope_payload(envelope)
    duplicate = payload.replace(
        b'"phase":"cold-generation"',
        b'"phase":"foreign","phase":"cold-generation"',
        1,
    )
    assert duplicate != payload
    try:
        with pytest.raises(DuplicateJsonKeyError, match="phase"):
            worker.publish_evo2_generation_sidecar(
                envelope_payload=duplicate,
                expected_envelope_sha256=envelope["envelope_sha256"],
            )
        assert not Path(envelope["publication_plan"]["final_path"]).exists()
    finally:
        cancel_file_publication_reservation(reservation)
