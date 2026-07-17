# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import builtins
import gzip
import hashlib
import json
import os
import random
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationRequestEnvelope,
    GenerationRequestIdentity,
    GenerationRequestLedgerAdmission,
    GenerationRequestScheduleAdmission,
    generation_prompt_token_ids_sha256,
)

import bionemo.evo2.vllm.nemo_runner as nemo_runner
import bionemo.evo2.vllm.runner as runner
import bionemo.evo2.vllm.artifact_io as artifact_io
from bionemo.evo2.vllm.artifact_io import cancel_file_publication_reservation, publish_reserved_bytes
from bionemo.evo2.vllm.benchmark import (
    GenerationRecord,
    WorkloadManifest,
    WorkloadRequest,
    build_request_waves,
)
from bionemo.evo2.vllm.nemo_runner import (
    NemoCallerPromptAnchorRoot,
    build_nemo_generation_config,
    build_nemo_generation_input,
    reserve_nemo_rank_publications,
    full_vocab_logprob_evidence_from_nemo_output,
    finalize_nemo_rank_publications,
    publish_nemo_rank_wave,
    rank_publication_contract_sha256,
    records_from_nemo_generation_output,
    run_nemo_generation_phase,
    snapshot_and_validate_nemo_resolved_configs,
    terminalize_failed_nemo_rank_publications,
)
from bionemo.evo2.vllm.nemo_publication_schema import (
    NEMO_RANK_PUBLICATION_OUTCOME_SCHEMA_VERSION,
    NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION,
)
from bionemo.evo2.vllm.profile import Evo2VllmProfile
from bionemo.evo2.vllm.runner import PeakMemoryMonitor


DATA = Path(__file__).with_name("data") / "gdpo_mixed_96.json"
_INDEPENDENT_REQUEST_SEED_STRIDE = 1_000_003
_INDEPENDENT_REQUEST_SEED_MODULUS = 2**31


def _require(condition: bool, message: str) -> None:
    if type(condition) is not bool or not condition:
        raise AssertionError(message)


def _independent_request_seed(
    *,
    base_seed: int,
    global_call_index: int,
    dp_rank: int,
    dp_size: int,
    immutable_local_ordinal: int,
) -> int:
    return (
        base_seed
        + (global_call_index * dp_size + dp_rank)
        * _INDEPENDENT_REQUEST_SEED_STRIDE
        + immutable_local_ordinal
    ) % _INDEPENDENT_REQUEST_SEED_MODULUS


def _independent_rank_prompt_anchor_sha256(envelope: dict[str, object]) -> str:
    payload = {
        "schema_version": "evo2-nemo-rank-prompt-anchor/v1",
        "publication_key": envelope["publication_key"],
        "manifest_sha256": envelope["manifest_sha256"],
        "phase": envelope["phase"],
        "wave_index": envelope["wave_index"],
        "generation_round": envelope["generation_round"],
        "call_index": envelope["call_index"],
        "dp_rank": envelope["dp_rank"],
        "requests": [
            {
                field: request[field]
                for field in (
                    "request_id",
                    "global_request_index",
                    "request_index_in_dp_stream",
                    "prompt_token_count",
                    "prompt_token_ids_sha256",
                )
            }
            for request in envelope["requests"]
        ],
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _coherent_foreign_prompt_tree(
    *,
    manifest: WorkloadManifest,
    caller_root: NemoCallerPromptAnchorRoot,
    envelopes: dict[str, dict[str, object]],
    request_index: int,
    foreign_prompt_token_ids: tuple[int, ...],
) -> tuple[WorkloadManifest, NemoCallerPromptAnchorRoot, dict[str, dict[str, object]]]:
    original_request = manifest.requests[request_index]
    if (
        type(foreign_prompt_token_ids) is not tuple
        or len(foreign_prompt_token_ids) != len(original_request.prompt_token_ids)
        or foreign_prompt_token_ids == original_request.prompt_token_ids
    ):
        raise AssertionError("foreign prompt fixture must use distinct same-length bytes")
    foreign_manifest = replace(
        manifest,
        requests=tuple(
            replace(request, prompt_token_ids=foreign_prompt_token_ids)
            if index == request_index
            else request
            for index, request in enumerate(manifest.requests)
        ),
    )
    foreign_prompt_sha256 = generation_prompt_token_ids_sha256(
        foreign_prompt_token_ids
    )
    original_prompt_sha256 = generation_prompt_token_ids_sha256(
        original_request.prompt_token_ids
    )
    _require(
        foreign_prompt_sha256 != original_prompt_sha256,
        "foreign prompt fixture did not change the canonical prompt digest",
    )
    semantic_replacements = {}
    foreign_semantic_requests = []
    for semantic in caller_root.semantic_requests:
        foreign_semantic = (
            replace(
                semantic,
                prompt_token_count=len(foreign_prompt_token_ids),
                prompt_token_ids_sha256=foreign_prompt_sha256,
            )
            if semantic.request_id == original_request.request_id
            else semantic
        )
        semantic_replacements[semantic.sha256] = foreign_semantic
        foreign_semantic_requests.append(foreign_semantic)
    foreign_occurrences = tuple(
        replace(
            occurrence,
            requests=tuple(
                semantic_replacements[request.sha256]
                for request in occurrence.requests
            ),
        )
        for occurrence in caller_root.execution_occurrences
    )
    payloads_by_sha256 = {}
    for request in foreign_manifest.requests:
        prompt_sha256 = generation_prompt_token_ids_sha256(request.prompt_token_ids)
        payloads_by_sha256[prompt_sha256] = nemo_runner.NemoPromptPayloadRecord(
            prompt_token_ids=tuple(request.prompt_token_ids),
            prompt_token_ids_sha256=prompt_sha256,
        )
    foreign_root = replace(
        caller_root,
        manifest_sha256=foreign_manifest.sha256,
        external_contract_sha256="b" * 64,
        prompt_payload_catalog=tuple(
            payloads_by_sha256[digest] for digest in sorted(payloads_by_sha256)
        ),
        semantic_requests=tuple(foreign_semantic_requests),
        execution_occurrences=foreign_occurrences,
    )
    foreign_envelopes = deepcopy(envelopes)
    for publication_key, envelope in foreign_envelopes.items():
        foreign_occurrence = foreign_root.require_occurrence(publication_key)
        semantic_by_global_index = {
            request.global_request_index: request
            for request in foreign_occurrence.requests
        }
        for request in envelope["requests"]:
            semantic = semantic_by_global_index[request["global_request_index"]]
            request.update(
                {
                    "request_id": semantic.request_id,
                    "request_index_in_dp_stream": semantic.request_index_in_dp_stream,
                    "prompt_token_count": semantic.prompt_token_count,
                    "prompt_token_ids_sha256": semantic.prompt_token_ids_sha256,
                    "semantic_request_sha256": semantic.sha256,
                    "seed": semantic.request_seed,
                }
            )
        envelope.update(
            {
                "manifest_sha256": foreign_manifest.sha256,
                "external_contract_sha256": foreign_root.external_contract_sha256,
                "caller_prompt_anchor_sha256": foreign_root.sha256,
                "execution_occurrence_sha256": foreign_occurrence.sha256,
            }
        )
        envelope["envelope_sha256"] = rank_publication_contract_sha256(envelope)
        plan = dict(envelope["publication_plan"])
        plan.update(
            {
                "external_contract_sha256": foreign_root.external_contract_sha256,
                "payload_contract_sha256": envelope["envelope_sha256"],
                "marker_sha256": "0" * 64,
            }
        )
        unsigned_plan = artifact_io.FilePublicationPlan(**plan)
        plan["marker_sha256"] = hashlib.sha256(
            artifact_io._file_publication_marker_payload(unsigned_plan)
        ).hexdigest()
        envelope["publication_plan"] = plan
    return foreign_manifest, foreign_root, foreign_envelopes


def _explicit_generation_metadata(
    manifest: WorkloadManifest,
    *,
    generation_round: int = 0,
    batch_call_slots_per_round: int = 1,
    batch_call_slot: int = 0,
    global_request_indices: tuple[int, ...] | None = None,
    dp_ranks: tuple[int, ...] | None = None,
    immutable_local_ordinals: tuple[int, ...] | None = None,
    data_parallel_size: int = 1,
) -> dict[str, object]:
    request_count = len(manifest.requests)
    if global_request_indices is None:
        global_request_indices = tuple(range(request_count))
    if dp_ranks is None:
        dp_ranks = (0,) * request_count
    if immutable_local_ordinals is None:
        immutable_local_ordinals = tuple(range(request_count))
    schedule = GenerationRequestScheduleAdmission(
        semantic_namespace_root="test",
        base_seed=manifest.seed,
        data_parallel_size=data_parallel_size,
        batch_call_slots_per_round=batch_call_slots_per_round,
        turns_per_call=1,
    )
    global_request_index_start = (
        global_request_indices[0] - batch_call_slot * request_count
    )
    ledger = GenerationRequestLedgerAdmission(
        schedule_admission=schedule,
        generation_round_start=generation_round,
        generation_round_count=1,
        global_request_index_start=global_request_index_start,
        request_counts_by_batch_call_slot=(request_count,)
        * batch_call_slots_per_round,
    )
    envelope = GenerationRequestEnvelope(
        schedule_admission=schedule,
        caller_ledger_sha256=ledger.sha256,
        semantic_namespace=ledger.semantic_namespace_for(
            generation_round,
            batch_call_slot,
        ),
        generation_round=generation_round,
        batch_call_slot=batch_call_slot,
        turn_index=0,
        execution_attempt_epoch=0,
        requests=tuple(
            GenerationRequestIdentity(
                local_request_id=request.request_id,
                global_request_index=global_request_indices[index],
                target_data_parallel_rank=dp_ranks[index],
                original_batch_ordinal=index,
                immutable_local_ordinal=immutable_local_ordinals[index],
                prompt_token_ids_sha256=generation_prompt_token_ids_sha256(
                    request.prompt_token_ids
                ),
            )
            for index, request in enumerate(manifest.requests)
        ),
    )
    ledger.require_envelope(envelope)
    return {
        "generation_request_seeds": torch.tensor(
            [
                _independent_request_seed(
                    base_seed=manifest.seed,
                    global_call_index=envelope.flattened_call_index,
                    dp_rank=request.target_data_parallel_rank,
                    dp_size=data_parallel_size,
                    immutable_local_ordinal=request.immutable_local_ordinal,
                )
                for request in envelope.requests
            ],
            dtype=torch.int64,
        ),
        "generation_global_request_indices": torch.tensor(
            global_request_indices, dtype=torch.int64
        ),
        "generation_rounds": torch.full(
            (request_count,), generation_round, dtype=torch.int64
        ),
        "generation_batch_call_indices": torch.full(
            (request_count,), envelope.batch_call_index, dtype=torch.int64
        ),
        "generation_flattened_call_indices": torch.full(
            (request_count,), envelope.flattened_call_index, dtype=torch.int64
        ),
        "generation_global_call_index_starts": torch.full(
            (request_count,),
            envelope.schedule_admission.global_call_index_start,
            dtype=torch.int64,
        ),
        "generation_dp_ranks": torch.tensor(dp_ranks, dtype=torch.int64),
        "generation_original_batch_ordinals": torch.arange(
            request_count, dtype=torch.int64
        ),
        "generation_immutable_local_ordinals": torch.tensor(
            immutable_local_ordinals, dtype=torch.int64
        ),
        "generation_base_seeds": torch.full(
            (request_count,), manifest.seed, dtype=torch.int64
        ),
        "generation_data_parallel_sizes": torch.full(
            (request_count,), data_parallel_size, dtype=torch.int64
        ),
        "generation_batch_call_slots_per_round": torch.full(
            (request_count,), batch_call_slots_per_round, dtype=torch.int64
        ),
        "generation_total_physical_calls_per_round": torch.full(
            (request_count,), batch_call_slots_per_round, dtype=torch.int64
        ),
        "generation_batch_call_slots": torch.full(
            (request_count,), batch_call_slot, dtype=torch.int64
        ),
        "generation_turn_indices": torch.zeros(
            request_count, dtype=torch.int64
        ),
        "generation_turns_per_call": torch.ones(
            request_count, dtype=torch.int64
        ),
        "generation_execution_attempt_epochs": torch.zeros(
            request_count, dtype=torch.int64
        ),
        "generation_local_request_ids": [
            request.request_id for request in manifest.requests
        ],
        "generation_semantic_namespaces": [
            envelope.semantic_namespace
        ]
        * request_count,
        "generation_schedule_semantic_namespace_roots": [
            schedule.semantic_namespace_root
        ]
        * request_count,
        "generation_prompt_token_ids_sha256": [
            request.prompt_token_ids_sha256 for request in envelope.requests
        ],
        "generation_schedule_admission_sha256": [schedule.sha256]
        * request_count,
        "generation_caller_ledger_sha256": [ledger.sha256] * request_count,
        "generation_request_envelope_sha256": [envelope.sha256] * request_count,
        "generation_execution_occurrence_sha256": [
            envelope.execution_occurrence_sha256
        ]
        * request_count,
        "generation_request_envelope_schema_versions": [
            envelope.schema_version
        ]
        * request_count,
    }


def _metadata_from_request_envelope(
    envelope: GenerationRequestEnvelope,
) -> dict[str, object]:
    request_count = len(envelope.requests)
    return {
        "generation_request_seeds": torch.tensor(
            [
                _independent_request_seed(
                    base_seed=envelope.base_seed,
                    global_call_index=envelope.flattened_call_index,
                    dp_rank=request.target_data_parallel_rank,
                    dp_size=envelope.data_parallel_size,
                    immutable_local_ordinal=request.immutable_local_ordinal,
                )
                for request in envelope.requests
            ],
            dtype=torch.int64,
        ),
        "generation_global_request_indices": torch.tensor(
            [request.global_request_index for request in envelope.requests],
            dtype=torch.int64,
        ),
        "generation_rounds": torch.full(
            (request_count,), envelope.generation_round, dtype=torch.int64
        ),
        "generation_batch_call_indices": torch.full(
            (request_count,), envelope.batch_call_index, dtype=torch.int64
        ),
        "generation_flattened_call_indices": torch.full(
            (request_count,), envelope.flattened_call_index, dtype=torch.int64
        ),
        "generation_global_call_index_starts": torch.full(
            (request_count,),
            envelope.schedule_admission.global_call_index_start,
            dtype=torch.int64,
        ),
        "generation_dp_ranks": torch.tensor(
            [
                request.target_data_parallel_rank
                for request in envelope.requests
            ],
            dtype=torch.int64,
        ),
        "generation_original_batch_ordinals": torch.tensor(
            [request.original_batch_ordinal for request in envelope.requests],
            dtype=torch.int64,
        ),
        "generation_immutable_local_ordinals": torch.tensor(
            [request.immutable_local_ordinal for request in envelope.requests],
            dtype=torch.int64,
        ),
        "generation_base_seeds": torch.full(
            (request_count,), envelope.base_seed, dtype=torch.int64
        ),
        "generation_data_parallel_sizes": torch.full(
            (request_count,), envelope.data_parallel_size, dtype=torch.int64
        ),
        "generation_batch_call_slots_per_round": torch.full(
            (request_count,),
            envelope.batch_call_slots_per_round,
            dtype=torch.int64,
        ),
        "generation_total_physical_calls_per_round": torch.full(
            (request_count,),
            envelope.total_physical_calls_per_round,
            dtype=torch.int64,
        ),
        "generation_batch_call_slots": torch.full(
            (request_count,), envelope.batch_call_slot, dtype=torch.int64
        ),
        "generation_turn_indices": torch.full(
            (request_count,), envelope.turn_index, dtype=torch.int64
        ),
        "generation_turns_per_call": torch.full(
            (request_count,), envelope.turns_per_call, dtype=torch.int64
        ),
        "generation_execution_attempt_epochs": torch.full(
            (request_count,), envelope.execution_attempt_epoch, dtype=torch.int64
        ),
        "generation_local_request_ids": [
            request.local_request_id for request in envelope.requests
        ],
        "generation_engine_request_ids": [
            (
                f"{envelope.execution_occurrence_sha256}:"
                f"{request.target_data_parallel_rank}:"
                f"{request.immutable_local_ordinal}"
            )
            for request in envelope.requests
        ],
        "generation_semantic_namespaces": [
            envelope.semantic_namespace
        ]
        * request_count,
        "generation_schedule_semantic_namespace_roots": [
            envelope.schedule_admission.semantic_namespace_root
        ]
        * request_count,
        "generation_prompt_token_ids_sha256": [
            request.prompt_token_ids_sha256 for request in envelope.requests
        ],
        "generation_schedule_admission_sha256": [
            envelope.schedule_admission.sha256
        ]
        * request_count,
        "generation_caller_ledger_sha256": [
            envelope.caller_ledger_sha256
        ]
        * request_count,
        "generation_request_envelope_sha256": [envelope.sha256] * request_count,
        "generation_execution_occurrence_sha256": [
            envelope.execution_occurrence_sha256
        ]
        * request_count,
        "generation_request_envelope_schema_versions": [
            envelope.schema_version
        ]
        * request_count,
    }


def _manifest_with_request_count(request_count: int) -> WorkloadManifest:
    base = WorkloadManifest.from_path(DATA).with_max_new_tokens(6_000)
    requests = tuple(
        WorkloadRequest(
            request_id=f"audit-{index:04d}",
            prompt_token_ids=base.requests[index % len(base.requests)].prompt_token_ids,
        )
        for index in range(request_count)
    )
    return replace(base, requests=requests)


def _phase_caller_ledger(
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    *,
    phase: str,
    generation_round: int,
    global_request_index_start: int,
    request_envelope_namespace: str,
) -> GenerationRequestLedgerAdmission:
    return nemo_runner.build_nemo_generation_caller_ledgers(
        manifest=manifest,
        profile=profile,
        phases=(phase,),
        generation_round=generation_round,
        global_request_index_start=global_request_index_start,
        request_envelope_namespace=request_envelope_namespace,
    )[phase]


def _reservation_caller_ledgers(
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    *,
    phases: tuple[str, ...],
    generation_round: int,
    external_contract_sha256: str,
    global_request_index_start: int = 0,
) -> dict[str, GenerationRequestLedgerAdmission]:
    return nemo_runner.build_nemo_generation_caller_ledgers(
        manifest=manifest,
        profile=profile,
        phases=phases,
        generation_round=generation_round,
        global_request_index_start=global_request_index_start,
        request_envelope_namespace=external_contract_sha256,
    )


def test_multi_phase_caller_ledgers_use_disjoint_caller_phase_coordinates(tmp_path) -> None:
    manifest = _manifest_with_request_count(1_000)
    external_contract_sha256 = "a" * 64
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=6_012,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=96,
        max_num_seqs=48,
    )
    coordinates = runner.benchmark_phase_coordinates(
        manifest,
        profile,
        generation_round_start=3,
        warmups=0,
        repetitions=2,
    )
    coordinates_by_phase = {row["phase"]: row for row in coordinates}

    ledgers = nemo_runner.build_nemo_generation_caller_ledgers(
        manifest=manifest,
        profile=profile,
        phases=tuple(coordinates_by_phase),
        generation_round=3,
        global_request_index_start=3_000,
        request_envelope_namespace=external_contract_sha256,
        phase_coordinates_by_name=coordinates_by_phase,
    )

    for phase, coordinate in coordinates_by_phase.items():
        ledger = ledgers[phase]
        _require(
            ledger.generation_round_start == coordinate["generation_round"],
            "DP2 phase ledger reused a generation round",
        )
        _require(
            ledger.global_request_index_start == coordinate["global_request_index_start"],
            "DP2 phase ledger reused a global request range",
        )
        _require(
            ledger.schedule_admission.batch_call_slots_per_round == 11,
            "DP2 phase ledger did not freeze eleven physical calls",
        )
    cold = ledgers["cold-generation"]
    steady_0 = ledgers["steady-0"]
    steady_1 = ledgers["steady-1"]
    _require(
        [
            cold.generation_round_start,
            steady_0.generation_round_start,
            steady_1.generation_round_start,
        ]
        == [3, 4, 5],
        "DP2 cold and steady rounds did not advance",
    )
    _require(
        [
            cold.global_request_index_start,
            steady_0.global_request_index_start,
            steady_1.global_request_index_start,
        ]
        == [3_000, 4_000, 5_000],
        "DP2 cold and steady request ranges did not advance",
    )
    reservations, _plans, envelopes, _caller_root = reserve_nemo_rank_publications(
        tmp_path / "proof.json",
        manifest=manifest,
        profile=profile,
        phases=tuple(coordinates_by_phase),
        generation_round=3,
        global_call_index_start=33,
        global_request_index_start=3_000,
        external_contract_sha256=external_contract_sha256,
        caller_ledgers_by_phase=ledgers,
        phase_coordinates_by_name=coordinates_by_phase,
    )
    try:
        for phase, coordinate in coordinates_by_phase.items():
            phase_envelopes = [
                envelope for envelope in envelopes.values() if envelope["phase"] == phase
            ]
            _require(
                {envelope["generation_round"] for envelope in phase_envelopes}
                == {coordinate["generation_round"]},
                "rank reservations reused a phase generation round",
            )
            _require(
                sorted({envelope["call_index"] for envelope in phase_envelopes})
                == list(
                    range(
                        coordinate["global_call_index_start"],
                        coordinate["global_call_index_start"] + 11,
                    )
                ),
                "rank reservations reused or skipped physical calls",
            )
            _require(
                min(envelope["global_wave_start"] for envelope in phase_envelopes)
                == coordinate["global_request_index_start"],
                "rank reservations reused a global request range",
            )
    finally:
        for reservation in reservations.values():
            cancel_file_publication_reservation(reservation)


def test_rank_publication_reservations_cover_exact1k_calls_and_dp_tail_before_launch(tmp_path) -> None:
    manifest = _manifest_with_request_count(1_000)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=6_012,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
    )

    reservations, plans, envelopes, _prompt_anchors = reserve_nemo_rank_publications(
        tmp_path / "proof.json",
        manifest=manifest,
        profile=profile,
        phases=("cold-generation", "steady-0"),
        generation_round=1,
        global_call_index_start=11,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=_reservation_caller_ledgers(
            manifest,
            profile,
            phases=("cold-generation", "steady-0"),
            generation_round=1,
            external_contract_sha256="a" * 64,
        ),
    )
    try:
        _require(
            len(reservations) == len(plans) == len(envelopes) == 44,
            "exact1k did not reserve every phase/wave/DP publication",
        )
        _require(
            set(reservations) == set(plans) == set(envelopes),
            "exact1k reservation, plan, and envelope inventories differ",
        )
        _require(
            all(Path(plan.marker_path).exists() for plan in plans.values()),
            "an exact1k reservation marker is missing before launch",
        )
        _require(
            all(not Path(plan.final_path).exists() for plan in plans.values()),
            "an exact1k final artifact exists before launch",
        )
        cold = [envelope for envelope in envelopes.values() if envelope["phase"] == "cold-generation"]
        _require(
            sorted({envelope["call_index"] for envelope in cold})
            == list(range(11, 22)),
            "exact1k physical calls are not exactly 11 through 21",
        )
        tail = [envelope for envelope in cold if envelope["wave_index"] == 10]
        ordered_tail = sorted(tail, key=lambda item: item["dp_rank"])
        _require(
            [envelope["expected_request_count"] for envelope in ordered_tail]
            == [20, 20],
            "exact1k DP2 tail is not exactly 20/20",
        )
        _require(
            [envelope["dp_rank"] for envelope in ordered_tail] == [0, 1],
            "exact1k tail does not contain the exact two DP ranks",
        )
        _require(
            sum(envelope["expected_request_count"] for envelope in cold) == 1_000,
            "exact1k cold phase does not own exactly 1,000 requests",
        )
        for envelope in cold:
            _require(
                envelope["external_contract_sha256"] == "a" * 64,
                "exact1k envelope lost the external contract anchor",
            )
            _require(
                envelope["publication_plan"]
                == plans[envelope["publication_key"]].to_dict(),
                "exact1k envelope publication plan drifted",
            )
            _require(
                len(envelope["requests"]) == envelope["expected_request_count"],
                "exact1k rank envelope request inventory is incomplete",
            )
    finally:
        for reservation in reservations.values():
            cancel_file_publication_reservation(reservation)


def test_rank_publication_envelope_seals_request_ids_global_indices_and_dp_seed_streams(tmp_path) -> None:
    manifest = _manifest_with_request_count(96)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=6_012,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
    )
    reservations, _plans, envelopes, prompt_anchors = reserve_nemo_rank_publications(
        tmp_path / "proof.json",
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=11,
        global_call_index_start=11,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=_reservation_caller_ledgers(
            manifest,
            profile,
            phases=("cold-generation",),
            generation_round=11,
            external_contract_sha256="a" * 64,
        ),
    )
    try:
        rank0, rank1 = sorted(envelopes.values(), key=lambda item: item["dp_rank"])
        _require(
            [request["request_id"] for request in rank0["requests"]]
            == [f"audit-{index:04d}" for index in range(48)],
            "rank 0 request identity partition drifted",
        )
        _require(
            [request["global_request_index"] for request in rank1["requests"]]
            == list(range(48, 96)),
            "rank 1 global request partition drifted",
        )
        _require(
            rank0["requests"][0]["seed"]
            == _independent_request_seed(
                base_seed=manifest.seed,
                global_call_index=11,
                dp_rank=0,
                dp_size=2,
                immutable_local_ordinal=0,
            ),
            "rank 0 first request seed differs from the independent formula",
        )
        _require(
            rank1["requests"][0]["seed"]
            == _independent_request_seed(
                base_seed=manifest.seed,
                global_call_index=11,
                dp_rank=1,
                dp_size=2,
                immutable_local_ordinal=0,
            ),
            "rank 1 first request seed differs from the independent formula",
        )
        _require(
            rank0["requests"][0]["prompt_token_ids_sha256"]
            == generation_prompt_token_ids_sha256(
                manifest.requests[0].prompt_token_ids
            ),
            "rank 0 prompt anchor differs from caller token bytes",
        )
        _require(
            rank1["requests"][0]["prompt_token_ids_sha256"]
            == generation_prompt_token_ids_sha256(
                manifest.requests[48].prompt_token_ids
            ),
            "rank 1 prompt anchor differs from caller token bytes",
        )
        _require(
            rank0["envelope_sha256"] != rank1["envelope_sha256"],
            "distinct DP rank envelopes collided",
        )
    finally:
        for reservation in reservations.values():
            cancel_file_publication_reservation(reservation)


def test_prompt_anchors_bind_distinct_prompts_to_identity_route_tail_and_retry(tmp_path) -> None:
    base = replace(_manifest_with_request_count(5), max_new_tokens=2)
    request_ids = ("req-z", "req-11", "req-a", "req-101", "tail-reactivated")
    manifest = replace(
        base,
        requests=tuple(
            WorkloadRequest(
                request_id=request_ids[index],
                prompt_token_ids=(1, 10 + index, 100 + index, 2),
            )
            for index in range(5)
        ),
    )
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    reservations, _plans, envelopes, caller_root = reserve_nemo_rank_publications(
        tmp_path / "proof.json",
        manifest=manifest,
        profile=profile,
        phases=("cold-generation", "retry-after-inactive"),
        generation_round=3,
        global_call_index_start=6,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=_reservation_caller_ledgers(
            manifest,
            profile,
            phases=("cold-generation", "retry-after-inactive"),
            generation_round=3,
            external_contract_sha256="a" * 64,
        ),
    )
    try:
        _require(
            "cold-generation/wave-001/dp-1"
            not in caller_root.index_by_publication_key(),
            "inactive cold DP rank received an active publication occurrence",
        )
        _require(
            "retry-after-inactive/wave-001/dp-1"
            not in caller_root.index_by_publication_key(),
            "inactive retry DP rank received an active publication occurrence",
        )
        tail = caller_root.require_occurrence("cold-generation/wave-001/dp-0")
        retry_tail = caller_root.require_occurrence("retry-after-inactive/wave-001/dp-0")
        _require(
            tuple(request.request_id for request in tail.requests)
            == ("tail-reactivated",),
            "tail occurrence lost its semantic request identity",
        )
        _require(
            tail.requests[0].request_index_in_dp_stream == 0,
            "tail request rank-local ordinal drifted",
        )
        _require(
            tail.requests[0].global_request_index == 4,
            "tail request global ordinal drifted",
        )
        _require(
            tail.requests[0].prompt_token_ids_sha256
            == generation_prompt_token_ids_sha256(
                manifest.requests[4].prompt_token_ids
            ),
            "tail request prompt anchor drifted",
        )
        _require(
            retry_tail.requests == tail.requests,
            "retry changed the semantic request inventory",
        )
        _require(
            retry_tail.phase == "retry-after-inactive",
            "retry occurrence lost its phase identity",
        )

        key = "cold-generation/wave-000/dp-0"
        reordered = deepcopy(envelopes[key])
        reordered["requests"] = list(reversed(reordered["requests"]))
        reordered["caller_prompt_anchor_sha256"] = _independent_rank_prompt_anchor_sha256(
            reordered
        )
        reordered["envelope_sha256"] = rank_publication_contract_sha256(reordered)
        with pytest.raises(RuntimeError, match="caller prompt anchor"):
            nemo_runner.validate_nemo_rank_prompt_anchor(reordered, caller_root)

        swapped = deepcopy(envelopes[key])
        first_digest = swapped["requests"][0]["prompt_token_ids_sha256"]
        swapped["requests"][0]["prompt_token_ids_sha256"] = swapped["requests"][1][
            "prompt_token_ids_sha256"
        ]
        swapped["requests"][1]["prompt_token_ids_sha256"] = first_digest
        swapped["caller_prompt_anchor_sha256"] = _independent_rank_prompt_anchor_sha256(
            swapped
        )
        swapped["envelope_sha256"] = rank_publication_contract_sha256(swapped)
        with pytest.raises(RuntimeError, match="caller prompt anchor"):
            nemo_runner.validate_nemo_rank_prompt_anchor(swapped, caller_root)
    finally:
        for reservation in reservations.values():
            cancel_file_publication_reservation(reservation)


def test_caller_prompt_root_separates_payload_semantic_and_execution_identity(tmp_path) -> None:
    base = replace(_manifest_with_request_count(5), max_new_tokens=2)
    shared_prompt = (1, 44, 144, 2)
    manifest = replace(
        base,
        requests=(
            WorkloadRequest(request_id="semantic-31", prompt_token_ids=shared_prompt),
            WorkloadRequest(request_id="semantic-7", prompt_token_ids=(1, 45, 145, 2)),
            WorkloadRequest(request_id="semantic-211", prompt_token_ids=shared_prompt),
            WorkloadRequest(request_id="semantic-19", prompt_token_ids=(1, 46, 146, 2)),
            WorkloadRequest(request_id="tail-503", prompt_token_ids=(1, 47, 147, 2)),
        ),
    )
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    reservations, _plans, envelopes, caller_root = reserve_nemo_rank_publications(
        tmp_path / "proof.json",
        manifest=manifest,
        profile=profile,
        phases=("cold-generation", "warm-generation"),
        generation_round=6,
        global_call_index_start=12,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=_reservation_caller_ledgers(
            manifest,
            profile,
            phases=("cold-generation", "warm-generation"),
            generation_round=6,
            external_contract_sha256="a" * 64,
        ),
    )
    try:
        _require(
            type(caller_root) is NemoCallerPromptAnchorRoot,
            "caller root is not the exact immutable record type",
        )
        _require(
            len(caller_root.prompt_payload_catalog) == 4,
            "caller root did not deduplicate prompt payloads exactly",
        )
        _require(
            len(caller_root.semantic_requests) == 5,
            "caller root semantic request inventory drifted",
        )
        _require(
            len(caller_root.execution_occurrences) == 8,
            "caller root occurrence inventory drifted",
        )
        cold = caller_root.require_occurrence("cold-generation/wave-000/dp-0")
        warm = caller_root.require_occurrence("warm-generation/wave-000/dp-0")
        _require(
            cold.semantic_request_sha256 == warm.semantic_request_sha256,
            "replay occurrence changed semantic request identity",
        )
        _require(
            cold.sha256 != warm.sha256,
            "distinct execution occurrences collided",
        )
        inactive = caller_root.require_occurrence("cold-generation/wave-001/dp-1")
        _require(inactive.active is False, "inactive occurrence is marked active")
        _require(inactive.requests == (), "inactive occurrence contains requests")
        _require(
            inactive.publication_key not in caller_root.index_by_publication_key(),
            "inactive occurrence entered the active publication index",
        )
        _require(
            envelopes[cold.publication_key]["caller_prompt_anchor_sha256"]
            == caller_root.sha256,
            "rank envelope lost the external caller root",
        )
        _require(
            envelopes[cold.publication_key]["execution_occurrence_sha256"]
            == cold.sha256,
            "rank envelope occurrence digest drifted",
        )
    finally:
        for reservation in reservations.values():
            cancel_file_publication_reservation(reservation)


def test_caller_prompt_root_rejects_digest_collision_before_reservation(
    monkeypatch,
    tmp_path,
) -> None:
    manifest = replace(
        _manifest_with_request_count(2),
        max_new_tokens=2,
        requests=(
            WorkloadRequest(request_id="collision-a", prompt_token_ids=(1, 11, 2)),
            WorkloadRequest(request_id="collision-b", prompt_token_ids=(1, 99, 2)),
        ),
    )
    profile = Evo2VllmProfile(
        topology="tp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=False,
        proof=True,
        global_wave_size=2,
        max_num_seqs=2,
    )
    monkeypatch.setattr(nemo_runner, "generation_prompt_token_ids_sha256", lambda _tokens: "d" * 64)

    with pytest.raises(RuntimeError, match="prompt digest collision"):
        reserve_nemo_rank_publications(
            tmp_path / "proof.json",
            manifest=manifest,
            profile=profile,
            phases=("cold-generation",),
            generation_round=0,
            global_call_index_start=0,
            external_contract_sha256="a" * 64,
            caller_ledgers_by_phase=_reservation_caller_ledgers(
                manifest,
                profile,
                phases=("cold-generation",),
                generation_round=0,
                external_contract_sha256="a" * 64,
            ),
        )

    assert not list(tmp_path.iterdir())


def test_nemo_request_envelope_owns_nonuniform_inactive_and_reordered_routes() -> None:
    base = _manifest_with_request_count(3).with_max_new_tokens(1)
    semantic_ids = ("semantic:900", "alpha", "request-above-9")
    manifest = replace(
        base,
        requests=tuple(
            replace(request, request_id=semantic_ids[index])
            for index, request in enumerate(base.requests)
        ),
    )
    wave = build_request_waves(
        request_count=3,
        global_batch_size=3,
        replica_count=2,
    )[0]
    schedule = GenerationRequestScheduleAdmission(
        semantic_namespace_root="test",
        base_seed=manifest.seed,
        data_parallel_size=2,
        batch_call_slots_per_round=1,
        turns_per_call=1,
    )
    ledger = GenerationRequestLedgerAdmission(
        schedule_admission=schedule,
        generation_round_start=7,
        generation_round_count=1,
        global_request_index_start=100,
        request_counts_by_batch_call_slot=(3,),
    )
    envelope = nemo_runner.build_nemo_request_envelope(
        wave_manifest=manifest,
        wave=wave,
        caller_ledger_admission=ledger,
        generation_round=7,
        batch_call_slot=0,
        execution_attempt_epoch=0,
    )

    assert envelope.partition_indices == ((0, 1), (2,))
    assert tuple(request.local_request_id for request in envelope.requests) == semantic_ids
    assert tuple(
        (request.target_data_parallel_rank, request.immutable_local_ordinal)
        for request in envelope.requests
    ) == ((0, 0), (0, 1), (1, 0))
    assert envelope.request_seeds == tuple(
        _independent_request_seed(
            base_seed=manifest.seed,
            global_call_index=7,
            dp_rank=dp_rank,
            dp_size=2,
            immutable_local_ordinal=ordinal,
        )
        for dp_rank, ordinal in ((0, 0), (0, 1), (1, 0))
    )

    executions = tuple(
        runner.RequestExecutionRecord(
            request_id=request.local_request_id,
            global_request_index=request.global_request_index,
            generation_round=envelope.generation_round,
            dp_rank=request.target_data_parallel_rank,
            call_index=envelope.flattened_call_index,
            seed=_independent_request_seed(
                base_seed=envelope.base_seed,
                global_call_index=envelope.flattened_call_index,
                dp_rank=request.target_data_parallel_rank,
                dp_size=envelope.data_parallel_size,
                immutable_local_ordinal=request.immutable_local_ordinal,
            ),
        )
        for request in envelope.requests
    )
    nemo_runner._validate_wave_execution(
        manifest,
        executions,
        expected_envelope=envelope,
    )
    with pytest.raises(AssertionError, match="request order"):
        nemo_runner._validate_wave_execution(
            manifest,
            (executions[2], executions[0], executions[1]),
            expected_envelope=envelope,
        )
    with pytest.raises(AssertionError, match="DP ownership"):
        nemo_runner._validate_wave_execution(
            manifest,
            (replace(executions[0], dp_rank=1), *executions[1:]),
            expected_envelope=envelope,
        )

    single_manifest = manifest.request_slice(0, 1)
    single_wave = build_request_waves(
        request_count=1,
        global_batch_size=2,
        replica_count=2,
    )[0]
    single_ledger = GenerationRequestLedgerAdmission(
        schedule_admission=schedule,
        generation_round_start=8,
        generation_round_count=1,
        global_request_index_start=103,
        request_counts_by_batch_call_slot=(1,),
    )
    single_envelope = nemo_runner.build_nemo_request_envelope(
        wave_manifest=single_manifest,
        wave=single_wave,
        caller_ledger_admission=single_ledger,
        generation_round=8,
        batch_call_slot=0,
        execution_attempt_epoch=0,
    )
    assert single_envelope.partition_indices == ((0,), ())


def _publish_synthetic_rank_outcome(
    envelope,
    manifest,
    *,
    row_schema_version=NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION,
):
    prompts = {request.request_id: list(request.prompt_token_ids) for request in manifest.requests}
    rows = []
    for request in envelope["requests"]:
        prompt = prompts[request["request_id"]]
        generated = [65, 67]
        rows.append(
            {
                "schema_version": row_schema_version,
                "request_id": request["request_id"],
                "engine_request_id": f"engine:{envelope['publication_key']}:{request['request_id']}",
                "global_request_index": request["global_request_index"],
                "request_index_in_dp_stream": request["request_index_in_dp_stream"],
                "semantic_request_sha256": request["semantic_request_sha256"],
                "seed": request["seed"],
                "generation_round": envelope["generation_round"],
                "call_index": envelope["call_index"],
                "dp_rank": envelope["dp_rank"],
                "phase": envelope["phase"],
                "wave_index": envelope["wave_index"],
                "prompt_token_ids": prompt,
                "output_token_ids": generated,
                "chosen_token_logprobs": [-0.125, -0.25],
                "selected_logprob_valid_count": 2,
                "requested_max_tokens": 2,
                "observed_prompt_tokens": len(prompt),
                "observed_new_tokens": 2,
                "observed_total_tokens": len(prompt) + 2,
                "finish_reason": "length",
                "stopped_on_eos": False,
                "external_contract_sha256": envelope["external_contract_sha256"],
                "envelope_sha256": envelope["envelope_sha256"],
                "caller_prompt_anchor_sha256": envelope["caller_prompt_anchor_sha256"],
                "execution_occurrence_sha256": envelope["execution_occurrence_sha256"],
            }
        )
    jsonl = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows
    )
    payload = gzip.compress(jsonl, mtime=0)
    receipt = publish_reserved_bytes(envelope["publication_plan"], payload)
    semantic_coordinates = [
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
    ]
    return {
        "schema_version": NEMO_RANK_PUBLICATION_OUTCOME_SCHEMA_VERSION,
        "publication_key": envelope["publication_key"],
        "external_contract_sha256": envelope["external_contract_sha256"],
        "envelope_sha256": envelope["envelope_sha256"],
        "caller_prompt_anchor_sha256": envelope["caller_prompt_anchor_sha256"],
        "execution_occurrence_sha256": envelope["execution_occurrence_sha256"],
        "phase": envelope["phase"],
        "wave_index": envelope["wave_index"],
        "generation_round": envelope["generation_round"],
        "call_index": envelope["call_index"],
        "dp_rank": envelope["dp_rank"],
        "publisher": True,
        "payload_sha256": receipt.sha256,
        "payload_size_bytes": receipt.size_bytes,
        "row_count": len(rows),
        "request_ids": [row["request_id"] for row in rows],
        "semantic_request_coordinates": semantic_coordinates,
        "tp_sibling_evidence": [
            {
                "tp_rank": 0,
                "publisher": True,
                "payload_sha256": receipt.sha256,
                "payload_size_bytes": receipt.size_bytes,
                "envelope_sha256": envelope["envelope_sha256"],
            }
        ],
        "provisional_receipt": receipt.to_dict(),
    }


def test_coordinator_reopens_finalizes_and_reassembles_every_dp_rank_sidecar(tmp_path) -> None:
    output = tmp_path / "proof.json"
    marker = runner.reserve_output_namespace(output)
    manifest = replace(_manifest_with_request_count(4), max_new_tokens=2)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    caller_ledgers = _reservation_caller_ledgers(
        manifest,
        profile,
        phases=("cold-generation",),
        generation_round=0,
        external_contract_sha256="a" * 64,
    )
    reservations, _plans, envelopes, prompt_anchors = reserve_nemo_rank_publications(
        output,
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=0,
        global_call_index_start=0,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=caller_ledgers,
    )
    outcomes = {
        key: _publish_synthetic_rank_outcome(envelope, manifest) for key, envelope in envelopes.items()
    }

    class FakeWorkerGroup:
        dp_leader_worker_indices = [5, 9]

        def __init__(self):
            self.calls = []

        def run_single_worker_single_data(
            self,
            method_name,
            worker_index,
            *,
            envelope_payload,
            expected_envelope_sha256,
        ):
            envelope = json.loads(envelope_payload)
            _require(
                envelope["envelope_sha256"] == expected_envelope_sha256,
                "worker RPC received an unsealed rank envelope",
            )
            self.calls.append((method_name, worker_index, envelope["publication_key"]))
            return outcomes[envelope["publication_key"]]

    worker_group = FakeWorkerGroup()
    generation = type("Generation", (), {"worker_group": worker_group})()
    wave = build_request_waves(request_count=4, global_batch_size=4, replica_count=2)[0]
    rpc_outcomes = publish_nemo_rank_wave(
        generation,
        wave=wave,
        phase="cold-generation",
        envelopes=envelopes,
        caller_prompt_anchor_roots=prompt_anchors.index_by_publication_key(),
        caller_ledger_admission=caller_ledgers["cold-generation"],
        ray_get=lambda futures: futures,
    )
    outcomes = {outcome["publication_key"]: outcome for outcome in rpc_outcomes}

    proof = finalize_nemo_rank_publications(
        reservations=reservations,
        envelopes=envelopes,
        caller_prompt_anchor_roots=prompt_anchors.index_by_publication_key(),
        caller_ledgers_by_phase=caller_ledgers,
        outcomes=outcomes,
        namespace_output_path=output,
    )
    runner.write_json_artifact(
        output,
        {"rank_publication_proof": proof},
        ownership_validator=lambda: runner.require_output_namespace_reservation(output),
    )
    runner.complete_output_namespace(marker, output_path=output)

    _require(
        proof["provenance_assurance_tier"] == "adapter_integrity",
        "coordinator overstated provenance assurance",
    )
    _require(
        proof["trusted_node_reproducibility_admission"] is False,
        "adapter-only proof falsely claimed trusted-node admission",
    )
    _require(
        proof["full_runtime_adversarial_admission"] is False,
        "adapter-only proof falsely claimed adversarial admission",
    )
    _require(
        proof["live_standard_engine_enqueue_wait_admission"] is False,
        "CPU adapter proof falsely claimed live enqueue/wait admission",
    )
    _require(
        proof["nemo_rl_gdpo_e2e_admission"] is False,
        "CPU adapter proof falsely claimed NeMo-RL GDPO end-to-end admission",
    )
    _require(proof["publication_count"] == 2, "coordinator missed a DP sidecar")
    _require(
        list(proof["reassembled_wave_sha256"])
        == ["cold-generation/wave-000"],
        "coordinator reassembled the wrong wave inventory",
    )
    _require(
        all(
            not Path(envelope["publication_plan"]["marker_path"]).exists()
            for envelope in envelopes.values()
        ),
        "coordinator left a live reservation marker",
    )
    _require(output.exists(), "coordinator did not publish the terminal artifact")
    _require(
        [(method, worker_index) for method, worker_index, _key in worker_group.calls]
        == [
            ("publish_evo2_generation_sidecar", 5),
            ("publish_evo2_generation_sidecar", 9),
        ],
        "coordinator did not dispatch exactly one publication per DP rank",
    )


def test_coordinator_rejects_stale_rank_sidecar_schema_during_reopen(tmp_path) -> None:
    output = tmp_path / "stale-row-schema.json"
    runner.reserve_output_namespace(output)
    manifest = replace(_manifest_with_request_count(4), max_new_tokens=2)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    caller_ledgers = _reservation_caller_ledgers(
        manifest,
        profile,
        phases=("cold-generation",),
        generation_round=0,
        external_contract_sha256="a" * 64,
    )
    reservations, _plans, envelopes, prompt_anchors = reserve_nemo_rank_publications(
        output,
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=0,
        global_call_index_start=0,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=caller_ledgers,
    )
    outcomes = {}
    for index, (key, envelope) in enumerate(envelopes.items()):
        outcomes[key] = _publish_synthetic_rank_outcome(
            envelope,
            manifest,
            row_schema_version=(
                "evo2-nemo-rank-sidecar-row/v0"
                if index == 0
                else NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION
            ),
        )
    try:
        with pytest.raises(RuntimeError, match="sidecar row schema"):
            finalize_nemo_rank_publications(
                reservations=reservations,
                envelopes=envelopes,
                caller_prompt_anchor_roots=prompt_anchors.index_by_publication_key(),
                caller_ledgers_by_phase=caller_ledgers,
                outcomes=outcomes,
                namespace_output_path=output,
            )
        _require(
            all(not reservation.closed for reservation in reservations.values()),
            "stale sidecar schema falsely finalized a rank reservation",
        )
    finally:
        terminalize_failed_nemo_rank_publications(reservations)


def test_coordinator_rejects_coherently_shifted_worker_coordinate_bundle(tmp_path) -> None:
    output = tmp_path / "proof.json"
    runner.reserve_output_namespace(output)
    manifest = replace(_manifest_with_request_count(4), max_new_tokens=2)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    caller_ledgers = _reservation_caller_ledgers(
        manifest,
        profile,
        phases=("cold-generation",),
        generation_round=0,
        external_contract_sha256="a" * 64,
    )
    reservations, _plans, envelopes, prompt_anchors = reserve_nemo_rank_publications(
        output,
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=0,
        global_call_index_start=0,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=caller_ledgers,
    )
    outcomes = {
        key: _publish_synthetic_rank_outcome(envelope, manifest) for key, envelope in envelopes.items()
    }
    shifted = next(iter(outcomes.values()))
    shifted["generation_round"] = 1
    shifted["semantic_request_coordinates"] = [
        {**row, "generation_round": 1} for row in shifted["semantic_request_coordinates"]
    ]

    with pytest.raises(RuntimeError, match="caller field generation_round"):
        finalize_nemo_rank_publications(
            reservations=reservations,
            envelopes=envelopes,
            caller_prompt_anchor_roots=prompt_anchors.index_by_publication_key(),
            caller_ledgers_by_phase=caller_ledgers,
            outcomes=outcomes,
            namespace_output_path=output,
        )

    terminalize_failed_nemo_rank_publications(reservations)
    _require(
        all(reservation.closed for reservation in reservations.values()),
        "shifted worker bundle left an unterminated reservation",
    )
    _require(
        all(
            not os.path.lexists(path)
            for envelope in envelopes.values()
            for path in (
                envelope["publication_plan"]["final_path"],
                envelope["publication_plan"]["marker_path"],
                envelope["publication_plan"]["staging_directory_path"],
            )
        ),
        "shifted worker bundle left a live publication path",
    )


def test_finalizer_reconstructs_global_generation_envelope_from_rank_records(
    tmp_path,
) -> None:
    output = tmp_path / "proof.json"
    runner.reserve_output_namespace(output)
    manifest = replace(_manifest_with_request_count(4), max_new_tokens=2)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    caller_ledgers = _reservation_caller_ledgers(
        manifest,
        profile,
        phases=("cold-generation",),
        generation_round=0,
        external_contract_sha256="a" * 64,
    )
    reservations, _plans, envelopes, prompt_anchors = reserve_nemo_rank_publications(
        output,
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=0,
        global_call_index_start=0,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=caller_ledgers,
    )
    outcomes = {
        key: _publish_synthetic_rank_outcome(envelope, manifest)
        for key, envelope in envelopes.items()
    }
    rewritten = deepcopy(envelopes)
    for envelope in rewritten.values():
        envelope["generation_request_envelope_sha256"] = "f" * 64
        envelope["generation_execution_occurrence_sha256"] = "e" * 64
    _require(
        all(
            rewritten[key]["generation_request_envelope_sha256"]
            != envelopes[key]["generation_request_envelope_sha256"]
            and rewritten[key]["generation_execution_occurrence_sha256"]
            != envelopes[key]["generation_execution_occurrence_sha256"]
            for key in envelopes
        ),
        "finalizer rewrite fixture did not alter both sealed generation digests",
    )

    try:
        with pytest.raises(RuntimeError, match="generation envelope"):
            finalize_nemo_rank_publications(
                reservations=reservations,
                envelopes=rewritten,
                caller_prompt_anchor_roots=prompt_anchors.index_by_publication_key(),
                caller_ledgers_by_phase=caller_ledgers,
                outcomes=outcomes,
                namespace_output_path=output,
            )
    finally:
        terminalize_failed_nemo_rank_publications(reservations)


def test_caller_prompt_anchor_rejects_coherent_foreign_prompt_before_rpc_and_terminalization(
    tmp_path,
) -> None:
    output = tmp_path / "proof.json"
    base = replace(_manifest_with_request_count(5), max_new_tokens=2)
    manifest = replace(
        base,
        requests=tuple(
            WorkloadRequest(
                request_id=f"semantic-{index * 17 + 3}",
                prompt_token_ids=(1, 20 + index, 120 + index, 2),
            )
            for index in range(5)
        ),
    )
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    caller_ledgers = _reservation_caller_ledgers(
        manifest,
        profile,
        phases=("cold-generation",),
        generation_round=4,
        external_contract_sha256="a" * 64,
    )
    reservations, _plans, envelopes, anchors = reserve_nemo_rank_publications(
        output,
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=4,
        global_call_index_start=8,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=caller_ledgers,
    )
    key = "cold-generation/wave-000/dp-0"
    foreign_manifest = replace(
        manifest,
        requests=(
            replace(manifest.requests[0], prompt_token_ids=(1, 77, 177, 2)),
            *manifest.requests[1:],
        ),
    )
    rewritten = envelopes[key]
    rewritten["requests"][0]["prompt_token_ids_sha256"] = generation_prompt_token_ids_sha256(
        foreign_manifest.requests[0].prompt_token_ids
    )
    rewritten["caller_prompt_anchor_sha256"] = _independent_rank_prompt_anchor_sha256(
        rewritten
    )
    rewritten["envelope_sha256"] = rank_publication_contract_sha256(rewritten)

    class NoCallWorkerGroup:
        dp_leader_worker_indices = [0, 1]

        def __init__(self) -> None:
            self.call_count = 0

        def run_single_worker_single_data(self, *args, **kwargs):
            self.call_count += 1
            raise AssertionError("foreign prompt must fail before worker publication")

    worker_group = NoCallWorkerGroup()
    generation = type("Generation", (), {"worker_group": worker_group})()
    wave = build_request_waves(request_count=5, global_batch_size=4, replica_count=2)[0]
    try:
        with pytest.raises(RuntimeError, match="caller prompt anchor"):
            publish_nemo_rank_wave(
                generation,
                wave=wave,
                phase="cold-generation",
                envelopes=envelopes,
                caller_prompt_anchor_roots=anchors.index_by_publication_key(),
                caller_ledger_admission=caller_ledgers["cold-generation"],
                ray_get=lambda futures: futures,
            )
        _require(
            worker_group.call_count == 0,
            "foreign caller prompt reached worker publication",
        )
        _require(
            all(
                not Path(reservation.plan.final_path).exists()
                for reservation in reservations.values()
            ),
            "foreign caller prompt published a final artifact",
        )

        self_attested_outcomes = {publication_key: {} for publication_key in envelopes}
        with pytest.raises(RuntimeError, match="caller prompt anchor"):
            finalize_nemo_rank_publications(
                reservations=reservations,
                envelopes=envelopes,
                caller_prompt_anchor_roots=anchors.index_by_publication_key(),
                caller_ledgers_by_phase=caller_ledgers,
                outcomes=self_attested_outcomes,
                namespace_output_path=output,
            )
        _require(
            all(not reservation.closed for reservation in reservations.values()),
            "failed foreign-root finalization falsely closed a reservation",
        )
        _require(
            all(
                not Path(reservation.plan.final_path).exists()
                for reservation in reservations.values()
            ),
            "failed foreign-root finalization published a final artifact",
        )
    finally:
        terminalize_failed_nemo_rank_publications(reservations)


def test_finalizer_rejects_coherent_foreign_prompt_sidecar_against_original_anchor(
    tmp_path,
) -> None:
    output = tmp_path / "proof.json"
    manifest = replace(
        _manifest_with_request_count(4),
        max_new_tokens=2,
        requests=tuple(
            WorkloadRequest(
                request_id=f"identity-{index * 13 + 5}",
                prompt_token_ids=(1, 30 + index, 130 + index, 2),
            )
            for index in range(4)
        ),
    )
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    caller_ledgers = _reservation_caller_ledgers(
        manifest,
        profile,
        phases=("cold-generation",),
        generation_round=0,
        external_contract_sha256="a" * 64,
    )
    reservations, _plans, envelopes, anchors = reserve_nemo_rank_publications(
        output,
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=0,
        global_call_index_start=0,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=caller_ledgers,
    )
    key = "cold-generation/wave-000/dp-0"
    foreign_manifest = replace(
        manifest,
        requests=(
            replace(manifest.requests[0], prompt_token_ids=(1, 88, 188, 2)),
            *manifest.requests[1:],
        ),
    )
    rewritten = envelopes[key]
    rewritten["requests"][0]["prompt_token_ids_sha256"] = generation_prompt_token_ids_sha256(
        foreign_manifest.requests[0].prompt_token_ids
    )
    rewritten["caller_prompt_anchor_sha256"] = _independent_rank_prompt_anchor_sha256(
        rewritten
    )
    rewritten["envelope_sha256"] = rank_publication_contract_sha256(rewritten)
    outcomes = {
        publication_key: _publish_synthetic_rank_outcome(envelope, foreign_manifest)
        for publication_key, envelope in envelopes.items()
    }
    try:
        with pytest.raises(RuntimeError, match="caller prompt anchor"):
            finalize_nemo_rank_publications(
                reservations=reservations,
                envelopes=envelopes,
                caller_prompt_anchor_roots=anchors.index_by_publication_key(),
                caller_ledgers_by_phase=caller_ledgers,
                outcomes=outcomes,
                namespace_output_path=output,
            )
        _require(
            all(not reservation.closed for reservation in reservations.values()),
            "foreign prompt sidecar falsely closed a reservation",
        )
    finally:
        terminalize_failed_nemo_rank_publications(reservations)


def test_marker_preflight_rejects_nonvacuous_coherent_foreign_root_before_rpc(
    tmp_path,
) -> None:
    output = tmp_path / "coherent-foreign-marker.json"
    manifest = replace(
        _manifest_with_request_count(4),
        max_new_tokens=2,
        requests=tuple(
            WorkloadRequest(
                request_id=f"semantic-{index * 19 + 7}",
                prompt_token_ids=(1, 40 + index, 140 + index, 2),
            )
            for index in range(4)
        ),
    )
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    caller_ledgers = _reservation_caller_ledgers(
        manifest,
        profile,
        phases=("cold-generation",),
        generation_round=0,
        external_contract_sha256="a" * 64,
    )
    reservations, _plans, envelopes, caller_root = reserve_nemo_rank_publications(
        output,
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=0,
        global_call_index_start=0,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=caller_ledgers,
    )
    key = "cold-generation/wave-000/dp-0"
    original_occurrence = caller_root.require_occurrence(key)
    original_envelope_sha256 = envelopes[key]["envelope_sha256"]
    original_marker_sha256 = envelopes[key]["publication_plan"]["marker_sha256"]
    foreign_prompt = (1, 99, 199, 2)
    foreign_manifest, foreign_root, foreign_envelopes = _coherent_foreign_prompt_tree(
        manifest=manifest,
        caller_root=caller_root,
        envelopes=envelopes,
        request_index=0,
        foreign_prompt_token_ids=foreign_prompt,
    )
    foreign_occurrence = foreign_root.require_occurrence(key)
    _require(
        manifest.requests[0].prompt_token_ids != foreign_prompt,
        "foreign-root prompt fixture is not distinct",
    )
    _require(
        len(manifest.requests[0].prompt_token_ids) == len(foreign_prompt),
        "foreign-root prompt fixture changed prompt length",
    )
    _require(
        manifest.sha256 != foreign_manifest.sha256,
        "foreign-root fixture did not change the manifest digest",
    )
    _require(
        caller_root.external_contract_sha256
        != foreign_root.external_contract_sha256,
        "foreign-root fixture did not change the external contract",
    )
    _require(
        caller_root.sha256 != foreign_root.sha256,
        "foreign-root fixture did not change the caller root",
    )
    _require(
        original_occurrence.requests[0].prompt_token_ids_sha256
        != foreign_occurrence.requests[0].prompt_token_ids_sha256,
        "foreign-root fixture did not change the prompt payload digest",
    )
    _require(
        original_occurrence.requests[0].sha256
        != foreign_occurrence.requests[0].sha256,
        "foreign-root fixture did not change the semantic request digest",
    )
    _require(
        original_occurrence.sha256 != foreign_occurrence.sha256,
        "foreign-root fixture did not change the occurrence digest",
    )
    _require(
        original_envelope_sha256 != foreign_envelopes[key]["envelope_sha256"],
        "foreign-root fixture did not change the rank envelope digest",
    )
    _require(
        original_marker_sha256
        != foreign_envelopes[key]["publication_plan"]["marker_sha256"],
        "foreign-root fixture did not change the reservation marker digest",
    )

    class NoCallWorkerGroup:
        dp_leader_worker_indices = [0, 1]

        def __init__(self) -> None:
            self.calls = []

        def run_single_worker_single_data(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            raise AssertionError("marker/root preflight must precede worker RPC")

    worker_group = NoCallWorkerGroup()
    generation = type("Generation", (), {"worker_group": worker_group})()
    wave = build_request_waves(request_count=4, global_batch_size=4, replica_count=2)[0]
    try:
        with pytest.raises(RuntimeError, match="reservation marker"):
            publish_nemo_rank_wave(
                generation,
                wave=wave,
                phase="cold-generation",
                envelopes=foreign_envelopes,
                caller_prompt_anchor_roots=foreign_root.index_by_publication_key(),
                caller_ledger_admission=caller_ledgers["cold-generation"],
                ray_get=lambda futures: futures,
            )
        _require(worker_group.calls == [], "foreign marker/root reached worker RPC")
        _require(
            all(
                not Path(reservation.plan.final_path).exists()
                for reservation in reservations.values()
            ),
            "foreign marker/root published a final artifact",
        )
    finally:
        terminalize_failed_nemo_rank_publications(reservations)


def test_rank_publication_two_pass_preflight_rejects_malformed_later_rank_before_first_rpc(
    tmp_path,
) -> None:
    output = tmp_path / "malformed-later-rank.json"
    manifest = replace(_manifest_with_request_count(4), max_new_tokens=2)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    caller_ledgers = _reservation_caller_ledgers(
        manifest,
        profile,
        phases=("cold-generation",),
        generation_round=0,
        external_contract_sha256="a" * 64,
    )
    reservations, _plans, envelopes, caller_root = reserve_nemo_rank_publications(
        output,
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=0,
        global_call_index_start=0,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=caller_ledgers,
    )
    malformed = deepcopy(envelopes)
    later_key = "cold-generation/wave-000/dp-1"
    original_digest = malformed[later_key]["requests"][0]["prompt_token_ids_sha256"]
    malformed[later_key]["requests"][0]["prompt_token_ids_sha256"] = "f" * 64
    _require(
        malformed[later_key]["requests"][0]["prompt_token_ids_sha256"]
        != original_digest,
        "later-rank hostile fixture did not alter the prompt digest",
    )

    class NoCallWorkerGroup:
        dp_leader_worker_indices = [0, 1]

        def __init__(self) -> None:
            self.calls = []

        def run_single_worker_single_data(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            raise AssertionError("all ranks must pass preflight before the first RPC")

    worker_group = NoCallWorkerGroup()
    generation = type("Generation", (), {"worker_group": worker_group})()
    wave = build_request_waves(request_count=4, global_batch_size=4, replica_count=2)[0]
    try:
        with pytest.raises(RuntimeError, match="caller prompt anchor"):
            publish_nemo_rank_wave(
                generation,
                wave=wave,
                phase="cold-generation",
                envelopes=malformed,
                caller_prompt_anchor_roots=caller_root.index_by_publication_key(),
                caller_ledger_admission=caller_ledgers["cold-generation"],
                ray_get=lambda futures: futures,
            )
        _require(
            worker_group.calls == [],
            "malformed later rank was detected only after an earlier worker RPC",
        )
        _require(
            all(
                not Path(reservation.plan.final_path).exists()
                for reservation in reservations.values()
            ),
            "malformed later rank published a final artifact",
        )
    finally:
        terminalize_failed_nemo_rank_publications(reservations)


def test_generation_phase_rejects_foreign_root_before_provider_rng_or_publication_side_effects(
    tmp_path,
) -> None:
    output = tmp_path / "foreign-phase.outputs.jsonl.gz"
    manifest = replace(_manifest_with_request_count(4), max_new_tokens=2)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    caller_ledgers = _reservation_caller_ledgers(
        manifest,
        profile,
        phases=("cold-generation",),
        generation_round=0,
        external_contract_sha256="a" * 64,
    )
    reservations, _plans, envelopes, caller_root = reserve_nemo_rank_publications(
        tmp_path / "phase-proof.json",
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=0,
        global_call_index_start=0,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=caller_ledgers,
    )
    foreign_manifest, foreign_root, foreign_envelopes = _coherent_foreign_prompt_tree(
        manifest=manifest,
        caller_root=caller_root,
        envelopes=envelopes,
        request_index=0,
        foreign_prompt_token_ids=(1, 99, 199, 2),
    )
    _require(
        foreign_manifest.sha256 != manifest.sha256,
        "foreign phase fixture did not alter the manifest root",
    )
    side_effects = []

    class ForbiddenGeneration:
        def __getattribute__(self, name):
            if name.startswith("__"):
                return object.__getattribute__(self, name)
            side_effects.append(name)
            raise AssertionError(f"provider side effect before caller-root admission: {name}")

    python_rng_before = random.getstate()
    torch_rng_before = torch.random.get_rng_state().clone()
    try:
        with pytest.raises(RuntimeError, match="caller prompt root differs"):
            run_nemo_generation_phase(
                generation=ForbiddenGeneration(),
                manifest=manifest,
                profile=profile,
                phase="cold-generation",
                sample_index=0,
                generation_round=0,
                global_call_index_start=0,
                global_request_index_start=0,
                caller_ledger_admission=caller_ledgers["cold-generation"],
                full_output_path=output,
                memory_monitor_factory=lambda: pytest.fail(
                    "memory/proof setup ran before caller-root admission"
                ),
                ray_get=lambda _value: pytest.fail(
                    "Ray/provider work ran before caller-root admission"
                ),
                rank_publication_envelopes=foreign_envelopes,
                rank_prompt_anchor_roots=foreign_root.index_by_publication_key(),
                request_envelope_namespace="a" * 64,
            )
        _require(side_effects == [], "foreign phase touched the generation provider")
        _require(
            random.getstate() == python_rng_before,
            "foreign phase advanced Python RNG state",
        )
        _require(
            bool(torch.equal(torch.random.get_rng_state(), torch_rng_before)),
            "foreign phase advanced Torch RNG state",
        )
        _require(not output.exists(), "foreign phase published its output sidecar")
        _require(
            all(
                not Path(reservation.plan.final_path).exists()
                for reservation in reservations.values()
            ),
            "foreign phase published a rank final artifact",
        )
    finally:
        terminalize_failed_nemo_rank_publications(reservations)


def test_generation_phase_rejects_coherent_foreign_ledger_before_provider_side_effects(
    tmp_path,
) -> None:
    manifest = replace(_manifest_with_request_count(4), max_new_tokens=2)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    admitted_schedule = GenerationRequestScheduleAdmission(
        semantic_namespace_root="proof/cold/cold-generation",
        base_seed=manifest.seed,
        data_parallel_size=2,
        batch_call_slots_per_round=1,
        turns_per_call=1,
    )
    admitted_ledger = GenerationRequestLedgerAdmission(
        schedule_admission=admitted_schedule,
        generation_round_start=0,
        generation_round_count=1,
        global_request_index_start=0,
        request_counts_by_batch_call_slot=(4,),
    )
    foreign_ledger = replace(
        admitted_ledger,
        schedule_admission=replace(admitted_schedule, turns_per_call=2),
    )
    _require(
        foreign_ledger.sha256 != admitted_ledger.sha256,
        "foreign caller-ledger fixture did not change the ledger digest",
    )
    side_effects = []

    class ForbiddenGeneration:
        def __getattribute__(self, name):
            if name.startswith("__"):
                return object.__getattribute__(self, name)
            side_effects.append(name)
            raise AssertionError(f"provider side effect before caller-ledger admission: {name}")

    with pytest.raises(ValueError, match="caller ledger|turns per call"):
        run_nemo_generation_phase(
            generation=ForbiddenGeneration(),
            manifest=manifest,
            profile=profile,
            phase="cold-generation",
            sample_index=0,
            generation_round=0,
            global_call_index_start=0,
            global_request_index_start=0,
            caller_ledger_admission=foreign_ledger,
            full_output_path=tmp_path / "foreign-ledger.outputs.jsonl.gz",
            memory_monitor_factory=lambda: pytest.fail(
                "memory/proof setup ran before caller-ledger admission"
            ),
            ray_get=lambda _value: pytest.fail(
                "Ray/provider work ran before caller-ledger admission"
            ),
            request_envelope_namespace="proof/cold",
        )
    _require(side_effects == [], "foreign caller ledger touched the generation provider")


@pytest.mark.parametrize("published_dp_count", (0, 1))
def test_failed_attempt_terminalizes_every_rank_plan_before_or_after_partial_dp_publish(
    tmp_path,
    published_dp_count,
) -> None:
    output = tmp_path / "failed-proof.json"
    manifest = replace(_manifest_with_request_count(4), max_new_tokens=2)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
        global_wave_size=4,
        max_num_seqs=2,
    )
    reservations, _plans, envelopes, _prompt_anchors = reserve_nemo_rank_publications(
        output,
        manifest=manifest,
        profile=profile,
        phases=("cold-generation",),
        generation_round=0,
        global_call_index_start=0,
        external_contract_sha256="a" * 64,
        caller_ledgers_by_phase=_reservation_caller_ledgers(
            manifest,
            profile,
            phases=("cold-generation",),
            generation_round=0,
            external_contract_sha256="a" * 64,
        ),
    )
    for envelope in list(envelopes.values())[:published_dp_count]:
        _publish_synthetic_rank_outcome(envelope, manifest)
    primary = RuntimeError("ray failed")

    terminalize_failed_nemo_rank_publications(reservations, primary_error=primary)

    _require(
        all(reservation.closed for reservation in reservations.values()),
        "failed attempt left a live rank publication reservation",
    )
    _require(
        all(
            not os.path.lexists(path)
            for envelope in envelopes.values()
            for path in (
                envelope["publication_plan"]["final_path"],
                envelope["publication_plan"]["marker_path"],
                envelope["publication_plan"]["staging_directory_path"],
            )
        ),
        "failed attempt left a rank publication path behind",
    )
    _require(not output.exists(), "failed attempt published a terminal PASS artifact")


def test_nemo_dp2_gpu_preflight_failure_precedes_ray_actor_setup(tmp_path, monkeypatch) -> None:
    output = tmp_path / "dp2-gpu-preflight.json"
    args = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--manifest",
            str(DATA),
            "--topology",
            "dp2",
            "--max-model-len",
            "7000",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.92",
            "--proof",
            "--output",
            str(output),
        ]
    )
    manifest = WorkloadManifest.from_path(DATA)
    runner.reserve_output_namespace(output)
    failure = runner.GpuPreflightError(
        "wrong assigned GPU",
        evidence={
            "schema_version": 2,
            "passed": False,
            "devices": [],
            "failure": {"stage": "assignment", "message": "wrong assigned GPU"},
        },
    )
    monkeypatch.setattr(nemo_runner, "context_length_preflight", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        nemo_runner,
        "gpu_hardware_provenance",
        lambda: (_ for _ in ()).throw(failure),
    )
    real_import = builtins.__import__
    actor_imports = []

    def track_import(name, *args, **kwargs):
        if name == "ray" or name.startswith("nemo_rl.distributed.virtual_cluster"):
            actor_imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_import)

    with pytest.raises(runner.GpuPreflightError, match="wrong assigned GPU"):
        nemo_runner.run_nemo_dp2_benchmark(args, manifest)

    assert actor_imports == []
    assert not output.exists()


def test_nemo_dp2_config_owns_two_exact_48_request_engines() -> None:
    manifest = WorkloadManifest.from_path(DATA).with_max_new_tokens(6_000)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=6_012,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
        proof=True,
    )

    config = build_nemo_generation_config(
        profile,
        manifest,
        checkpoint=Path("/checkpoint"),
        load_format="safetensors",
    )

    assert config["model_name"] == "/checkpoint"
    assert config["request_seed"] == 42
    assert config["generation_batch_size"] == 96
    assert config["max_new_tokens"] == 6_000
    assert config["temperature"] == 1.0
    assert config["top_p"] == 1.0
    assert config["top_k"] == 4
    assert config["ignore_eos"] is True
    assert config["_pad_token_id"] == 0
    assert config["colocated"]["enabled"] is False
    assert config["vllm_cfg"]["tensor_parallel_size"] == 1
    assert config["vllm_kwargs"]["max_num_seqs"] == 48
    assert config["vllm_kwargs"]["cudagraph_metrics"] is True
    assert config["generation_worker_cls"].endswith("Evo2NemoRlGenerationWorker")
    assert config["vllm_kwargs"]["worker_extension_cls"].endswith("Evo2NemoRlProofVllmWorkerExtension")


def test_nemo_generation_input_right_pads_mixed_prompts_without_semantic_padding() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 3).with_max_new_tokens(3)

    batch = build_nemo_generation_input(manifest)

    assert batch["input_ids"].shape == (3, 6)
    assert batch["input_lengths"].tolist() == [4, 5, 6]
    for row, request in zip(batch["input_ids"], manifest.requests, strict=True):
        length = len(request.prompt_token_ids)
        assert row[:length].tolist() == list(request.prompt_token_ids)
        assert row[length:].tolist() == [0] * (6 - length)


def test_nemo_tp2_parity_config_requests_all_512_processed_logprobs() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 1).with_max_new_tokens(4)
    profile = Evo2VllmProfile(
        topology="tp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        proof=True,
    )

    config = build_nemo_generation_config(
        profile,
        manifest,
        checkpoint=Path("/checkpoint"),
        load_format="safetensors",
        num_logprobs=512,
    )

    assert config["num_logprobs"] == 512
    assert config["vllm_cfg"]["tensor_parallel_size"] == 2
    assert config["vllm_kwargs"]["max_num_seqs"] == 96
    assert config["vllm_kwargs"]["max_logprobs"] == 512


def test_nemo_output_adapter_retains_full_tokens_logprobs_seeds_and_timings() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    inputs = build_nemo_generation_input(manifest)
    output_ids = torch.zeros((2, inputs["input_ids"].shape[1] + 3), dtype=torch.long)
    logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
    selected_logprob_valid = torch.zeros_like(output_ids, dtype=torch.bool)
    generated = ((65, 67, 71), (66, 68, 72))
    for index, request in enumerate(manifest.requests):
        prompt_length = len(request.prompt_token_ids)
        output_ids[index, :prompt_length] = torch.tensor(request.prompt_token_ids)
        output_ids[index, prompt_length : prompt_length + 3] = torch.tensor(generated[index])
        logprobs[index, prompt_length : prompt_length + 3] = torch.tensor([-0.1, -0.2, -0.3])
        selected_logprob_valid[index, prompt_length : prompt_length + 3] = True
    outputs = BatchedDataDict(
        {
            "output_ids": output_ids,
            "logprobs": logprobs,
            "generation_lengths": torch.tensor([3, 3]),
            "unpadded_sequence_lengths": torch.tensor(
                [len(request.prompt_token_ids) + 3 for request in manifest.requests]
            ),
            "truncated": torch.tensor([True, True]),
            **_explicit_generation_metadata(
                manifest,
                generation_round=3,
                batch_call_slots_per_round=2,
                batch_call_slot=1,
                global_request_indices=(48, 49),
                dp_ranks=(1, 1),
                immutable_local_ordinals=(0, 1),
                data_parallel_size=2,
            ),
            "generation_first_token_latency_s": torch.tensor([0.4, 0.5]),
            "generation_decode_s": torch.tensor([0.2, 0.3]),
            "generation_selected_logprob_valid": selected_logprob_valid,
            "generation_selected_logprob_counts": torch.tensor([3, 3]),
        }
    )

    records, executions, timings = records_from_nemo_generation_output(manifest, outputs)

    assert records[0].output_token_ids == (65, 67, 71)
    assert records[0].output_logprobs == pytest.approx((-0.1, -0.2, -0.3))
    assert executions[0].to_dict() == {
        "execution_uid": "round=3/call=7/global=48/dp=1/request=gdpo-000",
        "request_id": "gdpo-000",
        "global_request_index": 48,
        "generation_round": 3,
        "dp_rank": 1,
        "call_index": 7,
        "seed": 15_000_087,
    }
    assert records[0].requested_max_tokens == 3
    assert records[0].finish_reason == "length"
    assert records[0].stop_reason is None
    assert records[0].stopped_on_eos is False
    assert timings["ttft_s"] == pytest.approx((0.4, 0.5))
    assert timings["decode_s"] == pytest.approx((0.2, 0.3))

    maskless = BatchedDataDict(dict(outputs.data))
    del maskless["generation_selected_logprob_valid"]
    with pytest.raises(TypeError, match="generation_selected_logprob_valid"):
        records_from_nemo_generation_output(manifest, maskless)

    float_seed = BatchedDataDict(dict(outputs.data))
    float_seed["generation_request_seeds"] = float_seed[
        "generation_request_seeds"
    ].to(torch.float64)
    with pytest.raises(TypeError, match="generation_request_seeds"):
        records_from_nemo_generation_output(manifest, float_seed)


def test_full_vocab_evidence_validates_every_chosen_token_semantically() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 1).with_max_new_tokens(3)
    inputs = build_nemo_generation_input(manifest)
    prompt_length = len(manifest.requests[0].prompt_token_ids)
    output_ids = torch.zeros((1, inputs["input_ids"].shape[1] + 3), dtype=torch.long)
    output_ids[0, :prompt_length] = torch.tensor(manifest.requests[0].prompt_token_ids)
    output_ids[0, prompt_length : prompt_length + 3] = torch.tensor([1, 2, 3])
    logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
    dense = torch.full((1, 3, 512), -torch.inf)
    dense[0, 0] = torch.linspace(-8.0, -1.0, 512)
    dense[0, 1] = torch.linspace(-7.0, -0.5, 512)
    dense[0, 2] = torch.linspace(-6.0, -0.25, 512)
    chosen = dense[0, torch.arange(3), torch.tensor([1, 2, 3])]
    logprobs[0, prompt_length : prompt_length + 3] = chosen
    outputs = BatchedDataDict(
        {
            "output_ids": output_ids,
            "logprobs": logprobs,
            "generation_lengths": torch.tensor([3]),
            "unpadded_sequence_lengths": torch.tensor([prompt_length + 3]),
            "truncated": torch.tensor([True]),
            **_explicit_generation_metadata(
                manifest,
                generation_round=3,
                batch_call_slots_per_round=2,
                batch_call_slot=1,
                global_request_indices=(48,),
                dp_ranks=(0,),
                immutable_local_ordinals=(0,),
                data_parallel_size=2,
            ),
            "generation_first_token_latency_s": torch.tensor([0.4]),
            "generation_decode_s": torch.tensor([0.2]),
            "generation_selected_logprob_valid": torch.tensor(
                [[False] * prompt_length + [True, True, True]], dtype=torch.bool
            ),
            "generation_selected_logprob_counts": torch.tensor([3]),
            "generation_vocab_logprobs": dense,
            "generation_logprob_counts": torch.full((1, 3), 512),
        }
    )
    records, _, _ = records_from_nemo_generation_output(manifest, outputs)

    evidence = full_vocab_logprob_evidence_from_nemo_output(
        outputs,
        records=records,
        require_full=True,
    )

    assert evidence["shape"] == [1, 3, 512]
    assert evidence["coverage_counts"] == [[512, 512, 512]]
    assert evidence["chosen_token_oracle_passed"] is True
    assert evidence["logprobs"][0][1][2] == pytest.approx(float(dense[0, 1, 2]))

    outputs["logprobs"][0, prompt_length] += 0.5
    broken_records, _, _ = records_from_nemo_generation_output(manifest, outputs)
    with pytest.raises(AssertionError, match="chosen-token"):
        full_vocab_logprob_evidence_from_nemo_output(
            outputs,
            records=broken_records,
            require_full=True,
        )


def test_full_vocab_evidence_reports_exact_processed_topk_finite_mismatch() -> None:
    dense = torch.full((2, 4, 512), -torch.inf)
    dense[:, :, :4] = torch.tensor([-4.0, -3.0, -2.0, -1.0])
    outputs = BatchedDataDict(
        {
            "generation_vocab_logprobs": dense,
            "generation_logprob_counts": torch.full((2, 4), 512),
        }
    )
    records = tuple(
        GenerationRecord(
            request_id=f"request-{index}",
            prompt_token_ids=(43, 126, 71, 65),
            output_token_ids=(0, 1, 2, 3),
            output_logprobs=(-4.0, -3.0, -2.0, -1.0),
            requested_max_tokens=4,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        )
        for index in range(2)
    )

    expected = (
        "shape=[2, 4, 512], mismatched_positions=8/8, "
        "reported_counts={512: 8}, finite_counts={4: 8}, "
        "first_mismatch=(request=0, step=0, reported=512, finite=4)"
    )
    with pytest.raises(AssertionError, match=re.escape(expected)):
        full_vocab_logprob_evidence_from_nemo_output(
            outputs,
            records=records,
            require_full=True,
        )


def test_full_vocab_evidence_retains_exact_processed_topk_support() -> None:
    dense = torch.full((2, 4, 512), -torch.inf)
    dense[:, :, :4] = torch.tensor([-4.0, -3.0, -2.0, -1.0])
    outputs = BatchedDataDict(
        {
            "generation_vocab_logprobs": dense,
            "generation_logprob_counts": torch.full((2, 4), 512),
        }
    )
    records = tuple(
        GenerationRecord(
            request_id=f"request-{index}",
            prompt_token_ids=(43, 126, 71, 65),
            output_token_ids=(0, 1, 2, 3),
            output_logprobs=(-4.0, -3.0, -2.0, -1.0),
            requested_max_tokens=4,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        )
        for index in range(2)
    )

    evidence = full_vocab_logprob_evidence_from_nemo_output(
        outputs,
        records=records,
        require_full=True,
        expected_finite_support=4,
    )

    assert evidence["shape"] == [2, 4, 512]
    assert evidence["coverage_counts"] == [[512] * 4] * 2
    assert evidence["finite_support_counts"] == [[4] * 4] * 2
    assert evidence["negative_infinity_counts"] == [[508] * 4] * 2
    assert evidence["expected_finite_support"] == 4
    assert evidence["chosen_token_in_finite_support"] is True
    assert evidence["chosen_token_ids"] == [[0, 1, 2, 3]] * 2
    assert evidence["chosen_token_logprobs"] == [[-4.0, -3.0, -2.0, -1.0]] * 2


def test_full_vocab_evidence_rejects_one_ulp_chosen_value_drift() -> None:
    dense = torch.full((1, 1, 512), -torch.inf)
    dense[0, 0, :4] = torch.tensor([-4.0, -3.0, -2.0, -1.0])
    one_ulp_drift = torch.nextafter(dense[0, 0, 0], torch.tensor(-3.0)).item()
    outputs = BatchedDataDict(
        {
            "generation_vocab_logprobs": dense,
            "generation_logprob_counts": torch.full((1, 1), 512),
        }
    )
    records = (
        GenerationRecord(
            request_id="request-0",
            prompt_token_ids=(43, 126, 71, 65),
            output_token_ids=(0,),
            output_logprobs=(one_ulp_drift,),
            requested_max_tokens=1,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        ),
    )

    with pytest.raises(AssertionError, match="bitwise"):
        full_vocab_logprob_evidence_from_nemo_output(
            outputs,
            records=records,
            require_full=True,
            expected_finite_support=4,
        )


def test_full_vocab_evidence_rejects_nonfinite_chosen_processed_logprob() -> None:
    dense = torch.full((1, 1, 512), -torch.inf)
    dense[0, 0, :4] = torch.tensor([-4.0, -3.0, -2.0, -1.0])
    dense[0, 0, 0] = -torch.inf
    dense[0, 0, 4] = -5.0
    outputs = BatchedDataDict(
        {
            "generation_vocab_logprobs": dense,
            "generation_logprob_counts": torch.tensor([[512]]),
        }
    )
    records = (
        GenerationRecord(
            request_id="request-0",
            prompt_token_ids=(43, 126, 71, 65),
            output_token_ids=(0,),
            output_logprobs=(-torch.inf,),
            requested_max_tokens=1,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        ),
    )

    with pytest.raises(AssertionError, match="chosen token is outside finite processed support"):
        full_vocab_logprob_evidence_from_nemo_output(
            outputs,
            records=records,
            require_full=True,
            expected_finite_support=4,
        )


def test_production_nemo_phase_proves_exact_dp_ownership_and_persists_full_outputs(tmp_path) -> None:
    manifest = (
        WorkloadManifest.from_path(DATA)
        .request_slice(0, 1)
        .with_uniform_prompt_length(
            32,
            request_count=8,
            request_id_prefix="shared",
        )
        .with_max_new_tokens(3)
    )
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=35,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        proof=True,
        shared_prefix_state_reuse=True,
        global_wave_size=4,
        max_num_seqs=2,
    )

    def attention_groups():
        block_ids = [101]
        return [
            {
                "kv_cache_group_id": 1,
                "layer_names": ["model.layers.3.self_attention"],
                "block_size_tokens": 16,
                "physical_block_count": 1,
                "physical_block_ids": block_ids,
                "physical_block_ids_sha256": hashlib.sha256(
                    json.dumps(block_ids, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        ]

    def clone_record(phase, index):
        state_copies = [
            {
                "kv_cache_group_id": 0,
                "layer_name": f"model.layers.{state_index}.mixer",
                "state_index": state_index,
                "dtype": "torch.float32",
                "state_shape": [17, 128],
                "block_shape": [128],
                "source_logical_block_index": 0,
                "destination_logical_block_index": 1,
                "source_physical_block_id": 2 * state_index,
                "destination_physical_block_id": 2 * state_index + 1,
                "source_data_ptr": 10_000 + 2 * state_index,
                "destination_data_ptr": 10_001 + 2 * state_index,
                "copied_elements": 128,
                "copied_bytes": 512,
            }
            for state_index in range(8)
        ]
        return {
            "request_id": f"{phase}-clone-{index}",
            "source_miss_request_id": "source",
            "source_snapshot_index": 0,
            "attention_kv_identity_verified": True,
            "num_computed_tokens": 16,
            "prompt_tokens": 32,
            "block_size": 16,
            "source_attention_kv_groups": attention_groups(),
            "reused_attention_kv_groups": attention_groups(),
            "runtime_state_layout": [
                {
                    key: entry[key]
                    for key in (
                        "kv_cache_group_id",
                        "layer_name",
                        "state_index",
                        "dtype",
                        "state_shape",
                        "block_shape",
                        "copied_elements",
                        "copied_bytes",
                    )
                }
                for entry in state_copies
            ],
            "state_copies": state_copies,
            "copy_entries": 8,
            "copied_elements": 1_024,
            "copied_bytes": 4_096,
            "expected_copy_entries": 8,
            "expected_copied_elements": 1_024,
            "expected_copied_bytes": 4_096,
            "all_state_dtypes_fp32": True,
        }

    class FakeWorkerGroup:
        def __init__(self) -> None:
            self.phase = ""
            self.shard_sizes = (0, 0)

        def run_all_workers_single_data(self, method_name, *, run_rank_0_only_axes, phase):
            assert run_rank_0_only_axes == ["tensor_parallel", "pipeline_parallel"]
            if method_name == "reset_evo2_proof_phase":
                self.phase = phase
                return [{"phase": phase, "worker_reset": []}] * 2
            assert method_name == "snapshot_evo2_proof_phase"
            return [
                {
                    "phase": phase,
                    "resolved_config": {"cache": {"block_size": 16}},
                    "cudagraph_observations": [
                        {
                            "phase": phase,
                            "engine_index": 0,
                            "num_unpadded_tokens": shard_size,
                            "num_padded_tokens": shard_size,
                            "num_paddings": 0,
                            "runtime_mode": "CUDAGraphMode.FULL",
                        }
                        for _ in range(2)
                    ],
                    "cudagraph_summary": [],
                    "scheduler_observations": [
                        {
                            "phase": phase,
                            "engine_index": 0,
                            "preemption_events": 0,
                            "recompute_events": 0,
                            "prefix_preempted_requests": 0,
                            "prefix_preempted_queries": 0,
                            "prefix_preempted_hits": 0,
                            "preempted_prompt_recomputed_tokens": 0,
                            "prompt_tokens_computed": 32,
                            "prompt_tokens_cached": 16 * (shard_size - int(phase.endswith("wave-000"))),
                            "prompt_tokens_total": 32 * shard_size,
                            "num_running_requests": shard_size,
                            "num_waiting_requests": 0,
                            "num_skipped_waiting_requests": 0,
                        }
                    ],
                    "worker_proof": [
                        {
                            "rank": 0,
                            "fir_routes": {"direct": {"calls": 9, "requests": shard_size, "tokens": 18}},
                            "mamba_state_copies": {
                                "copy_calls": 8,
                                "copied_elements": 1_024,
                                "copied_bytes": 4_096,
                            },
                            "mamba_prefix_clones": {
                                "cache_miss_count": int(phase.endswith("wave-000")),
                                "cache_miss_request_ids": (["source"] if phase.endswith("wave-000") else []),
                                "prefix_sources": [
                                    {
                                        "request_id": "source",
                                        "prompt_tokens": 32,
                                        "snapshots": [
                                            {
                                                "snapshot_index": 0,
                                                "num_computed_tokens_before_step": 0,
                                                "num_scheduled_tokens": 32,
                                                "directly_observed_prefix_tokens": 16,
                                                "attention_kv_groups": attention_groups(),
                                            }
                                        ],
                                    }
                                ],
                                "clone_count": shard_size - int(phase.endswith("wave-000")),
                                "requests": [
                                    clone_record(phase, index)
                                    for index in range(shard_size - int(phase.endswith("wave-000")))
                                ],
                            },
                        }
                    ],
                }
                for shard_size in self.shard_sizes
            ]

    class FakeGeneration:
        def __init__(self) -> None:
            self.worker_group = FakeWorkerGroup()
            self.calls = []
            self.cache_invalidations = 0

        def invalidate_kv_cache(self):
            self.cache_invalidations += 1
            return True

        def generate(
            self,
            data,
            greedy=False,
            *,
            request_envelope,
        ):
            assert greedy is True
            assert isinstance(request_envelope, GenerationRequestEnvelope)
            assert request_envelope.generation_round == 3
            assert request_envelope.flattened_call_index == 6 + len(self.calls)
            assert request_envelope.requests[0].global_request_index == (
                48 + 4 * len(self.calls)
            )
            request_count = len(data["input_ids"])
            self.calls.append(
                (
                    request_envelope.generation_round,
                    request_envelope.flattened_call_index,
                    request_envelope.requests[0].global_request_index,
                    request_count,
                )
            )
            self.worker_group.shard_sizes = tuple(
                len(indices) for indices in request_envelope.partition_indices
            )
            output_ids = torch.zeros((request_count, data["input_ids"].shape[1] + 3), dtype=torch.long)
            logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
            selected_logprob_valid = torch.zeros_like(output_ids, dtype=torch.bool)
            for row_index, prompt_length in enumerate(data["input_lengths"].tolist()):
                output_ids[row_index, :prompt_length] = data["input_ids"][row_index, :prompt_length]
                output_ids[row_index, prompt_length : prompt_length + 3] = torch.tensor([65, 67, 71])
                logprobs[row_index, prompt_length : prompt_length + 3] = torch.tensor([-0.1, -0.2, -0.3])
                selected_logprob_valid[row_index, prompt_length : prompt_length + 3] = True
            dense = torch.full((request_count, 3, 512), -20.0)
            for row_index in range(request_count):
                dense[row_index, torch.arange(3), torch.tensor([65, 67, 71])] = torch.tensor([-0.1, -0.2, -0.3])
            outputs = BatchedDataDict(
                {
                    "output_ids": output_ids,
                    "logprobs": logprobs,
                    "generation_lengths": torch.full((request_count,), 3),
                    "unpadded_sequence_lengths": data["input_lengths"] + 3,
                    "truncated": torch.ones(request_count, dtype=torch.bool),
                    **_metadata_from_request_envelope(request_envelope),
                    "generation_first_token_latency_s": torch.full((request_count,), 0.4),
                    "generation_decode_s": torch.full((request_count,), 0.2),
                    "generation_selected_logprob_valid": selected_logprob_valid,
                    "generation_selected_logprob_counts": torch.full(
                        (request_count,), 3, dtype=torch.long
                    ),
                    "generation_num_cached_tokens": torch.tensor(
                        [0, 16, 0, 16]
                        if request_envelope.flattened_call_index == 6
                        else [16, 16, 16, 16]
                    ),
                    "generation_vocab_logprobs": dense,
                    "generation_logprob_counts": torch.full((request_count, 3), 512),
                }
            )
            return outputs

    times = iter((10.0, 11.0, 20.0, 21.0))
    generation = FakeGeneration()
    result = run_nemo_generation_phase(
        generation=generation,
        manifest=manifest,
        profile=profile,
        phase="steady-0",
        sample_index=0,
        generation_round=3,
        global_call_index_start=6,
        global_request_index_start=48,
        caller_ledger_admission=_phase_caller_ledger(
            manifest,
            profile,
            phase="steady-0",
            generation_round=3,
            global_request_index_start=48,
            request_envelope_namespace="test/contract",
        ),
        full_output_path=tmp_path / "nemo.outputs.jsonl.gz",
        memory_monitor_factory=lambda: PeakMemoryMonitor(lambda: (1_000, 2_000)),
        ray_get=lambda futures: futures,
        clock=lambda: next(times),
        greedy=True,
        require_full_vocab_logprobs=True,
        request_envelope_namespace="test/contract",
    )

    assert result.sample.generation_s == 2.0
    assert result.sample.generated_tokens == 24
    assert [record.global_request_index for record in result.request_executions] == list(range(48, 56))
    assert [record.dp_rank for record in result.request_executions] == [0, 0, 1, 1] * 2
    assert [record.generation_round for record in result.request_executions] == [3] * 8
    assert [record.call_index for record in result.request_executions] == [6] * 4 + [7] * 4
    assert [record.seed for record in result.request_executions] == [
        12_000_078,
        12_000_079,
        13_000_081,
        13_000_082,
        14_000_084,
        14_000_085,
        15_000_087,
        15_000_088,
    ]
    assert generation.calls == [(3, 6, 48, 4), (3, 7, 52, 4)]
    assert len(result.wave_proofs) == 2
    assert generation.cache_invalidations == 1
    assert [engine["full_decode_proof"]["passed"] for engine in result.wave_proofs[0]["engines"]] == [
        True,
        True,
    ]
    assert result.full_output_artifact["generated_token_count"] == 24
    assert result.wave_proofs[0]["full_vocab_logprobs"]["shape"] == [4, 3, 512]
    assert result.wave_proofs[0]["full_vocab_logprobs"]["chosen_token_oracle_passed"] is True
    assert all(
        engine["scheduler_capacity_proof"]["batch_fit_without_preemption"]
        for engine in result.wave_proofs[0]["engines"]
    )
    reuse = result.wave_proofs[0]["shared_prefix_state_reuse"]
    assert reuse["cached_tokens_by_request"] == [0, 16, 0, 16]
    assert reuse["cache_hit_request_count"] == 2
    assert reuse["physical_state_copy_proven"] is True
    assert reuse["phase_prefix_cache_reset_before_first_wave"] is True
    later_reuse = result.wave_proofs[1]["shared_prefix_state_reuse"]
    assert later_reuse["cache_miss_request_count"] == 0
    assert later_reuse["cache_hit_request_count"] == 4
    assert [worker["clone_count"] for worker in later_reuse["worker_state_clones"]] == [2, 2]


def test_nemo_speed_phase_registers_sidecar_and_skips_proof_rpcs(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 4).with_max_new_tokens(1)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=13,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        proof=False,
        global_wave_size=4,
        max_num_seqs=2,
    )
    output = tmp_path / "nemo-speed.json"
    marker = runner.reserve_output_namespace(output)
    sidecar = runner.phase_output_artifact_path(output, phase="steady-0")

    class ForbiddenWorkerGroup:
        def run_all_workers_single_data(self, *args, **kwargs):
            pytest.fail(f"speed lane issued a proof RPC: {args!r} {kwargs!r}")

    class FakeGeneration:
        worker_group = ForbiddenWorkerGroup()

        def generate(
            self,
            data,
            greedy=False,
            *,
            request_envelope,
        ):
            _require(greedy is False, "speed lane changed the sealed sampling mode")
            _require(
                type(request_envelope) is GenerationRequestEnvelope,
                "speed lane did not receive an exact request envelope",
            )
            _require(
                (
                    request_envelope.generation_round,
                    request_envelope.flattened_call_index,
                    request_envelope.requests[0].global_request_index,
                )
                == (0, 0, 0),
                "speed lane semantic coordinates drifted",
            )
            request_count = len(data["input_ids"])
            prompt_width = data["input_ids"].shape[1]
            output_ids = torch.zeros((request_count, prompt_width + 1), dtype=torch.long)
            logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
            selected_logprob_valid = torch.zeros_like(output_ids, dtype=torch.bool)
            for row_index, prompt_length in enumerate(data["input_lengths"].tolist()):
                output_ids[row_index, :prompt_length] = data["input_ids"][row_index, :prompt_length]
                output_ids[row_index, prompt_length] = 65
                logprobs[row_index, prompt_length] = -0.1
                selected_logprob_valid[row_index, prompt_length] = True
            return BatchedDataDict(
                {
                    "output_ids": output_ids,
                    "logprobs": logprobs,
                    "generation_lengths": torch.ones(request_count, dtype=torch.long),
                    "unpadded_sequence_lengths": data["input_lengths"] + 1,
                    "truncated": torch.ones(request_count, dtype=torch.bool),
                    **_metadata_from_request_envelope(request_envelope),
                    "generation_first_token_latency_s": torch.full((request_count,), 0.4),
                    "generation_decode_s": torch.zeros(request_count),
                    "generation_selected_logprob_valid": selected_logprob_valid,
                    "generation_selected_logprob_counts": torch.ones(
                        request_count, dtype=torch.long
                    ),
                }
            )

    times = iter((10.0, 11.0))
    result = run_nemo_generation_phase(
        generation=FakeGeneration(),
        manifest=manifest,
        profile=profile,
        phase="steady-0",
        sample_index=0,
        generation_round=0,
        global_call_index_start=0,
        global_request_index_start=0,
        caller_ledger_admission=_phase_caller_ledger(
            manifest,
            profile,
            phase="steady-0",
            generation_round=0,
            global_request_index_start=0,
            request_envelope_namespace="test/contract",
        ),
        full_output_path=sidecar,
        namespace_output_path=output,
        memory_monitor_factory=lambda: pytest.fail("speed lane started peak-memory polling"),
        ray_get=lambda futures: pytest.fail(f"speed lane resolved proof futures: {futures!r}"),
        clock=lambda: next(times),
        request_envelope_namespace="test/contract",
    )

    _require(result.sample.generation_s == 1.0, "speed-lane generation timer drifted")
    _require(
        result.sample.peak_device_memory_bytes == (),
        "speed lane retained proof-only memory polling",
    )
    _require(result.proof_collected is False, "speed lane collected proof callbacks")
    _require(
        result.wave_proofs[0]["reset_proof"] is None,
        "speed lane reset proof state inside the timed route",
    )
    _require(
        result.wave_proofs[0]["engines"] == [],
        "speed lane retained per-engine proof evidence",
    )
    _require(
        result.full_output_artifact["generated_token_count"] == 4,
        "speed lane output accounting is incomplete",
    )
    runner.write_json_artifact(
        output,
        {"phase": result.phase},
        ownership_validator=lambda: runner.require_output_namespace_reservation(output),
    )
    runner.complete_output_namespace(marker, output_path=output)
    _require(
        not marker.exists() and output.is_file() and sidecar.is_file(),
        "NeMo generation publications did not complete as one owned namespace",
    )


def test_nemo_phase_rejects_engine_request_id_reset_across_waves(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 1).with_uniform_prompt_length(
        12,
        request_count=8,
        request_id_prefix="engine-id",
    ).with_max_new_tokens(1)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        proof=False,
        global_wave_size=4,
        max_num_seqs=2,
    )

    class FakeGeneration:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, data, greedy=False, *, request_envelope):
            _require(greedy is False, "engine-ID hostile fixture changed sampling mode")
            _require(
                type(request_envelope) is GenerationRequestEnvelope,
                "engine-ID hostile fixture received a non-exact envelope",
            )
            self.calls.append(request_envelope.execution_occurrence_sha256)
            request_count = len(data["input_ids"])
            prompt_width = data["input_ids"].shape[1]
            output_ids = torch.zeros(
                (request_count, prompt_width + 1),
                dtype=torch.long,
            )
            logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
            selected_logprob_valid = torch.zeros_like(output_ids, dtype=torch.bool)
            for row_index, prompt_length in enumerate(data["input_lengths"].tolist()):
                output_ids[row_index, :prompt_length] = data["input_ids"][
                    row_index, :prompt_length
                ]
                output_ids[row_index, prompt_length] = 65
                logprobs[row_index, prompt_length] = -0.1
                selected_logprob_valid[row_index, prompt_length] = True
            metadata = _metadata_from_request_envelope(request_envelope)
            metadata["generation_engine_request_ids"] = [
                (
                    f"reset:{request.target_data_parallel_rank}:"
                    f"{request.immutable_local_ordinal}"
                )
                for request in request_envelope.requests
            ]
            return BatchedDataDict(
                {
                    "output_ids": output_ids,
                    "logprobs": logprobs,
                    "generation_lengths": torch.ones(request_count, dtype=torch.long),
                    "unpadded_sequence_lengths": data["input_lengths"] + 1,
                    "truncated": torch.ones(request_count, dtype=torch.bool),
                    **metadata,
                    "generation_first_token_latency_s": torch.full(
                        (request_count,), 0.1
                    ),
                    "generation_decode_s": torch.zeros(request_count),
                    "generation_selected_logprob_valid": selected_logprob_valid,
                    "generation_selected_logprob_counts": torch.ones(
                        request_count, dtype=torch.long
                    ),
                }
            )

    generation = FakeGeneration()
    with pytest.raises(RuntimeError, match="reused within one DP replica"):
        run_nemo_generation_phase(
            generation=generation,
            manifest=manifest,
            profile=profile,
            phase="steady-0",
            sample_index=0,
            generation_round=0,
            global_call_index_start=0,
            global_request_index_start=0,
            caller_ledger_admission=_phase_caller_ledger(
                manifest,
                profile,
                phase="steady-0",
                generation_round=0,
                global_request_index_start=0,
                request_envelope_namespace="test/contract",
            ),
            full_output_path=tmp_path / "reset-engine-id.outputs.jsonl.gz",
            memory_monitor_factory=lambda: pytest.fail(
                "speed lane started peak-memory polling"
            ),
            clock=iter((10.0, 11.0, 20.0, 21.0)).__next__,
            request_envelope_namespace="test/contract",
        )
    _require(
        len(generation.calls) == 2
        and len(set(generation.calls)) == 2,
        "engine-ID reset fixture did not execute two distinct wave occurrences",
    )


@pytest.mark.parametrize("inactive_executes_work", [False, True])
def test_nemo_phase_requires_inactive_dp_replica_to_report_zero_scheduler_and_graph_work(
    tmp_path,
    inactive_executes_work,
) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 1).with_max_new_tokens(1)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=16,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        proof=True,
        global_wave_size=2,
        max_num_seqs=1,
    )
    phase = "steady-0.wave-000"
    active_observation = {
        "phase": phase,
        "engine_index": 0,
        "num_unpadded_tokens": 1,
        "num_padded_tokens": 1,
        "num_paddings": 0,
        "runtime_mode": "CUDAGraphMode.FULL",
    }
    active_scheduler = {
        "phase": phase,
        "engine_index": 0,
        "preemption_events": 0,
        "recompute_events": 0,
        "prefix_preempted_requests": 0,
        "prefix_preempted_queries": 0,
        "prefix_preempted_hits": 0,
        "preempted_prompt_recomputed_tokens": 0,
        "prompt_tokens_computed": len(manifest.requests[0].prompt_token_ids),
        "prompt_tokens_cached": 0,
        "prompt_tokens_total": len(manifest.requests[0].prompt_token_ids),
        "num_running_requests": 1,
        "num_waiting_requests": 0,
        "num_skipped_waiting_requests": 0,
    }

    class FakeWorkerGroup:
        def run_all_workers_single_data(self, method_name, *, run_rank_0_only_axes, phase):
            assert run_rank_0_only_axes == ["tensor_parallel", "pipeline_parallel"]
            if method_name == "reset_evo2_proof_phase":
                return [{"phase": phase}, {"phase": phase}]
            assert method_name == "snapshot_evo2_proof_phase"
            return [
                {
                    "phase": phase,
                    "resolved_config": {"cache": {"block_size": 16}},
                    "cudagraph_observations": [active_observation],
                    "scheduler_observations": [active_scheduler],
                    "worker_proof": [],
                },
                {
                    "phase": phase,
                    "resolved_config": {"cache": {"block_size": 16}},
                    "cudagraph_observations": ([active_observation] if inactive_executes_work else []),
                    "scheduler_observations": ([active_scheduler] if inactive_executes_work else []),
                    "worker_proof": [],
                },
            ]

    class FakeGeneration:
        worker_group = FakeWorkerGroup()

        def generate(self, data, greedy=False, *, request_envelope):
            assert isinstance(request_envelope, GenerationRequestEnvelope)
            prompt_length = int(data["input_lengths"][0])
            output_ids = torch.zeros((1, prompt_length + 1), dtype=torch.long)
            output_ids[0, :prompt_length] = data["input_ids"][0, :prompt_length]
            output_ids[0, prompt_length] = 65
            logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
            logprobs[0, prompt_length] = -0.125
            selected_logprob_valid = torch.zeros_like(output_ids, dtype=torch.bool)
            selected_logprob_valid[0, prompt_length] = True
            return BatchedDataDict(
                {
                    "output_ids": output_ids,
                    "logprobs": logprobs,
                    "generation_lengths": torch.ones(1, dtype=torch.long),
                    "unpadded_sequence_lengths": data["input_lengths"] + 1,
                    "truncated": torch.ones(1, dtype=torch.bool),
                    **_metadata_from_request_envelope(request_envelope),
                    "generation_first_token_latency_s": torch.tensor([0.1]),
                    "generation_decode_s": torch.tensor([0.0]),
                    "generation_selected_logprob_valid": selected_logprob_valid,
                    "generation_selected_logprob_counts": torch.ones(
                        1, dtype=torch.long
                    ),
                }
            )

    kwargs = dict(
        generation=FakeGeneration(),
        manifest=manifest,
        profile=profile,
        phase="steady-0",
        sample_index=0,
        generation_round=0,
        global_call_index_start=0,
        global_request_index_start=0,
        caller_ledger_admission=_phase_caller_ledger(
            manifest,
            profile,
            phase="steady-0",
            generation_round=0,
            global_request_index_start=0,
            request_envelope_namespace="test/contract",
        ),
        full_output_path=tmp_path / "inactive.outputs.jsonl.gz",
        memory_monitor_factory=lambda: PeakMemoryMonitor(lambda: (1_000, 2_000)),
        ray_get=lambda futures: futures,
        clock=iter((10.0, 11.0)).__next__,
        request_envelope_namespace="test/contract",
    )
    if inactive_executes_work:
        with pytest.raises(AssertionError, match="inactive DP replica executed"):
            run_nemo_generation_phase(**kwargs)
    else:
        result = run_nemo_generation_phase(**kwargs)
        assert result.wave_proofs[0]["inactive_engines"] == [
            {
                "dp_rank": 1,
                "request_count": 0,
                "inactive": True,
                "phase": phase,
                "resolved_config": {"cache": {"block_size": 16}},
                "cudagraph_observations": [],
                "scheduler_observations": [],
                "worker_proof": [],
            }
        ]


def test_nemo_speed_snapshots_actual_resolved_configs_outside_generation() -> None:
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=64,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        proof=False,
        global_wave_size=4,
        max_num_seqs=2,
    )
    calls = []

    class FakeWorkerGroup:
        def run_all_workers_single_data(self, method_name, *, run_rank_0_only_axes):
            calls.append((method_name, run_rank_0_only_axes))
            return [profile.expected_resolved_config(), profile.expected_resolved_config()]

    generation = type("Generation", (), {"worker_group": FakeWorkerGroup()})()
    resolved = snapshot_and_validate_nemo_resolved_configs(
        generation,
        profile=profile,
        ray_get=lambda futures: futures,
    )

    assert calls == [
        (
            "snapshot_evo2_resolved_config",
            ["tensor_parallel", "pipeline_parallel"],
        )
    ]
    assert resolved == [
        profile.expected_resolved_config(),
        profile.expected_resolved_config(),
    ]
