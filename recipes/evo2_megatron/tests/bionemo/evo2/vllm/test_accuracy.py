# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from dataclasses import replace
from pathlib import Path

import pytest

from bionemo.evo2.vllm.accuracy import (
    CANONICAL_7B_CHECKPOINT,
    CANONICAL_7B_PROMPTS_SHA256,
    build_canonical_identity_contract,
    build_canonical_identity_manifest,
    build_common_prefix_identity_manifest,
    build_homogeneous_identity_schedule,
    build_mixed_canonical_identity_contract,
    build_mixed_canonical_identity_manifest,
    build_mixed_identity_admission_bundle,
    build_mixed_identity_schedule,
    load_canonical_7b_identity_cases,
    load_common_prefix_identity_cases,
    validate_canonical_identity_manifest,
    validate_canonical_identity_output_artifact,
    validate_common_prefix_identity_manifest,
    validate_common_prefix_identity_output_artifacts,
    validate_homogeneous_identity_phase_evidence,
    validate_mixed_canonical_identity_output_artifact,
    validate_mixed_identity_phase_evidence,
)
from bionemo.evo2.vllm.benchmark import GenerationRecord, WorkloadManifest, WorkloadRequest
from bionemo.evo2.vllm.runner import (
    RequestExecutionRecord,
    build_request_execution_records,
    write_full_generation_records_artifact,
)
from bionemo.evo2.vllm.tokenizer_io import SnapshotBoundTokenizer


PROMPTS_CSV = Path(__file__).resolve().parent.parent / "data" / "prompts.csv"
TOKENIZER_JSON = Path(__file__).resolve().parents[4] / "tokenizers/nucleotide_fast_tokenizer_512/tokenizer.json"


def _tokenizer() -> SnapshotBoundTokenizer:
    return SnapshotBoundTokenizer.from_path(TOKENIZER_JSON)


def _base_manifest(*, source_checkpoint: str = CANONICAL_7B_CHECKPOINT) -> WorkloadManifest:
    return WorkloadManifest(
        schema_version=1,
        name="canonical-identity-base",
        source_checkpoint=source_checkpoint,
        checkpoint_manifest_sha256="1" * 64,
        checkpoint_index_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        requests=(WorkloadRequest(request_id="base", prompt_token_ids=(1,)),),
        max_new_tokens=1,
        temperature=0.5,
        top_p=0.5,
        top_k=4,
        seed=7,
        dtype="bfloat16",
        ignore_eos=False,
        stop_token_ids=(0,),
    )


def test_canonical_7b_identity_cases_pin_unchanged_protocol_and_inputs() -> None:
    cases = load_canonical_7b_identity_cases(PROMPTS_CSV)

    assert CANONICAL_7B_CHECKPOINT == "evo2/7b-1m:1.0"
    assert CANONICAL_7B_PROMPTS_SHA256 == "7e525370e8fb66ef20c0e8d7959f6a0f8e78e5e973819cf3db6f4d23b0e19c0c"
    assert [case.case_index for case in cases] == [0, 1, 2, 3]
    assert [case.prompt_length for case in cases] == [3269, 3528, 3080, 3808]
    assert [case.target_length for case in cases] == [500] * 4
    assert [case.expected_identity_percent for case in cases] == [97.60, 89.63, 80.03, 84.57]
    assert [case.minimum_identity_percent for case in cases] == pytest.approx([87.84, 80.667, 72.027, 76.113])
    assert [case.minimum_matches for case in cases] == [440, 404, 361, 381]
    assert [case.prompt_sha256 for case in cases] == [
        "a92c212221dbae0b8c8afef6f4cf53ec247efe3c41ca50ce8a5084fe8b275d8e",
        "b3fc30040112d22c7bc8c699128c3848e56ab3e92fdc0d72dfdcf2fb7bf43db2",
        "acc63af126428db972b9eda5187db2e3daa5c26d4222365a70d50e00adb91738",
        "ba411e918b8a75de8be7853075df7e12efff5964bb28fe14e9b3af9cbedc0cbf",
    ]
    assert [case.target_sha256 for case in cases] == [
        "40917dd09186f4ddf54a8963bb7165fb3c91b3141a569dc82682a766cffb500c",
        "fdf6b6006f176f2c9e8aea829d5360a39718d434cd359e4eb9b29a8832fa8f88",
        "48db0b8e5a58737820ee7839adfc7b4443990c790eb789c35e7da76127957a96",
        "29478edc8329aaf37faef18abc5e6c18457f96f9d796b72dd4cf22bebee8dd74",
    ]
    assert all(case.midpoint_fraction == 0.5 for case in cases)
    assert all(case.max_new_tokens == 500 for case in cases)
    assert all(case.temperature == 1.0 and case.top_k == 1 and case.seed == 42 for case in cases)


@pytest.mark.parametrize("request_count", (4, 96))
def test_mixed_canonical_identity_manifest_is_one_interleaved_physical_batch(request_count) -> None:
    cases = load_canonical_7b_identity_cases(PROMPTS_CSV)
    stage_manifests = {
        stage: build_mixed_canonical_identity_manifest(
            _base_manifest(),
            cases=cases,
            prompts_csv=PROMPTS_CSV,
            tokenizer=_tokenizer(),
            request_count=count,
            request_id_prefix=f"mixed-{'b4' if count == 4 else 'b96'}",
        )
        for stage, count in (("mixed-b4", 4), ("mixed-b96", 96))
    }
    manifest = stage_manifests["mixed-b4" if request_count == 4 else "mixed-b96"]
    expected_case_indices = [index % 4 for index in range(request_count)]
    expected_prompt_lengths = [cases[index].prompt_length for index in expected_case_indices]
    expected_request_ids = [
        f"mixed-{'b4' if request_count == 4 else 'b96'}-case{case_index}-occurrence{index // 4:04d}"
        for index, case_index in enumerate(expected_case_indices)
    ]

    if [len(request.prompt_token_ids) for request in manifest.requests] != expected_prompt_lengths:
        raise AssertionError("mixed canonical prompts were padded, truncated, or reordered")
    if [request.request_id for request in manifest.requests] != expected_request_ids:
        raise AssertionError("mixed canonical semantic case/occurrence identities drifted")
    if len({request.prompt_token_ids for request in manifest.requests}) != 4:
        raise AssertionError("mixed canonical manifest does not contain all four distinct prompts")

    schedule = build_mixed_identity_schedule(
        topology="tp2",
        request_count=request_count,
        global_wave_size=request_count,
        request_id_prefix=f"mixed-{'b4' if request_count == 4 else 'b96'}",
    )
    contract = build_mixed_canonical_identity_contract(
        cases=cases,
        schedule=schedule,
        stage_manifests=stage_manifests,
        prompts_csv=PROMPTS_CSV,
        tokenizer_path=TOKENIZER_JSON,
    )

    if schedule.global_request_shapes != (request_count,) or schedule.engine_request_shapes != ((request_count,),):
        raise AssertionError("mixed canonical TP2 schedule is not one physical call")
    if schedule.mixed_case_batching is not True or schedule.semantic_padding is not False:
        raise AssertionError("mixed canonical schedule did not bind real unpadded mixed batching")
    if contract["case_order"] != expected_case_indices:
        raise AssertionError("mixed canonical contract lost interleaved case ownership")
    if contract["minimum_matches"] != [440, 404, 361, 381]:
        raise AssertionError("mixed canonical contract changed the original integer identity floors")
    if contract["schedule"]["global_request_shapes"] != [request_count]:
        raise AssertionError("mixed canonical contract did not bind the one-call physical shape")
    if [stage["manifest_sha256"] for stage in contract["stages"]] != [
        stage_manifests["mixed-b4"].sha256,
        stage_manifests["mixed-b96"].sha256,
    ]:
        raise AssertionError("mixed canonical contract did not independently bind both stage manifests")


def test_mixed_identity_admission_orders_b4_then_b96_with_advancing_seed_coordinates() -> None:
    bundle = build_mixed_identity_admission_bundle(topology="tp2", base_seed=42)
    attempts = bundle["attempts"]

    if [attempt["stage"] for attempt in attempts] != ["mixed-b4", "mixed-b96"]:
        raise AssertionError("mixed admission did not preserve B4-then-B96 order")
    if [attempt["global_call_index"] for attempt in attempts] != [0, 1]:
        raise AssertionError("mixed admission reset the B96 call index")
    if [attempt["global_request_index_start"] for attempt in attempts] != [0, 4]:
        raise AssertionError("mixed admission reused the B4 request range")
    if [attempt["request_count"] for attempt in attempts] != [4, 96]:
        raise AssertionError("mixed admission changed B4/B96 cardinality")
    if attempts[0]["request_seeds"] != [42, 43, 44, 45]:
        raise AssertionError("mixed B4 request seeds drifted")
    if attempts[1]["request_seeds"] != [1_000_045 + index for index in range(96)]:
        raise AssertionError("mixed B96 request seeds did not advance to call 1")
    if set(attempts[0]["request_seeds"]) & set(attempts[1]["request_seeds"]):
        raise AssertionError("mixed B4/B96 request seed streams overlap")
    if set(attempts[0]["request_ids"]) & set(attempts[1]["request_ids"]):
        raise AssertionError("mixed B4/B96 semantic request identities overlap")
    if attempts[0]["request_ids"][0] != "mixed-b4-case0-occurrence0000":
        raise AssertionError("mixed B4 request identity is not stage-qualified")
    if attempts[1]["request_ids"][0] != "mixed-b96-case0-occurrence0000":
        raise AssertionError("mixed B96 request identity is not stage-qualified")


def test_mixed_identity_dp2_uses_contiguous_local2_and_local48_case_geometry() -> None:
    b4 = build_mixed_identity_schedule(topology="dp2", request_count=4, global_wave_size=4)
    b96 = build_mixed_identity_schedule(topology="dp2", request_count=96, global_wave_size=96)

    if b4.engine_request_shapes != ((2, 2),):
        raise AssertionError("mixed DP2 B4 did not shard to contiguous local2/local2")
    if b4.engine_case_counts != (((1, 1, 0, 0), (0, 0, 1, 1)),):
        raise AssertionError("mixed DP2 B4 case ownership is not contiguous")
    if b96.engine_request_shapes != ((48, 48),):
        raise AssertionError("mixed DP2 B96 did not shard to contiguous local48/local48")
    if b96.engine_case_counts != (((12, 12, 12, 12), (12, 12, 12, 12)),):
        raise AssertionError("mixed DP2 B96 does not retain exactly 12 occurrences per case and rank")
    if b4.case_order != (0, 1, 2, 3) or b96.case_order != tuple(range(4)) * 24:
        raise AssertionError("mixed DP2 schedule changed global interleaved order")


def test_canonical_identity_loader_rejects_modified_source(tmp_path) -> None:
    modified = tmp_path / "prompts.csv"
    modified.write_bytes(PROMPTS_CSV.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA256"):
        load_canonical_7b_identity_cases(modified)


def test_common_2048_cases_are_deterministically_derived_from_frozen_7b_protocol() -> None:
    canonical = load_canonical_7b_identity_cases(PROMPTS_CSV)
    common = load_common_prefix_identity_cases(PROMPTS_CSV)

    assert [case.case_index for case in common] == [0, 1, 2, 3]
    assert [case.prompt for case in common] == [case.sequence[:2048] for case in canonical]
    assert [case.target for case in common] == [case.sequence[2048:2548] for case in canonical]
    assert all(case.prompt_length == 2048 and case.target_length == 500 for case in common)
    assert all(case.max_new_tokens == 500 for case in common)
    assert all(case.temperature == 1.0 and case.top_k == 1 and case.seed == 42 for case in common)


def test_common_2048_manifest_repeats_only_one_case_without_padding() -> None:
    case = load_common_prefix_identity_cases(PROMPTS_CSV)[1]
    manifest = build_common_prefix_identity_manifest(
        _base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=96,
        request_id_prefix="common-case1",
    )

    assert manifest.name == "common-2048-7b-identity-case1-n96"
    assert len(manifest.requests) == 96
    assert {len(request.prompt_token_ids) for request in manifest.requests} == {2048}
    assert len({request.prompt_token_ids for request in manifest.requests}) == 1
    assert manifest.max_new_tokens == 500
    assert manifest.top_k == 1
    validate_common_prefix_identity_manifest(manifest, case=case, request_count=96)


def _write_common_outputs(path, manifest, outputs):
    records = tuple(
        GenerationRecord(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=tuple(ord(character) for character in output),
            output_logprobs=(-0.1,) * 500,
            requested_max_tokens=500,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        )
        for request, output in zip(manifest.requests, outputs, strict=True)
    )
    return write_full_generation_records_artifact(
        path,
        records=records,
        execution_records=build_request_execution_records(
            manifest,
            global_request_offset=0,
            dp_rank=0,
            dp_size=1,
            generation_round=0,
            call_index=0,
        ),
        decode_output_token_ids=lambda token_ids: "".join(chr(token_id) for token_id in token_ids),
    )


def test_common_2048_output_gate_compares_every_request_to_serial_target_identity(tmp_path) -> None:
    case = load_common_prefix_identity_cases(PROMPTS_CSV)[0]
    reference_manifest = build_common_prefix_identity_manifest(
        _base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=1,
        request_id_prefix="serial",
    )
    candidate_manifest = build_common_prefix_identity_manifest(
        _base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=2,
        request_id_prefix="batched",
    )

    reference_artifact = _write_common_outputs(
        tmp_path / "reference.outputs.jsonl.gz",
        reference_manifest,
        [case.target],
    )
    candidate_artifact = _write_common_outputs(
        tmp_path / "candidate.outputs.jsonl.gz",
        candidate_manifest,
        [case.target, case.target],
    )

    evidence = validate_common_prefix_identity_output_artifacts(
        reference_artifact,
        candidate_artifact,
        case=case,
        expected_candidate_request_ids=[request.request_id for request in candidate_manifest.requests],
        decode_output_token_ids=lambda token_ids: "".join(chr(token_id) for token_id in token_ids),
    )

    assert evidence["serial_target_identity_percent"] == 100.0
    assert evidence["minimum_candidate_target_identity_percent"] == 100.0
    assert evidence["candidate_request_count"] == 2
    assert evidence["passed"] is True


def test_common_2048_output_gate_rejects_one_request_more_than_five_points_below_serial(tmp_path) -> None:
    case = load_common_prefix_identity_cases(PROMPTS_CSV)[0]
    reference_manifest = build_common_prefix_identity_manifest(
        _base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=1,
        request_id_prefix="serial",
    )
    candidate_manifest = build_common_prefix_identity_manifest(
        _base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=2,
        request_id_prefix="batched",
    )
    mismatched_prefix = "".join("A" if base != "A" else "C" for base in case.target[:26])
    low_identity = mismatched_prefix + case.target[26:]
    reference_artifact = _write_common_outputs(
        tmp_path / "reference.outputs.jsonl.gz",
        reference_manifest,
        [case.target],
    )
    candidate_artifact = _write_common_outputs(
        tmp_path / "candidate.outputs.jsonl.gz",
        candidate_manifest,
        [case.target, low_identity],
    )

    with pytest.raises(AssertionError, match="serial-reference bound"):
        validate_common_prefix_identity_output_artifacts(
            reference_artifact,
            candidate_artifact,
            case=case,
            expected_candidate_request_ids=[request.request_id for request in candidate_manifest.requests],
            decode_output_token_ids=lambda token_ids: "".join(chr(token_id) for token_id in token_ids),
        )


def test_canonical_identity_manifest_is_one_homogeneous_case() -> None:
    case = load_canonical_7b_identity_cases(PROMPTS_CSV)[2]
    manifest = build_canonical_identity_manifest(
        _base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=1_000,
        request_id_prefix="identity-case2",
    )

    assert manifest.source_checkpoint == CANONICAL_7B_CHECKPOINT
    assert manifest.name == "canonical-7b-identity-case2-n1000"
    assert len(manifest.requests) == 1_000
    assert len({request.prompt_token_ids for request in manifest.requests}) == 1
    assert len(manifest.requests[0].prompt_token_ids) == case.prompt_length
    assert manifest.max_new_tokens == 500
    assert manifest.temperature == 1.0
    assert manifest.top_p == 1.0
    assert manifest.top_k == 1
    assert manifest.seed == 42
    assert manifest.ignore_eos is True
    assert manifest.stop_token_ids == ()
    assert manifest.prompt_source_path == str(PROMPTS_CSV.resolve())
    assert manifest.prompt_source_sha256 == CANONICAL_7B_PROMPTS_SHA256
    validate_canonical_identity_manifest(manifest, case=case, request_count=1_000)

    requests = list(manifest.requests)
    requests[-1] = WorkloadRequest(request_id=requests[-1].request_id, prompt_token_ids=(99,))
    mixed = WorkloadManifest(**{**manifest.constructor_kwargs(), "requests": tuple(requests)})
    with pytest.raises(AssertionError, match="homogeneous"):
        validate_canonical_identity_manifest(mixed, case=case, request_count=1_000)


def test_canonical_identity_manifest_rejects_noncanonical_checkpoint() -> None:
    case = load_canonical_7b_identity_cases(PROMPTS_CSV)[0]

    with pytest.raises(ValueError, match="evo2/7b-1m:1.0"):
        build_canonical_identity_manifest(
            _base_manifest(source_checkpoint="microviridae"),
            case=case,
            prompts_csv=PROMPTS_CSV,
            tokenizer=_tokenizer(),
            request_count=96,
            request_id_prefix="identity-case0",
        )


@pytest.mark.parametrize(
    ("topology", "request_count", "global_wave_size", "global_shapes", "engine_shapes"),
    (
        ("tp2", 96, 96, (96,), ((96,),)),
        ("dp2", 96, 96, (96,), ((48, 48),)),
        ("tp2", 1_000, 96, (96,) * 10 + (40,), ((96,),) * 10 + ((40,),)),
        ("dp2", 1_000, 96, (96,) * 10 + (40,), ((48, 48),) * 10 + ((20, 20),)),
        ("tp2", 85, 32, (32, 32, 21), ((32,), (32,), (21,))),
        ("dp2", 85, 32, (32, 32, 21), ((16, 16), (16, 16), (11, 10))),
    ),
)
def test_homogeneous_identity_schedule_pins_every_physical_shape(
    topology,
    request_count,
    global_wave_size,
    global_shapes,
    engine_shapes,
) -> None:
    schedule = build_homogeneous_identity_schedule(
        topology=topology,
        request_count=request_count,
        global_wave_size=global_wave_size,
    )

    assert schedule.topology == topology
    assert schedule.request_count == request_count
    assert schedule.global_wave_size == global_wave_size
    assert schedule.global_request_shapes == global_shapes
    assert schedule.engine_request_shapes == engine_shapes
    assert schedule.semantic_padding is False
    assert schedule.mixed_case_batching is False


def test_canonical_identity_contract_binds_case_sampling_and_physical_schedule(tmp_path) -> None:
    case = load_canonical_7b_identity_cases(PROMPTS_CSV)[1]
    schedule = build_homogeneous_identity_schedule(
        topology="dp2",
        request_count=1_000,
        global_wave_size=96,
    )
    contract = build_canonical_identity_contract(
        case=case,
        schedule=schedule,
        prompts_csv=PROMPTS_CSV,
        tokenizer_path=TOKENIZER_JSON,
    )

    assert contract["schema_version"] == 1
    assert contract["checkpoint"] == "evo2/7b-1m:1.0"
    assert contract["case_index"] == 1
    assert contract["midpoint_fraction"] == 0.5
    assert contract["prompt_length"] == 3528
    assert contract["target_length"] == 500
    assert contract["target_sha256"] == "fdf6b6006f176f2c9e8aea829d5360a39718d434cd359e4eb9b29a8832fa8f88"
    assert contract["expected_identity_percent"] == 89.63
    assert contract["minimum_identity_percent"] == pytest.approx(80.667)
    assert contract["sampling"] == {
        "max_new_tokens": 500,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 42,
        "ignore_eos": True,
        "stop_token_ids": [],
    }
    assert contract["schedule"]["global_request_shapes"] == [96] * 10 + [40]
    assert contract["schedule"]["engine_request_shapes"] == [[48, 48]] * 10 + [[20, 20]]
    assert contract["schedule"]["mixed_case_batching"] is False
    assert contract["schedule"]["semantic_padding"] is False


def _physical_phase(schedule, *, phase_name="identity", execution_coordinates=None):
    waves = []
    observations = []
    start = 0
    for wave_index, (global_shape, engine_shapes) in enumerate(
        zip(schedule.global_request_shapes, schedule.engine_request_shapes, strict=True)
    ):
        wave_phase = f"{phase_name}.wave-{wave_index:03d}"
        stop = start + global_shape
        decode_replay_count = 499 if schedule.mixed_case_batching else 1
        if schedule.topology == "tp2":
            observations.extend(
                {
                    "phase": wave_phase,
                    "runtime_mode": "CUDAGraphMode.FULL",
                    "num_unpadded_tokens": global_shape,
                    "num_padded_tokens": global_shape,
                    "num_paddings": 0,
                    "request_dimensions": {
                        "schema_version": 1,
                        "source": "iteration-stats-bound-to-cudagraph-dispatch",
                        "prefill_req_count": 0,
                        "decode_req_count": global_shape,
                        "token_count": global_shape,
                        "prompt_token_count": 0,
                        "first_token_event_count": 0,
                    },
                }
                for _ in range(decode_replay_count)
            )
            waves.append(
                {
                    "wave_index": wave_index,
                    "start": start,
                    "stop": stop,
                    "request_count": global_shape,
                    "full_decode_proof": {
                        "batch_size": global_shape,
                        "max_new_tokens": 500,
                        "maximum_full_batch": global_shape,
                        "passed": True,
                    },
                    "scheduler_capacity_proof": {
                        "global_wave_size": global_shape,
                        "engine_request_count": global_shape,
                        "maximum_running_requests": global_shape,
                        "batch_fit_without_preemption": True,
                    },
                }
            )
        else:
            engines = []
            for dp_rank, engine_shape in enumerate(engine_shapes):
                engine_observations = [
                    {
                        "phase": wave_phase,
                        "runtime_mode": "CUDAGraphMode.FULL",
                        "num_unpadded_tokens": engine_shape,
                        "num_padded_tokens": engine_shape,
                        "num_paddings": 0,
                        "request_dimensions": {
                            "schema_version": 1,
                            "source": "iteration-stats-bound-to-cudagraph-dispatch",
                            "prefill_req_count": 0,
                            "decode_req_count": engine_shape,
                            "token_count": engine_shape,
                            "prompt_token_count": 0,
                            "first_token_event_count": 0,
                        },
                    }
                    for _ in range(decode_replay_count)
                ]
                engines.append(
                    {
                        "dp_rank": dp_rank,
                        "request_count": engine_shape,
                        "cudagraph_observations": engine_observations,
                        "full_decode_proof": {
                            "batch_size": engine_shape,
                            "max_new_tokens": 500,
                            "maximum_full_batch": engine_shape,
                            "passed": True,
                        },
                        "scheduler_capacity_proof": {
                            "global_wave_size": global_shape,
                            "engine_request_count": engine_shape,
                            "maximum_running_requests": engine_shape,
                            "batch_fit_without_preemption": True,
                        },
                    }
                )
            waves.append(
                {
                    "wave_index": wave_index,
                    "start": start,
                    "stop": stop,
                    "request_count": global_shape,
                    "engines": engines,
                    "inactive_engines": [
                        {
                            "dp_rank": dp_rank,
                            "request_count": 0,
                            "inactive": True,
                            "phase": wave_phase,
                            "cudagraph_observations": [],
                            "scheduler_observations": [],
                        }
                        for dp_rank in range(len(engine_shapes), 2)
                    ],
                }
            )
        start = stop
    phase = {
        "phase": phase_name,
        "sample": {
            "request_count": schedule.request_count,
            "generated_tokens": schedule.request_count * 500,
            "output_lengths": [500] * schedule.request_count,
        },
        "cudagraph_observations_retained": observations,
    }
    phase["waves" if schedule.topology == "dp2" else "wave_proofs"] = waves
    if execution_coordinates is not None:
        phase["request_executions"] = list(execution_coordinates)
    return phase


@pytest.mark.parametrize("topology", ("tp2", "dp2"))
def test_homogeneous_identity_phase_evidence_requires_actual_physical_shapes(topology) -> None:
    schedule = build_homogeneous_identity_schedule(
        topology=topology,
        request_count=1_000,
        global_wave_size=96,
    )
    phase = _physical_phase(schedule)

    evidence = validate_homogeneous_identity_phase_evidence(phase, schedule=schedule)

    assert evidence["passed"] is True
    assert evidence["global_request_shapes"] == [96] * 10 + [40]
    expected_engines = [[96]] * 10 + [[40]] if topology == "tp2" else [[48, 48]] * 10 + [[20, 20]]
    assert evidence["engine_request_shapes"] == expected_engines


@pytest.mark.parametrize(
    ("topology", "expected_engine_shapes", "wrong_decode_dimension"),
    (("tp2", [[96]], 95), ("dp2", [[48, 48]], 47)),
)
def test_mixed_identity_phase_evidence_requires_exact_singleton_decode_dimension(
    topology,
    expected_engine_shapes,
    wrong_decode_dimension,
) -> None:
    attempt = build_mixed_identity_admission_bundle(topology=topology, base_seed=42)["attempts"][1]
    schedule = build_mixed_identity_schedule(
        topology=topology,
        request_count=96,
        global_wave_size=96,
        request_id_prefix="mixed-b96",
    )
    phase = _physical_phase(
        schedule,
        phase_name="mixed-b96",
        execution_coordinates=attempt["execution_coordinates"],
    )

    evidence = validate_mixed_identity_phase_evidence(
        phase,
        schedule=schedule,
        expected_execution_coordinates=attempt["execution_coordinates"],
    )

    if evidence["global_request_shapes"] != [96] or evidence["engine_request_shapes"] != expected_engine_shapes:
        raise AssertionError("mixed physical evidence did not prove its exact B96 engine dimensions")
    if evidence["mixed_case_batching"] is not True or evidence["passed"] is not True:
        raise AssertionError("mixed physical evidence was not admitted as real mixed batching")

    if topology == "tp2":
        observations = phase["cudagraph_observations_retained"]
    else:
        observations = phase["waves"][0]["engines"][0]["cudagraph_observations"]
    removed = observations.pop()
    with pytest.raises(AssertionError, match="exactly 499"):
        validate_mixed_identity_phase_evidence(
            phase,
            schedule=schedule,
            expected_execution_coordinates=attempt["execution_coordinates"],
        )
    observations.append(removed)

    dimensions = observations[0]["request_dimensions"]
    dimensions["decode_req_count"] = wrong_decode_dimension
    dimensions["token_count"] = wrong_decode_dimension
    with pytest.raises(AssertionError, match="singleton decode dimension"):
        validate_mixed_identity_phase_evidence(
            phase,
            schedule=schedule,
            expected_execution_coordinates=attempt["execution_coordinates"],
        )


def test_mixed_identity_dp2_rejects_swapped_rank_case_ownership() -> None:
    attempt = build_mixed_identity_admission_bundle(topology="dp2", base_seed=42)["attempts"][1]
    schedule = build_mixed_identity_schedule(
        topology="dp2",
        request_count=96,
        global_wave_size=96,
        request_id_prefix="mixed-b96",
    )
    phase = _physical_phase(
        schedule,
        phase_name="mixed-b96",
        execution_coordinates=attempt["execution_coordinates"],
    )
    phase["request_executions"][0], phase["request_executions"][48] = (
        phase["request_executions"][48],
        phase["request_executions"][0],
    )

    with pytest.raises(AssertionError, match="caller admission"):
        validate_mixed_identity_phase_evidence(
            phase,
            schedule=schedule,
            expected_execution_coordinates=attempt["execution_coordinates"],
        )


def test_homogeneous_identity_phase_evidence_rejects_subdivided_capture_shape() -> None:
    schedule = build_homogeneous_identity_schedule(
        topology="tp2",
        request_count=96,
        global_wave_size=96,
    )
    phase = _physical_phase(schedule)
    phase["cudagraph_observations_retained"][0]["num_unpadded_tokens"] = 24
    phase["cudagraph_observations_retained"][0]["num_padded_tokens"] = 24

    with pytest.raises(AssertionError, match="physical.*shape"):
        validate_homogeneous_identity_phase_evidence(phase, schedule=schedule)


def test_dp2_serial_identity_phase_requires_unused_replica_to_remain_inactive() -> None:
    schedule = build_homogeneous_identity_schedule(
        topology="dp2",
        request_count=1,
        global_wave_size=96,
    )
    phase = _physical_phase(schedule)

    evidence = validate_homogeneous_identity_phase_evidence(phase, schedule=schedule)

    assert evidence["engine_request_shapes"] == [[1]]
    inactive = phase["waves"][0]["inactive_engines"][0]
    inactive["scheduler_observations"] = [{"phase": "identity.wave-000"}]
    with pytest.raises(AssertionError, match="inactive"):
        validate_homogeneous_identity_phase_evidence(phase, schedule=schedule)


def _identity_records(manifest, target):
    return tuple(
        GenerationRecord(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=tuple(target.encode("ascii")),
            output_logprobs=(-0.1,) * 500,
            requested_max_tokens=500,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        )
        for request in manifest.requests
    )


def test_canonical_identity_artifact_retains_raw_bytes_and_checks_every_request(tmp_path) -> None:
    case = load_canonical_7b_identity_cases(PROMPTS_CSV)[0]
    manifest = build_canonical_identity_manifest(
        _base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=3,
        request_id_prefix="identity-case0",
    )
    records = _identity_records(manifest, case.target)
    executions = build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    artifact = write_full_generation_records_artifact(
        tmp_path / "identity.outputs.jsonl.gz",
        records=records,
        execution_records=executions,
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )

    evidence = validate_canonical_identity_output_artifact(
        artifact,
        case=case,
        expected_request_ids=tuple(request.request_id for request in manifest.requests),
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )

    assert artifact["decoded_output_bytes_retained"] is True
    assert artifact["decoded_output_byte_count"] == 1_500
    assert evidence["passed"] is True
    assert evidence["request_count"] == 3
    assert evidence["minimum_observed_identity_percent"] == 100.0
    assert evidence["raw_output_bytes_retained"] is True
    assert len(evidence["requests"]) == 3


def test_canonical_identity_artifact_rejects_one_low_scoring_request(tmp_path) -> None:
    case = load_canonical_7b_identity_cases(PROMPTS_CSV)[1]
    manifest = build_canonical_identity_manifest(
        _base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=2,
        request_id_prefix="identity-case1",
    )
    records = list(_identity_records(manifest, case.target))
    records[1] = replace(records[1], output_token_ids=(ord("N"),) * 500)
    executions = build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    artifact = write_full_generation_records_artifact(
        tmp_path / "identity-fail.outputs.jsonl.gz",
        records=records,
        execution_records=executions,
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )

    with pytest.raises(AssertionError, match="request.*identity"):
        validate_canonical_identity_output_artifact(
            artifact,
            case=case,
            expected_request_ids=tuple(request.request_id for request in manifest.requests),
            decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
        )


def test_mixed_canonical_identity_artifact_scores_every_case_occurrence(tmp_path) -> None:
    cases = load_canonical_7b_identity_cases(PROMPTS_CSV)
    manifest = build_mixed_canonical_identity_manifest(
        _base_manifest(),
        cases=cases,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=4,
        request_id_prefix="mixed-b4",
    )
    expected_cases = cases
    records = tuple(
        GenerationRecord(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=tuple(case.target.encode("ascii")),
            output_logprobs=(-0.1,) * 500,
            requested_max_tokens=500,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        )
        for request, case in zip(manifest.requests, expected_cases, strict=True)
    )
    executions = build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    artifact = write_full_generation_records_artifact(
        tmp_path / "mixed.outputs.jsonl.gz",
        records=records,
        execution_records=executions,
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )

    evidence = validate_mixed_canonical_identity_output_artifact(
        artifact,
        cases_by_request=expected_cases,
        expected_request_ids=tuple(request.request_id for request in manifest.requests),
        expected_prompt_token_ids=tuple(request.prompt_token_ids for request in manifest.requests),
        expected_execution_coordinates=tuple(record.to_dict() for record in executions),
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )

    if [request["case_index"] for request in evidence["requests"]] != [0, 1, 2, 3]:
        raise AssertionError("mixed output evidence lost per-occurrence case ownership")
    if evidence["case_request_counts"] != [1, 1, 1, 1] or evidence["minimum_observed_matches"] != 500:
        raise AssertionError("mixed output evidence did not score every retained occurrence")


def test_mixed_canonical_identity_artifact_rejects_one_low_occurrence(tmp_path) -> None:
    cases = load_canonical_7b_identity_cases(PROMPTS_CSV)
    manifest = build_mixed_canonical_identity_manifest(
        _base_manifest(),
        cases=cases,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=4,
        request_id_prefix="mixed-b4",
    )
    records = list(
        GenerationRecord(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=tuple(cases[index].target.encode("ascii")),
            output_logprobs=(-0.1,) * 500,
            requested_max_tokens=500,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        )
        for index, request in enumerate(manifest.requests)
    )
    records[2] = replace(records[2], output_token_ids=(ord("A"),) * 500)
    executions = build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    artifact = write_full_generation_records_artifact(
        tmp_path / "mixed-low.outputs.jsonl.gz",
        records=tuple(records),
        execution_records=executions,
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )

    with pytest.raises(AssertionError, match="case 2.*identity"):
        validate_mixed_canonical_identity_output_artifact(
            artifact,
            cases_by_request=cases,
            expected_request_ids=tuple(request.request_id for request in manifest.requests),
            expected_prompt_token_ids=tuple(request.prompt_token_ids for request in manifest.requests),
            expected_execution_coordinates=tuple(record.to_dict() for record in executions),
            decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
        )


@pytest.mark.parametrize("mutation", ("prompt", "call-and-seed", "global-range", "stage-request-id"))
def test_mixed_canonical_output_rejects_caller_admission_drift(tmp_path, mutation) -> None:
    cases = load_canonical_7b_identity_cases(PROMPTS_CSV)
    manifest = build_mixed_canonical_identity_manifest(
        _base_manifest(),
        cases=cases,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=96,
        request_id_prefix="mixed-b96",
    )
    attempt = build_mixed_identity_admission_bundle(topology="tp2", base_seed=42)["attempts"][1]
    expected_coordinates = tuple(attempt["execution_coordinates"])
    executions = [RequestExecutionRecord(**coordinate) for coordinate in expected_coordinates]
    records = [
        GenerationRecord(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=tuple(cases[index % 4].target.encode("ascii")),
            output_logprobs=(-0.1,) * 500,
            requested_max_tokens=500,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        )
        for index, request in enumerate(manifest.requests)
    ]
    if mutation == "prompt":
        records[0] = replace(records[0], prompt_token_ids=(ord("A"),))
    elif mutation == "call-and-seed":
        executions[0] = replace(executions[0], call_index=0, seed=42)
    elif mutation == "global-range":
        executions[0] = replace(executions[0], global_request_index=0)
    else:
        b4_request_id = "mixed-b4-case0-occurrence0000"
        records[0] = replace(records[0], request_id=b4_request_id)
        executions[0] = replace(executions[0], request_id=b4_request_id)
    artifact = write_full_generation_records_artifact(
        tmp_path / f"mixed-{mutation}.outputs.jsonl.gz",
        records=tuple(records),
        execution_records=tuple(executions),
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )

    with pytest.raises(AssertionError):
        validate_mixed_canonical_identity_output_artifact(
            artifact,
            cases_by_request=tuple(cases[index % 4] for index in range(96)),
            expected_request_ids=attempt["request_ids"],
            expected_prompt_token_ids=tuple(request.prompt_token_ids for request in manifest.requests),
            expected_execution_coordinates=expected_coordinates,
            decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
        )


def test_canonical_identity_artifact_accepts_n_but_rejects_non_dna_or_token_byte_alias(tmp_path) -> None:
    case = load_canonical_7b_identity_cases(PROMPTS_CSV)[0]
    manifest = build_canonical_identity_manifest(
        _base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=_tokenizer(),
        request_count=1,
        request_id_prefix="identity-case0",
    )
    executions = build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    output_with_n = "N" + case.target[1:]
    accepted_artifact = write_full_generation_records_artifact(
        tmp_path / "identity-allowed-n.outputs.jsonl.gz",
        records=_identity_records(manifest, output_with_n),
        execution_records=executions,
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )
    accepted = validate_canonical_identity_output_artifact(
        accepted_artifact,
        case=case,
        expected_request_ids=(manifest.requests[0].request_id,),
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )
    if accepted["minimum_observed_identity_percent"] != 99.8:
        raise AssertionError("one valid N must be retained and scored against the unchanged 500-base target")

    invalid_record = replace(
        _identity_records(manifest, case.target)[0],
        output_token_ids=(ord("X"),) * 500,
    )
    with pytest.raises(AssertionError, match="A/C/G/N/T"):
        write_full_generation_records_artifact(
            tmp_path / "identity-invalid-base.outputs.jsonl.gz",
            records=(invalid_record,),
            execution_records=executions,
            decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
        )

    aliased_record = replace(_identity_records(manifest, case.target)[0], output_token_ids=(ord("A"),) * 500)
    with pytest.raises(AssertionError, match="token IDs"):
        write_full_generation_records_artifact(
            tmp_path / "identity-token-alias.outputs.jsonl.gz",
            records=(aliased_record,),
            execution_records=executions,
            decode_output_token_ids=lambda token_ids: "C" * len(token_ids),
        )
