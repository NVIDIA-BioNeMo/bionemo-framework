# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from types import SimpleNamespace

import pytest

from bionemo.evo2.vllm.benchmark import WorkloadManifest
from bionemo.evo2.vllm.runner import (
    CUDAGraphProofRecorder,
    PeakMemoryMonitor,
    build_request_sampling_params,
    prepare_workload,
    request_seed,
    run_generation_phase,
    validate_full_decode_proof,
)


DATA = __import__("pathlib").Path(__file__).with_name("data") / "gdpo_mixed_96.json"


def _scheduler_stats(
    *,
    unpadded: int,
    padded: int,
    mode: str,
):
    return SimpleNamespace(
        cudagraph_stats=SimpleNamespace(
            num_unpadded_tokens=unpadded,
            num_padded_tokens=padded,
            num_paddings=padded - unpadded,
            runtime_mode=mode,
        )
    )


def test_cudagraph_recorder_persists_phase_and_runtime_mode_without_log_resets() -> None:
    recorder = CUDAGraphProofRecorder()
    recorder.start_phase("steady-0")
    recorder.record(
        _scheduler_stats(unpadded=768, padded=768, mode="CUDAGraphMode.PIECEWISE"),
        None,
        engine_idx=0,
    )
    recorder.record(
        _scheduler_stats(unpadded=96, padded=96, mode="CUDAGraphMode.FULL"),
        None,
        engine_idx=0,
    )
    recorder.record(None, None, engine_idx=0)

    assert recorder.observations == [
        {
            "phase": "steady-0",
            "engine_index": 0,
            "num_unpadded_tokens": 768,
            "num_padded_tokens": 768,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.PIECEWISE",
        },
        {
            "phase": "steady-0",
            "engine_index": 0,
            "num_unpadded_tokens": 96,
            "num_padded_tokens": 96,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        },
    ]


def test_full_decode_proof_requires_full_unpadded_replay_and_rejects_fallback() -> None:
    observations = [
        {
            "phase": "steady-0",
            "engine_index": 0,
            "num_unpadded_tokens": 768,
            "num_padded_tokens": 768,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.NONE",
        },
        *[
            {
                "phase": "steady-0",
                "engine_index": 0,
                "num_unpadded_tokens": 96,
                "num_padded_tokens": 96,
                "num_paddings": 0,
                "runtime_mode": "CUDAGraphMode.FULL",
            }
            for _ in range(2)
        ],
    ]

    validate_full_decode_proof(
        observations,
        phase="steady-0",
        batch_size=96,
        max_new_tokens=3,
    )

    with pytest.raises(AssertionError, match="NONE"):
        validate_full_decode_proof(
            [
                {
                    **observations[-1],
                    "runtime_mode": "CUDAGraphMode.NONE",
                }
            ],
            phase="steady-0",
            batch_size=96,
            max_new_tokens=2,
        )
    with pytest.raises(AssertionError, match="FULL"):
        validate_full_decode_proof(
            observations[:1],
            phase="steady-0",
            batch_size=96,
            max_new_tokens=3,
        )
    with pytest.raises(AssertionError, match="padding"):
        validate_full_decode_proof(
            [
                {
                    **observations[-1],
                    "num_padded_tokens": 128,
                    "num_paddings": 32,
                }
            ],
            phase="steady-0",
            batch_size=96,
            max_new_tokens=2,
        )


def test_full_decode_proof_allows_staggered_admission_with_full_global_batch() -> None:
    observations = [
        {
            "phase": "cold-generation",
            "engine_index": 0,
            "num_unpadded_tokens": 4,
            "num_padded_tokens": 4,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.PIECEWISE",
        },
        {
            "phase": "cold-generation",
            "engine_index": 0,
            "num_unpadded_tokens": 6,
            "num_padded_tokens": 8,
            "num_paddings": 2,
            "runtime_mode": "CUDAGraphMode.PIECEWISE",
        },
        {
            "phase": "cold-generation",
            "engine_index": 0,
            "num_unpadded_tokens": 2,
            "num_padded_tokens": 2,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        },
        {
            "phase": "cold-generation",
            "engine_index": 0,
            "num_unpadded_tokens": 1,
            "num_padded_tokens": 1,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        },
    ]

    validate_full_decode_proof(
        observations,
        phase="cold-generation",
        batch_size=2,
        max_new_tokens=3,
    )

    steady_observations = [{**item, "phase": "steady-0"} for item in observations]
    validate_full_decode_proof(
        steady_observations,
        phase="steady-0",
        batch_size=2,
        max_new_tokens=3,
    )

    with pytest.raises(AssertionError, match="global batch"):
        validate_full_decode_proof(
            [item for item in steady_observations if item["num_unpadded_tokens"] != 2],
            phase="steady-0",
            batch_size=2,
            max_new_tokens=3,
        )


def test_long_full_decode_proof_rejects_missing_work_and_serialization() -> None:
    def full_observation(unpadded: int) -> dict[str, object]:
        return {
            "phase": "steady-0",
            "engine_index": 0,
            "num_unpadded_tokens": unpadded,
            "num_padded_tokens": unpadded,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        }

    batched = [full_observation(10) for _ in range(98)]
    validate_full_decode_proof(
        batched,
        phase="steady-0",
        batch_size=10,
        max_new_tokens=100,
    )

    with pytest.raises(AssertionError, match="coverage"):
        validate_full_decode_proof(
            batched[:40],
            phase="steady-0",
            batch_size=10,
            max_new_tokens=100,
        )

    with pytest.raises(AssertionError, match="occupancy"):
        validate_full_decode_proof(
            [full_observation(10), *[full_observation(1) for _ in range(980)]],
            phase="steady-0",
            batch_size=10,
            max_new_tokens=100,
        )


def test_peak_memory_monitor_records_each_device_maximum() -> None:
    samples = iter(((100, 200), (150, 180), (125, 240)))
    monitor = PeakMemoryMonitor(lambda: next(samples))

    monitor.sample_now()
    monitor.sample_now()
    monitor.sample_now()

    assert monitor.peak_device_memory_bytes == (150, 240)


def test_peak_memory_monitor_rejects_device_count_changes() -> None:
    samples = iter(((100, 200), (150,)))
    monitor = PeakMemoryMonitor(lambda: next(samples))
    monitor.sample_now()

    with pytest.raises(RuntimeError, match="device count"):
        monitor.sample_now()


def test_request_seeds_match_between_tp2_and_dp2_and_advance_by_round() -> None:
    tp2 = [request_seed(42, generation_round=0, global_request_index=index) for index in range(96)]
    dp2 = [
        *[request_seed(42, generation_round=0, global_request_index=index) for index in range(48)],
        *[request_seed(42, generation_round=0, global_request_index=index) for index in range(48, 96)],
    ]
    next_round = [request_seed(42, generation_round=1, global_request_index=index) for index in range(96)]

    assert tp2 == dp2
    assert len(set(tp2)) == 96
    assert set(tp2).isdisjoint(next_round)
    assert tp2[0] == 42


def test_request_sampling_params_apply_global_seed_offsets() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)

    params = build_request_sampling_params(
        manifest,
        sampling_params_factory=SimpleNamespace,
        generation_round=2,
        global_request_offset=48,
    )

    assert [param.seed for param in params] == [
        request_seed(42, generation_round=2, global_request_index=48),
        request_seed(42, generation_round=2, global_request_index=49),
    ]
    assert all(param.max_tokens == 3 and param.min_tokens == 3 for param in params)
    assert all(param.detokenize is False and param.logprobs == 0 for param in params)


def test_prepare_workload_builds_exact_pressure_shape_without_mutating_manifest() -> None:
    manifest = WorkloadManifest.from_path(DATA)

    pressure = prepare_workload(
        manifest,
        request_count=3,
        uniform_prompt_length=25_000,
        request_id_prefix="pressure",
        max_new_tokens=25_000,
    )

    assert len(manifest.requests) == 96
    assert manifest.max_new_tokens == 5_989
    assert [len(request.prompt_token_ids) for request in pressure.requests] == [25_000] * 3
    assert pressure.max_new_tokens == 25_000
    assert [request.request_id for request in pressure.requests] == [
        "pressure-0000",
        "pressure-0001",
        "pressure-0002",
    ]


def _fake_outputs(manifest: WorkloadManifest):
    outputs = []
    for index, request in enumerate(manifest.requests):
        token_ids = (65 + index, 67 + index, 71 + index)
        completion = SimpleNamespace(
            token_ids=token_ids,
            logprobs=[{token_id: SimpleNamespace(logprob=-0.1)} for token_id in token_ids],
        )
        outputs.append(
            SimpleNamespace(
                prompt_token_ids=list(request.prompt_token_ids),
                outputs=[completion],
                finished=True,
                metrics=SimpleNamespace(
                    first_token_latency=0.2,
                    first_token_ts=10.0,
                    last_token_ts=10.2,
                    num_generation_tokens=3,
                ),
            )
        )
    return outputs


def test_generation_phase_times_one_complete_batch_and_preserves_exact_outputs() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    recorder = CUDAGraphProofRecorder()

    class FakeLLM:
        def generate(self, prompts, sampling_params, *, use_tqdm):
            assert prompts == [{"prompt_token_ids": list(request.prompt_token_ids)} for request in manifest.requests]
            assert len(sampling_params) == 2
            assert use_tqdm is False
            recorder.record(
                _scheduler_stats(unpadded=2, padded=2, mode="CUDAGraphMode.FULL"),
                None,
            )
            recorder.record(
                _scheduler_stats(unpadded=2, padded=2, mode="CUDAGraphMode.FULL"),
                None,
            )
            return _fake_outputs(manifest)

    times = iter((10.0, 12.5))
    proof_events = []
    result = run_generation_phase(
        llm=FakeLLM(),
        manifest=manifest,
        sampling_params=build_request_sampling_params(
            manifest,
            sampling_params_factory=SimpleNamespace,
            generation_round=0,
            global_request_offset=0,
        ),
        phase="steady-0",
        sample_index=0,
        recorder=recorder,
        memory_monitor_factory=lambda: PeakMemoryMonitor(lambda: (1_000, 2_000)),
        reset_worker_proof=lambda: proof_events.append("reset"),
        snapshot_worker_proof=lambda: (
            {
                "rank": 0,
                "fir_routes": {"equal_length_conv": {"calls": 9, "requests": 2, "tokens": 10}},
            },
        ),
        clock=lambda: next(times),
    )

    assert result.sample.generation_s == 2.5
    assert result.sample.request_count == 2
    assert result.sample.generated_tokens == 6
    assert result.sample.peak_device_memory_bytes == (1_000, 2_000)
    assert len(result.observations) == 2
    assert [summary["output_length"] for summary in result.output_summaries] == [3, 3]
    assert proof_events == ["reset"]
    assert result.worker_proof[0]["rank"] == 0
    assert result.to_dict()["worker_proof"][0]["fir_routes"]["equal_length_conv"]["calls"] == 9
