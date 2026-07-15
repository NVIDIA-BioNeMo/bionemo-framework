# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from types import SimpleNamespace

import pytest

from bionemo.evo2.vllm.benchmark import WorkloadManifest
from bionemo.evo2.vllm.runner import (
    CUDAGraphProofRecorder,
    PeakMemoryMonitor,
    build_request_sampling_params,
    request_seed,
    run_generation_phase,
    validate_full_decode_proof,
)


DATA = (
    __import__("pathlib").Path(__file__).with_name("data")
    / "gdpo_mixed_96.json"
)


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
            "runtime_mode": "CUDAGraphMode.PIECEWISE",
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
    next_round = [
        request_seed(42, generation_round=1, global_request_index=index) for index in range(96)
    ]

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


def _fake_outputs(manifest: WorkloadManifest):
    outputs = []
    for index, request in enumerate(manifest.requests):
        token_ids = (65 + index, 67 + index, 71 + index)
        completion = SimpleNamespace(
            token_ids=token_ids,
            logprobs=[
                {token_id: SimpleNamespace(logprob=-0.1)}
                for token_id in token_ids
            ],
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
            assert prompts == [
                {"prompt_token_ids": list(request.prompt_token_ids)}
                for request in manifest.requests
            ]
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
        clock=lambda: next(times),
    )

    assert result.sample.generation_s == 2.5
    assert result.sample.request_count == 2
    assert result.sample.generated_tokens == 6
    assert result.sample.peak_device_memory_bytes == (1_000, 2_000)
    assert len(result.observations) == 2
    assert [summary["output_length"] for summary in result.output_summaries] == [3, 3]
