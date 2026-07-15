# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from pathlib import Path

import pytest
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from bionemo.evo2.vllm.benchmark import WorkloadManifest
from bionemo.evo2.vllm.nemo_runner import (
    build_nemo_generation_config,
    build_nemo_generation_input,
    records_from_nemo_generation_output,
    run_nemo_generation_phase,
)
from bionemo.evo2.vllm.profile import Evo2VllmProfile
from bionemo.evo2.vllm.runner import PeakMemoryMonitor, request_seed


DATA = Path(__file__).with_name("data") / "gdpo_mixed_96.json"


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
    assert config["vllm_kwargs"]["worker_extension_cls"].endswith("Evo2NemoRlVllmWorkerExtension")


def test_nemo_generation_input_right_pads_mixed_prompts_without_semantic_padding() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 3).with_max_new_tokens(3)

    batch = build_nemo_generation_input(manifest)

    assert batch["input_ids"].shape == (3, 6)
    assert batch["input_lengths"].tolist() == [4, 5, 6]
    for row, request in zip(batch["input_ids"], manifest.requests, strict=True):
        length = len(request.prompt_token_ids)
        assert row[:length].tolist() == list(request.prompt_token_ids)
        assert row[length:].tolist() == [0] * (6 - length)


def test_nemo_output_adapter_retains_full_tokens_logprobs_seeds_and_timings() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    inputs = build_nemo_generation_input(manifest)
    output_ids = torch.zeros((2, inputs["input_ids"].shape[1] + 3), dtype=torch.long)
    logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
    generated = ((65, 67, 71), (66, 68, 72))
    for index, request in enumerate(manifest.requests):
        prompt_length = len(request.prompt_token_ids)
        output_ids[index, :prompt_length] = torch.tensor(request.prompt_token_ids)
        output_ids[index, prompt_length : prompt_length + 3] = torch.tensor(generated[index])
        logprobs[index, prompt_length : prompt_length + 3] = torch.tensor([-0.1, -0.2, -0.3])
    outputs = BatchedDataDict(
        {
            "output_ids": output_ids,
            "logprobs": logprobs,
            "generation_lengths": torch.tensor([3, 3]),
            "unpadded_sequence_lengths": torch.tensor(
                [len(request.prompt_token_ids) + 3 for request in manifest.requests]
            ),
            "truncated": torch.tensor([True, True]),
            "generation_request_seeds": torch.tensor([101, 202]),
            "generation_global_request_indices": torch.tensor([48, 49]),
            "generation_rounds": torch.tensor([3, 3]),
            "generation_call_indices": torch.tensor([7, 7]),
            "generation_dp_ranks": torch.tensor([1, 1]),
            "generation_first_token_latency_s": torch.tensor([0.4, 0.5]),
            "generation_decode_s": torch.tensor([0.2, 0.3]),
        }
    )

    records, executions, timings = records_from_nemo_generation_output(manifest, outputs)

    assert records[0].output_token_ids == (65, 67, 71)
    assert records[0].output_logprobs == pytest.approx((-0.1, -0.2, -0.3))
    assert executions[0].to_dict() == {
        "request_id": "gdpo-000",
        "global_request_index": 48,
        "generation_round": 3,
        "dp_rank": 1,
        "call_index": 7,
        "seed": 101,
    }
    assert timings["ttft_s"] == pytest.approx((0.4, 0.5))
    assert timings["decode_s"] == pytest.approx((0.2, 0.3))


def test_production_nemo_phase_proves_exact_dp_ownership_and_persists_full_outputs(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 4).with_max_new_tokens(3)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=15,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        proof=True,
    )

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
                    "worker_proof": [
                        {
                            "rank": 0,
                            "fir_routes": {"direct": {"calls": 9, "requests": shard_size, "tokens": 18}},
                        }
                    ],
                }
                for shard_size in self.shard_sizes
            ]

    class FakeGeneration:
        def __init__(self) -> None:
            self.worker_group = FakeWorkerGroup()
            self.call_index = 7
            self.global_index = 48

        def generate(self, data):
            request_count = len(data["input_ids"])
            first_shard = (request_count + 1) // 2
            self.worker_group.shard_sizes = (first_shard, request_count - first_shard)
            output_ids = torch.zeros((request_count, data["input_ids"].shape[1] + 3), dtype=torch.long)
            logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
            for row_index, prompt_length in enumerate(data["input_lengths"].tolist()):
                output_ids[row_index, :prompt_length] = data["input_ids"][row_index, :prompt_length]
                output_ids[row_index, prompt_length : prompt_length + 3] = torch.tensor([65, 67, 71])
                logprobs[row_index, prompt_length : prompt_length + 3] = torch.tensor([-0.1, -0.2, -0.3])
            global_indices = torch.arange(self.global_index, self.global_index + request_count)
            seeds = torch.tensor(
                [
                    request_seed(42, generation_round=self.call_index, global_request_index=int(index))
                    for index in global_indices
                ]
            )
            outputs = BatchedDataDict(
                {
                    "output_ids": output_ids,
                    "logprobs": logprobs,
                    "generation_lengths": torch.full((request_count,), 3),
                    "unpadded_sequence_lengths": data["input_lengths"] + 3,
                    "truncated": torch.ones(request_count, dtype=torch.bool),
                    "generation_request_seeds": seeds,
                    "generation_global_request_indices": global_indices,
                    "generation_rounds": torch.full((request_count,), self.call_index),
                    "generation_call_indices": torch.full((request_count,), self.call_index),
                    "generation_dp_ranks": torch.tensor([0] * first_shard + [1] * (request_count - first_shard)),
                    "generation_first_token_latency_s": torch.full((request_count,), 0.4),
                    "generation_decode_s": torch.full((request_count,), 0.2),
                }
            )
            self.call_index += 1
            self.global_index += request_count
            return outputs

    times = iter((10.0, 12.5))
    result = run_nemo_generation_phase(
        generation=FakeGeneration(),
        manifest=manifest,
        profile=profile,
        phase="steady-0",
        sample_index=0,
        full_output_path=tmp_path / "nemo.outputs.jsonl.gz",
        memory_monitor_factory=lambda: PeakMemoryMonitor(lambda: (1_000, 2_000)),
        ray_get=lambda futures: futures,
        clock=lambda: next(times),
    )

    assert result.sample.generation_s == 2.5
    assert result.sample.generated_tokens == 12
    assert [record.global_request_index for record in result.request_executions] == [48, 49, 50, 51]
    assert [record.dp_rank for record in result.request_executions] == [0, 0, 1, 1]
    assert len(result.wave_proofs) == 1
    assert [engine["full_decode_proof"]["passed"] for engine in result.wave_proofs[0]["engines"]] == [
        True,
        True,
    ]
    assert result.full_output_artifact["generated_token_count"] == 12
