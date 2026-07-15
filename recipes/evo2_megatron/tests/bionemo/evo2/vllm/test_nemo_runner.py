# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import hashlib
import json
import re
from pathlib import Path

import pytest
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from bionemo.evo2.vllm.benchmark import GenerationRecord, WorkloadManifest
from bionemo.evo2.vllm.nemo_runner import (
    build_nemo_generation_config,
    build_nemo_generation_input,
    full_vocab_logprob_evidence_from_nemo_output,
    records_from_nemo_generation_output,
    run_nemo_generation_phase,
    snapshot_and_validate_nemo_resolved_configs,
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
        "execution_uid": "round=3/call=7/global=48/dp=1/request=gdpo-000",
        "request_id": "gdpo-000",
        "global_request_index": 48,
        "generation_round": 3,
        "dp_rank": 1,
        "call_index": 7,
        "seed": 101,
    }
    assert records[0].requested_max_tokens == 3
    assert records[0].finish_reason == "length"
    assert records[0].stop_reason is None
    assert records[0].stopped_on_eos is False
    assert timings["ttft_s"] == pytest.approx((0.4, 0.5))
    assert timings["decode_s"] == pytest.approx((0.2, 0.3))


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
            "generation_request_seeds": torch.tensor([101]),
            "generation_global_request_indices": torch.tensor([48]),
            "generation_rounds": torch.tensor([3]),
            "generation_call_indices": torch.tensor([7]),
            "generation_dp_ranks": torch.tensor([0]),
            "generation_first_token_latency_s": torch.tensor([0.4]),
            "generation_decode_s": torch.tensor([0.2]),
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
            self.call_index = 7
            self.global_index = 48
            self.cache_invalidations = 0

        def invalidate_kv_cache(self):
            self.cache_invalidations += 1
            return True

        def generate(self, data, greedy=False):
            assert greedy is True
            request_count = len(data["input_ids"])
            first_shard = (request_count + 1) // 2
            self.worker_group.shard_sizes = (first_shard, request_count - first_shard)
            output_ids = torch.zeros((request_count, data["input_ids"].shape[1] + 3), dtype=torch.long)
            logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
            for row_index, prompt_length in enumerate(data["input_lengths"].tolist()):
                output_ids[row_index, :prompt_length] = data["input_ids"][row_index, :prompt_length]
                output_ids[row_index, prompt_length : prompt_length + 3] = torch.tensor([65, 67, 71])
                logprobs[row_index, prompt_length : prompt_length + 3] = torch.tensor([-0.1, -0.2, -0.3])
            dense = torch.full((request_count, 3, 512), -20.0)
            for row_index in range(request_count):
                dense[row_index, torch.arange(3), torch.tensor([65, 67, 71])] = torch.tensor([-0.1, -0.2, -0.3])
            global_indices = torch.arange(self.global_index, self.global_index + request_count)
            seeds = torch.tensor(
                [
                    request_seed(
                        42,
                        call_index=self.call_index,
                        dp_rank=dp_rank,
                        dp_size=2,
                        request_index_in_stream=local_index,
                    )
                    for dp_rank, shard_size in enumerate(self.worker_group.shard_sizes)
                    for local_index in range(shard_size)
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
                    "generation_num_cached_tokens": torch.tensor(
                        [0, 16, 0, 16] if self.call_index == 7 else [16, 16, 16, 16]
                    ),
                    "generation_vocab_logprobs": dense,
                    "generation_logprob_counts": torch.full((request_count, 3), 512),
                }
            )
            self.call_index += 1
            self.global_index += request_count
            return outputs

    times = iter((10.0, 11.0, 20.0, 21.0))
    generation = FakeGeneration()
    result = run_nemo_generation_phase(
        generation=generation,
        manifest=manifest,
        profile=profile,
        phase="steady-0",
        sample_index=0,
        full_output_path=tmp_path / "nemo.outputs.jsonl.gz",
        memory_monitor_factory=lambda: PeakMemoryMonitor(lambda: (1_000, 2_000)),
        ray_get=lambda futures: futures,
        clock=lambda: next(times),
        greedy=True,
        require_full_vocab_logprobs=True,
    )

    assert result.sample.generation_s == 2.0
    assert result.sample.generated_tokens == 24
    assert [record.global_request_index for record in result.request_executions] == list(range(48, 56))
    assert [record.dp_rank for record in result.request_executions] == [0, 0, 1, 1] * 2
    assert [record.generation_round for record in result.request_executions] == [7] * 4 + [8] * 4
    assert [record.call_index for record in result.request_executions] == [7] * 4 + [8] * 4
    assert [record.seed for record in result.request_executions] == [
        14_000_084,
        14_000_085,
        15_000_087,
        15_000_088,
        16_000_090,
        16_000_091,
        17_000_093,
        17_000_094,
    ]
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


def test_nemo_speed_phase_skips_proof_rpcs_and_memory_polling(tmp_path) -> None:
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

    class ForbiddenWorkerGroup:
        def run_all_workers_single_data(self, *args, **kwargs):
            pytest.fail(f"speed lane issued a proof RPC: {args!r} {kwargs!r}")

    class FakeGeneration:
        worker_group = ForbiddenWorkerGroup()

        def generate(self, data, greedy=False):
            assert greedy is False
            request_count = len(data["input_ids"])
            prompt_width = data["input_ids"].shape[1]
            output_ids = torch.zeros((request_count, prompt_width + 1), dtype=torch.long)
            logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
            for row_index, prompt_length in enumerate(data["input_lengths"].tolist()):
                output_ids[row_index, :prompt_length] = data["input_ids"][row_index, :prompt_length]
                output_ids[row_index, prompt_length] = 65
                logprobs[row_index, prompt_length] = -0.1
            return BatchedDataDict(
                {
                    "output_ids": output_ids,
                    "logprobs": logprobs,
                    "generation_lengths": torch.ones(request_count, dtype=torch.long),
                    "unpadded_sequence_lengths": data["input_lengths"] + 1,
                    "truncated": torch.ones(request_count, dtype=torch.bool),
                    "generation_request_seeds": torch.tensor([42, 43, 1_000_045, 1_000_046]),
                    "generation_global_request_indices": torch.arange(request_count),
                    "generation_rounds": torch.zeros(request_count, dtype=torch.long),
                    "generation_call_indices": torch.zeros(request_count, dtype=torch.long),
                    "generation_dp_ranks": torch.tensor([0, 0, 1, 1]),
                    "generation_first_token_latency_s": torch.full((request_count,), 0.4),
                    "generation_decode_s": torch.zeros(request_count),
                }
            )

    times = iter((10.0, 11.0))
    result = run_nemo_generation_phase(
        generation=FakeGeneration(),
        manifest=manifest,
        profile=profile,
        phase="steady-0",
        sample_index=0,
        full_output_path=tmp_path / "nemo-speed.outputs.jsonl.gz",
        memory_monitor_factory=lambda: pytest.fail("speed lane started peak-memory polling"),
        ray_get=lambda futures: pytest.fail(f"speed lane resolved proof futures: {futures!r}"),
        clock=lambda: next(times),
    )

    assert result.sample.generation_s == 1.0
    assert result.sample.peak_device_memory_bytes == ()
    assert result.proof_collected is False
    assert result.wave_proofs[0]["reset_proof"] is None
    assert result.wave_proofs[0]["engines"] == []
    assert result.full_output_artifact["generated_token_count"] == 4


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
