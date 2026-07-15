# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import json
from dataclasses import replace

import pytest
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    VLLM_EXECUTABLE,
)

from bionemo.evo2.vllm.profile import (
    Evo2VllmProfile,
    compilation_counter_snapshot,
    optimized_profile_sweep,
    resolved_config_snapshot,
    validate_resolved_profile,
)


def test_tp2_profile_pins_optimized_vllm_020_settings() -> None:
    profile = Evo2VllmProfile(
        topology="tp2",
        max_model_len=50_000,
        max_num_batched_tokens=32_768,
        gpu_memory_utilization=0.95,
        proof=True,
    )

    kwargs = profile.engine_kwargs(model="/checkpoint", seed=17)

    assert profile.global_batch_size == 96
    assert profile.replica_count == 1
    assert profile.per_engine_batch_size == 96
    assert kwargs["model"] == "/checkpoint"
    assert kwargs["tensor_parallel_size"] == 2
    assert kwargs["distributed_executor_backend"] == "ray"
    assert kwargs["worker_extension_cls"] == "bionemo.evo2.vllm.worker.Evo2VllmWorkerExtension"
    assert kwargs["max_num_seqs"] == 96
    assert kwargs["max_model_len"] == 50_000
    assert kwargs["max_num_batched_tokens"] == 32_768
    assert kwargs["gpu_memory_utilization"] == 0.95
    assert kwargs["optimization_level"] == 2
    assert kwargs["performance_mode"] == "balanced"
    assert kwargs["enforce_eager"] is False
    assert kwargs["enable_chunked_prefill"] is True
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["mamba_cache_mode"] == "none"
    assert kwargs["mamba_cache_dtype"] == "float32"
    assert kwargs["mamba_ssm_cache_dtype"] == "float32"
    assert "mamba_block_size" not in kwargs
    assert kwargs["async_scheduling"] is False
    assert kwargs["cudagraph_metrics"] is True
    assert kwargs["hf_overrides"]["max_position_embeddings"] == 50_000

    compilation = kwargs["compilation_config"]
    assert compilation["mode"] == 3
    assert compilation["backend"] == "inductor"
    assert compilation["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert compilation["compile_sizes"] == [96]
    assert compilation["cudagraph_capture_sizes"][-1] == 96
    assert 48 in compilation["cudagraph_capture_sizes"]
    assert 96 in compilation["cudagraph_capture_sizes"]


def test_dp2_profile_maps_to_two_independent_48_request_nemo_rl_engines() -> None:
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=10_240,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
        async_scheduling=True,
    )

    kwargs = profile.engine_kwargs(model="/checkpoint", seed=29, load_format="dummy")
    nemo_rl = profile.nemo_rl_generation_config(load_format="dummy", request_seed=29)

    assert profile.global_batch_size == 96
    assert profile.replica_count == 2
    assert profile.per_engine_batch_size == 48
    assert kwargs["tensor_parallel_size"] == 1
    assert "distributed_executor_backend" not in kwargs
    assert kwargs["worker_extension_cls"] == "bionemo.evo2.vllm.worker.Evo2VllmWorkerExtension"
    assert kwargs["max_num_seqs"] == 48
    assert kwargs["async_scheduling"] is True
    assert kwargs["compilation_config"]["compile_sizes"] == [48]
    assert kwargs["compilation_config"]["cudagraph_capture_sizes"][-1] == 48
    assert 20 in kwargs["compilation_config"]["cudagraph_capture_sizes"]
    assert 96 not in kwargs["compilation_config"]["cudagraph_capture_sizes"]

    assert nemo_rl["backend"] == "vllm"
    assert nemo_rl["generation_batch_size"] == 96
    assert nemo_rl["request_seed"] == 29
    assert nemo_rl["generation_worker_cls"] == ("bionemo.evo2.vllm.nemo_generation_worker.Evo2NemoRlGenerationWorker")
    assert ACTOR_ENVIRONMENT_REGISTRY[nemo_rl["generation_worker_cls"]] == VLLM_EXECUTABLE
    assert nemo_rl["vllm_cfg"] == {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.92,
        "max_model_len": 10_240,
        "skip_tokenizer_init": True,
        "async_engine": False,
        "load_format": "dummy",
        "precision": "bfloat16",
        "kv_cache_dtype": "auto",
        "enforce_eager": False,
        "enable_prefix_caching": False,
    }
    assert nemo_rl["vllm_kwargs"]["max_num_seqs"] == 48
    assert nemo_rl["vllm_kwargs"]["async_scheduling"] is True
    assert nemo_rl["vllm_kwargs"]["mamba_cache_mode"] == "none"
    assert "mamba_block_size" not in nemo_rl["vllm_kwargs"]
    assert nemo_rl["vllm_kwargs"]["worker_extension_cls"] == (
        "bionemo.evo2.vllm.nemo_worker.Evo2NemoRlVllmWorkerExtension"
    )
    nemo_owned_llm_kwargs = {
        "model",
        "served_model_name",
        "load_format",
        "skip_tokenizer_init",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "enable_expert_parallel",
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "dtype",
        "seed",
        "enforce_eager",
        "max_model_len",
        "trust_remote_code",
        "enable_sleep_mode",
        "disable_log_stats",
        "logprobs_mode",
    }
    assert nemo_owned_llm_kwargs.isdisjoint(nemo_rl["vllm_kwargs"])


def test_long_prefill_profile_admits_multiple_packed_partial_requests() -> None:
    profile = Evo2VllmProfile(
        topology="tp2",
        max_model_len=300_008,
        max_num_batched_tokens=32_768,
        gpu_memory_utilization=0.92,
        max_concurrent_partial_prefills=8,
        long_prefill_chunk_tokens=4_096,
    )

    kwargs = profile.engine_kwargs(model="/checkpoint")

    assert kwargs["max_num_partial_prefills"] == 8
    assert kwargs["max_long_partial_prefills"] == 8
    assert kwargs["long_prefill_token_threshold"] == 4_096
    assert kwargs["enable_chunked_prefill"] is True


def test_o3_throughput_profile_is_explicit_and_resolved_without_batch_drift() -> None:
    profile = Evo2VllmProfile(
        topology="tp2",
        max_model_len=50_000,
        max_num_batched_tokens=32_768,
        gpu_memory_utilization=0.95,
        optimization_level=3,
        performance_mode="throughput",
        proof=True,
    )

    kwargs = profile.engine_kwargs(model="/checkpoint")
    resolved = profile.expected_resolved_config()

    assert kwargs["optimization_level"] == 3
    assert kwargs["performance_mode"] == "throughput"
    assert resolved["runtime"] == {
        "optimization_level": 3,
        "performance_mode": "throughput",
    }
    assert resolved["scheduler"]["max_num_seqs"] == 96
    assert resolved["scheduler"]["max_num_batched_tokens"] == 32_768
    validate_resolved_profile(profile, resolved)


def test_requested_profile_sweep_includes_performance_and_long_prefill_dimensions() -> None:
    profiles = optimized_profile_sweep(topology="tp2", max_model_len=10_240)

    dimensions = {
        (
            profile.max_num_batched_tokens,
            profile.gpu_memory_utilization,
            profile.optimization_level,
            profile.performance_mode,
            profile.max_concurrent_partial_prefills,
            profile.long_prefill_chunk_tokens,
        )
        for profile in profiles
    }
    assert len(profiles) == 36
    assert dimensions == {
        (tokens, utilization, optimization, performance, concurrency, threshold)
        for tokens in (16_384, 32_768)
        for utilization in (0.92, 0.95, 0.97)
        for optimization, performance in ((2, "balanced"), (3, "throughput"))
        for concurrency, threshold in ((1, 0), (2, 4_096), (4, 4_096))
    }


def test_profile_rejects_unsupported_or_misleading_settings() -> None:
    base = Evo2VllmProfile(
        topology="tp2",
        max_model_len=10_240,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.92,
    )

    with pytest.raises(ValueError, match="TP2 Ray"):
        replace(base, async_scheduling=True)
    with pytest.raises(ValueError, match="partial prefill"):
        replace(base, max_concurrent_partial_prefills=8, long_prefill_chunk_tokens=0)
    with pytest.raises(ValueError, match="max_model_len"):
        replace(base, max_model_len=0)
    with pytest.raises(ValueError, match="optimization_level"):
        replace(base, optimization_level=4)
    with pytest.raises(ValueError, match="performance_mode"):
        replace(base, performance_mode="latency")


def test_resolved_profile_validation_requires_full_decode_graphs_and_no_fallback() -> None:
    profile = Evo2VllmProfile(
        topology="tp2",
        max_model_len=50_000,
        max_num_batched_tokens=32_768,
        gpu_memory_utilization=0.95,
        proof=True,
    )
    resolved = profile.expected_resolved_config()

    validate_resolved_profile(profile, resolved)

    for path, value, message in [
        (("model", "enforce_eager"), True, "enforce_eager"),
        (("compilation", "mode"), 0, "VLLM_COMPILE"),
        (("compilation", "cudagraph_mode"), "PIECEWISE", "FULL_AND_PIECEWISE"),
        (("cache", "enable_prefix_caching"), True, "prefix caching"),
        (("cache", "mamba_cache_mode"), "all", "mamba_cache_mode"),
        (("scheduler", "max_num_seqs"), 48, "max_num_seqs"),
        (("runtime", "optimization_level"), 0, "optimization_level"),
        (("runtime", "performance_mode"), "throughput", "performance_mode"),
    ]:
        invalid = json.loads(json.dumps(resolved))
        invalid[path[0]][path[1]] = value
        with pytest.raises(AssertionError, match=message):
            validate_resolved_profile(profile, invalid)


def test_compilation_counter_snapshot_is_json_serializable() -> None:
    class Counter:
        num_models_seen = 1
        num_backend_compilations = 2
        num_inductor_compiles = 3
        num_eager_compiles = 0
        num_gpu_runner_capture_triggers = 1
        num_cudagraph_captured = 7
        stock_torch_compile_count = 0

    snapshot = compilation_counter_snapshot(Counter())

    assert snapshot == {
        "num_models_seen": 1,
        "num_backend_compilations": 2,
        "num_inductor_compiles": 3,
        "num_eager_compiles": 0,
        "num_gpu_runner_capture_triggers": 1,
        "num_cudagraph_captured": 7,
        "stock_torch_compile_count": 0,
    }
    json.dumps(snapshot)


def test_pinned_vllm_resolver_preserves_profile_and_resolves_omitted_mamba_block(tmp_path) -> None:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    from bionemo.evo2.vllm.config import Evo2Config
    from bionemo.evo2.vllm.plugin import register

    register()
    Evo2Config(max_position_embeddings=257).save_pretrained(tmp_path)
    profile = Evo2VllmProfile(
        topology="dp2",
        max_model_len=257,
        max_num_batched_tokens=4_096,
        gpu_memory_utilization=0.92,
        proof=True,
    )
    engine_config = EngineArgs(**profile.engine_kwargs(model=str(tmp_path))).create_engine_config(
        usage_context=UsageContext.LLM_CLASS
    )

    resolved = resolved_config_snapshot(engine_config)

    assert "mamba_block_size" not in profile.engine_kwargs(model=str(tmp_path))
    assert resolved["cache"]["mamba_block_size"] == 257
    validate_resolved_profile(profile, resolved)
