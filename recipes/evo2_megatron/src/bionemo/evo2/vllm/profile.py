# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Validated vLLM 0.20 profiles for Evo2 rollout inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Literal


Topology = Literal["tp2", "dp2"]
PerformanceMode = Literal["balanced", "throughput"]

_GLOBAL_BATCH_SIZE = 96
_TP2_CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96)
_DP2_CAPTURE_SIZES = (1, 2, 4, 8, 16, 20, 24, 32, 40, 48)
_COUNTER_FIELDS = (
    "num_models_seen",
    "num_backend_compilations",
    "num_inductor_compiles",
    "num_eager_compiles",
    "num_gpu_runner_capture_triggers",
    "num_cudagraph_captured",
    "stock_torch_compile_count",
)


@dataclass(frozen=True)
class Evo2VllmProfile:
    """One topology-local, accuracy-preserving Evo2 vLLM engine profile."""

    topology: Topology
    max_model_len: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    async_scheduling: bool = False
    proof: bool = False
    max_concurrent_partial_prefills: int = 1
    long_prefill_chunk_tokens: int = 0
    optimization_level: int = 2
    performance_mode: PerformanceMode = "balanced"

    def __post_init__(self) -> None:
        """Validate settings before an engine can consume them."""
        if self.topology not in ("tp2", "dp2"):
            raise ValueError(f"unsupported topology: {self.topology}")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.max_num_batched_tokens < self.per_engine_batch_size:
            raise ValueError("max_num_batched_tokens must cover one decode token per active request")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if self.topology == "tp2" and self.async_scheduling:
            raise ValueError("TP2 Ray does not support async scheduling in vLLM 0.20")
        if self.max_concurrent_partial_prefills < 1:
            raise ValueError("max_concurrent_partial_prefills must be positive")
        if self.max_concurrent_partial_prefills > 1 and self.long_prefill_chunk_tokens <= 0:
            raise ValueError("concurrent partial prefill requires a positive long_prefill_chunk_tokens")
        if self.long_prefill_chunk_tokens < 0:
            raise ValueError("long_prefill_chunk_tokens cannot be negative")
        if self.long_prefill_chunk_tokens > self.max_model_len:
            raise ValueError("long_prefill_chunk_tokens cannot exceed max_model_len")
        if self.optimization_level not in (2, 3):
            raise ValueError("optimized Evo2 profiles require optimization_level 2 or 3")
        if self.performance_mode not in ("balanced", "throughput"):
            raise ValueError(f"unsupported performance_mode: {self.performance_mode}")

    @property
    def global_batch_size(self) -> int:
        """Return the GDPO-wide request count."""
        return _GLOBAL_BATCH_SIZE

    @property
    def replica_count(self) -> int:
        """Return the number of independently scheduled engines."""
        return 1 if self.topology == "tp2" else 2

    @property
    def tensor_parallel_size(self) -> int:
        """Return the model tensor-parallel width per engine."""
        return 2 if self.topology == "tp2" else 1

    @property
    def per_engine_batch_size(self) -> int:
        """Return the real request count handled by each engine."""
        return self.global_batch_size // self.replica_count

    @property
    def cudagraph_capture_sizes(self) -> tuple[int, ...]:
        """Return explicit decode graph sizes through the real batch size."""
        return _TP2_CAPTURE_SIZES if self.topology == "tp2" else _DP2_CAPTURE_SIZES

    def engine_kwargs(
        self,
        *,
        model: str,
        seed: int = 42,
        load_format: str = "safetensors",
    ) -> dict[str, Any]:
        """Return kwargs accepted directly by ``vllm.LLM``."""
        kwargs = {
            "model": model,
            "load_format": load_format,
            "skip_tokenizer_init": True,
            "model_impl": "vllm",
            "dtype": "bfloat16",
            "seed": seed,
            "worker_extension_cls": "bionemo.evo2.vllm.worker.Evo2VllmWorkerExtension",
            "optimization_level": self.optimization_level,
            "performance_mode": self.performance_mode,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "hf_overrides": {"max_position_embeddings": self.max_model_len},
            "max_num_seqs": self.per_engine_batch_size,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "enable_chunked_prefill": True,
            "max_num_partial_prefills": self.max_concurrent_partial_prefills,
            "max_long_partial_prefills": self.max_concurrent_partial_prefills,
            "long_prefill_token_threshold": self.long_prefill_chunk_tokens,
            "enable_prefix_caching": False,
            "mamba_cache_mode": "none",
            "mamba_cache_dtype": "float32",
            "mamba_ssm_cache_dtype": "float32",
            "kv_cache_dtype": "auto",
            "enforce_eager": False,
            "async_scheduling": self.async_scheduling,
            "cudagraph_metrics": self.proof,
            "disable_log_stats": False,
            "compilation_config": {
                "mode": 3,
                "backend": "inductor",
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_num_of_warmups": 1,
                "cudagraph_capture_sizes": list(self.cudagraph_capture_sizes),
                "compile_sizes": [self.per_engine_batch_size],
            },
        }
        if self.topology == "tp2":
            kwargs["distributed_executor_backend"] = "ray"
        return kwargs

    def nemo_rl_generation_config(
        self,
        *,
        load_format: str = "dummy",
        request_seed: int = 42,
    ) -> dict[str, Any]:
        """Return the stock NeMo-RL generation subtree for this profile."""
        if request_seed < 0:
            raise ValueError("request_seed must be nonnegative")
        from nemo_rl.distributed.ray_actor_environment_registry import (
            ACTOR_ENVIRONMENT_REGISTRY,
            VLLM_EXECUTABLE,
        )

        generation_worker_cls = "bionemo.evo2.vllm.nemo_generation_worker.Evo2NemoRlGenerationWorker"
        ACTOR_ENVIRONMENT_REGISTRY[generation_worker_cls] = VLLM_EXECUTABLE
        engine_kwargs = self.engine_kwargs(model="unused-by-nemo-rl", load_format=load_format)
        for key in (
            "model",
            "load_format",
            "skip_tokenizer_init",
            "dtype",
            "seed",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "gpu_memory_utilization",
            "max_model_len",
            "enable_prefix_caching",
            "enforce_eager",
            "kv_cache_dtype",
            "worker_extension_cls",
            "disable_log_stats",
        ):
            engine_kwargs.pop(key)

        return {
            "backend": "vllm",
            "generation_batch_size": self.global_batch_size,
            "request_seed": request_seed,
            "generation_worker_cls": generation_worker_cls,
            "vllm_cfg": {
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": 1,
                "expert_parallel_size": 1,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "skip_tokenizer_init": True,
                "async_engine": False,
                "load_format": load_format,
                "precision": "bfloat16",
                "kv_cache_dtype": "auto",
                "enforce_eager": False,
                "enable_prefix_caching": False,
            },
            "vllm_kwargs": {
                **engine_kwargs,
                "worker_extension_cls": ("bionemo.evo2.vllm.nemo_worker.Evo2NemoRlVllmWorkerExtension"),
            },
        }

    def expected_resolved_config(self) -> dict[str, Any]:
        """Return the pinned subset expected after ``EngineArgs`` resolution."""
        return {
            "runtime": {
                "optimization_level": self.optimization_level,
                "performance_mode": self.performance_mode,
            },
            "model": {
                "max_model_len": self.max_model_len,
                "enforce_eager": False,
            },
            "parallel": {
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": 1,
            },
            "scheduler": {
                "max_num_seqs": self.per_engine_batch_size,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "enable_chunked_prefill": True,
                "max_num_partial_prefills": self.max_concurrent_partial_prefills,
                "max_long_partial_prefills": self.max_concurrent_partial_prefills,
                "long_prefill_token_threshold": self.long_prefill_chunk_tokens,
                "async_scheduling": self.async_scheduling,
            },
            "cache": {
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "enable_prefix_caching": False,
                "mamba_cache_mode": "none",
                # vLLM 0.20 rejects an explicit value when prefix caching is off,
                # then resolves the omitted accounting block to max_model_len.
                "mamba_block_size": self.max_model_len,
            },
            "compilation": {
                "mode": 3,
                "backend": "inductor",
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": list(self.cudagraph_capture_sizes),
                "compile_sizes": [self.per_engine_batch_size],
            },
            "observability": {"cudagraph_metrics": self.proof},
        }


def optimized_profile_sweep(*, topology: Topology, max_model_len: int) -> list[Evo2VllmProfile]:
    """Build scheduler, memory, runtime-policy, and long-prefill sweep points."""
    long_prefill_threshold = min(4_096, max_model_len)
    return [
        Evo2VllmProfile(
            topology=topology,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            optimization_level=optimization_level,
            performance_mode=performance_mode,
            max_concurrent_partial_prefills=max_concurrent_partial_prefills,
            long_prefill_chunk_tokens=long_prefill_chunk_tokens,
        )
        for max_num_batched_tokens in (16_384, 32_768)
        for gpu_memory_utilization in (0.92, 0.95, 0.97)
        for optimization_level, performance_mode in ((2, "balanced"), (3, "throughput"))
        for max_concurrent_partial_prefills, long_prefill_chunk_tokens in (
            (1, 0),
            (2, long_prefill_threshold),
            (4, long_prefill_threshold),
        )
    ]


def _enum_name_or_value(value: Any) -> Any:
    if isinstance(value, IntEnum):
        return int(value)
    if isinstance(value, Enum):
        return value.name
    return value


def resolved_config_snapshot(vllm_config: Any) -> dict[str, Any]:
    """Extract the stable, JSON-safe profile subset from a vLLM config."""
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    compilation = vllm_config.compilation_config
    observability = vllm_config.observability_config
    return {
        "runtime": {
            "optimization_level": _enum_name_or_value(vllm_config.optimization_level),
            "performance_mode": vllm_config.performance_mode,
        },
        "model": {
            "max_model_len": model.max_model_len,
            "enforce_eager": model.enforce_eager,
        },
        "parallel": {
            "tensor_parallel_size": parallel.tensor_parallel_size,
            "pipeline_parallel_size": parallel.pipeline_parallel_size,
        },
        "scheduler": {
            "max_num_seqs": scheduler.max_num_seqs,
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "enable_chunked_prefill": scheduler.enable_chunked_prefill,
            "max_num_partial_prefills": scheduler.max_num_partial_prefills,
            "max_long_partial_prefills": scheduler.max_long_partial_prefills,
            "long_prefill_token_threshold": scheduler.long_prefill_token_threshold,
            "async_scheduling": scheduler.async_scheduling,
        },
        "cache": {
            "gpu_memory_utilization": cache.gpu_memory_utilization,
            "enable_prefix_caching": cache.enable_prefix_caching,
            "mamba_cache_mode": cache.mamba_cache_mode,
            "mamba_block_size": cache.mamba_block_size,
        },
        "compilation": {
            "mode": _enum_name_or_value(compilation.mode),
            "backend": compilation.backend,
            "cudagraph_mode": _enum_name_or_value(compilation.cudagraph_mode),
            "cudagraph_capture_sizes": list(compilation.cudagraph_capture_sizes),
            "compile_sizes": list(compilation.compile_sizes),
        },
        "observability": {"cudagraph_metrics": observability.cudagraph_metrics},
    }


def validate_resolved_profile(profile: Evo2VllmProfile, resolved: dict[str, Any]) -> None:
    """Reject any resolved setting that weakens the optimized proof profile."""
    assert resolved["runtime"]["optimization_level"] == profile.optimization_level, "optimization_level drifted"
    assert resolved["runtime"]["performance_mode"] == profile.performance_mode, "performance_mode drifted"
    assert resolved["model"]["enforce_eager"] is False, "enforce_eager must remain false"
    assert resolved["model"]["max_model_len"] == profile.max_model_len, "max_model_len drifted"
    assert resolved["parallel"]["tensor_parallel_size"] == profile.tensor_parallel_size, "tensor_parallel_size drifted"
    assert resolved["scheduler"]["max_num_seqs"] == profile.per_engine_batch_size, "max_num_seqs drifted"
    assert resolved["scheduler"]["max_num_batched_tokens"] == profile.max_num_batched_tokens, (
        "max_num_batched_tokens drifted"
    )
    assert resolved["scheduler"]["enable_chunked_prefill"] is True, "chunked prefill must remain enabled"
    assert resolved["scheduler"]["async_scheduling"] is profile.async_scheduling, "async scheduling drifted"
    assert resolved["cache"]["enable_prefix_caching"] is False, "prefix caching must remain disabled"
    assert resolved["cache"]["mamba_cache_mode"] == "none", "mamba_cache_mode must remain none"
    assert resolved["cache"]["mamba_block_size"] == profile.max_model_len, (
        "resolved mamba_block_size must equal max_model_len when prefix caching is disabled"
    )
    assert resolved["compilation"]["mode"] == 3, "VLLM_COMPILE mode 3 is required"
    assert resolved["compilation"]["backend"] == "inductor", "Inductor backend is required"
    assert resolved["compilation"]["cudagraph_mode"] == "FULL_AND_PIECEWISE", (
        "FULL_AND_PIECEWISE is required for FULL steady decode"
    )
    capture_sizes = resolved["compilation"]["cudagraph_capture_sizes"]
    assert profile.per_engine_batch_size in capture_sizes, "real topology batch must be CUDA-graph captured"
    assert resolved["compilation"]["compile_sizes"] == [profile.per_engine_batch_size], (
        "real topology batch must be statically compiled"
    )
    if profile.proof:
        assert resolved["observability"]["cudagraph_metrics"] is True, "proof run requires cudagraph_metrics"


def compilation_counter_snapshot(counter: Any | None = None) -> dict[str, int]:
    """Return the vLLM compilation counters needed by proof artifacts."""
    if counter is None:
        from vllm.compilation.counter import compilation_counter

        counter = compilation_counter
    return {field: int(getattr(counter, field)) for field in _COUNTER_FIELDS}


__all__ = [
    "Evo2VllmProfile",
    "PerformanceMode",
    "Topology",
    "compilation_counter_snapshot",
    "optimized_profile_sweep",
    "resolved_config_snapshot",
    "validate_resolved_profile",
]
