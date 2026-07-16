# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Validated vLLM 0.20 profiles for Evo2 rollout inference."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Literal

from bionemo.evo2.vllm.artifact_io import read_json_snapshot


Topology = Literal["tp2", "dp2"]
PerformanceMode = Literal["balanced", "throughput"]

_GLOBAL_BATCH_SIZE = 96
_PREFIX_CACHE_BLOCK_SIZE = 16
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
    shared_prefix_state_reuse: bool = False
    global_wave_size: int = _GLOBAL_BATCH_SIZE
    max_num_seqs: int | None = None

    def __post_init__(self) -> None:
        """Validate settings before an engine can consume them."""
        if type(self.topology) is not str:
            raise TypeError("topology must be a built-in string")
        for field in (
            "max_model_len",
            "max_num_batched_tokens",
            "max_concurrent_partial_prefills",
            "long_prefill_chunk_tokens",
            "optimization_level",
            "global_wave_size",
        ):
            if type(getattr(self, field)) is not int:
                raise TypeError(f"{field} must be a built-in integer")
        if self.max_num_seqs is not None and type(self.max_num_seqs) is not int:
            raise TypeError("max_num_seqs must be a built-in integer or None")
        for field in ("async_scheduling", "proof", "shared_prefix_state_reuse"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be a built-in bool")
        if type(self.gpu_memory_utilization) is not float or not math.isfinite(self.gpu_memory_utilization):
            raise TypeError("gpu_memory_utilization must be a finite built-in float")
        if type(self.performance_mode) is not str:
            raise TypeError("performance_mode must be a built-in string")
        if self.topology not in ("tp2", "dp2"):
            raise ValueError(f"unsupported topology: {self.topology}")
        if (
            isinstance(self.global_wave_size, bool)
            or not isinstance(self.global_wave_size, int)
            or self.global_wave_size <= 0
        ):
            raise ValueError("global_wave_size must be a positive integer")
        if self.global_wave_size % self.replica_count:
            raise ValueError("global_wave_size must be divisible by the topology replica count")
        if self.resolved_max_num_seqs < self.per_engine_batch_size:
            raise ValueError("max_num_seqs must cover every request in one per-engine wave")
        if self.resolved_max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.max_num_batched_tokens < self.resolved_max_num_seqs:
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
        """Return the physical global request count submitted in one call."""
        return self.global_wave_size

    @property
    def gdpo_target_batch_size(self) -> int:
        """Return the fixed request target used for one GDPO optimization batch."""
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
        return self.global_wave_size // self.replica_count

    @property
    def resolved_max_num_seqs(self) -> int:
        """Return the explicit per-engine scheduler request ceiling."""
        return self.per_engine_batch_size if self.max_num_seqs is None else self.max_num_seqs

    @property
    def gdpo_waves_to_96(self) -> int:
        """Return the number of explicit global calls needed for one 96-request GDPO batch."""
        return (self.gdpo_target_batch_size + self.global_wave_size - 1) // self.global_wave_size

    @property
    def cudagraph_capture_sizes(self) -> tuple[int, ...]:
        """Return explicit decode graph sizes through the real batch size."""
        defaults = _TP2_CAPTURE_SIZES if self.topology == "tp2" else _DP2_CAPTURE_SIZES
        sizes = {size for size in defaults if size <= self.resolved_max_num_seqs}
        sizes.add(self.per_engine_batch_size)
        sizes.add(self.resolved_max_num_seqs)
        return tuple(sorted(sizes))

    def engine_kwargs(
        self,
        *,
        model: str,
        seed: int = 42,
        load_format: str = "safetensors",
    ) -> dict[str, Any]:
        """Return kwargs accepted directly by ``vllm.LLM``."""
        from bionemo.evo2.vllm.sampler import sampler_runtime_environment_contract

        sampler_runtime_environment_contract()
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
            "logprobs_mode": "processed_logprobs",
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.resolved_max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "enable_chunked_prefill": True,
            "max_num_partial_prefills": self.max_concurrent_partial_prefills,
            "max_long_partial_prefills": self.max_concurrent_partial_prefills,
            "long_prefill_token_threshold": self.long_prefill_chunk_tokens,
            "enable_prefix_caching": self.shared_prefix_state_reuse,
            "mamba_cache_mode": "align" if self.shared_prefix_state_reuse else "none",
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
                "compile_sizes": sorted({self.per_engine_batch_size, self.resolved_max_num_seqs}),
            },
        }
        if self.shared_prefix_state_reuse:
            kwargs["block_size"] = _PREFIX_CACHE_BLOCK_SIZE
            kwargs["mamba_block_size"] = _PREFIX_CACHE_BLOCK_SIZE
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
            "logprobs_mode",
        ):
            engine_kwargs.pop(key)

        return {
            "backend": "vllm",
            "generation_batch_size": self.global_batch_size,
            "request_seed": request_seed,
            "evo2_collect_proof": self.proof,
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
                "enable_prefix_caching": self.shared_prefix_state_reuse,
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
                "logprobs_mode": "processed_logprobs",
            },
            "parallel": {
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": 1,
            },
            "scheduler": {
                "max_num_seqs": self.resolved_max_num_seqs,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "enable_chunked_prefill": True,
                "max_num_partial_prefills": self.max_concurrent_partial_prefills,
                "max_long_partial_prefills": self.max_concurrent_partial_prefills,
                "long_prefill_token_threshold": self.long_prefill_chunk_tokens,
                "async_scheduling": self.async_scheduling,
            },
            "cache": {
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "enable_prefix_caching": self.shared_prefix_state_reuse,
                "block_size": _PREFIX_CACHE_BLOCK_SIZE,
                "mamba_cache_mode": "align" if self.shared_prefix_state_reuse else "none",
                "mamba_block_size": _PREFIX_CACHE_BLOCK_SIZE if self.shared_prefix_state_reuse else self.max_model_len,
            },
            "compilation": {
                "mode": 3,
                "backend": "inductor",
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": list(self.cudagraph_capture_sizes),
                "compile_sizes": sorted({self.per_engine_batch_size, self.resolved_max_num_seqs}),
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
            "logprobs_mode": model.logprobs_mode,
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
            "block_size": cache.block_size,
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

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    require(
        resolved["runtime"]["optimization_level"] == profile.optimization_level,
        "optimization_level drifted",
    )
    require(resolved["runtime"]["performance_mode"] == profile.performance_mode, "performance_mode drifted")
    require(resolved["model"]["enforce_eager"] is False, "enforce_eager must remain false")
    require(resolved["model"]["max_model_len"] == profile.max_model_len, "max_model_len drifted")
    require(
        resolved["model"]["logprobs_mode"] == "processed_logprobs",
        "logprobs_mode must remain processed_logprobs",
    )
    require(
        resolved["parallel"]["tensor_parallel_size"] == profile.tensor_parallel_size,
        "tensor_parallel_size drifted",
    )
    require(resolved["scheduler"]["max_num_seqs"] == profile.resolved_max_num_seqs, "max_num_seqs drifted")
    require(
        resolved["scheduler"]["max_num_batched_tokens"] == profile.max_num_batched_tokens,
        "max_num_batched_tokens drifted",
    )
    require(resolved["scheduler"]["enable_chunked_prefill"] is True, "chunked prefill must remain enabled")
    require(
        resolved["scheduler"]["async_scheduling"] is profile.async_scheduling,
        "async scheduling drifted",
    )
    require(
        resolved["cache"]["enable_prefix_caching"] is profile.shared_prefix_state_reuse,
        "prefix caching drifted",
    )
    if profile.shared_prefix_state_reuse:
        require(resolved["cache"]["mamba_cache_mode"] == "align", "prefix reuse requires align mode")
        require(
            resolved["cache"]["block_size"] == _PREFIX_CACHE_BLOCK_SIZE,
            "prefix-cache block size drifted",
        )
        require(
            resolved["cache"]["mamba_block_size"] == resolved["cache"]["block_size"],
            "Mamba and attention block sizes must match for align mode",
        )
    else:
        require(resolved["cache"]["mamba_cache_mode"] == "none", "mamba_cache_mode must remain none")
        require(
            resolved["cache"]["mamba_block_size"] == profile.max_model_len,
            "resolved mamba_block_size must equal max_model_len when prefix caching is disabled",
        )
    require(resolved["compilation"]["mode"] == 3, "VLLM_COMPILE mode 3 is required")
    require(resolved["compilation"]["backend"] == "inductor", "Inductor backend is required")
    require(
        resolved["compilation"]["cudagraph_mode"] == "FULL_AND_PIECEWISE",
        "FULL_AND_PIECEWISE is required for FULL steady decode",
    )
    capture_sizes = resolved["compilation"]["cudagraph_capture_sizes"]
    require(profile.resolved_max_num_seqs in capture_sizes, "scheduler ceiling must be CUDA-graph captured")
    require(profile.per_engine_batch_size in capture_sizes, "real topology batch must be CUDA-graph captured")
    require(
        resolved["compilation"]["compile_sizes"]
        == sorted({profile.per_engine_batch_size, profile.resolved_max_num_seqs}),
        "real topology batch must be statically compiled",
    )
    if profile.proof:
        require(resolved["observability"]["cudagraph_metrics"] is True, "proof run requires cudagraph_metrics")


def context_length_preflight(
    profile: Evo2VllmProfile,
    *,
    model: str | Path,
    workload_max_total_tokens: int,
    load_format: str = "safetensors",
) -> dict[str, Any]:
    """Resolve the pinned vLLM length contract without loading model weights or GPUs."""
    if (
        isinstance(workload_max_total_tokens, bool)
        or not isinstance(workload_max_total_tokens, int)
        or workload_max_total_tokens <= 0
    ):
        raise ValueError("workload_max_total_tokens must be a positive integer")
    if profile.max_model_len < workload_max_total_tokens:
        raise ValueError(
            f"profile max_model_len={profile.max_model_len} is smaller than "
            f"workload max_total_tokens={workload_max_total_tokens}"
        )

    checkpoint = Path(model).resolve()
    config_path = checkpoint / "config.json"
    config_snapshot = read_json_snapshot(config_path, label="checkpoint config")
    config_data = config_snapshot.value
    if not isinstance(config_data, dict):
        raise ValueError("checkpoint config must be a JSON object")
    declared = config_data.get("max_position_embeddings")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
        raise ValueError("checkpoint config must declare a positive integer max_position_embeddings")

    environment_variable = "VLLM_ALLOW_LONG_MAX_MODEL_LEN"
    raw_override = os.environ.get(environment_variable)
    override_enabled = raw_override is not None and raw_override.strip().lower() in ("1", "true")
    override_required = profile.max_model_len > declared
    if override_required and not override_enabled:
        raise RuntimeError(
            f"requested max_model_len={profile.max_model_len} exceeds checkpoint provenance {declared}; "
            f"set {environment_variable}=1 explicitly"
        )

    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    from bionemo.evo2.vllm.model import evo2_context_length_contract
    from bionemo.evo2.vllm.plugin import register

    register()
    engine_config = EngineArgs(
        **profile.engine_kwargs(model=str(checkpoint), load_format=load_format)
    ).create_engine_config(usage_context=UsageContext.LLM_CLASS)
    resolved = resolved_config_snapshot(engine_config)
    validate_resolved_profile(profile, resolved)
    resolved_hf_max = int(engine_config.model_config.hf_config.max_position_embeddings)
    if resolved_hf_max != declared:
        raise AssertionError("vLLM resolution rewrote checkpoint max_position_embeddings provenance")
    length_contract = evo2_context_length_contract(engine_config)
    if length_contract["rotary_max_position_embeddings"] != profile.max_model_len:
        raise AssertionError("Evo2 rotary length did not follow resolved max_model_len")
    if length_contract["hyena_state_length_dependent"] is not False:
        raise AssertionError("Evo2 recurrent state unexpectedly depends on max_model_len")

    return {
        "schema_version": 1,
        "checkpoint_config_path": str(config_path),
        "checkpoint_config_sha256": config_snapshot.sha256,
        "checkpoint_declared_max_position_embeddings": declared,
        "requested_max_model_len": profile.max_model_len,
        "resolved_max_model_len": int(engine_config.model_config.max_model_len),
        "workload_max_total_tokens": workload_max_total_tokens,
        "workload_fits_resolved_max_model_len": True,
        "workload_headroom_tokens": int(engine_config.model_config.max_model_len) - workload_max_total_tokens,
        "resolved_hf_max_position_embeddings": resolved_hf_max,
        "long_length_override": {
            "environment_variable": environment_variable,
            "raw_value": raw_override,
            "enabled": override_enabled,
            "required": override_required,
        },
        "provenance_rewritten": False,
        "length_clipped": int(engine_config.model_config.max_model_len) != profile.max_model_len,
        "evo2_length_contract": length_contract,
        "resolved_config": resolved,
    }


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
    "context_length_preflight",
    "optimized_profile_sweep",
    "resolved_config_snapshot",
    "validate_resolved_profile",
]
