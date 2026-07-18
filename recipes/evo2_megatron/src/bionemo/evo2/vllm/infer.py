# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Production Evo2 inference through the qualified public vLLM path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from bionemo.evo2.vllm.artifact_io import read_json_snapshot, read_jsonl_snapshot
from bionemo.evo2.vllm.benchmark import validate_dna_output_token_ids
from bionemo.evo2.vllm.sampler_runtime import NEMO_VLLM_ACTOR_FQN, sampler_runtime_environment_contract
from bionemo.evo2.vllm.tokenizer_io import SnapshotBoundTokenizer


_CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96)
_DNA_OUTPUT_TOKEN_IDS = (65, 67, 71, 78, 84)
_VLLM_020_SPLITTING_OPS = (
    "vllm::unified_attention_with_output",
    "vllm::unified_mla_attention_with_output",
    "vllm::mamba_mixer2",
    "vllm::mamba_mixer",
    "vllm::short_conv",
    "vllm::linear_attention",
    "vllm::plamo2_mamba_mixer",
    "vllm::gdn_attention_core",
    "vllm::gdn_attention_core_xpu",
    "vllm::olmo_hybrid_gdn_full_forward",
    "vllm::kda_attention",
    "vllm::sparse_attn_indexer",
    "vllm::rocm_aiter_sparse_attn_indexer",
    "vllm::deepseek_v4_attention",
    "vllm::unified_kv_cache_update",
    "vllm::unified_mla_kv_cache_update",
    "bionemo_evo2::hyena_mixer",
)


@dataclass(frozen=True)
class InferenceRequest:
    """One caller-owned public inference request."""

    request_id: str
    prompt: str
    prompt_id: str | None = None
    length_stratum: int | None = None
    rollout_ordinal: int | None = None
    order_index: int | None = None
    validation_seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise TypeError("request_id must be a nonempty built-in string")
        if type(self.prompt) is not str:
            raise TypeError("prompt must be a built-in string")
        coordinates = (
            self.prompt_id,
            self.length_stratum,
            self.rollout_ordinal,
            self.order_index,
            self.validation_seed,
        )
        if all(value is None for value in coordinates):
            return
        if type(self.prompt_id) is not str or not self.prompt_id:
            raise TypeError("prompt_id must be a nonempty built-in string")
        for label, value in (
            ("length_stratum", self.length_stratum),
            ("rollout_ordinal", self.rollout_ordinal),
            ("order_index", self.order_index),
            ("validation_seed", self.validation_seed),
        ):
            if type(value) is not int or value < 0:
                raise TypeError(f"{label} must be a nonnegative built-in integer")

    def rollout_coordinates(self) -> dict[str, str | int]:
        """Return recipe prompt-group coordinates without adding null legacy fields."""
        if self.prompt_id is None:
            return {}
        return {
            "prompt_id": self.prompt_id,
            "length_stratum": self.length_stratum,
            "rollout_ordinal": self.rollout_ordinal,
            "order_index": self.order_index,
        }


@dataclass(frozen=True)
class ExportIdentity:
    """Cheap run-scoped binding to one Evo2 safetensors export."""

    root: Path
    manifest_sha256: str
    config_sha256: str
    index_sha256: str
    architecture: str
    source_checkpoint: str
    source_iteration: int
    tensor_parallel_divisor: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["root"] = str(self.root)
        return value


def _require_builtin_int(value: Any, *, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{label} must be a built-in integer >= {minimum}")
    return value


def _configured_vllm_python() -> Path | None:
    explicit = os.environ.get("EVO2_VLLM_PYTHON")
    if explicit is not None:
        if not explicit:
            raise RuntimeError("EVO2_VLLM_PYTHON must name an executable interpreter")
        return Path(explicit).expanduser()
    actor_root = os.environ.get("NEMO_RL_VENV_DIR")
    if actor_root is None:
        return None
    return Path(actor_root).expanduser() / NEMO_VLLM_ACTOR_FQN / "bin" / "python"


def require_vllm_runtime(*, argv: Sequence[str] | None = None) -> None:
    """Use the current vLLM runtime or replace this CLI with the locked actor runtime."""
    if importlib.util.find_spec("vllm") is not None:
        return
    if os.environ.get("EVO2_VLLM_REEXEC") == "1":
        raise RuntimeError("the configured isolated vLLM environment still cannot import vllm")
    actor_python = _configured_vllm_python()
    if actor_python is None:
        raise RuntimeError(
            "vLLM is not installed in this environment and no locked actor environment is configured; "
            "run the recipe .ci_build.sh or set EVO2_VLLM_PYTHON"
        )
    if not actor_python.is_file():
        raise RuntimeError(f"configured Evo2 vLLM interpreter does not exist: {actor_python}")
    if not os.access(actor_python, os.X_OK):
        raise RuntimeError(f"configured Evo2 vLLM interpreter is not executable: {actor_python}")
    arguments = tuple(sys.argv[1:]) if argv is None else tuple(argv)
    environment = dict(os.environ)
    environment["EVO2_VLLM_REEXEC"] = "1"
    os.execve(
        str(actor_python),
        [str(actor_python), "-m", "bionemo.evo2.vllm.infer", *arguments],
        environment,
    )
    raise RuntimeError("failed to enter the configured Evo2 vLLM environment")


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [entry.strip() for entry in visible.split(",") if entry.strip() and entry.strip() != "-1"]
        return len(devices)
    import torch

    return int(torch.cuda.device_count())


def resolve_tensor_parallel_size(value: str | int) -> int:
    """Resolve explicit or all-visible-GPU tensor parallelism."""
    visible_gpu_count = _visible_gpu_count()
    if visible_gpu_count <= 0:
        raise RuntimeError("Evo2 vLLM inference requires at least one visible GPU")
    if value == "auto":
        return visible_gpu_count
    if type(value) is int:
        tensor_parallel_size = value
    elif type(value) is str and value.isdecimal():
        tensor_parallel_size = int(value)
    else:
        raise TypeError("tensor_parallel_size must be 'auto' or a positive decimal integer")
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if tensor_parallel_size > visible_gpu_count:
        raise ValueError(
            f"tensor_parallel_size={tensor_parallel_size} exceeds {visible_gpu_count} visible GPU(s)"
        )
    return tensor_parallel_size


def _capture_sizes(batch_size: int, extra: Sequence[int] = ()) -> list[int]:
    sizes = {size for size in _CAPTURE_SIZES if size <= batch_size}
    sizes.add(batch_size)
    for size in extra:
        _require_builtin_int(size, label="additional capture size")
        if size <= batch_size:
            sizes.add(size)
    return sorted(sizes)


def build_engine_kwargs(
    *,
    model: str,
    tensor_parallel_size: int,
    batch_size: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    optimization_level: int,
    performance_mode: str,
    async_scheduling: bool,
    additional_capture_sizes: Sequence[int] = (),
) -> dict[str, Any]:
    """Build the qualified public ``vllm.LLM`` constructor arguments."""
    if type(model) is not str or not model:
        raise TypeError("model must be a nonempty built-in string")
    _require_builtin_int(tensor_parallel_size, label="tensor_parallel_size")
    _require_builtin_int(batch_size, label="batch_size")
    _require_builtin_int(max_model_len, label="max_model_len")
    _require_builtin_int(max_num_batched_tokens, label="max_num_batched_tokens")
    if max_num_batched_tokens < batch_size:
        raise ValueError("max_num_batched_tokens must cover one token for every active request")
    if type(gpu_memory_utilization) is not float or not math.isfinite(gpu_memory_utilization):
        raise TypeError("gpu_memory_utilization must be a finite built-in float")
    if not 0.0 < gpu_memory_utilization < 1.0:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    if type(optimization_level) is not int or optimization_level not in (2, 3):
        raise ValueError("optimization_level must be the integer 2 or 3")
    if type(performance_mode) is not str or performance_mode not in ("balanced", "throughput"):
        raise ValueError("performance_mode must be 'balanced' or 'throughput'")
    if type(async_scheduling) is not bool:
        raise TypeError("async_scheduling must be a built-in bool")

    sampler_runtime_environment_contract()
    capture_sizes = _capture_sizes(batch_size, additional_capture_sizes)
    kwargs: dict[str, Any] = {
        "model": model,
        "load_format": "safetensors",
        "skip_tokenizer_init": True,
        "model_impl": "vllm",
        "dtype": "bfloat16",
        "seed": 42,
        "optimization_level": optimization_level,
        "performance_mode": performance_mode,
        "logprobs_mode": "processed_logprobs",
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": batch_size,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_chunked_prefill": True,
        "max_num_partial_prefills": 1,
        "max_long_partial_prefills": 1,
        "long_prefill_token_threshold": 0,
        "enable_prefix_caching": False,
        "mamba_cache_mode": "none",
        "mamba_cache_dtype": "float32",
        "mamba_ssm_cache_dtype": "float32",
        "kv_cache_dtype": "auto",
        "enforce_eager": False,
        "async_scheduling": async_scheduling,
        "cudagraph_metrics": False,
        "disable_log_stats": False,
        "compilation_config": {
            "mode": 3,
            "backend": "inductor",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_num_of_warmups": 1,
            "cudagraph_capture_sizes": capture_sizes,
            "compile_sizes": sorted({batch_size, *additional_capture_sizes}),
            "splitting_ops": list(_VLLM_020_SPLITTING_OPS),
        },
    }
    if tensor_parallel_size > 1:
        kwargs["distributed_executor_backend"] = "mp"
    return kwargs


def load_export_identity(model: str | Path) -> ExportIdentity:
    """Validate the cheap authoritative files for one Evo2 vLLM export."""
    root = Path(model).resolve()
    manifest = read_json_snapshot(root / "manifest.json", label="Evo2 export manifest")
    config = read_json_snapshot(root / "config.json", label="Evo2 export config")
    index = read_json_snapshot(root / "model.safetensors.index.json", label="Evo2 safetensors index")
    if type(manifest.value) is not dict or manifest.value.get("schema_version") != 1:
        raise RuntimeError("Evo2 export manifest schema is unsupported")
    if manifest.value.get("config_sha256") != config.sha256:
        raise RuntimeError("Evo2 export config digest does not match its manifest")
    if manifest.value.get("index_sha256") != index.sha256:
        raise RuntimeError("Evo2 export index digest does not match its manifest")
    if type(config.value) is not dict:
        raise RuntimeError("Evo2 export config must be a JSON object")
    architectures = config.value.get("architectures")
    if architectures != ["Evo2ForCausalLM"] or config.value.get("model_type") != "evo2":
        raise RuntimeError("vLLM export must declare Evo2ForCausalLM")
    attention_heads = config.value.get("num_attention_heads")
    _require_builtin_int(attention_heads, label="num_attention_heads")
    if type(index.value) is not dict or type(index.value.get("weight_map")) is not dict:
        raise RuntimeError("Evo2 safetensors index weight_map is missing")
    shard_names = set(index.value["weight_map"].values())
    if not shard_names or any(type(name) is not str or Path(name).name != name for name in shard_names):
        raise RuntimeError("Evo2 safetensors index contains an invalid shard name")
    for shard_name in shard_names:
        if not (root / shard_name).is_file():
            raise RuntimeError(f"Evo2 safetensors shard is missing: {shard_name}")
    source_checkpoint = manifest.value.get("source_checkpoint")
    source_iteration = manifest.value.get("source_iteration")
    if type(source_checkpoint) is not str or not source_checkpoint:
        raise RuntimeError("Evo2 export source_checkpoint is missing")
    _require_builtin_int(source_iteration, label="source_iteration", minimum=0)
    return ExportIdentity(
        root=root,
        manifest_sha256=manifest.sha256,
        config_sha256=config.sha256,
        index_sha256=index.sha256,
        architecture="Evo2ForCausalLM",
        source_checkpoint=source_checkpoint,
        source_iteration=source_iteration,
        tensor_parallel_divisor=attention_heads,
    )


def load_prompt_requests(*, prompt: str | None, prompt_file: str | Path | None) -> tuple[InferenceRequest, ...]:
    """Load caller requests from one strict JSONL snapshot or one direct prompt."""
    if prompt_file is None:
        if type(prompt) is not str:
            raise ValueError("either --prompt or --prompt-file is required")
        return (InferenceRequest(request_id="0", prompt=prompt),)
    snapshot = read_jsonl_snapshot(prompt_file, label="Evo2 inference prompt JSONL")
    requests = []
    for index, value in enumerate(snapshot.values):
        if type(value) is not dict:
            raise TypeError(f"prompt JSONL row {index} must be an object")
        if "prompt" in value:
            request_id = value.get("id", str(index))
            requests.append(InferenceRequest(request_id=request_id, prompt=value["prompt"]))
            continue
        messages = value.get("messages")
        if (
            type(messages) is not list
            or len(messages) != 2
            or type(messages[0]) is not dict
            or type(messages[1]) is not dict
            or set(messages[0]) != {"role", "content"}
            or set(messages[1]) != {"role", "content"}
            or messages[0].get("role") != "user"
            or type(messages[0].get("content")) is not str
            or messages[1] != {"role": "assistant", "content": ""}
        ):
            raise TypeError(
                f"prompt JSONL row {index} must contain a flat prompt or one user and empty assistant message"
            )
        prompt_id = value.get("prompt_id")
        rollout_ordinal = value.get("rollout_ordinal")
        if type(prompt_id) is not str or not prompt_id or type(rollout_ordinal) is not int:
            raise TypeError(f"prompt JSONL row {index} has invalid prompt_id or rollout_ordinal")
        requests.append(
            InferenceRequest(
                request_id=f"{prompt_id}-rollout-{rollout_ordinal:04d}",
                prompt=messages[0]["content"],
                prompt_id=prompt_id,
                length_stratum=value.get("length_stratum"),
                rollout_ordinal=rollout_ordinal,
                order_index=value.get("order_index"),
                validation_seed=value.get("validation_seed"),
            )
        )
    request_ids = [request.request_id for request in requests]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("prompt JSONL request IDs must be unique")
    return tuple(requests)


def resolve_tokenizer_json(
    *,
    export_root: str | Path,
    tokenizer_json: str | Path | None,
) -> Path:
    """Resolve an explicit tokenizer or the tokenizer packaged by the exporter."""
    root = Path(export_root).expanduser().resolve()
    if tokenizer_json is not None:
        candidate = Path(tokenizer_json).expanduser()
        if candidate.is_dir():
            candidate = candidate / "tokenizer.json"
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"tokenizer.json does not exist: {candidate}")
        return candidate
    candidates = (root / "tokenizer.json", root / "tokenizer" / "tokenizer.json")
    existing = tuple(candidate for candidate in candidates if candidate.is_file())
    if not existing:
        raise FileNotFoundError(f"export does not contain tokenizer.json under {root}")
    if len(existing) != 1:
        raise RuntimeError(f"export contains ambiguous tokenizer.json files: {existing}")
    return existing[0]


def build_sampling_params_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    """Build one exact-length, chosen-logprob, request-seeded sampling policy."""
    _require_builtin_int(max_new_tokens, label="max_new_tokens")
    _require_builtin_int(seed, label="seed", minimum=0)
    if type(temperature) is not float or not math.isfinite(temperature) or temperature < 0.0:
        raise TypeError("temperature must be a finite nonnegative built-in float")
    if type(top_p) is not float or not math.isfinite(top_p) or not 0.0 <= top_p <= 1.0:
        raise TypeError("top_p must be a finite built-in float between zero and one")
    if type(top_k) is not int or top_k < 0:
        raise TypeError("top_k must be a nonnegative built-in integer")
    greedy = top_k == 1 or temperature == 0.0
    return {
        "temperature": 0.0 if greedy else temperature,
        "top_p": 1.0 if greedy or top_p == 0.0 else top_p,
        "top_k": 0 if greedy else top_k,
        "max_tokens": max_new_tokens,
        "min_tokens": max_new_tokens,
        "logprobs": 0,
        "ignore_eos": True,
        "detokenize": False,
        "allowed_token_ids": list(_DNA_OUTPUT_TOKEN_IDS),
        "seed": seed,
    }


def records_from_public_outputs(
    *,
    requests: Sequence[InferenceRequest],
    prompt_token_ids: Sequence[Sequence[int]],
    request_seeds: Sequence[int],
    outputs: Sequence[Any],
    tokenizer: SnapshotBoundTokenizer,
    max_new_tokens: int,
) -> tuple[dict[str, Any], ...]:
    """Validate public ordered vLLM outputs and preserve legacy JSONL fields."""
    expected_count = len(requests)
    if not (
        len(prompt_token_ids) == len(request_seeds) == len(outputs) == expected_count
    ):
        raise AssertionError("inference requests, seeds, prompts, and outputs must align exactly")
    engine_request_ids: set[str] = set()
    records = []
    for request, expected_prompt_ids, seed, output in zip(
        requests,
        prompt_token_ids,
        request_seeds,
        outputs,
        strict=True,
    ):
        if output.finished is not True or type(output.outputs) is not list or len(output.outputs) != 1:
            raise AssertionError(f"request {request.request_id} did not return one finished public output")
        engine_request_id = output.request_id
        if type(engine_request_id) is not str or not engine_request_id or engine_request_id in engine_request_ids:
            raise AssertionError("vLLM public request IDs must be nonempty and unique")
        engine_request_ids.add(engine_request_id)
        expected_prompt_tuple = tuple(expected_prompt_ids)
        if tuple(output.prompt_token_ids or ()) != expected_prompt_tuple:
            raise AssertionError(f"request {request.request_id} prompt token IDs changed or were reordered")
        completion = output.outputs[0]
        output_token_ids = tuple(completion.token_ids)
        validate_dna_output_token_ids(output_token_ids, request_id=request.request_id)
        if len(output_token_ids) != max_new_tokens:
            raise AssertionError(f"request {request.request_id} must generate exactly {max_new_tokens} tokens")
        if completion.finish_reason != "length" or completion.stop_reason is not None:
            raise AssertionError(f"request {request.request_id} did not finish at the exact length limit")
        if type(completion.logprobs) is not list or len(completion.logprobs) != max_new_tokens:
            raise AssertionError(f"request {request.request_id} is missing aligned chosen logprobs")
        chosen_logprobs = []
        for token_id, position in zip(output_token_ids, completion.logprobs, strict=True):
            if type(position) is not dict or token_id not in position:
                raise AssertionError(f"request {request.request_id} is missing a chosen-token logprob")
            logprob = float(position[token_id].logprob)
            if not math.isfinite(logprob) or logprob > 0.0:
                raise ValueError(f"request {request.request_id} chosen logprobs must be finite and nonpositive")
            chosen_logprobs.append(logprob)
        completion_text = tokenizer.decode(output_token_ids)
        if tuple(map(ord, completion_text)) != output_token_ids:
            raise AssertionError(f"request {request.request_id} output IDs do not decode losslessly")
        records.append(
            {
                "id": request.request_id,
                "prompt": request.prompt,
                "completion": completion_text,
                "finish_reason": "length",
                "usage": {
                    "prompt_tokens": len(expected_prompt_tuple),
                    "completion_tokens": max_new_tokens,
                    "total_tokens": len(expected_prompt_tuple) + max_new_tokens,
                },
                "seed": seed,
                "engine_request_id": engine_request_id,
                "prompt_token_ids": list(expected_prompt_tuple),
                "token_ids": list(output_token_ids),
                "logprobs": {"completion_logprobs": chosen_logprobs},
                **request.rollout_coordinates(),
            }
        )
    return tuple(records)


def _ensure_evo2_plugin() -> None:
    configured = os.environ.get("VLLM_PLUGINS")
    if configured is None:
        os.environ["VLLM_PLUGINS"] = "evo2"
        return
    names = {name.strip() for name in configured.split(",") if name.strip()}
    if "evo2" not in names:
        raise RuntimeError("VLLM_PLUGINS must include the installed 'evo2' entry-point name")


def validate_optional_rl_load_parity(
    *,
    checkpoint: str | Path | None,
    export: str | Path,
    tokenizer_json: str | Path | None,
) -> dict[str, Any] | None:
    """Bind standalone inference to an RL checkpoint when caller authority is provided."""
    if (checkpoint is None) != (tokenizer_json is None):
        raise ValueError("RL checkpoint and tokenizer must be provided together")
    if checkpoint is None:
        return None
    if tokenizer_json is None:
        raise RuntimeError("RL tokenizer authority is unexpectedly missing")

    from bionemo.evo2.vllm.load_parity import validate_rl_inference_load_parity

    return validate_rl_inference_load_parity(
        checkpoint=checkpoint,
        export=export,
        rl_tokenizer=tokenizer_json,
    )


def run_inference(
    *,
    model: str | Path,
    tokenizer_json: str | Path | None,
    requests: Sequence[InferenceRequest],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    base_seed: int,
    tensor_parallel_size: int,
    batch_size: int,
    max_model_len: int | None,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    optimization_level: int,
    performance_mode: str,
    async_scheduling: bool,
    rl_checkpoint: str | Path | None = None,
    rl_tokenizer_json: str | Path | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Run ordered public vLLM generation and return strict rows plus provenance."""
    _require_builtin_int(base_seed, label="base_seed", minimum=0)
    request_seeds = tuple(
        request.validation_seed if request.validation_seed is not None else base_seed + index
        for index, request in enumerate(requests)
    )
    if len(set(request_seeds)) != len(request_seeds):
        raise ValueError("inference request seeds must be unique")
    rl_load_parity = validate_optional_rl_load_parity(
        checkpoint=rl_checkpoint,
        export=model,
        tokenizer_json=rl_tokenizer_json,
    )
    identity = load_export_identity(model)
    if identity.tensor_parallel_divisor % tensor_parallel_size:
        raise ValueError(
            f"tensor_parallel_size={tensor_parallel_size} must divide "
            f"{identity.tensor_parallel_divisor} attention heads"
        )
    tokenizer_path = resolve_tokenizer_json(
        export_root=identity.root,
        tokenizer_json=tokenizer_json,
    )
    tokenizer = SnapshotBoundTokenizer.from_path(tokenizer_path)
    prompt_ids = tuple(tokenizer.encode(request.prompt) for request in requests)
    required_model_len = max(len(token_ids) for token_ids in prompt_ids) + max_new_tokens
    if max_model_len is None:
        resolved_model_len = required_model_len
    else:
        resolved_model_len = _require_builtin_int(max_model_len, label="max_model_len")
        if resolved_model_len < required_model_len:
            raise ValueError(
                f"max_model_len={resolved_model_len} does not cover the required {required_model_len} tokens"
            )
    _require_builtin_int(batch_size, label="batch_size")
    resolved_batch_size = min(batch_size, len(requests))
    tail_size = len(requests) % resolved_batch_size
    extra_capture_sizes = () if tail_size == 0 else (tail_size,)
    engine_kwargs = build_engine_kwargs(
        model=str(identity.root),
        tensor_parallel_size=tensor_parallel_size,
        batch_size=resolved_batch_size,
        max_model_len=resolved_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        optimization_level=optimization_level,
        performance_mode=performance_mode,
        async_scheduling=async_scheduling,
        additional_capture_sizes=extra_capture_sizes,
    )
    _ensure_evo2_plugin()
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm = LLM(**engine_kwargs)
    initialized = time.perf_counter()
    records: list[dict[str, Any]] = []
    for wave_start in range(0, len(requests), resolved_batch_size):
        wave_end = min(wave_start + resolved_batch_size, len(requests))
        wave_requests = tuple(requests[wave_start:wave_end])
        wave_prompt_ids = prompt_ids[wave_start:wave_end]
        wave_seeds = request_seeds[wave_start:wave_end]
        sampling_params = [
            SamplingParams(
                **build_sampling_params_kwargs(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                )
            )
            for seed in wave_seeds
        ]
        outputs = llm.generate(
            [{"prompt_token_ids": list(token_ids)} for token_ids in wave_prompt_ids],
            sampling_params,
            use_tqdm=True,
        )
        records.extend(
            records_from_public_outputs(
                requests=wave_requests,
                prompt_token_ids=wave_prompt_ids,
                request_seeds=wave_seeds,
                outputs=outputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
        )
    completed = time.perf_counter()
    tokenizer_provenance = tokenizer.verify_source()
    manifest = {
        "schema_version": 1,
        "backend": "vllm",
        "export": identity.to_dict(),
        "rl_load_parity": rl_load_parity,
        "tokenizer": tokenizer_provenance,
        "request_count": len(requests),
        "batch_size": resolved_batch_size,
        "request_seeds": list(request_seeds),
        "engine_kwargs": engine_kwargs,
        "engine_init_wall_seconds": initialized - started,
        "generation_wall_seconds": completed - initialized,
        "end_to_end_wall_seconds": completed - started,
    }
    return tuple(records), manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate DNA with Evo2 through the qualified vLLM path",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "--ckpt-dir", dest="model", type=Path, required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--tokenizer-json", type=Path, default=None)
    parser.add_argument("--rl-checkpoint", type=Path, default=None)
    parser.add_argument("--rl-tokenizer-json", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", default="auto")
    parser.add_argument("--batch-size", "--prompt-batch-size", dest="batch_size", type=int, default=96)
    parser.add_argument("--max-model-len", "--max-seq-length", dest="max_model_len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16_384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.91)
    parser.add_argument("--optimization-level", type=int, choices=(2, 3), default=2)
    parser.add_argument("--performance-mode", choices=("balanced", "throughput"), default="balanced")
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--run-manifest-file", type=Path, default=None)
    parser.add_argument(
        "--return-log-probs",
        action="store_true",
        help="Accepted for legacy CLI compatibility; chosen logprobs are always retained",
    )
    return parser


def main() -> None:
    """Run the vLLM-backed ``infer_evo2`` command."""
    args = _parser().parse_args()
    require_vllm_runtime()
    requests = load_prompt_requests(prompt=args.prompt, prompt_file=args.prompt_file)
    tensor_parallel_size = resolve_tensor_parallel_size(args.tensor_parallel_size)
    records, manifest = run_inference(
        model=args.model,
        tokenizer_json=args.tokenizer_json,
        requests=requests,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        base_seed=args.seed,
        tensor_parallel_size=tensor_parallel_size,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        optimization_level=args.optimization_level,
        performance_mode=args.performance_mode,
        async_scheduling=args.async_scheduling,
        rl_checkpoint=args.rl_checkpoint,
        rl_tokenizer_json=args.rl_tokenizer_json,
    )
    output = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    if args.output_file is None:
        print(output, end="")
    else:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(output, encoding="utf-8")
    manifest_path = args.run_manifest_file
    if manifest_path is None and args.output_file is not None:
        manifest_path = args.output_file.with_suffix(args.output_file.suffix + ".manifest.json")
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "ExportIdentity",
    "InferenceRequest",
    "build_engine_kwargs",
    "build_sampling_params_kwargs",
    "load_export_identity",
    "load_prompt_requests",
    "main",
    "records_from_public_outputs",
    "resolve_tokenizer_json",
    "require_vllm_runtime",
    "resolve_tensor_parallel_size",
    "run_inference",
    "validate_optional_rl_load_parity",
]
