# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evo2-specific generation helpers for NeMo-RL's Megatron worker."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch


logger = logging.getLogger(__name__)


def resume_generation_call_offset(completed_steps: int, *, val_period: int, val_at_start: bool) -> int:
    """Count generation calls completed before resuming a checkpointed RL step."""
    periodic_validations = completed_steps // val_period if val_period > 0 else 0
    initial_validation = int(val_at_start and completed_steps > 0)
    return completed_steps + periodic_validations + initial_validation


def _evo2_batched_decode_size(cfg: dict[str, Any]) -> int:
    generation = cfg.get("generation", {}) or {}
    mcore_generation_config = generation.get("mcore_generation_config", {}) or {}
    return int(
        mcore_generation_config.get("prompt_batch_size")
        or mcore_generation_config.get("evo2_batched_decode_size")
        or 1
    )


def _evo2_native_batched_decode_size(worker: Any) -> int:
    """Return native request capacity rounded up for MCore tensor parallelism."""
    configured_size = max(1, _evo2_batched_decode_size(worker.cfg))
    megatron_cfg = worker.cfg.get("megatron_cfg", {}) or {}
    tensor_parallel_size = int(megatron_cfg.get("tensor_model_parallel_size") or 1)
    if tensor_parallel_size < 1:
        raise ValueError(
            f"tensor_model_parallel_size must be positive for Evo2 native generation, got {tensor_parallel_size}"
        )
    return ((configured_size + tensor_parallel_size - 1) // tensor_parallel_size) * tensor_parallel_size


def _reseed_evo2_native_dynamic(native_dynamic: Any, seed: int) -> None:
    """Apply an adapter-call seed even when native inference components are cached."""
    native_dynamic.evo2_seed = int(seed)
    native_dynamic.sampling_rng = None


def _unwrap_evo2_model(model: Any) -> Any:
    """Unwrap Megatron/DDP/fp16 wrappers until the Evo2 model is visible."""
    current = model
    seen: set[int] = set()
    while hasattr(current, "module") and id(current) not in seen:
        seen.add(id(current))
        next_model = getattr(current, "module")
        if next_model is current:
            break
        current = next_model
    return current


def should_use_evo2_native_batched_generation(cfg: dict[str, Any], model: Any, batch_size: int) -> bool:
    """Return whether NeMo-RL should bypass MCore's generic coordinator for Evo2."""
    generation = cfg.get("generation", {}) or {}
    mcore_generation_config = generation.get("mcore_generation_config", {}) or {}
    adapter_path = str(mcore_generation_config.get("generation_adapter") or "")
    enabled = bool(mcore_generation_config.get("evo2_native_batched_generation", False)) or adapter_path.endswith(
        ":Evo2MegatronGenerationAdapter"
    )
    if not enabled:
        return False
    if batch_size < 1 or _evo2_batched_decode_size(cfg) <= 1:
        return False
    raw_model = _unwrap_evo2_model(model)
    return hasattr(getattr(raw_model, "decoder", raw_model), "hyena_state_shapes_per_request")


@dataclass
class Evo2GenerationResult:
    """Minimal result object consumed by NeMo-RL's generation output parser."""

    prompt_tokens: torch.Tensor
    generated_tokens: list[int]
    generated_log_probs: list[float]
    finish_reason: str = "length"
    stopped_on_eos: bool = False
    truncated: bool = False
    timings: dict[str, Any] | None = None
    memory: dict[str, int] | None = None


class _PromptTokenProxy:
    """Tokenizer proxy that preserves already-tokenized NeMo-RL prompts."""

    def __init__(self, tokenizer: Any, prompt_token_ids: list[list[int]]):
        self._tokenizer = tokenizer
        self.prompts = [f"__nemo_rl_evo2_prompt_{idx}__" for idx in range(len(prompt_token_ids))]
        self._prompt_tokens = dict(zip(self.prompts, prompt_token_ids, strict=True))

    def tokenize(self, text: str) -> list[int]:
        if text in self._prompt_tokens:
            return list(self._prompt_tokens[text])
        return list(self._tokenizer.tokenize(text))

    def detokenize(self, token_ids: list[int]) -> str:
        return self._tokenizer.detokenize(token_ids)

    def __getattr__(self, name: str) -> Any:
        if name == "_tokenizer":
            raise AttributeError(name)
        return getattr(self._tokenizer, name)


def _sampling_value(sampling_params: Any, name: str, default: Any) -> Any:
    return getattr(sampling_params, name, default)


def _required_sampling_value(sampling_params: Any, name: str) -> Any:
    if not hasattr(sampling_params, name):
        raise ValueError(f"Evo2 native batched generation requires sampling_params.{name}")
    return getattr(sampling_params, name)


def _batched_sampling_value(sampling_params: list[Any], name: str, *, required: bool, default: Any = None) -> Any:
    first_value = (
        _required_sampling_value(sampling_params[0], name)
        if required
        else _sampling_value(sampling_params[0], name, default)
    )
    for idx, params in enumerate(sampling_params[1:], start=1):
        value = _required_sampling_value(params, name) if required else _sampling_value(params, name, default)
        if value != first_value:
            raise ValueError(
                "Evo2 native batched generation requires homogeneous sampling params; "
                f"{name} differs at batch index {idx}: {value!r} != {first_value!r}"
            )
    return first_value


def _native_generated_token_ids(result: Any) -> list[int]:
    """Return native generated token IDs without detokenize/tokenize replay."""
    generated_tokens = getattr(result, "generated_tokens", None)
    if generated_tokens is None:
        generated_tokens = getattr(result, "generated_token_ids", None)
    if generated_tokens is None:
        raise ValueError("Evo2 native generation result did not include generated token IDs")
    return [int(token_id) for token_id in generated_tokens]


def generate_evo2_native_batched(
    worker: Any,
    prompt_tokens_tensor: torch.Tensor,
    prompt_lengths_tensor: torch.Tensor,
    sampling_params: list[Any],
    *,
    evo2_seed: int | None = None,
    ignore_eos: bool = False,
    strict_generation: bool = False,
) -> list[Evo2GenerationResult]:
    """Generate Evo2 completions with the standalone batched dynamic-decode lifecycle."""
    from megatron.core.utils import unwrap_model

    from bionemo.evo2.run.infer import (
        Evo2InferenceComponents,
        _generate_native_dynamic,
        _setup_native_dynamic_components,
    )

    if not sampling_params:
        return []

    mcore_generation_config = worker.cfg["generation"]["mcore_generation_config"]
    batched_decode_size = _evo2_native_batched_decode_size(worker)
    initial_seed = int(
        evo2_seed if evo2_seed is not None else mcore_generation_config.get("seed", torch.initial_seed() % (2**31))
    )
    prompt_token_ids = [
        row[: int(prompt_len.item())].detach().cpu().tolist()
        for row, prompt_len in zip(prompt_tokens_tensor, prompt_lengths_tensor, strict=True)
    ]
    tokenizer = _PromptTokenProxy(worker.megatron_tokenizer, prompt_token_ids)

    native_dynamic = getattr(worker, "_evo2_native_dynamic_components", None)
    if native_dynamic is None:
        raw_model = _unwrap_evo2_model(unwrap_model(worker.model))
        native_dynamic = _setup_native_dynamic_components(
            model=raw_model,
            raw_model=raw_model,
            max_seq_length=int(mcore_generation_config["max_model_len"]),
            evo2_seed=initial_seed,
            cuda_graphs_enabled=mcore_generation_config.get("cuda_graph_impl") != "none",
        )
        worker._evo2_native_dynamic_components = native_dynamic
    _reseed_evo2_native_dynamic(native_dynamic, initial_seed)

    components = Evo2InferenceComponents(
        tokenizer=tokenizer,
        model=native_dynamic.forward_model,
        native_dynamic=native_dynamic,
    )
    max_new_tokens = int(_batched_sampling_value(sampling_params, "num_tokens_to_generate", required=True))
    temperature = float(_batched_sampling_value(sampling_params, "temperature", required=False, default=1.0))
    top_k = int(_batched_sampling_value(sampling_params, "top_k", required=False, default=0))
    top_p = float(_batched_sampling_value(sampling_params, "top_p", required=False, default=0.0))
    native_results = _generate_native_dynamic(
        components,
        tokenizer.prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_log_probs=True,
        ignore_eos=ignore_eos,
        strict_generation=strict_generation,
        enable_chunked_prefill=bool(mcore_generation_config.get("enable_chunked_prefill", False))
        and batched_decode_size <= 1,
        inference_dynamic_batching_max_tokens=mcore_generation_config.get("max_tokens"),
        inference_dynamic_batching_block_size=int(mcore_generation_config.get("block_size_tokens", 256)),
        evo2_batched_decode_size=batched_decode_size,
        result_callback=None,
    )

    results = []
    for result in native_results:
        generated_tokens = _native_generated_token_ids(result)
        generated_log_probs = list(result.generated_log_probs or [])
        if len(generated_tokens) != len(generated_log_probs):
            raise ValueError(
                "Evo2 native generation returned mismatched token/log-prob lengths: "
                f"{len(generated_tokens)} tokens != {len(generated_log_probs)} log-probs"
            )
        results.append(
            Evo2GenerationResult(
                prompt_tokens=torch.tensor(
                    result.prompt_tokens,
                    dtype=torch.long,
                    device=prompt_tokens_tensor.device,
                ),
                generated_tokens=generated_tokens,
                generated_log_probs=generated_log_probs,
                finish_reason=str(getattr(result, "finish_reason", "length")),
                stopped_on_eos=bool(getattr(result, "stopped_on_eos", False)),
                truncated=bool(getattr(result, "truncated", False)),
                timings=getattr(result, "timings", None),
                memory=getattr(result, "memory", None),
            )
        )
    return results


class Evo2MegatronGenerationAdapter:
    """Recipe-owned adapter for DP-sharded, model-parallel Evo2 generation."""

    requires_all_workers = True

    def __init__(self, config: dict[str, Any] | None = None):
        """Create an adapter from NeMo-RL generation adapter config."""
        self.config = dict(config or {})
        self.seed_stride = int(self.config.get("seed_stride", 1_000_003))
        self.call_index_offset = int(self.config.get("call_index_offset", 0))

    def _distributed_rank(self, worker: Any) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return int(getattr(worker, "rank", 0))

    def _data_parallel_coordinates(self, worker: Any) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            from megatron.core import parallel_state

            return (
                int(parallel_state.get_data_parallel_rank()),
                int(parallel_state.get_data_parallel_world_size()),
            )
        return (
            int(getattr(worker, "data_parallel_rank", 0)),
            int(getattr(worker, "dp_size", 1)),
        )

    def _tensor_parallel_coordinates(self, worker: Any) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            from megatron.core import parallel_state

            return (
                int(parallel_state.get_tensor_model_parallel_rank()),
                int(parallel_state.get_tensor_model_parallel_world_size()),
            )
        return (
            int(getattr(worker, "tensor_parallel_rank", 0)),
            int(getattr(worker, "tp_size", 1)),
        )

    def _is_model_parallel_leader(self) -> bool:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return True

        from megatron.core import parallel_state

        return (
            int(parallel_state.get_tensor_model_parallel_rank()) == 0
            and int(parallel_state.get_pipeline_model_parallel_rank()) == 0
            and int(parallel_state.get_context_parallel_rank()) == 0
        )

    @staticmethod
    def _broadcast_seed_in_group(candidate_seed: int, *, group: Any, src: int) -> int:
        """Broadcast one seed using a tensor compatible with the group's backend."""
        backend = str(torch.distributed.get_backend(group)).lower()
        if backend.endswith("nccl"):
            seed_device = torch.device("cuda", torch.cuda.current_device())
        else:
            seed_device = torch.device("cpu")
        seed_tensor = torch.tensor([int(candidate_seed)], dtype=torch.int64, device=seed_device)
        torch.distributed.broadcast(
            seed_tensor,
            src=int(src),
            group=group,
        )
        return int(seed_tensor.item())

    def _shared_implicit_base_seed(self, candidate_seed: int) -> int:
        """Broadcast an unconfigured base seed across this DP replica's MP and CP dimensions."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return int(candidate_seed)

        from megatron.core import parallel_state

        shared_seed = self._broadcast_seed_in_group(
            candidate_seed,
            group=parallel_state.get_model_parallel_group(),
            src=int(parallel_state.get_model_parallel_src_rank()),
        )
        if int(parallel_state.get_context_parallel_world_size()) > 1:
            shared_seed = self._broadcast_seed_in_group(
                shared_seed,
                group=parallel_state.get_context_parallel_group(),
                src=int(parallel_state.get_context_parallel_global_ranks()[0]),
            )
        return shared_seed

    def _next_seed(self, worker: Any) -> int:
        mcore_generation_config = worker.cfg["generation"].get("mcore_generation_config", {}) or {}
        configured_seed = self.config.get("seed")
        if configured_seed is None:
            configured_seed = mcore_generation_config.get("seed")
        if configured_seed is None:
            base_seed = self._shared_implicit_base_seed(torch.initial_seed() % (2**31))
        else:
            base_seed = int(configured_seed)
        call_index = int(getattr(worker, "_evo2_generation_call_index", self.call_index_offset))
        setattr(worker, "_evo2_generation_call_index", call_index + 1)
        data_parallel_rank, data_parallel_size = self._data_parallel_coordinates(worker)
        if data_parallel_size < 1 or not 0 <= data_parallel_rank < data_parallel_size:
            raise ValueError(
                "Invalid Evo2 generation data-parallel coordinates: "
                f"rank={data_parallel_rank}, size={data_parallel_size}"
            )
        tensor_parallel_rank, tensor_parallel_size = self._tensor_parallel_coordinates(worker)
        if tensor_parallel_size < 1 or not 0 <= tensor_parallel_rank < tensor_parallel_size:
            raise ValueError(
                "Invalid Evo2 generation tensor-parallel coordinates: "
                f"rank={tensor_parallel_rank}, size={tensor_parallel_size}"
            )
        seed_index = call_index * data_parallel_size + data_parallel_rank
        seed = int((base_seed + seed_index * self.seed_stride) % (2**31))

        trace = getattr(worker, "_evo2_generation_rng_trace", [])
        trace_item = {
            "rank": self._distributed_rank(worker),
            "data_parallel_rank": data_parallel_rank,
            "data_parallel_size": data_parallel_size,
            "tensor_parallel_rank": tensor_parallel_rank,
            "tensor_parallel_size": tensor_parallel_size,
            "call_index": call_index,
            "seed_index": seed_index,
            "seed": seed,
            "base_seed": base_seed,
            "seed_stride": self.seed_stride,
        }
        trace.append(trace_item)
        setattr(worker, "_evo2_generation_rng_trace", trace[-100:])
        if self._is_model_parallel_leader():
            logger.info("EVO2_SEED_TRACE %s", json.dumps(trace_item, sort_keys=True))
        return seed

    def generate_worker(
        self,
        worker: Any,
        *,
        data: Any,
        greedy: bool = False,
    ) -> Any | None:
        """Generate a replicated DP shard on every model-parallel worker."""
        adapter_begin_unix_s = time.time()
        adapter_start = time.perf_counter()
        prompt_tokens_tensor, prompt_lengths_tensor, sampling_params = worker._prepare_data_for_generation(
            data, greedy
        )
        if not should_use_evo2_native_batched_generation(
            worker.cfg, worker.model, batch_size=prompt_tokens_tensor.size(0)
        ):
            raise RuntimeError(
                "Evo2MegatronGenerationAdapter is configured, but this worker/model is not eligible for native batched Evo2 generation."
            )

        seed = self._next_seed(worker)
        data_parallel_rank, data_parallel_size = self._data_parallel_coordinates(worker)
        native_begin_unix_s = time.time()
        native_start = time.perf_counter()
        result: list[Evo2GenerationResult] | None = None
        try:
            result = generate_evo2_native_batched(
                worker,
                prompt_tokens_tensor,
                prompt_lengths_tensor,
                sampling_params,
                evo2_seed=seed,
                ignore_eos=bool(self.config.get("ignore_eos", False)),
                strict_generation=bool(self.config.get("strict_generation", False)),
            )
            expected_results = int(prompt_tokens_tensor.size(0))
            if len(result) != expected_results:
                raise RuntimeError(
                    "Evo2 native batched generation returned "
                    f"{len(result)} results for {expected_results} prompts on "
                    f"data-parallel rank {data_parallel_rank}"
                )
        finally:
            native_end_unix_s = time.time()
            timing = {
                "timing/train/generation/evo2_native_begin_unix_s": native_begin_unix_s,
                "timing/train/generation/evo2_native_end_unix_s": native_end_unix_s,
                "timing/train/generation/evo2_native_elapsed_s": time.perf_counter() - native_start,
                "timing/train/generation/evo2_adapter_begin_unix_s": adapter_begin_unix_s,
                "timing/train/generation/evo2_adapter_end_unix_s": time.time(),
                "timing/train/generation/evo2_adapter_elapsed_s": time.perf_counter() - adapter_start,
                "timing/train/generation/evo2_native_batch_size": float(prompt_tokens_tensor.size(0)),
                "timing/train/generation/evo2_data_parallel_rank": float(data_parallel_rank),
                "timing/train/generation/evo2_data_parallel_size": float(data_parallel_size),
            }
            if result is not None:
                phase_totals = {
                    "engine_setup_elapsed_s": 0.0,
                    "context_setup_elapsed_s": 0.0,
                    "cuda_graph_capture_elapsed_s": 0.0,
                    "prefill_elapsed_s": 0.0,
                    "decode_elapsed_s": 0.0,
                    "generation_elapsed_s": 0.0,
                    "total_elapsed_s": 0.0,
                }
                memory_peak_maxima = {
                    f"{phase_name}_peak_{memory_kind}_bytes": 0
                    for phase_name in (
                        "engine_setup",
                        "context_setup",
                        "cuda_graph_capture",
                        "prefill",
                        "decode",
                        "generation",
                        "total",
                    )
                    for memory_kind in ("allocated", "reserved")
                }
                seen_evidence_groups: set[tuple[str, str]] = set()
                for generation_result in result:
                    result_timings = generation_result.timings
                    result_memory = generation_result.memory
                    if result_timings is None and result_memory is None:
                        continue
                    timing_group_id = result_timings.get("timing_group_id") if result_timings is not None else None
                    if timing_group_id is not None:
                        evidence_group_key = (
                            str(result_timings.get("timing_scope", "native_generation_group")),
                            str(timing_group_id),
                        )
                    elif result_timings is not None:
                        evidence_group_key = ("legacy_timing_object", str(id(result_timings)))
                    else:
                        evidence_group_key = ("legacy_memory_object", str(id(result_memory)))
                    if evidence_group_key in seen_evidence_groups:
                        continue
                    seen_evidence_groups.add(evidence_group_key)
                    if result_timings is not None:
                        for phase_name in phase_totals:
                            phase_totals[phase_name] += float(result_timings.get(phase_name, 0.0))
                    if result_memory is not None:
                        for memory_name in memory_peak_maxima:
                            memory_peak_maxima[memory_name] = max(
                                memory_peak_maxima[memory_name],
                                int(result_memory.get(memory_name, 0)),
                            )
                timing.update(
                    {
                        "timing/train/generation/evo2_engine_setup_elapsed_s": phase_totals["engine_setup_elapsed_s"],
                        "timing/train/generation/evo2_context_setup_elapsed_s": phase_totals[
                            "context_setup_elapsed_s"
                        ],
                        "timing/train/generation/evo2_cuda_graph_capture_elapsed_s": phase_totals[
                            "cuda_graph_capture_elapsed_s"
                        ],
                        "timing/train/generation/evo2_prefill_elapsed_s": phase_totals["prefill_elapsed_s"],
                        "timing/train/generation/evo2_decode_elapsed_s": phase_totals["decode_elapsed_s"],
                        "timing/train/generation/evo2_generation_elapsed_s": phase_totals["generation_elapsed_s"],
                        "timing/train/generation/evo2_total_elapsed_s": phase_totals["total_elapsed_s"],
                    }
                )
                timing.update(
                    {
                        f"memory/train/generation/evo2_{memory_name}": value
                        for memory_name, value in memory_peak_maxima.items()
                    }
                )
            setattr(worker, "_evo2_generation_timing", timing)
            if self._is_model_parallel_leader():
                logger.info("%s", " ".join(f"{key}={value:.6f}" for key, value in timing.items()))
        return worker._parse_result_to_batched_data_dict(data, result)

    def finish_worker(self, worker: Any) -> None:
        """Release native Evo2 generation state before logprob/training phases."""
        if hasattr(worker, "_evo2_native_dynamic_components"):
            delattr(worker, "_evo2_native_dynamic_components")
        torch.cuda.empty_cache()
