# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Out-of-tree vLLM model implementation for Evo2 Vortex checkpoints."""

import hashlib
import json
from collections.abc import Iterable
from copy import deepcopy
from itertools import islice
from typing import Any

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaCopySpec, MambaStateCopyFunc
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory, make_layers, maybe_prefix
from vllm.sequence import IntermediateTensors

from bionemo.evo2.vllm.config import Evo2Config
from bionemo.evo2.vllm.hyena import Evo2HyenaDecoderLayer
from bionemo.evo2.vllm.layers import Evo2AttentionDecoderLayer
from bionemo.evo2.vllm.weights import IncrementalEvo2WeightLoader


_MAMBA_STATE_COPY_STATS = {
    "copy_calls": 0,
    "copied_elements": 0,
    "copied_bytes": 0,
}

_MAMBA_PREFIX_CLONE_RECORDS: list[dict[str, Any]] = []
_MAMBA_PREFIX_CACHE_MISS_IDS: list[str] = []
_MAMBA_PREFIX_SOURCE_RECORDS: dict[str, dict[str, Any]] = {}
_MAMBA_PREFIX_CLONE_STATE: dict[str, dict[str, Any] | None] = {"active": None}


def reset_mamba_state_copy_stats() -> None:
    """Reset phase-local physical state-copy telemetry."""
    for key in _MAMBA_STATE_COPY_STATS:
        _MAMBA_STATE_COPY_STATS[key] = 0


def get_mamba_state_copy_stats() -> dict[str, int]:
    """Return physical whole-state copies queued through vLLM's align manager."""
    return dict(_MAMBA_STATE_COPY_STATS)


def reset_mamba_prefix_clone_stats(*, reset_prefix_sources: bool = True) -> None:
    """Reset phase-local request-scoped prefix clone telemetry."""
    if _MAMBA_PREFIX_CLONE_STATE["active"] is not None:
        raise RuntimeError("cannot reset prefix clone telemetry during Mamba preprocessing")
    _MAMBA_PREFIX_CLONE_RECORDS.clear()
    _MAMBA_PREFIX_CACHE_MISS_IDS.clear()
    if reset_prefix_sources:
        _MAMBA_PREFIX_SOURCE_RECORDS.clear()


def get_mamba_prefix_clone_stats() -> dict[str, Any]:
    """Return exact per-request recurrent-state clones caused by prefix hits."""
    return {
        "cache_miss_count": len(_MAMBA_PREFIX_CACHE_MISS_IDS),
        "cache_miss_request_ids": list(_MAMBA_PREFIX_CACHE_MISS_IDS),
        "prefix_sources": deepcopy(list(_MAMBA_PREFIX_SOURCE_RECORDS.values())),
        "clone_count": len(_MAMBA_PREFIX_CLONE_RECORDS),
        "requests": deepcopy(_MAMBA_PREFIX_CLONE_RECORDS),
    }


def _physical_block_ids_sha256(block_ids: list[int]) -> str:
    payload = json.dumps(block_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _attention_prefix_block_groups(
    kv_cache_config: Any,
    mamba_group_ids: list[int],
    req_state: Any,
    *,
    prefix_tokens: int,
    expected_block_size: int,
) -> list[dict[str, Any]]:
    if prefix_tokens <= 0 or prefix_tokens % expected_block_size:
        raise AssertionError("attention KV proof requires a positive block-aligned prefix")
    if len(req_state.block_ids) != len(kv_cache_config.kv_cache_groups):
        raise AssertionError("request block tables must align with every KV cache group")
    mamba_groups = set(mamba_group_ids)
    retained = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        if group_id in mamba_groups:
            continue
        block_size = int(group.kv_cache_spec.block_size)
        if block_size != expected_block_size:
            raise AssertionError("Evo2 prefix proof requires matching Mamba and attention block sizes")
        physical_block_count = prefix_tokens // block_size
        physical_block_ids = [int(value) for value in req_state.block_ids[group_id][:physical_block_count]]
        if len(physical_block_ids) != physical_block_count:
            raise AssertionError("attention KV block table does not cover the full cached prefix")
        if len(set(physical_block_ids)) != len(physical_block_ids):
            raise AssertionError("attention KV cached-prefix physical block IDs must be unique")
        layer_names = [str(name) for name in group.layer_names]
        if not layer_names:
            raise AssertionError("attention KV cache group must own at least one layer")
        retained.append(
            {
                "kv_cache_group_id": group_id,
                "layer_names": layer_names,
                "block_size_tokens": block_size,
                "physical_block_count": physical_block_count,
                "physical_block_ids": physical_block_ids,
                "physical_block_ids_sha256": _physical_block_ids_sha256(physical_block_ids),
            }
        )
    if not retained:
        raise AssertionError("Evo2 shared-prefix proof found no attention KV cache groups")
    return retained


def _record_prefix_source_snapshot(
    scheduler_output: Any,
    kv_cache_config: Any,
    mamba_group_ids: list[int],
    req_state: Any,
    *,
    block_size: int,
) -> None:
    request_id = str(req_state.req_id)
    source = _MAMBA_PREFIX_SOURCE_RECORDS[request_id]
    if source["prompt_tokens"] != int(req_state.num_prompt_tokens):
        raise AssertionError("cache-miss source prompt length changed while retaining prefix evidence")
    num_scheduled_tokens = int(scheduler_output.num_scheduled_tokens[req_state.req_id])
    observed_tokens = min(
        int(req_state.num_computed_tokens) + num_scheduled_tokens,
        int(req_state.num_prompt_tokens) - 1,
    )
    directly_observed_prefix_tokens = observed_tokens // block_size * block_size
    if directly_observed_prefix_tokens <= 0:
        return
    groups = _attention_prefix_block_groups(
        kv_cache_config,
        mamba_group_ids,
        req_state,
        prefix_tokens=directly_observed_prefix_tokens,
        expected_block_size=block_size,
    )
    snapshots = source["snapshots"]
    if snapshots:
        previous_groups = snapshots[-1]["attention_kv_groups"]
        if len(previous_groups) != len(groups):
            raise AssertionError("cache-miss source attention KV group count changed")
        for old, new in zip(previous_groups, groups, strict=True):
            for key in ("kv_cache_group_id", "layer_names", "block_size_tokens"):
                if old[key] != new[key]:
                    raise AssertionError("cache-miss source attention KV group or layer ownership changed")
            old_ids = old["physical_block_ids"]
            if old_ids != new["physical_block_ids"][: len(old_ids)]:
                raise AssertionError("cache-miss source attention KV physical blocks changed during prefill")
        if groups == previous_groups:
            return
    snapshots.append(
        {
            "snapshot_index": len(snapshots),
            "num_computed_tokens_before_step": int(req_state.num_computed_tokens),
            "num_scheduled_tokens": num_scheduled_tokens,
            "directly_observed_prefix_tokens": directly_observed_prefix_tokens,
            "attention_kv_groups": groups,
        }
    )


def _verified_prefix_source_for_hit(
    kv_cache_config: Any,
    mamba_group_ids: list[int],
    req_state: Any,
    *,
    block_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sources = [record for record in _MAMBA_PREFIX_SOURCE_RECORDS.values() if record["snapshots"]]
    if len(sources) != 1:
        raise AssertionError("prefix reuse proof requires exactly one directly observed cache-miss source")
    source = sources[0]
    if source["prompt_tokens"] != int(req_state.num_prompt_tokens):
        raise AssertionError("cache hit prompt length does not match its directly observed source")
    source_snapshot = source["snapshots"][-1]
    if source_snapshot["directly_observed_prefix_tokens"] != int(req_state.num_computed_tokens):
        raise AssertionError("cache hit token count does not match the direct cache-miss source snapshot")
    reused_groups = _attention_prefix_block_groups(
        kv_cache_config,
        mamba_group_ids,
        req_state,
        prefix_tokens=int(req_state.num_computed_tokens),
        expected_block_size=block_size,
    )
    source_groups = source_snapshot["attention_kv_groups"]
    if len(source_groups) != len(reused_groups):
        raise AssertionError("cache hit attention KV group count does not match its source")
    for source_group, reused_group in zip(source_groups, reused_groups, strict=True):
        for key in (
            "kv_cache_group_id",
            "layer_names",
            "block_size_tokens",
            "physical_block_count",
        ):
            if source_group[key] != reused_group[key]:
                raise AssertionError("cache hit attention KV group, layer, or block count does not match")
        if source_group["physical_block_ids"] != reused_group["physical_block_ids"]:
            raise AssertionError("cache hit attention KV physical block IDs do not exactly match its source")
    return source, reused_groups


def _mamba_copy_entry_specs(
    kv_cache_config: Any,
    mamba_group_ids: list[int],
    req_state: Any,
    forward_context: dict[str, Any],
    *,
    src_block_idx: int,
    dest_block_idx: int,
) -> list[dict[str, Any]]:
    entries = []
    for group_id in mamba_group_ids:
        block_ids = req_state.block_ids[group_id]
        if not 0 <= src_block_idx < len(block_ids) or not 0 <= dest_block_idx < len(block_ids):
            raise AssertionError("Mamba clone logical block index is outside the request block table")
        source_block_id = int(block_ids[src_block_idx])
        destination_block_id = int(block_ids[dest_block_idx])
        if source_block_id == destination_block_id:
            raise AssertionError("Mamba prefix clone source and destination blocks must differ")
        group = kv_cache_config.kv_cache_groups[group_id]
        for layer_name in group.layer_names:
            states = forward_context[layer_name].kv_cache
            for state_index, state in enumerate(states):
                if not isinstance(state, torch.Tensor) or state.dtype != torch.float32:
                    raise AssertionError("every Evo2 cloned recurrent state must be an FP32 tensor")
                source = state[source_block_id]
                destination = state[destination_block_id]
                entries.append(
                    {
                        "kv_cache_group_id": int(group_id),
                        "layer_name": str(layer_name),
                        "state_index": state_index,
                        "dtype": str(state.dtype),
                        "state_shape": list(state.shape),
                        "block_shape": list(source.shape),
                        "source_logical_block_index": src_block_idx,
                        "destination_logical_block_index": dest_block_idx,
                        "source_physical_block_id": source_block_id,
                        "destination_physical_block_id": destination_block_id,
                        "source_data_ptr": int(source.data_ptr()),
                        "destination_data_ptr": int(destination.data_ptr()),
                        "copied_elements": int(source.numel()),
                        "copied_bytes": int(source.numel() * source.element_size()),
                    }
                )
    return entries


def _expected_mamba_prefix_clone_layout(
    kv_cache_config: Any,
    mamba_group_ids: list[int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[Any, ...],
) -> dict[str, int | bool]:
    expected_entries = 0
    expected_elements = 0
    expected_bytes = 0
    all_state_dtypes_fp32 = True
    for group_id in mamba_group_ids:
        group = kv_cache_config.kv_cache_groups[group_id]
        for layer_name in group.layer_names:
            states = forward_context[layer_name].kv_cache
            if len(states) != len(mamba_state_copy_funcs):
                raise AssertionError("Mamba state tensors and copy functions do not align")
            for state in states:
                if not isinstance(state, torch.Tensor) or state.ndim < 1 or state.shape[0] < 1:
                    raise AssertionError("Mamba prefix clone state must be a nonempty tensor")
                block = state[0]
                expected_entries += 1
                expected_elements += block.numel()
                expected_bytes += block.numel() * block.element_size()
                all_state_dtypes_fp32 &= state.dtype == torch.float32
    if expected_entries == 0:
        raise AssertionError("Mamba prefix clone layout contains no recurrent state")
    if not all_state_dtypes_fp32:
        raise AssertionError("Evo2 prefix clone state must remain FP32")
    return {
        "expected_copy_entries": expected_entries,
        "expected_copied_elements": expected_elements,
        "expected_copied_bytes": expected_bytes,
        "all_state_dtypes_fp32": all_state_dtypes_fp32,
    }


def install_mamba_prefix_clone_proof_hook(mamba_utils_module: Any | None = None) -> None:
    """Install request-scoped telemetry around vLLM's physical align-mode copies."""
    if mamba_utils_module is None:
        from vllm.v1.worker import mamba_utils as mamba_utils_module

    original_preprocess = mamba_utils_module.preprocess_mamba
    if getattr(original_preprocess, "_evo2_prefix_clone_proof_hook", False):
        return
    original_collect = mamba_utils_module.collect_mamba_copy_meta

    def tracked_collect(
        copy_bufs,
        kv_cache_config,
        mamba_state_copy_funcs,
        mamba_group_ids,
        src_block_idx,
        dest_block_idx,
        accept_token_bias,
        req_state,
        forward_context,
    ):
        active = _MAMBA_PREFIX_CLONE_STATE["active"]
        entry_specs = []
        if active is not None and req_state.req_id in active:
            if accept_token_bias != 0:
                raise AssertionError("Evo2 prefix clones must copy one exact recurrent state block")
            entry_specs = _mamba_copy_entry_specs(
                kv_cache_config,
                mamba_group_ids,
                req_state,
                forward_context,
                src_block_idx=src_block_idx,
                dest_block_idx=dest_block_idx,
            )
        before = int(copy_bufs.offset)
        result = original_collect(
            copy_bufs,
            kv_cache_config,
            mamba_state_copy_funcs,
            mamba_group_ids,
            src_block_idx,
            dest_block_idx,
            accept_token_bias,
            req_state,
            forward_context,
        )
        if active is not None and req_state.req_id in active:
            after = int(copy_bufs.offset)
            if after - before != len(entry_specs):
                raise AssertionError("Mamba prefix clone copy entries do not match the physical state layout")
            source_ptrs = [int(value) for value in copy_bufs.src_ptrs.np[before:after]]
            destination_ptrs = [int(value) for value in copy_bufs.dst_ptrs.np[before:after]]
            sizes = [int(value) for value in copy_bufs.sizes.np[before:after]]
            if any(size <= 0 or size % 4 for size in sizes):
                raise AssertionError("FP32 Mamba prefix clone copy sizes must be positive multiples of four")
            record = active[req_state.req_id]
            for entry, source_ptr, destination_ptr, size in zip(
                entry_specs,
                source_ptrs,
                destination_ptrs,
                sizes,
                strict=True,
            ):
                if source_ptr != entry["source_data_ptr"]:
                    raise AssertionError("Mamba prefix clone source pointer does not match its physical block")
                if destination_ptr != entry["destination_data_ptr"]:
                    raise AssertionError("Mamba prefix clone destination pointer does not match its physical block")
                if size != entry["copied_bytes"]:
                    raise AssertionError("Mamba prefix clone byte count does not match its tensor shape")
            record["state_copies"].extend(entry_specs)
            record["copy_entries"] += after - before
            record["copied_bytes"] += sum(sizes)
            record["copied_elements"] += sum(sizes) // 4
        return result

    def tracked_preprocess(
        scheduler_output,
        kv_cache_config,
        cache_config,
        mamba_state_idx,
        input_batch,
        requests,
        forward_context,
        mamba_state_copy_funcs,
        copy_bufs,
    ):
        if _MAMBA_PREFIX_CLONE_STATE["active"] is not None:
            raise RuntimeError("Mamba prefix clone telemetry does not support reentrant preprocessing")

        new_request_ids = {request.req_id for request in scheduler_output.scheduled_new_reqs}
        block_size = int(copy_bufs.mamba_spec.block_size)
        for req_id in input_batch.req_ids:
            if req_id not in new_request_ids or requests[req_id].num_computed_tokens != 0:
                continue
            retained_req_id = str(req_id)
            if retained_req_id not in _MAMBA_PREFIX_CACHE_MISS_IDS:
                _MAMBA_PREFIX_CACHE_MISS_IDS.append(retained_req_id)
            if retained_req_id in _MAMBA_PREFIX_SOURCE_RECORDS:
                raise AssertionError("cache-miss request IDs must remain unique within a prefix-cache epoch")
            _MAMBA_PREFIX_SOURCE_RECORDS[retained_req_id] = {
                "request_id": retained_req_id,
                "prompt_tokens": int(requests[req_id].num_prompt_tokens),
                "snapshots": [],
            }
        for req_id in input_batch.req_ids:
            source = _MAMBA_PREFIX_SOURCE_RECORDS.get(str(req_id))
            if source is None:
                continue
            _record_prefix_source_snapshot(
                scheduler_output,
                kv_cache_config,
                copy_bufs.mamba_group_ids,
                requests[req_id],
                block_size=block_size,
            )
        candidate_ids = [
            req_id
            for req_id in input_batch.req_ids
            if req_id in new_request_ids and requests[req_id].num_computed_tokens > 0
        ]
        active: dict[str, dict[str, Any]] = {}
        if candidate_ids:
            layout = _expected_mamba_prefix_clone_layout(
                kv_cache_config,
                copy_bufs.mamba_group_ids,
                forward_context,
                mamba_state_copy_funcs,
            )
            for req_id in candidate_ids:
                req_state = requests[req_id]
                if req_state.output_token_ids:
                    raise AssertionError("a newly cloned prefix request cannot already have output tokens")
                if req_state.num_computed_tokens >= req_state.num_prompt_tokens:
                    raise AssertionError("vLLM must recompute at least the final prompt token")
                source, reused_groups = _verified_prefix_source_for_hit(
                    kv_cache_config,
                    copy_bufs.mamba_group_ids,
                    req_state,
                    block_size=block_size,
                )
                source_snapshot = source["snapshots"][-1]
                active[req_id] = {
                    "request_id": str(req_id),
                    "source_miss_request_id": source["request_id"],
                    "source_snapshot_index": source_snapshot["snapshot_index"],
                    "attention_kv_identity_verified": True,
                    "num_computed_tokens": int(req_state.num_computed_tokens),
                    "prompt_tokens": int(req_state.num_prompt_tokens),
                    "block_size": block_size,
                    "source_attention_kv_groups": deepcopy(source_snapshot["attention_kv_groups"]),
                    "reused_attention_kv_groups": reused_groups,
                    "state_copies": [],
                    "copy_entries": 0,
                    "copied_elements": 0,
                    "copied_bytes": 0,
                    **layout,
                }

        _MAMBA_PREFIX_CLONE_STATE["active"] = active
        succeeded = False
        try:
            result = original_preprocess(
                scheduler_output,
                kv_cache_config,
                cache_config,
                mamba_state_idx,
                input_batch,
                requests,
                forward_context,
                mamba_state_copy_funcs,
                copy_bufs,
            )
            succeeded = True
            return result
        finally:
            _MAMBA_PREFIX_CLONE_STATE["active"] = None
            if succeeded:
                _MAMBA_PREFIX_CLONE_RECORDS.extend(active[req_id] for req_id in candidate_ids)

    tracked_preprocess._evo2_prefix_clone_proof_hook = True
    mamba_utils_module.collect_mamba_copy_meta = tracked_collect
    mamba_utils_module.preprocess_mamba = tracked_preprocess


def _copy_whole_evo2_state_block(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Describe one contiguous channel-major Evo2 state block copy."""
    if num_accepted_tokens != 1:
        raise ValueError("Evo2 vLLM inference does not support speculative state copies")
    if not 0 <= cur_block_idx < len(block_ids):
        raise IndexError("Evo2 state copy block index is out of range")
    source = state[block_ids[cur_block_idx]]
    elements = source.numel()
    _MAMBA_STATE_COPY_STATS["copy_calls"] += 1
    _MAMBA_STATE_COPY_STATS["copied_elements"] += elements
    _MAMBA_STATE_COPY_STATS["copied_bytes"] += elements * source.element_size()
    return MambaCopySpec(start_addr=source.data_ptr(), num_elements=source.numel())


def evo2_context_length_contract(vllm_config: VllmConfig) -> dict[str, object]:
    """Describe the resolved position and length-independent recurrent-state contract."""
    config = vllm_config.model_config.hf_config
    if not isinstance(config, Evo2Config):
        raise TypeError("Evo2 context length contract requires Evo2Config")
    resolved_max_model_len = int(vllm_config.model_config.max_model_len)
    if resolved_max_model_len <= 0:
        raise ValueError("resolved max_model_len must be positive")
    state_shapes = config.local_state_shapes(vllm_config.parallel_config.tensor_parallel_size)
    return {
        "checkpoint_declared_max_position_embeddings": int(config.max_position_embeddings),
        "resolved_max_model_len": resolved_max_model_len,
        "rotary_max_position_embeddings": resolved_max_model_len,
        "attention_kv_position_limit": resolved_max_model_len,
        "position_source": "vllm_config.model_config.max_model_len",
        "position_clipping": False,
        "hyena_state_shapes": [list(shape) for shape in state_shapes],
        "hyena_state_dtypes": ["float32", "float32"],
        "hyena_state_length_dependent": False,
        "hyena_decode_scratch_request_capacity": int(vllm_config.scheduler_config.max_num_seqs),
        "hyena_prefill_allocation_basis": "num_actual_tokens",
    }


class Evo2Embedding(nn.Module):
    """Vocab-parallel embedding with the native MBridge module path."""

    def __init__(
        self,
        config: Evo2Config,
        *,
        quant_config=None,
        prefix: str = "",
        params_dtype: torch.dtype | None = None,
    ) -> None:
        """Construct the checkpoint-compatible word embedding."""
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.word_embeddings",
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed one packed vLLM token tensor."""
        return self.word_embeddings(input_ids)


class Evo2Decoder(nn.Module):
    """Pipeline-aware hybrid Evo2 decoder with native checkpoint names."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        """Construct the operator-pattern layers and final RMS normalization."""
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, Evo2Config):
            raise TypeError("Evo2ForCausalLM requires Evo2Config")
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        params_dtype = vllm_config.model_config.dtype
        self.context_length_contract = evo2_context_length_contract(vllm_config)
        max_position_embeddings = int(self.context_length_contract["rotary_max_position_embeddings"])
        self.config = config

        def make_layer(prefix: str) -> nn.Module:
            layer_index = int(prefix.rsplit(".", maxsplit=1)[-1])
            operator_type = config.hybrid_override_pattern[layer_index]
            if operator_type == "*":
                return Evo2AttentionDecoderLayer(
                    config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=prefix,
                    max_position_embeddings=max_position_embeddings,
                    params_dtype=params_dtype,
                )
            return Evo2HyenaDecoderLayer(
                config,
                operator_type=operator_type,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                params_dtype=params_dtype,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            make_layer,
            prefix=f"{prefix}.layers",
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | IntermediateTensors:
        """Apply local pipeline layers and the final norm on the last rank."""
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            if residual is None:
                raise RuntimeError("Evo2 pipeline stage produced no residual")
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        if residual is None:
            return self.final_norm(hidden_states)
        normalized, _ = self.final_norm(hidden_states, residual)
        return normalized


@support_torch_compile
class Evo2Model(nn.Module):
    """Evo2 embedding and hybrid decoder compiled as one vLLM graph."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Construct the checkpoint-compatible Evo2 backbone."""
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, Evo2Config):
            raise TypeError("Evo2Model requires Evo2Config")
        self.config = config
        self.hybrid_override_pattern = config.hybrid_override_pattern
        self.embedding = Evo2Embedding(
            config,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embedding"),
            params_dtype=vllm_config.model_config.dtype,
        )
        self.decoder = Evo2Decoder(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "decoder"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"],
            config.hidden_size,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed packed token ids through the vocab-parallel table."""
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Run one packed vLLM scheduler batch through the local pipeline stage."""
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids are required when inputs_embeds are absent")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            if intermediate_tensors is None:
                raise ValueError("non-first Evo2 pipeline ranks require intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        return self.decoder(hidden_states, positions, residual)


class Evo2ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """vLLM causal language model wrapper for Evo2 Vortex checkpoints."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Construct Evo2, tie its LM head, and expose vLLM state interfaces."""
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, Evo2Config):
            raise TypeError("Evo2ForCausalLM requires Evo2Config")
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = Evo2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if not config.tie_word_embeddings:
            raise ValueError("Evo2 vLLM checkpoints require tied input and output embeddings")
        self.lm_head = self.model.embedding.word_embeddings
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self._weight_loader = IncrementalEvo2WeightLoader(self)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input ids through the tied Evo2 table."""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """Return hidden states for vLLM's logits and sampling pipeline."""
        del kwargs
        self._weight_loader.assert_ready_for_inference()
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return length-independent channel-major cache shapes for one TP rank."""
        config = vllm_config.model_config.hf_config
        if not isinstance(config, Evo2Config):
            raise TypeError("Evo2 state shapes require Evo2Config")
        return config.local_state_shapes(vllm_config.parallel_config.tensor_parallel_size)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        """Keep both Evo2 recurrent states in fp32."""
        return (torch.float32, torch.float32)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        """Return whole-block copy functions independent of vLLM's conv-layout setting."""
        return (_copy_whole_evo2_state_block, _copy_whole_evo2_state_block)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        """Delegate graph input copies to vLLM's injected Mamba cache manager."""
        return self.mamba_cache.copy_inputs_before_cuda_graphs(input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """Delegate state capture buffers to vLLM's injected Mamba cache manager."""
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Compute vocabulary-trimmed logits with the tied embedding table."""
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream native MBridge or Vortex names into the vLLM parameter tree."""
        return self._weight_loader.load(weights)
