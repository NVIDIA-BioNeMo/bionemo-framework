# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Packed vLLM cache adapter and sequence mixers for Evo2 Hyena layers."""

from functools import partial
from typing import cast

import torch
from torch import nn
from torch.library import Library
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata

from bionemo.evo2.vllm.config import Evo2Config
from bionemo.evo2.vllm.layers import Evo2MLP, _add_norm_weight, apply_pre_norm_residual
from bionemo.evo2.vllm.packed_fir import fir_route_telemetry_context, packed_causal_fir
from bionemo.evo2.vllm.packed_iir import packed_modal_iir
from bionemo.evo2.vllm.weights import load_tensor_parallel_weight


_EVO2_LIBRARY = Library("bionemo_evo2", "FRAGMENT")
_PREFILL_MAX_QUERY_LEN_KEY = "bionemo_evo2_prefill_max_query_len"


def _parameter(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    shard_dim: int = 0,
) -> nn.Parameter:
    from vllm.model_executor.utils import set_weight_attrs

    parameter = nn.Parameter(torch.empty(shape, dtype=dtype, device=device))
    set_weight_attrs(
        parameter,
        {"weight_loader": partial(load_tensor_parallel_weight, shard_dim=shard_dim)},
    )
    return parameter


class _ProjectionConv(nn.Module):
    def __init__(self, channels: int, taps: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.short_conv_weight = _parameter((channels, taps), dtype=dtype, device=device)


class _ShortConv(nn.Module):
    def __init__(self, groups: int, taps: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.short_conv_weight = _parameter((groups, 1, taps), dtype=dtype, device=device)


class _HCSOperator(nn.Module):
    def __init__(
        self,
        groups: int,
        taps: int,
        *,
        use_bias: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.short_conv = _ShortConv(groups, taps, dtype=dtype, device=device)
        if use_bias:
            self.conv_bias = _parameter((groups,), dtype=dtype, device=device)


class _HCMFilter(nn.Module):
    def __init__(self, groups: int, taps: int, *, device: torch.device) -> None:
        super().__init__()
        self.h = _parameter((groups, taps), dtype=torch.float32, device=device)
        self.decay = _parameter((groups, taps), dtype=torch.float32, device=device)
        self.register_buffer(
            "effective_filter",
            torch.empty((groups, taps), dtype=torch.float32, device=device),
            persistent=False,
        )


class _HCMOperator(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int,
        taps: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.conv_bias = _parameter((channels,), dtype=dtype, device=device)
        self.filter = _HCMFilter(groups, taps, device=device)


class _HCLFilter(nn.Module):
    def __init__(self, groups: int, state_size: int, *, device: torch.device) -> None:
        super().__init__()
        shape = (groups, state_size)
        self.p = _parameter(shape, dtype=torch.float32, device=device)
        self.gamma = _parameter(shape, dtype=torch.float32, device=device)
        self.R = _parameter(shape, dtype=torch.float32, device=device)
        self.register_buffer(
            "modal_decay",
            torch.empty(shape, dtype=torch.float32, device=device),
            persistent=False,
        )


class _HCLOperator(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int,
        state_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.conv_bias = _parameter((channels,), dtype=dtype, device=device)
        self.filter = _HCLFilter(groups, state_size, device=device)


def _validate_evo2_cache_config(cache_config: CacheConfig) -> None:
    """Accept only uncached execution or vLLM's physical align-mode state reuse."""
    if not cache_config.enable_prefix_caching:
        if cache_config.mamba_cache_mode != "none":
            raise ValueError("mamba cache mode must be 'none' when prefix caching is disabled")
        return
    if cache_config.mamba_cache_mode != "align":
        raise ValueError("Evo2 prefix caching supports only mamba_cache_mode='align'")
    if cache_config.mamba_block_size != cache_config.block_size:
        raise ValueError("Evo2 align mode requires matching Mamba and attention block sizes")


def _one_dimensional_state_indices(state_indices: torch.Tensor, *, kind: str) -> torch.Tensor:
    if state_indices.ndim == 1:
        return state_indices
    if state_indices.ndim == 2 and state_indices.shape[1] == 1:
        return state_indices[:, 0]
    raise ValueError(f"Evo2 {kind} state indices must identify exactly one cache block per request")


def _prefill_max_query_len(
    forward_context: ForwardContext,
    query_start_loc: torch.Tensor,
) -> int:
    cached = forward_context.additional_kwargs.get(_PREFILL_MAX_QUERY_LEN_KEY)
    if cached is None:
        # vLLM does not expose max_query_len on Mamba1 metadata. Cache the one scalar
        # synchronization in the shared model-forward context so it is not repeated per layer.
        cached = int(torch.diff(query_start_loc).max().item())
        if cached < 1:
            raise ValueError("Evo2 packed prefill requests must contain at least one token")
        forward_context.additional_kwargs[_PREFILL_MAX_QUERY_LEN_KEY] = cached
    return cast(int, cached)


@PluggableLayer.register("evo2_hyena_mixer")
class Evo2HyenaMixer(MambaBase, PluggableLayer):
    """One checkpoint-compatible Evo2 Hyena mixer over vLLM packed batches."""

    def __init__(
        self,
        config: Evo2Config,
        *,
        operator_type: str,
        cache_config: CacheConfig,
        quant_config=None,
        prefix: str = "",
        params_dtype: torch.dtype | None = None,
        disable_tp: bool = False,
    ) -> None:
        """Construct one tensor-parallel mixer and register its vLLM cache owner."""
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

        if operator_type not in ("S", "D", "H"):
            raise ValueError(f"unsupported Evo2 Hyena operator type: {operator_type}")
        _validate_evo2_cache_config(cache_config)

        tp_size = 1 if disable_tp else get_tensor_model_parallel_world_size()
        if config.hidden_size % tp_size:
            raise ValueError("hidden size must be divisible by tensor parallel size")
        local_hidden_size = config.hidden_size // tp_size
        global_groups = {
            "S": config.num_groups_hyena_short,
            "D": config.num_groups_hyena_medium,
            "H": config.num_groups_hyena,
        }[operator_type]
        if global_groups % tp_size:
            raise ValueError(f"{operator_type} groups must be divisible by tensor parallel size")
        local_groups = global_groups // tp_size
        if local_hidden_size % local_groups:
            raise ValueError(f"{operator_type} groups must partition the local hidden size")

        dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.operator_type = operator_type
        self.local_hidden_size = local_hidden_size
        self.operator_group_size = local_hidden_size // local_groups
        self.operator_state_size = config.hcl_state_size
        self.prefix = prefix
        self.cache_config = cache_config
        self._state_shapes = config.local_state_shapes(tp_size)

        self.dense_projection = ColumnParallelLinear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_projection",
            return_bias=False,
            disable_tp=disable_tp,
        )
        _add_norm_weight(self.dense_projection, config.hidden_size, params_dtype)
        device = self.dense_projection.weight.device
        self.hyena_proj_conv = _ProjectionConv(
            3 * local_hidden_size,
            config.short_conv_length,
            dtype=dtype,
            device=device,
        )
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
            return_bias=False,
            disable_tp=disable_tp,
        )
        if operator_type == "S":
            self.mixer = _HCSOperator(
                local_groups,
                config.hcs_filter_length,
                use_bias=config.use_short_conv_bias,
                dtype=dtype,
                device=device,
            )
        elif operator_type == "D":
            self.mixer = _HCMOperator(
                local_hidden_size,
                local_groups,
                config.hcm_filter_length,
                dtype=dtype,
                device=device,
            )
        else:
            self.mixer = _HCLOperator(
                local_hidden_size,
                local_groups,
                config.hcl_state_size,
                dtype=dtype,
                device=device,
            )

        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"duplicate Evo2 layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.register_buffer(
            "_decode_query_start_loc",
            torch.arange(max_num_seqs + 1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_decode_has_initial_state",
            torch.ones(max_num_seqs, dtype=torch.bool, device=device),
            persistent=False,
        )

    def get_state_shape(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return projection-FIR and uniform operator-state shapes per request."""
        return self._state_shapes

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        """Keep both recurrence caches in fp32 across bf16 model execution."""
        return (torch.float32, torch.float32)

    @property
    def mamba_type(self) -> str:
        """Reuse vLLM's Mamba1 metadata and cache manager."""
        return "mamba1"

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        """Invoke the opaque state-mutating op used by vLLM compilation and graphs."""
        projection_state, operator_state = self.kv_cache
        torch.ops.bionemo_evo2.hyena_mixer(
            hidden_states,
            output,
            projection_state,
            operator_state,
            _encode_layer_name(self.prefix),
        )

    def _mix_segment(
        self,
        projected_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        projection_state: torch.Tensor,
        operator_state: torch.Tensor,
        *,
        max_query_len: int,
    ) -> torch.Tensor:
        projected_states = packed_causal_fir(
            projected_states,
            self.hyena_proj_conv.short_conv_weight,
            None,
            projection_state,
            query_start_loc,
            state_indices,
            has_initial_state,
            max_query_len=max_query_len,
        )
        x1, x2, value = projected_states.view(
            projected_states.shape[0],
            self.local_hidden_size,
            3,
        ).unbind(-1)
        drive = x2 * value

        if self.operator_type == "S":
            filtered = packed_causal_fir(
                drive,
                self.mixer.short_conv.short_conv_weight.squeeze(1),
                getattr(self.mixer, "conv_bias", None),
                operator_state,
                query_start_loc,
                state_indices,
                has_initial_state,
                group_size=self.operator_group_size,
                gated_bias=True,
                max_query_len=max_query_len,
            )
            return x1 * filtered
        if self.operator_type == "D":
            filtered = packed_causal_fir(
                drive,
                self.mixer.filter.effective_filter,
                self.mixer.conv_bias,
                operator_state,
                query_start_loc,
                state_indices,
                has_initial_state,
                group_size=self.operator_group_size,
                gated_bias=True,
                flip_filter=True,
                max_query_len=max_query_len,
            )
            return x1 * filtered

        decay = self.mixer.filter.modal_decay
        residues = self.mixer.filter.R
        if self.operator_group_size != 1:
            decay = decay.repeat_interleave(self.operator_group_size, dim=0)
            residues = residues.repeat_interleave(self.operator_group_size, dim=0)
        return packed_modal_iir(
            drive,
            x1,
            decay,
            residues,
            self.mixer.conv_bias,
            operator_state,
            query_start_loc,
            state_indices,
            has_initial_state,
            state_size=self.operator_state_size,
            max_query_len=max_query_len,
        )

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        projection_state: torch.Tensor,
        operator_state: torch.Tensor,
    ) -> None:
        """Project, mix each packed scheduler segment, and update vLLM's cache in place."""
        forward_context: ForwardContext = get_forward_context()
        metadata_raw = forward_context.attn_metadata
        projected_states = cast(torch.Tensor, self.dense_projection(hidden_states))

        if metadata_raw is None:
            profile_values = projected_states.view(
                projected_states.shape[0],
                self.local_hidden_size,
                3,
            )[..., 2]
            output.copy_(cast(torch.Tensor, self.dense(profile_values)))
            return

        if not isinstance(metadata_raw, dict):
            raise TypeError("Evo2 Hyena attention metadata must be keyed by layer prefix")
        metadata = metadata_raw[self.prefix]
        if not isinstance(metadata, Mamba1AttentionMetadata):
            raise TypeError("Evo2 Hyena layers require Mamba1AttentionMetadata")
        if metadata.num_decode_tokens != metadata.num_decodes:
            raise ValueError("Evo2 Hyena vLLM inference does not support speculative decode")

        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        num_actual_tokens = num_decode_tokens + num_prefill_tokens
        if num_actual_tokens == 0:
            return
        if projected_states.shape[0] < num_actual_tokens:
            raise ValueError("packed Evo2 metadata references more tokens than the model input")
        mixed = torch.empty(
            (num_actual_tokens, self.local_hidden_size),
            dtype=projected_states.dtype,
            device=projected_states.device,
        )

        if num_decode_tokens:
            if metadata.state_indices_tensor_d is None:
                raise ValueError("decode metadata is missing Evo2 state indices")
            if num_decode_tokens > self._decode_has_initial_state.numel():
                raise ValueError("decode batch exceeds the configured vLLM max_num_seqs")
            decode_state_indices = _one_dimensional_state_indices(
                metadata.state_indices_tensor_d,
                kind="decode",
            )
            mixed[:num_decode_tokens] = self._mix_segment(
                projected_states[:num_decode_tokens],
                self._decode_query_start_loc[: num_decode_tokens + 1],
                decode_state_indices,
                self._decode_has_initial_state[:num_decode_tokens],
                projection_state,
                operator_state,
                max_query_len=1,
            )

        if num_prefill_tokens:
            if (
                metadata.query_start_loc_p is None
                or metadata.state_indices_tensor_p is None
                or metadata.has_initial_states_p is None
            ):
                raise ValueError("prefill metadata is missing Evo2 packed sequence boundaries")
            prefill_state_indices = _one_dimensional_state_indices(
                metadata.state_indices_tensor_p,
                kind="prefill",
            )
            max_query_len = _prefill_max_query_len(
                forward_context,
                metadata.query_start_loc_p,
            )
            mixed[num_decode_tokens:num_actual_tokens] = self._mix_segment(
                projected_states[num_decode_tokens:num_actual_tokens],
                metadata.query_start_loc_p,
                prefill_state_indices,
                metadata.has_initial_states_p,
                projection_state,
                operator_state,
                max_query_len=max_query_len,
            )

        output[:num_actual_tokens] = cast(torch.Tensor, self.dense(mixed))


class Evo2HyenaDecoderLayer(nn.Module):
    """Evo2 Hyena decoder layer with vLLM's residual-carrying ABI."""

    def __init__(
        self,
        config: Evo2Config,
        *,
        operator_type: str,
        cache_config: CacheConfig,
        quant_config=None,
        prefix: str = "",
        params_dtype: torch.dtype | None = None,
        disable_tp: bool = False,
    ) -> None:
        """Construct the Hyena mixer and its checkpoint-compatible SwiGLU MLP."""
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.mixer = Evo2HyenaMixer(
            config,
            operator_type=operator_type,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
            params_dtype=params_dtype,
            disable_tp=disable_tp,
        )
        self.mlp = Evo2MLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            params_dtype=params_dtype,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply pre-norm Hyena mixing and MLP while carrying the residual."""
        del positions
        hidden_states, residual = apply_pre_norm_residual(
            hidden_states,
            residual,
            self.mixer.dense_projection.layer_norm_weight,
            eps=self.rms_norm_eps,
        )
        mixer_output = torch.empty_like(hidden_states)
        self.mixer(hidden_states, mixer_output)
        hidden_states, residual = apply_pre_norm_residual(
            mixer_output,
            residual,
            self.mlp.linear_fc1.layer_norm_weight,
            eps=self.rms_norm_eps,
        )
        return self.mlp(hidden_states), residual


def _hyena_mixer_custom_op(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    projection_state: torch.Tensor,
    operator_state: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    mixer = forward_context.no_compile_layers[layer_name]
    if not isinstance(mixer, Evo2HyenaMixer):
        raise TypeError(f"registered Evo2 layer {layer_name!r} is not an Evo2HyenaMixer")
    with fir_route_telemetry_context():
        mixer.forward_impl(hidden_states, output, projection_state, operator_state)


def _hyena_mixer_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    projection_state: torch.Tensor,
    operator_state: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return None


direct_register_custom_op(
    op_name="hyena_mixer",
    op_func=_hyena_mixer_custom_op,
    mutates_args=["output", "projection_state", "operator_state"],
    fake_impl=_hyena_mixer_fake,
    target_lib=_EVO2_LIBRARY,
)
