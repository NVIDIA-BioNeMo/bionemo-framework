# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""vLLM attention, MLP, normalization, and residual layers for Evo2."""

import re
from typing import cast

import torch
from torch import nn

from bionemo.evo2.vllm.config import Evo2Config


_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def apply_pre_norm_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Evo2's fused residual addition and pre-normalization contract."""
    from vllm.model_executor.layers.layernorm import RMSNorm

    if residual is None:
        normalized = RMSNorm.forward_static(
            hidden_states,
            eps,
            hidden_states.shape[-1],
            hidden_states.dtype,
            weight,
        )
        return cast(torch.Tensor, normalized), hidden_states

    normalized_and_residual = RMSNorm.forward_static(
        hidden_states,
        eps,
        hidden_states.shape[-1],
        hidden_states.dtype,
        weight,
        residual,
    )
    return cast(tuple[torch.Tensor, torch.Tensor], normalized_and_residual)


def _add_norm_weight(module: nn.Module, hidden_size: int, params_dtype: torch.dtype | None) -> None:
    dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
    module.register_parameter("layer_norm_weight", nn.Parameter(torch.ones(hidden_size, dtype=dtype)))


class IdentityAndMul(nn.Module):
    """Apply Evo2's post-layer-zero identity GLU: ``first_half * second_half``."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Multiply equally sized fused FC1 halves without a nonlinear activation."""
        if hidden_states.shape[-1] % 2:
            raise ValueError("a gated MLP projection must have an even final dimension")
        gate, up = hidden_states.chunk(2, dim=-1)
        return gate * up


def _global_layer_index(prefix: str) -> int:
    match = _LAYER_INDEX_PATTERN.search(prefix)
    if match is None:
        raise ValueError(f"Evo2 MLP prefix does not contain a global layer index: {prefix!r}")
    return int(match.group(1))


class Evo2MLP(nn.Module):
    """Tensor-parallel gated MLP with Evo2's global-layer activation policy."""

    def __init__(
        self,
        config: Evo2Config,
        *,
        quant_config=None,
        prefix: str = "",
        params_dtype: torch.dtype | None = None,
        disable_tp: bool = False,
    ) -> None:
        """Construct the fused column-parallel and row-parallel projections."""
        super().__init__()
        from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear

        self.linear_fc1 = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
            return_bias=False,
            disable_tp=disable_tp,
        )
        _add_norm_weight(self.linear_fc1, config.hidden_size, params_dtype)
        self.global_layer_index = _global_layer_index(prefix)
        if config.remove_activation_post_first_layer and self.global_layer_index > 0:
            self.activation = IdentityAndMul()
        elif config.hidden_act == "gelu":
            self.activation = GeluAndMul(approximate=config.gelu_approximate)
        elif config.hidden_act == "silu":
            self.activation = SiluAndMul()
        elif config.hidden_act == "identity":
            self.activation = IdentityAndMul()
        else:
            raise ValueError(f"unsupported Evo2 MLP activation: {config.hidden_act}")
        self.linear_fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
            return_bias=False,
            disable_tp=disable_tp,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the fused SwiGLU projection and output projection."""
        hidden_states = cast(torch.Tensor, self.linear_fc1(hidden_states))
        hidden_states = self.activation(hidden_states)
        return cast(torch.Tensor, self.linear_fc2(hidden_states))


class Evo2Attention(nn.Module):
    """vLLM paged attention using Evo2's fused QKV checkpoint layout."""

    def __init__(
        self,
        config: Evo2Config,
        *,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        max_position_embeddings: int | None = None,
        params_dtype: torch.dtype | None = None,
        disable_tp: bool = False,
    ) -> None:
        """Construct tensor-parallel QKV, RoPE, paged attention, and output projection."""
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
        from vllm.model_executor.layers.rotary_embedding import get_rope

        tp_size = 1 if disable_tp else get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp_size:
            raise ValueError("attention heads must be divisible by tensor parallel size")
        if config.num_key_value_heads >= tp_size:
            if config.num_key_value_heads % tp_size:
                raise ValueError("key/value heads must be divisible by tensor parallel size")
        elif tp_size % config.num_key_value_heads:
            raise ValueError("tensor parallel size must be divisible by replicated key/value heads")

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.linear_qkv = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_qkv",
            return_bias=False,
            disable_tp=disable_tp,
        )
        _add_norm_weight(self.linear_qkv, config.hidden_size, params_dtype)
        self.linear_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_proj",
            return_bias=False,
            disable_tp=disable_tp,
        )

        max_position = config.max_position_embeddings if max_position_embeddings is None else max_position_embeddings
        if max_position <= 0:
            raise ValueError("max_position_embeddings must be positive")
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            is_neox_style=True,
            rope_parameters={"rope_theta": config.rotary_base},
            dtype=params_dtype,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply rotary self-attention to one packed scheduler token tensor."""
        qkv = cast(torch.Tensor, self.linear_qkv(hidden_states))
        query, key, value = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        query, key = self.rotary_emb(positions, query, key)
        if key is None:
            raise RuntimeError("Evo2 self-attention requires keys")
        hidden_states = self.attn(query, key, value)
        return cast(torch.Tensor, self.linear_proj(hidden_states))


class Evo2AttentionDecoderLayer(nn.Module):
    """Evo2 attention decoder layer with vLLM's residual-carrying ABI."""

    def __init__(
        self,
        config: Evo2Config,
        *,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        max_position_embeddings: int | None = None,
        params_dtype: torch.dtype | None = None,
        disable_tp: bool = False,
    ) -> None:
        """Construct the attention and MLP sublayers with native checkpoint paths."""
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.self_attention = Evo2Attention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attention",
            max_position_embeddings=max_position_embeddings,
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
        """Apply attention and MLP while carrying vLLM's deferred residual."""
        hidden_states, residual = apply_pre_norm_residual(
            hidden_states,
            residual,
            self.self_attention.linear_qkv.layer_norm_weight,
            eps=self.rms_norm_eps,
        )
        hidden_states = self.self_attention(positions, hidden_states)
        hidden_states, residual = apply_pre_norm_residual(
            hidden_states,
            residual,
            self.mlp.linear_fc1.layer_norm_weight,
            eps=self.rms_norm_eps,
        )
        return self.mlp(hidden_states), residual
