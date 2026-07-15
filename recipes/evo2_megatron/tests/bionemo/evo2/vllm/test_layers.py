# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from contextlib import contextmanager
from math import isclose
from tempfile import TemporaryDirectory

import pytest
import torch
from torch import nn
from torch.nn import functional

from bionemo.evo2.vllm.config import Evo2Config
from bionemo.evo2.vllm.layers import (
    Evo2Attention,
    Evo2AttentionDecoderLayer,
    Evo2MLP,
    apply_pre_norm_residual,
)


DEVICE = "cuda"
DTYPE = torch.bfloat16
CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _config() -> Evo2Config:
    return Evo2Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        hybrid_override_pattern="*",
        rms_norm_eps=1e-6,
        rotary_base=10_000,
    )


@contextmanager
def _vllm_module_context():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    vllm_config = VllmConfig()
    with (
        TemporaryDirectory() as temporary_directory,
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(DTYPE),
    ):
        torch.cuda.set_device(0)
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temporary_directory}/distributed_init",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        try:
            yield vllm_config
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


class _DeterministicAttention(nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return query + 2 * key + 3 * value


def _randomize(module: nn.Module, seed: int = 17) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("layer_norm_weight"):
                parameter.copy_(
                    1 + 0.05 * torch.randn(parameter.shape, device=DEVICE, dtype=DTYPE, generator=generator)
                )
            else:
                parameter.copy_(0.05 * torch.randn(parameter.shape, device=DEVICE, dtype=DTYPE, generator=generator))


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def _neox_rope(x: torch.Tensor, positions: torch.Tensor, *, heads: int, base: float) -> torch.Tensor:
    tokens, hidden = x.shape
    head_dim = hidden // heads
    x = x.view(tokens, heads, head_dim)
    inverse_frequency = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
    frequency = positions.float().unsqueeze(-1) * inverse_frequency.unsqueeze(0)
    cosine = frequency.cos().to(x.dtype).unsqueeze(1)
    sine = frequency.sin().to(x.dtype).unsqueeze(1)
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1).reshape(tokens, hidden)


def _attention_reference(
    attention: Evo2Attention, positions: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    config = _config()
    qkv = functional.linear(hidden_states, attention.linear_qkv.weight)
    query, key, value = qkv.chunk(3, dim=-1)
    query = _neox_rope(query, positions, heads=config.num_attention_heads, base=config.rotary_base)
    key = _neox_rope(key, positions, heads=config.num_key_value_heads, base=config.rotary_base)
    mixed = query + 2 * key + 3 * value
    return functional.linear(mixed, attention.linear_proj.weight, attention.linear_proj.bias)


def _mlp_reference(mlp: Evo2MLP, hidden_states: torch.Tensor) -> torch.Tensor:
    gate_up = functional.linear(hidden_states, mlp.linear_fc1.weight)
    gate, up = gate_up.chunk(2, dim=-1)
    return functional.linear(functional.silu(gate) * up, mlp.linear_fc2.weight)


def test_apply_pre_norm_residual_matches_two_stage_reference() -> None:
    hidden_states = torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -0.5, -1.5]])
    weight = torch.tensor([0.75, 1.25, 1.5, 0.5])

    normalized, residual = apply_pre_norm_residual(hidden_states, None, weight, eps=1e-6)
    torch.testing.assert_close(normalized, _rms_norm(hidden_states, weight, 1e-6))
    torch.testing.assert_close(residual, hidden_states)

    update = torch.tensor([[0.25, 0.5, -0.75, 1.0], [-1.0, 0.25, 0.5, -0.25]])
    normalized, new_residual = apply_pre_norm_residual(update, residual, weight, eps=1e-6)
    expected_residual = update + hidden_states
    torch.testing.assert_close(new_residual, expected_residual)
    torch.testing.assert_close(normalized, _rms_norm(expected_residual, weight, 1e-6))


@CUDA_REQUIRED
def test_mlp_matches_dense_bf16_reference() -> None:
    with _vllm_module_context():
        mlp = Evo2MLP(_config(), prefix="decoder.layers.0.mlp", disable_tp=True).to(DEVICE).eval()
        _randomize(mlp)
        hidden_states = torch.randn((7, 64), device=DEVICE, dtype=DTYPE)

        with torch.inference_mode():
            actual = mlp(hidden_states)
            expected = _mlp_reference(mlp, hidden_states)

    assert actual.dtype == DTYPE
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@CUDA_REQUIRED
def test_attention_matches_qkv_rotary_and_projection_reference() -> None:
    with _vllm_module_context():
        attention = (
            Evo2Attention(
                _config(),
                prefix="decoder.layers.0.self_attention",
                max_position_embeddings=257,
                disable_tp=True,
            )
            .to(DEVICE)
            .eval()
        )
        attention.attn = _DeterministicAttention()
        _randomize(attention)
        hidden_states = torch.randn((4, 64), device=DEVICE, dtype=DTYPE)
        positions = torch.tensor([0, 1, 7, 15], device=DEVICE, dtype=torch.long)

        with torch.inference_mode():
            actual = attention(positions, hidden_states)
            expected = _attention_reference(attention, positions, hidden_states)

    assert attention.rotary_emb.max_position_embeddings == 257
    assert isclose(attention.rotary_emb.base, _config().rotary_base)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@CUDA_REQUIRED
def test_decoder_preserves_mbridge_names_and_residual_order() -> None:
    config = _config()
    with _vllm_module_context():
        layer = (
            Evo2AttentionDecoderLayer(
                config,
                prefix="decoder.layers.0",
                disable_tp=True,
            )
            .to(DEVICE)
            .eval()
        )
        layer.self_attention.attn = _DeterministicAttention()
        _randomize(layer)
        names = set(dict(layer.named_parameters()))
        assert names == {
            "self_attention.linear_qkv.layer_norm_weight",
            "self_attention.linear_qkv.weight",
            "self_attention.linear_proj.weight",
            "self_attention.linear_proj.bias",
            "mlp.linear_fc1.layer_norm_weight",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc2.weight",
        }

        hidden_states = torch.randn((5, 64), device=DEVICE, dtype=DTYPE)
        positions = torch.arange(5, device=DEVICE, dtype=torch.long)
        first_norm = _rms_norm(hidden_states, layer.self_attention.linear_qkv.layer_norm_weight, config.rms_norm_eps)
        attention_output = _attention_reference(layer.self_attention, positions, first_norm)
        expected_residual = hidden_states + attention_output
        second_norm = _rms_norm(expected_residual, layer.mlp.linear_fc1.layer_norm_weight, config.rms_norm_eps)
        expected_hidden = _mlp_reference(layer.mlp, second_norm)

        with torch.inference_mode():
            actual_hidden, actual_residual = layer(positions, hidden_states, None)

    torch.testing.assert_close(actual_residual, expected_residual, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=2e-2, atol=2e-2)


@CUDA_REQUIRED
def test_mlp_supports_fullgraph_compilation() -> None:
    with _vllm_module_context():
        mlp = Evo2MLP(_config(), prefix="decoder.layers.0.mlp_compile", disable_tp=True).to(DEVICE).eval()
        _randomize(mlp)
        hidden_states = torch.randn((7, 64), device=DEVICE, dtype=DTYPE)

        with torch.inference_mode():
            expected = mlp(hidden_states)
            actual = torch.compile(mlp, fullgraph=True)(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
