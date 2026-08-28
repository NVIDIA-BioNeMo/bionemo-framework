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

"""Tests for selective native-NVFP4 inference conversion."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import bionemo.evo2.run.native_nvfp4 as native_nvfp4
from bionemo.evo2.run.low_precision import validate_inference_precision
from bionemo.evo2.run.native_nvfp4 import (
    NativeNVFP4LayerNormColumnLinear,
    NativeNVFP4RowParallelLinear,
    native_nvfp4_target_kind,
    prepare_model_for_native_nvfp4_inference,
    rmsnorm_no_clip_amax,
    validate_native_nvfp4_decode,
    validate_native_nvfp4_policy,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("decoder.layers.0.mlp.linear_fc1", "fc1"),
        ("decoder.layers.0.mixer.dense_projection", "hyena_projection"),
        ("decoder.layers.0.self_attention.linear_qkv", "attention_qkv"),
        ("decoder.layers.0.mlp.linear_fc2", "fc2"),
        ("decoder.layers.0.mixer.dense", "hyena_output"),
        ("decoder.layers.0.self_attention.linear_proj", "attention_output"),
    ],
)
def test_target_kinds(name, expected):
    assert native_nvfp4_target_kind(name) == expected


@pytest.mark.parametrize("policy", ["off", "fc1", "expansion", "full"])
def test_valid_policies(policy):
    validate_native_nvfp4_policy(policy, activation_amax=8.0)


@pytest.mark.parametrize("policy", ["all", "nvfp4", ""])
def test_invalid_policies(policy):
    with pytest.raises(ValueError, match="native NVFP4 policy"):
        validate_native_nvfp4_policy(policy, activation_amax=8.0)


def test_invalid_activation_range():
    with pytest.raises(ValueError, match="activation amax"):
        validate_native_nvfp4_policy("fc1", activation_amax=0.0)


def test_automatic_activation_range():
    validate_native_nvfp4_policy("fc1", activation_amax=None)
    bound = rmsnorm_no_clip_amax(
        torch.tensor([0.5, -2.0]),
        in_features=16,
        zero_centered_gamma=False,
    )
    torch.testing.assert_close(bound, torch.tensor(8.0))


def test_zero_centered_gamma_bound():
    bound = rmsnorm_no_clip_amax(
        torch.tensor([0.0, 1.0]),
        in_features=16,
        zero_centered_gamma=True,
    )
    torch.testing.assert_close(bound, torch.tensor(8.0))


@pytest.mark.parametrize("decode_mode", ["bf16", "fp8", "nvfp4"])
def test_valid_decode_modes(decode_mode):
    validate_native_nvfp4_decode(decode_mode)


def test_invalid_decode_mode():
    with pytest.raises(ValueError, match="decode mode"):
        validate_native_nvfp4_decode("auto")


def test_te_rmsnorm_calls_transformer_engine_with_named_parameters(monkeypatch):
    from transformer_engine.pytorch.module import _common

    calls = {}
    normalized = torch.ones(2, 4)

    def apply_normalization(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return normalized, None, None

    monkeypatch.setattr(_common, "apply_normalization", apply_normalization)
    module = SimpleNamespace(
        layer_norm_weight=torch.ones(4),
        eps=1e-5,
        inf_ln_sm_margin=7,
        zero_centered_gamma=False,
    )
    value = torch.zeros(2, 4)

    assert native_nvfp4._te_rmsnorm(module, value) is normalized
    assert calls["args"] == ()
    assert calls["kwargs"] == {
        "inputmat": value,
        "ln_out": None,
        "ln_weight": module.layer_norm_weight,
        "ln_bias": None,
        "eps": module.eps,
        "output_quantizer": None,
        "output_dtype": value.dtype,
        "normalization": "RMSNorm",
        "fwd_ln_sm_margin": module.inf_ln_sm_margin,
        "zero_centered_gamma": module.zero_centered_gamma,
    }


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(fp8="e4m3", fp4=None),
        SimpleNamespace(fp8=None, fp4="e2m1"),
    ],
)
def test_native_policy_rejects_global_quantization(config):
    with pytest.raises(ValueError, match="native NVFP4"):
        validate_inference_precision(config, vortex_style_fp8=False, native_nvfp4_policy="fc1")


def test_native_policy_rejects_vortex_fp8():
    config = SimpleNamespace(fp8=None, fp4=None)
    with pytest.raises(ValueError, match="native NVFP4"):
        validate_inference_precision(config, vortex_style_fp8=True, native_nvfp4_policy="fc1")


class _ColumnCandidate(nn.Module):
    pass


class _RowCandidate(nn.Module):
    pass


class _Converted(nn.Module):
    def __init__(self, source, *, activation_amax, decode_mode):
        super().__init__()
        self.source = source
        self.activation_amax = activation_amax
        self.decode_mode = decode_mode


class _RowConverted(_Converted):
    pass


def _fake_model():
    model = nn.Module()
    model.decoder = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.linear_fc1 = _ColumnCandidate()
    layer.mlp.linear_fc2 = _RowCandidate()
    layer.mixer = nn.Module()
    layer.mixer.dense_projection = _ColumnCandidate()
    layer.mixer.dense = _RowCandidate()
    layer.self_attention = nn.Module()
    layer.self_attention.linear_qkv = _ColumnCandidate()
    layer.self_attention.linear_proj = _RowCandidate()
    model.decoder.layers = nn.ModuleList([layer])
    return model.eval()


@pytest.mark.parametrize(
    ("policy", "expected_names"),
    [
        ("fc1", {"decoder.layers.0.mlp.linear_fc1"}),
        (
            "expansion",
            {
                "decoder.layers.0.mlp.linear_fc1",
                "decoder.layers.0.mixer.dense_projection",
                "decoder.layers.0.self_attention.linear_qkv",
            },
        ),
        (
            "full",
            {
                "decoder.layers.0.mlp.linear_fc1",
                "decoder.layers.0.mlp.linear_fc2",
                "decoder.layers.0.mixer.dense_projection",
                "decoder.layers.0.mixer.dense",
                "decoder.layers.0.self_attention.linear_qkv",
                "decoder.layers.0.self_attention.linear_proj",
            },
        ),
    ],
)
def test_conversion_scope(monkeypatch, policy, expected_names):
    model = _fake_model()
    monkeypatch.setattr(
        native_nvfp4,
        "_is_te_layernorm_column_linear",
        lambda module: isinstance(module, _ColumnCandidate),
    )
    monkeypatch.setattr(
        native_nvfp4,
        "_is_te_row_parallel_linear",
        lambda module: isinstance(module, _RowCandidate),
    )
    monkeypatch.setattr(native_nvfp4, "NativeNVFP4LayerNormColumnLinear", _Converted)
    monkeypatch.setattr(native_nvfp4, "NativeNVFP4RowParallelLinear", _RowConverted)

    report = prepare_model_for_native_nvfp4_inference(model, policy=policy, activation_amax=7.5)

    assert set(report.module_names) == expected_names
    assert report.converted_modules == len(expected_names)
    for name, module in model.named_modules():
        if name in expected_names:
            assert isinstance(module, (_Converted, _RowConverted))
            assert module.activation_amax == 7.5
            assert module.decode_mode == "bf16"


def test_off_policy_is_noop():
    model = _fake_model()
    report = prepare_model_for_native_nvfp4_inference(model, policy="off")
    assert report.converted_modules == 0
    assert report.module_names == ()
    assert isinstance(model.decoder.layers[0].mlp.linear_fc1, _ColumnCandidate)


def test_forward_requests_native_nvfp4(monkeypatch):
    calls = {}

    class _FlashInfer:
        @staticmethod
        def fp4_quantize(activation, global_scale, **kwargs):
            calls["quantize"] = (activation, global_scale, kwargs)
            packed_activation = torch.ones(2, 8, dtype=torch.uint8)
            calls["packed_activation"] = packed_activation
            return packed_activation, torch.ones(1, dtype=torch.uint8)

        @staticmethod
        def mm_fp4(activation, weight, activation_scales, weight_scales, alpha, **kwargs):
            calls["gemm"] = (activation, weight, activation_scales, weight_scales, alpha, kwargs)
            return torch.zeros(2, 8, dtype=torch.bfloat16)

    projection = NativeNVFP4LayerNormColumnLinear.__new__(NativeNVFP4LayerNormColumnLinear)
    nn.Module.__init__(projection)
    projection.in_features = 16
    projection.local_out_features = 8
    projection.eps = 1e-5
    projection.zero_centered_gamma = False
    projection.inf_ln_sm_margin = 0
    projection.te_return_bias = False
    projection.use_bias = False
    projection.bias = None
    projection.decode_mode = "nvfp4"
    projection.bf16_decode_module = None
    projection._flashinfer = _FlashInfer()
    projection.layer_norm_weight = nn.Parameter(torch.ones(16, dtype=torch.bfloat16))
    projection.register_buffer("packed_weight", torch.full((8, 8), 3, dtype=torch.uint8))
    projection.register_buffer("weight_block_scales", torch.full((1, 8), 4, dtype=torch.uint8))
    projection.register_buffer("activation_global_scale", torch.ones(1, dtype=torch.float32))
    projection.register_buffer("gemm_alpha", torch.ones(1, dtype=torch.float32))
    projection.register_buffer("fp8_activation_global_scale", None)
    projection.register_buffer("fp8_decode_weight", None)
    projection.register_buffer("fp8_gemm_alpha", None)
    monkeypatch.setattr(native_nvfp4, "_te_rmsnorm", lambda _module, value: value)

    with torch.no_grad():
        output, bias = projection(torch.ones(2, 1, 16, dtype=torch.bfloat16))

    assert output.shape == (2, 1, 8)
    assert bias is None
    assert calls["quantize"][2] == {
        "sf_vec_size": 16,
        "is_sf_swizzled_layout": True,
        "backend": "cuda",
    }
    gemm_activation, gemm_weight, _, _, _, gemm_options = calls["gemm"]
    assert gemm_activation is calls["packed_activation"]
    assert torch.equal(gemm_weight, projection.packed_weight.T)
    assert gemm_options == {
        "out_dtype": torch.bfloat16,
        "block_size": 16,
        "use_8x4_sf_layout": False,
        "backend": "cutlass",
        "use_nvfp4": True,
    }


def test_row_forward_requests_native_nvfp4_then_tensor_parallel_reduce(monkeypatch):
    calls = {}

    class _FlashInfer:
        @staticmethod
        def fp4_quantize(activation, global_scale, **kwargs):
            calls["quantize"] = (activation, global_scale, kwargs)
            return torch.ones(2, 8, dtype=torch.uint8), torch.ones(1, dtype=torch.uint8)

        @staticmethod
        def mm_fp4(*args, **kwargs):
            calls["gemm"] = (args, kwargs)
            return torch.full((2, 8), 3.0, dtype=torch.bfloat16)

    projection = NativeNVFP4RowParallelLinear.__new__(NativeNVFP4RowParallelLinear)
    nn.Module.__init__(projection)
    projection.in_features = 16
    projection.local_out_features = 8
    projection.te_return_bias = False
    projection.use_bias = True
    projection.bias = nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    projection.decode_mode = "nvfp4"
    projection.bf16_decode_module = None
    projection._flashinfer = _FlashInfer()
    projection.tp_size = 2
    projection.tp_group = object()
    projection.register_buffer("packed_weight", torch.full((8, 8), 3, dtype=torch.uint8))
    projection.register_buffer("weight_block_scales", torch.full((1, 8), 4, dtype=torch.uint8))
    projection.register_buffer("activation_global_scale", torch.ones(1, dtype=torch.float32))
    projection.register_buffer("gemm_alpha", torch.ones(1, dtype=torch.float32))
    projection.register_buffer("fp8_activation_global_scale", None)
    projection.register_buffer("fp8_decode_weight", None)
    projection.register_buffer("fp8_gemm_alpha", None)

    def reduce_output(value, *, group, tp_size):
        calls["reduce"] = (value, group, tp_size)
        return value + 2

    monkeypatch.setattr(native_nvfp4, "_reduce_row_parallel_output", reduce_output)

    with torch.no_grad():
        output, bias = projection(torch.ones(2, 1, 16, dtype=torch.bfloat16))

    assert output.shape == (2, 1, 8)
    assert torch.equal(output, torch.full_like(output, 6.0))
    assert bias is None
    assert calls["reduce"][1:] == (projection.tp_group, 2)
