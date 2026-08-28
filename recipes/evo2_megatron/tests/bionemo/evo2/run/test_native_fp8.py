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

"""Tests for inference-only native FP8 projection conversion."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import bionemo.evo2.run.native_fp8 as native_fp8
from bionemo.evo2.run.native_fp8 import (
    NativeFP8LayerNormColumnLinear,
    prepare_model_for_native_fp8_inference,
    validate_native_fp8_decode,
    validate_native_fp8_policy,
)


@pytest.mark.parametrize("policy", ["off", "hyena", "fc1", "expansion"])
def test_valid_policies(policy):
    validate_native_fp8_policy(policy)


@pytest.mark.parametrize("policy", ["all", "fp8", "vortex", ""])
def test_invalid_policies(policy):
    with pytest.raises(ValueError, match="native FP8 policy"):
        validate_native_fp8_policy(policy)


@pytest.mark.parametrize("decode_mode", ["none", "bf16", "fp8"])
def test_valid_decode_modes(decode_mode):
    validate_native_fp8_decode(decode_mode)


def test_invalid_decode_mode():
    with pytest.raises(ValueError, match="decode mode"):
        validate_native_fp8_decode("auto")


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

    assert native_fp8._te_rmsnorm(module, value) is normalized
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


class _Candidate(nn.Module):
    pass


class _Converted(nn.Module):
    def __init__(self, source, *, decode_mode):
        super().__init__()
        self.source = source
        self.decode_mode = decode_mode
        self.original_weight_bytes = 16
        self.quantized_weight_bytes = 8


def _fake_model():
    model = nn.Module()
    model.decoder = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.linear_fc1 = _Candidate()
    layer.mlp.linear_fc2 = _Candidate()
    layer.mixer = nn.Module()
    layer.mixer.dense_projection = _Candidate()
    layer.mixer.dense = _Candidate()
    layer.self_attention = nn.Module()
    layer.self_attention.linear_qkv = _Candidate()
    layer.self_attention.linear_proj = _Candidate()
    model.decoder.layers = nn.ModuleList([layer])
    return model.eval()


@pytest.mark.parametrize(
    ("policy", "expected_names"),
    [
        ("hyena", {"decoder.layers.0.mixer.dense_projection"}),
        ("fc1", {"decoder.layers.0.mlp.linear_fc1"}),
        (
            "expansion",
            {
                "decoder.layers.0.mlp.linear_fc1",
                "decoder.layers.0.mixer.dense_projection",
                "decoder.layers.0.self_attention.linear_qkv",
            },
        ),
    ],
)
def test_conversion_scope(monkeypatch, policy, expected_names):
    model = _fake_model()
    monkeypatch.setattr(native_fp8, "_is_te_layernorm_column_linear", lambda module: isinstance(module, _Candidate))
    monkeypatch.setattr(native_fp8, "NativeFP8LayerNormColumnLinear", _Converted)

    report = prepare_model_for_native_fp8_inference(model, policy=policy, decode_mode="bf16")

    assert set(report.module_names) == expected_names
    assert report.converted_modules == len(expected_names)
    assert report.original_weight_bytes == 16 * len(expected_names)
    assert report.quantized_weight_bytes == 8 * len(expected_names)
    for name, module in model.named_modules():
        if name in expected_names:
            assert isinstance(module, _Converted)
            assert module.decode_mode == "bf16"


def test_off_policy_is_noop():
    model = _fake_model()
    report = prepare_model_for_native_fp8_inference(model, policy="off")
    assert report.converted_modules == 0
    assert isinstance(model.decoder.layers[0].mlp.linear_fc1, _Candidate)


def test_forward_uses_native_mxfp8_tensor_core(monkeypatch):
    calls = {}

    def quantize(activation):
        calls["quantize"] = activation
        return activation.to(torch.float8_e4m3fn), torch.ones(3, 1, dtype=torch.uint8)

    def scaled_mm(activation, weight, scale_a, scale_b, **kwargs):
        calls["gemm"] = (activation, weight, scale_a, scale_b, kwargs)
        return torch.zeros(6, 8, dtype=torch.bfloat16)

    projection = NativeFP8LayerNormColumnLinear.__new__(NativeFP8LayerNormColumnLinear)
    nn.Module.__init__(projection)
    projection.in_features = 16
    projection.local_out_features = 8
    projection.eps = 1e-5
    projection.zero_centered_gamma = False
    projection.inf_ln_sm_margin = 0
    projection.te_return_bias = False
    projection.use_bias = False
    projection.bias = None
    projection.decode_mode = "none"
    projection.bf16_decode_module = None
    projection.layer_norm_weight = nn.Parameter(torch.ones(16, dtype=torch.bfloat16))
    projection.register_buffer("weight_mxfp8", torch.ones(8, 16, dtype=torch.float8_e4m3fn))
    projection.register_buffer("weight_block_scales", torch.ones(1, 1, dtype=torch.uint8))
    projection.register_buffer("decode_weight_fp8", None)
    monkeypatch.setattr(native_fp8, "_te_rmsnorm", lambda _module, value: value)
    monkeypatch.setattr(native_fp8, "_mxfp8_quantize", quantize)
    monkeypatch.setattr(native_fp8, "_native_mxfp8_mm", scaled_mm)

    with torch.no_grad():
        output, bias = projection(torch.linspace(-2, 2, 96, dtype=torch.bfloat16).reshape(3, 2, 16))

    assert output.shape == (3, 2, 8)
    assert bias is None
    activation, weight, scale_a, scale_b, options = calls["gemm"]
    assert activation.dtype is torch.float8_e4m3fn
    assert weight.dtype is torch.float8_e4m3fn
    assert weight.stride(0) == 1
    assert scale_a.dtype is torch.uint8
    assert scale_b is projection.weight_block_scales
    assert options == {"out_dtype": torch.bfloat16}


def test_bf16_decode_bypasses_fp8(monkeypatch):
    expected = torch.randn(1, 4, 8, dtype=torch.bfloat16)

    class _BF16Decode(nn.Module):
        def forward(self, _x):
            return expected, None

    projection = NativeFP8LayerNormColumnLinear.__new__(NativeFP8LayerNormColumnLinear)
    nn.Module.__init__(projection)
    projection.in_features = 16
    projection.decode_mode = "bf16"
    projection.bf16_decode_module = _BF16Decode()
    monkeypatch.setattr(native_fp8, "_native_mxfp8_mm", lambda *_args, **_kwargs: pytest.fail("FP8 GEMM called"))

    with torch.no_grad():
        actual, bias = projection(torch.ones(1, 4, 16, dtype=torch.bfloat16))

    assert actual is expected
    assert bias is None


def test_bf16_decode_rejects_mismatched_hidden_size():
    class _BF16Decode(nn.Module):
        def forward(self, _x):
            raise AssertionError("invalid input must fail before the BF16 bypass")

    projection = NativeFP8LayerNormColumnLinear.__new__(NativeFP8LayerNormColumnLinear)
    nn.Module.__init__(projection)
    projection.in_features = 16
    projection.decode_mode = "bf16"
    projection.bf16_decode_module = _BF16Decode()

    with torch.no_grad(), pytest.raises(ValueError, match="Expected hidden size 16"):
        projection(torch.ones(1, 4, 8, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(fp8="e4m3", fp4=None),
        SimpleNamespace(fp8=None, fp4="e2m1"),
    ],
)
def test_native_fp8_rejects_global_recipe(config):
    with pytest.raises(ValueError, match="native FP8"):
        native_fp8.validate_native_fp8_precision(config, vortex_style_fp8=False, native_nvfp4_policy="off")


def test_native_fp8_rejects_vortex_flag():
    config = SimpleNamespace(fp8=None, fp4=None)
    with pytest.raises(ValueError, match="native FP8"):
        native_fp8.validate_native_fp8_precision(config, vortex_style_fp8=True, native_nvfp4_policy="off")
