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

"""Behavioral tests for Evo2 inference precision setup."""

from functools import wraps
from types import SimpleNamespace

import pytest
import torch
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.core import fp8_utils

from bionemo.evo2.models.evo2_provider import Hyena7bModelProvider
from bionemo.evo2.run import low_precision as low_precision_module
from bionemo.evo2.run.low_precision import (
    configure_global_fp8_layer_scope,
    configure_quantized_parameter_storage,
    inference_parameter_storage,
    inference_precision_kind,
    prepare_model_for_quantized_inference,
    validate_inference_precision,
)


@pytest.mark.parametrize(
    ("config", "expected_kind"),
    [
        (SimpleNamespace(fp8=None, fp4=None, bf16=True, fp8_recipe=None), "bf16"),
        (SimpleNamespace(fp8="e4m3", fp4=None, bf16=True, fp8_recipe="mxfp8"), "mxfp8"),
        (SimpleNamespace(fp8="e4m3", fp4=None, bf16=True, fp8_recipe="current_scaling"), "fp8"),
        (
            SimpleNamespace(
                fp8="e4m3",
                fp4=None,
                bf16=True,
                fp8_recipe="current_scaling",
                evo2_fp8_all_layers=True,
            ),
            "fp8-all-layers",
        ),
        (SimpleNamespace(fp8=None, fp4="e2m1", bf16=True, fp8_recipe=None), "nvfp4"),
    ],
)
def test_inference_precision_kind_reports_the_active_compute_format(config, expected_kind):
    assert inference_precision_kind(config) == expected_kind


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(fp8="e4m3", fp4=None),
        SimpleNamespace(fp8=None, fp4="e2m1"),
    ],
)
def test_quantized_inference_prepares_te_linears_for_single_token_decode(monkeypatch, config):
    """Removing FP4/FP8 padding must fail because one-token decode would become unaligned."""
    model = SimpleNamespace(token_alignment_prepared=False)

    def prepare(candidate):
        candidate.token_alignment_prepared = True

    monkeypatch.setattr("megatron.core.fp8_utils.prepare_model_for_fp8_inference", prepare)

    assert prepare_model_for_quantized_inference(model, config) is True
    assert model.token_alignment_prepared is True


@pytest.mark.parametrize("fp8_recipe", ["tensorwise", "delayed"])
def test_regular_fp8_skips_per_layer_padding_when_flattened_gemm_rows_are_aligned(monkeypatch, fp8_recipe):
    calls: list[str] = []

    class FakeTELinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(32, 16), requires_grad=False)
            self.sequence_parallel = False

        def forward(self, input_tensor):
            calls.append("original")
            return input_tensor

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = FakeTELinear()

    def install_upstream_padding(candidate):
        original_forward = candidate.linear.forward

        @wraps(original_forward)
        def padded_forward(input_tensor):
            calls.append("padding")
            return original_forward(input_tensor)

        candidate.linear.forward = padded_forward

    monkeypatch.setattr(fp8_utils, "TE_LINEAR_TYPES", (FakeTELinear,))
    monkeypatch.setattr(fp8_utils, "prepare_model_for_fp8_inference", install_upstream_padding)
    model = FakeModel()
    config = SimpleNamespace(fp8="hybrid", fp4=None, fp8_recipe=fp8_recipe)

    prepare_model_for_quantized_inference(model, config)

    assert model.evo2_regular_fp8_aligned_fast_path_modules == 1
    model.linear(torch.zeros(1, 8, 16))
    assert calls == ["original"]

    calls.clear()
    model.linear(torch.zeros(1, 1, 16))
    assert calls == ["padding", "original"]

    calls.clear()
    model.linear.sequence_parallel = True
    model.linear(torch.zeros(1, 8, 16))
    assert calls == ["padding", "original"]

    calls.clear()
    model.linear.sequence_parallel = False
    model.linear.weight = torch.nn.Parameter(torch.empty(31, 16), requires_grad=False)
    model.linear(torch.zeros(1, 8, 16))
    assert calls == ["padding", "original"]


def test_bf16_inference_does_not_wrap_te_linears(monkeypatch):
    model = SimpleNamespace()
    calls = []

    def prepare(candidate):
        calls.append(candidate)

    monkeypatch.setattr(
        "megatron.core.fp8_utils.prepare_model_for_fp8_inference",
        prepare,
    )

    assert prepare_model_for_quantized_inference(model, SimpleNamespace(fp8=None, fp4=None)) is False
    assert calls == []


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(fp8="e4m3", fp4=None),
        SimpleNamespace(fp8=None, fp4="e2m1"),
    ],
)
def test_vortex_fp8_rejects_a_second_global_quantization_recipe(config):
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_inference_precision(config, vortex_style_fp8=True)


def test_vortex_fp8_accepts_the_required_bf16_global_recipe():
    validate_inference_precision(SimpleNamespace(fp8=None, fp4=None), vortex_style_fp8=True)


def test_global_fp8_all_layers_disables_the_mbridge_bf16_boundary_blocks():
    config = SimpleNamespace(
        fp8="hybrid",
        first_last_layers_bf16=True,
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
    )

    configure_global_fp8_layer_scope(config, all_layers=True)

    assert config.first_last_layers_bf16 is False
    assert config.num_layers_at_start_in_bf16 == 0
    assert config.num_layers_at_end_in_bf16 == 0


@pytest.mark.parametrize(
    "recipe_name",
    [
        "bf16_with_fp8_current_scaling_mixed",
        "bf16_with_fp8_delayed_scaling_mixed",
    ],
)
def test_global_fp8_all_layers_covers_every_block_in_the_installed_regular_fp8_recipes(recipe_name):
    config = get_mixed_precision_config(recipe_name)

    configure_global_fp8_layer_scope(config, all_layers=True)

    assert config.fp8 == "hybrid"
    assert config.first_last_layers_bf16 is False
    assert config.num_layers_at_start_in_bf16 == 0
    assert config.num_layers_at_end_in_bf16 == 0


def test_global_fp8_all_layers_propagates_into_the_actual_evo2_7b_provider():
    config = get_mixed_precision_config("bf16_with_fp8_current_scaling_mixed")
    configure_global_fp8_layer_scope(config, all_layers=True)
    config.finalize()
    provider = Hyena7bModelProvider()

    config.setup(provider)

    assert provider.fp8 == "hybrid"
    assert provider.fp8_recipe == "tensorwise"
    assert provider.first_last_layers_bf16 is False
    assert provider.num_layers_at_start_in_bf16 == 0
    assert provider.num_layers_at_end_in_bf16 == 0


def test_global_fp8_all_layers_rejects_a_non_fp8_recipe():
    with pytest.raises(ValueError, match="requires a global FP8 mixed-precision recipe"):
        configure_global_fp8_layer_scope(SimpleNamespace(fp8=None), all_layers=True)


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(
            fp8="e4m3",
            fp4=None,
            fp8_param=True,
            fp8_param_gather=True,
            fp4_param=False,
            fp4_param_gather=False,
            reuse_grad_buf_for_mxfp8_param_ag=True,
        ),
        SimpleNamespace(
            fp8=None,
            fp4="e2m1",
            fp8_param=False,
            fp8_param_gather=False,
            fp4_param=True,
            fp4_param_gather=True,
            reuse_grad_buf_for_mxfp8_param_ag=False,
        ),
    ],
)
def test_bf16_parameter_storage_disables_native_quantized_parameter_allocation(config):
    """The fallback must avoid constructing quantized target tensors before checkpoint load."""
    configure_quantized_parameter_storage(config, "bf16")

    assert config.fp8_param is False
    assert config.fp8_param_gather is False
    assert config.fp4_param is False
    assert config.fp4_param_gather is False
    assert config.reuse_grad_buf_for_mxfp8_param_ag is False
    assert inference_parameter_storage(config) == "bf16"


def test_recipe_parameter_storage_preserves_quantized_parameters():
    config = SimpleNamespace(fp8="e4m3", fp4=None, fp8_param=True, fp4_param=False)

    configure_quantized_parameter_storage(config, "recipe")

    assert config.fp8_param is True
    assert inference_parameter_storage(config) == "quantized"


def test_bf16_parameter_storage_rejects_a_non_quantized_compute_recipe():
    with pytest.raises(ValueError, match="requires an FP8 or FP4"):
        configure_quantized_parameter_storage(SimpleNamespace(fp8=None, fp4=None), "bf16")


@pytest.mark.parametrize(
    ("policy", "config", "tp", "expected"),
    [
        pytest.param("auto", SimpleNamespace(fp8=None, fp4=None), 2, True, id="auto-bf16"),
        pytest.param("auto", SimpleNamespace(fp8="hybrid", fp4=None), 2, False, id="auto-fp8"),
        pytest.param("auto", SimpleNamespace(fp8=None, fp4="e2m1"), 2, False, id="auto-fp4"),
        pytest.param("off", SimpleNamespace(fp8=None, fp4=None), 2, False, id="off-bf16"),
        pytest.param("on", SimpleNamespace(fp8=None, fp4=None), 2, True, id="on-bf16"),
        pytest.param("on", SimpleNamespace(fp8="hybrid", fp4=None), 2, True, id="on-fp8-upstream"),
        pytest.param("on", SimpleNamespace(fp8=None, fp4=None), 1, False, id="on-tp1"),
    ],
)
def test_prediction_sequence_parallel_policy_resolves_requested_mode(policy, config, tp, expected):
    """A wrong policy branch changes the provider passed to MBridge setup."""
    provider = SimpleNamespace(tensor_model_parallel_size=tp, sequence_parallel=None)

    enabled = low_precision_module.configure_prediction_sequence_parallel(
        provider,
        config,
        policy=policy,
    )

    assert enabled is expected
    assert provider.sequence_parallel is expected


def test_prediction_sequence_parallel_legacy_disable_overrides_auto():
    provider = SimpleNamespace(tensor_model_parallel_size=2, sequence_parallel=None)

    enabled = low_precision_module.configure_prediction_sequence_parallel(
        provider,
        SimpleNamespace(fp8=None, fp4=None),
        policy="auto",
        legacy_disabled=True,
    )

    assert enabled is False
    assert provider.sequence_parallel is False


def test_prediction_sequence_parallel_rejects_conflicting_legacy_disable():
    provider = SimpleNamespace(tensor_model_parallel_size=2, sequence_parallel=None)

    with pytest.raises(ValueError, match="conflicts with"):
        low_precision_module.configure_prediction_sequence_parallel(
            provider,
            SimpleNamespace(fp8=None, fp4=None),
            policy="on",
            legacy_disabled=True,
        )
