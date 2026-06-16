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

"""Tests for model provider instantiation, naming, and checkpoint converters."""

from pathlib import Path

import pytest
import torch

from bionemo.evo2.models.evo2_provider import (
    HYENA_MODEL_OPTIONS,
    MODEL_OPTIONS,
    Hyena1bModelProvider,
    _patch_megatron_dataset_helper_compile,
    infer_model_type,
)
from bionemo.evo2.utils.checkpoint.mbridge_to_vortex import _split_fc1, mbridge_to_vortex_state_dict
from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import savanna_to_mbridge_state_dict


def test_evo2_prefix_for_arc_models():
    """Verify evo2-prefixed ARC model keys exist in HYENA_MODEL_OPTIONS."""
    for key in ["evo2_1b_base", "evo2_7b_base", "evo2_7b", "evo2_40b_base", "evo2_40b"]:
        assert key in HYENA_MODEL_OPTIONS


def test_striped_hyena_prefix_for_nv_models():
    """Verify striped_hyena-prefixed NV model keys exist in HYENA_MODEL_OPTIONS."""
    for key in ["striped_hyena_1b_nv", "striped_hyena_7b_nv", "striped_hyena_40b_nv"]:
        assert key in HYENA_MODEL_OPTIONS


def test_old_keys_removed():
    """Verify deprecated short keys are no longer in HYENA_MODEL_OPTIONS."""
    for key in ["1b", "7b", "40b", "1b_nv", "7b_nv", "40b_nv", "test", "test_nv"]:
        assert key not in HYENA_MODEL_OPTIONS, f"Old key '{key}' still present"


def test_model_options_equals_hyena():
    """Verify MODEL_OPTIONS equals HYENA_MODEL_OPTIONS (Eden removed)."""
    assert set(MODEL_OPTIONS.keys()) == set(HYENA_MODEL_OPTIONS.keys())


def test_infer_model_type_hyena():
    """Verify infer_model_type returns 'hyena' for all HYENA model keys."""
    for key in HYENA_MODEL_OPTIONS:
        assert infer_model_type(key) == "hyena"


def test_infer_model_type_unknown():
    """Verify infer_model_type raises ValueError for unknown model keys."""
    with pytest.raises(ValueError, match="Unknown model size"):
        infer_model_type("nonexistent_model")


@pytest.mark.parametrize(
    ("has_makefile", "has_prebuilt_extension", "expected_original_calls"),
    [
        (False, True, 0),
        (True, True, 1),
        (False, False, 1),
    ],
)
def test_megatron_dataset_helper_compile_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    has_makefile: bool,
    has_prebuilt_extension: bool,
    expected_original_calls: int,
):
    """Skip Megatron's runtime make step only when a prebuilt helper extension exists."""
    from megatron.bridge.training import initialize as bridge_initialize
    from megatron.core.datasets import utils as dataset_utils

    calls = []

    def original_compile_helpers():
        calls.append("called")

    if has_makefile:
        (tmp_path / "Makefile").write_text("all:\n")
    if has_prebuilt_extension:
        (tmp_path / "helpers_cpp.cpython-312-x86_64-linux-gnu.so").touch()

    monkeypatch.setattr(dataset_utils, "__file__", str(tmp_path / "utils.py"))
    monkeypatch.setattr(dataset_utils, "compile_helpers", original_compile_helpers)
    monkeypatch.setattr(bridge_initialize, "compile_helpers", original_compile_helpers)

    _patch_megatron_dataset_helper_compile()

    dataset_utils.compile_helpers()
    assert bridge_initialize.compile_helpers is dataset_utils.compile_helpers
    assert len(calls) == expected_original_calls


def _make_mock_savanna_sd(pattern: str) -> dict[str, torch.Tensor]:
    """Create a minimal mock savanna state dict for the given pattern.

    Savanna layout: 0=embedding, 1=lambda(no params), 2..N+1=layers, N+2=lambda, N+3=final_norm.
    """
    sd = {}
    sd["sequential.0.word_embeddings.weight"] = torch.randn(512, 1920)
    num_layers = len(pattern)

    for i, symbol in enumerate(pattern):
        src_idx = i + 2
        sd[f"sequential.{src_idx}.pre_mlp_layernorm.weight"] = torch.randn(1920)
        sd[f"sequential.{src_idx}.mlp.w1.weight"] = torch.randn(5120, 1920)
        sd[f"sequential.{src_idx}.mlp.w2.weight"] = torch.randn(5120, 1920)
        sd[f"sequential.{src_idx}.mlp.w3.weight"] = torch.randn(1920, 5120)
        sd[f"sequential.{src_idx}.input_layernorm.weight"] = torch.randn(1920)

        if symbol != "*":
            sd[f"sequential.{src_idx}.mixer.dense_projection.weight"] = torch.randn(5760, 1920)
            sd[f"sequential.{src_idx}.mixer.hyena_proj_conv.short_conv_weight"] = torch.randn(5760, 3)
            sd[f"sequential.{src_idx}.mixer.dense.weight"] = torch.randn(1920, 1920)
            sd[f"sequential.{src_idx}.mixer.dense.bias"] = torch.randn(1920)
            if symbol == "S":
                sd[f"sequential.{src_idx}.mixer.mixer.short_conv.short_conv_weight"] = torch.randn(1920, 1, 7)
            elif symbol == "D":
                sd[f"sequential.{src_idx}.mixer.mixer.conv_bias"] = torch.randn(1920)
                sd[f"sequential.{src_idx}.mixer.mixer.filter.h"] = torch.randn(1920, 256)
                sd[f"sequential.{src_idx}.mixer.mixer.filter.decay"] = torch.randn(1920, 256)
            elif symbol == "H":
                sd[f"sequential.{src_idx}.mixer.mixer.conv_bias"] = torch.randn(1920)
                sd[f"sequential.{src_idx}.mixer.mixer.filter.gamma"] = torch.randn(1920)
                sd[f"sequential.{src_idx}.mixer.mixer.filter.R"] = torch.randn(1920 * 128)
                sd[f"sequential.{src_idx}.mixer.mixer.filter.p"] = torch.randn(1920 * 128)
        else:
            sd[f"sequential.{src_idx}.mixer.dense_projection.weight"] = torch.randn(5760, 1920)
            sd[f"sequential.{src_idx}.mixer.dense.weight"] = torch.randn(1920, 1920)
            sd[f"sequential.{src_idx}.mixer.dense.bias"] = torch.randn(1920)

    sd[f"sequential.{num_layers + 3}.norm.weight"] = torch.randn(1920)
    return sd


def test_savanna_embedding_mapped():
    """Verify savanna embedding is mapped to mbridge embedding.word_embeddings.weight."""
    sd = _make_mock_savanna_sd("S")
    result = savanna_to_mbridge_state_dict(sd, "S", te_enabled=True)
    assert "embedding.word_embeddings.weight" in result


def test_savanna_final_norm_mapped():
    """Verify savanna final norm is mapped to mbridge decoder.final_norm.weight."""
    sd = _make_mock_savanna_sd("S")
    result = savanna_to_mbridge_state_dict(sd, "S", te_enabled=True)
    assert "decoder.final_norm.weight" in result


def test_savanna_mlp_merge():
    """Verify savanna MLP w1/w3 are merged into mbridge linear_fc1 with correct shape."""
    sd = _make_mock_savanna_sd("S")
    result = savanna_to_mbridge_state_dict(sd, "S", te_enabled=True)
    fc1 = result["decoder.layers.0.mlp.linear_fc1.weight"]
    assert fc1.shape[0] == 5120 * 2


def test_savanna_all_layer_types():
    """Verify savanna-to-mbridge conversion produces MLP keys for all layer types (S, D, H, *)."""
    pattern = "SDH*"
    sd = _make_mock_savanna_sd(pattern)
    result = savanna_to_mbridge_state_dict(sd, pattern, te_enabled=True)
    for i in range(4):
        assert f"decoder.layers.{i}.mlp.linear_fc1.weight" in result
        assert f"decoder.layers.{i}.mlp.linear_fc2.weight" in result


def test_savanna_attention_keys():
    """Verify attention-only (*) layers get linear_qkv and linear_proj keys in mbridge format."""
    sd = _make_mock_savanna_sd("*")
    result = savanna_to_mbridge_state_dict(sd, "*", te_enabled=True)
    assert "decoder.layers.0.self_attention.linear_qkv.weight" in result
    assert "decoder.layers.0.self_attention.linear_proj.weight" in result


def test_mlp_fc1_split_merge_roundtrip():
    """Verify _split_fc1 correctly splits merged w1/w3 back to original tensors."""
    w1 = torch.randn(5120, 1920)
    w2 = torch.randn(5120, 1920)
    merged = torch.cat([w1, w2], dim=0)
    split_w1, split_w2 = _split_fc1(merged)
    assert torch.equal(w1, split_w1)
    assert torch.equal(w2, split_w2)


def test_vortex_embedding_duplicated():
    """Verify mbridge-to-vortex duplicates embedding into embedding_layer and unembed."""
    mock_provider = Hyena1bModelProvider()
    sd = {"embedding.word_embeddings.weight": torch.randn(512, 1920)}
    sd["decoder.final_norm.weight"] = torch.randn(1920)
    result = mbridge_to_vortex_state_dict(sd, mock_provider, te_enabled=True)
    assert "embedding_layer.weight" in result
    assert "unembed.weight" in result
    assert torch.equal(result["embedding_layer.weight"], result["unembed.weight"])


def test_vortex_final_norm_mapped():
    """Verify mbridge decoder.final_norm is mapped to vortex norm.scale."""
    mock_provider = Hyena1bModelProvider()
    sd = {"decoder.final_norm.weight": torch.randn(1920)}
    result = mbridge_to_vortex_state_dict(sd, mock_provider, te_enabled=True)
    assert "norm.scale" in result


def test_vortex_mlp_split():
    """Verify mbridge MLP linear_fc1 is split into vortex l1/l2/l3 with correct shapes."""
    mock_provider = Hyena1bModelProvider()
    sd = {
        "decoder.layers.0.mlp.linear_fc1.weight": torch.randn(10240, 1920),
        "decoder.layers.0.mlp.linear_fc2.weight": torch.randn(1920, 5120),
        "decoder.layers.0.mlp.linear_fc1.layer_norm_weight": torch.randn(1920),
        "decoder.final_norm.weight": torch.randn(1920),
    }
    result = mbridge_to_vortex_state_dict(sd, mock_provider, te_enabled=True)
    assert "blocks.0.mlp.l1.weight" in result
    assert "blocks.0.mlp.l2.weight" in result
    assert "blocks.0.mlp.l3.weight" in result
    assert result["blocks.0.mlp.l1.weight"].shape[0] == 5120
    assert result["blocks.0.mlp.l2.weight"].shape[0] == 5120
