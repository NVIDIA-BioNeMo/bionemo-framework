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

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import sentinel
from zipfile import ZipFile

import numpy as np
import pytest
import torch

import bionemo.evo2.models.evo2_provider as evo2_provider
from bionemo.evo2.models.evo2_provider import (
    HYENA_MODEL_OPTIONS,
    MODEL_OPTIONS,
    Hyena1bModelProvider,
    HyenaTestModelProvider,
    _patch_megatron_dataset_helper_compile,
    bind_hyena_packed_views_to_static_context,
    build_evo2_mamba_inference_state_config,
    infer_model_type,
    reset_hyena_packed_views_for_new_request,
)
from bionemo.evo2.utils.checkpoint.mbridge_to_vortex import _split_fc1, mbridge_to_vortex_state_dict
from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import load_savanna_state_dict, savanna_to_mbridge_state_dict


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


def test_hyena_provider_leaves_te_context_parallel_transport_unset():
    """Checkpoint model definitions do not persist a runtime transport optimization."""
    per_layer = ["a2a", "p2p"]
    assert HyenaTestModelProvider().cp_comm_type is None
    assert HyenaTestModelProvider(cp_comm_type=None).cp_comm_type is None
    assert HyenaTestModelProvider(cp_comm_type="p2p").cp_comm_type == "p2p"
    assert HyenaTestModelProvider(cp_comm_type=per_layer).cp_comm_type is per_layer


@pytest.mark.parametrize(
    ("device_capability", "configured_backend", "expected_fa4"),
    [
        ((8, 0), evo2_provider.AttnBackend.flash, False),
        ((8, 6), evo2_provider.AttnBackend.flash, False),
        ((8, 9), evo2_provider.AttnBackend.flash, False),
        ((8, 9), evo2_provider.AttnBackend.fused, False),
        ((8, 9), evo2_provider.AttnBackend.auto, False),
        ((9, 0), evo2_provider.AttnBackend.flash, True),
    ],
)
def test_fa4_backend_respects_supported_device_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    device_capability: tuple[int, int],
    configured_backend,
    expected_fa4: bool,
):
    """FA4 remains enabled on Hopper while SM8x keeps flash dispatch through FA2."""
    from megatron.core.transformer import attention as mcore_attention
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils

    provider = SimpleNamespace(attention_backend=configured_backend)
    monkeypatch.setattr(mcore_attention, "HAVE_FA4", True)
    monkeypatch.setattr(FlashAttentionUtils, "v4_is_installed", True)

    evo2_provider._configure_fa4_for_device(provider, device_capability=device_capability)

    assert provider.attention_backend is configured_backend
    assert mcore_attention.HAVE_FA4 is expected_fa4
    assert FlashAttentionUtils.v4_is_installed is expected_fa4


def test_fa4_backend_falls_back_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from megatron.core.transformer import attention as mcore_attention

    provider = SimpleNamespace(attention_backend=evo2_provider.AttnBackend.flash)
    monkeypatch.setattr(mcore_attention, "HAVE_FA4", True)
    monkeypatch.setattr(evo2_provider.torch.cuda, "is_available", lambda: False)

    assert evo2_provider._configure_fa4_for_device(provider) is False
    assert provider.attention_backend is evo2_provider.AttnBackend.flash
    assert mcore_attention.HAVE_FA4 is False


def test_sm89_disables_te_fa4_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCore disabling FA4 must not leave TE free to select it on SM89."""
    from megatron.core.transformer import attention as mcore_attention
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils

    provider = SimpleNamespace(attention_backend=evo2_provider.AttnBackend.flash)
    monkeypatch.setattr(mcore_attention, "HAVE_FA4", False)
    monkeypatch.setattr(FlashAttentionUtils, "v4_is_installed", True)

    assert evo2_provider._configure_fa4_for_device(provider, device_capability=(8, 9)) is False
    assert provider.attention_backend is evo2_provider.AttnBackend.flash
    assert FlashAttentionUtils.v4_is_installed is False


def test_fa4_selection_clears_derived_te_backend_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inherited TE selectors cannot override the provider-selected backend."""
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.transformer import attention as mcore_attention
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils

    provider = SimpleNamespace(attention_backend=evo2_provider.AttnBackend.flash)
    monkeypatch.setattr(mcore_attention, "HAVE_FA4", True)
    monkeypatch.setattr(FlashAttentionUtils, "v4_is_installed", True)
    for variable in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.setenv(variable, "1")

    assert evo2_provider._configure_fa4_for_device(provider, device_capability=(8, 9)) is False

    assert provider.attention_backend is evo2_provider.AttnBackend.flash
    assert FlashAttentionUtils.v4_is_installed is False
    assert all(variable not in os.environ for variable in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"))

    LanguageModule._set_attention_backend(SimpleNamespace(config=provider))

    assert os.environ["NVTE_FLASH_ATTN"] == "1"
    assert os.environ["NVTE_FUSED_ATTN"] == "0"
    assert os.environ["NVTE_UNFUSED_ATTN"] == "0"


@pytest.mark.parametrize(("requested", "expected"), [(None, "p2p"), ("p2p", "p2p"), ("a2a", "a2a")])
def test_configure_runtime_context_parallel_comm_type(requested: str | None, expected: str):
    """Runtime selection defaults to P2P and overrides stale checkpoint metadata."""
    provider = SimpleNamespace(cp_comm_type="a2a")

    resolved = evo2_provider.configure_runtime_context_parallel_comm_type(provider, requested)

    assert resolved == expected
    assert provider.cp_comm_type == expected


def test_configure_runtime_context_parallel_comm_type_rejects_unknown_transport():
    provider = SimpleNamespace(cp_comm_type=None)

    with pytest.raises(ValueError, match="context-parallel communication type"):
        evo2_provider.configure_runtime_context_parallel_comm_type(provider, "all_gather")


def test_static_hyena_state_binding_keeps_graph_stable_full_rings() -> None:
    """Static prefill copies even short FIR tails into persistent full-ring views."""
    shapes = SimpleNamespace(
        conv_shape=(6, 2),
        conv_owner_id=101,
        ssm_shape=(2, 4),
        ssm_kind="inner_fir",
        ssm_owner_id=202,
    )
    layer = SimpleNamespace(
        mixer=SimpleNamespace(hyena_state_shapes_per_request=lambda: shapes),
    )
    decoder = SimpleNamespace(
        layers=[layer],
        hyena_state_shapes_per_request=lambda: ((6, 2), (2, 4), [shapes]),
    )
    model = SimpleNamespace(decoder=decoder)
    context = SimpleNamespace()

    bind_hyena_packed_views_to_static_context(model, context, batch_size=2, device=torch.device("cpu"))
    projection_tail = torch.arange(12, dtype=torch.float32).reshape(2, 6, 1)
    context.fir_filter_state_dict[101] = projection_tail

    first_view = context.fir_filter_state_dict[101]
    assert first_view.shape == (2, 6, 2)
    torch.testing.assert_close(first_view[..., 0], torch.zeros(2, 6))
    torch.testing.assert_close(first_view[..., 1:], projection_tail)
    first_ptr = first_view.data_ptr()

    reset_hyena_packed_views_for_new_request(context)
    assert 101 not in context.fir_filter_state_dict
    assert torch.count_nonzero(context._evo2_hyena_conv_states) == 0

    context.fir_filter_state_dict[101] = torch.full((2, 6, 2), 7.0)
    assert context.fir_filter_state_dict[101].data_ptr() == first_ptr
    torch.testing.assert_close(context.fir_filter_state_dict[101], torch.full((2, 6, 2), 7.0))


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


def test_get_batch_passes_the_context_parallel_group(monkeypatch: pytest.MonkeyPatch):
    """Keep TP/PP ranks out of MCore's context-parallel batch partitioning."""
    batch = {
        "tokens": sentinel.tokens,
        "labels": sentinel.labels,
        "loss_mask": sentinel.loss_mask,
        "attention_mask": None,
        "position_ids": sentinel.position_ids,
        "cu_seqlens": None,
    }
    pg_collection = SimpleNamespace(pp=sentinel.pp_group, cp=sentinel.cp_group)

    monkeypatch.setattr(evo2_provider, "is_pp_first_stage", lambda group: True)
    monkeypatch.setattr(evo2_provider, "is_pp_last_stage", lambda group: True)
    monkeypatch.setattr(evo2_provider, "get_batch_from_iterator", lambda *args, **kwargs: batch)

    def partition_on_cp_group(actual_batch, *, is_hybrid_cp, cp_group):
        assert actual_batch is batch
        assert is_hybrid_cp is False
        assert cp_group is sentinel.cp_group
        return actual_batch

    monkeypatch.setattr(evo2_provider, "get_batch_on_this_cp_rank", partition_on_cp_group)

    cfg = SimpleNamespace(dataset=SimpleNamespace(skip_getting_attention_mask_from_dataset=True))
    result = evo2_provider.get_batch(iter(()), cfg, pg_collection=pg_collection)

    assert result[:3] == (sentinel.tokens, sentinel.labels, sentinel.loss_mask)


def test_hyena_only_stage_gets_kv_sentinel():
    """MCore dynamic inference gets its required KV slot without shifting real layers."""
    from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols

    layer_types = [Symbols.MAMBA] * 4
    model = SimpleNamespace(
        decoder=SimpleNamespace(
            layer_type_list=layer_types,
            mamba_state_shapes_per_request=lambda: ((2, 3), (4, 5)),
        )
    )

    state_config = build_evo2_mamba_inference_state_config(model)

    assert state_config.layer_type_list == [*layer_types, Symbols.ATTENTION]
    assert layer_types == [Symbols.MAMBA] * 4


def test_load_legacy_savanna_numpy_metadata(tmp_path: Path):
    """Legacy Savanna metadata can be loaded without disabling weights-only safety."""
    source_path = tmp_path / "current_numpy.pt"
    checkpoint_path = tmp_path / "legacy_savanna.pt"
    expected = torch.arange(4)
    torch.save(
        {
            "module": {"module.sequential.0.weight": expected},
            "rng_state": np.array([1234], dtype=np.uint32),
        },
        source_path,
    )

    # NumPy 2 writes ``numpy._core``; the pinned Savanna checkpoint was written
    # by NumPy 1 and therefore records the legacy ``numpy.core`` module path.
    replaced_module_path = False
    with ZipFile(source_path) as source, ZipFile(checkpoint_path, "w") as legacy:
        for member in source.infolist():
            payload = source.read(member.filename)
            if member.filename.endswith("data.pkl"):
                updated = payload.replace(b"numpy._core.multiarray", b"numpy.core.multiarray")
                replaced_module_path = updated != payload
                payload = updated
            legacy.writestr(member, payload)
    assert replaced_module_path

    state_dict = load_savanna_state_dict(checkpoint_path)

    assert torch.equal(state_dict["sequential.0.weight"], expected)


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
