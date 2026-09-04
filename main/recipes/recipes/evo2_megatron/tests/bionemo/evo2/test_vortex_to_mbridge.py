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

"""Tests for ``bionemo.evo2.utils.checkpoint.vortex_to_mbridge``."""

import os
from collections import OrderedDict
from io import BytesIO

import pytest
import torch
import torch.distributed.checkpoint as dcp
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor

from bionemo.evo2.models.evo2_provider import HYENA_MODEL_OPTIONS, HyenaTestModelProvider
from bionemo.evo2.utils.checkpoint.mbridge_to_vortex import (
    _split_fc1,
    load_mbridge_state_dict,
    mbridge_to_vortex_state_dict,
)
from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import _to_mcore_sharded_state_dict
from bionemo.evo2.utils.checkpoint.vortex_to_mbridge import (
    _add_mbridge_te_extra_states,
    _merge_fc1,
    _validate_rotary_frequencies,
    download_vortex_checkpoint,
    load_vortex_state_dict,
    vortex_to_mbridge_state_dict,
)


def _make_test_provider() -> HyenaTestModelProvider:
    """Use tiny dimensions while preserving all Evo2 layer symbols."""
    return HyenaTestModelProvider(
        hybrid_override_pattern="SDH*",
        hidden_size=4,
        num_groups_hyena=4,
        num_groups_hyena_medium=4,
        num_groups_hyena_short=4,
        num_attention_heads=2,
        ffn_hidden_size=6,
    )


def _make_mock_vortex_sd(pattern: str) -> dict[str, torch.Tensor]:
    """Create a compact mock Vortex state dict that exercises all layer symbols."""
    sd = {
        "embedding_layer.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "norm.scale": torch.randn(4, dtype=torch.bfloat16),
    }
    sd["unembed.weight"] = sd["embedding_layer.weight"].clone()

    for i, symbol in enumerate(pattern):
        bp = f"blocks.{i}"
        sd[f"{bp}.pre_norm.scale"] = torch.randn(4, dtype=torch.bfloat16)
        sd[f"{bp}.post_norm.scale"] = torch.randn(4, dtype=torch.bfloat16)
        sd[f"{bp}.mlp.l1.weight"] = torch.randn(6, 4, dtype=torch.bfloat16)
        sd[f"{bp}.mlp.l2.weight"] = torch.randn(6, 4, dtype=torch.bfloat16)
        sd[f"{bp}.mlp.l3.weight"] = torch.randn(4, 6, dtype=torch.bfloat16)

        if symbol == "*":
            sd[f"{bp}.inner_mha_cls.Wqkv.weight"] = torch.randn(12, 4, dtype=torch.bfloat16)
            sd[f"{bp}.inner_mha_cls.out_proj.weight"] = torch.randn(4, 4, dtype=torch.bfloat16)
            sd[f"{bp}.inner_mha_cls.out_proj.bias"] = torch.randn(4, dtype=torch.bfloat16)
            sd[f"{bp}.inner_mha_cls.rotary_emb.inv_freq"] = torch.ones(1, dtype=torch.float32)
        else:
            sd[f"{bp}.projections.weight"] = torch.randn(12, 4, dtype=torch.bfloat16)
            sd[f"{bp}.filter.short_filter_weight"] = torch.randn(12, 1, 3, dtype=torch.bfloat16)
            sd[f"{bp}.out_filter_dense.weight"] = torch.randn(4, 4, dtype=torch.bfloat16)
            sd[f"{bp}.out_filter_dense.bias"] = torch.randn(4, dtype=torch.bfloat16)
            if symbol == "S":
                sd[f"{bp}.filter.h"] = torch.randn(4, 1, 7, dtype=torch.bfloat16)
            elif symbol == "D":
                sd[f"{bp}.filter.D"] = torch.randn(4, dtype=torch.bfloat16)
                sd[f"{bp}.filter.h"] = 1e-8 * torch.randn(4, 1, 128, dtype=torch.float32)
            elif symbol == "H":
                sd[f"{bp}.filter.D"] = torch.randn(4, dtype=torch.bfloat16)
                # Negative log poles are required for a valid inverse.
                sd[f"{bp}.filter.log_poles"] = -(0.25 + torch.rand(4, 1, 1, dtype=torch.float32))
                sd[f"{bp}.filter.residues"] = torch.randn(4, 1, dtype=torch.float32)

    return sd


def test_mlp_fc1_merge_split_roundtrip():
    """Verify Vortex l1/l2 merge into MBridge fc1 and split back exactly."""
    w1 = torch.randn(8, 4)
    w2 = torch.randn(8, 4)
    merged = _merge_fc1(w1, w2)
    split_w1, split_w2 = _split_fc1(merged)
    assert torch.equal(w1, split_w1)
    assert torch.equal(w2, split_w2)


def test_vortex_to_mbridge_all_layer_types():
    """Verify Vortex-to-MBridge conversion emits keys for S, D, H, and attention layers."""
    result = vortex_to_mbridge_state_dict(_make_mock_vortex_sd("SDH*"), _make_test_provider(), te_enabled=True)

    assert "embedding.word_embeddings.weight" in result
    assert "decoder.final_norm.weight" in result
    assert "decoder.layers.0.mixer.mixer.short_conv.short_conv_weight" in result
    assert "decoder.layers.1.mixer.mixer.filter.h" in result
    assert "decoder.layers.1.mixer.mixer.filter.decay" in result
    assert "decoder.layers.2.mixer.mixer.filter.p" in result
    assert "decoder.layers.2.mixer.mixer.filter.gamma" in result
    assert "decoder.layers.2.mixer.mixer.filter.R" in result
    assert "decoder.layers.3.self_attention.linear_qkv.weight" in result
    assert "decoder.layers.3.self_attention.linear_proj.weight" in result
    assert "decoder.layers.3.self_attention.rotary_emb.inv_freq" not in result


def test_load_vortex_state_dict_allows_vortex_te_metadata(tmp_path):
    """The weights-only loader should safely deserialize the BytesIO metadata in released checkpoints."""
    checkpoint = OrderedDict(
        [
            ("embedding_layer.weight", torch.ones(2, 2)),
            ("blocks.0.projections._extra_state", BytesIO(b"vortex-fp8")),
        ]
    )
    checkpoint_path = tmp_path / "vortex.pt"
    torch.save(checkpoint, checkpoint_path)

    loaded = load_vortex_state_dict(checkpoint_path)

    assert torch.equal(loaded["embedding_layer.weight"], torch.ones(2, 2))
    assert loaded["blocks.0.projections._extra_state"].getvalue() == b"vortex-fp8"


def test_load_vortex_state_dict_rejects_duplicate_normalized_keys(tmp_path):
    """A module prefix must not make one checkpoint tensor silently overwrite another."""
    checkpoint = OrderedDict(
        [
            ("weight", torch.tensor([1.0])),
            ("module.weight", torch.tensor([2.0])),
        ]
    )
    checkpoint_path = tmp_path / "vortex.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="duplicate normalized Vortex key"):
        load_vortex_state_dict(checkpoint_path)


def test_vortex_to_mbridge_discards_vortex_runtime_state():
    """Vortex runtime caches and FP8 state must not leak into the MBridge checkpoint."""
    vortex_sd = _make_mock_vortex_sd("SDH*")
    vortex_sd["blocks.0.projections._extra_state"] = BytesIO(b"legacy-vortex-fp8")
    vortex_sd["blocks.2.filter.t"] = torch.arange(8, dtype=torch.float32).view(1, 1, -1)

    result = vortex_to_mbridge_state_dict(vortex_sd, _make_test_provider(), te_enabled=True)

    runtime_suffixes = ("._extra_state", ".filter.t")
    assert not any(key.endswith(runtime_suffixes) for key in result)


def test_mbridge_loader_ignores_adjacent_vortex_sidecar(tmp_path):
    """Only DCP entries, never an adjacent legacy sidecar, should affect export."""
    iter_dir = tmp_path / "iter_0000001"
    dcp.save({"weight": torch.tensor([3.0])}, checkpoint_id=str(iter_dir))
    torch.save({"injected": torch.tensor([9.0])}, tmp_path / "vortex_passthrough.pt")

    loaded = load_mbridge_state_dict(tmp_path)

    assert set(loaded) == {"weight"}
    assert torch.equal(loaded["weight"], torch.tensor([3.0]))


def test_vortex_to_mbridge_rejects_missing_required_model_keys():
    """Malformed or wrong-model checkpoints must fail before packaging."""
    with pytest.raises(ValueError, match="missing required MBridge keys"):
        vortex_to_mbridge_state_dict({}, _make_test_provider(), te_enabled=True)


@pytest.mark.parametrize(
    "missing_key",
    [
        "blocks.0.out_filter_dense.bias",
        "blocks.3.inner_mha_cls.out_proj.bias",
    ],
)
def test_vortex_to_mbridge_rejects_missing_learned_bias(missing_key: str):
    """Dropping a learned bias must fail during conversion instead of producing an incomplete checkpoint."""
    vortex_sd = _make_mock_vortex_sd("SDH*")
    vortex_sd.pop(missing_key)

    with pytest.raises(ValueError, match="missing required MBridge keys"):
        vortex_to_mbridge_state_dict(vortex_sd, _make_test_provider(), te_enabled=True)


def test_vortex_to_mbridge_rejects_unmapped_model_tensor():
    """Unknown model tensors must not be silently discarded by a checkpoint conversion."""
    vortex_sd = _make_mock_vortex_sd("SDH*")
    vortex_sd["blocks.0.new_model_weight"] = torch.ones(4)

    with pytest.raises(ValueError, match="unmapped Vortex entries are fatal"):
        vortex_to_mbridge_state_dict(vortex_sd, _make_test_provider(), te_enabled=True)


def test_vortex_to_mbridge_rejects_invalid_mlp_geometry():
    """An invalid gated-MLP shape should fail before writing a DCP checkpoint."""
    vortex_sd = _make_mock_vortex_sd("SDH*")
    vortex_sd["blocks.0.mlp.l1.weight"] = torch.randn(5, 4, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="invalid Vortex gated-MLP geometry"):
        vortex_to_mbridge_state_dict(vortex_sd, _make_test_provider(), te_enabled=True)


def test_vortex_to_mbridge_rejects_invalid_projection_fir_geometry():
    """An invalid projection FIR should fail before writing a DCP checkpoint."""
    vortex_sd = _make_mock_vortex_sd("SDH*")
    vortex_sd["blocks.0.filter.short_filter_weight"] = torch.randn(12, 1, 5, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="invalid Vortex projection FIR"):
        vortex_to_mbridge_state_dict(vortex_sd, _make_test_provider(), te_enabled=True)


def test_vortex_to_mbridge_rejects_invalid_medium_filter_geometry():
    """A medium filter must match the provider's group count and convolution length."""
    vortex_sd = _make_mock_vortex_sd("SDH*")
    vortex_sd["blocks.1.filter.h"] = torch.randn(4, 1, 127, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="invalid Vortex medium filter"):
        vortex_to_mbridge_state_dict(vortex_sd, _make_test_provider(), te_enabled=True)


def test_vortex_to_mbridge_rejects_mismatched_long_filter_geometry():
    """Long-filter residues must align with the Vortex log-pole dimensions."""
    vortex_sd = _make_mock_vortex_sd("SDH*")
    vortex_sd["blocks.2.filter.residues"] = torch.randn(4, 2, dtype=torch.float32)

    with pytest.raises(ValueError, match="invalid Vortex poles/residues"):
        vortex_to_mbridge_state_dict(vortex_sd, _make_test_provider(), te_enabled=True)


def test_vortex_to_mbridge_uses_provider_parameter_dtype():
    """Ordinary model weights should use the target provider dtype in the canonical checkpoint."""
    vortex_sd = _make_mock_vortex_sd("SDH*")
    embedding = torch.randn(8, 4, dtype=torch.float32)
    vortex_sd["embedding_layer.weight"] = embedding
    vortex_sd["unembed.weight"] = embedding.clone()

    result = vortex_to_mbridge_state_dict(vortex_sd, _make_test_provider(), te_enabled=True)

    assert result["embedding.word_embeddings.weight"].dtype == torch.bfloat16


def test_vortex_to_mbridge_rejects_rotary_frequencies_that_disagree_with_provider():
    """A derived rotary buffer must agree with the target provider before it is omitted."""
    vortex_sd = _make_mock_vortex_sd("SDH*")
    vortex_sd["blocks.3.inner_mha_cls.rotary_emb.inv_freq"] = torch.tensor([0.5])

    with pytest.raises(ValueError, match="rotary frequencies"):
        vortex_to_mbridge_state_dict(vortex_sd, _make_test_provider(), te_enabled=True)


def test_rotary_validation_accepts_bfloat16_quantized_values_stored_as_float32():
    """Released FP32 rotary buffers may contain values quantized through BF16."""
    provider = _make_test_provider()
    provider.hidden_size = 8
    rotary = torch.tensor([1.0, 0.01], dtype=torch.bfloat16).float()

    _validate_rotary_frequencies(rotary, provider)


def test_add_mbridge_te_extra_states_initializes_mbridge_runtime_state():
    """Packaging should add MBridge TE extra states without importing Vortex FP8 runtime state."""
    provider = _make_test_provider()
    mbridge_sd = vortex_to_mbridge_state_dict(_make_mock_vortex_sd("SDH*"), provider, te_enabled=True)

    _add_mbridge_te_extra_states(mbridge_sd, provider)

    assert torch.equal(
        mbridge_sd["decoder.layers.0.mixer.dense_projection._extra_state"], torch.empty(0, dtype=torch.uint8)
    )
    assert torch.equal(
        mbridge_sd["decoder.layers.3.self_attention.linear_proj._extra_state"], torch.empty(0, dtype=torch.uint8)
    )
    assert torch.equal(mbridge_sd["decoder.layers.0.mlp.linear_fc1._extra_state"], torch.empty(0, dtype=torch.uint8))
    assert torch.equal(
        mbridge_sd["decoder.layers.3.self_attention.linear_qkv._extra_state"], torch.empty(0, dtype=torch.uint8)
    )
    assert torch.equal(mbridge_sd["decoder.final_norm._extra_state"], torch.empty(0, dtype=torch.uint8))
    assert mbridge_sd["output_layer._extra_state"] is None


def test_mcore_packaging_wraps_extra_state_as_sharded_object():
    """MCore treats module extra state as a replicated ShardedObject, not a tensor shard."""
    state_dict = {
        "decoder.final_norm.weight": torch.randn(4),
        "decoder.final_norm._extra_state": torch.empty(0, dtype=torch.uint8),
        "output_layer._extra_state": None,
    }

    sharded = _to_mcore_sharded_state_dict(state_dict)

    assert isinstance(sharded["decoder.final_norm.weight"], ShardedTensor)
    extra_state = sharded["decoder.final_norm._extra_state"]
    assert isinstance(extra_state, ShardedObject)
    assert extra_state.unique_key == "decoder.final_norm._extra_state/shard_0_1"
    output_extra_state = sharded["output_layer._extra_state"]
    assert isinstance(output_extra_state, ShardedObject)
    assert output_extra_state.unique_key == "output_layer._extra_state/shard_0_1"


def test_long_filter_inverse_reconstructs_log_poles_exactly():
    """The ambiguous long-filter inverse should round-trip log poles exactly."""
    vortex_sd = _make_mock_vortex_sd("SDH*")
    balanced = torch.tensor([[-0.7], [-0.5], [-0.3], [-0.1]], dtype=torch.float32)
    vortex_sd["blocks.2.filter.log_poles"] = (-torch.exp(balanced) * torch.exp(balanced))[..., None]
    result = vortex_to_mbridge_state_dict(vortex_sd, _make_test_provider(), te_enabled=True)
    original = vortex_sd["blocks.2.filter.log_poles"]
    p = result["decoder.layers.2.mixer.mixer.filter.p"].reshape(4, 1)
    gamma = result["decoder.layers.2.mixer.mixer.filter.gamma"]
    reconstructed = (-torch.exp(p) * torch.exp(gamma))[..., None]

    assert torch.equal(reconstructed, original)
    assert torch.allclose(p, gamma, rtol=0.0, atol=1e-6)


def test_vortex_to_mbridge_to_vortex_synthetic_roundtrip():
    """Round-trip a compact Vortex state dict through MBridge conversion."""
    provider = _make_test_provider()
    vortex_sd = _make_mock_vortex_sd("SDH*")
    mbridge_sd = vortex_to_mbridge_state_dict(vortex_sd, provider, te_enabled=True)
    roundtrip_sd = mbridge_to_vortex_state_dict(dict(mbridge_sd), provider, te_enabled=True)

    assert set(roundtrip_sd) == set(vortex_sd)
    for key, original in vortex_sd.items():
        _assert_exact_value_equal(roundtrip_sd[key], original, key)


def _assert_exact_value_equal(actual, expected, key: str) -> None:
    """Assert exact equality except for descriptor-derived rotary frequencies."""
    if isinstance(expected, BytesIO):
        assert isinstance(actual, BytesIO), key
        assert actual.getvalue() == expected.getvalue(), key
    elif key.endswith(".rotary_emb.inv_freq"):
        expected_float = expected.float()
        precision_dtype = expected.dtype
        if expected.dtype == torch.float32 and torch.equal(expected_float, expected_float.to(torch.bfloat16).float()):
            precision_dtype = torch.bfloat16
        tolerance = torch.finfo(precision_dtype).eps
        assert torch.allclose(actual.float(), expected_float, rtol=tolerance, atol=tolerance), key
    else:
        assert torch.equal(actual, expected), key


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.skipif(
    not os.environ.get("LONG_TESTS"),
    reason="Set LONG_TESTS=1 to run (downloads the public 1B Vortex checkpoint)",
)
def test_1b_base_checkpoint_weight_roundtrip(tmp_path):
    """Download the public 1B checkpoint and exactly round-trip its model tensors."""
    cache_dir = os.environ.get("EVO2_CHECKPOINT_CACHE_DIR")
    original_path = download_vortex_checkpoint(
        "arcinstitute/evo2_1b_base",
        filename="evo2_1b_base.pt",
        revision="2279e1df422c991037470302360edd40d0d2ea1e",
        cache_dir=cache_dir or tmp_path,
    )
    provider = HYENA_MODEL_OPTIONS["evo2_1b_base"]()
    original_sd = load_vortex_state_dict(original_path)
    mbridge_sd = vortex_to_mbridge_state_dict(original_sd, provider, te_enabled=True)
    roundtrip_sd = mbridge_to_vortex_state_dict(dict(mbridge_sd), provider, te_enabled=True)

    runtime_suffixes = ("._extra_state", ".filter.t")
    expected_sd = {key: value for key, value in original_sd.items() if not key.endswith(runtime_suffixes)}
    assert set(roundtrip_sd) == set(expected_sd)
    for key, original in expected_sd.items():
        _assert_exact_value_equal(roundtrip_sd[key], original, key)
