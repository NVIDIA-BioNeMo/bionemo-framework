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

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import bionemo.evo2.vllm.weights as weights_module
from bionemo.evo2.vllm.weights import IncrementalEvo2WeightLoader, load_evo2_weights, load_tensor_parallel_weight


PATTERN = "SDH*"


def _mcore_group_major_qkv_reference(
    tensor: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    q_heads_per_group = num_attention_heads // num_key_value_heads
    grouped = tensor.reshape(num_key_value_heads, q_heads_per_group + 2, head_dim, *tensor.shape[1:])
    query = grouped[:, :q_heads_per_group].reshape(num_attention_heads * head_dim, *tensor.shape[1:])
    key = grouped[:, q_heads_per_group].reshape(num_key_value_heads * head_dim, *tensor.shape[1:])
    value = grouped[:, q_heads_per_group + 1].reshape(num_key_value_heads * head_dim, *tensor.shape[1:])
    return torch.cat((query, key, value), dim=0)


def test_mcore_group_major_qkv_is_permuted_to_vllm_global_qkv_for_weight_and_bias() -> None:
    weight = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    bias = torch.arange(16, dtype=torch.float32)
    expected_weight_rows = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]

    converted_weight = weights_module.mcore_grouped_qkv_to_vllm(
        weight,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
    )
    converted_bias = weights_module.mcore_grouped_qkv_to_vllm(
        bias,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
    )

    torch.testing.assert_close(converted_weight, weight[expected_weight_rows])
    torch.testing.assert_close(converted_bias, bias[expected_weight_rows])


def _make_qkv_transaction_model() -> nn.Module:
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=8, num_attention_heads=4, num_key_value_heads=2)
    decoder = _child(model, "decoder")
    layers = nn.ModuleList([nn.Module()])
    decoder.add_module("layers", layers)
    self_attention = _child(layers[0], "self_attention")
    linear_qkv = _child(self_attention, "linear_qkv")
    linear_qkv.register_parameter("weight", nn.Parameter(torch.zeros(16, 3)))
    linear_qkv.register_parameter("bias", nn.Parameter(torch.zeros(16)))
    return model


def test_mcore_qkv_permutation_applies_to_initial_load_and_every_incremental_refit() -> None:
    model = _make_qkv_transaction_model()
    loader = IncrementalEvo2WeightLoader(model)
    weight_name = "decoder.layers.0.self_attention.linear_qkv.weight"
    bias_name = "decoder.layers.0.self_attention.linear_qkv.bias"
    initial_weight = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    initial_bias = torch.arange(16, dtype=torch.float32)

    loader.load([(weight_name, initial_weight)])
    with pytest.raises(RuntimeError, match="initial weight load is incomplete"):
        loader.assert_ready_for_inference()
    loader.load([(bias_name, initial_bias)])
    loader.assert_ready_for_inference()
    assert loader.completed_transactions == 1
    torch.testing.assert_close(
        model.decoder.layers[0].self_attention.linear_qkv.weight,
        _mcore_group_major_qkv_reference(
            initial_weight,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
        ),
    )
    torch.testing.assert_close(
        model.decoder.layers[0].self_attention.linear_qkv.bias,
        _mcore_group_major_qkv_reference(
            initial_bias,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
        ),
    )

    refit_weight = initial_weight + 1_000
    refit_bias = initial_bias + 1_000
    loader.load([(bias_name, refit_bias)])
    with pytest.raises(RuntimeError, match="refit weight load is incomplete"):
        loader.assert_ready_for_inference()
    loader.load([(weight_name, refit_weight)])
    loader.assert_ready_for_inference()
    assert loader.completed_transactions == 2
    torch.testing.assert_close(
        model.decoder.layers[0].self_attention.linear_qkv.weight,
        _mcore_group_major_qkv_reference(
            refit_weight,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
        ),
    )
    torch.testing.assert_close(
        model.decoder.layers[0].self_attention.linear_qkv.bias,
        _mcore_group_major_qkv_reference(
            refit_bias,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
        ),
    )


def _child(parent, name):
    module = nn.Module()
    parent.add_module(name, module)
    return module


def _add_parameter(module, name, global_shape, *, tp_rank, tp_size, shard_dim=None):
    local_shape = list(global_shape)
    if shard_dim is not None:
        assert global_shape[shard_dim] % tp_size == 0
        local_shape[shard_dim] //= tp_size
    parameter = nn.Parameter(torch.zeros(tuple(local_shape), dtype=torch.float32))
    module.register_parameter(name, parameter)
    parameter._evo2_global_shape = tuple(global_shape)
    parameter._evo2_shard_dim = shard_dim
    if shard_dim is not None:
        parameter.weight_loader = partial(
            load_tensor_parallel_weight,
            shard_dim=shard_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
    return parameter


def _make_synthetic_model(*, tp_rank=0, tp_size=1):
    hidden = 4
    intermediate = 6
    groups = 4
    state_size = 2
    medium_taps = 4
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=hidden, num_attention_heads=2, num_key_value_heads=2)
    model.hybrid_override_pattern = PATTERN
    embedding = _child(model, "embedding")
    word_embeddings = nn.Embedding(8 // tp_size, hidden)
    embedding.add_module("word_embeddings", word_embeddings)
    word_embeddings.weight = _add_parameter(
        word_embeddings,
        "weight",
        (8, hidden),
        tp_rank=tp_rank,
        tp_size=tp_size,
        shard_dim=0,
    )

    decoder = _child(model, "decoder")
    layers = nn.ModuleList()
    decoder.add_module("layers", layers)
    for symbol in PATTERN:
        layer = nn.Module()
        layers.append(layer)
        if symbol == "*":
            self_attention = _child(layer, "self_attention")
            linear_qkv = _child(self_attention, "linear_qkv")
            _add_parameter(linear_qkv, "layer_norm_weight", (hidden,), tp_rank=tp_rank, tp_size=tp_size)
            _add_parameter(
                linear_qkv,
                "weight",
                (3 * hidden, hidden),
                tp_rank=tp_rank,
                tp_size=tp_size,
                shard_dim=0,
            )
            linear_proj = _child(self_attention, "linear_proj")
            _add_parameter(
                linear_proj,
                "weight",
                (hidden, hidden),
                tp_rank=tp_rank,
                tp_size=tp_size,
                shard_dim=1,
            )
            _add_parameter(linear_proj, "bias", (hidden,), tp_rank=tp_rank, tp_size=tp_size)
        else:
            mixer = _child(layer, "mixer")
            dense_projection = _child(mixer, "dense_projection")
            _add_parameter(dense_projection, "layer_norm_weight", (hidden,), tp_rank=tp_rank, tp_size=tp_size)
            _add_parameter(
                dense_projection,
                "weight",
                (3 * hidden, hidden),
                tp_rank=tp_rank,
                tp_size=tp_size,
                shard_dim=0,
            )
            projection_conv = _child(mixer, "hyena_proj_conv")
            _add_parameter(
                projection_conv,
                "short_conv_weight",
                (3 * hidden, 3),
                tp_rank=tp_rank,
                tp_size=tp_size,
                shard_dim=0,
            )
            dense = _child(mixer, "dense")
            _add_parameter(
                dense,
                "weight",
                (hidden, hidden),
                tp_rank=tp_rank,
                tp_size=tp_size,
                shard_dim=1,
            )
            _add_parameter(dense, "bias", (hidden,), tp_rank=tp_rank, tp_size=tp_size)
            operator = _child(mixer, "mixer")
            if symbol == "S":
                short_conv = _child(operator, "short_conv")
                _add_parameter(
                    short_conv,
                    "short_conv_weight",
                    (groups, 1, 7),
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    shard_dim=0,
                )
            elif symbol == "D":
                _add_parameter(
                    operator,
                    "conv_bias",
                    (hidden,),
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    shard_dim=0,
                )
                filter_module = _child(operator, "filter")
                h = _add_parameter(
                    filter_module,
                    "h",
                    (groups, medium_taps),
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    shard_dim=0,
                )
                _add_parameter(
                    filter_module,
                    "decay",
                    (groups, medium_taps),
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    shard_dim=0,
                )
                filter_module.register_buffer("effective_filter", torch.empty_like(h), persistent=False)
            elif symbol == "H":
                _add_parameter(
                    operator,
                    "conv_bias",
                    (hidden,),
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    shard_dim=0,
                )
                filter_module = _child(operator, "filter")
                p = _add_parameter(
                    filter_module,
                    "p",
                    (hidden, state_size),
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    shard_dim=0,
                )
                _add_parameter(
                    filter_module,
                    "gamma",
                    (hidden, state_size),
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    shard_dim=0,
                )
                _add_parameter(
                    filter_module,
                    "R",
                    (hidden, state_size),
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    shard_dim=0,
                )
                filter_module.register_buffer("modal_decay", torch.empty_like(p), persistent=False)

        mlp = _child(layer, "mlp")
        linear_fc1 = _child(mlp, "linear_fc1")
        _add_parameter(linear_fc1, "layer_norm_weight", (hidden,), tp_rank=tp_rank, tp_size=tp_size)
        _add_parameter(
            linear_fc1,
            "weight",
            (2 * intermediate, hidden),
            tp_rank=tp_rank,
            tp_size=tp_size,
            shard_dim=0,
        )
        linear_fc2 = _child(mlp, "linear_fc2")
        _add_parameter(
            linear_fc2,
            "weight",
            (hidden, intermediate),
            tp_rank=tp_rank,
            tp_size=tp_size,
            shard_dim=1,
        )

    final_norm = _child(decoder, "final_norm")
    _add_parameter(final_norm, "weight", (hidden,), tp_rank=tp_rank, tp_size=tp_size)
    return model


def _make_source_weights():
    model = _make_synthetic_model()
    weights = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        weights[name] = values * 0.001 + index * 0.01
    return weights


def _expected_mcore_parameter(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name.endswith("self_attention.linear_qkv.weight"):
        return _mcore_group_major_qkv_reference(
            tensor,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=2,
        )
    return tensor


def _make_vortex_weights(source):
    vortex = {
        "embedding_layer.weight": source["embedding.word_embeddings.weight"],
        "unembed.weight": source["embedding.word_embeddings.weight"].clone(),
        "norm.scale": source["decoder.final_norm.weight"],
    }
    for layer_index, symbol in enumerate(PATTERN):
        prefix = f"decoder.layers.{layer_index}"
        block = f"blocks.{layer_index}"
        if symbol == "*":
            vortex[f"{block}.pre_norm.scale"] = source[f"{prefix}.self_attention.linear_qkv.layer_norm_weight"]
            vortex[f"{block}.inner_mha_cls.Wqkv.weight"] = _expected_mcore_parameter(
                f"{prefix}.self_attention.linear_qkv.weight",
                source[f"{prefix}.self_attention.linear_qkv.weight"],
            )
            vortex[f"{block}.inner_mha_cls.out_proj.weight"] = source[f"{prefix}.self_attention.linear_proj.weight"]
            vortex[f"{block}.inner_mha_cls.out_proj.bias"] = source[f"{prefix}.self_attention.linear_proj.bias"]
        else:
            vortex[f"{block}.pre_norm.scale"] = source[f"{prefix}.mixer.dense_projection.layer_norm_weight"]
            vortex[f"{block}.projections.weight"] = source[f"{prefix}.mixer.dense_projection.weight"]
            vortex[f"{block}.filter.short_filter_weight"] = source[
                f"{prefix}.mixer.hyena_proj_conv.short_conv_weight"
            ].unsqueeze(1)
            vortex[f"{block}.out_filter_dense.weight"] = source[f"{prefix}.mixer.dense.weight"]
            vortex[f"{block}.out_filter_dense.bias"] = source[f"{prefix}.mixer.dense.bias"]
            if symbol == "S":
                vortex[f"{block}.filter.h"] = source[f"{prefix}.mixer.mixer.short_conv.short_conv_weight"]
            elif symbol == "D":
                vortex[f"{block}.filter.D"] = source[f"{prefix}.mixer.mixer.conv_bias"]
                vortex[f"{block}.filter.h"] = (
                    source[f"{prefix}.mixer.mixer.filter.h"] * source[f"{prefix}.mixer.mixer.filter.decay"]
                ).unsqueeze(1)
            elif symbol == "H":
                vortex[f"{block}.filter.D"] = source[f"{prefix}.mixer.mixer.conv_bias"]
                vortex[f"{block}.filter.log_poles"] = (
                    -torch.exp(source[f"{prefix}.mixer.mixer.filter.p"] + source[f"{prefix}.mixer.mixer.filter.gamma"])
                ).unsqueeze(-1)
                vortex[f"{block}.filter.residues"] = source[f"{prefix}.mixer.mixer.filter.R"]

        vortex[f"{block}.post_norm.scale"] = source[f"{prefix}.mlp.linear_fc1.layer_norm_weight"]
        fc1 = source[f"{prefix}.mlp.linear_fc1.weight"]
        vortex[f"{block}.mlp.l1.weight"], vortex[f"{block}.mlp.l2.weight"] = fc1.chunk(2, dim=0)
        vortex[f"{block}.mlp.l3.weight"] = source[f"{prefix}.mlp.linear_fc2.weight"]
    return vortex


def _expected_shard(source, parameter, *, tp_rank, tp_size):
    shard_dim = parameter._evo2_shard_dim
    if shard_dim is None:
        return source
    shard_size = source.shape[shard_dim] // tp_size
    return source.narrow(shard_dim, tp_rank * shard_size, shard_size)


def _assert_derived_buffers(model, source):
    hcm = model.decoder.layers[1].mixer.mixer.filter
    expected_hcm = (
        source["decoder.layers.1.mixer.mixer.filter.h"] * source["decoder.layers.1.mixer.mixer.filter.decay"]
    )
    expected_hcm = _expected_shard(expected_hcm, hcm.h, tp_rank=getattr(model, "tp_rank", 0), tp_size=1)
    torch.testing.assert_close(hcm.effective_filter, expected_hcm)

    hcl = model.decoder.layers[2].mixer.mixer.filter
    expected_hcl = torch.exp(
        -torch.exp(
            source["decoder.layers.2.mixer.mixer.filter.p"] + source["decoder.layers.2.mixer.mixer.filter.gamma"]
        )
    )
    torch.testing.assert_close(hcl.modal_decay, expected_hcl)


def test_load_mbridge_weights_is_order_independent_and_refreshes_filters():
    source = _make_source_weights()
    normal_model = _make_synthetic_model()
    reversed_model = _make_synthetic_model()

    loaded = load_evo2_weights(normal_model, source.items())
    reversed_loaded = load_evo2_weights(reversed_model, reversed(list(source.items())))

    assert loaded == set(source)
    assert reversed_loaded == set(source)
    for name, parameter in normal_model.named_parameters():
        torch.testing.assert_close(parameter, _expected_mcore_parameter(name, source[name]))
        torch.testing.assert_close(parameter, dict(reversed_model.named_parameters())[name])
    _assert_derived_buffers(normal_model, source)
    _assert_derived_buffers(reversed_model, source)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_tp2_loads_expected_axis_shards(tp_rank):
    source = _make_source_weights()
    model = _make_synthetic_model(tp_rank=tp_rank, tp_size=2)

    load_evo2_weights(model, source.items())

    for name, parameter in model.named_parameters():
        expected = _expected_shard(
            _expected_mcore_parameter(name, source[name]),
            parameter,
            tp_rank=tp_rank,
            tp_size=2,
        )
        torch.testing.assert_close(parameter, expected)
    hcm = model.decoder.layers[1].mixer.mixer.filter
    expected_hcm = (
        source["decoder.layers.1.mixer.mixer.filter.h"] * source["decoder.layers.1.mixer.mixer.filter.decay"]
    )
    torch.testing.assert_close(
        hcm.effective_filter,
        _expected_shard(expected_hcm, hcm.h, tp_rank=tp_rank, tp_size=2),
    )
    hcl = model.decoder.layers[2].mixer.mixer.filter
    expected_hcl = torch.exp(
        -torch.exp(
            source["decoder.layers.2.mixer.mixer.filter.p"] + source["decoder.layers.2.mixer.mixer.filter.gamma"]
        )
    )
    torch.testing.assert_close(
        hcl.modal_decay,
        _expected_shard(expected_hcl, hcl.p, tp_rank=tp_rank, tp_size=2),
    )


def test_native_vortex_names_load_equivalent_parameters_and_filters():
    source = _make_source_weights()
    vortex = _make_vortex_weights(source)
    model = _make_synthetic_model()

    loaded = load_evo2_weights(model, reversed(list(vortex.items())))

    assert loaded == set(dict(model.named_parameters()))
    parameters = dict(model.named_parameters())
    hcm_prefix = "decoder.layers.1.mixer.mixer.filter"
    effective_hcm = source[f"{hcm_prefix}.h"] * source[f"{hcm_prefix}.decay"]
    torch.testing.assert_close(parameters[f"{hcm_prefix}.h"], effective_hcm)
    torch.testing.assert_close(parameters[f"{hcm_prefix}.decay"], torch.ones_like(effective_hcm))
    torch.testing.assert_close(model.decoder.layers[1].mixer.mixer.filter.effective_filter, effective_hcm)

    hcl_prefix = "decoder.layers.2.mixer.mixer.filter"
    combined_log_parameter = source[f"{hcl_prefix}.p"] + source[f"{hcl_prefix}.gamma"]
    torch.testing.assert_close(parameters[f"{hcl_prefix}.p"], combined_log_parameter)
    torch.testing.assert_close(parameters[f"{hcl_prefix}.gamma"], torch.zeros_like(combined_log_parameter))
    torch.testing.assert_close(parameters[f"{hcl_prefix}.R"], source[f"{hcl_prefix}.R"])
    torch.testing.assert_close(
        model.decoder.layers[2].mixer.mixer.filter.modal_decay,
        torch.exp(-torch.exp(combined_log_parameter)),
    )

    replaced = {
        f"{hcm_prefix}.h",
        f"{hcm_prefix}.decay",
        f"{hcl_prefix}.p",
        f"{hcl_prefix}.gamma",
    }
    for name, parameter in parameters.items():
        if name not in replaced:
            torch.testing.assert_close(parameter, _expected_mcore_parameter(name, source[name]))


def test_partial_refit_refreshes_after_either_derived_source_changes():
    source = _make_source_weights()
    model = _make_synthetic_model()
    h_key = "decoder.layers.1.mixer.mixer.filter.h"
    decay_key = "decoder.layers.1.mixer.mixer.filter.decay"
    p_key = "decoder.layers.2.mixer.mixer.filter.p"
    gamma_key = "decoder.layers.2.mixer.mixer.filter.gamma"

    load_evo2_weights(model, [(h_key, source[h_key])], strict=False)
    torch.testing.assert_close(
        model.decoder.layers[1].mixer.mixer.filter.effective_filter, torch.zeros_like(source[h_key])
    )
    load_evo2_weights(model, [(decay_key, source[decay_key])], strict=False)
    torch.testing.assert_close(
        model.decoder.layers[1].mixer.mixer.filter.effective_filter,
        source[h_key] * source[decay_key],
    )

    load_evo2_weights(model, [(p_key, source[p_key])], strict=False)
    expected_after_p = torch.exp(-torch.exp(source[p_key]))
    torch.testing.assert_close(model.decoder.layers[2].mixer.mixer.filter.modal_decay, expected_after_p)
    load_evo2_weights(model, [(gamma_key, source[gamma_key])], strict=False)
    torch.testing.assert_close(
        model.decoder.layers[2].mixer.mixer.filter.modal_decay,
        torch.exp(-torch.exp(source[p_key] + source[gamma_key])),
    )


def test_unknown_and_missing_mandatory_weights_are_rejected():
    source = _make_source_weights()
    with pytest.raises(ValueError, match="decoder.layers.0.unknown.weight"):
        load_evo2_weights(
            _make_synthetic_model(), [*source.items(), ("decoder.layers.0.unknown.weight", torch.ones(1))]
        )

    missing_name = "decoder.final_norm.weight"
    incomplete = [(name, tensor) for name, tensor in source.items() if name != missing_name]
    with pytest.raises(ValueError, match=missing_name):
        load_evo2_weights(_make_synthetic_model(), incomplete)


def test_incremental_loader_requires_complete_initial_load_and_each_refit():
    source = _make_source_weights()
    model = _make_synthetic_model()
    loader = IncrementalEvo2WeightLoader(model)
    chunks = [list(source.items())[index : index + 7] for index in range(0, len(source), 7)]

    loader.load(chunks[0])
    with pytest.raises(RuntimeError, match="initial weight load is incomplete"):
        loader.assert_ready_for_inference()

    for chunk in chunks[1:]:
        loader.load(chunk)
    loader.assert_ready_for_inference()
    assert loader.completed_transactions == 1
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, _expected_mcore_parameter(name, source[name]))

    refit_source = {name: tensor + 0.25 for name, tensor in source.items()}
    refit_items = list(refit_source.items())
    loader.load(refit_items[:-1])
    with pytest.raises(RuntimeError, match="refit weight load is incomplete"):
        loader.assert_ready_for_inference()

    loader.load(refit_items[-1:])
    loader.assert_ready_for_inference()
    assert loader.completed_transactions == 2
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, _expected_mcore_parameter(name, refit_source[name]))


def test_incremental_loader_preserves_vortex_fc1_fusion_across_chunks():
    source = _make_source_weights()
    model = _make_synthetic_model()
    loader = IncrementalEvo2WeightLoader(model)
    loaded = set()

    for item in _make_vortex_weights(source).items():
        loaded.update(loader.load([item]))

    loader.assert_ready_for_inference()
    assert loaded == set(dict(model.named_parameters()))
    assert loader.completed_transactions == 1
