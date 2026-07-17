# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import json

import pytest
import torch
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from safetensors.torch import save_file

from bionemo.evo2.vllm.config import Evo2Config
from bionemo.evo2.vllm.refit_bridge import Evo2RefitBridge


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_checkpoint(tmp_path, names: tuple[str, ...]) -> None:
    Evo2Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=128,
        hybrid_override_pattern="S",
        num_groups_hyena=8,
        num_groups_hyena_short=8,
        num_groups_hyena_medium=8,
    ).save_pretrained(tmp_path)
    shard_name = "model-00001-of-00001.safetensors"
    save_file({name: torch.ones(2, 2) for name in names}, tmp_path / shard_name)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard_name for name in names}}),
        encoding="utf-8",
    )


def test_evo2_refit_bridge_uses_exact_same_name_tp_mappings(tmp_path) -> None:
    names = (
        "embedding.word_embeddings.weight",
        "decoder.layers.0.mixer.dense_projection.weight",
        "decoder.layers.0.mixer.dense.weight",
        "decoder.layers.0.mixer.dense.bias",
        "decoder.final_norm.weight",
    )
    _write_checkpoint(tmp_path, names)
    transformer_config = object()

    bridge = Evo2RefitBridge.from_pretrained(
        tmp_path,
        transformer_config=transformer_config,
    )
    registry = bridge.mapping_registry()

    expected_types = {
        names[0]: ColumnParallelMapping,
        names[1]: ColumnParallelMapping,
        names[2]: RowParallelMapping,
        names[3]: ReplicatedMapping,
        names[4]: ReplicatedMapping,
    }
    for name, expected_type in expected_types.items():
        mapping = registry.megatron_to_hf_lookup(name)
        _require(type(mapping) is expected_type, f"wrong mapping class for {name}")
        _require(mapping.hf_param == name, f"refit renamed {name}")
    _require(bridge.transformer_config is transformer_config, "training model config was not retained")
    _require(bridge.expected_weight_names == frozenset(names), "indexed weight inventory drifted")


def test_evo2_refit_bridge_rejects_unknown_indexed_weight(tmp_path) -> None:
    _write_checkpoint(tmp_path, ("decoder.layers.0.foreign.weight",))

    with pytest.raises(ValueError, match="unsupported Evo2 refit weights"):
        Evo2RefitBridge.from_pretrained(tmp_path, transformer_config=object())


def test_evo2_refit_bridge_rejects_duplicate_or_incomplete_stream(monkeypatch, tmp_path) -> None:
    names = ("decoder.final_norm.weight", "embedding.word_embeddings.weight")
    _write_checkpoint(tmp_path, names)
    bridge = Evo2RefitBridge.from_pretrained(tmp_path, transformer_config=object())

    monkeypatch.setattr(
        bridge,
        "_stream_converted_weights",
        lambda *_args, **_kwargs: iter(((names[0], torch.ones(1)), (names[0], torch.ones(1)))),
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        list(bridge.export_hf_weights([object()], conversion_tasks=[]))
