"""Tests for lossless MBridge runtime-state checkpoint upgrades."""

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch

from bionemo.evo2.models.evo2_provider import HyenaTestModelProvider
from bionemo.evo2.utils.checkpoint.mbridge_to_vortex import load_mbridge_state_dict
from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import package_mbridge_checkpoint
from bionemo.evo2.utils.checkpoint.upgrade_mbridge_runtime_state import (
    augment_mbridge_runtime_state,
    upgrade_mbridge_checkpoint,
)


TOKENIZER_PATH = Path(__file__).resolve().parents[5] / "tokenizers/nucleotide_fast_tokenizer_512"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _provider() -> HyenaTestModelProvider:
    return HyenaTestModelProvider(
        hybrid_override_pattern="SDH*",
        hidden_size=4,
        num_groups_hyena=4,
        num_attention_heads=2,
        ffn_hidden_size=6,
    )


def _required_extra_state_keys(pattern: str) -> set[str]:
    keys: set[str] = set()
    for layer_index, symbol in enumerate(pattern):
        prefix = f"decoder.layers.{layer_index}"
        if symbol == "*":
            keys.update(
                {
                    f"{prefix}.self_attention.core_attention._extra_state",
                    f"{prefix}.self_attention.linear_qkv._extra_state",
                    f"{prefix}.self_attention.linear_proj._extra_state",
                }
            )
        else:
            keys.update(
                {
                    f"{prefix}.mixer.dense_projection._extra_state",
                    f"{prefix}.mixer.dense._extra_state",
                }
            )
        keys.update(
            {
                f"{prefix}.mlp.linear_fc1._extra_state",
                f"{prefix}.mlp.linear_fc2._extra_state",
            }
        )
    keys.update({"decoder.final_norm._extra_state", "output_layer._extra_state"})
    return keys


def test_augment_mbridge_runtime_state_preserves_source_and_adds_only_required_keys() -> None:
    provider = _provider()
    existing_extra_key = "decoder.layers.0.mixer.dense_projection._extra_state"
    source: OrderedDict[str, Any] = OrderedDict(
        {
            "decoder.layers.1.mixer.mixer.filter.h": torch.randn(4, 7),
            "decoder.layers.2.mixer.mixer.filter.p": torch.randn(4, 1),
            existing_extra_key: torch.empty(0, dtype=torch.uint8),
        }
    )
    original_keys = tuple(source)
    original_values = dict(source)

    augmentation = augment_mbridge_runtime_state(source, provider)

    upgraded = augmentation.state_dict
    _require(tuple(source) == original_keys, "augmentation mutated source keys")
    for key, value in original_values.items():
        _require(source[key] is value, f"augmentation replaced source value {key}")
        _require(upgraded[key] is value, f"upgrade copied or replaced source value {key}")

    rotary_key = "decoder.layers.3.self_attention.rotary_emb.inv_freq"
    expected_added = (_required_extra_state_keys(provider.hybrid_override_pattern) - {existing_extra_key}) | {
        rotary_key
    }
    _require(set(augmentation.added_keys) == expected_added, "added runtime-state key inventory mismatch")
    _require(set(upgraded) == set(source) | expected_added, "upgrade added an unexpected key")
    for key in _required_extra_state_keys(provider.hybrid_override_pattern) - {"output_layer._extra_state"}:
        value = upgraded[key]
        _require(type(value) is torch.Tensor, f"{key} must be a tensor")
        _require(value.dtype is torch.uint8 and value.numel() == 0, f"{key} must be empty uint8")
    _require(upgraded["output_layer._extra_state"] is None, "output-layer extra state must be None")

    rotary_dim = provider.hidden_size // provider.num_attention_heads
    expected_inv_freq = 1.0 / (
        float(provider.rotary_base)
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    _require(torch.equal(upgraded[rotary_key], expected_inv_freq), "derived RoPE inverse frequencies differ")


def test_augment_mbridge_runtime_state_preserves_matching_existing_rope() -> None:
    provider = _provider()
    rotary_key = "decoder.layers.3.self_attention.rotary_emb.inv_freq"
    rotary_dim = provider.hidden_size // provider.num_attention_heads
    inv_freq = 1.0 / (
        float(provider.rotary_base)
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    source = OrderedDict({rotary_key: inv_freq})

    augmentation = augment_mbridge_runtime_state(source, provider)

    _require(augmentation.state_dict[rotary_key] is inv_freq, "matching source RoPE buffer was replaced")
    _require(rotary_key not in augmentation.added_keys, "existing RoPE buffer was reported as added")


def test_augment_mbridge_runtime_state_rejects_conflicting_rope_without_mutation() -> None:
    provider = _provider()
    rotary_key = "decoder.layers.3.self_attention.rotary_emb.inv_freq"
    conflicting = torch.full((1,), 7.0, dtype=torch.float32)
    source = OrderedDict({rotary_key: conflicting})
    before = tuple(source.items())

    with pytest.raises(ValueError, match="conflicts with provider-derived value"):
        augment_mbridge_runtime_state(source, provider)

    _require(tuple(source.items()) == before, "rejected augmentation mutated its source")


def test_upgrade_mbridge_checkpoint_reopens_losslessly_and_refuses_overwrite(tmp_path: Path) -> None:
    provider = _provider()
    source_state = OrderedDict(
        {
            "embedding.word_embeddings.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            "decoder.layers.1.mixer.mixer.filter.h": torch.randn(4, 7, dtype=torch.float32),
            "decoder.layers.1.mixer.mixer.filter.decay": torch.randn(4, 7, dtype=torch.float32),
            "decoder.layers.2.mixer.mixer.filter.p": torch.randn(4, 1, dtype=torch.float32),
            "decoder.layers.2.mixer.mixer.filter.gamma": torch.randn(4, 1, dtype=torch.float32),
            "decoder.final_norm.weight": torch.randn(4, dtype=torch.bfloat16),
        }
    )
    source = tmp_path / "legacy"
    destination = tmp_path / "current"
    package_mbridge_checkpoint(source_state, source, provider, TOKENIZER_PATH)
    reopened_source = load_mbridge_state_dict(source)

    receipt = upgrade_mbridge_checkpoint(
        source_checkpoint=source,
        destination_checkpoint=destination,
        model_provider=_provider(),
        tokenizer_path=TOKENIZER_PATH,
    )

    reopened_destination = load_mbridge_state_dict(destination)
    _require(receipt.source_checkpoint == source.resolve(), "receipt source mismatch")
    _require(receipt.destination_checkpoint == destination.resolve(), "receipt destination mismatch")
    _require(receipt.source_key_count == len(reopened_source), "receipt source key count mismatch")
    _require(receipt.destination_key_count == len(reopened_destination), "receipt destination key count mismatch")
    _require(
        set(receipt.added_keys) == set(reopened_destination) - set(reopened_source),
        "receipt added-key inventory mismatch",
    )
    for key, expected in reopened_source.items():
        actual = reopened_destination[key]
        _require(type(actual) is type(expected), f"reopened type changed for {key}")
        if type(expected) is torch.Tensor:
            _require(torch.equal(actual, expected), f"reopened tensor changed for {key}")
        else:
            _require(actual == expected, f"reopened value changed for {key}")

    source_run_config = source / "iter_0000001/run_config.yaml"
    destination_run_config = destination / "iter_0000001/run_config.yaml"
    _require(
        receipt.source_run_config_sha256 == hashlib.sha256(source_run_config.read_bytes()).hexdigest(),
        "source run-config digest mismatch",
    )
    _require(
        receipt.destination_run_config_sha256 == hashlib.sha256(destination_run_config.read_bytes()).hexdigest(),
        "destination run-config digest mismatch",
    )
    metadata_before = (destination / "iter_0000001/.metadata").read_bytes()
    with pytest.raises(FileExistsError):
        upgrade_mbridge_checkpoint(
            source_checkpoint=source,
            destination_checkpoint=destination,
            model_provider=_provider(),
            tokenizer_path=TOKENIZER_PATH,
        )
    _require(
        (destination / "iter_0000001/.metadata").read_bytes() == metadata_before,
        "refused overwrite modified destination metadata",
    )
