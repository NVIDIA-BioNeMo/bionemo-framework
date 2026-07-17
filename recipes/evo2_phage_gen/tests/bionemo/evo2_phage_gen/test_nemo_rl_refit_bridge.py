# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from types import SimpleNamespace

import pytest

from nemo_rl.models.megatron import setup


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _policy_config(*, bridge_cls: object = "recipe.bridge.Evo2RefitBridge", backend: str = "vllm") -> dict:
    return {
        "generation": {
            "backend": backend,
            "refit_bridge_cls": bridge_cls,
            "colocated": {"enabled": True},
        }
    }


def test_explicit_vllm_refit_bridge_bypasses_auto_bridge(monkeypatch) -> None:
    training_config = object()
    created = SimpleNamespace(
        get_conversion_tasks=lambda *_args, **_kwargs: [],
        export_hf_weights=lambda *_args, **_kwargs: iter(()),
        transformer_config=training_config,
    )
    calls = []

    class FakeBridge:
        @classmethod
        def from_pretrained(cls, model_name, *, transformer_config):
            calls.append((model_name, transformer_config))
            return created

    monkeypatch.setattr(
        setup.AutoBridge,
        "from_hf_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("AutoBridge must not run")),
    )
    monkeypatch.setattr(
        setup.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(Evo2RefitBridge=FakeBridge),
    )

    result = setup._select_megatron_bridge(
        _policy_config(),
        "/exports/evo2",
        SimpleNamespace(config=training_config),
    )

    _require(result is created, "explicit bridge result was replaced")
    _require(calls == [("/exports/evo2", training_config)], "explicit bridge inputs drifted")


def test_default_vllm_refit_bridge_preserves_auto_bridge(monkeypatch) -> None:
    created = object()
    calls = []
    monkeypatch.setattr(
        setup.AutoBridge,
        "from_hf_pretrained",
        lambda model_name, **kwargs: calls.append((model_name, kwargs)) or created,
    )

    result = setup._select_megatron_bridge(
        {"generation": {"backend": "vllm", "colocated": {"enabled": True}}},
        "generic-model",
        SimpleNamespace(config=object()),
    )

    _require(result is created, "default AutoBridge result was replaced")
    _require(calls == [("generic-model", {"trust_remote_code": True})], "default AutoBridge call drifted")


@pytest.mark.parametrize("bridge_cls", [True, 1, "", "NoModule"])
def test_explicit_refit_bridge_requires_qualified_builtin_string(bridge_cls) -> None:
    with pytest.raises((TypeError, ValueError), match="refit_bridge_cls"):
        setup._select_megatron_bridge(
            _policy_config(bridge_cls=bridge_cls),
            "/exports/evo2",
            SimpleNamespace(config=object()),
        )


def test_explicit_refit_bridge_is_vllm_only() -> None:
    with pytest.raises(ValueError, match="vLLM"):
        setup._select_megatron_bridge(
            _policy_config(backend="megatron"),
            "/exports/evo2",
            SimpleNamespace(config=object()),
        )
