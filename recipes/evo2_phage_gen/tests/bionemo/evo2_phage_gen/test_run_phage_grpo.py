# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Tests for the recipe-owned NeMo-RL launcher extensions."""

import inspect
from types import SimpleNamespace

from bionemo.evo2.vllm import load_parity
from bionemo.evo2_phage_gen import run_phage_grpo
from bionemo.evo2_phage_gen.run_phage_grpo import (
    _bind_vllm_prompt_group_sharding,
    _register_recipe_extensions,
    _unpack_grpo_setup_result,
    _validate_vllm_load_parity,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_unpack_grpo_setup_result_accepts_pinned_thirteen_value_contract() -> None:
    expected = tuple(object() for _ in range(13))

    actual = _unpack_grpo_setup_result(expected)

    _require(actual == expected, "launcher did not preserve the pinned setup result order")


def test_register_recipe_extensions_registers_evo2_vllm_worker(tmp_path, monkeypatch) -> None:
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY

    worker_fqn = "bionemo.evo2.vllm.nemo_generation_worker.Evo2NemoRlGenerationWorker"
    actor_python = tmp_path / worker_fqn / "bin" / "python"
    actor_python.parent.mkdir(parents=True)
    actor_python.write_text("")
    monkeypatch.setenv("NEMO_RL_VENV_DIR", str(tmp_path))
    monkeypatch.delitem(ACTOR_ENVIRONMENT_REGISTRY, worker_fqn, raising=False)

    _register_recipe_extensions()

    _require(worker_fqn in ACTOR_ENVIRONMENT_REGISTRY, "Evo2 vLLM worker environment was not registered")
    _require(
        ACTOR_ENVIRONMENT_REGISTRY[worker_fqn] == str(actor_python),
        "Evo2 vLLM worker did not use the isolated pinned actor environment",
    )


def test_launcher_resolves_and_validates_vllm_load_parity(monkeypatch) -> None:
    captured: dict[str, object] = {}
    expected = {
        "export_manifest_sha256": "a" * 64,
        "source_run_config_sha256": "b" * 64,
        "tensor_count": 330,
        "tokenizer_semantic_sha256": "c" * 64,
    }

    def _capture(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(load_parity, "validate_rl_inference_load_parity", _capture)
    config = SimpleNamespace(
        policy={
            "model_name": "data/checkpoints/export",
            "tokenizer": {"name": "tokenizers/canonical"},
            "generation": {"backend": "vllm"},
        },
        checkpointing={
            "pretrained_checkpoint": {"path": "data/checkpoints/mbridge"},
        },
    )

    actual = _validate_vllm_load_parity(config)

    _require(actual == expected, "launcher changed load-parity evidence")
    _require(
        captured["checkpoint"] == run_phage_grpo.RECIPE_ROOT / "data/checkpoints/mbridge",
        "launcher did not resolve the MBridge checkpoint recipe-relatively",
    )
    _require(
        captured["export"] == run_phage_grpo.RECIPE_ROOT / "data/checkpoints/export",
        "launcher did not resolve the vLLM export recipe-relatively",
    )
    _require(
        captured["rl_tokenizer"] == run_phage_grpo.RECIPE_ROOT / "tokenizers/canonical",
        "launcher did not resolve the RL tokenizer recipe-relatively",
    )


def test_launcher_preserves_non_vllm_generation_without_parity_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        load_parity,
        "validate_rl_inference_load_parity",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("non-vLLM route invoked parity")),
    )
    config = SimpleNamespace(
        policy={"model_name": "mcore", "tokenizer": {"name": "tokenizer"}, "generation": {"backend": "megatron"}},
        checkpointing={"pretrained_checkpoint": {"path": "checkpoint"}},
    )

    _require(_validate_vllm_load_parity(config) is None, "non-vLLM generation gained parity side effects")


def test_launcher_binds_vllm_dp_sharding_to_rollouts_per_prompt() -> None:
    config = SimpleNamespace(
        grpo={"num_generations_per_prompt": 12},
        policy={"generation": {"backend": "vllm"}},
    )

    _bind_vllm_prompt_group_sharding(config)

    _require(
        config.policy["generation"]["dp_shard_batch_size"] == 12,
        "vLLM DP sharding was not bound to K",
    )

    config.policy["generation"]["dp_shard_batch_size"] = 6
    try:
        _bind_vllm_prompt_group_sharding(config)
    except ValueError as error:
        _require(
            "num_generations_per_prompt" in str(error),
            "mismatch error lost its authority",
        )
    else:
        raise AssertionError("a conflicting vLLM prompt-group shard size was accepted")


def test_launcher_validates_vllm_load_parity_before_ray_initialization() -> None:
    source = inspect.getsource(run_phage_grpo.main)
    parity_position = source.find("_validate_vllm_load_parity(config)")
    ray_position = source.find("    init_ray()")

    _require(
        parity_position >= 0,
        "production launcher does not call the load-parity preflight",
    )
    _require(
        ray_position >= 0,
        "production launcher no longer exposes the Ray initialization boundary",
    )
    _require(
        parity_position < ray_position,
        "load parity runs after Ray can allocate resources",
    )
