# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Tests for the recipe-owned NeMo-RL launcher extensions."""

from bionemo.evo2_phage_gen.run_phage_grpo import _register_recipe_extensions


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
