# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Contracts for the recipe-owned pinned NeMo-RL vLLM actor environment."""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

from bionemo.evo2_phage_gen import nemo_rl_patches, run_phage_grpo


EVO2_VLLM_WORKER = "bionemo.evo2.vllm.nemo_generation_worker.Evo2NemoRlGenerationWorker"
VLLM_ACTORS = (
    EVO2_VLLM_WORKER,
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
    "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
    "nemo_rl.algorithms.async_utils.ReplayBuffer",
    "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor",
)
SYSTEM_ACTORS = (
    "bionemo.evo2_phage_gen.nemo_rl_env.PhageQCEnvironment",
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_clone_pinned_nemo_rl_initializes_exact_recursive_submodules(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(nemo_rl_patches.subprocess, "run", fake_run)
    destination = tmp_path / "source"

    result = nemo_rl_patches._clone_nemo_rl_source("https://example.invalid/nemo-rl.git", "abc123", destination)

    _require(result == destination, "clone helper returned the wrong persistent source path")
    _require(
        calls
        == [
            ["git", "clone", "--filter=blob:none", "https://example.invalid/nemo-rl.git", str(destination)],
            ["git", "-C", str(destination), "checkout", "abc123"],
            ["git", "-C", str(destination), "submodule", "update", "--init", "--recursive"],
        ],
        f"pinned clone did not initialize the exact workspace recursively: {calls}",
    )


def test_repair_install_retains_exact_source_and_uses_editable_main_install(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "nemo-rl-source"
    source_root.mkdir()
    pyproject = source_root / "pyproject.toml"
    original_pyproject = '[project]\nrequires-python = ">=3.13.13,<3.14"\n'
    pyproject.write_text(original_pyproject)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        nemo_rl_patches,
        "_ensure_pinned_nemo_rl_source",
        lambda *, force_reinstall: source_root,
    )

    def fake_run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(nemo_rl_patches.subprocess, "run", fake_run)

    result = nemo_rl_patches.repair_nemo_rl_install(force_reinstall=True)

    _require(pyproject.read_text() == original_pyproject, "repair install rewrote pinned project metadata")
    _require(len(calls) == 1, f"repair install ran unexpected commands: {calls}")
    install = calls[0]
    _require(install[:4] == [nemo_rl_patches.sys.executable, "-m", "pip", "install"], "pip command drifted")
    for required in ("--no-deps", "--force-reinstall", "--ignore-requires-python", "--no-build-isolation", "-e"):
        _require(required in install, f"editable main install omitted {required}")
    _require(install[-1] == str(source_root), "main install did not retain the pinned source")
    _require(result.endswith(str(source_root)), "repair receipt omitted the retained source path")


def test_prepare_vllm_actor_environment_uses_locked_vllm_extra_without_deep_ep(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    actor_root = tmp_path / "actor-venvs"
    recipe_root = tmp_path / "recipe"
    recipe_root.mkdir()
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    monkeypatch.setenv("NEMO_RL_VENV_DIR", str(actor_root))
    monkeypatch.setattr(nemo_rl_patches, "RECIPE_ROOT", recipe_root)
    monkeypatch.setattr(
        nemo_rl_patches,
        "_ensure_pinned_nemo_rl_source",
        lambda *, force_reinstall: source_root,
    )
    monkeypatch.setattr(nemo_rl_patches, "_verify_vllm_actor_environment", lambda python_path: None)

    def fake_run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs.get("env")))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(nemo_rl_patches.subprocess, "run", fake_run)

    actor_python = nemo_rl_patches.prepare_vllm_actor_environment(force_rebuild=True)

    expected_python = actor_root / EVO2_VLLM_WORKER / "bin" / "python"
    _require(actor_python == expected_python, "actor environment used a noncanonical path")
    commands = [command for command, _env in calls]
    _require(
        ["uv", "venv", "--clear", str(expected_python.parents[1])] in commands,
        f"actor venv was not rebuilt explicitly: {commands}",
    )
    sync = next((command for command in commands if command[:2] == ["uv", "sync"]), None)
    _require(sync is not None, "actor environment omitted uv sync")
    for required in (
        "--locked",
        "--extra",
        "vllm",
        "--no-install-package",
        "deep-ep",
        "--directory",
        str(source_root),
    ):
        _require(required in sync, f"locked vLLM actor sync omitted {required}")
    sync_env = next(env for command, env in calls if command is sync)
    _require(sync_env is not None, "actor sync omitted its isolated environment")
    _require(
        sync_env.get("UV_PROJECT_ENVIRONMENT") == str(expected_python.parents[1]), "actor sync targeted the wrong venv"
    )
    recipe_install = next((command for command in commands if command[:3] == ["uv", "pip", "install"]), None)
    _require(recipe_install is not None, "actor environment omitted the Evo2 plugin install")
    for required in ("--python", str(expected_python), "--no-deps", "--no-build-isolation", "-e", str(recipe_root)):
        _require(required in recipe_install, f"actor plugin install omitted {required}")


def test_recipe_registry_splits_vllm_actors_from_system_training_runtime(tmp_path: Path, monkeypatch) -> None:
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    actor_root = tmp_path / "actor-venvs"
    actor_python = actor_root / EVO2_VLLM_WORKER / "bin" / "python"
    actor_python.parent.mkdir(parents=True)
    actor_python.write_text("")
    monkeypatch.setenv("NEMO_RL_VENV_DIR", str(actor_root))
    for actor_fqn in (*VLLM_ACTORS, *SYSTEM_ACTORS):
        monkeypatch.setitem(ACTOR_ENVIRONMENT_REGISTRY, actor_fqn, "sentinel")

    run_phage_grpo._register_recipe_extensions()

    for actor_fqn in VLLM_ACTORS:
        _require(
            ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] == str(actor_python),
            f"{actor_fqn} did not use the isolated pinned vLLM environment",
        )
    for actor_fqn in SYSTEM_ACTORS:
        _require(
            ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] == PY_EXECUTABLES.SYSTEM,
            f"{actor_fqn} left the main training environment",
        )


def test_launcher_does_not_globally_force_every_actor_to_system_python() -> None:
    source = inspect.getsource(run_phage_grpo.main)

    _require("NEMO_RL_PY_EXECUTABLES_SYSTEM" not in source, "launcher still globally bypasses actor environments")


def test_retained_source_authority_rejects_lockfile_and_submodule_drift(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    pyproject = source_root / "pyproject.toml"
    lockfile = source_root / "uv.lock"
    pyproject.write_bytes(b"pinned pyproject\n")
    lockfile.write_bytes(b"pinned lockfile\n")
    status_output = {"value": " abc123 dep\n"}

    monkeypatch.setattr(nemo_rl_patches, "PINNED_NEMO_RL_PYPROJECT_SHA256", nemo_rl_patches._sha256_file(pyproject))
    monkeypatch.setattr(nemo_rl_patches, "PINNED_NEMO_RL_UV_LOCK_SHA256", nemo_rl_patches._sha256_file(lockfile))
    monkeypatch.setattr(nemo_rl_patches, "PINNED_NEMO_RL_SUBMODULES", {Path("dep"): "abc123"})
    monkeypatch.setattr(nemo_rl_patches, "_git_head", lambda path: "revision")

    def fake_run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout=status_output["value"], stderr="")

    monkeypatch.setattr(nemo_rl_patches.subprocess, "run", fake_run)

    _require(
        nemo_rl_patches._pinned_nemo_rl_source_is_complete(source_root, "revision"),
        "exact retained source was rejected",
    )
    lockfile.write_bytes(b"foreign lockfile\n")
    _require(
        not nemo_rl_patches._pinned_nemo_rl_source_is_complete(source_root, "revision"),
        "foreign lockfile was accepted",
    )
    lockfile.write_bytes(b"pinned lockfile\n")
    status_output["value"] = "+abc123 dep\n"
    _require(
        not nemo_rl_patches._pinned_nemo_rl_source_is_complete(source_root, "revision"),
        "dirty recursive submodule status was accepted",
    )


def test_actor_runtime_probe_pins_supported_torch_vllm_and_plugin(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(nemo_rl_patches.subprocess, "run", fake_run)
    actor_python = Path("/actor/bin/python")

    nemo_rl_patches._verify_vllm_actor_environment(actor_python)

    _require(len(calls) == 1 and calls[0][:2] == [str(actor_python), "-c"], "runtime probe command drifted")
    probe = calls[0][2]
    for expected in ("(3, 13, 13)", '"2.11.0"', '"0.20.0"', "_opaque_base", "deep_ep", "evo2"):
        _require(expected in probe, f"runtime probe omitted {expected}")


def test_ci_build_and_test_env_export_persistent_actor_authority() -> None:
    build = (nemo_rl_patches.RECIPE_ROOT / ".ci_build.sh").read_text()
    test_env = (nemo_rl_patches.RECIPE_ROOT / ".ci_test_env.sh").read_text()

    for source in (build, test_env):
        _require("EVO2_PHAGE_NEMO_RL_SOURCE_DIR" in source, "CI environment omitted retained source authority")
        _require("NEMO_RL_VENV_DIR" in source, "CI environment omitted actor venv root")
    _require("--prepare-vllm-actor-env" in build, "clean CI build omitted isolated actor environment creation")
