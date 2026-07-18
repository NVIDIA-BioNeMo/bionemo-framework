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

"""Apply Evo2 integration patches to an installed NeMo-RL package."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[3]
EVO2_VLLM_WORKER = "bionemo.evo2.vllm.nemo_generation_worker.Evo2NemoRlGenerationWorker"
DEFAULT_NEMO_RL_SOURCE_DIR = RECIPE_ROOT / ".venv" / "nemo-rl-source"
DEFAULT_NEMO_RL_VENV_DIR = RECIPE_ROOT / ".venv" / "nemo-rl-venvs"
PYPI_INDEX = "https://pypi.org/simple"
PYTORCH_CU130_INDEX = "https://download.pytorch.org/whl/cu130"
FLASHINFER_CU130_INDEX = "https://flashinfer.ai/whl/cu130"
PINNED_NEMO_RL_PYPROJECT_SHA256 = "e12030d7494eeb6b63c17882fdebeec977d07a08a82bf8d848a1692f89baea1b"
PINNED_NEMO_RL_UV_LOCK_SHA256 = "472f9e1bd8e6f4a8a54468a3946fbe721d4560fbaee3cb0085a94952660644ff"
PINNED_NEMO_RL_SUBMODULES = {
    Path("3rdparty/Automodel-workspace/Automodel"): "24b47e856263d313b942f0ed666c63fff83306b4",
    Path("3rdparty/Gym-workspace/Gym"): "d67ad6611cfe21dbaeb301c59e59df32ce22ec50",
    Path("3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"): "554c7b9324225aa863eee52e8b8fdde7abced2b1",
    Path(
        "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
    ): "002255075c3728fded9a2e435677840b08560d55",
}
DEFAULT_PATCHES = (
    RECIPE_ROOT / "patches" / "nemo-rl-evo2-policy.patch",
    RECIPE_ROOT / "patches" / "nemo-rl-evo2-vllm.patch",
    RECIPE_ROOT / "patches" / "nemo-rl-evo2-sampling.patch",
)
REQUIRED_NEMO_RL_MODULES = [
    "nemo_rl.algorithms.grpo",
    "nemo_rl.data.processors",
    "nemo_rl.models.generation.megatron.megatron_worker",
    "nemo_rl.models.megatron.setup",
]
EXPECTED_PATCHED_SYMBOLS = [
    ("nemo_rl.algorithms.logits_sampling_utils", "_canonical_allowed_token_ids"),
    ("nemo_rl.models.generation.interfaces", "generation_prompt_token_ids_sha256"),
    ("nemo_rl.models.megatron.setup", "_apply_target_allowlist_prefixes"),
    ("nemo_rl.models.megatron.setup", "NoRefitMegatronBridge"),
    ("nemo_rl.models.megatron.setup", "_uses_colocated_megatron_generation"),
    ("nemo_rl.models.megatron.setup", "_select_megatron_bridge"),
    ("nemo_rl.models.generation.vllm.config", "VllmActorExecutionConfig"),
    ("nemo_rl.models.generation.vllm.vllm_generation", "_request_seeds_for_dp_stream"),
]


def _nemo_rl_source_root() -> Path:
    spec = importlib.util.find_spec("nemo_rl")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("nemo_rl is not importable in this environment")
    package_dir = Path(spec.origin).resolve().parent
    return package_dir.parent


def _run_patch(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run GNU patch in noninteractive forward-only mode."""
    patch_args = list(args)
    if "--forward" not in patch_args:
        patch_args.insert(0, "--forward")
    return subprocess.run(
        ["patch", *patch_args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _patch_paths(patch_path: Path | None = None) -> tuple[Path, ...]:
    return DEFAULT_PATCHES if patch_path is None else (Path(patch_path),)


def patch_sha256(patch_path: Path | None = None) -> str:
    """Return a domain-separated SHA256 for one patch or the maintained series."""
    paths = _patch_paths(patch_path)
    if len(paths) == 1:
        return hashlib.sha256(paths[0].read_bytes()).hexdigest()
    digest = hashlib.sha256()
    digest.update(b"bionemo.evo2_phage_gen.nemo_rl_patch_series.v1\0")
    for path in paths:
        digest.update(path.name.encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def assert_nemo_rl_patch_symbols() -> None:
    """Fail early if the runtime NeMo-RL package is missing symbols installed by the patch."""
    missing = []
    for module_name, qualified_symbol in EXPECTED_PATCHED_SYMBOLS:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(f"{module_name}:{qualified_symbol}")
            continue
        obj = module
        for attr in qualified_symbol.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                missing.append(f"{module_name}:{qualified_symbol}")
                break
    if missing:
        raise RuntimeError(
            "NeMo-RL is missing Evo2 phage patch symbols: "
            f"{', '.join(missing)}. Run evo2_phage_patch_nemo_rl --repair-install before launching GRPO."
        )


def assert_nemo_rl_patch_runtime(patch_path: Path | None = None) -> None:
    """Fail unless the importable NeMo-RL runtime matches every maintained patch."""
    source_root = _nemo_rl_source_root()
    for path in _patch_paths(patch_path):
        resolved = path.resolve()
        reverse_dry_run = _run_patch(
            ["--batch", "--dry-run", "-R", "-p1", "-i", str(resolved)],
            cwd=source_root,
        )
        if reverse_dry_run.returncode != 0:
            forward_dry_run = _run_patch(
                ["--batch", "--dry-run", "-p1", "-i", str(resolved)],
                cwd=source_root,
            )
            raise RuntimeError(
                "The importable NeMo-RL runtime is not reverse-patch-equivalent to the maintained Evo2 patch series.\n"
                f"Runtime root: {source_root}\n"
                f"Patch: {resolved}\n"
                f"Patch SHA256: {patch_sha256(resolved)}\n"
                f"Reverse dry-run output:\n{reverse_dry_run.stdout}\n"
                f"Forward dry-run output:\n{forward_dry_run.stdout}"
            )
    assert_nemo_rl_patch_symbols()


def _nemo_rl_source_pin() -> tuple[str, str]:
    """Read the pinned NeMo-RL source URL and revision from this recipe."""
    pyproject = tomllib.loads((RECIPE_ROOT / "pyproject.toml").read_text())
    source = pyproject["tool"]["uv"]["sources"]["nemo-rl"]
    return source["git"], source["rev"]


def _is_complete_nemo_rl_install() -> bool:
    """Return whether the importable NeMo-RL package contains the modules GRPO needs."""
    for module_name in REQUIRED_NEMO_RL_MODULES:
        try:
            if importlib.util.find_spec(module_name) is None:
                return False
        except ModuleNotFoundError:
            return False
    return True


def _git_head(path: Path) -> str | None:
    """Return the Git HEAD for ``path`` when it is a Git checkout."""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _clone_nemo_rl_source(git_url: str, revision: str, destination: Path) -> Path:
    """Clone the pinned NeMo-RL source when it is not already in uv's cache."""
    subprocess.run(["git", "clone", "--filter=blob:none", git_url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", revision], check=True)
    subprocess.run(["git", "-C", str(destination), "submodule", "update", "--init", "--recursive"], check=True)
    return destination


def _nemo_rl_source_dir() -> Path:
    """Return the recipe-owned persistent checkout for the pinned NeMo-RL source."""
    return Path(os.environ.get("EVO2_PHAGE_NEMO_RL_SOURCE_DIR", DEFAULT_NEMO_RL_SOURCE_DIR)).expanduser()


def vllm_actor_python_executable() -> Path:
    """Return the canonical Python executable for Evo2 vLLM actors."""
    actor_root = Path(os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_NEMO_RL_VENV_DIR)).expanduser()
    return actor_root / EVO2_VLLM_WORKER / "bin" / "python"


def _sha256_file(path: Path) -> str:
    """Hash one source-authority file without rewriting it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned_nemo_rl_source_is_complete(source_root: Path, revision: str) -> bool:
    """Validate the retained source revision, lockfiles, and recursive workspace pins."""
    pyproject = source_root / "pyproject.toml"
    lockfile = source_root / "uv.lock"
    if not pyproject.is_file() or not lockfile.is_file() or _git_head(source_root) != revision:
        return False
    if _sha256_file(pyproject) != PINNED_NEMO_RL_PYPROJECT_SHA256:
        return False
    if _sha256_file(lockfile) != PINNED_NEMO_RL_UV_LOCK_SHA256:
        return False
    status = subprocess.run(
        ["git", "-C", str(source_root), "submodule", "status", "--recursive"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if status.returncode != 0:
        return False
    observed: dict[Path, str] = {}
    for line in status.stdout.splitlines():
        if not line.startswith(" "):
            return False
        fields = line.strip().split()
        if len(fields) < 2:
            return False
        observed[Path(fields[1])] = fields[0]
    return observed == PINNED_NEMO_RL_SUBMODULES


def _ensure_pinned_nemo_rl_source(*, force_reinstall: bool) -> Path:
    """Create or validate the persistent exact NeMo-RL source checkout."""
    git_url, revision = _nemo_rl_source_pin()
    source_root = _nemo_rl_source_dir()
    if force_reinstall and source_root.exists():
        shutil.rmtree(source_root)
    if not source_root.exists():
        source_root.parent.mkdir(parents=True, exist_ok=True)
        _clone_nemo_rl_source(git_url, revision, source_root)
    if not _pinned_nemo_rl_source_is_complete(source_root, revision):
        raise RuntimeError(
            f"Retained NeMo-RL source does not match pinned revision, lockfiles, and submodules: {source_root}"
        )
    return source_root


def repair_nemo_rl_install(*, force_reinstall: bool = False) -> str:
    """Install the retained exact NeMo-RL checkout into the main training environment."""
    source_root = _ensure_pinned_nemo_rl_source(force_reinstall=force_reinstall)
    if not force_reinstall and _is_complete_nemo_rl_install():
        return f"nemo-rl install already contains required modules from {source_root}"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            "--ignore-requires-python",
            "--no-build-isolation",
            "-e",
            str(source_root),
        ],
        check=True,
    )
    source_path = str(source_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    importlib.invalidate_caches()
    return f"reinstalled editable nemo-rl from retained source {source_root}"


def _verify_vllm_actor_environment(python_path: Path) -> None:
    """Verify the isolated actor runtime and Evo2 plugin before Ray can launch it."""
    probe = """
import importlib.metadata
import importlib.util
import sys

import ray
import torch
import vllm

if sys.version_info[:2] != __EXPECTED_PYTHON_MINOR__:
    raise RuntimeError(f"unexpected actor Python: {sys.version}")
if torch.__version__.split("+", 1)[0] != "2.11.0":
    raise RuntimeError(f"unexpected actor Torch: {torch.__version__}")
if vllm.__version__ != "0.20.0":
    raise RuntimeError(f"unexpected actor vLLM: {vllm.__version__}")
if ray.__version__ != "2.55.1":
    raise RuntimeError(f"unexpected actor Ray: {ray.__version__}")
if not hasattr(torch, "_opaque_base"):
    raise RuntimeError("actor Torch lacks torch._opaque_base")
if importlib.util.find_spec("deep_ep") is not None:
    raise RuntimeError("dense Evo2 actor environment unexpectedly installed deep_ep")
plugins = [entry for entry in importlib.metadata.entry_points(group="vllm.general_plugins") if entry.name == "evo2"]
if len(plugins) != 1 or plugins[0].value != "bionemo.evo2.vllm.plugin:register":
    raise RuntimeError(f"unexpected Evo2 vLLM plugin entry points: {plugins}")
from bionemo.evo2.vllm.nemo_generation_worker import Evo2NemoRlGenerationWorker  # noqa: F401
"""
    subprocess.run([str(python_path), "-c", probe.replace("__EXPECTED_PYTHON_MINOR__", repr(sys.version_info[:2]))], check=True)


def prepare_vllm_actor_environment(*, force_rebuild: bool = False) -> Path:
    """Build the locked isolated NeMo-RL vLLM actor environment."""
    source_root = _ensure_pinned_nemo_rl_source(force_reinstall=False)
    actor_python = vllm_actor_python_executable()
    actor_environment = actor_python.parents[1]
    actor_environment.parent.mkdir(parents=True, exist_ok=True)

    venv_args = ["uv", "venv"]
    venv_args.append("--clear" if force_rebuild else "--allow-existing")
    venv_args.extend(["--seed", "--python", sys.executable])
    venv_args.append(str(actor_environment))
    subprocess.run(venv_args, cwd=source_root, check=True)

    requirements_path = actor_environment / ".nemo-rl-vllm-lock.txt"
    subprocess.run(
        [
            "uv",
            "export",
            "--locked",
            "--extra",
            "vllm",
            "--no-dev",
            "--no-emit-workspace",
            "--no-emit-package",
            "deep-ep",
            "--no-emit-package",
            "deep-gemm",
            "--no-hashes",
            "--output-file",
            str(requirements_path),
        ],
        cwd=source_root,
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(actor_python),
            "--requirements",
            str(requirements_path),
            "--no-deps",
            "--default-index",
            PYPI_INDEX,
            "--index",
            PYTORCH_CU130_INDEX,
            "--index",
            FLASHINFER_CU130_INDEX,
            "--index-strategy",
            "unsafe-best-match",
        ],
        cwd=source_root,
        check=True,
    )
    subprocess.run(
        [
            str(actor_python),
            "-m",
            "pip",
            "install",
            "--ignore-requires-python",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(source_root),
        ],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(actor_python),
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(RECIPE_ROOT),
        ],
        check=True,
    )
    _verify_vllm_actor_environment(actor_python)
    return actor_python


def _apply_nemo_rl_patch_file(patch_path: Path, *, source_root: Path, check_only: bool) -> str:
    """Apply one patch file to an importable NeMo-RL package root."""
    patch_path = Path(patch_path).resolve()
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")

    dry_run = _run_patch(["--batch", "--dry-run", "-p1", "-i", str(patch_path)], cwd=source_root)
    if dry_run.returncode == 0:
        if check_only:
            return f"patch can apply cleanly to {source_root}"
        applied = _run_patch(["--batch", "-p1", "-i", str(patch_path)], cwd=source_root)
        if applied.returncode != 0:
            raise RuntimeError(applied.stdout)
        return f"patch applied to {source_root}"

    reverse_dry_run = _run_patch(["--batch", "--dry-run", "-R", "-p1", "-i", str(patch_path)], cwd=source_root)
    if reverse_dry_run.returncode == 0:
        return f"patch already applied to {source_root}"

    raise RuntimeError(
        "Patch does not apply cleanly and does not appear to be already applied.\n"
        f"Forward dry-run output:\n{dry_run.stdout}\n"
        f"Reverse dry-run output:\n{reverse_dry_run.stdout}"
    )


def apply_nemo_rl_patch(patch_path: Path | None = None, *, check_only: bool = False) -> str:
    """Apply one requested patch or the maintained recipe patch series."""
    source_root = _nemo_rl_source_root()
    results = [
        _apply_nemo_rl_patch_file(path, source_root=source_root, check_only=check_only)
        for path in _patch_paths(patch_path)
    ]
    return "\n".join(results)


def main() -> None:
    """CLI entry point for patching an installed NeMo-RL package."""
    parser = argparse.ArgumentParser(description="Apply Evo2 phage NeMo-RL integration patches")
    parser.add_argument(
        "--patch",
        type=Path,
        default=None,
        help="Apply one explicit patch instead of the maintained recipe patch series.",
    )
    parser.add_argument("--check", action="store_true", help="Only check whether the patch can be applied")
    parser.add_argument(
        "--repair-install",
        action="store_true",
        help="Reinstall the pinned NeMo-RL checkout with all nemo_rl subpackages before applying the patch.",
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Force a fresh reinstall even if the current nemo-rl package looks complete.",
    )
    parser.add_argument(
        "--prepare-vllm-actor-env",
        action="store_true",
        help="Build and verify the locked isolated vLLM actor environment.",
    )
    parser.add_argument(
        "--verify-runtime",
        action="store_true",
        help="Verify the importable nemo-rl runtime is reverse-patch-equivalent to the requested patch or series.",
    )
    args = parser.parse_args()
    if args.repair_install:
        print(repair_nemo_rl_install(force_reinstall=args.force_reinstall))
        importlib.invalidate_caches()
        for module_name in list(sys.modules):
            if module_name == "nemo_rl" or module_name.startswith("nemo_rl."):
                sys.modules.pop(module_name)
    print(apply_nemo_rl_patch(args.patch, check_only=args.check))
    if args.prepare_vllm_actor_env:
        print(
            f"prepared vLLM actor environment at {prepare_vllm_actor_environment(force_rebuild=args.force_reinstall)}"
        )
    if args.verify_runtime:
        assert_nemo_rl_patch_runtime(args.patch)
        print(f"verified patched nemo-rl runtime with patch SHA256 {patch_sha256(args.patch)}")


if __name__ == "__main__":
    main()
