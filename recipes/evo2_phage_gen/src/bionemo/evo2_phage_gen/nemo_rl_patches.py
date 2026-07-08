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
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATCH = RECIPE_ROOT / "patches" / "nemo-rl-evo2-mbridge-grpo.patch"
REQUIRED_NEMO_RL_MODULES = [
    "nemo_rl.algorithms.grpo",
    "nemo_rl.data.processors",
    "nemo_rl.models.generation.megatron.megatron_worker",
    "nemo_rl.models.megatron.setup",
]
EXPECTED_PATCHED_SYMBOLS = [
    ("nemo_rl.experience.rollouts", "collect_environment_metrics"),
    ("nemo_rl.models.generation.megatron.megatron_generation", "_load_generation_adapter"),
    ("nemo_rl.models.generation.megatron.megatron_worker", "MegatronGenerationMixin._load_generation_adapter"),
    ("nemo_rl.models.generation.megatron.megatron_worker", "MegatronGenerationMixin.generate_with_adapter"),
    ("nemo_rl.models.megatron.setup", "_apply_target_allowlist_prefixes"),
    ("nemo_rl.models.megatron.setup", "NoRefitMegatronBridge"),
    ("nemo_rl.models.megatron.setup", "_uses_colocated_megatron_generation"),
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


def patch_sha256(patch_path: Path = DEFAULT_PATCH) -> str:
    """Return the SHA256 hash of the maintained NeMo-RL patch."""
    return hashlib.sha256(Path(patch_path).read_bytes()).hexdigest()


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


def assert_nemo_rl_patch_runtime(patch_path: Path = DEFAULT_PATCH) -> None:
    """Fail unless the importable NeMo-RL runtime matches the maintained patch."""
    source_root = _nemo_rl_source_root()
    patch_path = Path(patch_path).resolve()
    reverse_dry_run = _run_patch(["--batch", "--dry-run", "-R", "-p1", "-i", str(patch_path)], cwd=source_root)
    if reverse_dry_run.returncode != 0:
        forward_dry_run = _run_patch(["--batch", "--dry-run", "-p1", "-i", str(patch_path)], cwd=source_root)
        raise RuntimeError(
            "The importable NeMo-RL runtime is not reverse-patch-equivalent to the maintained Evo2 patch.\n"
            f"Runtime root: {source_root}\n"
            f"Patch SHA256: {patch_sha256(patch_path)}\n"
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


def _uv_cache_dir() -> Path | None:
    """Return uv's cache dir when uv is available."""
    result = subprocess.run(
        ["uv", "cache", "dir"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).expanduser()


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


def _git_worktree_is_clean(path: Path) -> bool:
    """Return whether ``path`` is a Git checkout with no tracked or untracked changes."""
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == ""


def _looks_like_nemo_rl_source(path: Path) -> bool:
    """Check for files that identify a full NeMo-RL source checkout."""
    return (path / "pyproject.toml").exists() and (path / "nemo_rl" / "algorithms" / "grpo.py").exists()


def _find_cached_nemo_rl_source(revision: str) -> Path | None:
    """Find the pinned NeMo-RL checkout in uv's git cache, if present."""
    cache_dir = _uv_cache_dir()
    if cache_dir is None:
        return None
    checkouts_dir = cache_dir / "git-v0" / "checkouts"
    if not checkouts_dir.exists():
        return None
    for pyproject_path in checkouts_dir.rglob("pyproject.toml"):
        candidate = pyproject_path.parent
        if (
            _looks_like_nemo_rl_source(candidate)
            and _git_head(candidate) == revision
            and _git_worktree_is_clean(candidate)
        ):
            return candidate
    return None


def _clone_nemo_rl_source(git_url: str, revision: str, destination: Path) -> Path:
    """Clone the pinned NeMo-RL source when it is not already in uv's cache."""
    subprocess.run(["git", "clone", "--filter=blob:none", git_url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", revision], check=True)
    return destination


def _copy_minimal_nemo_rl_source(source_root: Path, destination: Path) -> Path:
    """Copy only files needed to build the NeMo-RL Python package."""
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root / "nemo_rl", destination / "nemo_rl")
    for filename in ["pyproject.toml", "README.md", "LICENSE"]:
        source_file = source_root / filename
        if source_file.exists():
            shutil.copy2(source_file, destination / filename)
    return destination


def _patch_nemo_rl_packaging_metadata(source_root: Path) -> None:
    """Patch upstream packaging metadata so setuptools includes all ``nemo_rl`` subpackages."""
    pyproject_path = source_root / "pyproject.toml"
    text = pyproject_path.read_text()
    text = text.replace('packages = ["nemo_rl"]', 'packages = { find = { include = ["nemo_rl*"] } }')
    text = text.replace('requires-python = ">=3.13.13,<3.14"', 'requires-python = ">=3.10"')
    pyproject_path.write_text(text)


def repair_nemo_rl_install(*, force_reinstall: bool = False) -> str:
    """Reinstall the pinned NeMo-RL checkout with complete package discovery."""
    if not force_reinstall and _is_complete_nemo_rl_install():
        return "nemo-rl install already contains required modules"

    git_url, revision = _nemo_rl_source_pin()
    source_root = _find_cached_nemo_rl_source(revision)
    with tempfile.TemporaryDirectory(prefix="evo2-phage-nemo-rl-") as temp_dir:
        temp_path = Path(temp_dir)
        if source_root is None:
            source_root = _clone_nemo_rl_source(git_url, revision, temp_path / "source")
        build_root = _copy_minimal_nemo_rl_source(source_root, temp_path / "build")
        _patch_nemo_rl_packaging_metadata(build_root)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(build_root)],
            check=True,
        )
    return f"reinstalled nemo-rl from {revision} with complete package discovery"


def apply_nemo_rl_patch(patch_path: Path = DEFAULT_PATCH, *, check_only: bool = False) -> str:
    """Apply the recipe's NeMo-RL patch to the importable package root."""
    patch_path = Path(patch_path).resolve()
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    source_root = _nemo_rl_source_root()

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


def main() -> None:
    """CLI entry point for patching an installed NeMo-RL package."""
    parser = argparse.ArgumentParser(description="Apply Evo2 phage NeMo-RL integration patch")
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
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
        "--verify-runtime",
        action="store_true",
        help="Verify the importable nemo-rl runtime is reverse-patch-equivalent to the maintained patch.",
    )
    args = parser.parse_args()
    if args.repair_install:
        print(repair_nemo_rl_install(force_reinstall=args.force_reinstall))
        importlib.invalidate_caches()
        for module_name in list(sys.modules):
            if module_name == "nemo_rl" or module_name.startswith("nemo_rl."):
                sys.modules.pop(module_name)
    print(apply_nemo_rl_patch(args.patch, check_only=args.check))
    if args.verify_runtime:
        assert_nemo_rl_patch_runtime(args.patch)
        print(f"verified patched nemo-rl runtime with patch SHA256 {patch_sha256(args.patch)}")


if __name__ == "__main__":
    main()
