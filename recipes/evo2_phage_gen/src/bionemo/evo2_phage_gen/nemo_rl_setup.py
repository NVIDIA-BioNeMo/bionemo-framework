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

"""Install the NeMo-RL source used by the Evo2 phage recipe."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import logging
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


logger = logging.getLogger(__name__)
RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATCH = RECIPE_ROOT / "patches" / "nemo-rl-evo2-mbridge-grpo.patch"
REQUIRED_MODULES = (
    "nemo_rl.algorithms.grpo",
    "nemo_rl.algorithms.logits_sampling_utils",
    "nemo_rl.data.processors",
    "nemo_rl.models.generation.megatron.megatron_worker",
    "nemo_rl.models.megatron.setup",
)


def _configured_source() -> tuple[str, str]:
    pyproject = tomllib.loads((RECIPE_ROOT / "pyproject.toml").read_text())
    source = pyproject.get("tool", {}).get("uv", {}).get("sources", {}).get("nemo-rl", {})
    if not isinstance(source, dict) or not source.get("git") or not source.get("rev"):
        raise RuntimeError("pyproject.toml must identify the NeMo-RL source revision")
    return str(source["git"]), str(source["rev"])


def _run_patch(source_root: Path, patch_path: Path, *, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    command = ["patch", "--batch", "--forward"]
    if dry_run:
        command.append("--dry-run")
    command.extend(["-p1", "-i", str(patch_path.resolve())])
    return subprocess.run(
        command,
        cwd=source_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def apply_source_patch(source_root: Path, patch_path: Path = DEFAULT_PATCH, *, check_only: bool = False) -> str:
    """Apply the recipe patch to an unmodified NeMo-RL source tree."""
    source_root = Path(source_root)
    patch_path = Path(patch_path)
    if not patch_path.is_file():
        raise FileNotFoundError(patch_path)
    result = _run_patch(source_root, patch_path, dry_run=check_only)
    if result.returncode:
        raise RuntimeError(f"NeMo-RL patch did not apply to {source_root}:\n{result.stdout}")
    return result.stdout.strip()


def _uv_cache_dir() -> Path | None:
    result = subprocess.run(
        ["uv", "cache", "dir"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return Path(result.stdout.strip()).expanduser() if result.returncode == 0 else None


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _find_cached_source(revision: str) -> Path | None:
    cache = _uv_cache_dir()
    if cache is None:
        return None
    checkouts = cache / "git-v0" / "checkouts"
    if not checkouts.exists():
        return None
    for pyproject in checkouts.rglob("pyproject.toml"):
        candidate = pyproject.parent
        if (candidate / "nemo_rl" / "algorithms" / "grpo.py").is_file() and _git_revision(candidate) == revision:
            return candidate
    return None


def _clone_source(url: str, revision: str, destination: Path) -> Path:
    subprocess.run(["git", "clone", "--filter=blob:none", url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", revision], check=True)
    return destination


def _copy_build_source(source_root: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    shutil.copytree(source_root / "nemo_rl", destination / "nemo_rl")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        source = source_root / name
        if source.exists():
            shutil.copy2(source, destination / name)
    return destination


def _runtime_is_complete() -> bool:
    for name in REQUIRED_MODULES:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except ModuleNotFoundError:
            return False
    return True


def assert_nemo_rl_runtime() -> None:
    """Check the small set of runtime capabilities used by this recipe."""
    if not _runtime_is_complete():
        raise RuntimeError("NeMo-RL is missing modules required by the Evo2 phage recipe")
    grpo = importlib.import_module("nemo_rl.algorithms.grpo")
    logits_sampling = importlib.import_module("nemo_rl.algorithms.logits_sampling_utils")
    cluster = importlib.import_module("nemo_rl.distributed.virtual_cluster")
    dataset_utils = importlib.import_module("nemo_rl.data.datasets.utils")
    generation_worker = importlib.import_module("nemo_rl.models.generation.megatron.megatron_worker")
    if not callable(getattr(grpo, "split_environment_timing_metrics", None)):
        raise RuntimeError("NeMo-RL is missing environment timing support")
    if not callable(getattr(dataset_utils, "resolve_external_dataset_class", None)):
        raise RuntimeError("NeMo-RL is missing support for external recipe datasets")
    sampling_parameters = inspect.signature(logits_sampling.apply_top_k_top_p).parameters
    if "target_token_ids" not in sampling_parameters:
        raise RuntimeError("NeMo-RL cannot retain sampled actions in filtered log-probability support")
    generation_mixin = getattr(generation_worker, "MegatronGenerationMixin", None)
    if not callable(getattr(generation_mixin, "_generation_adapter_requires_persistent_model_storage", None)):
        raise RuntimeError("NeMo-RL cannot preserve CUDA-graph model storage across colocated refits")
    if not callable(getattr(generation_mixin, "_generation_adapter_model_refit_complete", None)):
        raise RuntimeError("NeMo-RL cannot refresh quantized CUDA graphs after colocated refits")
    parameters = inspect.signature(cluster.init_ray).parameters
    if not {"include_dashboard", "num_cpus"}.issubset(parameters):
        raise RuntimeError("NeMo-RL is missing local Ray resource controls")


def setup_nemo_rl(*, patch_path: Path = DEFAULT_PATCH, force_reinstall: bool = False, check_only: bool = False) -> str:
    """Patch the configured NeMo-RL source once, then install that source."""
    if not check_only and not force_reinstall and _runtime_is_complete():
        assert_nemo_rl_runtime()
        return "NeMo-RL runtime is ready"

    url, revision = _configured_source()
    with tempfile.TemporaryDirectory(prefix="evo2-phage-nemo-rl-") as temporary:
        temporary_root = Path(temporary)
        source = _find_cached_source(revision)
        if source is None:
            source = _clone_source(url, revision, temporary_root / "source")
        build_source = _copy_build_source(source, temporary_root / "build")
        output = apply_source_patch(build_source, patch_path, check_only=check_only)
        if check_only:
            return output or f"patch applies to NeMo-RL {revision}"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(build_source)],
            check=True,
        )

    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name == "nemo_rl" or name.startswith("nemo_rl."):
            sys.modules.pop(name)
    assert_nemo_rl_runtime()
    return f"installed NeMo-RL {revision} with Evo2 support"


def main() -> None:
    """Install and verify the NeMo-RL integration used by this recipe."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Set up NeMo-RL for the Evo2 phage recipe")
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--check", action="store_true", help="Check that the patch applies without installing")
    parser.add_argument("--force-reinstall", action="store_true")
    args = parser.parse_args()
    print(setup_nemo_rl(patch_path=args.patch, force_reinstall=args.force_reinstall, check_only=args.check))


if __name__ == "__main__":
    main()
