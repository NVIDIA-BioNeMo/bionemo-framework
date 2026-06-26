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
import importlib.util
import subprocess
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATCH = RECIPE_ROOT / "patches" / "nemo-rl-evo2-mbridge-grpo.patch"


def _nemo_rl_source_root() -> Path:
    spec = importlib.util.find_spec("nemo_rl")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("nemo_rl is not importable in this environment")
    package_dir = Path(spec.origin).resolve().parent
    return package_dir.parent


def _run_patch(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["patch", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def apply_nemo_rl_patch(patch_path: Path = DEFAULT_PATCH, *, check_only: bool = False) -> str:
    """Apply the recipe's NeMo-RL patch to the importable package root."""
    patch_path = Path(patch_path).resolve()
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    source_root = _nemo_rl_source_root()

    dry_run = _run_patch(["--dry-run", "-p1", "-i", str(patch_path)], cwd=source_root)
    if dry_run.returncode == 0:
        if check_only:
            return f"patch can apply cleanly to {source_root}"
        applied = _run_patch(["-p1", "-i", str(patch_path)], cwd=source_root)
        if applied.returncode != 0:
            raise RuntimeError(applied.stdout)
        return f"patch applied to {source_root}"

    reverse_dry_run = _run_patch(["--dry-run", "-R", "-p1", "-i", str(patch_path)], cwd=source_root)
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
    args = parser.parse_args()
    print(apply_nemo_rl_patch(args.patch, check_only=args.check))


if __name__ == "__main__":
    main()
