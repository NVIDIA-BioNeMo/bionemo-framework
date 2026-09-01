#!/usr/bin/env python3

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

"""Probe the local GPU and Transformer Engine install for quantization-recipe support.

Run this inside the *target* codebase's environment -- the same Python interpreter and virtual
environment (or conda env) that your model training script uses. If torch or
transformer_engine are missing the script will offer to install them; if you decline it
exits so you can install them yourself in the right environment first.

Support is determined by delegating to Transformer Engine's own ``check_*_support()``
functions -- the same approach used by
``$BIONEMO_RECIPES/models/esm2/tests/common/fixtures.py::_check_recipe_support``.

Usage:
    python probe_hardware.py [-o .bionemo-accel/hardware.json] [--no-install]

Arguments:
    -o, --output PATH   Write JSON output to PATH (default: stdout only).
    --no-install        Never prompt to install missing packages; exit 1 immediately
                        if torch or transformer_engine are absent.

Output:
    JSON object written to stdout (and to --output if given) with keys:
      torch               -- version, CUDA availability, device name, compute capability
      transformer_engine  -- version, per-recipe support flags, caveats
      te_version_recommendation -- recommended NGC image and pip pin
      supported_recipes   -- sorted list of recipe names that passed their check function

Exit codes:
    0  Probe succeeded; torch and transformer_engine are importable and a CUDA device is visible.
    1  A required package is missing and was not installed, or no CUDA device is available.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


# The NGC image the BioNeMo recipes standardize on (see $BIONEMO_RECIPES/recipes/*/Dockerfile).
RECOMMENDED_NGC_IMAGE = "nvcr.io/nvidia/pytorch:26.04-py3"

# The repo's only explicit pin, for non-NGC environments
# ($BIONEMO_RECIPES/recipes/esm2_native_te/Dockerfile.cuda).
RECOMMENDED_TE_PIN = "transformer-engine[pytorch]==2.9.0"
RECOMMENDED_TORCH_PIN = "torch==2.9.0"

# Recipe name -> the TE support-check function that gates it.
RECIPE_SUPPORT_CHECKS = {
    "DelayedScaling": "check_fp8_support",
    "Float8CurrentScaling": "check_fp8_support",
    "Float8BlockScaling": "check_fp8_block_scaling_support",
    "MXFP8BlockScaling": "check_mxfp8_support",
    "NVFP4BlockScaling": "check_nvfp4_support",
}

# Recipes that BioNeMo treats as unvalidated on consumer Blackwell, even where TE reports support.
# See $BIONEMO_RECIPES/recipes/eden_megatron/tests/bionemo/eden/utils.py, which bounds at cc < (12, 0).
SM120_UNVALIDATED_RECIPES = ("MXFP8BlockScaling", "NVFP4BlockScaling")

# GPUs that the shared test harness treats as data-center class
# ($BIONEMO_RECIPES/models/esm2/tests/common/__init__.py).
DATA_CENTER_GPUS = ("H100", "H200", "B100", "B200", "B300")

# Compute capability tuples used for BioNeMo-specific hardware caveats.
SM_120_COMPUTE_CAPABILITY = (12, 0)  # Consumer Blackwell (RTX 50xx)
SM_80_COMPUTE_CAPABILITY = (8, 0)  # Ampere (A100)

_ENVIRONMENT_NOTICE = """
NOTE: This script must run in your model's training environment — the same Python
interpreter, virtual environment, or conda environment that your training script uses.
If you are running it in a different environment, the install will land in the wrong
place and the acceleration port will not be able to import these packages.

To confirm you are in the right environment, check:
  which python      (or: conda info --envs)
  pip show torch
"""


def _pip_cmd(no_build_isolation: bool = False) -> list[str]:
    """Return the best available install command prefix.

    Prefers ``uv pip install`` when ``uv`` is on PATH (faster, better resolver),
    falls back to ``python -m pip install``.

    Args:
        no_build_isolation: When True, append ``--no-build-isolation``.

    Returns:
        argv prefix up to and including ``install``.
    """
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install"]
    else:
        cmd = [sys.executable, "-m", "pip", "install"]
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    return cmd


def _ask_install(package_desc: str, packages: list[str], no_build_isolation: bool = False) -> bool:
    """Prompt the user to confirm an install, then run it.

    Args:
        package_desc: Human-readable description of what will be installed.
        packages: Package specifiers to pass to the installer.
        no_build_isolation: Whether to pass ``--no-build-isolation``.

    Returns:
        True if installation succeeded, False if the user declined or it failed.
    """
    install_cmd = _pip_cmd(no_build_isolation) + packages
    print(_ENVIRONMENT_NOTICE, file=sys.stderr)
    print(f"{package_desc} is not installed in this environment.", file=sys.stderr)
    print(f"Install command: {' '.join(install_cmd)}", file=sys.stderr)
    try:
        answer = input("Install now? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        print("Skipping install. Install it yourself in the correct environment, then re-run.", file=sys.stderr)
        return False

    print(f"Running: {' '.join(install_cmd)}", file=sys.stderr)
    result = subprocess.run(install_cmd)
    if result.returncode != 0:
        print("Install failed. Check the output above and resolve before continuing.", file=sys.stderr)
        return False
    return True


def ensure_torch(no_install: bool) -> bool:
    """Ensure torch is importable; offer to install if missing.

    Args:
        no_install: When True, never prompt — just return False if unavailable.

    Returns:
        True if torch is importable after this call.
    """
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        pass

    if no_install:
        print(
            "torch is not installed. Re-run without --no-install to be prompted, or install manually.", file=sys.stderr
        )
        return False

    if not _ask_install("torch", [RECOMMENDED_TORCH_PIN]):
        return False

    try:
        import importlib

        importlib.import_module("torch")
        return True
    except ImportError:
        return False


def ensure_transformer_engine(no_install: bool) -> bool:
    """Ensure transformer_engine is importable; offer to install if missing.

    Args:
        no_install: When True, never prompt — just return False if unavailable.

    Returns:
        True if transformer_engine is importable after this call.
    """
    try:
        import transformer_engine  # noqa: F401

        return True
    except ImportError:
        pass

    if no_install:
        print(
            "transformer_engine is not installed. Re-run without --no-install to be prompted, or install manually.",
            file=sys.stderr,
        )
        return False

    # TE must be installed without build isolation because it links against the
    # installed torch and CUDA headers at build time.
    if not _ask_install("transformer-engine", [RECOMMENDED_TE_PIN], no_build_isolation=True):
        return False

    try:
        import importlib

        importlib.import_module("transformer_engine")
        return True
    except ImportError:
        return False


def probe_torch() -> dict:
    """Collect torch and device information.

    Returns:
        A dict of torch/device facts, with an ``error`` key if torch is unavailable.
    """
    try:
        import torch
    except ImportError as e:
        return {"available": False, "error": f"torch not importable: {e}"}

    info: dict = {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        info["error"] = "No CUDA device visible; recipe support cannot be determined."
        return info

    capability = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name(0)
    info.update(
        {
            "device_name": device_name,
            "device_count": torch.cuda.device_count(),
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "compute_capability_tuple": list(capability),
            "has_data_center_gpu": any(gpu in device_name.upper() for gpu in DATA_CENTER_GPUS),
        }
    )
    return info


def probe_transformer_engine(compute_capability: tuple[int, int] | None) -> dict:
    """Query Transformer Engine for per-recipe support.

    Args:
        compute_capability: Device compute capability for BioNeMo-specific caveats.

    Returns:
        A dict with the TE version, per-recipe support, and any caveats.
    """
    try:
        import transformer_engine
        import transformer_engine.pytorch
        from transformer_engine.pytorch import fp8
    except (ImportError, OSError, RuntimeError) as e:
        return {"available": False, "error": f"transformer_engine not importable: {e}"}

    result: dict = {
        "available": True,
        "version": getattr(transformer_engine, "__version__", "unknown"),
        "has_modern_autocast": hasattr(transformer_engine.pytorch, "autocast"),
        "recipes": {},
        "caveats": [],
    }

    for recipe_name, check_name in RECIPE_SUPPORT_CHECKS.items():
        check_fn = getattr(fp8, check_name, None)
        if check_fn is None:
            result["recipes"][recipe_name] = {
                "supported": False,
                "reason": f"transformer_engine.pytorch.fp8.{check_name} not present in TE {result['version']}",
                "check": check_name,
            }
            continue
        try:
            supported, reason = check_fn()
        except Exception as e:  # report the failure; never crash the probe
            supported, reason = False, f"{check_name} raised: {e}"
        result["recipes"][recipe_name] = {
            "supported": bool(supported),
            "reason": str(reason),
            "check": check_name,
        }

    if not result["has_modern_autocast"]:
        result["caveats"].append(
            "transformer_engine.pytorch.autocast is unavailable; this TE predates the API the "
            "BioNeMo recipes use. Upgrade before porting -- do not fall back to the legacy "
            "te.fp8_autocast."
        )

    if compute_capability is not None:
        major, minor = compute_capability
        if (major, minor) == SM_120_COMPUTE_CAPABILITY:
            for recipe_name in SM120_UNVALIDATED_RECIPES:
                if result["recipes"].get(recipe_name, {}).get("supported"):
                    result["recipes"][recipe_name]["bionemo_unvalidated"] = True
            result["caveats"].append(
                "sm_120 (consumer Blackwell): BioNeMo bounds MXFP8/NVFP4 support at compute "
                "capability < 12.0 ($BIONEMO_RECIPES/recipes/eden_megatron/tests/bionemo/eden/"
                "utils.py). Treat these as unvalidated here."
            )
            result["caveats"].append(
                "sm_120: fused_attn THD is xfailed in "
                "$BIONEMO_RECIPES/models/esm2/tests/common/test_modeling_common.py. "
                "Sequence packing still works via flash_attn."
            )
        if (major, minor) == SM_80_COMPUTE_CAPABILITY:
            result["caveats"].append(
                "sm_80 (A100): fused_attn THD is xfailed in "
                "$BIONEMO_RECIPES/models/esm2/tests/common/test_modeling_common.py. "
                "Sequence packing still works via flash_attn."
            )

    return result


def build_version_recommendation(te_info: dict) -> dict:
    """Build the Transformer Engine version recommendation for the report.

    Args:
        te_info: The output of :func:`probe_transformer_engine`.

    Returns:
        A dict describing the installed version and the recommended target.
    """
    return {
        "installed": te_info.get("version") if te_info.get("available") else None,
        "recommended_ngc_image": RECOMMENDED_NGC_IMAGE,
        "recommended_pin_non_ngc": [RECOMMENDED_TORCH_PIN, RECOMMENDED_TE_PIN],
        "note": (
            "Inside the NGC image, TE comes from the image and requirements.txt lists an "
            "unversioned transformer_engine[pytorch]. The explicit pin is only for fresh-venv "
            "installs ($BIONEMO_RECIPES/recipes/esm2_native_te/Dockerfile.cuda), and must be "
            "installed --no-build-isolation. Recommend only; never upgrade the target "
            "environment without asking."
        ),
    }


def main() -> int:
    """Run the probe, installing missing packages if the user agrees.

    Returns:
        0 on success, 1 if a required package is missing and not installed.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write JSON here (default: stdout only)")
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Never prompt to install missing packages; exit 1 immediately if they are absent.",
    )
    args = parser.parse_args()

    if not ensure_torch(args.no_install):
        return 1
    if not ensure_transformer_engine(args.no_install):
        return 1

    torch_info = probe_torch()
    capability = torch_info.get("compute_capability_tuple")
    te_info = probe_transformer_engine(tuple(capability) if capability else None)

    report = {
        "torch": torch_info,
        "transformer_engine": te_info,
        "te_version_recommendation": build_version_recommendation(te_info),
        "supported_recipes": sorted(
            name for name, info in te_info.get("recipes", {}).items() if info.get("supported")
        ),
    }

    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")

    if not torch_info.get("cuda_available") or not te_info.get("available"):
        print("\nprobe_hardware: no usable CUDA device or no Transformer Engine install.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
