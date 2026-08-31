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

r"""Drive Transformer Engine's GEMM benchmark to pick a quantization recipe.

Wraps ``benchmarks/gemm/benchmark_gemm.py`` from the Transformer Engine *source tree* (it does not
ship in the pip wheel). Runs it twice -- autocast and ``--pre-quantize`` -- because the gap between
them is the dynamic quantization overhead that a kernel-only number hides.

See https://nvidia.github.io/TransformerEngine/examples/gemm_profiling/gemm_profiling.html

Usage:
    python run_gemm_benchmark.py \
        --inventory .bionemo-accel/inventory.json \
        --hardware  .bionemo-accel/hardware.json \
        -o .bionemo-accel/gemm

Arguments:
    --inventory PATH    inventory.json written by probe_hardware.py Phase 0.
    --hardware PATH     hardware.json written by probe_hardware.py Phase 2.
    -o, --output-dir DIR  Directory for benchmark plots and logs (default: .bionemo-accel/gemm).
    --shapes MxKxN      Comma-separated manual shape triplets; mutually exclusive with model config.
    --te-source PATH    Path to a Transformer Engine source checkout (benchmark_gemm.py location).
    --allow-clone       Shallow-clone Transformer Engine if benchmark_gemm.py is not found locally.
    --verbose-kernels   Set NVTE_LOG_LEVEL=1 to confirm kernel dispatch in benchmark output.

Output:
    <output-dir>/autocast.log          -- stdout/stderr from the autocast run.
    <output-dir>/pre_quantized.log     -- stdout/stderr from the pre-quantize run.
    <output-dir>/gemm_speedup_autocast.png      -- speedup plot for autocast mode.
    <output-dir>/gemm_speedup_pre_quantized.png -- speedup plot for pre-quantize mode.
    <output-dir>/summary.json          -- benchmark script path, mode, per-run return codes,
                                         supported recipes, and interpretation reminders.

Exit codes:
    0  Both benchmark runs (autocast and pre-quantize) completed with return code 0.
    1  One or both runs failed, benchmark_gemm.py was not found, or a required argument is missing.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


BENCHMARK_RELPATH = Path("benchmarks/gemm/benchmark_gemm.py")

TE_REPO_URL = "https://github.com/NVIDIA/TransformerEngine.git"

# Recipe names (from probe_hardware.py) that map to each benchmark skip flag.
# --no-fp8 covers block-scaled FP8 families; --no-fp4 covers NVFP4.
BLOCK_SCALED_FP8_RECIPES = frozenset({"Float8BlockScaling", "MXFP8BlockScaling"})

# Model-config keys we forward, mapped to the benchmark's flag names.
CONFIG_FLAGS = {
    "hidden_size": "--hidden_size",
    "intermediate_size": "--intermediate_size",
    "num_attention_heads": "--num_attention_heads",
    "num_hidden_layers": "--num_hidden_layers",
    "micro_batch_size": "--micro_batch_size",
    "sequence_length": "--sequence_length",
}


def find_benchmark_script(search_roots: list[Path]) -> Path | None:
    """Locate ``benchmark_gemm.py`` in a Transformer Engine source tree.

    Args:
        search_roots: Directories that might be, or contain, a TE checkout.

    Returns:
        The path to the benchmark script, or None if not found.
    """
    for root in search_roots:
        if root is None:
            continue
        candidate = root / BENCHMARK_RELPATH
        if candidate.is_file():
            return candidate
    return None


def default_search_roots() -> list[Path]:
    """Build the list of plausible Transformer Engine source locations.

    Returns:
        Candidate roots, ordered most to least likely.
    """
    roots: list[Path] = []
    env_root = os.environ.get("TE_SOURCE_DIR") or os.environ.get("TRANSFORMER_ENGINE_SOURCE")
    if env_root:
        roots.append(Path(env_root))

    # NGC containers ship the TE source alongside the install.
    roots += [Path("/opt/transformerengine"), Path("/opt/TransformerEngine"), Path("/workspace/TransformerEngine")]

    try:
        import transformer_engine

        # site-packages/transformer_engine -> walk up looking for a checkout root.
        pkg_dir = Path(transformer_engine.__file__).resolve().parent
        roots += [pkg_dir.parent, pkg_dir.parent.parent]
    except ImportError:
        pass

    return roots


def clone_transformer_engine(dest: Path, version: str | None) -> Path | None:
    """Shallow-clone Transformer Engine so the benchmark script is available.

    Args:
        dest: Directory to clone into.
        version: Installed TE version, used to pick a matching tag.

    Returns:
        The clone root on success, else None.
    """
    if dest.exists():
        return dest
    cmd = ["git", "clone", "--depth", "1"]
    if version:
        cmd += ["--branch", f"v{version.split('+')[0]}"]
    cmd += [TE_REPO_URL, str(dest)]
    print(f"Cloning Transformer Engine: {' '.join(cmd)}", file=sys.stderr)
    if subprocess.run(cmd).returncode == 0:
        return dest
    if version:
        print("Tagged clone failed; retrying on the default branch.", file=sys.stderr)
        fallback = ["git", "clone", "--depth", "1", TE_REPO_URL, str(dest)]
        if subprocess.run(fallback).returncode == 0:
            return dest
    return None


def skip_flags_from_hardware(hardware: dict) -> list[str]:
    """Derive benchmark skip flags from the probe's supported-recipe list.

    Args:
        hardware: Parsed ``hardware.json`` from ``probe_hardware.py``.

    Returns:
        List of ``--no-*`` flags to pass to ``benchmark_gemm.py``.
    """
    supported = set(hardware.get("supported_recipes", []))
    flags: list[str] = []
    if not (BLOCK_SCALED_FP8_RECIPES & supported):
        flags.append("--no-fp8")
    if "NVFP4BlockScaling" not in supported:
        flags.append("--no-fp4")
    return flags


def build_command(
    script: Path, inventory: dict, hardware: dict, output_png: Path, shapes: str | None, pre_quantize: bool
) -> list[str]:
    """Assemble the benchmark command line.

    Args:
        script: Path to ``benchmark_gemm.py``.
        inventory: Parsed ``inventory.json`` with the target's model dimensions.
        hardware: Parsed ``hardware.json``.
        output_png: Where the benchmark writes its plot.
        shapes: Manual ``MxKxN`` triplets; mutually exclusive with model-config flags.
        pre_quantize: Whether to add ``--pre-quantize``.

    Returns:
        The argv list to run.
    """
    cmd = [sys.executable, str(script)]

    if shapes:
        cmd += ["--shapes", shapes]
    else:
        model = inventory.get("model", inventory)
        missing = [key for key in CONFIG_FLAGS if model.get(key) is None]
        if missing:
            raise SystemExit(
                f"inventory is missing required model dimensions: {missing}. "
                "Fill them in during Phase 0, or pass --shapes for manual shape mode."
            )
        for key, flag in CONFIG_FLAGS.items():
            cmd += [flag, str(model[key])]
        cmd += skip_flags_from_hardware(hardware)

    if pre_quantize:
        cmd.append("--pre-quantize")
    cmd += ["-o", str(output_png)]
    return cmd


def run_mode(cmd: list[str], log_path: Path, verbose_kernels: bool) -> dict:
    """Run one benchmark invocation and capture its output.

    Args:
        cmd: The argv list.
        log_path: Where to write combined stdout/stderr.
        verbose_kernels: Set ``NVTE_LOG_LEVEL=1`` to surface kernel dispatch.

    Returns:
        A dict with the command, return code, and log path.
    """
    env = dict(os.environ)
    if verbose_kernels:
        env["NVTE_LOG_LEVEL"] = "1"

    print(f"\n$ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + "\n" + proc.stderr)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
    return {"command": cmd, "returncode": proc.returncode, "log": str(log_path)}


def main() -> int:
    """Locate the benchmark, run both modes, and write a summary.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inventory", type=Path, required=True, help="inventory.json from Phase 0")
    parser.add_argument("--hardware", type=Path, required=True, help="hardware.json from probe_hardware.py")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path(".bionemo-accel/gemm"))
    parser.add_argument(
        "--shapes",
        default=None,
        help="Manual shape mode: comma-separated MxKxN triplets. Mutually exclusive with model config.",
    )
    parser.add_argument("--te-source", type=Path, default=None, help="Path to a Transformer Engine checkout")
    parser.add_argument(
        "--allow-clone",
        action="store_true",
        help="Shallow-clone Transformer Engine if benchmark_gemm.py is not found locally",
    )
    parser.add_argument(
        "--verbose-kernels",
        action="store_true",
        help="Set NVTE_LOG_LEVEL=1 to confirm kernel dispatch (use when a speedup comes back near 1.0x)",
    )
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text())
    hardware = json.loads(args.hardware.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roots = ([args.te_source] if args.te_source else []) + default_search_roots()
    script = find_benchmark_script(roots)
    if script is None and args.allow_clone:
        if shutil.which("git") is None:
            raise SystemExit("git is not available; cannot clone Transformer Engine.")
        version = hardware.get("transformer_engine", {}).get("version")
        clone_root = clone_transformer_engine(args.output_dir.parent / "TransformerEngine", version)
        script = find_benchmark_script([clone_root]) if clone_root else None
    if script is None:
        raise SystemExit(
            f"Could not find {BENCHMARK_RELPATH} in any of: {[str(r) for r in roots if r]}.\n"
            "It ships in the Transformer Engine source tree, not the pip wheel. "
            "Pass --te-source <checkout>, set TE_SOURCE_DIR, or re-run with --allow-clone."
        )
    print(f"Using benchmark script: {script}", file=sys.stderr)

    results = {}
    for mode, pre_quantize in (("autocast", False), ("pre_quantized", True)):
        cmd = build_command(
            script,
            inventory,
            hardware,
            args.output_dir / f"gemm_speedup_{mode}.png",
            args.shapes,
            pre_quantize,
        )
        results[mode] = run_mode(cmd, args.output_dir / f"{mode}.log", args.verbose_kernels)

    skip_flags = skip_flags_from_hardware(hardware) if not args.shapes else []
    successful_runs = {mode: res for mode, res in results.items() if res["returncode"] == 0}

    if successful_runs:
        summary = {
            "benchmark_script": str(script),
            "mode": "manual_shapes" if args.shapes else "model_config",
            "skip_flags_applied": skip_flags,
            "supported_recipes": hardware.get("supported_recipes", []),
            "runs": successful_runs,
            "interpretation_reminders": [
                "GEMM speedup is an UPPER BOUND on end-to-end speedup: attention, LayerNorm/RMSNorm, "
                "activations, and AllReduce are precision-agnostic and are not measured here.",
                "A speedup near 1.0x usually means a silent fallback to a lower-precision kernel. "
                "Re-run with --verbose-kernels (NVTE_LOG_LEVEL=1) and confirm dispatch before concluding.",
                "The autocast-vs-pre_quantized gap is the dynamic quantization overhead. Report both.",
                "NVFP4 carries costs outside the GEMM kernels (random Hadamard transforms, stochastic "
                "rounding, 2D block scaling, amax passes). Discount its headline number.",
                "DelayedScaling always runs in autocast mode; --pre-quantize does not apply to it.",
            ],
        }
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nWrote {summary_path}", file=sys.stderr)
    else:
        print("\nNo benchmark runs succeeded; summary.json not written.", file=sys.stderr)

    return 0 if len(successful_runs) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
