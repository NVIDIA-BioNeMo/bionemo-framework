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

r"""Evo2 SAE steering harness CLI — clamp a feature and measure the causal effect on generation.

Thin wrapper: builds an ``Evo2SAE`` and calls ``evo2_sae.steer_analysis.run_steering`` (the
engine-driven harness + pure metrics live there, CPU-tested). Writes a structured
``steering_results.json`` so the evidence is persisted and reproducible — the steering analog of
how ``extract.py`` persists its outputs.

GPU harness — run on an H100 with the inference engine available; this is not a CPU unit test.

    python steer.py --evo2-ckpt-dir <mbridge> --sae-checkpoint <sae.pt> --layer 26 \
        --sequence ATGGCC... --feature 29244 --controls 12345,54321 --strengths 0,50,100,200 \
        --out steering_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evo2_sae.core import MAX_CLAMP_STRENGTH, Evo2SAE
from evo2_sae.steer_analysis import pick_target, run_steering


def main():
    """Encode a sequence, then steer a target feature (dose-response) + control features (selectivity)."""
    p = argparse.ArgumentParser(description="Evo2 SAE steering harness (clamp -> continuation effect).")
    p.add_argument("--evo2-ckpt-dir", required=True)
    p.add_argument("--sae-checkpoint", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--sequence", required=True)
    p.add_argument("--organism", default="None (raw DNA)")
    p.add_argument("--feature", type=int, default=None, help="Target feature id (default: top labeled feature).")
    p.add_argument("--controls", default="", help="Comma-separated control feature ids (selectivity).")
    p.add_argument("--strengths", default="0,50,100,200", help="Comma-separated clamp strengths to sweep.")
    p.add_argument("--n-tokens", type=int, default=60)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="write the structured steering_results JSON here")
    a = p.parse_args()

    try:
        controls = [int(c) for c in a.controls.split(",") if c.strip()]
        strengths = [float(s) for s in a.strengths.split(",")]
    except ValueError as e:
        raise SystemExit(f"bad --controls/--strengths (need comma-separated ints/floats): {e}")
    if not strengths:
        raise SystemExit("--strengths must list at least one clamp strength")

    eng = Evo2SAE(a.evo2_ckpt_dir, a.sae_checkpoint, a.layer, device=a.device).load()

    # 1. Encode -> the sequence's most-active features (pick a target if not given).
    target, rows = pick_target(eng, a.sequence, a.feature)
    print(f"top features on {a.sequence[:24]}...:")
    for r in rows:
        print(f"  feat {r['feature_id']:6d}  {str(r['label']):18s}  max_act {r['max_activation']:7.2f}")
    if a.feature is None and not any(r["label"] for r in rows):
        print(f"  (no labeled feature; defaulting target to top-active {target})")

    # 2-4. Run the sweep (reuses the production generate path; pure metrics score it).
    result = run_steering(
        eng, a.sequence, a.organism, target, controls, strengths, a.n_tokens, MAX_CLAMP_STRENGTH
    )

    print(f"\nbaseline:  {result['baseline'][:60]}")
    print(f"\n=== dose-response: feature {target} ({eng.labels.get(target)}) ===")
    for r in result["dose_response"]:
        print(f"  strength {r['strength']:7.1f}: prefix@{r['first_divergence']:3d}  {r['frac_changed']:6.1%} rewritten")
    sel = result["selectivity"]
    if sel is not None:
        print(f"\n=== selectivity @ strength {strengths[-1]} (target/control ratio {sel['selectivity_ratio']}) ===")
        print(f"  target {target:6d}: {sel['target_frac_changed']:6.1%} rewritten")
        for c, frac in sel["control_frac_changed"].items():
            print(f"  control {int(c):6d}: {frac:6.1%} rewritten  ({eng.labels.get(int(c))})")

    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote steering results -> {a.out}")


if __name__ == "__main__":
    main()
