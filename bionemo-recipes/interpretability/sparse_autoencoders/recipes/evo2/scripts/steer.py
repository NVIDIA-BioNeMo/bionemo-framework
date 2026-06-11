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

r"""Evo2 SAE steering harness — clamp features and measure the causal effect on generation.

Uses ``sae.steering.clamp_hook`` (the shared delta-clamp) registered on the Evo2 decoder layer
the SAE was trained on. Workflow: encode a sequence to find its active features, then for a
**target** feature sweep the clamp strength (dose-response) and for **control** features apply
the same clamp (selectivity), each time comparing the steered continuation to the baseline.

GPU harness — run on an H100 with the inference engine available; this is not a CPU unit test.

    python steer.py --evo2-ckpt-dir <mbridge> --sae-checkpoint <sae.pt> --layer 26 \
        --sequence ATGGCC... --feature 29244 --controls 12345,54321 --strengths 0,50,100,200

Note: ``sae.steering.clamp_hook`` clamps on *every* forward (prefill + decode), so it steers
the prompt as well as the continuation. The decode-only ("continuation-only") variant lives in
``evo2_sae.core.Evo2SAE._clamp_hook``; unifying the two onto ``sae.steering`` (with a
``decode_only`` flag) is a planned follow-up.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))  # recipes/evo2/src -> evo2_sae package
sys.path.insert(0, str(_HERE.parents[2] / "sae" / "src"))

from sae.steering import steer  # noqa: E402


def _divergence(a: str, b: str):
    """Return (first differing index, fraction of differing chars) over the shared prefix length."""
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), n)
    diff = sum(1 for i in range(n) if a[i] != b[i]) / max(1, n)
    return first, diff


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
    a = p.parse_args()

    from bionemo.evo2.run import infer as INF  # noqa: E402, I001, RUF100
    from evo2_sae.core import Evo2SAE, clean_dna  # noqa: E402, RUF100
    from megatron.core.utils import unwrap_model  # noqa: E402, RUF100

    eng = Evo2SAE(a.evo2_ckpt_dir, a.sae_checkpoint, a.layer, device=a.device).load()

    # 1. Encode -> the sequence's most-active features (pick a target if not given).
    codes = eng.encode(a.sequence)
    vals, ids = codes.max(0).values.topk(10)
    print(f"top features on {a.sequence[:24]}...:")
    target = a.feature
    for v, i in zip(vals.tolist(), ids.tolist()):
        lab = eng.labels.get(int(i))
        print(f"  feat {int(i):6d}  {str(lab):18s}  max_act {v:7.2f}")
        if target is None and lab:
            target = int(i)
    controls = [int(c) for c in a.controls.split(",") if c.strip()]
    strengths = [float(s) for s in a.strengths.split(",")]

    # 2. The Evo2 decoder layer the SAE hooks + a clean (tag + DNA) prompt.
    comp = eng._ensure_engine()
    prompt = (eng.resolve_tag(a.organism, None) or "") + clean_dna(a.sequence)
    layer_mod = unwrap_model(comp.model).decoder.layers[a.layer]

    def gen(clamps):
        ctx = steer(layer_mod, eng.sae, clamps) if clamps else nullcontext()
        with ctx:
            out = INF.generate(comp, [prompt], max_new_tokens=a.n_tokens, temperature=0.0, top_k=1)
        return clean_dna(INF._unwrap_result(out[0]).generated_text)

    base = gen({})
    print(f"\nbaseline:  {base[:60]}")
    print(f"\n=== dose-response: feature {target} ({eng.labels.get(target)}) ===")
    for s in strengths:
        steered = gen({target: s})
        first, diff = _divergence(base, steered)
        print(f"  strength {s:7.1f}: diverges@{first:3d}  {diff:6.1%} changed   {steered[:44]}")

    if controls:
        s = strengths[-1]
        print(f"\n=== selectivity: control features clamped to {s} ===")
        for c in controls:
            steered = gen({c: s})
            first, diff = _divergence(base, steered)
            print(f"  control {c:6d} ({str(eng.labels.get(c)):16s}): diverges@{first:3d}  {diff:6.1%} changed")


if __name__ == "__main__":
    main()
