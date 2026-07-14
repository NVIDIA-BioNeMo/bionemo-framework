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

"""SAE feature-steering analysis: the engine-driven harness + the pure metrics it scores with.

``run_steering`` clamps a feature via the production ``Evo2SAE.generate`` path (the same
decode-only ``evo2_sae.steering`` hook the server/CLI use) and measures its effect on generation:

  - divergence:    how far a steered continuation departs from the baseline
  - dose_response: how that effect scales with clamp strength
  - selectivity:   target vs control features at one strength (bigger than the controls?)

**Scope — a steering smoke test.** These metrics quantify the *magnitude* of a clamp's effect
(dose_response) and whether it exceeds a few *hand-picked* control features (selectivity). They do
not check the effect's *direction* — that the output moved toward the target feature's labeled
concept — and ``selectivity`` is only as trustworthy as the chosen controls. Verifying steering is
concept-correct (does clamping a "stop-codon" feature yield more stop codons?) against an
activation-matched null is future work: concept-density scoring via ``evo2_sae.eval.probing.labelers``.

**Why edit distance, not positional Hamming.** Steering decodes greedily (``temperature=0``),
so generation is deterministic and autoregressive: the first token a clamp flips shifts every
downstream token, which would pin a position-by-position mismatch fraction at ~1.0 and erase any
dose curve. We therefore measure effect magnitude with a *normalized edit (Levenshtein) distance*,
which does not saturate from a single early shift, and report ``first_divergence`` (the length of
the shared prefix — how many leading bases survive the clamp; smaller = the effect bites earlier)
as the complementary, monotone-friendly signal.

The engine is *injected* into ``pick_target``/``run_steering`` (rather than imported), so this whole
module stays torch-free and CPU-unit-testable with a stub; ``scripts/steer.py`` is just the CLI that
builds a real ``Evo2SAE`` and calls in. Lives in the package (not ``scripts/``) so it imports as a
normal module — no ``sys.path`` games — the same way ``evo2_sae.fasta`` does.
"""

from __future__ import annotations


def common_prefix_len(a: str, b: str) -> int:
    """Number of leading characters ``a`` and ``b`` share (the shared-prefix length)."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings (insert/delete/substitute = cost 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def divergence(a: str, b: str) -> tuple[int, float]:
    """Return ``(shared-prefix length, normalized edit distance)``.

    The first element is how many leading characters survive unchanged (``len`` when identical);
    the second is the edit distance normalized by the longer string's length, in ``[0, 1]`` — an
    insertion-robust measure of how much of the continuation the clamp rewrote.
    """
    first = common_prefix_len(a, b)
    n = max(len(a), len(b))
    frac = edit_distance(a, b) / n if n else 0.0
    return first, frac


def dose_response(baseline: str, steered_by_strength: dict[float, str]) -> list[dict]:
    """Per clamp strength, the divergence from baseline — rows sorted by ascending strength.

    ``frac_changed`` (normalized edit distance) rising and ``first_divergence`` (shared-prefix
    length) shrinking as strength grows is the signature of a feature that genuinely steers
    generation (a stronger clamp rewrites more, and bites earlier).
    """
    rows = []
    for s in sorted(steered_by_strength):
        first, frac = divergence(baseline, steered_by_strength[s])
        rows.append({"strength": float(s), "first_divergence": int(first), "frac_changed": round(frac, 4)})
    return rows


def selectivity(baseline: str, target_steered: str, control_steered: dict[int, str]) -> dict:
    """Target effect vs control features clamped to the same strength.

    Effect magnitude is the normalized edit distance from baseline (see module docstring).
    ``selectivity_ratio`` > 1 means the target feature rewrites generation more than the average
    control — evidence the steering is feature-specific, not a generic "any clamp perturbs output".
    """
    target = divergence(baseline, target_steered)[1]
    controls = {int(c): round(divergence(baseline, seq)[1], 4) for c, seq in control_steered.items()}
    mean_c = sum(controls.values()) / len(controls) if controls else None
    return {
        "target_frac_changed": round(target, 4),
        "control_frac_changed": controls,
        "mean_control_frac_changed": round(mean_c, 4) if mean_c is not None else None,
        # None when there are no controls (ratio undefined) or controls produced zero change
        "selectivity_ratio": round(target / mean_c, 2) if mean_c else None,
    }


# --------------------------------------------------------------------- harness (engine injected)
def pick_target(eng, sequence: str, feature: int | None = None, k: int = 10) -> tuple[int, list[dict]]:
    """Return ``(target_feature, top_rows)``: the steered feature + the printable top-k table.

    Reuses ``Evo2SAE.top_features`` (same ranking the CLI/server show) instead of re-deriving the
    top-k. Honors an explicit ``feature``; else the top-active *labeled* feature; else the single
    most-active feature (``top_rows`` is sorted by activation, strictly-positive features only).
    """
    rows = eng.top_features(eng.encode(sequence), k=k)
    target = feature
    if target is None:
        target = next((r["feature_id"] for r in rows if r["label"]), None)
    if target is None:  # no --feature and no labeled feature in the top-k: steer the most-active one
        target = rows[0]["feature_id"] if rows else 0
    return target, rows


def run_steering(eng, sequence, organism, target, controls, strengths, n_tokens, max_clamp) -> dict:
    """Drive the (real or fake) engine to build the steering result dict — no argparse, no I/O.

    ``max_clamp`` is the engine's ``MAX_CLAMP_STRENGTH``: requested strengths beyond it are
    *silently capped inside* ``generate``, which would make two requested strengths produce an
    identical clamp (a fake "plateau"). We surface that — steer at the effective (capped) strength,
    warn, and record both the cap and which requests were capped.
    """

    def gen(clamps):  # clamps already effective (within +/- max_clamp)
        feats = [{"feature_id": f, "strength": v} for f, v in clamps.items()]
        out = eng.generate(
            prompt=sequence, organism=organism, features=feats, n_tokens=n_tokens, temperature=0.0, top_k=1
        )
        return out["generation"]["sequence"]

    def effective(s: float) -> float:  # mirrors core._sanitize_steering's clamp on |strength|
        return max(-max_clamp, min(max_clamp, s))

    capped = sorted({s for s in strengths if effective(s) != s})
    if capped:
        print(
            f"  WARNING: strength(s) {capped} exceed MAX_CLAMP_STRENGTH={max_clamp}; capped before steering "
            "(equal-after-cap requests will look like a plateau)."
        )

    base = gen({})
    # Dose-response: key rows by the *requested* strength (so the sweep reads as asked), but steer
    # with the effective (capped) value so the result matches what the engine actually applies.
    steered_by_strength = {s: gen({target: effective(s)}) for s in strengths}
    dose = dose_response(base, steered_by_strength)

    sel = None
    if controls:
        s = strengths[-1]
        control_steered = {c: gen({c: effective(s)}) for c in controls}
        sel = selectivity(base, steered_by_strength[s], control_steered)

    return {
        "target_feature": target,
        "sequence": sequence[:80],
        "organism": organism,
        "baseline": base,
        "max_clamp_strength": max_clamp,
        "capped_strengths": capped,
        "dose_response": dose,
        "selectivity": sel,
    }
