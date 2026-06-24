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

"""CPU tests for the steering metrics + harness (pure string math + a torch-free fake engine).

No GPU, no checkpoints, no torch — the engine is injected, so ``pick_target``/``run_steering`` run
against the in-file ``FakeEngine``. The fake is intentionally local (not a conftest fixture) so it
stays decoupled from the GPU engine fixtures the sibling server PR adds to ``conftest.py``. The real
engine path is covered by the GPU ``test_steering.py``.
"""

from typing import ClassVar

from evo2_sae.eval.steering import divergence, dose_response, edit_distance, pick_target, run_steering, selectivity


# ----------------------------------------------------------------------------- pure metrics
def test_divergence_identical_and_single_change():
    """Identical strings -> no divergence; one substitution -> shared-prefix len + edit fraction."""
    assert divergence("ACGTACGT", "ACGTACGT") == (8, 0.0)
    first, frac = divergence("ACGTACGT", "ACGAACGT")  # differs only at index 3
    assert first == 3 and abs(frac - 0.125) < 1e-9


def test_divergence_edit_distance_does_not_saturate_on_a_shift():
    """A single early insertion shifts every downstream base.

    Positional Hamming would call this ~100% changed (the saturation that erases a dose curve under
    greedy decode); normalized edit distance sees one insert and reports a tiny effect.
    """
    base = "ACGTACGTACGT"
    shifted = "C" + base[:-1]  # one insertion at the front, everything else shunted right by one
    first, frac = divergence(base, shifted)
    assert first == 0  # diverges immediately...
    assert edit_distance(base, shifted) <= 2  # ...but it's a cheap edit, not a full rewrite
    assert frac < 0.2  # not pinned at ~1.0 the way Hamming would be


def test_dose_response_sorted_and_rises_with_strength():
    """Rows come back ascending in strength; a stronger clamp changes more of the continuation."""
    base = "AAAAAAAA"
    steered = {0.0: "AAAAAAAA", 100.0: "AAAACCCC", 50.0: "AAAAAACC"}  # given out of order
    rows = dose_response(base, steered)
    assert [r["strength"] for r in rows] == [0.0, 50.0, 100.0]
    assert rows[0]["frac_changed"] == 0.0
    assert rows[1]["frac_changed"] < rows[2]["frac_changed"]


def test_selectivity_ratio_high_when_target_specific():
    """Target clamp moves generation, controls barely do -> ratio >> 1 (feature-specific)."""
    base = "AAAAAAAA"
    sel = selectivity(base, "CCCCCCCC", {1: "AAAAAAAA", 2: "AAAAAAAC"})
    assert sel["target_frac_changed"] == 1.0
    assert sel["mean_control_frac_changed"] < 0.1
    assert sel["selectivity_ratio"] > 5


# ----------------------------------------------------------------------------- harness (fake engine)
class FakeEngine:
    """Deterministic, torch-free stand-in for ``Evo2SAE``.

    ``top_features`` returns a fixed ranking whose top entry is *unlabeled* (so target selection has
    to skip it for the labeled one); ``generate`` rewrites a strength-scaled tail, with the target
    feature (7) 10x more potent than any other -> monotone dose + high selectivity.
    """

    labels: ClassVar[dict[int, str]] = {7: "promoter_like", 3: "tata_box"}
    # (feature_id, label, max_activation), sorted by activation descending
    ranking: ClassVar[list[tuple[int, str | None, float]]] = [
        (9, None, 6.0),  # most active overall, but UNLABELED
        (7, "promoter_like", 5.0),
        (3, "tata_box", 2.0),
    ]

    def encode(self, dna: str):
        return dna  # opaque token; top_features below ignores it and uses `ranking`

    def top_features(self, codes, tag_len: int = 0, k: int = 8):
        return [{"feature_id": f, "label": lab, "max_activation": v} for f, lab, v in self.ranking[:k]]

    def generate(self, prompt, organism, features, n_tokens, temperature, top_k):
        assert temperature == 0.0 and top_k == 1  # harness must drive greedy/deterministic decode
        if not features:
            return {"generation": {"sequence": "A" * n_tokens}}
        fid, strength = features[0]["feature_id"], abs(features[0]["strength"])
        potency = 10 if fid == 7 else 100  # feature 7 rewrites 10x more per unit strength
        changed = min(n_tokens, int(strength / potency))
        return {"generation": {"sequence": "A" * (n_tokens - changed) + "C" * changed}}


def test_pick_target_prefers_top_labeled_over_top_active():
    """No --feature -> the most-active *labeled* feature (7), skipping the unlabeled top-active (9)."""
    target, rows = pick_target(FakeEngine(), "ACGTACGTACGT", feature=None)
    assert target == 7
    assert rows[0]["feature_id"] == 9  # table still surfaces the top-active feature


def test_pick_target_honors_explicit_feature():
    target, _ = pick_target(FakeEngine(), "ACGTACGTACGT", feature=3)
    assert target == 3


def test_run_steering_dose_response_is_monotone():
    """A stronger clamp rewrites more (frac up) and bites earlier (shared prefix down)."""
    res = run_steering(
        FakeEngine(),
        "ACGT" * 20,
        "None (raw DNA)",
        target=7,
        controls=[],
        strengths=[0.0, 50.0, 100.0, 200.0],
        n_tokens=60,
        max_clamp=300.0,
    )
    fracs = [r["frac_changed"] for r in res["dose_response"]]
    prefixes = [r["first_divergence"] for r in res["dose_response"]]
    assert fracs == sorted(fracs) and fracs[0] == 0.0 and fracs[-1] > fracs[0]
    assert prefixes == sorted(prefixes, reverse=True)  # non-increasing
    assert res["capped_strengths"] == []
    assert res["selectivity"] is None


def test_run_steering_selectivity_is_feature_specific():
    """Target (potent) vs control (10x weaker) at the strongest clamp -> ratio > 1."""
    res = run_steering(
        FakeEngine(),
        "ACGT" * 20,
        "None (raw DNA)",
        target=7,
        controls=[3],
        strengths=[0.0, 200.0],
        n_tokens=60,
        max_clamp=300.0,
    )
    sel = res["selectivity"]
    assert sel["selectivity_ratio"] > 1
    assert sel["target_frac_changed"] > sel["mean_control_frac_changed"]


def test_run_steering_surfaces_the_clamp_cap():
    """A requested strength above the cap is recorded and steered at the effective (capped) value."""
    res = run_steering(
        FakeEngine(),
        "ACGT" * 20,
        "None (raw DNA)",
        target=7,
        controls=[],
        strengths=[50.0, 200.0],
        n_tokens=60,
        max_clamp=100.0,
    )
    assert res["max_clamp_strength"] == 100.0
    assert res["capped_strengths"] == [200.0]
    # Requested 200 is steered at effective 100 -> int(100/10)=10 of 60 bases rewritten.
    row_200 = next(r for r in res["dose_response"] if r["strength"] == 200.0)
    assert row_200["frac_changed"] == round(10 / 60, 4)
