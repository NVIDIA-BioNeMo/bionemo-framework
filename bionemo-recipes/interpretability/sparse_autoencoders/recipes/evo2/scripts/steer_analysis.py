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

"""Pure metrics for quantifying SAE feature-steering effects (no model — CPU-testable).

Given a baseline generation and steered generations (produced by ``steer.py`` via
``Evo2SAE.generate``), quantify the causal effect of a feature clamp:

  - divergence:    how far a steered continuation departs from the baseline
  - dose_response: how that effect scales with clamp strength
  - selectivity:   target vs control features at one strength (is the effect feature-specific?)

These are pure functions of the output strings, so the steering analysis is reproducible and
unit-tested without a GPU; ``steer.py`` runs the model and calls them to build the persisted
result.
"""

from __future__ import annotations


def divergence(a: str, b: str) -> tuple[int, float]:
    """Return (first differing index, fraction of differing chars) over the shared prefix length."""
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), n)
    diff = sum(1 for i in range(n) if a[i] != b[i]) / max(1, n)
    return first, diff


def dose_response(baseline: str, steered_by_strength: dict[float, str]) -> list[dict]:
    """Per clamp strength, the divergence from baseline — rows sorted by ascending strength.

    A monotonically rising ``frac_changed`` is the signature of a feature that genuinely steers
    generation (stronger clamp -> larger effect).
    """
    rows = []
    for s in sorted(steered_by_strength):
        first, frac = divergence(baseline, steered_by_strength[s])
        rows.append({"strength": float(s), "first_divergence": int(first), "frac_changed": round(frac, 4)})
    return rows


def selectivity(baseline: str, target_steered: str, control_steered: dict[int, str]) -> dict:
    """Target effect vs control features clamped to the same strength.

    ``selectivity_ratio`` > 1 means the target feature moves generation more than the average
    control — evidence the steering is feature-specific, not a generic "any clamp perturbs output".
    """
    target = divergence(baseline, target_steered)[1]
    controls = {int(c): round(divergence(baseline, seq)[1], 4) for c, seq in control_steered.items()}
    mean_c = sum(controls.values()) / len(controls) if controls else 0.0
    return {
        "target_frac_changed": round(target, 4),
        "control_frac_changed": controls,
        "mean_control_frac_changed": round(mean_c, 4),
        "selectivity_ratio": round(target / max(mean_c, 1e-9), 2),
    }
