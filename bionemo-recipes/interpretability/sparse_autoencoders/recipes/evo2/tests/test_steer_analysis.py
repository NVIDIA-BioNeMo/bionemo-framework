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

"""CPU tests for the steering-effect metrics (pure string math, no model)."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from steer_analysis import divergence, dose_response, selectivity


def test_divergence_identical_and_single_change():
    """Identical strings -> no divergence; one substitution -> first index + fraction."""
    assert divergence("ACGTACGT", "ACGTACGT") == (8, 0.0)
    first, frac = divergence("ACGTACGT", "ACGAACGT")  # differs only at index 3
    assert first == 3 and abs(frac - 0.125) < 1e-9


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
