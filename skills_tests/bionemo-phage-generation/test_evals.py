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

"""Tests for the release-facing BioNeMo phage generation eval dataset."""

import json
from pathlib import Path


EVALS = Path(__file__).parents[2] / "skills" / "bionemo-phage-generation" / "evals" / "evals.json"


def test_portable_handoff_evals() -> None:
    """The eval dataset should exercise the portable handoff contract."""
    evals_text = json.dumps(json.loads(EVALS.read_text(encoding="utf-8")))

    for marker in (
        "VERSION == 2.4",
        "no recipe-local controller",
        "aggregation",
        "absolute-root Codex",
        "absolute-root Claude",
        "original request",
        "fixed required sibling allowlist",
        "integrity-failed skill",
    ):
        assert marker in evals_text
