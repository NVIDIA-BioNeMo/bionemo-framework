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

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "summarize_paper_sheet1_candidates.py"


def test_average_protein_identity_uses_exclusive_95_percent_limit(load_script) -> None:
    module = load_script(SCRIPT, "paper_sheet1_summary")
    spec = next(item for item in module.METRIC_SPECS if item.column == "average_protein_percent_identity")

    observed = module._passes_threshold(pd.Series([94.999, 95.0]), spec)

    assert spec.threshold == "<95%"
    assert observed.tolist() == [True, False]
