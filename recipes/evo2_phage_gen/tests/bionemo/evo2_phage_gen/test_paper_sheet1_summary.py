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

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "summarize_paper_sheet1_candidates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("paper_sheet1_summary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_average_protein_identity_uses_exclusive_95_percent_limit() -> None:
    module = _load_script()
    spec = next(item for item in module.METRIC_SPECS if item.column == "average_protein_percent_identity")

    observed = module._passes_threshold(pd.Series([94.999, 95.0]), spec)

    assert spec.threshold == "<95%"
    assert observed.tolist() == [True, False]
