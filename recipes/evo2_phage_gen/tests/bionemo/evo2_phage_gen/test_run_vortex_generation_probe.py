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

"""Tests for the standalone Vortex generation probe helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_vortex_generation_probe.py"


def _load_probe(monkeypatch):
    monkeypatch.setitem(sys.modules, "evo2", types.SimpleNamespace(Evo2=object))
    spec = importlib.util.spec_from_file_location("run_vortex_generation_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_score_values_preserves_batch_alignment(monkeypatch):
    module = _load_probe(monkeypatch)

    assert module._normalize_score_values(None, 2) == [None, None]
    assert module._normalize_score_values(torch.tensor([1.25, 2.5]), 2) == [1.25, 2.5]
    assert module._normalize_score_values([[1.0], ["bad"]], 2) == [1.0, None]
    assert module._normalize_score_values([1.0], 2) == [None, None]
    assert module._normalize_score_values(3.0, 1) == [3.0]


def test_completion_token_accepts_empty_generation(monkeypatch):
    module = _load_probe(monkeypatch)

    assert module._completion_token(" \n\t") == ""
    assert module._completion_token("ACGT extra\n") == "ACGT"
    assert module._completion_token("ACGT<EOS>ignored") == "ACGT"
