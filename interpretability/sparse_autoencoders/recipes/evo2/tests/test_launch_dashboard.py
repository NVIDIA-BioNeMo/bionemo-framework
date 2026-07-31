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

"""CPU tests for the dashboard launcher's data staging (no npm/vite, no model)."""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import launch_dashboard as L


def _write(path, with_feature_id=True):
    cols = {"feature_id": [0, 1], "x": [0.1, 0.2]} if with_feature_id else {"x": [0.1, 0.2]}
    pq.write_table(pa.table(cols), path)


def test_stage_copies_required_parquets(tmp_path):
    data, public = tmp_path / "data", tmp_path / "public"
    data.mkdir()
    for f in L.REQUIRED_PARQUETS:
        _write(data / f)
    staged = L.stage_dashboard_data(data, public)
    assert set(staged) == set(L.REQUIRED_PARQUETS)
    assert all((public / f).exists() for f in L.REQUIRED_PARQUETS)


def test_stage_missing_parquet_fails_fast(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "features_atlas.parquet")  # only one of three
    with pytest.raises(FileNotFoundError):
        L.stage_dashboard_data(data, tmp_path / "public")


def test_stage_rejects_wrong_schema(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for f in L.REQUIRED_PARQUETS:
        _write(data / f, with_feature_id=False)  # no feature_id column
    with pytest.raises(ValueError):
        L.stage_dashboard_data(data, tmp_path / "public")
