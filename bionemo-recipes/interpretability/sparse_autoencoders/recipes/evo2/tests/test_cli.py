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

"""CPU test for the generate CLI's --clamp parsing (no model)."""

from evo2_sae.cli import _parse_clamps


def test_parse_clamps_id_and_strength():
    assert _parse_clamps(["29244:300", "88:1.5"]) == [
        {"feature_id": 29244, "strength": 300.0},
        {"feature_id": 88, "strength": 1.5},
    ]


def test_parse_clamps_default_strength():
    assert _parse_clamps(["29244"]) == [{"feature_id": 29244, "strength": 1.0}]


def test_parse_clamps_empty():
    assert _parse_clamps([]) == []
