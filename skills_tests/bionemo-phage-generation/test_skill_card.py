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

"""Tests for the release-facing BioNeMo phage generation skill card."""

from pathlib import Path


SKILL_CARD = Path(__file__).parents[2] / "skills" / "bionemo-phage-generation" / "skill-card.md"


def test_declared_license_is_linked() -> None:
    """The release-facing skill card should link its declared license."""
    card = SKILL_CARD.read_text(encoding="utf-8")

    assert "[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)" in card
