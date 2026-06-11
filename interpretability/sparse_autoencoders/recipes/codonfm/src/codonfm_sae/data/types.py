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

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CodonRecord:
    """Container for a codon DNA sequence record."""

    id: str
    sequence: str  # raw DNA string (e.g., "ATGCGT...")
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_codons(self) -> int:  # noqa: D102
        return len(self.sequence) // 3
