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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Genomic sequence masking functions for data preprocessing."""

from dataclasses import dataclass
from typing import Any

import torch


def _make_upper_case(tokens, lowercase_start=97, lowercase_end=122, case_diff=32):
    """Replace lowercase ASCII characters with uppercase."""
    lowercase_mask = (tokens >= lowercase_start) & (tokens <= lowercase_end)
    uppercase_tensor = torch.where(lowercase_mask, tokens - case_diff, tokens)
    return uppercase_tensor, lowercase_mask


@dataclass
class GenomicDataCollator:
    """Wrapper collator that adds genomic-specific masking to any base collator."""

    base_collator: Any
    uppercase_labels: bool = False
    mask_degenerate_bases: bool = True
    dna_tokens: tuple[int, ...] = (65, 67, 71, 84, 97, 99, 103, 116)
    control_tags: tuple[int, ...] = (64, 35)

    def __call__(self, features: list) -> dict[str, Any]:
        """Apply base collator, then add genomic masking."""
        batch = self.base_collator(features)
        labels = batch["labels"]

        if self.uppercase_labels:
            labels, _ = _make_upper_case(labels)

        if self.mask_degenerate_bases:
            dna_tokens_tensor = torch.tensor(self.dna_tokens, device=labels.device)
            control_tensor = torch.tensor(self.control_tags, device=labels.device)
            not_dna = ~torch.isin(labels, dna_tokens_tensor)
            is_control = torch.isin(labels, control_tensor)
            labels[(not_dna | is_control) & (labels != -100)] = -100

        batch["labels"] = labels
        return batch
