# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Genomic sequence masking functions for data preprocessing.

Core functions for genomic data preprocessing during training:
- make_upper_case: Convert lowercase tokens to uppercase
- Evo2MaskingConstants: Standard DNA tokens and control characters

Adapted from NeMo's Evo2 implementation.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch


def _make_upper_case(tokens, lowercase_start=97, lowercase_end=122, case_diff=32):
    """Replace lowercase ASCII characters with uppercase.

    Adapted from: nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils.make_upper_case

    Args:
        tokens: Input tensor containing token IDs (ASCII values)
        lowercase_start: ASCII value for 'a' (default: 97)
        lowercase_end: ASCII value for 'z' (default: 122)
        case_diff: Difference between lowercase and uppercase (default: 32)

    Returns:
        tuple: (uppercase_tensor, lowercase_mask)
    """
    lowercase_mask = (tokens >= lowercase_start) & (tokens <= lowercase_end)
    uppercase_tensor = torch.where(lowercase_mask, tokens - case_diff, tokens)
    return uppercase_tensor, lowercase_mask


class Evo2MaskingConstants:
    """Constants used in Evo2 genomic sequence masking."""

    # Standard DNA tokens: A, C, G, T (both uppercase and lowercase)
    DNA_TOKENS: ClassVar[list[int]] = [65, 67, 71, 84, 97, 99, 103, 116]

    # Control characters used in data formatting
    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@', '#'


@dataclass
class GenomicDataCollator:
    """Wrapper collator that adds genomic-specific masking to any base collator.

    This collator uses composition to wrap any base collator (BSHD, THD, etc.) and
    applies genomic masking to the labels after batching.

    Args:
        base_collator: The underlying collator (e.g., DataCollatorForLanguageModeling)
        uppercase_labels: Whether to uppercase labels. Default: False.
        mask_degenerate_bases: Whether to mask non-ACGT bases. Default: True.
        dna_tokens: Tuple of valid DNA token IDs (A, C, G, T upper+lowercase)
        control_tags: Tuple of control character token IDs (@, #)

    Example:
        >>> from transformers.data.data_collator import DataCollatorForLanguageModeling
        >>> base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        >>> collator = GenomicDataCollator(
        ...     base_collator=base,
        ...     uppercase_labels=False,
        ...     mask_degenerate_bases=True,
        ... )
    """

    base_collator: Any
    uppercase_labels: bool = False
    mask_degenerate_bases: bool = True
    dna_tokens: tuple[int, ...] = (65, 67, 71, 84, 97, 99, 103, 116)  # A, C, G, T (upper+lower)
    control_tags: tuple[int, ...] = (64, 35)  # '@', '#'

    def __call__(self, features: list) -> dict[str, Any]:
        """Apply base collator, then add genomic masking."""
        # Base collator handles batching and CLM label creation
        batch = self.base_collator(features)

        labels = batch["labels"]

        # Step 1: Uppercase labels (inputs stay mixed case)
        if self.uppercase_labels:
            labels, _ = _make_upper_case(labels)

        # Step 2: Mask degenerate bases and control characters
        if self.mask_degenerate_bases:
            dna_tokens_tensor = torch.tensor(self.dna_tokens, device=labels.device)
            control_tensor = torch.tensor(self.control_tags, device=labels.device)

            # Identify non-DNA tokens
            not_dna = ~torch.isin(labels, dna_tokens_tensor)
            is_control = torch.isin(labels, control_tensor)

            # Mask both, but preserve existing -100 values
            labels[(not_dna | is_control) & (labels != -100)] = -100

        batch["labels"] = labels
        return batch
