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

"""Shared constants and helper functions for the BaseCamp Research dataloader.

This module is used by multiple recipes via ``bionemo.common``.
**It must not import megatron-core, megatron-bridge, or NeMo.**
"""

SEQUENCE_ID_COLUMN_NAME = "contig_id"
SEQUENCE_LENGTH_COLUMN_NAME = "length"
SEQUENCE_COLUMN_NAME = "nt_sequence"


def extract_sample_id(sequence_id: str) -> str:
    """Extract sample ID from sequence ID format: BCR__EXT-SAMPLE1__CT1-1."""
    parts = sequence_id.split("__")[1].split("-")[1:]
    return ".".join(parts)
