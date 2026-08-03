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

"""Reference: Layer-wise quantization assignment utilities.

Demonstrates how to resolve user-specified layer lists into per-layer precision assignments.
"""


def resolve_layer_precision(
    num_layers: int,
    fp8_enabled: bool,
    fp4_enabled: bool,
    fp8_layers: list[int] | None,
    fp4_layers: list[int] | None,
) -> list[str | None]:
    """Resolve layer-wise quantization from user config.

    Takes 1-indexed layer lists and returns 0-indexed precision list.

    Examples:
        # All layers FP8
        resolve_layer_precision(6, fp8_enabled=True, fp4_enabled=False, None, None)
        # -> ["fp8", "fp8", "fp8", "fp8", "fp8", "fp8"]

        # Mixed: layers 1-3 FP8, layers 4-6 FP4
        resolve_layer_precision(6, True, True, [1,2,3], [4,5,6])
        # -> ["fp8", "fp8", "fp8", "fp4", "fp4", "fp4"]
    """
    all_layers = set(range(1, num_layers + 1))

    if fp8_enabled and fp4_enabled and fp8_layers is None and fp4_layers is None:
        raise ValueError("Both fp8 and fp4 enabled but no layer lists specified. Provide explicit layer assignments.")

    # Auto-fill: if one format has explicit layers, other gets remaining
    if fp8_enabled and fp8_layers is None:
        claimed = set(fp4_layers) if fp4_layers else set()
        fp8_layers = sorted(all_layers - claimed)

    if fp4_enabled and fp4_layers is None:
        claimed = set(fp8_layers) if fp8_layers else set()
        fp4_layers = sorted(all_layers - claimed)

    if not fp8_enabled:
        fp8_layers = None
    if not fp4_enabled:
        fp4_layers = None

    # Validate no overlap
    if fp8_layers and fp4_layers:
        overlap = set(fp8_layers) & set(fp4_layers)
        if overlap:
            raise ValueError(f"Overlapping layers: {overlap}")

    fp8_set = set(fp8_layers) if fp8_layers else set()
    fp4_set = set(fp4_layers) if fp4_layers else set()
    return ["fp8" if i in fp8_set else "fp4" if i in fp4_set else None for i in range(1, num_layers + 1)]
