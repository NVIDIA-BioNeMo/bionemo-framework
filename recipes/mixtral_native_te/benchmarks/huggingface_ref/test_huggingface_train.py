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

"""Tests for the Hugging Face reference benchmark training helpers."""

import torch

from .train import _pad_expert_groups


def test_pad_expert_groups_preserves_rows_and_gradients() -> None:
    values = torch.arange(15, dtype=torch.float32).reshape(5, 3).requires_grad_()
    # Group sizes [2, 0, 3], including an empty expert.
    offsets = torch.tensor([2, 2, 5], dtype=torch.int32)
    sentinel_mask = torch.zeros(5, dtype=torch.bool)

    padded, indices, padded_offsets = _pad_expert_groups(values, offsets, sentinel_mask, alignment=4)

    torch.testing.assert_close(padded_offsets, torch.tensor([4, 4, 8], dtype=torch.int32))
    torch.testing.assert_close(indices, torch.tensor([0, 1, 4, 5, 6]))
    torch.testing.assert_close(padded[indices], values)
    padded.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_pad_expert_groups_places_ep_sentinels_in_tail() -> None:
    values = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    offsets = torch.tensor([1, 3], dtype=torch.int32)
    sentinel_mask = torch.tensor([False, False, False, True, True])

    padded, indices, padded_offsets = _pad_expert_groups(values, offsets, sentinel_mask, alignment=4)

    torch.testing.assert_close(padded_offsets, torch.tensor([4, 8], dtype=torch.int32))
    torch.testing.assert_close(indices, torch.tensor([0, 4, 5, 11, 12]))
    torch.testing.assert_close(padded[indices], values)
