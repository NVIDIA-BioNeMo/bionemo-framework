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

"""Tests for tensor-parallel comms helpers (CPU / gloo)."""

import pytest
import torch
from sae.parallel import all_gather_cat, all_reduce_sum, autograd_all_reduce_sum

from ._dist_utils import run_distributed


def _w_all_gather(rank, world):
    t = torch.tensor([rank * 10.0, rank * 10.0 + 1])
    out = all_gather_cat(t, dim=0)
    expected = torch.cat([torch.tensor([r * 10.0, r * 10.0 + 1]) for r in range(world)])
    assert torch.equal(out, expected), (rank, out, expected)
    return out


def _w_all_reduce(rank, world):
    t = torch.full((3,), float(rank + 1))
    all_reduce_sum(t)
    expected = float(sum(range(1, world + 1)))
    assert torch.allclose(t, torch.full((3,), expected)), (rank, t)
    return t


def _w_autograd_all_reduce(rank, world):
    x = (torch.ones(2) * (rank + 1)).requires_grad_(True)
    y = autograd_all_reduce_sum(x)
    y.sum().backward()
    total = float(sum(range(1, world + 1)))
    assert torch.allclose(y.detach(), torch.full((2,), total)), (rank, y)
    # d(sum_b sum_r x_r)/dx_r = 1 -> grad is ones on every rank.
    assert torch.allclose(x.grad, torch.ones(2)), (rank, x.grad)
    return x.grad


@pytest.mark.parametrize("world", [2, 4])
def test_all_gather_cat(world):
    run_distributed(_w_all_gather, world)


@pytest.mark.parametrize("world", [2, 4])
def test_all_reduce_sum(world):
    run_distributed(_w_all_reduce, world)


@pytest.mark.parametrize("world", [2, 4])
def test_autograd_all_reduce_sum(world):
    run_distributed(_w_autograd_all_reduce, world)
