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

"""Collective-communication helpers for tensor-parallel SAEs.

Thin wrappers over torch.distributed used by the latent-sharded TopK SAE:
- ``all_gather_cat``: gather a per-rank tensor and concatenate (used to collect
  each shard's top-k candidates before the global top-k). Non-differentiable.
- ``all_reduce_sum`` / ``autograd_all_reduce_sum``: sum a tensor across ranks. The
  autograd variant is used to combine per-rank partial reconstructions: the summed
  output is replicated, so the same loss-gradient lands on every rank and passes
  straight back to each rank's partial (identity backward).

For pure tensor parallelism (the Phase A / 1M case) the TP group is the default
process group, so ``group=None`` everywhere.
"""

import torch
import torch.distributed as dist


def all_gather_cat(tensor: torch.Tensor, group=None, dim: int = 0) -> torch.Tensor:
    """All-gather `tensor` from every rank and concatenate along `dim`."""
    world_size = dist.get_world_size(group)
    parts = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(parts, tensor.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def all_reduce_sum(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """In-place sum-reduce across ranks (for non-differentiable tensors / grads)."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # The summed output is replicated and the downstream loss is identical on
        # every rank, so grad_output is the same on all ranks and d(sum)/d(x_r)=I.
        # Each rank therefore keeps the incoming gradient unchanged.
        return grad_output, None


def autograd_all_reduce_sum(x: torch.Tensor, group=None) -> torch.Tensor:
    """Differentiable sum-all-reduce (combine per-rank partial reconstructions)."""
    return _AllReduceSum.apply(x, group)
