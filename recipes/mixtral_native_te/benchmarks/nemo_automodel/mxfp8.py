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

"""Register an MXFP8 expert implementation for Hugging Face Mixtral.

AutoModel 26.06 ships the differentiable TorchAO MXFP8 grouped GEMM for its
native MoE models, but Hugging Face Mixtral cannot select that backend. This
adapter keeps the HF router and parameter layout and replaces only its two
expert grouped GEMMs.
"""

from __future__ import annotations

import os

import torch


IMPLEMENTATION_NAME = "mxfp8_grouped_mm"
WGRAD_HIGH_PRECISION = os.environ.get("MXFP8_WGRAD_HIGH_PRECISION", "0") == "1"


def register_mxfp8_experts() -> None:
    """Make the local MXFP8 implementation selectable during model loading."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    if IMPLEMENTATION_NAME not in ALL_EXPERTS_FUNCTIONS:
        ALL_EXPERTS_FUNCTIONS.register(IMPLEMENTATION_NAME, mxfp8_grouped_mm_experts_forward)


def mxfp8_grouped_mm_experts_forward(
    experts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Run HF Mixtral routing around TorchAO's Blackwell MXFP8 training GEMM."""
    from torchao.prototype.moe_training import _to_mxfp8_then_scaled_grouped_mm

    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.shape
    sample_weights = top_k_weights.flatten()
    expert_ids, permutation = torch.sort(top_k_index.flatten())
    selected_states = hidden_states[permutation // num_top_k]
    selected_sample_weights = sample_weights[permutation]
    sentinel_mask = expert_ids >= experts.num_experts

    tokens_per_expert = torch.histc(
        expert_ids.int(),
        bins=experts.num_experts,
        min=0,
        max=experts.num_experts - 1,
    )
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    padded_states, padded_indices, padded_offsets = pad_expert_groups(
        selected_states,
        offsets,
        sentinel_mask,
    )

    gate_up_weight = experts.gate_up_proj if experts.is_transposed else experts.gate_up_proj.transpose(-2, -1)
    projected = _to_mxfp8_then_scaled_grouped_mm(
        padded_states,
        gate_up_weight,
        offs=padded_offsets,
        wgrad_with_hp=WGRAD_HIGH_PRECISION,
    )
    padded_row_ids = torch.arange(projected.shape[0], device=projected.device)
    padded_sentinel_mask = padded_row_ids >= padded_offsets[-1]
    projected = projected.masked_fill(padded_sentinel_mask.unsqueeze(-1), 0.0)
    projected = experts._apply_gate(projected)

    down_weight = experts.down_proj if experts.is_transposed else experts.down_proj.transpose(-2, -1)
    projected = _to_mxfp8_then_scaled_grouped_mm(
        projected,
        down_weight,
        offs=padded_offsets,
        wgrad_with_hp=WGRAD_HIGH_PRECISION,
    )
    projected = projected.masked_fill(padded_sentinel_mask.unsqueeze(-1), 0.0)
    projected = projected[padded_indices] * selected_sample_weights.unsqueeze(-1)
    projected = projected.masked_fill(sentinel_mask.unsqueeze(-1), 0.0)

    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    projected = projected[inverse_permutation]
    return projected.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def pad_expert_groups(
    values: torch.Tensor,
    offsets: torch.Tensor,
    sentinel_mask: torch.Tensor,
    alignment: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad expert groups to the alignment required by the installed CuTeDSL kernels."""
    num_rows = values.shape[0]
    num_groups = offsets.shape[0]
    zero = offsets.new_zeros(1)
    starts = torch.cat((zero, offsets[:-1]))
    sizes = offsets - starts
    padded_sizes = torch.div(sizes + alignment - 1, alignment, rounding_mode="floor") * alignment
    padded_offsets = torch.cumsum(padded_sizes, dim=0, dtype=torch.int32)
    padded_starts = torch.cat((zero, padded_offsets[:-1]))

    rows = torch.arange(num_rows, device=values.device, dtype=offsets.dtype)
    group_ids = torch.searchsorted(offsets, rows, right=True).clamp(max=num_groups - 1)
    real_indices = rows - starts[group_ids] + padded_starts[group_ids]
    sentinel_indices = rows + num_groups * alignment
    padded_indices = torch.where(sentinel_mask, sentinel_indices, real_indices).to(torch.long)
    padded = values.new_zeros((num_rows + num_groups * alignment, values.shape[1]))
    padded = padded.index_copy(0, padded_indices, values)
    return padded, padded_indices, padded_offsets
