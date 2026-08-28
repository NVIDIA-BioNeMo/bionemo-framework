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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Deterministic, copy-once rotary embedding for flat packed THD sequences."""

from __future__ import annotations

import torch
from megatron.core.models.common.embeddings import rope_utils
from torch import Tensor


try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


_ORIGINAL_FUSED_THD = rope_utils.fused_apply_rotary_pos_emb_thd
_MAX_SIGNED_INT32 = torch.iinfo(torch.int32).max


if TRITON_AVAILABLE:

    @triton.jit
    def _packed_rope_kernel(
        tokens,
        cos_sin,
        output,
        total_tokens: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        token_stride,
        head_stride,
        dim_stride,
        frequency_token_stride,
        frequency_pair_stride,
        frequency_dim_stride,
        output_token_stride,
        output_head_stride,
        output_dim_stride,
        interleaved: tl.constexpr,
        inverse: tl.constexpr,
        use_64bit_indexing: tl.constexpr,
        block_tokens: tl.constexpr,
        block_heads: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        token = tl.program_id(0) * block_tokens + tl.arange(0, block_tokens)
        if use_64bit_indexing:
            token = token.to(tl.int64)
        head = tl.program_id(1) * block_heads + tl.arange(0, block_heads)
        dimension = tl.arange(0, block_dim)
        token_grid = token[:, None, None]
        head_grid = head[None, :, None]
        dimension_grid = dimension[None, None, :]
        valid = (token_grid < total_tokens) & (head_grid < num_heads) & (dimension_grid < head_dim)
        rotary_valid = valid & (dimension_grid < rotary_dim)

        token_offset = token_grid * token_stride + head_grid * head_stride + dimension_grid * dim_stride
        value = tl.load(tokens + token_offset, mask=valid, other=0.0).to(tl.float32)

        if interleaved:
            partner_dimension = dimension_grid ^ 1
            rotate_sign = tl.where((dimension_grid & 1) == 0, -1.0, 1.0)
        else:
            half_rotary_dim = rotary_dim // 2
            partner_dimension = tl.where(
                dimension_grid < half_rotary_dim,
                dimension_grid + half_rotary_dim,
                dimension_grid - half_rotary_dim,
            )
            rotate_sign = tl.where(dimension_grid < half_rotary_dim, -1.0, 1.0)

        partner_offset = token_grid * token_stride + head_grid * head_stride + partner_dimension * dim_stride
        partner = tl.load(tokens + partner_offset, mask=rotary_valid, other=0.0).to(tl.float32)
        frequency_offset = token_grid * frequency_token_stride + dimension_grid * frequency_dim_stride
        cosine = tl.load(cos_sin + frequency_offset, mask=rotary_valid, other=1.0).to(tl.float32)
        sine = tl.load(
            cos_sin + frequency_pair_stride + frequency_offset,
            mask=rotary_valid,
            other=0.0,
        ).to(tl.float32)
        sine_scale = -1.0 if inverse else 1.0
        rotated = value * cosine + sine_scale * rotate_sign * partner * sine
        result = tl.where(dimension_grid < rotary_dim, rotated, value)

        output_offset = (
            token_grid * output_token_stride + head_grid * output_head_stride + dimension_grid * output_dim_stride
        )
        tl.store(output + output_offset, result, mask=valid)


def preindex_packed_rope_frequencies(
    frequencies: Tensor,
    position_ids: Tensor,
    *,
    total_tokens: int,
) -> Tensor:
    """Gather local-position RoPE frequencies once for a flat packed call."""
    if position_ids.numel() != total_tokens:
        raise ValueError(f"packed position_ids contain {position_ids.numel()} values for {total_tokens} tokens")
    flat_positions = position_ids.reshape(-1).to(device=frequencies.device, dtype=torch.long)
    return frequencies.index_select(0, flat_positions)


def precompute_packed_rope_cos_sin(
    frequencies: Tensor,
    position_ids: Tensor,
    *,
    total_tokens: int,
    dtype: torch.dtype,
) -> Tensor:
    """Gather local positions and compute cosine/sine once for all attention layers."""
    packed_frequencies = preindex_packed_rope_frequencies(
        frequencies,
        position_ids,
        total_tokens=total_tokens,
    )
    return torch.cat((packed_frequencies.cos(), packed_frequencies.sin()), dim=-2).to(dtype=dtype)


def _launch_packed_rope(
    tokens: Tensor,
    cos_sin: Tensor,
    *,
    interleaved: bool,
    inverse: bool,
) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for stride-safe packed RoPE")
    if tokens.ndim != 3 or not tokens.is_cuda:
        raise ValueError(f"packed RoPE tokens must be CUDA [T,H,D], got {tuple(tokens.shape)}")
    if cos_sin.ndim != 4 or cos_sin.shape[:3] != (tokens.shape[0], 1, 2) or cos_sin.shape[-1] > tokens.shape[-1]:
        raise ValueError("packed RoPE cos/sin must have shape [T,1,2,rotary_dim]")

    output = torch.empty(tokens.shape, dtype=tokens.dtype, device=tokens.device)
    block_tokens = 8
    block_heads = 8
    block_dim = triton.next_power_of_2(tokens.shape[-1])
    storage_span = (
        (tokens.shape[0] - 1) * tokens.stride(0)
        + (tokens.shape[1] - 1) * tokens.stride(1)
        + (tokens.shape[2] - 1) * tokens.stride(2)
        + 1
    )
    grid = (
        triton.cdiv(tokens.shape[0], block_tokens),
        triton.cdiv(tokens.shape[1], block_heads),
    )
    _packed_rope_kernel[grid](
        tokens,
        cos_sin,
        output,
        total_tokens=tokens.shape[0],
        num_heads=tokens.shape[1],
        head_dim=tokens.shape[2],
        rotary_dim=cos_sin.shape[-1],
        token_stride=tokens.stride(0),
        head_stride=tokens.stride(1),
        dim_stride=tokens.stride(2),
        frequency_token_stride=cos_sin.stride(0),
        frequency_pair_stride=cos_sin.stride(2),
        frequency_dim_stride=cos_sin.stride(3),
        output_token_stride=output.stride(0),
        output_head_stride=output.stride(1),
        output_dim_stride=output.stride(2),
        interleaved=interleaved,
        inverse=inverse,
        use_64bit_indexing=max(storage_span, output.numel()) > _MAX_SIGNED_INT32,
        block_tokens=block_tokens,
        block_heads=block_heads,
        block_dim=block_dim,
        num_warps=8,
    )
    return output


class _PackedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens: Tensor, cos_sin: Tensor, interleaved: bool) -> Tensor:
        ctx.save_for_backward(cos_sin)
        ctx.interleaved = interleaved
        return _launch_packed_rope(
            tokens,
            cos_sin,
            interleaved=interleaved,
            inverse=False,
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (cos_sin,) = ctx.saved_tensors
        grad_tokens = _launch_packed_rope(
            grad_output,
            cos_sin,
            interleaved=ctx.interleaved,
            inverse=True,
        )
        return grad_tokens, None, None


def apply_preindexed_fused_rope_thd(
    tokens: Tensor,
    cu_seqlens: Tensor,
    frequencies: Tensor,
    cp_size: int = 1,
    cp_rank: int = 0,
    interleaved: bool = False,
) -> Tensor:
    """Apply token-preindexed cosine/sine with a stride-safe flat Triton kernel.

    Transformer Engine's THD kernel reconstructs local positions from ``cu_seqlens``
    on every attention layer. Its 26.07 implementation also overflows 32-bit offsets
    for sufficiently long strided Q/K views. Preindexing once lets this kernel use
    64-bit token offsets when required, with no per-layer boundary lookup or copy.
    """
    if frequencies.shape[0] == tokens.shape[0] and frequencies.shape[-2] == 2:
        return _PackedRoPE.apply(tokens, frequencies, interleaved)

    if _ORIGINAL_FUSED_THD is None:
        raise RuntimeError("Transformer Engine fused THD RoPE is unavailable")
    return _ORIGINAL_FUSED_THD(
        tokens,
        cu_seqlens,
        frequencies,
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleaved=interleaved,
    )


def install_preindexed_packed_rope() -> None:
    """Install the THD dispatcher used by MCore's imported RoPE entry point."""
    if rope_utils.fused_apply_rotary_pos_emb_thd is not apply_preindexed_fused_rope_thd:
        rope_utils.fused_apply_rotary_pos_emb_thd = apply_preindexed_fused_rope_thd


install_preindexed_packed_rope()
