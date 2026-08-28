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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: recipes/evo2_megatron/src/bionemo/evo2/models/megatron/hyena/paged_kv_cache.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

"""Large-tensor-safe paged KV append for Evo2 dynamic inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


_MAX_SIGNED_INT32 = torch.iinfo(torch.int32).max


def storage_span_elements(shape: Sequence[int], strides: Sequence[int]) -> int:
    """Return the number of elements reachable through a strided tensor view."""
    if len(shape) != len(strides):
        raise ValueError(f"shape/stride rank mismatch: {len(shape)} != {len(strides)}")
    dimensions = tuple(int(dimension) for dimension in shape)
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError(f"tensor dimensions must be nonnegative, got {dimensions}")
    if any(dimension == 0 for dimension in dimensions):
        return 0
    return 1 + sum((dimension - 1) * abs(int(stride)) for dimension, stride in zip(dimensions, strides, strict=True))


def requires_64bit_indexing(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """Return whether signed-int32 pointer offsets cannot cover a tensor view."""
    return storage_span_elements(shape, strides) > _MAX_SIGNED_INT32


try:
    import triton
    import triton.language as tl
    from megatron.core.inference.contexts.fused_kv_append_kernel import (
        triton_append_key_value_cache as _mcore_append_key_value_cache,
    )

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    _mcore_append_key_value_cache = None


if HAVE_TRITON:

    @triton.jit
    def _append_kv_cache_64bit_kernel(
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_idx_ptr,
        local_kv_seq_idx_ptr,
        stride_key_token,
        stride_key_head,
        stride_key_hdim,
        stride_value_token,
        stride_value_head,
        stride_value_hdim,
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,
        stride_cache_hdim,
        n_tokens: tl.int32,
        num_heads: tl.int32,
        h_dim: tl.int32,
        block_size_h: tl.constexpr,
    ):
        """Append K/V after widening every pointer-contributing index."""
        token_idx = tl.program_id(0).to(tl.int64)
        head_idx = tl.program_id(1).to(tl.int64)
        if token_idx >= n_tokens or head_idx >= num_heads:
            return

        block_idx = tl.load(block_idx_ptr + token_idx).to(tl.int64)
        local_pos = tl.load(local_kv_seq_idx_ptr + token_idx).to(tl.int64)
        offs_h = tl.arange(0, block_size_h).to(tl.int64)
        mask_h = offs_h < h_dim

        key_head_ptr = key_ptr + token_idx * stride_key_token + head_idx * stride_key_head
        value_head_ptr = value_ptr + token_idx * stride_value_token + head_idx * stride_value_head
        key_to_write = tl.load(key_head_ptr + offs_h * stride_key_hdim, mask=mask_h, other=0.0)
        value_to_write = tl.load(value_head_ptr + offs_h * stride_value_hdim, mask=mask_h, other=0.0)

        dest_offset = block_idx * stride_cache_block + local_pos * stride_cache_pos + head_idx * stride_cache_head
        tl.store(key_cache_ptr + dest_offset + offs_h * stride_cache_hdim, key_to_write, mask=mask_h)
        tl.store(value_cache_ptr + dest_offset + offs_h * stride_cache_hdim, value_to_write, mask=mask_h)


def triton_append_key_value_cache(
    layer_number: int,
    key: Tensor,
    value: Tensor,
    memory_buffer: Tensor,
    padded_active_token_count: int,
    token_to_block_idx: Tensor,
    token_to_local_position_within_kv_block: Tensor,
) -> None:
    """Delegate narrow appends to MCore and widen only oversized tensor spans."""
    if not HAVE_TRITON or _mcore_append_key_value_cache is None:
        raise RuntimeError("Triton and MCore's fused paged KV append are required")

    n_tokens = int(padded_active_token_count)
    if n_tokens == 0:
        return _mcore_append_key_value_cache(
            layer_number,
            key,
            value,
            memory_buffer,
            padded_active_token_count,
            token_to_block_idx,
            token_to_local_position_within_kv_block,
        )

    key_cache = memory_buffer[0, layer_number]
    value_cache = memory_buffer[1, layer_number]
    key_to_cache = key.squeeze(1)[:n_tokens]
    value_to_cache = value.squeeze(1)[:n_tokens]
    block_idx_active = token_to_block_idx[:n_tokens]
    local_kv_seq_idx_active = token_to_local_position_within_kv_block[:n_tokens]
    participating_tensors = (
        key_to_cache,
        value_to_cache,
        key_cache,
        value_cache,
        block_idx_active,
        local_kv_seq_idx_active,
    )
    use_64bit_indexing = any(
        requires_64bit_indexing(tensor.shape, tensor.stride()) for tensor in participating_tensors
    )
    if not use_64bit_indexing:
        return _mcore_append_key_value_cache(
            layer_number,
            key,
            value,
            memory_buffer,
            padded_active_token_count,
            token_to_block_idx,
            token_to_local_position_within_kv_block,
        )

    if not (key.is_cuda and value.is_cuda and memory_buffer.is_cuda):
        raise ValueError("key, value, and memory_buffer must be CUDA tensors")
    if key.size(1) != 1 or value.size(1) != 1:
        raise ValueError("key and value must have sequence length 1")
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError("sliced key/value caches must be four-dimensional")
    if key_to_cache.dtype != key_cache.dtype:
        raise ValueError(f"key dtype does not match cache dtype: {key_to_cache.dtype} != {key_cache.dtype}")
    if value_to_cache.dtype != value_cache.dtype:
        raise ValueError(f"value dtype does not match cache dtype: {value_to_cache.dtype} != {value_cache.dtype}")

    _, num_heads, h_dim = key_to_cache.shape
    if num_heads != key_cache.shape[-2] or h_dim != key_cache.shape[-1]:
        raise ValueError(
            "key/value geometry does not match cache geometry: "
            f"key={tuple(key_to_cache.shape)}, cache={tuple(key_cache.shape)}"
        )

    block_idx_active = block_idx_active.contiguous()
    local_kv_seq_idx_active = local_kv_seq_idx_active.contiguous()
    cache_strides = key_cache.stride()
    grid = (n_tokens, num_heads)
    _append_kv_cache_64bit_kernel[grid](
        key_to_cache,
        value_to_cache,
        key_cache,
        value_cache,
        block_idx_active,
        local_kv_seq_idx_active,
        key_to_cache.stride(0),
        key_to_cache.stride(1),
        key_to_cache.stride(2),
        value_to_cache.stride(0),
        value_to_cache.stride(1),
        value_to_cache.stride(2),
        cache_strides[0],
        cache_strides[1],
        cache_strides[2],
        cache_strides[3],
        n_tokens=n_tokens,
        num_heads=num_heads,
        h_dim=h_dim,
        block_size_h=triton.next_power_of_2(h_dim),
    )


def install_large_tensor_safe_kv_append() -> bool:
    """Install the Evo2-scoped append into MCore's dynamic-context dispatch."""
    if not HAVE_TRITON or _mcore_append_key_value_cache is None:
        return False
    from megatron.core.inference.contexts import dynamic_context

    dynamic_context.triton_append_key_value_cache = triton_append_key_value_cache
    return True
