from __future__ import annotations

from typing import Literal

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


_FP8_MAX = 448.0
_MIN_SCALE = 1e-12


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _block32_launch_config(cols: int) -> tuple[int, int]:
    block_size = min(max(_next_power_of_two(cols), 32), 1024)
    if block_size <= 128:
        return block_size, 2
    if block_size <= 256:
        return block_size, 4
    return block_size, 8


if triton is not None:

    @triton.jit
    def _compute_group_scale_and_store(
        values,
        mask,
        row,
        cols,
        out_ptr,
        out_scale_ptr,
        GROUPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK_SIZE)
        group_ids = offs // 32
        for group_idx in range(GROUPS):
            group_mask = mask & (group_ids == group_idx)
            group_values = tl.where(group_mask, values, 0.0)
            max_abs = tl.max(tl.abs(group_values), axis=0)
            out_scale = tl.maximum(max_abs / 448.0, 1e-12)
            q = (group_values / out_scale).to(tl.float8e4nv)
            tl.store(out_ptr + row * cols + offs, q, mask=group_mask)
            tl.store(out_scale_ptr + row * GROUPS + group_idx, out_scale)

    @triton.jit
    def _fp8_unary_rowwise_kernel(
        x_ptr,
        x_scale_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        op: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols

        x_q = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
        x_scale = tl.load(x_scale_ptr + row).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale

        if op == 0:
            y = tl.maximum(x, 0.0)
        else:
            y = tl.sigmoid(x)

        max_abs = tl.max(tl.abs(y), axis=0)
        out_scale = tl.maximum(max_abs / 448.0, 1e-12)
        y_q = (y / out_scale).to(tl.float8e4nv)
        tl.store(out_ptr + row * cols + offs, y_q, mask=mask)
        tl.store(out_scale_ptr + row, out_scale)


    @triton.jit
    def _fp8_binary_rowwise_kernel(
        x_ptr,
        x_scale_ptr,
        y_ptr,
        y_scale_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        op: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols

        x_q = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
        x_scale = tl.load(x_scale_ptr + row).to(tl.float32)
        y_q = tl.load(y_ptr + row * cols + offs, mask=mask, other=0.0)
        y_scale = tl.load(y_scale_ptr + row).to(tl.float32)

        x = x_q.to(tl.float32) * x_scale
        y = y_q.to(tl.float32) * y_scale
        if op == 0:
            z = x + y
        else:
            z = x * y

        max_abs = tl.max(tl.abs(z), axis=0)
        out_scale = tl.maximum(max_abs / 448.0, 1e-12)
        z_q = (z / out_scale).to(tl.float8e4nv)
        tl.store(out_ptr + row * cols + offs, z_q, mask=mask)
        tl.store(out_scale_ptr + row, out_scale)


    @triton.jit
    def _fp8_mask_mul_rowwise_kernel(
        x_ptr,
        x_scale_ptr,
        mask_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols

        x_q = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
        x_scale = tl.load(x_scale_ptr + row).to(tl.float32)
        m = tl.load(mask_ptr + row).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale * m

        max_abs = tl.max(tl.abs(x), axis=0)
        out_scale = tl.maximum(max_abs / 448.0, 1e-12)
        x_q = (x / out_scale).to(tl.float8e4nv)
        tl.store(out_ptr + row * cols + offs, x_q, mask=mask)
        tl.store(out_scale_ptr + row, out_scale)


    @triton.jit
    def _fp8_layernorm_rowwise_kernel(
        x_ptr,
        x_scale_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols

        x_q = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
        x_scale = tl.load(x_scale_ptr + row).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale

        weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        denom = tl.full((), cols, tl.float32)
        mean = tl.sum(x, axis=0) / denom
        centered = x - mean
        var = tl.sum(centered * centered, axis=0) / denom
        inv_std = tl.rsqrt(var + eps)
        y = centered * inv_std * weight + bias

        max_abs = tl.max(tl.abs(y), axis=0)
        out_scale = tl.maximum(max_abs / 448.0, 1e-12)
        y_q = (y / out_scale).to(tl.float8e4nv)
        tl.store(out_ptr + row * cols + offs, y_q, mask=mask)
        tl.store(out_scale_ptr + row, out_scale)


    @triton.jit
    def _dense_to_fp8_rowwise_kernel(
        x_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        x = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0).to(tl.float32)
        max_abs = tl.max(tl.abs(x), axis=0)
        out_scale = tl.maximum(max_abs / 448.0, 1e-12)
        out = (x / out_scale).to(tl.float8e4nv)
        tl.store(out_ptr + row * cols + offs, out, mask=mask)
        tl.store(out_scale_ptr + row, out_scale)


    @triton.jit
    def _rowwise_max_abs_kernel(
        x_ptr,
        x_scale_ptr,
        out_ptr,
        cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        x_q = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
        x_scale = tl.load(x_scale_ptr + row).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale
        tl.store(out_ptr + row, tl.max(tl.abs(x), axis=0))


    @triton.jit
    def _rowwise_to_tensorwise_kernel(
        x_ptr,
        x_scale_ptr,
        out_ptr,
        cols,
        global_scale,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        x_q = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
        x_scale = tl.load(x_scale_ptr + row).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale
        out = (x / global_scale).to(tl.float8e4nv)
        tl.store(out_ptr + row * cols + offs, out, mask=mask)


    @triton.jit
    def _carrier_to_mxfp8_lhs_kernel(
        payload_ptr,
        scale_ptr,
        out_ptr,
        out_scale_ptr,
        stride_pb,
        stride_pi,
        stride_pj,
        stride_pd,
        stride_sb,
        stride_si,
        stride_sj,
        stride_ob,
        stride_oi,
        stride_ok,
        stride_tb,
        stride_tr,
        stride_tk,
        n,
        channel_offset,
        transpose: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        bd = tl.program_id(0)
        row = tl.program_id(1)
        kblock = tl.program_id(2)
        batch = bd // 32
        channel = (bd % 32) + channel_offset
        offs_k = kblock * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < n

        if transpose:
            payload_ptrs = payload_ptr + batch * stride_pb + offs_k * stride_pi + row * stride_pj + channel * stride_pd
            scale_ptrs = scale_ptr + batch * stride_sb + offs_k * stride_si + row * stride_sj
        else:
            payload_ptrs = payload_ptr + batch * stride_pb + row * stride_pi + offs_k * stride_pj + channel * stride_pd
            scale_ptrs = scale_ptr + batch * stride_sb + row * stride_si + offs_k * stride_sj

        x_q = tl.load(payload_ptrs, mask=mask, other=0.0)
        x_scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale
        max_abs = tl.max(tl.abs(x), axis=0)
        fp8_max = tl.full((), 448.0, tl.float32)
        min_scale = tl.full((), 5.877471754111438e-39, tl.float32)
        base_scale = tl.maximum(max_abs / fp8_max, min_scale)
        scale_inv = tl.maximum(tl.exp2(tl.ceil(tl.log2(base_scale))), min_scale)
        out = (x / scale_inv).to(tl.float8e4nv)

        tl.store(out_ptr + bd * stride_ob + row * stride_oi + offs_k * stride_ok, out, mask=mask)
        tl.store(out_scale_ptr + bd * stride_tb + row * stride_tr + kblock * stride_tk, scale_inv)


    @triton.jit
    def _carrier_to_mxfp8_rhs_kernel(
        payload_ptr,
        scale_ptr,
        out_ptr,
        out_scale_ptr,
        stride_pb,
        stride_pi,
        stride_pj,
        stride_pd,
        stride_sb,
        stride_si,
        stride_sj,
        stride_ob,
        stride_ok,
        stride_on,
        stride_tb,
        stride_tk,
        stride_tn,
        n,
        channel_offset,
        BLOCK_K: tl.constexpr,
    ):
        bd = tl.program_id(0)
        kblock = tl.program_id(1)
        col = tl.program_id(2)
        batch = bd // 32
        channel = (bd % 32) + channel_offset
        offs_k = kblock * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < n

        payload_ptrs = payload_ptr + batch * stride_pb + offs_k * stride_pi + col * stride_pj + channel * stride_pd
        scale_ptrs = scale_ptr + batch * stride_sb + offs_k * stride_si + col * stride_sj
        x_q = tl.load(payload_ptrs, mask=mask, other=0.0)
        x_scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale
        max_abs = tl.max(tl.abs(x), axis=0)
        fp8_max = tl.full((), 448.0, tl.float32)
        min_scale = tl.full((), 5.877471754111438e-39, tl.float32)
        base_scale = tl.maximum(max_abs / fp8_max, min_scale)
        scale_inv = tl.maximum(tl.exp2(tl.ceil(tl.log2(base_scale))), min_scale)
        out = (x / scale_inv).to(tl.float8e4nv)

        tl.store(out_ptr + bd * stride_ob + offs_k * stride_ok + col * stride_on, out, mask=mask)
        tl.store(out_scale_ptr + bd * stride_tb + kblock * stride_tk + col * stride_tn, scale_inv)


    @triton.jit
    def _tri_pair_to_fp8_carrier_kernel(
        x1_ptr,
        x2_ptr,
        out_ptr,
        out_scale_ptr,
        stride_x1b,
        stride_x1i,
        stride_x1j,
        stride_x2b,
        stride_x2i,
        stride_x2j,
        stride_ob,
        stride_oi,
        stride_oj,
        stride_od,
        stride_sb,
        stride_si,
        stride_sj,
        n,
        d_chunk,
        cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // (n * n)
        rem = row % (n * n)
        i = rem // n
        j = rem % n
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        first_half = offs < d_chunk
        second_half = offs >= d_chunk
        offs_second = offs - d_chunk
        x1_ptrs = x1_ptr + (batch * d_chunk + offs) * stride_x1b + i * stride_x1i + j * stride_x1j
        x2_ptrs = x2_ptr + (batch * d_chunk + offs_second) * stride_x2b + i * stride_x2i + j * stride_x2j
        x1 = tl.load(x1_ptrs, mask=mask & first_half, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptrs, mask=mask & second_half, other=0.0).to(tl.float32)
        x = tl.where(first_half, x1, x2)
        max_abs = tl.max(tl.abs(x), axis=0)
        out_scale = tl.maximum(max_abs / 448.0, 1e-12)
        out = (x / out_scale).to(tl.float8e4nv)
        out_ptrs = out_ptr + batch * stride_ob + i * stride_oi + j * stride_oj + offs * stride_od
        tl.store(out_ptrs, out, mask=mask)
        tl.store(out_scale_ptr + batch * stride_sb + i * stride_si + j * stride_sj, out_scale)


    @triton.jit
    def _dense_to_fp8_block32_kernel(
        x_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        BLOCK_SIZE: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        x = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0).to(tl.float32)
        _compute_group_scale_and_store(
            x,
            mask,
            row,
            cols,
            out_ptr,
            out_scale_ptr,
            GROUPS=GROUPS,
            BLOCK_SIZE=BLOCK_SIZE,
        )


    @triton.jit
    def _dense_to_fp8_block32_bias_kernel(
        x_ptr,
        bias_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        BLOCK_SIZE: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        x = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = x + bias
        _compute_group_scale_and_store(
            x,
            mask,
            row,
            cols,
            out_ptr,
            out_scale_ptr,
            GROUPS=GROUPS,
            BLOCK_SIZE=BLOCK_SIZE,
        )


    @triton.jit
    def _fp8_unary_block32_kernel(
        x_ptr,
        x_scale_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        op: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        group_ids = offs // 32
        x_q = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
        x_scale = tl.load(x_scale_ptr + row * GROUPS + group_ids, mask=mask, other=0.0).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale
        if op == 0:
            y = tl.maximum(x, 0.0)
        else:
            y = tl.sigmoid(x)
        _compute_group_scale_and_store(
            y,
            mask,
            row,
            cols,
            out_ptr,
            out_scale_ptr,
            GROUPS=GROUPS,
            BLOCK_SIZE=BLOCK_SIZE,
        )


    @triton.jit
    def _fp8_binary_block32_kernel(
        x_ptr,
        x_scale_ptr,
        y_ptr,
        y_scale_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        op: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        group_ids = offs // 32
        x_q = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
        y_q = tl.load(y_ptr + row * cols + offs, mask=mask, other=0.0)
        x_scale = tl.load(x_scale_ptr + row * GROUPS + group_ids, mask=mask, other=0.0).to(tl.float32)
        y_scale = tl.load(y_scale_ptr + row * GROUPS + group_ids, mask=mask, other=0.0).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale
        y = y_q.to(tl.float32) * y_scale
        if op == 0:
            z = x + y
        else:
            z = x * y
        _compute_group_scale_and_store(
            z,
            mask,
            row,
            cols,
            out_ptr,
            out_scale_ptr,
            GROUPS=GROUPS,
            BLOCK_SIZE=BLOCK_SIZE,
        )


    @triton.jit
    def _fp8_mask_mul_block32_kernel(
        x_ptr,
        x_scale_ptr,
        mask_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        BLOCK_SIZE: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        mask_value = tl.load(mask_ptr + row).to(tl.float32)
        group_ids = offs // 32
        x_q = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
        x_scale = tl.load(x_scale_ptr + row * GROUPS + group_ids, mask=mask, other=0.0).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale * mask_value
        _compute_group_scale_and_store(
            x,
            mask,
            row,
            cols,
            out_ptr,
            out_scale_ptr,
            GROUPS=GROUPS,
            BLOCK_SIZE=BLOCK_SIZE,
        )


    @triton.jit
    def _fp8_layernorm_block32_kernel(
        x_ptr,
        x_scale_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        out_scale_ptr,
        cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        group_ids = offs // 32
        x_q = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0)
        x_scale = tl.load(x_scale_ptr + row * GROUPS + group_ids, mask=mask, other=0.0).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale
        x_masked = tl.where(mask, x, 0.0)
        denom = tl.full((), cols, tl.float32)
        mean = tl.sum(x_masked, axis=0) / denom
        centered = x_masked - mean
        var = tl.sum(centered * centered, axis=0) / denom
        inv_std = tl.rsqrt(var + eps)
        weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = centered * inv_std * weight + bias
        _compute_group_scale_and_store(
            y,
            mask,
            row,
            cols,
            out_ptr,
            out_scale_ptr,
            GROUPS=GROUPS,
            BLOCK_SIZE=BLOCK_SIZE,
        )


    @triton.jit
    def _carrier_block32_mask_to_mxfp8_lhs_kernel(
        payload_ptr,
        scale_ptr,
        mask_ptr,
        out_ptr,
        out_scale_ptr,
        stride_pb,
        stride_pi,
        stride_pj,
        stride_pd,
        stride_sb,
        stride_si,
        stride_sj,
        stride_sg,
        stride_mb,
        stride_mi,
        stride_mj,
        stride_ob,
        stride_oi,
        stride_ok,
        stride_tb,
        stride_tr,
        stride_tk,
        n,
        channel_group,
        transpose: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        bd = tl.program_id(0)
        row = tl.program_id(1)
        kblock = tl.program_id(2)
        batch = bd // 32
        channel = (bd % 32) + channel_group * 32
        offs_k = kblock * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < n

        if transpose:
            payload_ptrs = payload_ptr + batch * stride_pb + offs_k * stride_pi + row * stride_pj + channel * stride_pd
            scale_ptrs = scale_ptr + batch * stride_sb + offs_k * stride_si + row * stride_sj + channel_group * stride_sg
            mask_ptrs = mask_ptr + batch * stride_mb + offs_k * stride_mi + row * stride_mj
        else:
            payload_ptrs = payload_ptr + batch * stride_pb + row * stride_pi + offs_k * stride_pj + channel * stride_pd
            scale_ptrs = scale_ptr + batch * stride_sb + row * stride_si + offs_k * stride_sj + channel_group * stride_sg
            mask_ptrs = mask_ptr + batch * stride_mb + row * stride_mi + offs_k * stride_mj

        x_q = tl.load(payload_ptrs, mask=mask, other=0.0)
        x_scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_mask = tl.load(mask_ptrs, mask=mask, other=0).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale * x_mask
        max_abs = tl.max(tl.abs(x), axis=0)
        fp8_max = tl.full((), 448.0, tl.float32)
        min_scale = tl.full((), 5.877471754111438e-39, tl.float32)
        base_scale = tl.maximum(max_abs / fp8_max, min_scale)
        scale_inv = tl.maximum(tl.exp2(tl.ceil(tl.log2(base_scale))), min_scale)
        out = (x / scale_inv).to(tl.float8e4nv)

        tl.store(out_ptr + bd * stride_ob + row * stride_oi + offs_k * stride_ok, out, mask=mask)
        tl.store(out_scale_ptr + bd * stride_tb + row * stride_tr + kblock * stride_tk, scale_inv)


    @triton.jit
    def _carrier_block32_mask_to_mxfp8_rhs_kernel(
        payload_ptr,
        scale_ptr,
        mask_ptr,
        out_ptr,
        out_scale_ptr,
        stride_pb,
        stride_pi,
        stride_pj,
        stride_pd,
        stride_sb,
        stride_si,
        stride_sj,
        stride_sg,
        stride_mb,
        stride_mi,
        stride_mj,
        stride_ob,
        stride_ok,
        stride_on,
        stride_tb,
        stride_tk,
        stride_tn,
        n,
        channel_group,
        BLOCK_K: tl.constexpr,
    ):
        bd = tl.program_id(0)
        kblock = tl.program_id(1)
        col = tl.program_id(2)
        batch = bd // 32
        channel = (bd % 32) + channel_group * 32
        offs_k = kblock * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < n

        payload_ptrs = payload_ptr + batch * stride_pb + offs_k * stride_pi + col * stride_pj + channel * stride_pd
        scale_ptrs = scale_ptr + batch * stride_sb + offs_k * stride_si + col * stride_sj + channel_group * stride_sg
        mask_ptrs = mask_ptr + batch * stride_mb + offs_k * stride_mi + col * stride_mj
        x_q = tl.load(payload_ptrs, mask=mask, other=0.0)
        x_scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_mask = tl.load(mask_ptrs, mask=mask, other=0).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale * x_mask
        max_abs = tl.max(tl.abs(x), axis=0)
        fp8_max = tl.full((), 448.0, tl.float32)
        min_scale = tl.full((), 5.877471754111438e-39, tl.float32)
        base_scale = tl.maximum(max_abs / fp8_max, min_scale)
        scale_inv = tl.maximum(tl.exp2(tl.ceil(tl.log2(base_scale))), min_scale)
        out = (x / scale_inv).to(tl.float8e4nv)

        tl.store(out_ptr + bd * stride_ob + offs_k * stride_ok + col * stride_on, out, mask=mask)
        tl.store(out_scale_ptr + bd * stride_tb + kblock * stride_tk + col * stride_tn, scale_inv)


    @triton.jit
    def _carrier_block32_to_mxfp8_lhs_kernel(
        payload_ptr,
        scale_ptr,
        out_ptr,
        out_scale_ptr,
        stride_pb,
        stride_pi,
        stride_pj,
        stride_pd,
        stride_sb,
        stride_si,
        stride_sj,
        stride_sg,
        stride_ob,
        stride_oi,
        stride_ok,
        stride_tb,
        stride_tr,
        stride_tk,
        n,
        channel_group,
        transpose: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        bd = tl.program_id(0)
        row = tl.program_id(1)
        kblock = tl.program_id(2)
        batch = bd // 32
        channel = (bd % 32) + channel_group * 32
        offs_k = kblock * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < n

        if transpose:
            payload_ptrs = payload_ptr + batch * stride_pb + offs_k * stride_pi + row * stride_pj + channel * stride_pd
            scale_ptrs = scale_ptr + batch * stride_sb + offs_k * stride_si + row * stride_sj + channel_group * stride_sg
        else:
            payload_ptrs = payload_ptr + batch * stride_pb + row * stride_pi + offs_k * stride_pj + channel * stride_pd
            scale_ptrs = scale_ptr + batch * stride_sb + row * stride_si + offs_k * stride_sj + channel_group * stride_sg

        x_q = tl.load(payload_ptrs, mask=mask, other=0.0)
        x_scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale
        max_abs = tl.max(tl.abs(x), axis=0)
        fp8_max = tl.full((), 448.0, tl.float32)
        min_scale = tl.full((), 5.877471754111438e-39, tl.float32)
        base_scale = tl.maximum(max_abs / fp8_max, min_scale)
        scale_inv = tl.maximum(tl.exp2(tl.ceil(tl.log2(base_scale))), min_scale)
        out = (x / scale_inv).to(tl.float8e4nv)

        tl.store(out_ptr + bd * stride_ob + row * stride_oi + offs_k * stride_ok, out, mask=mask)
        tl.store(out_scale_ptr + bd * stride_tb + row * stride_tr + kblock * stride_tk, scale_inv)


    @triton.jit
    def _carrier_block32_to_mxfp8_rhs_kernel(
        payload_ptr,
        scale_ptr,
        out_ptr,
        out_scale_ptr,
        stride_pb,
        stride_pi,
        stride_pj,
        stride_pd,
        stride_sb,
        stride_si,
        stride_sj,
        stride_sg,
        stride_ob,
        stride_ok,
        stride_on,
        stride_tb,
        stride_tk,
        stride_tn,
        n,
        channel_group,
        BLOCK_K: tl.constexpr,
    ):
        bd = tl.program_id(0)
        kblock = tl.program_id(1)
        col = tl.program_id(2)
        batch = bd // 32
        channel = (bd % 32) + channel_group * 32
        offs_k = kblock * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < n

        payload_ptrs = payload_ptr + batch * stride_pb + offs_k * stride_pi + col * stride_pj + channel * stride_pd
        scale_ptrs = scale_ptr + batch * stride_sb + offs_k * stride_si + col * stride_sj + channel_group * stride_sg
        x_q = tl.load(payload_ptrs, mask=mask, other=0.0)
        x_scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = x_q.to(tl.float32) * x_scale
        max_abs = tl.max(tl.abs(x), axis=0)
        fp8_max = tl.full((), 448.0, tl.float32)
        min_scale = tl.full((), 5.877471754111438e-39, tl.float32)
        base_scale = tl.maximum(max_abs / fp8_max, min_scale)
        scale_inv = tl.maximum(tl.exp2(tl.ceil(tl.log2(base_scale))), min_scale)
        out = (x / scale_inv).to(tl.float8e4nv)

        tl.store(out_ptr + bd * stride_ob + offs_k * stride_ok + col * stride_on, out, mask=mask)
        tl.store(out_scale_ptr + bd * stride_tb + kblock * stride_tk + col * stride_tn, scale_inv)


    @triton.jit
    def _tri_pair_to_fp8_block32_carrier_kernel(
        x1_ptr,
        x2_ptr,
        out_ptr,
        out_scale_ptr,
        stride_x1b,
        stride_x1i,
        stride_x1j,
        stride_x2b,
        stride_x2i,
        stride_x2j,
        stride_ob,
        stride_oi,
        stride_oj,
        stride_od,
        stride_sb,
        stride_si,
        stride_sj,
        stride_sg,
        n,
        d_chunk,
        BLOCK_SIZE: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // (n * n)
        rem = row % (n * n)
        i = rem // n
        j = rem % n
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < (d_chunk * 2)
        first_half = offs < d_chunk
        offs_second = offs - d_chunk
        x1_ptrs = x1_ptr + (batch * d_chunk + offs) * stride_x1b + i * stride_x1i + j * stride_x1j
        x2_ptrs = x2_ptr + (batch * d_chunk + offs_second) * stride_x2b + i * stride_x2i + j * stride_x2j
        x1 = tl.load(x1_ptrs, mask=mask & first_half, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptrs, mask=mask & (~first_half), other=0.0).to(tl.float32)
        x = tl.where(first_half, x1, x2)
        group_ids = offs // 32
        for group_idx in range(GROUPS):
            group_mask = mask & (group_ids == group_idx)
            group_values = tl.where(group_mask, x, 0.0)
            max_abs = tl.max(tl.abs(group_values), axis=0)
            out_scale = tl.maximum(max_abs / 448.0, 1e-12)
            out = (group_values / out_scale).to(tl.float8e4nv)
            out_ptrs = out_ptr + batch * stride_ob + i * stride_oi + j * stride_oj + offs * stride_od
            tl.store(out_ptrs, out, mask=group_mask)
            tl.store(out_scale_ptr + batch * stride_sb + i * stride_si + j * stride_sj + group_idx * stride_sg, out_scale)


def _use_triton(payload: torch.Tensor) -> bool:
    return triton is not None and payload.is_cuda


def _flatten_quantized(payload: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    original_shape = payload.shape
    cols = int(original_shape[-1])
    rows = payload.numel() // cols
    payload_2d = payload.reshape(rows, cols).contiguous()
    scale_1d = scale.reshape(rows).contiguous()
    return payload_2d, scale_1d, original_shape


def _flatten_dense(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    original_shape = tensor.shape
    cols = int(original_shape[-1])
    rows = tensor.numel() // cols
    tensor_2d = tensor.reshape(rows, cols).contiguous()
    return tensor_2d, original_shape


def _restore_quantized(
    payload_2d: torch.Tensor,
    scale_1d: torch.Tensor,
    original_shape: torch.Size,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = tuple(original_shape[:-1])
    payload = payload_2d.reshape(original_shape)
    scale = scale_1d.to(scale_dtype).reshape(*prefix, 1)
    return payload, scale


def _flatten_quantized_block32(payload: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    original_shape = payload.shape
    cols = int(original_shape[-1])
    if cols % 32 != 0:
        raise ValueError(f"block32 quantization requires last dimension divisible by 32, got {cols}")
    rows = payload.numel() // cols
    groups = cols // 32
    payload_2d = payload.reshape(rows, cols).contiguous()
    scale_2d = scale.reshape(rows, groups).contiguous()
    return payload_2d, scale_2d, original_shape


def _restore_quantized_block32(
    payload_2d: torch.Tensor,
    scale_2d: torch.Tensor,
    original_shape: torch.Size,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = tuple(original_shape[:-1])
    groups = int(original_shape[-1]) // 32
    payload = payload_2d.reshape(original_shape)
    scale = scale_2d.to(scale_dtype).reshape(*prefix, groups)
    return payload, scale


def _swizzle_mxfp8_scale_rowwise(scale: torch.Tensor) -> torch.Tensor:
    batch, rows, cols = scale.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    if padded_rows != rows or padded_cols != cols:
        padded = torch.full(
            (batch, padded_rows, padded_cols),
            5.877471754111438e-39,
            device=scale.device,
            dtype=scale.dtype,
        )
        padded[:, :rows, :cols] = scale
        scale = padded
        rows = padded_rows
        cols = padded_cols
    return (
        scale.view(batch, rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view_as(scale)
    )


def _fallback_unary(payload: torch.Tensor, scale: torch.Tensor, op: Literal["relu", "sigmoid"]) -> tuple[torch.Tensor, torch.Tensor]:
    x = payload.to(torch.bfloat16) * scale.to(torch.bfloat16)
    if op == "relu":
        y = torch.relu(x)
    else:
        y = torch.sigmoid(x)
    out_scale = (y.abs().amax(dim=-1, keepdim=True).clamp(min=_MIN_SCALE) / _FP8_MAX).to(scale.dtype)
    out = (y / out_scale.to(y.dtype)).to(torch.float8_e4m3fn)
    return out, out_scale


def _fallback_binary(
    x_payload: torch.Tensor,
    x_scale: torch.Tensor,
    y_payload: torch.Tensor,
    y_scale: torch.Tensor,
    op: Literal["add", "mul"],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_payload.to(torch.bfloat16) * x_scale.to(torch.bfloat16)
    y = y_payload.to(torch.bfloat16) * y_scale.to(torch.bfloat16)
    z = x + y if op == "add" else x * y
    out_scale = (z.abs().amax(dim=-1, keepdim=True).clamp(min=_MIN_SCALE) / _FP8_MAX).to(x_scale.dtype)
    out = (z / out_scale.to(z.dtype)).to(torch.float8_e4m3fn)
    return out, out_scale


def _fallback_mask_mul(payload: torch.Tensor, scale: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = payload.to(torch.bfloat16) * scale.to(torch.bfloat16)
    z = x * mask.unsqueeze(-1).to(x.dtype)
    out_scale = (z.abs().amax(dim=-1, keepdim=True).clamp(min=_MIN_SCALE) / _FP8_MAX).to(scale.dtype)
    out = (z / out_scale.to(z.dtype)).to(torch.float8_e4m3fn)
    return out, out_scale


def _fallback_layernorm(
    payload: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = payload.to(torch.bfloat16) * scale.to(torch.bfloat16)
    y = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=weight.to(torch.bfloat16), bias=bias.to(torch.bfloat16), eps=eps)
    out_scale = (y.abs().amax(dim=-1, keepdim=True).clamp(min=_MIN_SCALE) / _FP8_MAX).to(scale.dtype)
    out = (y / out_scale.to(y.dtype)).to(torch.float8_e4m3fn)
    return out, out_scale


def _fallback_quantize_block32(tensor: torch.Tensor, scale_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    cols = tensor.shape[-1]
    if cols % 32 != 0:
        raise ValueError(f"block32 quantization requires last dimension divisible by 32, got {cols}")
    compute_dtype = torch.float32 if scale_dtype == torch.float32 else torch.bfloat16
    x = tensor.to(compute_dtype).reshape(*tensor.shape[:-1], cols // 32, 32)
    out_scale = (x.abs().amax(dim=-1).clamp(min=_MIN_SCALE) / _FP8_MAX).to(scale_dtype)
    out = (x / out_scale.to(x.dtype).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape_as(tensor)
    return out, out_scale


def _fallback_unary_block32(
    payload: torch.Tensor,
    scale: torch.Tensor,
    op: Literal["relu", "sigmoid"],
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = payload.shape[-1]
    x = payload.reshape(*payload.shape[:-1], cols // 32, 32).to(torch.bfloat16) * scale.to(torch.bfloat16).unsqueeze(-1)
    if op == "relu":
        y = torch.relu(x)
    else:
        y = torch.sigmoid(x)
    out_scale = (y.abs().amax(dim=-1).clamp(min=_MIN_SCALE) / _FP8_MAX).to(scale.dtype)
    out = (y / out_scale.to(y.dtype).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape_as(payload)
    return out, out_scale


def _fallback_binary_block32(
    x_payload: torch.Tensor,
    x_scale: torch.Tensor,
    y_payload: torch.Tensor,
    y_scale: torch.Tensor,
    op: Literal["add", "mul"],
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = x_payload.shape[-1]
    x = x_payload.reshape(*x_payload.shape[:-1], cols // 32, 32).to(torch.bfloat16) * x_scale.to(torch.bfloat16).unsqueeze(-1)
    y = y_payload.reshape(*y_payload.shape[:-1], cols // 32, 32).to(torch.bfloat16) * y_scale.to(torch.bfloat16).unsqueeze(-1)
    z = x + y if op == "add" else x * y
    out_scale = (z.abs().amax(dim=-1).clamp(min=_MIN_SCALE) / _FP8_MAX).to(x_scale.dtype)
    out = (z / out_scale.to(z.dtype).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape_as(x_payload)
    return out, out_scale


def _fallback_mask_mul_block32(payload: torch.Tensor, scale: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cols = payload.shape[-1]
    x = payload.reshape(*payload.shape[:-1], cols // 32, 32).to(torch.bfloat16) * scale.to(torch.bfloat16).unsqueeze(-1)
    z = x * mask.unsqueeze(-1).unsqueeze(-1).to(x.dtype)
    out_scale = (z.abs().amax(dim=-1).clamp(min=_MIN_SCALE) / _FP8_MAX).to(scale.dtype)
    out = (z / out_scale.to(z.dtype).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape_as(payload)
    return out, out_scale


def _fallback_layernorm_block32(
    payload: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = payload.shape[-1]
    x = payload.reshape(*payload.shape[:-1], cols // 32, 32).to(torch.bfloat16) * scale.to(torch.bfloat16).unsqueeze(-1)
    x = x.reshape(*payload.shape[:-1], cols)
    y = torch.nn.functional.layer_norm(x, (cols,), weight=weight.to(torch.bfloat16), bias=bias.to(torch.bfloat16), eps=eps)
    return _fallback_quantize_block32(y, scale.dtype)


def fp8_unary_rows(payload: torch.Tensor, scale: torch.Tensor, op: Literal["relu", "sigmoid"]) -> tuple[torch.Tensor, torch.Tensor]:
    if not _use_triton(payload):
        return _fallback_unary(payload, scale, op)
    payload_2d, scale_1d, original_shape = _flatten_quantized(payload, scale)
    out_2d = torch.empty_like(payload_2d)
    out_scale_1d = torch.empty_like(scale_1d)
    cols = payload_2d.shape[1]
    block_size = min(max(_next_power_of_two(cols), 1), 1024)
    op_code = 0 if op == "relu" else 1
    _fp8_unary_rowwise_kernel[(payload_2d.shape[0],)](
        payload_2d,
        scale_1d,
        out_2d,
        out_scale_1d,
        cols,
        op=op_code,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return _restore_quantized(out_2d, out_scale_1d, original_shape, scale.dtype)


def fp8_binary_rows(
    x_payload: torch.Tensor,
    x_scale: torch.Tensor,
    y_payload: torch.Tensor,
    y_scale: torch.Tensor,
    op: Literal["add", "mul"],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _use_triton(x_payload):
        return _fallback_binary(x_payload, x_scale, y_payload, y_scale, op)
    x_payload_2d, x_scale_1d, original_shape = _flatten_quantized(x_payload, x_scale)
    y_payload_2d, y_scale_1d, other_shape = _flatten_quantized(y_payload, y_scale)
    if original_shape != other_shape:
        raise ValueError(f"Quantized tensors must match, got {tuple(original_shape)} and {tuple(other_shape)}")
    out_2d = torch.empty_like(x_payload_2d)
    out_scale_1d = torch.empty_like(x_scale_1d)
    cols = x_payload_2d.shape[1]
    block_size = min(max(_next_power_of_two(cols), 1), 1024)
    op_code = 0 if op == "add" else 1
    _fp8_binary_rowwise_kernel[(x_payload_2d.shape[0],)](
        x_payload_2d,
        x_scale_1d,
        y_payload_2d,
        y_scale_1d,
        out_2d,
        out_scale_1d,
        cols,
        op=op_code,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return _restore_quantized(out_2d, out_scale_1d, original_shape, x_scale.dtype)


def fp8_mask_mul_rows(payload: torch.Tensor, scale: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _use_triton(payload):
        return _fallback_mask_mul(payload, scale, mask)
    payload_2d, scale_1d, original_shape = _flatten_quantized(payload, scale)
    mask_1d = mask.reshape(-1).to(device=payload.device, dtype=torch.float32).contiguous()
    out_2d = torch.empty_like(payload_2d)
    out_scale_1d = torch.empty_like(scale_1d)
    cols = payload_2d.shape[1]
    block_size = min(max(_next_power_of_two(cols), 1), 1024)
    _fp8_mask_mul_rowwise_kernel[(payload_2d.shape[0],)](
        payload_2d,
        scale_1d,
        mask_1d,
        out_2d,
        out_scale_1d,
        cols,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return _restore_quantized(out_2d, out_scale_1d, original_shape, scale.dtype)


def fp8_layernorm_rows(
    payload: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _use_triton(payload):
        return _fallback_layernorm(payload, scale, weight, bias, eps)
    payload_2d, scale_1d, original_shape = _flatten_quantized(payload, scale)
    out_2d = torch.empty_like(payload_2d)
    out_scale_1d = torch.empty_like(scale_1d)
    cols = payload_2d.shape[1]
    block_size = min(max(_next_power_of_two(cols), 1), 1024)
    _fp8_layernorm_rowwise_kernel[(payload_2d.shape[0],)](
        payload_2d,
        scale_1d,
        weight.to(device=payload.device, dtype=torch.bfloat16).contiguous(),
        bias.to(device=payload.device, dtype=torch.bfloat16).contiguous(),
        out_2d,
        out_scale_1d,
        cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return _restore_quantized(out_2d, out_scale_1d, original_shape, scale.dtype)


def fp8_requantize_rows(tensor: torch.Tensor, *, scale_dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_2d, original_shape = _flatten_dense(tensor)
    if not _use_triton(tensor_2d):
        compute_dtype = torch.float32 if scale_dtype == torch.float32 else torch.bfloat16
        x = tensor_2d.to(compute_dtype)
        out_scale = (x.abs().amax(dim=-1, keepdim=True).clamp(min=_MIN_SCALE) / _FP8_MAX).to(scale_dtype)
        out = (x / out_scale.to(x.dtype)).to(torch.float8_e4m3fn)
        return out.reshape(original_shape), out_scale.reshape(*original_shape[:-1], 1)
    out_2d = torch.empty_like(tensor_2d, dtype=torch.float8_e4m3fn)
    out_scale_1d = torch.empty(tensor_2d.shape[0], device=tensor_2d.device, dtype=torch.float32)
    cols = tensor_2d.shape[1]
    block_size = min(max(_next_power_of_two(cols), 1), 1024)
    _dense_to_fp8_rowwise_kernel[(tensor_2d.shape[0],)](
        tensor_2d,
        out_2d,
        out_scale_1d,
        cols,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return _restore_quantized(out_2d, out_scale_1d, original_shape, scale_dtype)


def fp8_requantize_block32(
    tensor: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_2d, original_shape = _flatten_dense(tensor)
    cols = tensor_2d.shape[1]
    if cols % 32 != 0:
        raise ValueError(f"block32 quantization requires last dimension divisible by 32, got {cols}")
    groups = cols // 32
    if not _use_triton(tensor_2d):
        if bias is not None:
            tensor = tensor + bias.to(device=tensor.device, dtype=tensor.dtype)
        return _fallback_quantize_block32(tensor, scale_dtype)
    out_2d = torch.empty_like(tensor_2d, dtype=torch.float8_e4m3fn)
    out_scale_2d = torch.empty((tensor_2d.shape[0], groups), device=tensor_2d.device, dtype=torch.float32)
    block_size, num_warps = _block32_launch_config(cols)
    if bias is None:
        _dense_to_fp8_block32_kernel[(tensor_2d.shape[0],)](
            tensor_2d,
            out_2d,
            out_scale_2d,
            cols,
            BLOCK_SIZE=block_size,
            GROUPS=groups,
            num_warps=num_warps,
        )
    else:
        _dense_to_fp8_block32_bias_kernel[(tensor_2d.shape[0],)](
            tensor_2d,
            bias.to(device=tensor.device, dtype=tensor.dtype).contiguous(),
            out_2d,
            out_scale_2d,
            cols,
            BLOCK_SIZE=block_size,
            GROUPS=groups,
            num_warps=num_warps,
        )
    return _restore_quantized_block32(out_2d, out_scale_2d, original_shape, scale_dtype)


def fp8_unary_block32(payload: torch.Tensor, scale: torch.Tensor, op: Literal["relu", "sigmoid"]) -> tuple[torch.Tensor, torch.Tensor]:
    if not _use_triton(payload):
        return _fallback_unary_block32(payload, scale, op)
    payload_2d, scale_2d, original_shape = _flatten_quantized_block32(payload, scale)
    cols = payload_2d.shape[1]
    groups = cols // 32
    block_size, num_warps = _block32_launch_config(cols)
    out_2d = torch.empty_like(payload_2d)
    out_scale_2d = torch.empty_like(scale_2d, dtype=torch.float32)
    op_code = 0 if op == "relu" else 1
    _fp8_unary_block32_kernel[(payload_2d.shape[0],)](
        payload_2d,
        scale_2d.to(torch.float32),
        out_2d,
        out_scale_2d,
        cols,
        op=op_code,
        BLOCK_SIZE=block_size,
        GROUPS=groups,
        num_warps=num_warps,
    )
    return _restore_quantized_block32(out_2d, out_scale_2d, original_shape, scale.dtype)


def fp8_binary_block32(
    x_payload: torch.Tensor,
    x_scale: torch.Tensor,
    y_payload: torch.Tensor,
    y_scale: torch.Tensor,
    op: Literal["add", "mul"],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _use_triton(x_payload):
        return _fallback_binary_block32(x_payload, x_scale, y_payload, y_scale, op)
    x_payload_2d, x_scale_2d, original_shape = _flatten_quantized_block32(x_payload, x_scale)
    y_payload_2d, y_scale_2d, other_shape = _flatten_quantized_block32(y_payload, y_scale)
    if original_shape != other_shape:
        raise ValueError(f"Quantized tensors must match, got {tuple(original_shape)} and {tuple(other_shape)}")
    cols = x_payload_2d.shape[1]
    groups = cols // 32
    block_size, num_warps = _block32_launch_config(cols)
    out_2d = torch.empty_like(x_payload_2d)
    out_scale_2d = torch.empty_like(x_scale_2d, dtype=torch.float32)
    op_code = 0 if op == "add" else 1
    _fp8_binary_block32_kernel[(x_payload_2d.shape[0],)](
        x_payload_2d,
        x_scale_2d.to(torch.float32),
        y_payload_2d,
        y_scale_2d.to(torch.float32),
        out_2d,
        out_scale_2d,
        cols,
        op=op_code,
        BLOCK_SIZE=block_size,
        GROUPS=groups,
        num_warps=num_warps,
    )
    return _restore_quantized_block32(out_2d, out_scale_2d, original_shape, x_scale.dtype)


def fp8_mask_mul_block32(payload: torch.Tensor, scale: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _use_triton(payload):
        return _fallback_mask_mul_block32(payload, scale, mask)
    payload_2d, scale_2d, original_shape = _flatten_quantized_block32(payload, scale)
    mask_1d = mask.reshape(-1).to(device=payload.device, dtype=torch.float32).contiguous()
    cols = payload_2d.shape[1]
    groups = cols // 32
    block_size, num_warps = _block32_launch_config(cols)
    out_2d = torch.empty_like(payload_2d)
    out_scale_2d = torch.empty_like(scale_2d, dtype=torch.float32)
    _fp8_mask_mul_block32_kernel[(payload_2d.shape[0],)](
        payload_2d,
        scale_2d.to(torch.float32),
        mask_1d,
        out_2d,
        out_scale_2d,
        cols,
        BLOCK_SIZE=block_size,
        GROUPS=groups,
        num_warps=num_warps,
    )
    return _restore_quantized_block32(out_2d, out_scale_2d, original_shape, scale.dtype)


def fp8_layernorm_block32(
    payload: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _use_triton(payload):
        return _fallback_layernorm_block32(payload, scale, weight, bias, eps)
    payload_2d, scale_2d, original_shape = _flatten_quantized_block32(payload, scale)
    cols = payload_2d.shape[1]
    groups = cols // 32
    block_size, num_warps = _block32_launch_config(cols)
    out_2d = torch.empty_like(payload_2d)
    out_scale_2d = torch.empty_like(scale_2d, dtype=torch.float32)
    _fp8_layernorm_block32_kernel[(payload_2d.shape[0],)](
        payload_2d,
        scale_2d.to(torch.float32),
        weight.to(device=payload.device, dtype=torch.bfloat16).contiguous(),
        bias.to(device=payload.device, dtype=torch.bfloat16).contiguous(),
        out_2d,
        out_scale_2d,
        cols,
        eps,
        BLOCK_SIZE=block_size,
        GROUPS=groups,
        num_warps=num_warps,
    )
    return _restore_quantized_block32(out_2d, out_scale_2d, original_shape, scale.dtype)


def fp8_rowwise_to_tensorwise(payload: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _use_triton(payload):
        x = payload.to(torch.float32) * scale.to(torch.float32)
        global_scale = (x.abs().amax().clamp(min=_MIN_SCALE) / _FP8_MAX).to(torch.float32)
        out = (x / global_scale).to(torch.float8_e4m3fn)
        return out, global_scale.reshape(())

    payload_2d, scale_1d, original_shape = _flatten_quantized(payload, scale)
    cols = payload_2d.shape[1]
    block_size = min(max(_next_power_of_two(cols), 1), 1024)
    row_max = torch.empty(payload_2d.shape[0], device=payload.device, dtype=torch.float32)
    _rowwise_max_abs_kernel[(payload_2d.shape[0],)](
        payload_2d,
        scale_1d.to(torch.float32),
        row_max,
        cols,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    global_scale = (row_max.max().clamp(min=_MIN_SCALE) / _FP8_MAX).to(torch.float32)
    out_2d = torch.empty_like(payload_2d)
    _rowwise_to_tensorwise_kernel[(payload_2d.shape[0],)](
        payload_2d,
        scale_1d.to(torch.float32),
        out_2d,
        cols,
        float(global_scale.item()),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out_2d.reshape(original_shape), global_scale.reshape(())


def fp8_pack_carrier_to_mxfp8_lhs(
    payload: torch.Tensor,
    scale: torch.Tensor,
    *,
    channel_offset: int,
    transpose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if payload.dim() != 4 or scale.dim() != 4:
        raise ValueError("payload and scale must have shape (B, N, N, D) and (B, N, N, 1)")
    batch, n, n2, dim = payload.shape
    if n != n2:
        raise ValueError("carrier pack requires square spatial dimensions")
    if n % 32 != 0:
        raise ValueError("carrier pack requires sequence length divisible by 32")
    if channel_offset < 0 or channel_offset + 32 > dim:
        raise ValueError(f"channel_offset {channel_offset} is out of range for dim={dim}")
    if not _use_triton(payload):
        raise RuntimeError("carrier-to-MXFP8 packing requires Triton on CUDA")

    scale_3d = scale.squeeze(-1).contiguous().to(torch.float32)
    batch_channels = batch * 32
    out = torch.empty((batch_channels, n, n), device=payload.device, dtype=torch.float8_e4m3fn)
    temp_scale = torch.empty((batch_channels, n, n // 32), device=payload.device, dtype=torch.float32)
    _carrier_to_mxfp8_lhs_kernel[(batch_channels, n, n // 32)](
        payload.contiguous(),
        scale_3d,
        out,
        temp_scale,
        *payload.stride(),
        *scale_3d.stride(),
        *out.stride(),
        *temp_scale.stride(),
        n,
        channel_offset,
        transpose=transpose,
        BLOCK_K=32,
        num_warps=4,
    )
    scale_e8 = temp_scale.to(torch.float8_e8m0fnu)
    return out, _swizzle_mxfp8_scale_rowwise(scale_e8)


def fp8_pack_carrier_to_mxfp8_rhs(
    payload: torch.Tensor,
    scale: torch.Tensor,
    *,
    channel_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if payload.dim() != 4 or scale.dim() != 4:
        raise ValueError("payload and scale must have shape (B, N, N, D) and (B, N, N, 1)")
    batch, n, n2, dim = payload.shape
    if n != n2:
        raise ValueError("carrier pack requires square spatial dimensions")
    if n % 32 != 0:
        raise ValueError("carrier pack requires sequence length divisible by 32")
    if channel_offset < 0 or channel_offset + 32 > dim:
        raise ValueError(f"channel_offset {channel_offset} is out of range for dim={dim}")
    if not _use_triton(payload):
        raise RuntimeError("carrier-to-MXFP8 packing requires Triton on CUDA")

    scale_3d = scale.squeeze(-1).contiguous().to(torch.float32)
    batch_channels = batch * 32
    out = torch.empty((batch_channels, n, n), device=payload.device, dtype=torch.float8_e4m3fn)
    temp_scale = torch.empty((batch_channels, n // 32, n), device=payload.device, dtype=torch.float32)
    _carrier_to_mxfp8_rhs_kernel[(batch_channels, n // 32, n)](
        payload.contiguous(),
        scale_3d,
        out,
        temp_scale,
        *payload.stride(),
        *scale_3d.stride(),
        *out.stride(),
        *temp_scale.stride(),
        n,
        channel_offset,
        BLOCK_K=32,
        num_warps=4,
    )
    scale_e8 = temp_scale.to(torch.float8_e8m0fnu)
    return out, _swizzle_mxfp8_scale_rowwise(scale_e8.transpose(1, 2).contiguous())


def fp8_pack_block32_carrier_to_mxfp8_lhs(
    payload: torch.Tensor,
    scale: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    channel_group: int,
    transpose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if payload.dim() != 4 or scale.dim() != 4:
        raise ValueError("payload and scale must have shape (B, N, N, D) and (B, N, N, D/32)")
    batch, n, n2, dim = payload.shape
    groups = dim // 32
    if n != n2:
        raise ValueError("carrier pack requires square spatial dimensions")
    if n % 32 != 0:
        raise ValueError("carrier pack requires sequence length divisible by 32")
    if channel_group < 0 or channel_group >= groups:
        raise ValueError(f"channel_group {channel_group} is out of range for dim={dim}")
    if not _use_triton(payload):
        raise RuntimeError("carrier-to-MXFP8 packing requires Triton on CUDA")

    scale_4d = scale.contiguous().to(torch.float32)
    batch_channels = batch * 32
    out = torch.empty((batch_channels, n, n), device=payload.device, dtype=torch.float8_e4m3fn)
    temp_scale = torch.empty((batch_channels, n, n // 32), device=payload.device, dtype=torch.float32)
    if mask is None:
        _carrier_block32_to_mxfp8_lhs_kernel[(batch_channels, n, n // 32)](
            payload.contiguous(),
            scale_4d,
            out,
            temp_scale,
            *payload.stride(),
            *scale_4d.stride(),
            *out.stride(),
            *temp_scale.stride(),
            n,
            channel_group,
            transpose=transpose,
            BLOCK_K=32,
            num_warps=4,
        )
    else:
        mask_3d = mask.contiguous()
        _carrier_block32_mask_to_mxfp8_lhs_kernel[(batch_channels, n, n // 32)](
            payload.contiguous(),
            scale_4d,
            mask_3d,
            out,
            temp_scale,
            *payload.stride(),
            *scale_4d.stride(),
            *mask_3d.stride(),
            *out.stride(),
            *temp_scale.stride(),
            n,
            channel_group,
            transpose=transpose,
            BLOCK_K=32,
            num_warps=4,
        )
    scale_e8 = temp_scale.to(torch.float8_e8m0fnu)
    return out, _swizzle_mxfp8_scale_rowwise(scale_e8)


def fp8_pack_block32_carrier_to_mxfp8_rhs(
    payload: torch.Tensor,
    scale: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    channel_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if payload.dim() != 4 or scale.dim() != 4:
        raise ValueError("payload and scale must have shape (B, N, N, D) and (B, N, N, D/32)")
    batch, n, n2, dim = payload.shape
    groups = dim // 32
    if n != n2:
        raise ValueError("carrier pack requires square spatial dimensions")
    if n % 32 != 0:
        raise ValueError("carrier pack requires sequence length divisible by 32")
    if channel_group < 0 or channel_group >= groups:
        raise ValueError(f"channel_group {channel_group} is out of range for dim={dim}")
    if not _use_triton(payload):
        raise RuntimeError("carrier-to-MXFP8 packing requires Triton on CUDA")

    scale_4d = scale.contiguous().to(torch.float32)
    batch_channels = batch * 32
    out = torch.empty((batch_channels, n, n), device=payload.device, dtype=torch.float8_e4m3fn)
    temp_scale = torch.empty((batch_channels, n // 32, n), device=payload.device, dtype=torch.float32)
    if mask is None:
        _carrier_block32_to_mxfp8_rhs_kernel[(batch_channels, n // 32, n)](
            payload.contiguous(),
            scale_4d,
            out,
            temp_scale,
            *payload.stride(),
            *scale_4d.stride(),
            *out.stride(),
            *temp_scale.stride(),
            n,
            channel_group,
            BLOCK_K=32,
            num_warps=4,
        )
    else:
        mask_3d = mask.contiguous()
        _carrier_block32_mask_to_mxfp8_rhs_kernel[(batch_channels, n // 32, n)](
            payload.contiguous(),
            scale_4d,
            mask_3d,
            out,
            temp_scale,
            *payload.stride(),
            *scale_4d.stride(),
            *mask_3d.stride(),
            *out.stride(),
            *temp_scale.stride(),
            n,
            channel_group,
            BLOCK_K=32,
            num_warps=4,
        )
    scale_e8 = temp_scale.to(torch.float8_e8m0fnu)
    return out, _swizzle_mxfp8_scale_rowwise(scale_e8.transpose(1, 2).contiguous())


def fp8_tri_outputs_to_carrier(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    batch: int,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x1.dim() != 3 or x2.dim() != 3:
        raise ValueError("tri outputs must have shape (B*D, N, N)")
    if x1.shape != x2.shape:
        raise ValueError(f"tri outputs must match, got {tuple(x1.shape)} and {tuple(x2.shape)}")
    if x1.shape[0] % batch != 0:
        raise ValueError(f"tri output batch dimension {x1.shape[0]} is not divisible by batch={batch}")
    d_chunk = x1.shape[0] // batch
    n = x1.shape[1]
    if x1.shape[2] != n:
        raise ValueError("tri outputs must use square spatial dimensions")
    cols = d_chunk * 2

    if not _use_triton(x1):
        tri_out = torch.cat([x1.reshape(batch, d_chunk, n, n), x2.reshape(batch, d_chunk, n, n)], dim=1).permute(0, 2, 3, 1).contiguous()
        compute_dtype = torch.float32 if scale_dtype == torch.float32 else torch.bfloat16
        tri_out = tri_out.to(compute_dtype)
        out_scale = (tri_out.abs().amax(dim=-1, keepdim=True).clamp(min=_MIN_SCALE) / _FP8_MAX).to(scale_dtype)
        out = (tri_out / out_scale.to(tri_out.dtype)).to(torch.float8_e4m3fn)
        return out, out_scale

    out = torch.empty((batch, n, n, cols), device=x1.device, dtype=torch.float8_e4m3fn)
    out_scale_fp32 = torch.empty((batch, n, n, 1), device=x1.device, dtype=torch.float32)
    block_size = min(max(_next_power_of_two(cols), 1), 1024)
    _tri_pair_to_fp8_carrier_kernel[(batch * n * n,)](
        x1.contiguous(),
        x2.contiguous(),
        out,
        out_scale_fp32,
        *x1.stride(),
        *x2.stride(),
        *out.stride(),
        *out_scale_fp32.stride()[:3],
        n,
        d_chunk,
        cols,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out, out_scale_fp32.to(scale_dtype)


def fp8_tri_outputs_to_block32_carrier(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    batch: int,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x1.dim() != 3 or x2.dim() != 3:
        raise ValueError("tri outputs must have shape (B*D, N, N)")
    if x1.shape != x2.shape:
        raise ValueError(f"tri outputs must match, got {tuple(x1.shape)} and {tuple(x2.shape)}")
    if x1.shape[0] % batch != 0:
        raise ValueError(f"tri output batch dimension {x1.shape[0]} is not divisible by batch={batch}")
    d_chunk = x1.shape[0] // batch
    if d_chunk % 32 != 0:
        raise ValueError(f"block32 carrier requires d_chunk divisible by 32, got {d_chunk}")
    n = x1.shape[1]
    if x1.shape[2] != n:
        raise ValueError("tri outputs must use square spatial dimensions")
    cols = d_chunk * 2
    groups = cols // 32

    if not _use_triton(x1):
        tri_out = torch.cat([x1.reshape(batch, d_chunk, n, n), x2.reshape(batch, d_chunk, n, n)], dim=1).permute(0, 2, 3, 1).contiguous()
        return _fallback_quantize_block32(tri_out, scale_dtype)

    out = torch.empty((batch, n, n, cols), device=x1.device, dtype=torch.float8_e4m3fn)
    out_scale_fp32 = torch.empty((batch, n, n, groups), device=x1.device, dtype=torch.float32)
    block_size, num_warps = _block32_launch_config(cols)
    _tri_pair_to_fp8_block32_carrier_kernel[(batch * n * n,)](
        x1.contiguous(),
        x2.contiguous(),
        out,
        out_scale_fp32,
        *x1.stride(),
        *x2.stride(),
        *out.stride(),
        *out_scale_fp32.stride(),
        n,
        d_chunk,
        BLOCK_SIZE=block_size,
        GROUPS=groups,
        num_warps=num_warps,
    )
    return out, out_scale_fp32.to(scale_dtype)
