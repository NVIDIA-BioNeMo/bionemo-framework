from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def _tri_mul_kernel_k2(
    a_ptr,
    b_ptr,
    o_ptr,
    N,
    stride_ab,
    stride_ai,
    stride_ak,
    stride_ad,
    stride_bb,
    stride_bj,
    stride_bk,
    stride_bd,
    stride_ob,
    stride_oi,
    stride_oj,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_bd = tl.program_id(axis=2)
    batch_idx = pid_bd // 32
    d_idx = pid_bd % 32

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, N, BLOCK_K):
        k_idx = k0 + offs_k
        a_ptrs = (
            a_ptr
            + batch_idx * stride_ab
            + offs_m[:, None] * stride_ai
            + k_idx[None, :] * stride_ak
            + d_idx * stride_ad
        )
        b_ptrs = (
            b_ptr
            + batch_idx * stride_bb
            + offs_n[None, :] * stride_bj
            + k_idx[:, None] * stride_bk
            + d_idx * stride_bd
        )
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < N) & (k_idx[None, :] < N), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (k_idx[:, None] < N), other=0.0)
        acc += tl.dot(a, b)

    o_ptrs = (
        o_ptr
        + batch_idx * stride_ob
        + offs_m[:, None] * stride_oi
        + offs_n[None, :] * stride_oj
        + d_idx * stride_od
    )
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < N) & (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def _tri_mul_kernel_k1(
    a_ptr,
    b_ptr,
    o_ptr,
    N,
    stride_ab,
    stride_ak,
    stride_ai,
    stride_ad,
    stride_bb,
    stride_bk,
    stride_bj,
    stride_bd,
    stride_ob,
    stride_oi,
    stride_oj,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_bd = tl.program_id(axis=2)
    batch_idx = pid_bd // 32
    d_idx = pid_bd % 32

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, N, BLOCK_K):
        k_idx = k0 + offs_k
        a_ptrs = (
            a_ptr
            + batch_idx * stride_ab
            + k_idx[None, :] * stride_ak
            + offs_m[:, None] * stride_ai
            + d_idx * stride_ad
        )
        b_ptrs = (
            b_ptr
            + batch_idx * stride_bb
            + k_idx[:, None] * stride_bk
            + offs_n[None, :] * stride_bj
            + d_idx * stride_bd
        )
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < N) & (k_idx[None, :] < N), other=0.0)
        b = tl.load(b_ptrs, mask=(k_idx[:, None] < N) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    o_ptrs = (
        o_ptr
        + batch_idx * stride_ob
        + offs_m[:, None] * stride_oi
        + offs_n[None, :] * stride_oj
        + d_idx * stride_od
    )
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < N) & (offs_n[None, :] < N))


def tri_mul_fused_triton(a: torch.Tensor, b: torch.Tensor, *, k_dim: int, out_dtype: torch.dtype) -> torch.Tensor:
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("Triton fused tri-mul currently requires bfloat16 inputs")
    if a.dim() != 4 or b.dim() != 4 or a.shape != b.shape:
        raise ValueError("a and b must have matching shape (B, N, N, 32)")
    if a.shape[-1] != 32:
        raise ValueError(f"Triton fused tri-mul requires D_chunk=32, got {a.shape[-1]}")
    if k_dim not in {1, 2}:
        raise ValueError(f"k_dim must be 1 or 2, got {k_dim}")

    B, N, N2, D = a.shape
    if N != N2:
        raise ValueError("Triton fused tri-mul expects square spatial dimensions")

    out = torch.empty((B, N, N, D), device=a.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]), B * 32)
    if k_dim == 2:
        _tri_mul_kernel_k2[grid](
            a,
            b,
            out,
            N,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            a.stride(3),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            b.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
        )
    else:
        _tri_mul_kernel_k1[grid](
            a,
            b,
            out,
            N,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            a.stride(3),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            b.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
        )
    if out_dtype != torch.bfloat16:
        return out.to(out_dtype)
    return out
