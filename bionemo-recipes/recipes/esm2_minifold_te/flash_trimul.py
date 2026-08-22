import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _apply_gate_and_mask(proj: torch.Tensor, gate_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return proj * _sigmoid(gate_logits) * mask.unsqueeze(-1).to(dtype=proj.dtype)


def flash_trimul_reference(proj: torch.Tensor, gate_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if proj.shape != gate_logits.shape:
        raise ValueError(f"proj and gate_logits must match, got {tuple(proj.shape)} and {tuple(gate_logits.shape)}")
    if proj.dim() != 4:
        raise ValueError("proj and gate_logits must have shape (B, N, N, D)")
    if proj.shape[:-1] != mask.shape:
        raise ValueError(f"mask must have shape {tuple(proj.shape[:-1])}, got {tuple(mask.shape)}")
    if proj.shape[-1] % 4 != 0:
        raise ValueError(f"hidden dim must be divisible by 4, got {proj.shape[-1]}")

    gated = _apply_gate_and_mask(proj, gate_logits, mask)
    a1, b1, a2, b2 = torch.chunk(gated, 4, dim=-1)
    x1 = torch.matmul(a1.permute(0, 3, 1, 2), b1.permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
    x2 = torch.matmul(a2.permute(0, 3, 2, 1), b2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return torch.cat([x1, x2], dim=-1)


if triton is not None:

    @triton.jit
    def _flash_trimul_outgoing_kernel(
        proj_ptr,
        gate_ptr,
        mask_ptr,
        out_ptr,
        stride_pb,
        stride_pi,
        stride_pj,
        stride_pd,
        stride_mb,
        stride_mi,
        stride_mj,
        stride_ob,
        stride_oi,
        stride_oj,
        stride_od,
        N: tl.constexpr,
        C: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_bd = tl.program_id(2)
        batch = pid_bd // C
        d = pid_bd % C

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, N, BLOCK_K):
            k = k0 + offs_k
            a_mask = (offs_m[:, None] < N) & (k[None, :] < N)
            b_mask = (offs_n[:, None] < N) & (k[None, :] < N)

            a_proj = tl.load(
                proj_ptr + batch * stride_pb + offs_m[:, None] * stride_pi + k[None, :] * stride_pj + d * stride_pd,
                mask=a_mask,
                other=0.0,
            )
            a_gate = tl.load(
                gate_ptr + batch * stride_pb + offs_m[:, None] * stride_pi + k[None, :] * stride_pj + d * stride_pd,
                mask=a_mask,
                other=0.0,
            )
            a_m = tl.load(
                mask_ptr + batch * stride_mb + offs_m[:, None] * stride_mi + k[None, :] * stride_mj,
                mask=a_mask,
                other=0.0,
            ).to(a_proj.dtype)

            b_proj = tl.load(
                proj_ptr
                + batch * stride_pb
                + offs_n[:, None] * stride_pi
                + k[None, :] * stride_pj
                + (d + C) * stride_pd,
                mask=b_mask,
                other=0.0,
            )
            b_gate = tl.load(
                gate_ptr
                + batch * stride_pb
                + offs_n[:, None] * stride_pi
                + k[None, :] * stride_pj
                + (d + C) * stride_pd,
                mask=b_mask,
                other=0.0,
            )
            b_m = tl.load(
                mask_ptr + batch * stride_mb + offs_n[:, None] * stride_mi + k[None, :] * stride_mj,
                mask=b_mask,
                other=0.0,
            ).to(b_proj.dtype)

            a = a_proj * tl.sigmoid(a_gate.to(tl.float32)).to(a_proj.dtype) * a_m
            b = b_proj * tl.sigmoid(b_gate.to(tl.float32)).to(b_proj.dtype) * b_m
            acc += tl.dot(a, tl.trans(b))

        out_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        tl.store(
            out_ptr + batch * stride_ob + offs_m[:, None] * stride_oi + offs_n[None, :] * stride_oj + d * stride_od,
            acc,
            mask=out_mask,
        )


    @triton.jit
    def _flash_trimul_incoming_kernel(
        proj_ptr,
        gate_ptr,
        mask_ptr,
        out_ptr,
        stride_pb,
        stride_pi,
        stride_pj,
        stride_pd,
        stride_mb,
        stride_mi,
        stride_mj,
        stride_ob,
        stride_oi,
        stride_oj,
        stride_od,
        N: tl.constexpr,
        C: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_bd = tl.program_id(2)
        batch = pid_bd // C
        d = pid_bd % C

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, N, BLOCK_K):
            k = k0 + offs_k
            a_mask = (k[:, None] < N) & (offs_m[None, :] < N)
            b_mask = (k[:, None] < N) & (offs_n[None, :] < N)

            a_proj = tl.load(
                proj_ptr
                + batch * stride_pb
                + k[:, None] * stride_pi
                + offs_m[None, :] * stride_pj
                + (d + 2 * C) * stride_pd,
                mask=a_mask,
                other=0.0,
            )
            a_gate = tl.load(
                gate_ptr
                + batch * stride_pb
                + k[:, None] * stride_pi
                + offs_m[None, :] * stride_pj
                + (d + 2 * C) * stride_pd,
                mask=a_mask,
                other=0.0,
            )
            a_m = tl.load(
                mask_ptr + batch * stride_mb + k[:, None] * stride_mi + offs_m[None, :] * stride_mj,
                mask=a_mask,
                other=0.0,
            ).to(a_proj.dtype)

            b_proj = tl.load(
                proj_ptr
                + batch * stride_pb
                + k[:, None] * stride_pi
                + offs_n[None, :] * stride_pj
                + (d + 3 * C) * stride_pd,
                mask=b_mask,
                other=0.0,
            )
            b_gate = tl.load(
                gate_ptr
                + batch * stride_pb
                + k[:, None] * stride_pi
                + offs_n[None, :] * stride_pj
                + (d + 3 * C) * stride_pd,
                mask=b_mask,
                other=0.0,
            )
            b_m = tl.load(
                mask_ptr + batch * stride_mb + k[:, None] * stride_mi + offs_n[None, :] * stride_mj,
                mask=b_mask,
                other=0.0,
            ).to(b_proj.dtype)

            a = a_proj * tl.sigmoid(a_gate.to(tl.float32)).to(a_proj.dtype) * a_m
            b = b_proj * tl.sigmoid(b_gate.to(tl.float32)).to(b_proj.dtype) * b_m
            acc += tl.dot(tl.trans(a), b)

        out_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        tl.store(
            out_ptr + batch * stride_ob + offs_m[:, None] * stride_oi + offs_n[None, :] * stride_oj + d * stride_od,
            acc,
            mask=out_mask,
        )


def _flash_trimul_forward_cuda(proj: torch.Tensor, gate_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if triton is None or proj.device.type != "cuda" or proj.shape[-1] % 4 != 0:
        return flash_trimul_reference(proj, gate_logits, mask)

    B, N, _, D = proj.shape
    c = D // 4
    out1 = torch.empty((B, N, N, c), device=proj.device, dtype=proj.dtype)
    out2 = torch.empty_like(out1)
    grid = (triton.cdiv(N, 32), triton.cdiv(N, 32), B * c)

    _flash_trimul_outgoing_kernel[grid](
        proj,
        gate_logits,
        mask,
        out1,
        *proj.stride(),
        *mask.stride(),
        *out1.stride(),
        N=N,
        C=c,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )
    _flash_trimul_incoming_kernel[grid](
        proj,
        gate_logits,
        mask,
        out2,
        *proj.stride(),
        *mask.stride(),
        *out2.stride(),
        N=N,
        C=c,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )
    return torch.cat([out1, out2], dim=-1)


def _flash_trimul_backward_reference(
    proj: torch.Tensor,
    gate_logits: torch.Tensor,
    mask: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gated = _apply_gate_and_mask(proj, gate_logits, mask)
    mask_t = mask.unsqueeze(-1).to(dtype=proj.dtype)

    a1, b1, a2, b2 = torch.chunk(gated, 4, dim=-1)
    proj_a1, proj_b1, proj_a2, proj_b2 = torch.chunk(proj, 4, dim=-1)
    gate_a1, gate_b1, gate_a2, gate_b2 = torch.chunk(gate_logits, 4, dim=-1)
    g1, g2 = torch.chunk(grad_out, 2, dim=-1)

    ga1 = torch.matmul(g1.permute(0, 3, 1, 2), b1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    gb1 = torch.matmul(g1.permute(0, 3, 2, 1), a1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    ga2 = torch.matmul(b2.permute(0, 3, 1, 2), g2.permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
    gb2 = torch.matmul(a2.permute(0, 3, 1, 2), g2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def project_back(grad_gated: torch.Tensor, proj_chunk: torch.Tensor, gate_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sig = _sigmoid(gate_chunk)
        dproj = grad_gated * sig * mask_t
        dgate = grad_gated * proj_chunk * sig * (1.0 - sig) * mask_t
        return dproj, dgate

    dp1, dg1 = project_back(ga1, proj_a1, gate_a1)
    dp2, dg2 = project_back(gb1, proj_b1, gate_b1)
    dp3, dg3 = project_back(ga2, proj_a2, gate_a2)
    dp4, dg4 = project_back(gb2, proj_b2, gate_b2)
    return torch.cat([dp1, dp2, dp3, dp4], dim=-1), torch.cat([dg1, dg2, dg3, dg4], dim=-1)


class _FlashTriMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proj: torch.Tensor, gate_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if proj.shape != gate_logits.shape:
            raise ValueError(f"proj and gate_logits must match, got {tuple(proj.shape)} and {tuple(gate_logits.shape)}")
        if proj.dim() != 4:
            raise ValueError("proj and gate_logits must have shape (B, N, N, D)")
        if proj.shape[:-1] != mask.shape:
            raise ValueError(f"mask must have shape {tuple(proj.shape[:-1])}, got {tuple(mask.shape)}")

        ctx.save_for_backward(proj, gate_logits, mask)
        return _flash_trimul_forward_cuda(proj, gate_logits, mask)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        proj, gate_logits, mask = ctx.saved_tensors
        dproj, dgate = _flash_trimul_backward_reference(proj, gate_logits, mask, grad_out)
        return dproj, dgate, None


def flash_trimul(proj: torch.Tensor, gate_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fuse gate application into the triangular GEMM forward and recompute it in backward."""
    return _FlashTriMulFn.apply(proj, gate_logits, mask)
