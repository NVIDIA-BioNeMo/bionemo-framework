from __future__ import annotations

from typing import Iterable

import torch
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

from .ops import PackedNVFP4Tensor


_FP4_E2M1_CODEBOOK = torch.tensor(
    [-0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _require_cuda_3d(x: torch.Tensor, name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dim() != 3:
        raise ValueError(f"{name} must be 3D")


def _quantize_fp4_codes(x: torch.Tensor) -> torch.Tensor:
    codebook = _FP4_E2M1_CODEBOOK.to(device=x.device)
    distances = (x.unsqueeze(-1).float() - codebook).abs()
    return distances.argmin(dim=-1).to(torch.uint8)


def pack_nvfp4(x: torch.Tensor, *, role: str = "lhs") -> PackedNVFP4Tensor:
    _require_cuda_3d(x, "x")
    if x.shape[-1] % 16 != 0 or (x.shape[0] * x.shape[1]) % 16 != 0:
        raise ValueError("nvfp4 helper packing requires last dim divisible by 16 and flattened outer dims divisible by 16")
    role = role.lower()
    if role not in {"lhs", "rhs"}:
        raise ValueError("role must be 'lhs' or 'rhs'")

    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
    )
    packed_slices = []
    scale_slices = []
    amax_slices = []
    for batch_idx in range(x.shape[0]):
        src = x[batch_idx].contiguous()
        if role == "rhs":
            src = src.transpose(0, 1).contiguous()
        q = quantizer.quantize(src)
        packed_slices.append(q._rowwise_data)
        scale_slices.append(q._rowwise_scale_inv)
        amax_slices.append(q._amax_rowwise)
    packed = torch.stack(packed_slices, dim=0).contiguous()
    scale_inv = torch.stack(scale_slices, dim=0).contiguous()
    amax = torch.stack(amax_slices, dim=0).contiguous()
    return PackedNVFP4Tensor(
        data=packed,
        logical_shape=tuple(int(v) for v in x.shape),
        scale_inv=scale_inv,
        amax=amax,
        rhs_transposed=(role == "rhs"),
    )


def unpack_nvfp4(packed: PackedNVFP4Tensor) -> torch.Tensor:
    if not isinstance(packed, PackedNVFP4Tensor):
        raise TypeError("packed must be a PackedNVFP4Tensor")
    _require_cuda_3d(packed.data, "packed.data")

    lut = _FP4_E2M1_CODEBOOK.to(device=packed.data.device)
    data = packed.data
    lo = lut[(data & 0x0F).long()]
    hi = lut[((data >> 4) & 0x0F).long()]
    if packed.rhs_transposed:
        transposed_shape = (packed.logical_shape[0], packed.logical_shape[2], packed.logical_shape[1])
        out = torch.empty(transposed_shape, device=data.device, dtype=torch.float32)
    else:
        out = torch.empty(packed.logical_shape, device=data.device, dtype=torch.float32)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    if packed.scale_inv is None:
        return out

    if packed.amax is None:
        tensor_scale = 1.0
    else:
        tensor_scale = packed.amax.float().view(-1, 1, 1) / (6.0 * 448.0)
    rows = packed.logical_shape[2] if packed.rhs_transposed else packed.logical_shape[1]
    blocks = (packed.logical_shape[1] if packed.rhs_transposed else packed.logical_shape[2]) // 16
    scales_fp8 = packed.scale_inv.view(torch.float8_e4m3fn)
    scales = scales_fp8[:, :rows, :blocks].float() * tensor_scale
    out = out * scales.repeat_interleave(16, dim=2)
    if packed.rhs_transposed:
        return out.transpose(1, 2).contiguous()
    return out
