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

import os
import sys
import importlib
from contextlib import nullcontext
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions import split_quantize
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
from transformer_engine.pytorch.quantized_tensor import (
    QuantizedTensor,
    prepare_for_saving,
)
try:
    from transformer_engine.pytorch.quantized_tensor import restore_from_func_ctx
except ImportError:
    from transformer_engine.pytorch.quantized_tensor import restore_from_saved

    def restore_from_func_ctx(ctx):
        return restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
from transformer_engine.pytorch.tensor import MXFP8Quantizer, MXFP8TensorStorage
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

try:
    from tri_mul_ext import tri_mul_fused as _tri_mul_fused_ext
    from tri_mul_ext import tri_mul_xbdnn_cublas as _tri_mul_xbdnn_cublas_ext
except ImportError:
    _tri_mul_ext_root = Path(__file__).resolve().parent / "tri_mul_ext"
    if _tri_mul_ext_root.is_dir():
        sys.path.insert(0, str(_tri_mul_ext_root))
        try:
            sys.modules.pop("tri_mul_ext", None)
            _tri_mul_ext = importlib.import_module("tri_mul_ext")
            _tri_mul_fused_ext = _tri_mul_ext.tri_mul_fused
            _tri_mul_xbdnn_cublas_ext = _tri_mul_ext.tri_mul_xbdnn_cublas
        except ImportError:
            _tri_mul_fused_ext = None
            _tri_mul_xbdnn_cublas_ext = None
    else:
        _tri_mul_fused_ext = None
        _tri_mul_xbdnn_cublas_ext = None

try:
    from bmm_ext import mxfp8_bmm as _mxfp8_bmm_ext
    from bmm_ext import mxfp8_cublaslt_tri_mul_xbdnn as _mxfp8_cublaslt_tri_mul_xbdnn_ext
    from bmm_ext import mxfp8_cublaslt_tri_mul_xbdnn_inference as _mxfp8_cublaslt_tri_mul_xbdnn_inference_ext
    from bmm_ext import mxfp8_tri_mul_xbdnn as _mxfp8_tri_mul_xbdnn_ext
except ImportError:
    _bmm_ext_root = Path(__file__).resolve().parent / "fp8_bmm_ext"
    if _bmm_ext_root.is_dir():
        sys.path.insert(0, str(_bmm_ext_root))
        try:
            sys.modules.pop("bmm_ext", None)
            _bmm_ext = importlib.import_module("bmm_ext")
            _mxfp8_bmm_ext = _bmm_ext.mxfp8_bmm
            _mxfp8_cublaslt_tri_mul_xbdnn_ext = _bmm_ext.mxfp8_cublaslt_tri_mul_xbdnn
            _mxfp8_cublaslt_tri_mul_xbdnn_inference_ext = _bmm_ext.mxfp8_cublaslt_tri_mul_xbdnn_inference
            _mxfp8_tri_mul_xbdnn_ext = _bmm_ext.mxfp8_tri_mul_xbdnn
        except ImportError:
            _mxfp8_bmm_ext = None
            _mxfp8_cublaslt_tri_mul_xbdnn_ext = None
            _mxfp8_cublaslt_tri_mul_xbdnn_inference_ext = None
            _mxfp8_tri_mul_xbdnn_ext = None
    else:
        _mxfp8_bmm_ext = None
        _mxfp8_cublaslt_tri_mul_xbdnn_ext = None
        _mxfp8_cublaslt_tri_mul_xbdnn_inference_ext = None
        _mxfp8_tri_mul_xbdnn_ext = None


def te_linear_nd(
    module: te.Linear,
    x: torch.Tensor,
    *,
    fp8_output: bool = False,
    fp8_grad: bool = False,
) -> torch.Tensor:
    """Apply a te.Linear module to an N-dimensional tensor (N >= 2).

    te.Linear is validated for 2D (B, D) and 3D (B, S, D) inputs.
    For 4D+ inputs (e.g. pair representations with shape B, N, N, D),
    we flatten leading dimensions to 2D, apply the linear, and reshape back.

    Args:
        module: A transformer_engine.pytorch.Linear module.
        x: Input tensor of shape (*leading_dims, in_features).
        fp8_output: Forward quantized output request for TE FP8 autocast.
        fp8_grad: Backward quantized gradient request for TE FP8 autocast.

    Returns:
        Tensor of shape (*leading_dims, out_features).
    """
    if x.ndim <= 3:
        return module(x, fp8_output=fp8_output, fp8_grad=fp8_grad)
    leading = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    x = module(x, fp8_output=fp8_output, fp8_grad=fp8_grad)
    return x.reshape(*leading, -1)


def te_layernorm_nd(module: te.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """Apply a te.LayerNorm module to an N-dimensional tensor (N >= 2).

    Args:
        module: A transformer_engine.pytorch.LayerNorm module.
        x: Input tensor of shape (*leading_dims, normalized_shape).

    Returns:
        Tensor of same shape as input.
    """
    if x.ndim <= 3:
        return module(x)
    leading = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    x = module(x)
    return x.reshape(*leading, -1)


def _make_saved_activation_quantizer() -> Float8BlockQuantizer:
    """Create a TE block-scaled FP8 quantizer for saved activation packing."""
    return Float8BlockQuantizer(
        TE_DType[torch.float8_e4m3fn],
        rowwise=True,
        columnwise=False,
        block_scaling_dim=2,
    )


def quantize_tensor_for_fp8_save(tensor: torch.Tensor) -> torch.Tensor | QuantizedTensor:
    """Quantize a tensor to TE block-scaled FP8 when the shape supports it."""
    if isinstance(tensor, QuantizedTensor):
        return tensor
    if tensor.device.type != "cuda":
        return tensor
    quantizer = _make_saved_activation_quantizer()
    if tensor.dim() < 2 or not quantizer.is_quantizable(tensor):
        return tensor
    return quantizer.quantize(tensor.detach())


def save_fp8_tensors_for_backward(ctx, *tensors: torch.Tensor | QuantizedTensor) -> None:
    """Save tensors on an autograd context, preserving TE quantized storages."""
    saved_tensors, tensor_objects = prepare_for_saving(*tensors)
    ctx.tensor_objects = tensor_objects
    ctx.save_for_backward(*saved_tensors)


def restore_fp8_tensors_from_backward(ctx) -> list[torch.Tensor | QuantizedTensor | None]:
    """Restore tensors previously saved with `save_fp8_tensors_for_backward`."""
    return restore_from_func_ctx(ctx)


def saved_tensor_fp8_context(
    *,
    enabled: bool,
    include_bf16: bool = True,
    include_fp32: bool = False,
    include_leaf_tensors: bool = False,
):
    """Pack autograd-saved tensors into block-scaled FP8 and restore on backward.

    This is an experimental activation-memory path. It is intended for large
    non-leaf saved activations inside hotspot modules, not for model weights.
    """
    if not enabled or not torch.is_grad_enabled():
        return nullcontext()

    allowed_dtypes = set()
    if include_bf16:
        allowed_dtypes.add(torch.bfloat16)
    if include_fp32:
        allowed_dtypes.add(torch.float32)

    quantizer = _make_saved_activation_quantizer()

    def _pack(tensor):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if isinstance(tensor, QuantizedTensor):
            return tensor
        if tensor.device.type != "cuda" or tensor.dtype not in allowed_dtypes:
            return tensor
        if tensor.dim() < 2:
            return tensor
        if tensor.is_leaf and not include_leaf_tensors:
            return tensor
        if not quantizer.is_quantizable(tensor):
            return tensor

        packed = quantizer.quantize(tensor.detach())
        return ("te_fp8_saved_tensor", packed, tensor.dtype)

    def _unpack(packed):
        if isinstance(packed, tuple) and len(packed) == 3 and packed[0] == "te_fp8_saved_tensor":
            _, tensor, orig_dtype = packed
            return tensor.dequantize(dtype=orig_dtype)
        return packed

    return torch.autograd.graph.saved_tensors_hooks(_pack, _unpack)


def tri_mul_bmm(a: torch.Tensor, b: torch.Tensor, k_dim: int, mode: str = "off") -> torch.Tensor:
    """Batched GEMM equivalent of triangular multiplication einsum.

    Replaces:
      k_dim=2: torch.einsum("bikd,bjkd->bijd", a, b)
      k_dim=1: torch.einsum("bkid,bkjd->bijd", a, b)

    Args:
        a: Tensor of shape (B, N, N, D).
        b: Tensor of shape (B, N, N, D).
        k_dim: Spatial dimension to contract over (1 or 2).
        mode: Precision mode.
            "off": FP32 bmm (caller upcasts via .float(), default).
            "bf16": BF16 bmm (skip .float() upcast).

    Returns:
        Tensor of shape (B, N, N, D).
    """
    B, N1, N2, D = a.shape
    # Move D to dim 1: (B, D, N, N), then merge B*D for batched mm
    a = a.permute(0, 3, 1, 2).contiguous().reshape(B * D, N1, N2)
    b = b.permute(0, 3, 1, 2).contiguous().reshape(B * D, N1, N2)

    if k_dim == 2:
        # "bikd,bjkd->bijd": a is (batch, i, k), b is (batch, j, k)
        # result = a @ b^T = (batch, i, j)
        out = torch.bmm(a, b.transpose(1, 2))
    elif k_dim == 1:
        # "bkid,bkjd->bijd": a is (batch, k, i), b is (batch, k, j)
        # result = a^T @ b = (batch, i, j)
        out = torch.bmm(a.transpose(1, 2), b)
    else:
        raise ValueError(f"k_dim must be 1 or 2, got {k_dim}")

    # Reshape back: (B*D, N, N) -> (B, D, N, N) -> (B, N, N, D)
    return out.reshape(B, D, N1, N2).permute(0, 2, 3, 1)


def tri_mul_einsum(a: torch.Tensor, b: torch.Tensor, k_dim: int) -> torch.Tensor:
    """Literal einsum implementation of triangular multiplication."""
    if k_dim == 2:
        return torch.einsum("bikd,bjkd->bijd", a, b)
    if k_dim == 1:
        return torch.einsum("bkid,bkjd->bijd", a, b)
    raise ValueError(f"k_dim must be 1 or 2, got {k_dim}")


def tri_mul(
    a: torch.Tensor,
    b: torch.Tensor,
    k_dim: int,
    mode: str = "off",
    impl: str | None = None,
) -> torch.Tensor:
    """Triangular multiplication dispatch.

    Supported implementations:
    - `einsum`: literal `torch.einsum` contraction
    - `bmm`: reshape-to-batched-GEMM implementation
    - `fp8_bmm`: MXFP8 batched GEMM with custom autograd
    - `fp8_cublaslt`: MXFP8 batched GEMM via raw cuBLASLt
    - `fused`: experimental external `tri_mul_ext` BF16 kernel
    """
    impl = impl or os.environ.get("BIONEMO_TRI_MUL_IMPL", "bmm")
    if impl == "einsum":
        return tri_mul_einsum(a, b, k_dim=k_dim)
    if impl == "fp8_bmm":
        return tri_mul_fp8_bmm(a, b, k_dim=k_dim, out_dtype=a.dtype)
    if impl == "fp8_cublaslt":
        return tri_mul_fp8_bmm(a, b, k_dim=k_dim, out_dtype=a.dtype)
    if impl == "fused":
        if _tri_mul_fused_ext is None:
            raise RuntimeError("BIONEMO_TRI_MUL_IMPL=fused was requested but tri_mul_ext is not importable")
        return _tri_mul_fused_ext(a, b, k_dim=k_dim, out_dtype=a.dtype)
    if impl != "bmm":
        raise ValueError(f"Unsupported triangular multiplication implementation: {impl}")
    return tri_mul_bmm(a, b, k_dim=k_dim, mode=mode)


def tri_mul_fp8_bmm(a: torch.Tensor, b: torch.Tensor, k_dim: int, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Tri-mul via autograd-backed MXFP8 batched GEMM."""
    if _mxfp8_bmm_ext is None:
        raise RuntimeError("BIONEMO_TRI_MUL_IMPL=fp8_bmm was requested but bmm_ext is not importable")
    if a.dim() != 4 or b.dim() != 4:
        raise ValueError("a and b must have shape (B, N, N, D)")
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")

    B, N1, N2, D = a.shape
    out_dtype = a.dtype if out_dtype is None else out_dtype
    a_3d = a.permute(0, 3, 1, 2).contiguous().reshape(B * D, N1, N2)
    b_3d = b.permute(0, 3, 1, 2).contiguous().reshape(B * D, N1, N2)

    if k_dim == 2:
        lhs = a_3d
        rhs = b_3d.transpose(1, 2).contiguous()
    elif k_dim == 1:
        lhs = a_3d.transpose(1, 2).contiguous()
        rhs = b_3d
    else:
        raise ValueError(f"k_dim must be 1 or 2, got {k_dim}")

    out = _mxfp8_bmm_ext(lhs, rhs, out_dtype=out_dtype)
    return out.reshape(B, D, N1, N1).permute(0, 2, 3, 1)


def tri_mul_xbdnn(x_bdnn: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Tri-mul from a single pre-transposed `(B, 128, N, N)` tensor.

    This path is specialized to the production MiniFold shape:
    four 32-channel chunks laid out contiguously in the transposed tensor.
    """
    out_dtype = x_bdnn.dtype if out_dtype is None else out_dtype
    if _tri_mul_xbdnn_cublas_ext is None:
        raise RuntimeError("BIONEMO_TRI_MUL_IMPL=cublas_xbdnn was requested but tri_mul_ext is not importable")
    return _tri_mul_xbdnn_cublas_ext(x_bdnn, out_dtype=out_dtype)


def tri_mul_fp8_bdnn(x_bdnn: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Tri-mul from a pre-transposed `(B, 128, N, N)` tensor using packed MXFP8 state."""
    if _mxfp8_tri_mul_xbdnn_ext is None:
        raise RuntimeError("BIONEMO_TRI_MUL_IMPL=fp8_bmm was requested but bmm_ext is not importable")
    if x_bdnn.dim() != 4:
        raise ValueError("x_bdnn must have shape (B, D, N, N)")
    if x_bdnn.shape[1] % 4 != 0:
        raise ValueError(f"x_bdnn channel dimension must be divisible by 4, got {x_bdnn.shape[1]}")

    B, D, N1, N2 = x_bdnn.shape
    if N1 != N2:
        raise ValueError(f"fp8_bmm expects square sequence dimensions, got {(N1, N2)}")
    if N1 % 32 != 0:
        raise ValueError(f"fp8_bmm requires sequence length divisible by 32, got {N1}")

    out_dtype = x_bdnn.dtype if out_dtype is None else out_dtype
    x = _mxfp8_tri_mul_xbdnn_ext(x_bdnn, out_dtype=out_dtype)
    d_out = x.shape[1]
    return x.reshape(B, d_out, N1, N1).permute(0, 2, 3, 1)


def tri_mul_fp8_cublaslt_bdnn(x_bdnn: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Tri-mul from a pre-transposed `(B, 128, N, N)` tensor using raw cuBLASLt MXFP8 batched GEMMs."""
    if _mxfp8_cublaslt_tri_mul_xbdnn_ext is None:
        raise RuntimeError("BIONEMO_TRI_MUL_IMPL=fp8_cublaslt was requested but bmm_ext is not importable")
    if x_bdnn.dim() != 4:
        raise ValueError("x_bdnn must have shape (B, D, N, N)")
    if x_bdnn.shape[1] % 4 != 0:
        raise ValueError(f"x_bdnn channel dimension must be divisible by 4, got {x_bdnn.shape[1]}")

    B, D, N1, N2 = x_bdnn.shape
    if N1 != N2:
        raise ValueError(f"fp8_cublaslt expects square sequence dimensions, got {(N1, N2)}")
    if N1 % 32 != 0:
        raise ValueError(f"fp8_cublaslt requires sequence length divisible by 32, got {N1}")

    out_dtype = x_bdnn.dtype if out_dtype is None else out_dtype
    if not torch.is_grad_enabled() and _mxfp8_cublaslt_tri_mul_xbdnn_inference_ext is not None:
        x = _mxfp8_cublaslt_tri_mul_xbdnn_inference_ext(x_bdnn, out_dtype=out_dtype)
    else:
        x = _mxfp8_cublaslt_tri_mul_xbdnn_ext(x_bdnn, out_dtype=out_dtype)
    d_out = x.shape[1]
    return x.reshape(B, d_out, N1, N1).permute(0, 2, 3, 1)


def tri_mul_fp8_fused_bdnn(x_bdnn: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Fused FP8 tri-mul over a packed `(B, 128, N, N)` tensor.

    This exposes the paired native MXFP8 cuBLASLt implementation as a recipe-level
    fused backend. It keeps the two triangular contractions inside one FP8 backend
    path rather than issuing them through the older BF16 `tri_mul_ext` API.
    """
    return tri_mul_fp8_cublaslt_bdnn(x_bdnn, out_dtype=out_dtype)


def tri_mul_fp8_grouped_bdnn(x_bdnn: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Tri-mul from a pre-transposed `(B, 128, N, N)` tensor using TE grouped FP8 GEMMs."""
    if x_bdnn.dim() != 4:
        raise ValueError("x_bdnn must have shape (B, D, N, N)")
    if x_bdnn.shape[1] % 4 != 0:
        raise ValueError(f"x_bdnn channel dimension must be divisible by 4, got {x_bdnn.shape[1]}")

    B, D, N1, N2 = x_bdnn.shape
    if N1 != N2:
        raise ValueError(f"fp8_bmm expects square sequence dimensions, got {(N1, N2)}")
    if N1 % 32 != 0:
        raise ValueError(f"fp8_bmm requires sequence length divisible by 32, got {N1}")

    out_dtype = x_bdnn.dtype if out_dtype is None else out_dtype
    x = _GroupedFloat8TriMulXBDNN.apply(x_bdnn, out_dtype)
    d_out = x.shape[1]
    return x.reshape(B, d_out, N1, N1).permute(0, 2, 3, 1)


def _make_mxfp8_quantizer() -> MXFP8Quantizer:
    q = MXFP8Quantizer(TE_DType[torch.float8_e4m3fn], rowwise=True, columnwise=True)
    q.optimize_for_gemm = True
    return q


def _slice_mxfp8_storage(t: MXFP8TensorStorage, batch: int, n: int) -> list[MXFP8TensorStorage]:
    """Split a batched MXFP8 storage tensor into per-GEMM storage views."""
    row_data = t._rowwise_data
    row_scale = t._rowwise_scale_inv
    col_data = t._columnwise_data
    col_scale = t._columnwise_scale_inv
    fp8_dtype = t._fp8_dtype
    swizzled = t._with_gemm_swizzled_scales
    k_blocks = n // 32
    return [
        MXFP8TensorStorage(
            row_data[i : i + 1],
            row_scale[i * n : (i + 1) * n],
            col_data[i : i + 1],
            col_scale[i * k_blocks : (i + 1) * k_blocks],
            fp8_dtype,
            None,
            swizzled,
        )
        for i in range(batch)
    ]


def _pack_mxfp8_storages(chunks: list[torch.Tensor]) -> list[MXFP8TensorStorage]:
    if not chunks:
        return []
    split_sections = [chunk.shape[0] for chunk in chunks]
    quantizers = [_make_mxfp8_quantizer() for _ in chunks]
    return list(split_quantize(torch.cat(chunks, dim=0), split_sections, quantizers))


def _grouped_mxfp8_mm(
    left: list[MXFP8TensorStorage],
    right: list[MXFP8TensorStorage],
    *,
    op: str,
    n: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Run grouped MXFP8 GEMMs over per-slice storages.

    TE grouped GEMM uses:
    - `layout="NN"` on `(X, Y)` -> `Y @ X`
    - `layout="TN"` on `(X, Y)` -> `Y @ X.T`
    - `layout="NT"` on `(X, Y)` -> `Y.T @ X`
    """
    if len(left) != len(right):
        raise ValueError(f"grouped mxfp8 mm requires matching group counts, got {len(left)} and {len(right)}")
    groups = len(left)
    out = [torch.empty(n, n, device=left[0]._rowwise_data.device, dtype=out_dtype) for _ in range(groups)]
    quantization_params = [None] * groups
    m_splits = [n] * groups

    if op == "ab":
        xs, ys, layout = right, left, "NN"
    elif op == "abt":
        xs, ys, layout = right, left, "TN"
    elif op == "atb":
        xs, ys, layout = right, left, "NT"
    else:
        raise ValueError(f"Unsupported grouped mxfp8 op: {op}")

    general_grouped_gemm(
        xs,
        ys,
        out=out,
        quantization_params=quantization_params,
        out_dtype=out_dtype,
        layout=layout,
        m_splits=m_splits,
        single_output=False,
    )
    return torch.stack(out, dim=0)


class _GroupedFloat8TriMulXBDNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bdnn: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        B, DT, N, _ = x_bdnn.shape
        D = DT // 4
        a1, b1, a2, b2 = torch.chunk(x_bdnn, 4, dim=1)
        a1_3d = a1.reshape(B * D, N, N)
        b1_3d = b1.reshape(B * D, N, N)
        a2_3d = a2.reshape(B * D, N, N)
        b2_3d = b2.reshape(B * D, N, N)
        packed_a1, packed_b1, packed_a2, packed_b2 = _pack_mxfp8_storages([a1_3d, b1_3d, a2_3d, b2_3d])
        packed_a1_slices = _slice_mxfp8_storage(packed_a1, B * D, N)
        packed_b1_slices = _slice_mxfp8_storage(packed_b1, B * D, N)
        packed_a2_slices = _slice_mxfp8_storage(packed_a2, B * D, N)
        packed_b2_slices = _slice_mxfp8_storage(packed_b2, B * D, N)

        out1 = _grouped_mxfp8_mm(packed_a1_slices, packed_b1_slices, op="abt", n=N, out_dtype=out_dtype)
        out2 = _grouped_mxfp8_mm(packed_a2_slices, packed_b2_slices, op="atb", n=N, out_dtype=out_dtype)

        ctx.save_for_backward(
            packed_a1._rowwise_data,
            packed_a1._rowwise_scale_inv,
            packed_a1._columnwise_data,
            packed_a1._columnwise_scale_inv,
            packed_b1._rowwise_data,
            packed_b1._rowwise_scale_inv,
            packed_b1._columnwise_data,
            packed_b1._columnwise_scale_inv,
            packed_a2._rowwise_data,
            packed_a2._rowwise_scale_inv,
            packed_a2._columnwise_data,
            packed_a2._columnwise_scale_inv,
            packed_b2._rowwise_data,
            packed_b2._rowwise_scale_inv,
            packed_b2._columnwise_data,
            packed_b2._columnwise_scale_inv,
        )
        ctx.shape = (B, D, N)
        ctx.input_dtype = x_bdnn.dtype
        return torch.cat([out1.reshape(B, D, N, N), out2.reshape(B, D, N, N)], dim=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            a1_row,
            a1_row_scale,
            a1_col,
            a1_col_scale,
            b1_row,
            b1_row_scale,
            b1_col,
            b1_col_scale,
            a2_row,
            a2_row_scale,
            a2_col,
            a2_col_scale,
            b2_row,
            b2_row_scale,
            b2_col,
            b2_col_scale,
        ) = ctx.saved_tensors
        B, D, N = ctx.shape
        grad_output = grad_output.reshape(B, 2 * D, N, N)
        g1, g2 = torch.chunk(grad_output, 2, dim=1)
        g1_3d = g1.reshape(B * D, N, N)
        g2_3d = g2.reshape(B * D, N, N)
        packed_g1, packed_g2 = _pack_mxfp8_storages([g1_3d, g2_3d])

        packed_a1 = MXFP8TensorStorage(a1_row, a1_row_scale, a1_col, a1_col_scale, TE_DType[torch.float8_e4m3fn], None, True)
        packed_b1 = MXFP8TensorStorage(b1_row, b1_row_scale, b1_col, b1_col_scale, TE_DType[torch.float8_e4m3fn], None, True)
        packed_a2 = MXFP8TensorStorage(a2_row, a2_row_scale, a2_col, a2_col_scale, TE_DType[torch.float8_e4m3fn], None, True)
        packed_b2 = MXFP8TensorStorage(b2_row, b2_row_scale, b2_col, b2_col_scale, TE_DType[torch.float8_e4m3fn], None, True)

        packed_g1_slices = _slice_mxfp8_storage(packed_g1, B * D, N)
        packed_g2_slices = _slice_mxfp8_storage(packed_g2, B * D, N)
        packed_a1_slices = _slice_mxfp8_storage(packed_a1, B * D, N)
        packed_b1_slices = _slice_mxfp8_storage(packed_b1, B * D, N)
        packed_a2_slices = _slice_mxfp8_storage(packed_a2, B * D, N)
        packed_b2_slices = _slice_mxfp8_storage(packed_b2, B * D, N)

        grad_a1 = _grouped_mxfp8_mm(packed_g1_slices, packed_b1_slices, op="ab", n=N, out_dtype=torch.float32)
        grad_b1 = _grouped_mxfp8_mm(packed_g1_slices, packed_a1_slices, op="atb", n=N, out_dtype=torch.float32)
        grad_a2 = _grouped_mxfp8_mm(packed_b2_slices, packed_g2_slices, op="abt", n=N, out_dtype=torch.float32)
        grad_b2 = _grouped_mxfp8_mm(packed_a2_slices, packed_g2_slices, op="ab", n=N, out_dtype=torch.float32)

        grad_x = torch.cat(
            [
                grad_a1.reshape(B, D, N, N),
                grad_b1.reshape(B, D, N, N),
                grad_a2.reshape(B, D, N, N),
                grad_b2.reshape(B, D, N, N),
            ],
            dim=1,
        ).to(ctx.input_dtype)
        return grad_x, None


def tri_mul_bmm_bdnn(a: torch.Tensor, b: torch.Tensor, k_dim: int) -> torch.Tensor:
    """Tri-mul from pre-transposed `(B, D, N, N)` inputs.

    This keeps the current `torch.bmm` backend but lets callers permute once for
    both triangular contractions instead of permuting each chunk separately.
    """
    if a.dim() != 4 or b.dim() != 4:
        raise ValueError("a and b must have shape (B, D, N, N)")
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")

    B, D, N1, N2 = a.shape
    a = a.reshape(B * D, N1, N2)
    b = b.reshape(B * D, N1, N2)

    if k_dim == 2:
        out = torch.bmm(a, b.transpose(1, 2))
    elif k_dim == 1:
        out = torch.bmm(a.transpose(1, 2), b)
    else:
        raise ValueError(f"k_dim must be 1 or 2, got {k_dim}")

    return out.reshape(B, D, N1, N2).permute(0, 2, 3, 1)


def te_layernorm_linear_nd(module: te.LayerNormLinear, x: torch.Tensor) -> torch.Tensor:
    """Apply a te.LayerNormLinear module to an N-dimensional tensor (N >= 2).

    Args:
        module: A transformer_engine.pytorch.LayerNormLinear module.
        x: Input tensor of shape (*leading_dims, in_features).

    Returns:
        Tensor of shape (*leading_dims, out_features).
    """
    if x.ndim <= 3:
        return module(x)
    leading = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    x = module(x)
    return x.reshape(*leading, -1)
