from __future__ import annotations

from typing import Optional

import torch

try:
    from . import _C
except ImportError:
    _C = None

try:
    from .triton_kernel import tri_mul_fused_triton
except ImportError:
    tri_mul_fused_triton = None


_SUPPORTED_OUT_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_SUPPORTED_OUT_DTYPE_NAMES = {value: key for key, value in _SUPPORTED_OUT_DTYPES.items()}


def _extension_unavailable(name: str) -> RuntimeError:
    return RuntimeError(f"tri_mul_ext._C is not available for {name}")


def _normalize_out_dtype(out_dtype: Optional[torch.dtype], default: torch.dtype) -> torch.dtype:
    resolved = default if out_dtype is None else out_dtype
    if resolved not in _SUPPORTED_OUT_DTYPES:
        raise TypeError(f"unsupported out_dtype {resolved}")
    return resolved


def _normalize_out_dtype_name(out_dtype: Optional[torch.dtype], default: torch.dtype) -> str:
    return _SUPPORTED_OUT_DTYPES[_normalize_out_dtype(out_dtype, default)]


def _dtype_from_name(out_dtype: str) -> torch.dtype:
    try:
        return _SUPPORTED_OUT_DTYPE_NAMES[out_dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported out_dtype {out_dtype!r}") from exc


def extension_available() -> bool:
    return tri_mul_fused_triton is not None or _C is not None


def _tri_mul_reference(a: torch.Tensor, b: torch.Tensor, k_dim: int) -> torch.Tensor:
    if a.dim() != 4 or b.dim() != 4:
        raise ValueError("a and b must be rank-4 tensors with shape (B, N, N, D)")
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")
    if k_dim not in {1, 2}:
        raise ValueError(f"k_dim must be 1 or 2, got {k_dim}")

    batch, n1, n2, channels = a.shape
    a3 = a.permute(0, 3, 1, 2).contiguous().reshape(batch * channels, n1, n2)
    b3 = b.permute(0, 3, 1, 2).contiguous().reshape(batch * channels, n1, n2)
    if k_dim == 2:
        out3 = torch.bmm(a3, b3.transpose(1, 2))
    else:
        out3 = torch.bmm(a3.transpose(1, 2), b3)
    return out3.reshape(batch, channels, n1, n2).permute(0, 2, 3, 1)


def _tri_mul_xbdnn_reference(x_bdnn: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    if x_bdnn.dim() != 4:
        raise ValueError("x_bdnn must have shape (B, D, N, N)")
    if x_bdnn.shape[1] != 128:
        raise ValueError(f"x_bdnn must have channel dimension 128, got {x_bdnn.shape[1]}")

    a1, b1, a2, b2 = torch.chunk(x_bdnn, 4, dim=1)
    batch, d_chunk, n, _ = a1.shape
    x1 = torch.bmm(a1.reshape(batch * d_chunk, n, n), b1.reshape(batch * d_chunk, n, n).transpose(1, 2))
    x2 = torch.bmm(a2.reshape(batch * d_chunk, n, n).transpose(1, 2), b2.reshape(batch * d_chunk, n, n))
    out = torch.cat(
        [x1.reshape(batch, d_chunk, n, n), x2.reshape(batch, d_chunk, n, n)],
        dim=1,
    )
    return out.permute(0, 2, 3, 1).to(out_dtype)


@torch.library.custom_op("tri_mul_ext::tri_mul_fused", mutates_args=(), device_types="cuda")
def _tri_mul_fused_op(a: torch.Tensor, b: torch.Tensor, k_dim: int, out_dtype: str) -> torch.Tensor:
    if _C is None:
        raise _extension_unavailable("tri_mul_fused")
    return _C.tri_mul_fused(a, b, int(k_dim), out_dtype)


@_tri_mul_fused_op.register_fake
def _tri_mul_fused_fake(a: torch.Tensor, b: torch.Tensor, k_dim: int, out_dtype: str) -> torch.Tensor:
    return a.new_empty((a.shape[0], a.shape[1], b.shape[1], a.shape[3]), dtype=_dtype_from_name(out_dtype))


def _tri_mul_fused_setup_context(ctx, inputs, output) -> None:
    a, b, k_dim, _ = inputs
    ctx.k_dim = int(k_dim)
    ctx.save_for_backward(a, b)


def _tri_mul_fused_backward(ctx, grad_output: torch.Tensor):
    if _C is None:
        raise _extension_unavailable("tri_mul_fused_backward")
    a, b = ctx.saved_tensors
    grad_output = grad_output.contiguous()
    grad_a, grad_b = _C.tri_mul_fused_backward(grad_output, a, b, int(ctx.k_dim))
    return grad_a, grad_b, None, None


torch.library.register_autograd(_tri_mul_fused_op, _tri_mul_fused_backward, setup_context=_tri_mul_fused_setup_context)


@torch.library.custom_op("tri_mul_ext::tri_mul_bdnn_cublas", mutates_args=(), device_types="cuda")
def _tri_mul_bdnn_cublas_op(a: torch.Tensor, b: torch.Tensor, k_dim: int, out_dtype: str) -> torch.Tensor:
    if _C is None:
        raise _extension_unavailable("tri_mul_bdnn_cublas")
    return _C.tri_mul_bdnn_cublas(a, b, int(k_dim), out_dtype)


@_tri_mul_bdnn_cublas_op.register_fake
def _tri_mul_bdnn_cublas_fake(a: torch.Tensor, b: torch.Tensor, k_dim: int, out_dtype: str) -> torch.Tensor:
    return a.new_empty((a.shape[0], a.shape[2], b.shape[2], a.shape[1]), dtype=_dtype_from_name(out_dtype))


@torch.library.custom_op("tri_mul_ext::tri_mul_xbdnn_cublas", mutates_args=(), device_types="cuda")
def _tri_mul_xbdnn_cublas_op(x_bdnn: torch.Tensor, out_dtype: str) -> torch.Tensor:
    if _C is None:
        raise _extension_unavailable("tri_mul_xbdnn_cublas")
    return _C.tri_mul_xbdnn_cublas(x_bdnn, out_dtype)


@_tri_mul_xbdnn_cublas_op.register_fake
def _tri_mul_xbdnn_cublas_fake(x_bdnn: torch.Tensor, out_dtype: str) -> torch.Tensor:
    batch = int(x_bdnn.shape[0])
    n_i = int(x_bdnn.shape[2])
    n_j = int(x_bdnn.shape[3])
    channels = int(x_bdnn.shape[1] // 2)
    return torch.empty_strided(
        (batch, n_i, n_j, channels),
        (n_i * n_j * channels, n_j, 1, n_i * n_j),
        dtype=_dtype_from_name(out_dtype),
        device=x_bdnn.device,
    )


def _tri_mul_xbdnn_cublas_setup_context(ctx, inputs, output) -> None:
    (x_bdnn, _) = inputs
    ctx.save_for_backward(x_bdnn)
    ctx.input_dtype = x_bdnn.dtype


def _tri_mul_xbdnn_cublas_backward(ctx, grad_output: torch.Tensor):
    if _C is None:
        raise _extension_unavailable("tri_mul_xbdnn_cublas_backward")
    (x_bdnn,) = ctx.saved_tensors
    if grad_output.dtype != ctx.input_dtype:
        grad_output = grad_output.to(ctx.input_dtype)
    grad_output = grad_output.contiguous()

    a1, b1, a2, b2 = [t.permute(0, 2, 3, 1).contiguous() for t in torch.chunk(x_bdnn, 4, dim=1)]
    g1, g2 = [t.contiguous() for t in torch.chunk(grad_output, 2, dim=-1)]
    grad_a1, grad_b1 = _C.tri_mul_fused_backward(g1, a1, b1, 2)
    grad_a2, grad_b2 = _C.tri_mul_fused_backward(g2, a2, b2, 1)
    grad_x = torch.cat(
        [
            grad_a1.permute(0, 3, 1, 2).contiguous(),
            grad_b1.permute(0, 3, 1, 2).contiguous(),
            grad_a2.permute(0, 3, 1, 2).contiguous(),
            grad_b2.permute(0, 3, 1, 2).contiguous(),
        ],
        dim=1,
    )
    return grad_x.to(ctx.input_dtype), None


torch.library.register_autograd(
    _tri_mul_xbdnn_cublas_op,
    _tri_mul_xbdnn_cublas_backward,
    setup_context=_tri_mul_xbdnn_cublas_setup_context,
)


def tri_mul_fused(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    k_dim: int,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Fused triangular multiplication API."""
    resolved_out_dtype = _normalize_out_dtype(out_dtype, a.dtype)
    if (
        tri_mul_fused_triton is not None
        and a.is_cuda
        and a.dtype == torch.bfloat16
        and a.shape[-1] == 32
        and not torch.is_grad_enabled()
    ):
        return tri_mul_fused_triton(a, b, k_dim=k_dim, out_dtype=resolved_out_dtype)
    if _C is None or a.shape[-1] != 32:
        return _tri_mul_reference(a, b, k_dim).to(resolved_out_dtype)
    return _tri_mul_fused_op(a, b, int(k_dim), _SUPPORTED_OUT_DTYPES[resolved_out_dtype])


def tri_mul_bdnn_cublas(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    k_dim: int,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    resolved_out_dtype = _normalize_out_dtype_name(out_dtype, a.dtype)
    if _C is None:
        raise _extension_unavailable("tri_mul_bdnn_cublas")
    return _tri_mul_bdnn_cublas_op(a, b, int(k_dim), resolved_out_dtype)


def tri_mul_xbdnn_cublas(
    x_bdnn: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    resolved_out_dtype = _normalize_out_dtype(out_dtype, x_bdnn.dtype)
    if _C is None:
        raise _extension_unavailable("tri_mul_xbdnn_cublas")
    if x_bdnn.shape[1] != 128:
        return _tri_mul_xbdnn_reference(x_bdnn, resolved_out_dtype)
    return _tri_mul_xbdnn_cublas_op(x_bdnn, _SUPPORTED_OUT_DTYPES[resolved_out_dtype])
