from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from tri_mul_ext import tri_mul_xbdnn_cublas


def _rel_error(actual: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (actual.float() - ref.float()).norm()
    base = ref.float().norm().clamp_min(1e-12)
    return float((diff / base).item())


def _tri_mul_bmm_reference(x_bdnn: torch.Tensor) -> torch.Tensor:
    a1, b1, a2, b2 = torch.chunk(x_bdnn, 4, dim=1)
    batch, d_chunk, n, _ = a1.shape
    x1 = torch.bmm(a1.reshape(batch * d_chunk, n, n), b1.reshape(batch * d_chunk, n, n).transpose(1, 2))
    x2 = torch.bmm(a2.reshape(batch * d_chunk, n, n).transpose(1, 2), b2.reshape(batch * d_chunk, n, n))
    return torch.cat(
        [x1.reshape(batch, d_chunk, n, n), x2.reshape(batch, d_chunk, n, n)],
        dim=1,
    ).permute(0, 2, 3, 1)


class ToyTriangularUpdate(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.pi = nn.Linear(dim, dim)
        self.gi = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim // 2)
        self.po = nn.Linear(dim // 2, dim)
        self.go = nn.Linear(dim // 2, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, backend: str) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.pi(x) * torch.sigmoid(self.gi(x)) * mask.unsqueeze(-1)
        x_bdnn = x.permute(0, 3, 1, 2).contiguous()
        if backend == "cublas_xbdnn":
            x = tri_mul_xbdnn_cublas(x_bdnn, out_dtype=x.dtype)
        elif backend == "bmm":
            x = _tri_mul_bmm_reference(x_bdnn)
        else:
            raise ValueError(f"unsupported backend {backend}")
        x = self.output_norm(x)
        return self.po(x) * torch.sigmoid(self.go(x))


def _run_backend(backend: str, seed: int, batch: int, seq_len: int, dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    module = ToyTriangularUpdate(dim).to("cuda", dtype=torch.bfloat16)
    x = torch.randn((batch, seq_len, seq_len, dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones((batch, seq_len, seq_len), device="cuda", dtype=torch.bfloat16)
    out = module(x, mask, backend)
    loss = out.float().square().mean()
    loss.backward()
    return {
        "out": out.detach(),
        "x_grad": x.grad.detach(),
        "pi_grad": module.pi.weight.grad.detach(),
        "gi_grad": module.gi.weight.grad.detach(),
        "po_grad": module.po.weight.grad.detach(),
        "go_grad": module.go.weight.grad.detach(),
    }


def _compare(actual: torch.Tensor, ref: torch.Tensor) -> dict[str, float | bool]:
    diff = (actual.float() - ref.float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "rel_l2_error": _rel_error(actual, ref),
        "has_nan_or_inf": bool((~torch.isfinite(actual)).any().item()),
        "all_zero": bool(actual.eq(0).all().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify cuBLAS xbdnn gradients against torch.bmm at 3B-aligned shapes.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ref = _run_backend("bmm", args.seed, args.batch, args.seq_len, args.dim)
    test = _run_backend("cublas_xbdnn", args.seed, args.batch, args.seq_len, args.dim)

    metrics = {
        "shape": [args.batch, args.seq_len, args.seq_len, args.dim],
        "out": _compare(test["out"], ref["out"]),
        "x.grad": _compare(test["x_grad"], ref["x_grad"]),
        "pi.weight.grad": _compare(test["pi_grad"], ref["pi_grad"]),
        "gi.weight.grad": _compare(test["gi_grad"], ref["gi_grad"]),
        "po.weight.grad": _compare(test["po_grad"], ref["po_grad"]),
        "go.weight.grad": _compare(test["go_grad"], ref["go_grad"]),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
