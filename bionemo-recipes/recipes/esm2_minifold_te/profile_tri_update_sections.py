import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from miniformer_te import TriangularUpdateTE
from quantization import ComponentPrecisionConfig
from te_utils import (
    te_layernorm_nd,
    te_linear_nd,
    tri_mul_bmm_bdnn,
    tri_mul_fp8_bdnn,
    tri_mul_fp8_cublaslt_bdnn,
    tri_mul_fp8_grouped_bdnn,
    tri_mul_xbdnn,
)


def _cuda_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _section_forward(mod: TriangularUpdateTE, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    cp = mod._component_precision

    def _proj_ctx():
        return cp.get_context("tri_proj") if cp else nullcontext()

    def _gate_ctx():
        return cp.get_context("tri_gate") if cp else nullcontext()

    timings: dict[str, float] = {}
    keep_fp8_linear_outputs = FP8GlobalStateManager.is_fp8_enabled()
    values: dict[str, torch.Tensor] = {}

    def _input_gate():
        x_norm = te_layernorm_nd(mod.input_norm, x)
        with _proj_ctx():
            pi_out = te_linear_nd(mod.pi, x_norm, fp8_output=keep_fp8_linear_outputs)
        with _gate_ctx():
            gi_logits = te_linear_nd(mod.gi, x_norm, fp8_output=keep_fp8_linear_outputs)
            gi_out = gi_logits.sigmoid()
        values["gated"] = (pi_out * gi_out) * mask.unsqueeze(-1)

    timings["input_gate_ms"] = _cuda_ms(_input_gate)

    def _tri_backend():
        tri_mode = cp.tri_einsum if cp else "off"
        use_fp32 = tri_mode == "off"
        x_in = values["gated"].float() if use_fp32 else values["gated"]
        tri_impl = cp.tri_impl if cp else "bmm"
        x_bdnn = x_in.permute(0, 3, 1, 2).contiguous()
        if tri_impl == "cublas_xbdnn":
            x_tri = tri_mul_xbdnn(x_bdnn, out_dtype=x_in.dtype)
        elif tri_impl == "fp8_bmm":
            x_tri = tri_mul_fp8_bdnn(x_bdnn, out_dtype=x_in.dtype)
        elif tri_impl == "fp8_cublaslt":
            x_tri = tri_mul_fp8_cublaslt_bdnn(x_bdnn, out_dtype=x_in.dtype)
        elif tri_impl == "fp8_grouped":
            x_tri = tri_mul_fp8_grouped_bdnn(x_bdnn, out_dtype=x_in.dtype)
        elif tri_impl == "bmm":
            a1, b1, a2, b2 = torch.chunk(x_bdnn, 4, dim=1)
            x1 = tri_mul_bmm_bdnn(a1, b1, k_dim=2)
            x2 = tri_mul_bmm_bdnn(a2, b2, k_dim=1)
            x_tri = torch.cat([x1, x2], dim=-1)
        else:
            raise ValueError(f"unsupported backend {tri_impl}")
        if use_fp32:
            x_tri = x_tri.to(mask.dtype if mask.is_floating_point() else torch.float32)
        values["tri"] = x_tri

    timings["tri_backend_ms"] = _cuda_ms(_tri_backend)

    def _output_gate():
        x_norm = te_layernorm_nd(mod.output_norm, values["tri"])
        with _proj_ctx():
            po_out = te_linear_nd(mod.po, x_norm, fp8_output=keep_fp8_linear_outputs)
        with _gate_ctx():
            go_logits = te_linear_nd(mod.go, x_norm, fp8_output=keep_fp8_linear_outputs)
            go_out = go_logits.sigmoid()
        values["out"] = po_out * go_out

    timings["output_gate_ms"] = _cuda_ms(_output_gate)
    timings["forward_total_ms"] = timings["input_gate_ms"] + timings["tri_backend_ms"] + timings["output_gate_ms"]
    return values["out"], timings


def main():
    parser = argparse.ArgumentParser(description="Profile TriangularUpdateTE section timings for a chosen backend.")
    parser.add_argument("--backend", default="fp8_cublaslt", choices=["bmm", "cublas_xbdnn", "fp8_bmm", "fp8_cublaslt", "fp8_grouped"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.seq_len % 32 != 0:
        raise SystemExit("--seq-len must be divisible by 32")

    torch.cuda.set_device(0)
    cp = ComponentPrecisionConfig(tri_einsum="bf16", tri_impl=args.backend)
    mod = TriangularUpdateTE(dim=args.dim, component_precision=cp, params_dtype=torch.bfloat16).to("cuda")
    mod.eval()

    rows = []
    for step_idx in range(args.warmup + args.iters):
        x = torch.randn(
            args.batch,
            args.seq_len,
            args.seq_len,
            args.dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=not args.inference,
        )
        mask = torch.ones(args.batch, args.seq_len, args.seq_len, device="cuda", dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        if args.inference:
            with torch.no_grad():
                out, timings = _section_forward(mod, x, mask)
            del out
        else:
            out, timings = _section_forward(mod, x, mask)
            timings["backward_ms"] = _cuda_ms(lambda: out.square().mean().backward())
        timings["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        if step_idx >= args.warmup:
            rows.append(timings)
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    payload = {
        "backend": args.backend,
        "shape": [args.batch, args.seq_len, args.seq_len, args.dim],
        "warmup": args.warmup,
        "iters": args.iters,
        "inference": args.inference,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
