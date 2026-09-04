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

"""Compatibility matrix: compute x precision x EP x checkpoint for single_grouped_weight.

Precision axis distinguishes *how* MXFP8 is applied:

- ``bf16``           — bf16 parameters, no fp8 anywhere.
- ``mxfp8_autocast`` — bf16 master parameters, MXFP8 applied transiently inside ``te.autocast``.
                       This is the ONLY working MXFP8 mechanism for grouped experts on TE 2.16.
- ``mxfp8_params``   — *persistent* fp8 parameters via ``quantized_model_init`` (what a
                       ``FusedAdam(master_weights=True)`` recipe would want). Characterized
                       separately (``test_fp8_parameters_*``) because TE 2.16 does **not** quantize
                       ``GroupedLinear`` expert weights — the grouped weight stays bf16 — so this is
                       currently an xfail, not a matrix cell.

Fused-kernel cells assert ``ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8.is_supported()`` so the suite
never silently measures the unfused fallback (whose MXFP8 backward is numerically broken). The
fused CuteDSL kernel requires ``tokens-per-group % 256 == 0``; every cell drives the experts with a
deterministic even split of 256 tokens per local expert to stay on a supported shape.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import transformer_engine.pytorch as te


sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from distributed_helpers import DistributedConfig
from test_te_capabilities import _is_supported_blackwell

from grouped_dcp import load_consolidated, save_consolidated
from modeling_mixtral_te import _ensure_fused_grouped_mlp_registered


# Fused CuteDSL MXFP8 grouped GEMM requires each group's token count divisible by 256.
TOKENS_PER_EXPERT = 256
HIDDEN = 128
INTER = 256
VOCAB = 1000


@dataclass(frozen=True)
class MatrixCell:
    compute: str  # module_grouped_linear | fused_grouped_mlp
    precision: str  # bf16 | mxfp8_autocast
    ep_size: int  # 1 | 2 | 4
    checkpoint: bool  # grouped_dcp round-trip


def _mxfp8_supported() -> tuple[bool, str]:
    from transformer_engine.pytorch import fp8

    return fp8.check_mxfp8_support()


def _fused_mxfp8_kernel_supported() -> bool:
    """True only when the real CuteDSL fused MXFP8 kernel will fire (not the unfused fallback).

    ``is_supported`` is ``functools.lru_cache``d and may be called at fused-op import time before
    ``NVTE_CUTEDSL_FUSED_GROUPED_MLP`` is set, which would cache a stale ``False``. The model's
    compatibility helper clears that cache and registers the version-appropriate TE fusion.
    """
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    return _ensure_fused_grouped_mlp_registered()


def _cell_marks(cell: MatrixCell) -> list:
    marks: list = []
    if cell.compute == "fused_grouped_mlp" and not _is_supported_blackwell():
        marks.append(pytest.mark.skip(reason="fused GroupedMLP requires datacenter Blackwell (sm_100+)"))

    if cell.precision == "mxfp8_autocast":
        ok, reason = _mxfp8_supported()
        if not ok:
            marks.append(pytest.mark.skip(reason=f"MXFP8 unsupported: {reason}"))
        if cell.compute == "fused_grouped_mlp" and _is_supported_blackwell() and not _fused_mxfp8_kernel_supported():
            marks.append(
                pytest.mark.skip(
                    reason=(
                        "CuteDSL fused MXFP8 kernel unavailable — need nvidia-cutlass-dsl==4.4.1 + "
                        "nvidia-cudnn-frontend<1.24 + apache-tvm-ffi (else TE silently falls back to "
                        "the numerically-broken unfused path)"
                    )
                )
            )

    if cell.ep_size > 1 and (not torch.cuda.is_available() or torch.cuda.device_count() < cell.ep_size):
        marks.append(pytest.mark.skip(reason=f"requires {cell.ep_size} GPUs"))
    return marks


MATRIX: list = []
for compute in ("module_grouped_linear", "fused_grouped_mlp"):
    for precision in ("bf16", "mxfp8_autocast"):
        for ep in (1, 2, 4):
            for ckpt in (False, True):
                cell = MatrixCell(compute, precision, ep, ckpt)
                MATRIX.append(
                    pytest.param(
                        cell,
                        id=f"{compute[:3]}_{precision}_ep{ep}{'_dcp' if ckpt else ''}",
                        marks=_cell_marks(cell),
                    )
                )


class MatrixMoEBlock(nn.Module):
    """MoE block with selectable compute path, always single_grouped_weight.

    Tokens are dispatched with a deterministic even split (``TOKENS_PER_EXPERT`` per local expert)
    so every grouped GEMM sees a fused-kernel-supported shape.
    """

    def __init__(self, num_local_experts: int, compute: str) -> None:
        super().__init__()
        os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
        self.num_local_experts = num_local_experts
        self.compute = compute

        def _init(x):
            torch.nn.init.normal_(x, std=0.02)

        if compute == "module_grouped_linear":
            self.experts_gate_up = te.GroupedLinear(
                num_gemms=num_local_experts,
                in_features=HIDDEN,
                out_features=2 * INTER,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
                init_method=_init,
                single_grouped_weight=True,
            )
            self.experts_down = te.GroupedLinear(
                num_gemms=num_local_experts,
                in_features=INTER,
                out_features=HIDDEN,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
                init_method=_init,
                single_grouped_weight=True,
            )
            self._ffn = None
        else:
            os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
            from transformer_engine.pytorch.ops import GroupedLinear as OpsGL
            from transformer_engine.pytorch.ops import ScaledSwiGLU, Sequential

            gate_up = OpsGL(
                num_groups=num_local_experts,
                in_features=HIDDEN,
                out_features=2 * INTER,
                bias=False,
                dtype=torch.bfloat16,
                device="cuda",
                single_grouped_weight=True,
            )
            down = OpsGL(
                num_groups=num_local_experts,
                in_features=INTER,
                out_features=HIDDEN,
                bias=False,
                dtype=torch.bfloat16,
                device="cuda",
                single_grouped_weight=True,
            )
            self._ffn = Sequential(gate_up, ScaledSwiGLU(glu_interleave_size=32), down)
            self.experts_gate_up = gate_up
            self.experts_down = down

    def forward(self, x: torch.Tensor, *, use_mxfp8: bool = False) -> torch.Tensor:
        import transformer_engine.common.recipe as te_recipe

        b, s, h = x.shape
        flat = x.reshape(-1, h)
        n = flat.shape[0]
        per = n // self.num_local_experts
        assert per * self.num_local_experts == n, "batch must be TOKENS_PER_EXPERT * num_local_experts"
        m_splits = [per] * self.num_local_experts
        recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)

        if self.compute == "module_grouped_linear":

            def _run():
                gu = self.experts_gate_up(flat, m_splits=m_splits)
                gate, up = gu.chunk(2, dim=-1)
                mid = torch.nn.functional.silu(gate) * up
                return self.experts_down(mid, m_splits=m_splits)

            if use_mxfp8:
                with te.autocast(enabled=True, recipe=recipe):
                    out = _run()
            else:
                out = _run()
        else:
            split_sizes = torch.tensor(m_splits, dtype=torch.int32, device=x.device)
            probs = torch.ones(n, device=x.device, dtype=torch.float32)
            if use_mxfp8:
                with te.autocast(enabled=True, recipe=recipe):
                    out = self._ffn(flat, split_sizes, probs, split_sizes)
            else:
                out = self._ffn(flat, split_sizes, probs, split_sizes)

        return out.reshape(b, s, h)


class MatrixModel(nn.Module):
    def __init__(self, num_local: int, compute: str) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN, dtype=torch.bfloat16)
        self.moe = MatrixMoEBlock(num_local, compute)
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False, dtype=torch.bfloat16)

    def forward(self, ids: torch.Tensor, *, use_mxfp8: bool = False) -> torch.Tensor:
        x = self.embed(ids)
        x = self.moe(x, use_mxfp8=use_mxfp8)
        return self.head(x)


def _batch(num_local: int, device: str = "cuda") -> dict:
    """Even batch: TOKENS_PER_EXPERT tokens per local expert, so every group is ÷256."""
    tokens = TOKENS_PER_EXPERT * num_local
    torch.manual_seed(42)
    ids = torch.randint(0, VOCAB, (num_local, TOKENS_PER_EXPERT), device=device)
    labels = ids.clone()
    return {"input_ids": ids, "labels": labels, "tokens": tokens}


def _record(results_path: Path, cell: MatrixCell, status: str, detail: str = "") -> None:
    data: dict = {}
    if results_path.exists():
        data = json.loads(results_path.read_text())
    key = f"{cell.compute}|{cell.precision}|ep{cell.ep_size}|ckpt={cell.checkpoint}"
    data[key] = {"status": status, "detail": detail}
    results_path.write_text(json.dumps(data, indent=2))


def _loss(model: nn.Module, batch: dict, mxfp8: bool) -> torch.Tensor:
    logits = model(batch["input_ids"], use_mxfp8=mxfp8)
    return torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), batch["labels"].reshape(-1))


@pytest.mark.parametrize("cell", MATRIX)
def test_compat_matrix_cell(cell: MatrixCell, tmp_path):
    """Run one matrix cell (subprocess for EP>1)."""
    results_path = tmp_path / "compat_matrix_results.json"
    if cell.ep_size == 1:
        _run_cell_local(cell, tmp_path)
        _record(results_path, cell, "PASS")
        return

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={cell.ep_size}",
        str(Path(__file__).resolve()),
        "worker",
        json.dumps(cell.__dict__),
        str(tmp_path),
    ]
    env = os.environ.copy()
    env["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    if cell.compute == "fused_grouped_mlp":
        env["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        _record(results_path, cell, "FAIL", result.stderr[-500:])
        pytest.fail(f"cell {cell} failed:\n{result.stdout}\n{result.stderr}")
    _record(results_path, cell, "PASS")


def _train_and_maybe_checkpoint(cell: MatrixCell, num_local: int, ep_mesh, tmp_path, device) -> None:
    torch.manual_seed(0)
    model = MatrixModel(num_local, cell.compute).to(device)
    mx = cell.precision == "mxfp8_autocast"
    batch = _batch(num_local, device=str(device))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = _loss(model, batch, mx)
    loss.backward()
    opt.step()

    # Weights must stay finite through an MXFP8 train step (regression guard for the
    # unfused-fallback NaN bug — only the fused kernel is exercised here via is_supported gating).
    for gl in (model.moe.experts_gate_up, model.moe.experts_down):
        assert torch.isfinite(gl.weight.dequantize()).all(), "expert weight went non-finite after step"

    if cell.checkpoint:
        ckpt = str(Path(tmp_path) / f"ckpt_ep{ep_mesh.size()}")
        save_consolidated(model, ep_mesh, ckpt)
        snap = {
            "gu": model.moe.experts_gate_up.weight.dequantize().clone(),
            "dn": model.moe.experts_down.weight.dequantize().clone(),
        }
        model2 = MatrixModel(num_local, cell.compute).to(device)
        load_consolidated(model2, ep_mesh, ckpt)
        torch.testing.assert_close(model2.moe.experts_gate_up.weight.dequantize(), snap["gu"])
        torch.testing.assert_close(model2.moe.experts_down.weight.dequantize(), snap["dn"])


def _run_cell_local(cell: MatrixCell, tmp_path: Path) -> None:
    import torch.distributed as dist
    from torch.distributed.tensor.device_mesh import DeviceMesh

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    if cell.compute == "fused_grouped_mlp":
        os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", store=dist.HashStore(), rank=0, world_size=1)
    ep_mesh = DeviceMesh("cuda", [0])
    try:
        _train_and_maybe_checkpoint(cell, num_local=8, ep_mesh=ep_mesh, tmp_path=tmp_path, device=torch.device("cuda"))
    finally:
        dist.destroy_process_group()


def _worker_main(cell_dict: dict, tmp_dir: str) -> None:
    from torch.distributed.tensor.device_mesh import DeviceMesh

    cell = MatrixCell(**cell_dict)
    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    if cell.compute == "fused_grouped_mlp":
        os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    ep_mesh = DeviceMesh("cuda", list(range(dist_config.world_size)))
    num_local = 8 // dist_config.world_size
    _train_and_maybe_checkpoint(cell, num_local=num_local, ep_mesh=ep_mesh, tmp_path=tmp_dir, device=device)
    torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# fp8 *parameters* (persistent) — separate from the autocast matrix above.
# These document the two mechanisms a FusedAdam(master_weights=True) + quantized_model_init recipe
# would want, both currently blocked in TE 2.16.
# ---------------------------------------------------------------------------


def _looks_quantized(w: torch.Tensor) -> bool:
    tn = type(w).__name__
    if any(k in tn for k in ("FP8", "MXFP8", "NVFP4")):
        return True
    rd = getattr(w, "_rowwise_data", None)
    if rd is not None and rd.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    return w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("compute", ["module_grouped_linear", "fused_grouped_mlp"])
@pytest.mark.xfail(
    reason=(
        "TE 2.16: quantized_model_init does NOT quantize GroupedLinear expert weights — the grouped "
        "weight stays bf16, so persistent fp8 params are unsupported for MoE experts. XPASS here "
        "signals TE gained support."
    ),
    strict=False,
)
def test_fp8_parameters_persist_for_grouped_experts(compute):
    """quantized_model_init should yield a quantized (fp8) expert weight; today it stays bf16."""
    import transformer_engine.common.recipe as te_recipe

    ok, reason = _mxfp8_supported()
    if not ok:
        pytest.skip(f"MXFP8 unsupported: {reason}")
    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)
    with te.quantized_model_init(recipe=recipe, enabled=True):
        if compute == "module_grouped_linear":
            gl = te.GroupedLinear(
                4,
                HIDDEN,
                2 * INTER,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
                single_grouped_weight=True,
            )
        else:
            os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
            from transformer_engine.pytorch.ops import GroupedLinear as OpsGL

            gl = OpsGL(
                num_groups=4,
                in_features=HIDDEN,
                out_features=2 * INTER,
                bias=False,
                dtype=torch.bfloat16,
                device="cuda",
                single_grouped_weight=True,
            )
    assert _looks_quantized(gl.weight), (
        f"expected persistent fp8 params, got {type(gl.weight).__name__} dtype={gl.weight.dtype}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.xfail(
    reason=(
        "TE FusedAdam crashes with the single_grouped_weight grouped MLP (illegal memory access on "
        "the fused kernel; cuBLASLt error on the unfused path), regardless of master_weights. Plain "
        "torch.optim.Adam works. XPASS signals TE fixed the FusedAdam+GroupedTensor interaction."
    ),
    strict=False,
)
def test_te_fused_adam_supports_grouped_mlp(tmp_path):
    """A FusedAdam(master_weights=True) step over grouped experts should succeed; today it aborts."""
    if not _is_supported_blackwell() or not _fused_mxfp8_kernel_supported():
        pytest.skip("fused CuteDSL MXFP8 kernel unavailable")
    # Run in a subprocess: the failure is an illegal memory access that would abort the pytest
    # process, so we isolate it and assert on the exit code.
    cmd = [sys.executable, str(Path(__file__).resolve()), "fused_adam"]
    env = os.environ.copy()
    env["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    env["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    result = subprocess.run(
        cmd, env=env, cwd=str(Path(__file__).parent.parent), capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, f"FusedAdam grouped-MLP step failed (exit {result.returncode})"


def _fused_adam_worker() -> None:
    import torch.distributed as dist
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch.ops import GroupedLinear as OpsGL
    from transformer_engine.pytorch.ops import ScaledSwiGLU, Sequential
    from transformer_engine.pytorch.optimizers import FusedAdam

    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", store=dist.HashStore(), rank=0, world_size=1)
    ng, per = 4, TOKENS_PER_EXPERT
    gu = OpsGL(ng, HIDDEN, 2 * INTER, bias=False, dtype=torch.bfloat16, device="cuda", single_grouped_weight=True)
    dn = OpsGL(ng, INTER, HIDDEN, bias=False, dtype=torch.bfloat16, device="cuda", single_grouped_weight=True)
    ffn = Sequential(gu, ScaledSwiGLU(glu_interleave_size=32), dn)
    opt = FusedAdam(list(gu.parameters()) + list(dn.parameters()), lr=1e-3, master_weights=True)
    split = torch.tensor([per] * ng, dtype=torch.int32, device="cuda")
    x = torch.randn(per * ng, HIDDEN, device="cuda", dtype=torch.bfloat16)
    probs = torch.ones(per * ng, device="cuda", dtype=torch.float32)
    recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)
    with te.autocast(enabled=True, recipe=recipe):
        out = ffn(x, split, probs, split)
    out.sum().backward()
    opt.step()
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        _worker_main(json.loads(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "fused_adam":
        _fused_adam_worker()
