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

"""Tests for consolidated DCP checkpointing of single_grouped_weight expert-parallel models."""

import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import transformer_engine.pytorch as te


sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from distributed_helpers import DistributedConfig, get_dummy_batch
from test_te_capabilities import _is_supported_blackwell

from grouped_dcp import (
    load_consolidated,
    load_optimizer_consolidated,
    save_consolidated,
    save_optimizer_consolidated,
)
from modeling_mixtral_te import _ensure_fused_grouped_mlp_registered


requires_four_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="Test requires at least 4 GPUs",
)

requires_two_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

DISCRETE_TOKENS_PER_EXPERT = 256
DISCRETE_HIDDEN = 128
DISCRETE_INTER = 256


def _fused_mxfp8_kernel_supported() -> bool:
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    return _ensure_fused_grouped_mlp_registered()


def _maybe_skip_discrete(quantized: bool) -> None:
    if not _is_supported_blackwell():
        pytest.skip("fused GroupedMLP requires datacenter Blackwell (sm_100+)")
    if quantized:
        from transformer_engine.pytorch import fp8

        ok, reason = fp8.check_mxfp8_support()
        if not ok:
            pytest.skip(f"MXFP8 unsupported: {reason}")
        if not _fused_mxfp8_kernel_supported():
            pytest.skip(
                "CuteDSL fused MXFP8 kernel unavailable — need nvidia-cutlass-dsl==4.4.1 + "
                "nvidia-cudnn-frontend<1.24 + apache-tvm-ffi"
            )


class SingleGroupedWeightMoEBlock(nn.Module):
    """Minimal MoE block using native single_grouped_weight GroupedLinear experts."""

    def __init__(
        self,
        num_local_experts: int,
        total_num_experts: int,
        hidden_size: int = 128,
        intermediate_size: int = 256,
    ) -> None:
        super().__init__()
        os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
        self.hidden_size = hidden_size
        self.num_local_experts = num_local_experts
        self.total_num_experts = total_num_experts

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=0.02)

        self.experts_gate_up = te.GroupedLinear(
            num_gemms=num_local_experts,
            in_features=hidden_size,
            out_features=2 * intermediate_size,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
            init_method=_init_method,
            single_grouped_weight=True,
        )
        self.experts_down = te.GroupedLinear(
            num_gemms=num_local_experts,
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
            init_method=_init_method,
            single_grouped_weight=True,
        )
        # Router is replicated across EP ranks and always covers all global experts.
        self.router = nn.Linear(hidden_size, total_num_experts, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Simple top-1 routed forward for checkpoint parity testing."""
        batch, seq, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        logits = self.router(flat)
        expert_idx = logits.argmax(dim=-1)
        # For checkpoint tests each rank owns a disjoint expert slice; map global -> local.
        local_idx = expert_idx % self.num_local_experts
        m_splits = torch.bincount(local_idx, minlength=self.num_local_experts).tolist()
        sorted_idx = local_idx.argsort()
        tokens = flat[sorted_idx]

        gate_up = self.experts_gate_up(tokens, m_splits=m_splits)
        gate, up = gate_up.chunk(2, dim=-1)
        mid = torch.nn.functional.silu(gate) * up
        out = self.experts_down(mid, m_splits=m_splits)

        result = torch.empty_like(flat)
        result[sorted_idx] = out
        return result.reshape(batch, seq, hidden)


class SingleGroupedWeightMoE(nn.Module):
    """Tiny causal LM shell around one MoE block for EP checkpoint tests."""

    def __init__(self, num_local_experts: int, total_num_experts: int, vocab_size: int = 1000) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, 128, dtype=torch.bfloat16)
        self.moe = SingleGroupedWeightMoEBlock(
            num_local_experts=num_local_experts,
            total_num_experts=total_num_experts,
        )
        self.lm_head = nn.Linear(128, vocab_size, bias=False, dtype=torch.bfloat16)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.moe(x)
        return self.lm_head(x)


class DiscreteFusedMoEBlock(nn.Module):
    """Minimal MoE block using discrete ops.GroupedLinear experts (fused_grouped_mlp path)."""

    def __init__(
        self,
        num_local_experts: int,
        total_num_experts: int,
        quantized: bool = False,
        hidden_size: int = DISCRETE_HIDDEN,
        intermediate_size: int = DISCRETE_INTER,
    ) -> None:
        super().__init__()
        os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
        os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
        self.hidden_size = hidden_size
        self.num_local_experts = num_local_experts
        self.total_num_experts = total_num_experts
        self.quantized = quantized

        import transformer_engine.common.recipe as te_recipe
        from transformer_engine.pytorch.ops import GroupedLinear as OpsGL
        from transformer_engine.pytorch.ops import ScaledSwiGLU, Sequential

        recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)
        ctx = te.quantized_model_init(recipe=recipe, enabled=True) if quantized else nullcontext()
        with ctx:
            gate_up = OpsGL(
                num_groups=num_local_experts,
                in_features=hidden_size,
                out_features=2 * intermediate_size,
                bias=False,
                dtype=torch.bfloat16,
                device="cuda",
            )
            down = OpsGL(
                num_groups=num_local_experts,
                in_features=intermediate_size,
                out_features=hidden_size,
                bias=False,
                dtype=torch.bfloat16,
                device="cuda",
            )
        self.experts_gate_up = gate_up
        self.experts_down = down
        self._ffn = Sequential(gate_up, ScaledSwiGLU(glu_interleave_size=32), down)
        self.router = nn.Linear(hidden_size, total_num_experts, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        per = flat.shape[0] // self.num_local_experts
        assert per * self.num_local_experts == flat.shape[0]
        m_splits = [per] * self.num_local_experts
        split_sizes = torch.tensor(m_splits, dtype=torch.int32, device=flat.device)
        probs = torch.ones(flat.shape[0], device=flat.device, dtype=torch.float32)
        out = self._ffn(flat, split_sizes, probs, split_sizes)
        return out.reshape(batch, seq, hidden)


class DiscreteFusedMoE(nn.Module):
    """Tiny causal LM shell around discrete fused MoE experts for EP checkpoint tests."""

    def __init__(
        self,
        num_local_experts: int,
        total_num_experts: int,
        quantized: bool = False,
        vocab_size: int = 1000,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, DISCRETE_HIDDEN, dtype=torch.bfloat16)
        self.moe = DiscreteFusedMoEBlock(
            num_local_experts=num_local_experts,
            total_num_experts=total_num_experts,
            quantized=quantized,
        )
        self.lm_head = nn.Linear(DISCRETE_HIDDEN, vocab_size, bias=False, dtype=torch.bfloat16)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.moe(x)
        return self.lm_head(x)


def _run_torchrun(test_fn_name: str, tmp_dir: str, nproc: int, *extra_args: str):
    """Run a named worker function via torchrun."""
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(Path(__file__).resolve()),
        test_fn_name,
        tmp_dir,
        *extra_args,
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        cwd=str(Path(__file__).parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"{test_fn_name} failed with exit code {result.returncode}")


@requires_four_gpu
def test_grouped_dcp_ep4_roundtrip(tmp_path):
    """EP=4 consolidated DCP save/load stop-and-go parity."""
    _run_torchrun("ep4_roundtrip", str(tmp_path), nproc=4)


@requires_four_gpu
def test_grouped_dcp_ep4_to_ep2_reshard(tmp_path):
    """EP=4 save then EP=2 load reshard via consolidated DCP."""
    _run_torchrun("ep4_save", str(tmp_path), nproc=4)
    _run_torchrun("ep2_load_verify", str(tmp_path), nproc=2)


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "mxfp8"])
@requires_four_gpu
def test_grouped_dcp_discrete_ep4_roundtrip(tmp_path, quantized):
    """EP=4 consolidated DCP save/load for discrete fused experts + FusedAdam state."""
    _maybe_skip_discrete(quantized)
    _run_torchrun(
        "discrete_ep4_roundtrip",
        str(tmp_path),
        4,
        "1" if quantized else "0",
    )


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "mxfp8"])
@requires_four_gpu
def test_grouped_dcp_discrete_ep4_to_ep2_reshard(tmp_path, quantized):
    """EP=4 save then EP=2 load reshard for discrete fused experts + optimizer state."""
    _maybe_skip_discrete(quantized)
    _run_torchrun(
        "discrete_ep4_save",
        str(tmp_path),
        4,
        "1" if quantized else "0",
    )
    _run_torchrun(
        "discrete_ep2_load_verify",
        str(tmp_path),
        2,
        "1" if quantized else "0",
    )


def _init_distributed():
    from torch.distributed.tensor.device_mesh import DeviceMesh

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    ep_size = dist_config.world_size
    ep_mesh = DeviceMesh("cuda", list(range(ep_size)))
    return dist_config, device, ep_mesh


def _expert_weight_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Gather local dequantized expert weights for comparison."""
    snap: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        for attr in ("experts_gate_up", "experts_down"):
            gl = getattr(module, attr, None)
            if gl is not None and hasattr(gl, "weight"):
                snap[f"{name}.{attr}.weight"] = gl.weight.dequantize().detach().cpu().clone()
    return snap


def _gather_global_expert_weights(model: nn.Module, ep_mesh) -> dict[str, torch.Tensor]:
    """All-gather local expert shards into a full global tensor on every rank."""
    from torch.distributed.tensor import DTensor, Shard

    global_snap: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        for attr in ("experts_gate_up", "experts_down"):
            gl = getattr(module, attr, None)
            if gl is None or not hasattr(gl, "weight"):
                continue
            key = f"{name}.{attr}.weight"
            local = gl.weight.dequantize().contiguous()
            dt = DTensor.from_local(local, device_mesh=ep_mesh, placements=[Shard(0)])
            global_snap[key] = dt.full_tensor().detach().cpu()
    return global_snap


def _create_model(
    num_local_experts: int, total_num_experts: int, device: torch.device, seed: int
) -> SingleGroupedWeightMoE:
    torch.manual_seed(seed)
    model = SingleGroupedWeightMoE(
        num_local_experts=num_local_experts,
        total_num_experts=total_num_experts,
    ).to(device=device)
    return model


def _train_steps(model: nn.Module, device: torch.device, steps: int = 3) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = get_dummy_batch(model.vocab_size, seq_len=32, batch_size=2, device=str(device))
    for _ in range(steps):
        opt.zero_grad()
        logits = model(batch["input_ids"])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, model.vocab_size), batch["labels"].reshape(-1))
        loss.backward()
        opt.step()


def _worker_ep4_roundtrip(tmp_dir: str):
    """Save at EP=4, load into fresh EP=4 model, verify weights and forward."""
    dist_config, device, ep_mesh = _init_distributed()
    assert dist_config.world_size == 4
    num_local_experts = 8
    total_num_experts = num_local_experts * dist_config.world_size
    ckpt_dir = os.path.join(tmp_dir, "ep4_ckpt")

    model = _create_model(num_local_experts, total_num_experts, device, seed=0)
    _train_steps(model, device)
    model.eval()

    ref_snap = _expert_weight_snapshot(model)
    batch = get_dummy_batch(model.vocab_size, device=str(device))
    with torch.no_grad():
        ref_logits = model(batch["input_ids"]).detach().cpu()

    save_consolidated(model, ep_mesh, ckpt_dir)
    del model
    torch.cuda.empty_cache()

    model_new = _create_model(num_local_experts, total_num_experts, device, seed=999)
    load_consolidated(model_new, ep_mesh, ckpt_dir)
    model_new.eval()

    new_snap = _expert_weight_snapshot(model_new)
    for key, ref_w in ref_snap.items():
        torch.testing.assert_close(new_snap[key], ref_w, atol=0, rtol=0, msg=f"weight mismatch {key}")

    with torch.no_grad():
        new_logits = model_new(batch["input_ids"]).detach().cpu()
    if dist_config.is_main_process():
        torch.testing.assert_close(new_logits, ref_logits, atol=1e-3, rtol=1e-3)
        print("EP=4 grouped DCP round-trip PASSED")

    torch.distributed.destroy_process_group()


def _worker_ep4_save(tmp_dir: str):
    """Train EP=4 model, save consolidated checkpoint, write global reference on rank 0."""
    dist_config, device, ep_mesh = _init_distributed()
    assert dist_config.world_size == 4
    num_local_experts = 8
    total_num_experts = num_local_experts * dist_config.world_size
    ckpt_dir = os.path.join(tmp_dir, "reshard_ckpt")
    ref_path = os.path.join(tmp_dir, "global_expert_ref.pt")

    model = _create_model(num_local_experts, total_num_experts, device, seed=42)
    _train_steps(model, device)
    save_consolidated(model, ep_mesh, ckpt_dir)

    global_ref = _gather_global_expert_weights(model, ep_mesh)
    if dist_config.is_main_process():
        torch.save(global_ref, ref_path)
        print(f"Saved global reference with {len(global_ref)} expert tensors")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _worker_ep2_load_verify(tmp_dir: str):
    """Load EP=4 checkpoint into EP=2 model and verify against gathered reference."""
    dist_config, device, ep_mesh = _init_distributed()
    assert dist_config.world_size == 2
    total_num_experts = 32
    num_local_experts = total_num_experts // dist_config.world_size
    ckpt_dir = os.path.join(tmp_dir, "reshard_ckpt")
    ref_path = os.path.join(tmp_dir, "global_expert_ref.pt")

    model = _create_model(num_local_experts, total_num_experts, device, seed=999)
    load_consolidated(model, ep_mesh, ckpt_dir)
    model.eval()

    loaded_global = _gather_global_expert_weights(model, ep_mesh)

    if dist_config.is_main_process():
        ref_global = torch.load(ref_path, weights_only=True)
        for key, ref_w in ref_global.items():
            torch.testing.assert_close(
                loaded_global[key],
                ref_w,
                atol=0,
                rtol=0,
                msg=f"reshard weight mismatch {key}",
            )
        print("EP=4 -> EP=2 grouped DCP reshard PASSED")

    torch.distributed.destroy_process_group()


def _count_discrete_weights(gl: nn.Module) -> int:
    count = 0
    while getattr(gl, f"weight{count}", None) is not None:
        count += 1
    return count


def _discrete_expert_weight_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Stacked per-expert dequantized weights for discrete GroupedLinear modules."""
    snap: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        for attr in ("experts_gate_up", "experts_down"):
            gl = getattr(module, attr, None)
            if gl is None or getattr(gl, "weight0", None) is None:
                continue
            num_local = _count_discrete_weights(gl)
            stacked = torch.stack(
                [
                    getattr(gl, f"weight{i}").dequantize().detach().cpu().clone()
                    if hasattr(getattr(gl, f"weight{i}"), "dequantize")
                    else getattr(gl, f"weight{i}").detach().cpu().clone()
                    for i in range(num_local)
                ],
                dim=0,
            )
            snap[f"{name}.{attr}.weight"] = stacked
    return snap


def _gather_global_discrete_expert_weights(model: nn.Module, ep_mesh) -> dict[str, torch.Tensor]:
    """All-gather stacked local expert shards into a full global tensor on every rank."""
    from torch.distributed.tensor import DTensor, Shard

    global_snap: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        for attr in ("experts_gate_up", "experts_down"):
            gl = getattr(module, attr, None)
            if gl is None or getattr(gl, "weight0", None) is None:
                continue
            num_local = _count_discrete_weights(gl)
            deq = [
                getattr(gl, f"weight{i}").dequantize()
                if hasattr(getattr(gl, f"weight{i}"), "dequantize")
                else getattr(gl, f"weight{i}").detach()
                for i in range(num_local)
            ]
            local_stack = torch.stack(deq, dim=0).contiguous()
            key = f"{name}.{attr}.weight"
            dt = DTensor.from_local(local_stack, device_mesh=ep_mesh, placements=[Shard(0)])
            global_snap[key] = dt.full_tensor().detach().cpu()
    return global_snap


def _gather_global_optimizer_master(model: nn.Module, optimizer, ep_mesh) -> dict[str, torch.Tensor]:
    """All-gather stacked FusedAdam state shards (master_param, exp_avg, exp_avg_sq) for discrete experts."""
    from torch.distributed.tensor import DTensor, Shard

    out: dict[str, torch.Tensor] = {}
    state_names = ("master_param", "exp_avg", "exp_avg_sq")
    for name, module in model.named_modules():
        for attr in ("experts_gate_up", "experts_down"):
            gl = getattr(module, attr, None)
            if gl is None or getattr(gl, "weight0", None) is None:
                continue
            num_local = _count_discrete_weights(gl)
            for state_name in state_names:
                key = f"{name}.{attr}.{state_name}"
                locals_ = [optimizer.state[getattr(gl, f"weight{i}")][state_name] for i in range(num_local)]
                dt = DTensor.from_local(
                    torch.stack(locals_, dim=0).contiguous(),
                    device_mesh=ep_mesh,
                    placements=[Shard(0)],
                )
                out[key] = dt.full_tensor().detach().cpu()
    return out


def _create_discrete_model(
    num_local_experts: int,
    total_num_experts: int,
    device: torch.device,
    seed: int,
    quantized: bool,
) -> DiscreteFusedMoE:
    torch.manual_seed(seed)
    model = DiscreteFusedMoE(
        num_local_experts=num_local_experts,
        total_num_experts=total_num_experts,
        quantized=quantized,
    ).to(device=device)
    return model


def _discrete_train_steps(model: nn.Module, device: torch.device, quantized: bool, steps: int = 3) -> object:
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch.optimizers import FusedAdam

    expert_params = list(model.moe.experts_gate_up.parameters()) + list(model.moe.experts_down.parameters())
    opt = FusedAdam(expert_params, lr=1e-3, master_weights=True)
    per = DISCRETE_TOKENS_PER_EXPERT
    num_local = model.moe.num_local_experts
    split_sizes = torch.tensor([per] * num_local, dtype=torch.int32, device=device)
    recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)
    model.train()
    for step in range(steps):
        torch.manual_seed(step)
        x = torch.randn(per * num_local, DISCRETE_HIDDEN, device=device, dtype=torch.bfloat16)
        probs = torch.ones(per * num_local, device=device, dtype=torch.float32)
        opt.zero_grad()
        if quantized:
            with te.autocast(enabled=True, recipe=recipe):
                out = model.moe._ffn(x, split_sizes, probs, split_sizes)
        else:
            out = model.moe._ffn(x, split_sizes, probs, split_sizes)
        out.sum().backward()
        opt.step()
    return opt


def _worker_discrete_ep4_roundtrip(tmp_dir: str, quantized: bool):
    """Save at EP=4, load into fresh EP=4 discrete model; verify weights and optimizer state."""
    dist_config, device, ep_mesh = _init_distributed()
    assert dist_config.world_size == 4
    num_local_experts = 8
    total_num_experts = num_local_experts * dist_config.world_size
    ckpt_dir = os.path.join(tmp_dir, "discrete_ep4_ckpt")
    opt_ckpt_dir = os.path.join(tmp_dir, "discrete_ep4_opt_ckpt")

    model = _create_discrete_model(num_local_experts, total_num_experts, device, seed=0, quantized=quantized)
    opt = _discrete_train_steps(model, device, quantized)
    model.eval()

    ref_snap = _discrete_expert_weight_snapshot(model)
    ref_master = _gather_global_optimizer_master(model, opt, ep_mesh)

    save_consolidated(model, ep_mesh, ckpt_dir)
    save_optimizer_consolidated(model, opt, ep_mesh, opt_ckpt_dir)
    del model, opt
    torch.cuda.empty_cache()

    model_new = _create_discrete_model(num_local_experts, total_num_experts, device, seed=999, quantized=quantized)
    opt_new = _discrete_train_steps(model_new, device, quantized, steps=1)
    load_consolidated(model_new, ep_mesh, ckpt_dir)
    load_optimizer_consolidated(model_new, opt_new, ep_mesh, opt_ckpt_dir)
    model_new.eval()

    new_snap = _discrete_expert_weight_snapshot(model_new)
    for key, ref_w in ref_snap.items():
        torch.testing.assert_close(new_snap[key], ref_w, atol=0, rtol=0, msg=f"weight mismatch {key}")

    new_master = _gather_global_optimizer_master(model_new, opt_new, ep_mesh)
    for key, ref_w in ref_master.items():
        torch.testing.assert_close(new_master[key], ref_w, atol=0, rtol=0, msg=f"optimizer state mismatch {key}")

    if dist_config.is_main_process():
        print(f"EP=4 discrete grouped DCP round-trip PASSED (quantized={quantized})")

    torch.distributed.destroy_process_group()


def _worker_discrete_ep4_save(tmp_dir: str, quantized: bool):
    """Train EP=4 discrete model, save checkpoint, write global references on rank 0."""
    dist_config, device, ep_mesh = _init_distributed()
    assert dist_config.world_size == 4
    num_local_experts = 8
    total_num_experts = num_local_experts * dist_config.world_size
    ckpt_dir = os.path.join(tmp_dir, "discrete_reshard_ckpt")
    opt_ckpt_dir = os.path.join(tmp_dir, "discrete_reshard_opt_ckpt")
    ref_path = os.path.join(tmp_dir, "discrete_global_expert_ref.pt")
    opt_ref_path = os.path.join(tmp_dir, "discrete_global_opt_ref.pt")

    model = _create_discrete_model(num_local_experts, total_num_experts, device, seed=42, quantized=quantized)
    opt = _discrete_train_steps(model, device, quantized)
    save_consolidated(model, ep_mesh, ckpt_dir)
    save_optimizer_consolidated(model, opt, ep_mesh, opt_ckpt_dir)

    global_ref = _gather_global_discrete_expert_weights(model, ep_mesh)
    opt_ref = _gather_global_optimizer_master(model, opt, ep_mesh)
    if dist_config.is_main_process():
        torch.save(global_ref, ref_path)
        torch.save(opt_ref, opt_ref_path)
        print(f"Saved discrete global reference with {len(global_ref)} expert tensors")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _worker_discrete_ep2_load_verify(tmp_dir: str, quantized: bool):
    """Load EP=4 discrete checkpoint into EP=2 model and verify against gathered reference."""
    dist_config, device, ep_mesh = _init_distributed()
    assert dist_config.world_size == 2
    total_num_experts = 32
    num_local_experts = total_num_experts // dist_config.world_size
    ckpt_dir = os.path.join(tmp_dir, "discrete_reshard_ckpt")
    opt_ckpt_dir = os.path.join(tmp_dir, "discrete_reshard_opt_ckpt")
    ref_path = os.path.join(tmp_dir, "discrete_global_expert_ref.pt")
    opt_ref_path = os.path.join(tmp_dir, "discrete_global_opt_ref.pt")

    model = _create_discrete_model(num_local_experts, total_num_experts, device, seed=999, quantized=quantized)
    opt = _discrete_train_steps(model, device, quantized, steps=1)
    load_consolidated(model, ep_mesh, ckpt_dir)
    load_optimizer_consolidated(model, opt, ep_mesh, opt_ckpt_dir)
    model.eval()

    loaded_global = _gather_global_discrete_expert_weights(model, ep_mesh)
    loaded_opt = _gather_global_optimizer_master(model, opt, ep_mesh)

    if dist_config.is_main_process():
        ref_global = torch.load(ref_path, weights_only=True)
        ref_opt = torch.load(opt_ref_path, weights_only=True)
        for key, ref_w in ref_global.items():
            torch.testing.assert_close(
                loaded_global[key],
                ref_w,
                atol=0,
                rtol=0,
                msg=f"reshard weight mismatch {key}",
            )
        for key, ref_w in ref_opt.items():
            torch.testing.assert_close(
                loaded_opt[key],
                ref_w,
                atol=0,
                rtol=0,
                msg=f"reshard optimizer state mismatch {key}",
            )
        print(f"EP=4 -> EP=2 discrete grouped DCP reshard PASSED (quantized={quantized})")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_name = sys.argv[1]
    tmp_dir = sys.argv[2]
    quantized = sys.argv[3] == "1" if len(sys.argv) > 3 else False
    workers = {
        "ep4_roundtrip": _worker_ep4_roundtrip,
        "ep4_save": _worker_ep4_save,
        "ep2_load_verify": _worker_ep2_load_verify,
        "discrete_ep4_roundtrip": lambda d: _worker_discrete_ep4_roundtrip(d, quantized),
        "discrete_ep4_save": lambda d: _worker_discrete_ep4_save(d, quantized),
        "discrete_ep2_load_verify": lambda d: _worker_discrete_ep2_load_verify(d, quantized),
    }
    workers[test_name](tmp_dir)
