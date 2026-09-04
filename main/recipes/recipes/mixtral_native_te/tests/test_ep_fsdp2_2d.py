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

"""Combined EP=2 x FSDP2 dp=2 (2D mesh) tests for the Mixtral recipe."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.distributed.tensor import DTensor


DP_SIZE = 2
EP_SIZE = 2
TRAIN_SEED = 42
LOAD_SEED = 999

requires_four_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="Test requires at least 4 GPUs",
)

requires_sm100 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 10),
    reason="fused_grouped_mlp expert path requires compute capability 10.x; 12.0 support is pending",
)


def _run_torchrun(tmp_dir: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={DP_SIZE * EP_SIZE}",
        str(Path(__file__).resolve()),
        tmp_dir,
    ]
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    env["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    env["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        cwd=str(Path(__file__).parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        env=env,
    )


@requires_four_gpu
@requires_sm100
def test_ep2_fsdp2_dp2_2d_mesh(tmp_path):
    """EP=2 x FSDP2 dp=2: genuine dp sharding, train step, checkpoint roundtrip."""
    result = _run_torchrun(str(tmp_path))
    if result.returncode != 0:
        print(result.stdout)
        pytest.fail(f"2D mesh worker failed with exit code {result.returncode}")
    assert "EP=2 x FSDP2 dp=2 2D mesh test PASSED" in result.stdout


def _is_expert_key(name: str) -> bool:
    return ".experts_gate_up." in name or ".experts_down." in name


def _dequantized_tensor(param: torch.Tensor) -> torch.Tensor:
    # Experts and dense params are FSDP2 DTensors when dp>1; compare local shards (both the
    # reference and the reloaded model share the same sharding, so shard-wise equality suffices).
    if isinstance(param, DTensor):
        param = param.to_local()
    if hasattr(param, "dequantize"):
        return param.dequantize().detach()
    return param.detach()


def _get_dummy_batch(vocab_size: int, device: torch.device):
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (2, 32), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _build_model_and_optimizer(device: torch.device, seed: int):
    from distributed_setup import build_mesh_and_wrap
    from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM
    from scheduler import get_cosine_annealing_schedule_with_warmup
    from transformer_engine.pytorch.optimizers import FusedAdam

    torch.manual_seed(seed)
    config_path = Path(__file__).parent.parent / "model_configs" / "mixtral_tiny"
    config = NVMixtralConfig.from_pretrained(
        str(config_path),
        expert_parallel_size=EP_SIZE,
        expert_ffn_mode="fused_grouped_mlp",
        torch_dtype=torch.bfloat16,
    )
    model = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device=device)
    mesh = build_mesh_and_wrap(model, dp_size=DP_SIZE, ep_size=EP_SIZE)
    ep_mesh = mesh["ep"]
    dp_process_group = mesh["dp"].get_group()

    optimizer = FusedAdam(model.parameters(), lr=1e-3, master_weights=True)
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, num_warmup_steps=0, num_decay_steps=100)
    return model, optimizer, scheduler, mesh, ep_mesh, dp_process_group, config


def _assert_fsdp2_dp_sharding(model: torch.nn.Module, dp_size: int) -> None:
    from torch.distributed.tensor import DTensor, Shard

    lm_head_weight = model.lm_head.weight
    assert isinstance(lm_head_weight, DTensor), "lm_head weight must be FSDP2-sharded (DTensor)"
    assert any(isinstance(p, Shard) for p in lm_head_weight.placements), (
        "lm_head must have a Shard placement on the dp mesh"
    )
    assert lm_head_weight.device_mesh.size() == dp_size, (
        f"lm_head DTensor mesh size {lm_head_weight.device_mesh.size()} != dp_size={dp_size}"
    )

    local_weight = lm_head_weight.to_local()
    global_weight = lm_head_weight.full_tensor()
    assert local_weight.shape != global_weight.shape, (
        f"lm_head local shape {local_weight.shape} must differ from global {global_weight.shape} "
        f"when dp_size={dp_size}"
    )
    assert local_weight.numel() < global_weight.numel(), "lm_head must be genuinely sharded across dp"

    layer = model.model.layers[0]
    expert_weight = getattr(layer.mlp.experts_gate_up, "weight0")
    assert isinstance(expert_weight, DTensor), "expert weight0 must be FSDP2-sharded (DTensor) when dp>1"
    assert any(isinstance(p, Shard) for p in expert_weight.placements), (
        "expert weight0 must have a Shard placement on the dp mesh"
    )
    assert expert_weight.device_mesh.size() == dp_size, (
        f"expert DTensor mesh size {expert_weight.device_mesh.size()} != dp_size={dp_size}"
    )
    assert expert_weight.to_local().numel() < expert_weight.full_tensor().numel(), (
        "expert weight0 must be genuinely sharded across dp"
    )


def _expert_weight_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snap: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if _is_expert_key(name):
            snap[name] = _dequantized_tensor(param).cpu()
    return snap


def _expert_master_snapshot(model: torch.nn.Module, optimizer) -> dict[str, torch.Tensor]:
    snap: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not _is_expert_key(name):
            continue
        if param not in optimizer.state:
            continue
        master = optimizer.state[param].get("master_param")
        if master is not None:
            if isinstance(master, DTensor):
                master = master.to_local()
            snap[name] = master.detach().clone().cpu()
    return snap


def _run_2d_mesh_worker(tmp_dir: str) -> None:
    from checkpoint import load_checkpoint, save_checkpoint
    from distributed_config import DistributedConfig
    from distributed_setup import all_reduce_dense_grads_over_ep, clip_grad_norm_mixed
    from torch.distributed.tensor import DTensor

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    assert dist_config.world_size == DP_SIZE * EP_SIZE

    model, optimizer, scheduler, mesh, ep_mesh, dp_process_group, config = _build_model_and_optimizer(
        device, seed=TRAIN_SEED
    )

    _assert_fsdp2_dp_sharding(model, dp_size=DP_SIZE)

    batch = _get_dummy_batch(config.vocab_size, device)
    model.train()
    optimizer.zero_grad()
    loss = model(**batch).loss
    assert torch.isfinite(loss), f"forward loss not finite: {loss}"
    loss.backward()

    # EP+FSDP2: dense grads are FSDP-reduced over dp but only replicated over ep; average over ep.
    all_reduce_dense_grads_over_ep(model, mesh["ep"].get_group())

    # Build an independent reference: experts are unique across ep and sharded across dp, while
    # dense grads are sharded across dp but duplicated across ep after the average above.
    local_expert_sq = torch.zeros((), device=device, dtype=torch.float32)
    local_dense_sq = torch.zeros((), device=device, dtype=torch.float32)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        if _is_expert_key(name):
            local_expert_sq += grad.float().square().sum()
        else:
            local_dense_sq += grad.float().square().sum()
    torch.distributed.all_reduce(local_expert_sq)
    torch.distributed.all_reduce(local_dense_sq)
    expected_norm = torch.sqrt(local_expert_sq + local_dense_sq / EP_SIZE)

    grad_norm = clip_grad_norm_mixed(
        model,
        max_norm=1.0,
        ep_group=mesh["ep"].get_group(),
        dp_group=mesh["dp"].get_group(),
    )
    assert torch.isfinite(grad_norm), f"grad norm not finite: {grad_norm}"
    torch.testing.assert_close(grad_norm, expected_norm)

    optimizer.step()
    scheduler.step()

    for name, param in model.named_parameters():
        if name.endswith("_extra_state"):
            continue
        local = param.to_local() if isinstance(param, DTensor) else _dequantized_tensor(param)
        assert torch.isfinite(local).all(), f"param {name} not finite after optimizer step"

    ref_lm_head_full = model.lm_head.weight.full_tensor().detach().cpu()
    ref_expert = _expert_weight_snapshot(model)
    ref_master = _expert_master_snapshot(model, optimizer)

    ckpt_path = os.path.join(tmp_dir, "checkpoints")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ckpt_path=ckpt_path,
        step=1,
        epoch=0,
        dist_config=dist_config,
        ep_mesh=ep_mesh,
        dp_process_group=dp_process_group,
        max_checkpoints=2,
        async_save=False,
    )
    torch.distributed.barrier()

    model_new, optimizer_new, scheduler_new, _, _, _, _ = _build_model_and_optimizer(device, seed=LOAD_SEED)
    model_new, optimizer_new, scheduler_new, _, step, epoch = load_checkpoint(
        model=model_new,
        optimizer=optimizer_new,
        scheduler=scheduler_new,
        ckpt_path=ckpt_path,
        dist_config=dist_config,
        ep_mesh=ep_mesh,
        dp_process_group=dp_process_group,
    )
    assert step == 2
    assert epoch == 0

    loaded_lm_head_full = model_new.lm_head.weight.full_tensor().detach().cpu()
    torch.testing.assert_close(
        loaded_lm_head_full,
        ref_lm_head_full,
        rtol=0,
        atol=0,
        msg="FSDP2-sharded lm_head full_tensor mismatch after checkpoint roundtrip",
    )

    loaded_expert = _expert_weight_snapshot(model_new)
    for key, ref_w in ref_expert.items():
        assert key in loaded_expert, f"missing expert key {key}"
        torch.testing.assert_close(
            loaded_expert[key],
            ref_w,
            rtol=0,
            atol=0,
            msg=f"expert weight mismatch {key}",
        )

    loaded_master = _expert_master_snapshot(model_new, optimizer_new)
    for key, ref_w in ref_master.items():
        assert key in loaded_master, f"missing master_param key {key}"
        torch.testing.assert_close(
            loaded_master[key],
            ref_w,
            rtol=0,
            atol=0,
            msg=f"master_param mismatch {key}",
        )

    if dist_config.is_main_process():
        print("EP=2 x FSDP2 dp=2 2D mesh test PASSED")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    _run_2d_mesh_worker(sys.argv[1])
