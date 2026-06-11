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

"""Tests for FSDP2 + Expert Parallelism (EP) in the Mixtral MoE model.

Verifies that FSDP2 and EP can be composed together:
- FSDP=2, EP=1 (2 GPUs): Data-parallel sharding, all experts on each rank.
- FSDP=1, EP=2 (2 GPUs): Expert-parallel training, no data parallelism.
- FSDP=2, EP=2 (4 GPUs): Both data and expert parallelism (skipped on 2-GPU machines).
"""

import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import torch
from distributed_helpers import DistributedConfig, create_small_mixtral_config, get_dummy_batch

from modeling_mixtral_te import NVMixtralForCausalLM


requires_2_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

requires_4_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="Test requires at least 4 GPUs",
)


def _distribute_state_dict(full_state_dict: dict, model: torch.nn.Module, device: torch.device) -> dict:
    """Distribute a full (EP=1) state dict to match a model's DTensor sharding.

    After calling ``set_ep_groups``, expert weight parameters become DTensors with
    ``Shard(0)`` placement.  This function uses ``distribute_tensor`` to automatically
    shard full expert weights according to those annotations, avoiding manual slicing.

    Args:
        full_state_dict: Complete state dict from an EP=1 model (plain tensors).
        model: Target EP model whose expert parameters are already DTensors.
        device: Device to move source tensors to before distributing.
    """
    from torch.distributed.tensor import DTensor, distribute_tensor

    distributed_state: dict = {}
    # model.state_dict() filters _extra_state keys via the NVMixtralPreTrainedModel
    # override, so use nn.Module.state_dict to get the unfiltered dict that includes
    # TransformerEngine _extra_state entries required by load_state_dict(strict=True).
    for key, value in torch.nn.Module.state_dict(model).items():
        if key.endswith("_extra_state"):
            distributed_state[key] = value
        elif key not in full_state_dict:
            continue
        elif isinstance(value, DTensor):
            distributed_state[key] = distribute_tensor(
                full_state_dict[key].to(device),
                value.device_mesh,
                list(value.placements),
            )
        else:
            distributed_state[key] = full_state_dict[key]
    return distributed_state


def _train_step(model, batch):
    """Run a single forward + backward + optimizer step.

    Returns:
        Tuple of (loss value, dict of gradient norms, dict of weight change norms).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Snapshot weights before step
    pre_weights = {n: p.detach().clone() for n, p in model.named_parameters()}

    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            if hasattr(g, "full_tensor"):
                g = g.full_tensor()
            grad_norms[name] = g.detach().float().norm().item()

    optimizer.step()

    # Measure weight changes
    weight_changes = {}
    for name, param in model.named_parameters():
        pre = pre_weights[name]
        cur = param.detach()
        if hasattr(pre, "full_tensor"):
            pre = pre.full_tensor()
        if hasattr(cur, "full_tensor"):
            cur = cur.full_tensor()
        weight_changes[name] = (cur.float() - pre.float()).norm().item()

    return loss.detach().item(), grad_norms, weight_changes


# ---------------------------------------------------------------------------
# Pytest entry points — launch torchrun subprocesses
# ---------------------------------------------------------------------------


def _run_torchrun(test_fn_name: str, port: int, nproc: int = 2):
    """Run a named worker function via torchrun."""
    model_dir = str(Path(__file__).resolve().parent.parent)
    script = str(Path(__file__).resolve())
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=localhost:{port}",
        script,
        test_fn_name,
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        cwd=model_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"{test_fn_name} failed with exit code {result.returncode}")


@requires_2_gpus
def test_fsdp2_ep1(unused_tcp_port):
    """Test FSDP=2, EP=1: data-parallel training with all experts on each rank."""
    _run_torchrun("fsdp2_ep1", unused_tcp_port, nproc=2)


@requires_2_gpus
def test_fsdp1_ep2(unused_tcp_port):
    """Test FSDP=1, EP=2: expert-parallel training without data parallelism."""
    _run_torchrun("fsdp1_ep2", unused_tcp_port, nproc=2)


@requires_4_gpus
def test_fsdp2_ep2(unused_tcp_port):
    """Test FSDP=2, EP=2: combined data and expert parallelism."""
    _run_torchrun("fsdp2_ep2", unused_tcp_port, nproc=4)


# ---------------------------------------------------------------------------
# Distributed workers executed via torchrun
# ---------------------------------------------------------------------------


def _worker_fsdp2_ep1():
    """FSDP=2, EP=1: weights sharded by FSDP, all experts on each rank.

    Uses a 2D device mesh (dp=2, ep=1) so that DTensor multi-dimensional
    placement logic is exercised even though the EP dimension is trivial.

    1. Init distributed, create 2D device mesh with ep=1.
    2. Create model with EP=1, set EP groups on the trivial EP sub-mesh.
    3. Wrap with FSDP2 on the DP sub-mesh.
    4. Run one training step, verify loss/gradients are finite and weights update.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)

    ep_size = 1
    dp_size = dist_config.world_size
    device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, ep_size), mesh_dim_names=("dp", "ep"))

    config = create_small_mixtral_config(expert_parallel_size=ep_size)
    torch.manual_seed(0)
    model = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device=device)

    # EP setup with trivial (size-1) EP sub-mesh
    ep_mesh = device_mesh["ep"]
    ep_group = ep_mesh.get_group()
    model.model.set_ep_groups(ep_group, ep_mesh)

    # FSDP2 wrapping on DP sub-mesh
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"])
    fully_shard(model, mesh=device_mesh["dp"])

    model.train()
    batch = get_dummy_batch(config.vocab_size, device=str(device))

    loss_val, grad_norms, weight_changes = _train_step(model, batch)

    assert torch.isfinite(torch.tensor(loss_val)), f"Loss is not finite: {loss_val}"
    assert len(grad_norms) > 0, "No gradients computed"
    for name, gnorm in grad_norms.items():
        assert torch.isfinite(torch.tensor(gnorm)), f"Gradient for {name} is not finite: {gnorm}"
    assert any(wc > 0 for wc in weight_changes.values()), "No weights updated after optimizer step"

    torch.distributed.destroy_process_group()


def _worker_fsdp1_ep2():
    """FSDP=1, EP=2: experts sharded across ranks, trivial data parallelism.

    Uses a 2D device mesh (dp=1, ep=2) so that DTensor multi-dimensional
    placement logic is exercised even though the DP dimension is trivial.

    1. Init distributed, create 2D device mesh with dp=1.
    2. Create full EP=1 model for reference weights.
    3. Create EP=2 model, set EP groups (DTensor annotations), load via distribute_tensor.
    4. Wrap with FSDP2 on the trivial DP sub-mesh.
    5. Run one training step, verify loss/gradients are finite and weights update.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)

    ep_size = dist_config.world_size
    dp_size = 1
    device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, ep_size), mesh_dim_names=("dp", "ep"))

    ep_mesh = device_mesh["ep"]
    ep_group = ep_mesh.get_group()

    # Get reference weights from a full EP=1 model
    config_full = create_small_mixtral_config(expert_parallel_size=1)
    torch.manual_seed(0)
    full_model = NVMixtralForCausalLM(config_full).to(dtype=torch.bfloat16, device="cpu")
    full_state_dict = {k: v.clone() for k, v in full_model.state_dict().items()}
    del full_model

    # Create EP=2 model, set EP groups to create DTensor annotations, then load weights
    config_ep = create_small_mixtral_config(expert_parallel_size=ep_size)
    torch.manual_seed(0)
    model = NVMixtralForCausalLM(config_ep).to(dtype=torch.bfloat16, device=device)

    # EP setup on EP sub-mesh first (creates DTensor annotations on expert weights)
    model.model.set_ep_groups(ep_group, ep_mesh)

    # Load EP=1 weights — distribute_tensor handles expert sharding automatically
    distributed_state = _distribute_state_dict(full_state_dict, model, device)
    model.load_state_dict(distributed_state, strict=True)

    # FSDP2 wrapping on trivial (size-1) DP sub-mesh
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"])
    fully_shard(model, mesh=device_mesh["dp"])

    model.train()
    batch = get_dummy_batch(config_ep.vocab_size, device=str(device))

    loss_val, grad_norms, weight_changes = _train_step(model, batch)

    assert torch.isfinite(torch.tensor(loss_val)), f"Loss is not finite: {loss_val}"
    assert len(grad_norms) > 0, "No gradients computed"
    for name, gnorm in grad_norms.items():
        assert torch.isfinite(torch.tensor(gnorm)), f"Gradient for {name} is not finite: {gnorm}"
    assert any(wc > 0 for wc in weight_changes.values()), "No weights updated after optimizer step"

    torch.distributed.destroy_process_group()


def _worker_fsdp2_ep2():
    """FSDP=2, EP=2: both FSDP and EP active (requires 4 GPUs).

    1. Init distributed, create 2D device mesh (dp=2, ep=2).
    2. Create full EP=1 model for reference weights.
    3. Create EP=2 model, set EP groups (DTensor annotations), load via distribute_tensor.
    4. Wrap with FSDP2 on DP sub-mesh.
    5. Run one training step, verify loss/gradients are finite and weights update.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)

    ep_size = 2
    dp_size = dist_config.world_size // ep_size
    assert dp_size >= 2, f"Need at least 4 GPUs for FSDP=2 EP=2, got {dist_config.world_size}"

    device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    ep_mesh = device_mesh["ep"]
    ep_group = ep_mesh.get_group()

    # Get reference weights from a full EP=1 model
    config_full = create_small_mixtral_config(expert_parallel_size=1)
    torch.manual_seed(0)
    full_model = NVMixtralForCausalLM(config_full).to(dtype=torch.bfloat16, device="cpu")
    full_state_dict = {k: v.clone() for k, v in full_model.state_dict().items()}
    del full_model

    # Create EP=2 model, set EP groups to create DTensor annotations, then load weights
    config_ep = create_small_mixtral_config(expert_parallel_size=ep_size)
    torch.manual_seed(0)
    model = NVMixtralForCausalLM(config_ep).to(dtype=torch.bfloat16, device=device)

    # EP setup first: wrap expert weights as DTensors on EP sub-mesh
    model.model.set_ep_groups(ep_group, ep_mesh)

    # Load EP=1 weights — distribute_tensor handles expert sharding automatically
    distributed_state = _distribute_state_dict(full_state_dict, model, device)
    model.load_state_dict(distributed_state, strict=True)

    # FSDP2 wrapping on DP sub-mesh
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"])
    fully_shard(model, mesh=device_mesh["dp"])

    model.train()
    batch = get_dummy_batch(config_ep.vocab_size, device=str(device))

    loss_val, grad_norms, weight_changes = _train_step(model, batch)

    assert torch.isfinite(torch.tensor(loss_val)), f"Loss is not finite: {loss_val}"
    assert len(grad_norms) > 0, "No gradients computed"
    for name, gnorm in grad_norms.items():
        assert torch.isfinite(torch.tensor(gnorm)), f"Gradient for {name} is not finite: {gnorm}"
    assert any(wc > 0 for wc in weight_changes.values()), "No weights updated after optimizer step"

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_name = sys.argv[1]

    workers = {
        "fsdp2_ep1": _worker_fsdp2_ep1,
        "fsdp1_ep2": _worker_fsdp1_ep2,
        "fsdp2_ep2": _worker_fsdp2_ep2,
    }
    workers[test_name]()
