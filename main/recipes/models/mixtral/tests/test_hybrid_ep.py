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

"""Tests for the DeepEP-backed FusedTokenRouter in the Mixtral MoE model.

Verifies that FusedTokenRouter produces the same logits and loss as the
AllToAllTokenDispatcher when running with EP=2. Also verifies that the
backward pass produces matching gradients.
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


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


def _deep_ep_and_triton_available() -> bool:
    """Check if the deep_ep package is importable."""
    try:
        import deep_ep  # noqa: F401
        import triton
        import triton.language  # noqa: F401

        return True
    except ImportError:
        return False


def _cuda_peer_access_available() -> bool:
    """Check if CUDA peer access (NVLink IPC) is supported between GPU 0 and GPU 1."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    return torch.cuda.can_device_access_peer(0, 1)


requires_deep_ep = pytest.mark.skipif(
    not _deep_ep_and_triton_available(), reason="deep_ep and/or triton not available"
)

requires_peer_access = pytest.mark.skipif(
    not _cuda_peer_access_available(),
    reason="CUDA peer access (NVLink IPC) not supported between GPUs",
)


def _shard_expert_weights(full_state_dict: dict, ep_rank: int, ep_size: int, num_experts: int) -> dict:
    """Shard stacked expert weights from a full (EP=1) state dict for a given EP rank.

    Expert weight keys are ``...experts_gate_up_weight`` and ``...experts_down_weight``
    with shape ``[num_experts, ...]``. For EP, each rank keeps only its local slice.
    """
    experts_per_rank = num_experts // ep_size
    start_expert = ep_rank * experts_per_rank
    end_expert = start_expert + experts_per_rank

    new_state_dict = {}
    for key, value in full_state_dict.items():
        if key.endswith("experts_gate_up_weight") or key.endswith("experts_down_weight"):
            new_state_dict[key] = value[start_expert:end_expert]
        else:
            new_state_dict[key] = value

    return new_state_dict


# ---------------------------------------------------------------------------
# Pytest entry points — launch torchrun subprocesses
# ---------------------------------------------------------------------------


def _run_torchrun(test_name: str, port: int):
    """Run the equivalence test worker via torchrun with 2 GPUs."""
    model_dir = str(Path(__file__).resolve().parent.parent)
    script = str(Path(__file__).resolve())
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=localhost:{port}",
        script,
        test_name,
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
        pytest.fail(f"FusedTokenRouter {test_name} test failed with exit code {result.returncode}")


@requires_multi_gpu
@requires_deep_ep
@requires_peer_access
def test_fused_router_matches_alltoall(unused_tcp_port):
    """Test that FusedTokenRouter dispatcher matches AllToAll dispatcher at EP=2."""
    _run_torchrun("forward", unused_tcp_port)


@requires_multi_gpu
@requires_deep_ep
@requires_peer_access
def test_fused_router_backward(unused_tcp_port):
    """Test that backward pass with FusedTokenRouter dispatcher matches AllToAll at EP=2."""
    _run_torchrun("backward", unused_tcp_port)


# ---------------------------------------------------------------------------
# Distributed worker executed via torchrun
# ---------------------------------------------------------------------------


def _run_equivalence_test():
    """Main worker function for the FusedTokenRouter forward equivalence test.

    1. Init distributed with 2 GPUs.
    2. Create EP=1 model for reference weights.
    3. Create EP=2 model with AllToAll dispatcher -> reference logits/loss.
    4. Create EP=2 model with FusedTokenRouter dispatcher -> test logits/loss.
    5. Compare results.
    """
    from torch.distributed.tensor.device_mesh import DeviceMesh

    from fused_token_router import FusedTokenRouter

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    ep_rank = dist_config.rank
    ep_size = dist_config.world_size

    # --- Phase 1: Create EP=1 model for reference weights ---
    config_ep1 = create_small_mixtral_config(expert_parallel_size=1)
    torch.manual_seed(0)
    model_ep1 = NVMixtralForCausalLM(config_ep1).to(dtype=torch.bfloat16, device=device)
    full_state_dict = {k: v.clone().cpu() for k, v in model_ep1.state_dict().items()}
    del model_ep1
    torch.cuda.empty_cache()

    batch = get_dummy_batch(config_ep1.vocab_size, seq_len=32, batch_size=2, device=device)
    num_experts = config_ep1.num_local_experts

    # --- Phase 2: EP=2 + AllToAll dispatcher -> reference logits/loss ---
    config_ep2 = create_small_mixtral_config(expert_parallel_size=ep_size)
    torch.manual_seed(0)
    model_alltoall = NVMixtralForCausalLM(config_ep2).to(dtype=torch.bfloat16, device=device)

    sharded_state = _shard_expert_weights(full_state_dict, ep_rank, ep_size, num_experts)
    model_alltoall.load_state_dict(sharded_state, strict=False)
    model_alltoall.eval()

    ep_mesh = DeviceMesh("cuda", list(range(ep_size)))
    ep_group = ep_mesh.get_group()
    model_alltoall.model.set_ep_groups(ep_group, ep_mesh)

    with torch.no_grad():
        outputs_ref = model_alltoall(**batch)
    logits_ref = outputs_ref.logits.detach().clone().cpu()
    loss_ref = outputs_ref.loss.detach().clone().cpu()

    del model_alltoall, outputs_ref
    torch.cuda.empty_cache()

    # --- Phase 3: EP=2 + FusedTokenRouter dispatcher -> test logits/loss ---
    num_local_experts = num_experts // ep_size
    hidden_size = config_ep2.hidden_size

    dispatcher = FusedTokenRouter(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        ep_size=ep_size,
    )

    torch.manual_seed(0)
    model_fused = NVMixtralForCausalLM(config_ep2, dispatcher=dispatcher).to(dtype=torch.bfloat16, device=device)
    model_fused.load_state_dict(sharded_state, strict=False)
    model_fused.eval()

    model_fused.model.set_ep_groups(ep_group, ep_mesh)

    with torch.no_grad():
        outputs_test = model_fused(**batch)
    logits_test = outputs_test.logits.detach().cpu()
    loss_test = outputs_test.loss.detach().cpu()

    # --- Phase 4: Compare on rank 0 ---
    if dist_config.is_main_process():
        torch.testing.assert_close(
            logits_test,
            logits_ref,
            atol=1e-2,
            rtol=1e-2,
            msg="FusedTokenRouter logits do not match AllToAll logits",
        )

        torch.testing.assert_close(
            loss_test,
            loss_ref,
            atol=1e-3,
            rtol=1e-3,
            msg="FusedTokenRouter loss does not match AllToAll loss",
        )

    torch.distributed.destroy_process_group()


def _run_backward_test():
    """Worker function for the FusedTokenRouter backward pass equivalence test.

    1. Init distributed with 2 GPUs.
    2. Create EP=1 model for reference weights.
    3. Create EP=2 model with AllToAll dispatcher -> reference loss + gradients.
    4. Create EP=2 model with FusedTokenRouter dispatcher -> test loss + gradients.
    5. Compare loss and parameter gradients.
    """
    from torch.distributed.tensor.device_mesh import DeviceMesh

    from fused_token_router import FusedTokenRouter

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    ep_rank = dist_config.rank
    ep_size = dist_config.world_size

    # --- Phase 1: Create EP=1 model for reference weights ---
    config_ep1 = create_small_mixtral_config(expert_parallel_size=1)
    torch.manual_seed(0)
    model_ep1 = NVMixtralForCausalLM(config_ep1).to(dtype=torch.bfloat16, device=device)
    full_state_dict = {k: v.clone().cpu() for k, v in model_ep1.state_dict().items()}
    del model_ep1
    torch.cuda.empty_cache()

    batch = get_dummy_batch(config_ep1.vocab_size, seq_len=32, batch_size=2, device=device)
    num_experts = config_ep1.num_local_experts

    # --- Phase 2: EP=2 + AllToAll dispatcher -> reference loss + gradients ---
    config_ep2 = create_small_mixtral_config(expert_parallel_size=ep_size)
    torch.manual_seed(0)
    model_alltoall = NVMixtralForCausalLM(config_ep2).to(dtype=torch.bfloat16, device=device)

    sharded_state = _shard_expert_weights(full_state_dict, ep_rank, ep_size, num_experts)
    model_alltoall.load_state_dict(sharded_state, strict=False)

    ep_mesh = DeviceMesh("cuda", list(range(ep_size)))
    ep_group = ep_mesh.get_group()
    model_alltoall.model.set_ep_groups(ep_group, ep_mesh)

    outputs_ref = model_alltoall(**batch)
    loss_ref = outputs_ref.loss
    loss_ref.backward()

    ref_grads = {
        name: p.grad.detach().clone().cpu() for name, p in model_alltoall.named_parameters() if p.grad is not None
    }
    loss_ref_cpu = loss_ref.detach().cpu()

    del model_alltoall, outputs_ref
    torch.cuda.empty_cache()

    # --- Phase 3: EP=2 + FusedTokenRouter dispatcher -> test loss + gradients ---
    num_local_experts = num_experts // ep_size
    hidden_size = config_ep2.hidden_size

    dispatcher = FusedTokenRouter(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        ep_size=ep_size,
    )

    torch.manual_seed(0)
    model_fused = NVMixtralForCausalLM(config_ep2, dispatcher=dispatcher).to(dtype=torch.bfloat16, device=device)
    model_fused.load_state_dict(sharded_state, strict=False)

    model_fused.model.set_ep_groups(ep_group, ep_mesh)

    outputs_test = model_fused(**batch)
    loss_test = outputs_test.loss
    loss_test.backward()

    test_grads = {
        name: p.grad.detach().clone().cpu() for name, p in model_fused.named_parameters() if p.grad is not None
    }
    loss_test_cpu = loss_test.detach().cpu()

    # --- Phase 4: Compare on rank 0 ---
    if dist_config.is_main_process():
        torch.testing.assert_close(
            loss_test_cpu,
            loss_ref_cpu,
            atol=1e-3,
            rtol=1e-3,
            msg="FusedTokenRouter backward: loss does not match AllToAll loss",
        )

        assert len(ref_grads) > 0, "AllToAll model produced no gradients"
        assert len(test_grads) > 0, "FusedTokenRouter model produced no gradients"

        # Both AllToAll (via _DifferentiableAllToAll) and FusedTokenRouter provide
        # differentiable dispatch/combine, so all parameters should have gradients.
        assert ref_grads.keys() == test_grads.keys(), (
            f"Gradient key mismatch: "
            f"AllToAll-only={ref_grads.keys() - test_grads.keys()}, "
            f"Fused-only={test_grads.keys() - ref_grads.keys()}"
        )

        for name in ref_grads:
            torch.testing.assert_close(
                test_grads[name],
                ref_grads[name],
                atol=3e-2,
                rtol=3e-2,
                msg=f"FusedTokenRouter gradient mismatch for {name}",
            )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_name = sys.argv[1]
    if test_name == "backward":
        _run_backward_test()
    else:
        _run_equivalence_test()
