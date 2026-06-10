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

"""Tests for DCP (Distributed Checkpoint) with expert parallelism in the Mixtral MoE model.

Verifies:
1. DCP round-trip: save EP=2 model, load into fresh EP=2 model, weights and logits match.
2. Full gather: save_final_model_ep gathers all experts into a single safetensors file
   that can be loaded by an EP=1 model and produces matching logits.
"""

import os
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


# ---------------------------------------------------------------------------
# Pytest entry points — launch torchrun subprocesses
# ---------------------------------------------------------------------------


def _run_torchrun(test_fn_name: str, port: int, tmp_dir: str):
    """Run a named worker function via torchrun with 2 GPUs."""
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=localhost:{port}",
        str(Path(__file__).resolve()),
        test_fn_name,
        tmp_dir,
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        cwd=str(Path(__file__).parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"{test_fn_name} failed with exit code {result.returncode}")


@requires_multi_gpu
def test_ep_dcp_roundtrip(unused_tcp_port, tmp_path):
    """Test DCP save/load round-trip with EP=2."""
    _run_torchrun("dcp_roundtrip", unused_tcp_port, str(tmp_path))


@requires_multi_gpu
def test_ep_save_final_model(unused_tcp_port, tmp_path):
    """Test gathering EP-sharded weights into a single safetensors file."""
    _run_torchrun("save_final_model", unused_tcp_port, str(tmp_path))


# ---------------------------------------------------------------------------
# Distributed workers executed via torchrun
# ---------------------------------------------------------------------------


def _init_distributed():
    """Initialize distributed environment and return config, device, mesh."""
    from torch.distributed.tensor.device_mesh import DeviceMesh

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    ep_size = dist_config.world_size
    ep_mesh = DeviceMesh("cuda", list(range(ep_size)))
    ep_group = ep_mesh.get_group()
    return dist_config, device, ep_mesh, ep_group


def _create_ep_model(device, ep_mesh, ep_group, ep_size, seed=0):
    """Create a small EP model with DTensor expert weights."""
    config = create_small_mixtral_config(expert_parallel_size=ep_size)
    torch.manual_seed(seed)
    model = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device=device)
    model.model.set_ep_groups(ep_group, ep_mesh)
    return model


def _worker_dcp_roundtrip(tmp_dir: str):
    """DCP round-trip test worker.

    1. Create EP=2 model with known seed, save reference state dict.
    2. Save via DCP.
    3. Create fresh EP=2 model (different seed), load via DCP.
    4. Compare all weights and forward outputs.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
    from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
    from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
    from torch.distributed.tensor import DTensor

    dist_config, device, ep_mesh, ep_group = _init_distributed()
    ep_size = dist_config.world_size
    ckpt_dir = os.path.join(tmp_dir, "dcp_ckpt")

    # --- Phase 1: Create reference model and save ---
    model_ref = _create_ep_model(device, ep_mesh, ep_group, ep_size, seed=0)
    model_ref.eval()

    # Save reference weights (local tensors on CPU for comparison)
    ref_state = {}
    for k, v in model_ref.state_dict().items():
        if isinstance(v, DTensor):
            ref_state[k] = v.to_local().detach().clone().cpu()
        else:
            ref_state[k] = v.detach().clone().cpu()

    # Verify DTensors appear in state dict
    has_dtensor = any(isinstance(v, DTensor) for v in model_ref.state_dict().values())
    assert has_dtensor, "Expected DTensor parameters in EP model state_dict"

    # DCP save
    model_state_dict = get_model_state_dict(model_ref)
    dcp_save({"model": model_state_dict}, checkpoint_id=ckpt_dir)

    # Reference forward
    batch = get_dummy_batch(model_ref.config.vocab_size, device=str(device))
    with torch.no_grad():
        ref_logits = model_ref(**batch).logits.detach().cpu()

    del model_ref
    torch.cuda.empty_cache()

    # --- Phase 2: Create fresh model and load from DCP ---
    model_new = _create_ep_model(device, ep_mesh, ep_group, ep_size, seed=999)
    model_new.eval()

    # DCP load
    new_state_dict = get_model_state_dict(model_new)
    dcp_load({"model": new_state_dict}, checkpoint_id=ckpt_dir)
    set_model_state_dict(model_new, new_state_dict, options=StateDictOptions(strict=False))

    # --- Phase 3: Compare weights ---
    for k, v in model_new.state_dict().items():
        local_v = v.to_local() if isinstance(v, DTensor) else v
        local_v = local_v.detach().cpu()
        assert k in ref_state, f"Key {k} missing from reference"
        torch.testing.assert_close(
            local_v,
            ref_state[k],
            atol=0,
            rtol=0,
            msg=f"Weight mismatch for {k}",
        )

    # --- Phase 4: Compare forward outputs ---
    with torch.no_grad():
        new_logits = model_new(**batch).logits.detach().cpu()

    if dist_config.is_main_process():
        torch.testing.assert_close(
            new_logits,
            ref_logits,
            atol=1e-3,
            rtol=1e-3,
            msg="DCP round-trip logits do not match",
        )
        print("DCP round-trip test PASSED")

    torch.distributed.destroy_process_group()


def _worker_save_final_model(tmp_dir: str):
    """Save final model test worker.

    1. Create EP=2 model with known seed, run forward, save reference logits.
    2. Call save_final_model_ep to gather all experts into safetensors.
    3. On rank 0: verify safetensors has full expert shapes.
    4. Load into EP=1 model, run forward, compare logits.
    """
    from safetensors.torch import load_file

    from modeling_mixtral_te import save_final_model_ep

    dist_config, device, ep_mesh, ep_group = _init_distributed()
    ep_size = dist_config.world_size
    save_dir = os.path.join(tmp_dir, "final_model")

    # --- Phase 1: Create EP=2 model, get reference logits ---
    model_ep2 = _create_ep_model(device, ep_mesh, ep_group, ep_size, seed=0)
    model_ep2.eval()

    batch = get_dummy_batch(model_ep2.config.vocab_size, device=str(device))
    with torch.no_grad():
        ref_logits = model_ep2(**batch).logits.detach().cpu()

    # --- Phase 2: Save gathered model ---
    save_final_model_ep(model_ep2, save_dir, dist_config)

    # Barrier to ensure all ranks finish saving before rank 0 checks
    torch.distributed.barrier()

    # --- Phase 3: Verify and load on rank 0 ---
    if dist_config.is_main_process():
        safetensors_path = os.path.join(save_dir, "model.safetensors")
        assert os.path.exists(safetensors_path), f"Expected {safetensors_path} to exist"

        gathered_state = load_file(safetensors_path)

        # Verify expert weights have full (ungathered) shape
        num_experts = model_ep2.config.num_local_experts
        for key in gathered_state:
            if key.endswith("experts_gate_up_weight") or key.endswith("experts_down_weight"):
                assert gathered_state[key].shape[0] == num_experts, (
                    f"Expected {num_experts} experts in {key}, got {gathered_state[key].shape[0]}"
                )

        # Load into EP=1 model and compare logits
        config_ep1 = create_small_mixtral_config(expert_parallel_size=1)
        torch.manual_seed(0)
        model_ep1 = NVMixtralForCausalLM(config_ep1).to(dtype=torch.bfloat16, device=device)
        model_ep1.load_state_dict(gathered_state, strict=False)
        model_ep1.eval()

        with torch.no_grad():
            ep1_logits = model_ep1(**batch).logits.detach().cpu()

        torch.testing.assert_close(
            ep1_logits,
            ref_logits,
            atol=1e-2,
            rtol=1e-2,
            msg="Gathered EP=1 logits do not match EP=2 reference",
        )
        print("Save final model test PASSED")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_name = sys.argv[1]
    tmp_dir = sys.argv[2]

    workers = {
        "dcp_roundtrip": _worker_dcp_roundtrip,
        "save_final_model": _worker_save_final_model,
    }
    workers[test_name](tmp_dir)
