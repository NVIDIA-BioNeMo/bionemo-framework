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

"""Tests for FSDP2 non-expert DCP + consolidated expert/optimizer checkpointing."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

requires_sm100 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 10),
    reason="fused_grouped_mlp expert path requires compute capability 10.x; 12.0 support is pending",
)


def _run_torchrun(tmp_dir: str) -> None:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        tmp_dir,
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
        pytest.fail(f"checkpoint test failed with exit code {result.returncode}")


@requires_multi_gpu
@requires_sm100
def test_checkpoint_save_load_roundtrip(tmp_path):
    """Save/load checkpoint preserves non-expert, expert, and optimizer master state."""
    _run_torchrun(str(tmp_path))


def _is_expert_key(name: str) -> bool:
    return ".experts_gate_up." in name or ".experts_down." in name


def _param_local_tensor(param: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    if isinstance(param, DTensor):
        return param.to_local().detach().clone()
    if hasattr(param, "dequantize"):
        return param.dequantize().detach().clone()
    return param.detach().clone()


def _non_expert_param_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snap: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if _is_expert_key(name) or name.endswith("_extra_state"):
            continue
        snap[name] = _param_local_tensor(param).cpu()
    return snap


def _expert_weight_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snap: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if _is_expert_key(name):
            snap[name] = _param_local_tensor(param).cpu()
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
            snap[name] = master.detach().clone().cpu()
    return snap


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
        expert_parallel_size=2,
        expert_ffn_mode="fused_grouped_mlp",
        torch_dtype=torch.bfloat16,
    )
    model = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device=device)
    mesh = build_mesh_and_wrap(model, dp_size=1, ep_size=2)
    ep_mesh = mesh["ep"]
    dp_process_group = mesh["dp"].get_group()

    optimizer = FusedAdam(model.parameters(), lr=1e-3, master_weights=True)
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, num_warmup_steps=0, num_decay_steps=100)
    return model, optimizer, scheduler, mesh, ep_mesh, dp_process_group, config


def _run_checkpoint_worker(tmp_dir: str) -> None:
    from checkpoint import load_checkpoint, save_checkpoint
    from distributed_config import DistributedConfig

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)

    model, optimizer, scheduler, _mesh, ep_mesh, dp_process_group, config = _build_model_and_optimizer(device, seed=0)

    batch = _get_dummy_batch(config.vocab_size, device)
    model.train()
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    scheduler.step()

    ref_non_expert = _non_expert_param_snapshot(model)
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

    model_new, optimizer_new, scheduler_new, _, _, _, _ = _build_model_and_optimizer(device, seed=999)
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

    loaded_non_expert = _non_expert_param_snapshot(model_new)
    loaded_expert = _expert_weight_snapshot(model_new)
    loaded_master = _expert_master_snapshot(model_new, optimizer_new)

    for key, ref in ref_non_expert.items():
        assert key in loaded_non_expert, f"missing non-expert key {key}"
        torch.testing.assert_close(
            loaded_non_expert[key],
            ref,
            rtol=0,
            atol=0,
            msg=f"non-expert mismatch {key}",
        )

    for key, ref in ref_expert.items():
        assert key in loaded_expert, f"missing expert key {key}"
        torch.testing.assert_close(
            loaded_expert[key],
            ref,
            rtol=0,
            atol=0,
            msg=f"expert weight mismatch {key}",
        )

    for key, ref in ref_master.items():
        assert key in loaded_master, f"missing master_param key {key}"
        torch.testing.assert_close(
            loaded_master[key],
            ref,
            rtol=0,
            atol=0,
            msg=f"master_param mismatch {key}",
        )

    if dist_config.is_main_process():
        print("checkpoint save/load roundtrip PASSED")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    _run_checkpoint_worker(sys.argv[1])
