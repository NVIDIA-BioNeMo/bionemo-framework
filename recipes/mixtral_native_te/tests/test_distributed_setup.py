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

"""Tests for 2D (dp, ep) mesh setup with EP groups before selective FSDP2 wrapping."""

import os
import subprocess
from pathlib import Path

import pytest
import torch
from distributed_config import DistributedConfig
from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

requires_sm100 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 10),
    reason="fused_grouped_mlp expert path requires compute capability 10.x; 12.0 support is pending",
)


def _run_torchrun() -> None:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
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
        pytest.fail(f"distributed_setup test failed with exit code {result.returncode}")


@requires_multi_gpu
@requires_sm100
def test_build_mesh_and_wrap_ep_fsdp2():
    """EP-only (dp=1, ep=2): nothing FSDP-wrapped, all params plain, forward runs."""
    _run_torchrun()


def _get_dummy_batch(vocab_size: int, device: torch.device):
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (2, 32), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _run_distributed_setup_worker() -> None:
    from distributed_setup import build_mesh_and_wrap
    from torch.distributed.tensor import DTensor

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)

    config_path = Path(__file__).parent.parent / "model_configs" / "mixtral_tiny"
    config = NVMixtralConfig.from_pretrained(
        str(config_path),
        expert_parallel_size=2,
        expert_ffn_mode="fused_grouped_mlp",
        torch_dtype=torch.bfloat16,
    )

    torch.manual_seed(0)
    model = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device=device)

    mesh = build_mesh_and_wrap(model, dp_size=1, ep_size=2)

    # EP-only (dp=1): nothing is FSDP2-wrapped. Experts are EP-local plain weight{i} and dense
    # params are plain/replicated (kept in sync at train time via an ep all-reduce of dense grads).
    layer = model.model.layers[0]
    expert_weight = getattr(layer.mlp.experts_gate_up, "weight0")
    assert not isinstance(expert_weight, DTensor), "expert weight0 must remain a local tensor (EP-only)"

    lm_head_weight = model.lm_head.weight
    assert not isinstance(lm_head_weight, DTensor), "lm_head must stay a plain tensor at dp=1 (no FSDP)"

    attn_param = next(p for n, p in model.named_parameters() if "self_attention" in n and "experts" not in n)
    assert not isinstance(attn_param, DTensor), "attention param must stay plain at dp=1 (no FSDP)"

    batch = _get_dummy_batch(config.vocab_size, device)
    model.eval()
    with torch.no_grad():
        outputs = model(**batch)

    assert outputs.logits.shape == (2, 32, config.vocab_size)
    assert torch.isfinite(outputs.logits).all(), "forward logits must be finite"

    if dist_config.is_main_process():
        print(
            "distributed_setup test PASSED: dp=1 all params plain (experts EP-local), finite logits, "
            f"mesh shape={mesh.shape}"
        )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    _run_distributed_setup_worker()
