# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import pickle
import subprocess

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support


def requires_fp8(func):
    """Decorator to skip tests that require FP8 support."""
    fp8_available, reason = check_fp8_support()
    return pytest.mark.skipif(not fp8_available, reason=f"FP8 is not supported on this GPU: {reason}")(func)


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


@pytest.mark.parametrize(
    "strategy", ["ddp", "fsdp2", pytest.param("mfsdp", marks=pytest.mark.xfail(reason="BIONEMO-2999"))]
)
@requires_fp8
def test_single_process_attaches_correct_fp8_recipe(strategy, unused_tcp_port):
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=localhost:{unused_tcp_port}",
        os.path.relpath(__file__),
        "--strategy",
        strategy,
    ]

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


@pytest.mark.parametrize(
    "strategy", ["ddp", "fsdp2", pytest.param("mfsdp", marks=pytest.mark.xfail(reason="BIONEMO-2999"))]
)
@requires_fp8
@requires_multi_gpu
def test_multi_process_fp8_recipes_are_synced(strategy, unused_tcp_port):
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=localhost:{unused_tcp_port}",
        os.path.relpath(__file__),
        "--strategy",
        strategy,
    ]

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


if __name__ == "__main__":
    import argparse
    import enum
    import os
    import sys
    from dataclasses import dataclass, field
    from pathlib import Path

    # Ensure the model directory is on sys.path for bare module imports.
    sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

    import torch.distributed as dist
    import transformer_engine.pytorch
    from megatron_fsdp.fully_shard import fully_shard as megatron_fsdp_fully_shard
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.optim import AdamW
    from transformer_engine.pytorch.fp8 import DelayedScaling, Format

    from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM

    def recursive_assert(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            assert a.keys() == b.keys(), f"Dictionary keys mismatch: {a.keys()} != {b.keys()} at {path}"
            for k in a:
                recursive_assert(a[k], b[k], path=f"{path}.{k}")
        elif isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b), f"List lengths mismatch: {len(a)} != {len(b)} at {path}"
            for i in range(len(a)):
                recursive_assert(a[i], b[i], path=f"{path}.{i}")
        elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            torch.testing.assert_close(a, b, msg=f"Tensor mismatch at {path}")
        else:
            assert a == b, f"Value mismatch at {path}: {a} != {b}"

    class Strategy(enum.StrEnum):
        DDP = "ddp"
        FSDP2 = "fsdp2"
        MFSDP = "mfsdp"

    @dataclass
    class DistributedConfig:
        """Class to track distributed ranks."""

        rank: int = field(default_factory=dist.get_rank)
        local_rank: int = field(default_factory=lambda: int(os.environ["LOCAL_RANK"]))
        world_size: int = field(default_factory=dist.get_world_size)

        def is_main_process(self) -> bool:
            """This is the global rank 0 process, to be used for wandb logging, etc."""
            return self.rank == 0

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", type=Strategy, default=Strategy.DDP, choices=[Strategy.FSDP2, Strategy.MFSDP, Strategy.DDP]
    )
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    dist_config = DistributedConfig()
    torch.cuda.set_device(dist_config.local_rank)
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size, 1),
        mesh_dim_names=("dp", "tp"),
    )
    device = f"cuda:{dist_config.local_rank}"

    config = NVEsmConfig.from_pretrained("facebook/esm2_t6_8M_UR50D", dtype=torch.bfloat16, revision="c731040f")
    model = NVEsmForMaskedLM(config)

    if args.strategy is Strategy.FSDP2:
        for layer in model.model.encoder.layers:
            fully_shard(layer, mesh=device_mesh["dp"])
        fully_shard(model, mesh=device_mesh["dp"])
        model.to(device)

    elif args.strategy is Strategy.DDP:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_config.local_rank],
            output_device=dist_config.local_rank,
            device_mesh=device_mesh["dp"],
        )

    optimizer = AdamW(model.parameters())

    if args.strategy is Strategy.MFSDP:
        model, optimizer = megatron_fsdp_fully_shard(
            module=model,
            optimizer=optimizer,
            fsdp_unit_modules=[
                transformer_engine.pytorch.TransformerLayer,
                transformer_engine.pytorch.LayerNorm,
                transformer_engine.pytorch.LayerNormLinear,
            ],
            device_mesh=device_mesh,
            dp_shard_dim="dp",
            tp_dim="tp",
        )

    model.train()

    generator = torch.Generator()
    generator.manual_seed(torch.distributed.get_rank())

    fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_compute_algo="max", amax_history_len=10)

    for _ in range(3):
        input_data = {
            "input_ids": torch.randint(0, config.vocab_size, (1, 32), generator=generator),
            "labels": torch.randint(0, config.vocab_size, (1, 32), generator=generator),
            "attention_mask": torch.ones(1, 32),
        }
        input_data = {k: v.to(torch.cuda.current_device()) for k, v in input_data.items()}

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with transformer_engine.pytorch.autocast(enabled=True, recipe=fp8_recipe):
                outputs = model(**input_data)

        outputs.loss.backward()

    # Access FP8 extra states directly from modules instead of state_dict()
    # since state_dict() now filters them out for HuggingFace compatibility
    fp8_extra_states = {}
    for name, module in model.named_modules():
        if hasattr(module, "_extra_state") and callable(module._extra_state):
            extra_state = module._extra_state()
            if extra_state is not None and len(extra_state) > 0:
                fp8_extra_states[f"{name}._extra_state"] = extra_state

    # lm_head.dense and lm_head.decoder are BF16, not FP8, so exclude them from FP8 checks
    fp8_extra_states = {key: val for key, val in fp8_extra_states.items() if "lm_head." not in key}

    # 2 ranks, test to ensure that both ranks have the same FP8 extra states
    if torch.distributed.get_world_size() == 2:
        outputs_list = [None] * torch.distributed.get_world_size() if torch.distributed.get_rank() == 0 else None
        torch.distributed.gather_object(fp8_extra_states, outputs_list, dst=0)
        if torch.distributed.get_rank() == 0:
            assert outputs_list is not None

            for key in outputs_list[0]:
                state_1 = outputs_list[0][key]
                state_2 = outputs_list[1][key]
                assert len(state_1) > 0, f"No FP8 extra states for {key}, rank 0"
                assert len(state_2) > 0, f"No FP8 extra states for {key}, rank 1"
                dict_1 = pickle.loads(state_1.detach().numpy(force=True).tobytes())
                dict_2 = pickle.loads(state_2.detach().numpy(force=True).tobytes())
                recursive_assert(dict_1, dict_2)

    # One rank, test to ensure the correct FP8 extra states are saved
    if torch.distributed.get_world_size() == 1:
        for key, val in fp8_extra_states.items():
            assert len(val) > 0, f"No FP8 extra states for {key}"
            fp8_meta_dict = pickle.loads(val.detach().numpy(force=True).tobytes())
            assert fp8_meta_dict["recipe"] == fp8_recipe, f"Recipe mismatch for {key}"

    torch.distributed.destroy_process_group()
