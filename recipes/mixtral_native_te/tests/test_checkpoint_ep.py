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

"""EP checkpoint stop/go parity and EP=N→M reshard tests for the Mixtral recipe."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

requires_four_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="Test requires at least 4 GPUs",
)

requires_sm100 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 10),
    reason="fused_grouped_mlp expert path requires compute capability 10.x; 12.0 support is pending",
)

N_STEPS = 5
CHECKPOINT_STEP = 4
LOSS_RTOL = 0.05
LOSS_ATOL = 0.5
TEST_SEED = 42


def _init_determinism(seed: int = TEST_SEED) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _compose_config(recipe_path: Path, tmp_path: Path, overrides: list[str] | None = None):
    ckpt_dir = str(tmp_path / "ckpt")
    base = [
        f"checkpoint.ckpt_dir={ckpt_dir}",
        f"+wandb.dir={tmp_path}",
        "dataset.use_stateful_dataloader=true",
        "dataset.load_dataset_kwargs.streaming=false",
        "checkpoint.async_save=false",
        "wandb.mode=disabled",
        "logger.frequency=1",
    ]
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        return compose(config_name="L0_sanity", overrides=base + list(overrides or []))


def _run_torchrun(
    worker: str,
    tmp_dir: str,
    nproc: int,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(Path(__file__).resolve()),
        worker,
        tmp_dir,
        *extra_args,
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


def _capture_losses_with_main(cfg) -> tuple[float | None, list[tuple[int, float]]]:
    import train_fsdp2_ep

    captured: list[tuple[int, float]] = []
    original_log_step = train_fsdp2_ep.PerfLogger.log_step

    def _capture_log_step(self, step, grad_norm, lr):
        if (
            step > 0
            and step % self.logging_frequency == 0
            and self.grad_acc_step_count > 0
            and self._dist_config.is_main_process()
        ):
            avg_loss = (self.running_loss / self.grad_acc_step_count).item()
            captured.append((step, avg_loss))
        return original_log_step(self, step, grad_norm, lr)

    train_fsdp2_ep.PerfLogger.log_step = _capture_log_step
    try:
        min_loss = train_fsdp2_ep.main(cfg)
    finally:
        train_fsdp2_ep.PerfLogger.log_step = original_log_step
    return min_loss, captured


def _write_losses(path: Path, losses: list[tuple[int, float]]) -> None:
    path.write_text(json.dumps({"losses": losses}))


def _read_losses(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text())
    return {int(step): float(loss) for step, loss in payload["losses"]}


def _is_expert_key(name: str) -> bool:
    return ".experts_gate_up." in name or ".experts_down." in name


def _count_discrete_expert_weights(module: torch.nn.Module) -> int:
    count = 0
    while hasattr(module, f"weight{count}"):
        count += 1
    return count


def _dequantized_tensor(param: torch.Tensor) -> torch.Tensor:
    if hasattr(param, "dequantize"):
        return param.dequantize().detach()
    return param.detach()


def _gather_global_discrete_expert_weights(model: torch.nn.Module, ep_mesh) -> dict[str, torch.Tensor]:
    from torch.distributed.tensor import DTensor, Shard

    global_snap: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        for attr in ("experts_gate_up", "experts_down"):
            gl = getattr(module, attr, None)
            if gl is None or getattr(gl, "weight0", None) is None:
                continue
            num_local = _count_discrete_expert_weights(gl)
            stacked = torch.stack(
                [_dequantized_tensor(getattr(gl, f"weight{i}")) for i in range(num_local)],
                dim=0,
            ).contiguous()
            key = f"{name}.{attr}.weight"
            dt = DTensor.from_local(stacked, device_mesh=ep_mesh, placements=[Shard(0)])
            global_snap[key] = dt.full_tensor().detach().cpu()
    return global_snap


def _gather_global_optimizer_master(model: torch.nn.Module, optimizer, ep_mesh) -> dict[str, torch.Tensor]:
    from torch.distributed.tensor import DTensor, Shard

    out: dict[str, torch.Tensor] = {}
    state_names = ("master_param", "exp_avg", "exp_avg_sq")
    for name, module in model.named_modules():
        for attr in ("experts_gate_up", "experts_down"):
            gl = getattr(module, attr, None)
            if gl is None or getattr(gl, "weight0", None) is None:
                continue
            num_local = _count_discrete_expert_weights(gl)
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


def _get_dummy_batch(vocab_size: int, device: torch.device):
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (2, 32), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _build_model_and_optimizer(device: torch.device, ep_size: int, seed: int):
    from distributed_setup import build_mesh_and_wrap
    from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM
    from scheduler import get_cosine_annealing_schedule_with_warmup
    from transformer_engine.pytorch.optimizers import FusedAdam

    torch.manual_seed(seed)
    config_path = Path(__file__).parent.parent / "model_configs" / "mixtral_tiny"
    config = NVMixtralConfig.from_pretrained(
        str(config_path),
        expert_parallel_size=ep_size,
        expert_ffn_mode="fused_grouped_mlp",
        torch_dtype=torch.bfloat16,
    )
    model = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device=device)
    mesh = build_mesh_and_wrap(model, dp_size=1, ep_size=ep_size)
    ep_mesh = mesh["ep"]
    dp_process_group = mesh["dp"].get_group()

    optimizer = FusedAdam(model.parameters(), lr=1e-3, master_weights=True)
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, num_warmup_steps=0, num_decay_steps=100)
    return model, optimizer, scheduler, mesh, ep_mesh, dp_process_group, config


def _worker_stop_go_reference(tmp_dir: str) -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    _init_determinism()
    recipe_root = Path(__file__).parent.parent
    cfg = _compose_config(
        recipe_root,
        Path(tmp_dir),
        overrides=[
            f"num_train_steps={2 * N_STEPS}",
            "checkpoint.resume_from_checkpoint=false",
            "checkpoint.save_every_n_steps=0",
        ],
    )
    _, losses = _capture_losses_with_main(cfg)
    from distributed_config import DistributedConfig

    if DistributedConfig().is_main_process():
        _write_losses(Path(tmp_dir) / "reference_losses.json", losses)


def _worker_stop_go_phase1(tmp_dir: str) -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    _init_determinism()
    recipe_root = Path(__file__).parent.parent
    cfg = _compose_config(
        recipe_root,
        Path(tmp_dir),
        overrides=[
            f"num_train_steps={N_STEPS}",
            "checkpoint.resume_from_checkpoint=false",
            f"checkpoint.save_every_n_steps={CHECKPOINT_STEP}",
        ],
    )
    _, losses = _capture_losses_with_main(cfg)
    from distributed_config import DistributedConfig

    if DistributedConfig().is_main_process():
        _write_losses(Path(tmp_dir) / "phase1_losses.json", losses)


def _worker_stop_go_phase2(tmp_dir: str) -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    _init_determinism()
    recipe_root = Path(__file__).parent.parent
    cfg = _compose_config(
        recipe_root,
        Path(tmp_dir),
        overrides=[
            f"num_train_steps={2 * N_STEPS}",
            "checkpoint.resume_from_checkpoint=true",
            f"checkpoint.save_every_n_steps={CHECKPOINT_STEP}",
        ],
    )
    _, losses = _capture_losses_with_main(cfg)
    from distributed_config import DistributedConfig

    if DistributedConfig().is_main_process():
        phase1 = _read_losses(Path(tmp_dir) / "phase1_losses.json")
        combined = sorted(phase1.items()) + losses
        _write_losses(Path(tmp_dir) / "stopgo_losses.json", combined)


def _worker_ep4_save(tmp_dir: str) -> None:
    from checkpoint import save_checkpoint
    from distributed_config import DistributedConfig

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    assert dist_config.world_size == 4

    model, optimizer, scheduler, _, ep_mesh, dp_process_group, config = _build_model_and_optimizer(
        device, ep_size=4, seed=42
    )
    batch = _get_dummy_batch(config.vocab_size, device)
    model.train()
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    scheduler.step()

    ckpt_path = os.path.join(tmp_dir, "reshard_ckpt")
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

    global_expert = _gather_global_discrete_expert_weights(model, ep_mesh)
    global_master = _gather_global_optimizer_master(model, optimizer, ep_mesh)
    if dist_config.is_main_process():
        torch.save(global_expert, os.path.join(tmp_dir, "global_expert_ref.pt"))
        torch.save(global_master, os.path.join(tmp_dir, "global_opt_ref.pt"))
        print(f"Saved EP=4 reference with {len(global_expert)} expert tensors")

    torch.distributed.destroy_process_group()


def _worker_ep2_load_verify(tmp_dir: str) -> None:
    import grouped_dcp
    from checkpoint import (
        _bootstrap_expert_optimizer_state,
        _experts_ckpt_dir,
        _experts_optimizer_ckpt_dir,
    )
    from distributed_config import DistributedConfig

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    assert dist_config.world_size == 2

    model, optimizer, _scheduler, _, ep_mesh, _dp_process_group, _config = _build_model_and_optimizer(
        device, ep_size=2, seed=999
    )
    checkpoint_path = Path(tmp_dir) / "reshard_ckpt" / "step_1"
    grouped_dcp.load_consolidated(model, ep_mesh, _experts_ckpt_dir(checkpoint_path))
    _bootstrap_expert_optimizer_state(model, optimizer)
    grouped_dcp.load_optimizer_consolidated(model, optimizer, ep_mesh, _experts_optimizer_ckpt_dir(checkpoint_path))

    loaded_expert = _gather_global_discrete_expert_weights(model, ep_mesh)
    loaded_master = _gather_global_optimizer_master(model, optimizer, ep_mesh)

    if dist_config.is_main_process():
        ref_expert = torch.load(os.path.join(tmp_dir, "global_expert_ref.pt"), weights_only=True)
        ref_master = torch.load(os.path.join(tmp_dir, "global_opt_ref.pt"), weights_only=True)
        for key, ref_w in ref_expert.items():
            torch.testing.assert_close(
                loaded_expert[key],
                ref_w,
                atol=0,
                rtol=0,
                msg=f"reshard expert weight mismatch {key}",
            )
        for key, ref_w in ref_master.items():
            torch.testing.assert_close(
                loaded_master[key],
                ref_w,
                atol=0,
                rtol=0,
                msg=f"reshard optimizer master mismatch {key}",
            )
        print("EP=4 -> EP=2 recipe checkpoint reshard PASSED")

    torch.distributed.destroy_process_group()


@requires_multi_gpu
@requires_sm100
def test_ep2_stop_go_loss_parity(tmp_path):
    """Train N steps, checkpoint, resume in a fresh process, and match uninterrupted 2N losses."""
    result_ref = _run_torchrun("stop_go_reference", str(tmp_path), nproc=2)
    if result_ref.returncode != 0:
        print(result_ref.stdout)
        pytest.fail(f"Reference run failed: {result_ref.returncode}")

    result_p1 = _run_torchrun("stop_go_phase1", str(tmp_path), nproc=2)
    if result_p1.returncode != 0:
        print(result_p1.stdout)
        pytest.fail(f"Stop/go phase 1 failed: {result_p1.returncode}")

    ckpt_subdir = tmp_path / "ckpt" / "train_fsdp2_ep"
    assert (ckpt_subdir / f"step_{CHECKPOINT_STEP}").is_dir(), f"Missing checkpoint at step {CHECKPOINT_STEP}"

    result_p2 = _run_torchrun("stop_go_phase2", str(tmp_path), nproc=2)
    if result_p2.returncode != 0:
        print(result_p2.stdout)
        pytest.fail(f"Stop/go phase 2 failed: {result_p2.returncode}")

    ref_losses = _read_losses(tmp_path / "reference_losses.json")
    stopgo_losses = _read_losses(tmp_path / "stopgo_losses.json")

    assert set(ref_losses.keys()) == set(stopgo_losses.keys()), (
        f"Step mismatch: ref={sorted(ref_losses.keys())}, stopgo={sorted(stopgo_losses.keys())}"
    )

    for step in sorted(ref_losses.keys()):
        ref = ref_losses[step]
        got = stopgo_losses[step]
        assert torch.isfinite(torch.tensor(got)), f"Stop/go loss at step {step} not finite: {got}"
        assert torch.allclose(
            torch.tensor(ref),
            torch.tensor(got),
            rtol=LOSS_RTOL,
            atol=LOSS_ATOL,
        ), f"Loss mismatch at step {step}: ref={ref:.4f}, stopgo={got:.4f}"


@requires_four_gpu
@requires_sm100
def test_ep4_to_ep2_reshard_checkpoint(tmp_path):
    """Save consolidated checkpoint at EP=4 and verify EP=2 load matches gathered reference."""
    result_save = _run_torchrun("ep4_save", str(tmp_path), nproc=4)
    if result_save.returncode != 0:
        print(result_save.stdout)
        pytest.fail(f"EP=4 save failed: {result_save.returncode}")

    assert (tmp_path / "global_expert_ref.pt").exists()
    assert (tmp_path / "global_opt_ref.pt").exists()

    result_verify = _run_torchrun("ep2_load_verify", str(tmp_path), nproc=2)
    if result_verify.returncode != 0:
        print(result_verify.stdout)
        pytest.fail(f"EP=2 load/verify failed: {result_verify.returncode}")


if __name__ == "__main__":
    workers = {
        "stop_go_reference": _worker_stop_go_reference,
        "stop_go_phase1": _worker_stop_go_phase1,
        "stop_go_phase2": _worker_stop_go_phase2,
        "ep4_save": _worker_ep4_save,
        "ep2_load_verify": _worker_ep2_load_verify,
    }
    workers[sys.argv[1]](sys.argv[2])
