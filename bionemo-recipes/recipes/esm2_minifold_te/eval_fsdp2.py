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

"""FSDP2 evaluation script for ESM2-MiniFold TE structure prediction.

Loads a trained checkpoint and evaluates on a held-out dataset, reporting
structure quality metrics (lDDT, distogram accuracy, contact prediction)
to WandB and stdout.

Usage:
    # With FSDP2 distributed checkpoint
    torchrun --nproc_per_node=2 eval_fsdp2.py checkpoint.ckpt_dir=/path/to/checkpoints

    # With exported safetensors model
    torchrun --nproc_per_node=2 eval_fsdp2.py \
        checkpoint.ckpt_dir=/path/to/final_model \
        checkpoint.checkpoint_type=safetensors
"""

import json
import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from tqdm import tqdm

import wandb
from checkpoint import load_checkpoint_fsdp2
from dataset import create_dataloader
from distributed_config import DistributedConfig
from modeling_esm2_minifold_te import ESM2MiniFoldTE
from quantization import ComponentPrecisionConfig, resolve_layer_precision
from scheduler import get_linear_schedule_with_warmup
from torch_compile_diagnostics import maybe_capture_dynamo_diagnostics, maybe_compile
from train_fsdp2 import compute_distogram_loss, compute_distogram_metrics


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _resolve_fsdp2_checkpoint_path(ckpt_dir: Path) -> Path:
    if ckpt_dir.name.startswith("step_"):
        return ckpt_dir
    train_fsdp2_dir = ckpt_dir / "train_fsdp2"
    if train_fsdp2_dir.exists():
        return train_fsdp2_dir
    return ckpt_dir


def _move_batch_to_device(batch, device: torch.device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _forward_once(model_runner, batch, *, num_recycling: int):
    with torch.no_grad():
        return model_runner(batch, num_recycling=num_recycling)


@hydra.main(config_path="hydra_config", config_name="eval", version_base="1.2")
def main(args: DictConfig) -> None:
    """Evaluate ESM2-MiniFold TE on a held-out dataset."""
    os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("BIONEMO_ALLOW_CONFIG_ONLY_ESM2_INIT", "1")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Initialize distributed
    dist_config = DistributedConfig()
    logger.info("Initializing eval: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size,),
        mesh_dim_names=("dp",),
    )

    # Resolve per-block quantization precision
    block_precision = resolve_layer_precision(
        num_layers=args.model.num_blocks,
        fp8_enabled=args.fp8_config.enabled,
        fp4_enabled=args.fp4_config.enabled,
        fp8_layers=OmegaConf.to_container(args.fp8_layers, resolve=True) if args.fp8_layers is not None else None,
        fp4_layers=OmegaConf.to_container(args.fp4_layers, resolve=True) if args.fp4_layers is not None else None,
    )

    fp8_recipe = None
    fp4_recipe = None
    if args.fp8_config.enabled:
        from transformer_engine.common.recipe import Format

        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )
    if args.fp4_config.enabled:
        from transformer_engine.common.recipe import Format

        fp4_recipe = hydra.utils.get_class(args.fp4_config.fp4_recipe)(
            fp4_format=Format[args.fp4_config.fp4_format], **args.fp4_config.fp4_recipe_kwargs
        )

    # Component-level precision overrides
    component_precision = ComponentPrecisionConfig(**OmegaConf.to_container(args.component_precision, resolve=True))

    # Create model (same architecture as training)
    model = ESM2MiniFoldTE(
        esm_model_name=args.esm_model_name,
        c_s=args.model.c_s,
        c_z=args.model.c_z,
        num_blocks=args.model.num_blocks,
        no_bins=args.model.no_bins,
        use_structure_module=args.model.use_structure_module,
        block_precision=block_precision,
        fp8_recipe=fp8_recipe,
        fp4_recipe=fp4_recipe,
        component_precision=component_precision,
    ).to(device)

    # Load checkpoint
    ckpt_dir = Path(args.checkpoint.ckpt_dir)
    checkpoint_type = args.checkpoint.get("checkpoint_type", "fsdp2")
    loaded_step = None

    if checkpoint_type == "safetensors":
        # Load safetensors BEFORE FSDP2 sharding (plain tensors -> plain params)
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_dir / "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded safetensors model from %s", ckpt_dir)

    # FSDP2 sharding (must match training for FSDP2 checkpoint loading;
    # also needed for multi-GPU eval even with safetensors)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    for block in model.fold.miniformer.blocks:
        fully_shard(block, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    if checkpoint_type == "fsdp2":
        # Need dummy optimizer/scheduler for the checkpoint loader
        dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dummy_scheduler = get_linear_schedule_with_warmup(dummy_optimizer, num_warmup_steps=0, num_training_steps=1)
        ckpt_path = _resolve_fsdp2_checkpoint_path(ckpt_dir)
        model, _, _, _, loaded_step, _ = load_checkpoint_fsdp2(
            model=model,
            optimizer=dummy_optimizer,
            scheduler=dummy_scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
        )
        logger.info("Loaded FSDP2 checkpoint from step %d", loaded_step)
    elif checkpoint_type != "safetensors":
        raise ValueError(f"Unknown checkpoint_type: {checkpoint_type}")

    if dist_config.is_main_process():
        logger.info("Block precision: %s", block_precision)

    # Create eval dataloader (shuffle=False, drop_last=False from config)
    eval_dataloader, _ = create_dataloader(dist_config, **args.eval_dataset)
    logger.info("Eval dataset: %d batches", len(eval_dataloader))

    use_torch_compile = bool(args.get("use_torch_compile", False))
    torch_compile_mode = str(args.get("torch_compile_mode", "default"))
    diagnostics_output = args.get("diagnostics_output")
    output_json = args.get("output_json")
    model.eval()
    model_runner = maybe_compile(model, enabled=use_torch_compile, mode=torch_compile_mode)

    diagnostics = None
    if diagnostics_output:
        diagnostics_batch = _move_batch_to_device(next(iter(eval_dataloader)), device)
        diagnostics = maybe_capture_dynamo_diagnostics(
            _forward_once,
            args=(model_runner, diagnostics_batch),
            kwargs={"num_recycling": args.model.get("num_recycling", 0)},
            output_path=Path(str(diagnostics_output)),
            label="eval_fsdp2",
            rank=dist_config.rank,
            world_size=dist_config.world_size,
            extra_metadata={
                "use_torch_compile": use_torch_compile,
                "torch_compile_mode": torch_compile_mode,
                "checkpoint_dir": str(ckpt_dir),
                "checkpoint_type": checkpoint_type,
                "loaded_step": loaded_step,
                "tri_impl": args.component_precision.tri_impl,
            },
        )

    # Initialize WandB
    run_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    if dist_config.is_main_process():
        wandb.init(**args.wandb_init_args, config=run_config)

    # Eval loop
    all_metrics = {
        "loss": [],
        "disto_loss": [],
        "distogram_acc": [],
        "contact_precision_8A": [],
        "contact_recall_8A": [],
        "lddt_from_distogram": [],
        "mean_distance_error": [],
    }

    progress = tqdm(eval_dataloader, desc="Evaluating", disable=not dist_config.is_main_process())

    with torch.no_grad():
        for batch in progress:
            batch = _move_batch_to_device(batch, device)

            r_dict = _forward_once(model_runner, batch, num_recycling=args.model.get("num_recycling", 0))

            # Distogram loss
            disto_loss = compute_distogram_loss(
                preds=r_dict["preds"],
                coords=batch["coords"],
                mask=batch["mask"],
                no_bins=args.model.no_bins,
            )

            # Structure quality metrics
            metrics = compute_distogram_metrics(
                preds=r_dict["preds"].float(),
                coords=batch["coords"],
                mask=batch["mask"],
                no_bins=args.model.no_bins,
            )

            all_metrics["loss"].append(disto_loss.item())
            all_metrics["disto_loss"].append(disto_loss.item())
            for key, value in metrics.items():
                all_metrics[key].append(value.item())

            progress.set_postfix(
                {
                    "loss": f"{disto_loss.item():.3f}",
                    "lddt": f"{metrics['lddt_from_distogram'].item():.3f}",
                }
            )

    # Aggregate metrics
    summary = {}
    for key, values in all_metrics.items():
        if values:
            summary[f"eval/{key}"] = sum(values) / len(values)

    # Log to WandB and stdout
    if dist_config.is_main_process():
        wandb.log(summary)
        wandb.finish()

    payload = {
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_type": checkpoint_type,
        "loaded_step": loaded_step,
        "dataset_path": str(args.eval_dataset.parquet_path),
        "use_torch_compile": use_torch_compile,
        "torch_compile_mode": torch_compile_mode,
        "diagnostics_output": str(diagnostics_output) if diagnostics_output else None,
        "dynamo_diagnostics": diagnostics,
        "summary": summary,
        "batches_evaluated": len(all_metrics["loss"]),
    }
    if output_json and dist_config.is_main_process():
        output_path = Path(str(output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))

    if dist_config.local_rank == 0:
        logger.info("=== Evaluation Results ===")
        logger.info("Batches evaluated: %d", len(all_metrics["loss"]))
        for key, value in summary.items():
            logger.info("  %s: %.4f", key, value)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
