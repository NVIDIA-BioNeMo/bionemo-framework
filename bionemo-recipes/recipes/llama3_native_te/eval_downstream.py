#!/usr/bin/env python

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

"""Evaluate a trained Llama checkpoint on downstream NLP benchmarks using lm-eval.

Supports loading from:
  1. A consolidated final_model directory (model.safetensors + config.json)
  2. A distributed FSDP2 training checkpoint (step_N directory)
  3. A DDP checkpoint (checkpoint.pt file)

Examples:
    # From a consolidated final_model (single GPU, no torchrun needed):
    python eval_downstream.py \
        --checkpoint-path /path/to/ckpt_dir/train_fsdp2/final_model

    # From a distributed FSDP2 checkpoint (needs torchrun for weight gathering):
    torchrun --nproc_per_node=1 eval_downstream.py \
        --checkpoint-path /path/to/ckpt_dir/train_fsdp2/ \
        --from-distributed \
        --model-config ./model_configs/lingua-1B

    # Specific step from a distributed checkpoint:
    torchrun --nproc_per_node=2 eval_downstream.py \
        --checkpoint-path /path/to/ckpt_dir/train_fsdp2/ \
        --from-distributed \
        --checkpoint-step 20000 \
        --model-config ./model_configs/lingua-1B

    # Custom tasks and batch size:
    python eval_downstream.py \
        --checkpoint-path /path/to/final_model \
        --tasks arc_easy,hellaswag \
        --batch-size 16

    # Save results to a file:
    python eval_downstream.py \
        --checkpoint-path /path/to/final_model \
        --output-path ./eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DOWNSTREAM_TASKS = "arc_challenge,arc_easy,boolq,copa,hellaswag,piqa,winogrande"


# ---------------------------------------------------------------------------
# Checkpoint discovery (adapted from opengenome2 evaluate_fasta_lm_loss.py)
# ---------------------------------------------------------------------------


def find_checkpoint_path(checkpoint_dir: str, step: int | None = None) -> tuple[Path, str]:
    """Locate the checkpoint inside *checkpoint_dir* and return ``(path, type)``.

    Supports:
      - ``safetensors``: directory with ``model.safetensors``
      - ``dcp``: FSDP2 distributed checkpoint (``step_N/`` with ``.metadata``)
      - ``ddp``: DDP checkpoint (``checkpoint.pt``)

    Args:
        checkpoint_dir: Root checkpoint directory.
        step: Specific step to load. If None, uses the latest.

    Returns:
        Tuple of (checkpoint_path, checkpoint_type).
    """
    root = Path(checkpoint_dir)

    for candidate in [root, root / "final_model", root / "train_fsdp2" / "final_model"]:
        if (candidate / "model.safetensors").exists():
            return candidate, "safetensors"

    fsdp2_dir = root / "train_fsdp2" if (root / "train_fsdp2").exists() else root
    step_dirs = sorted(
        [d for d in fsdp2_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if step_dirs:
        if step is not None:
            target = fsdp2_dir / f"step_{step}"
            if not target.exists():
                raise FileNotFoundError(f"step_{step} not found. Available: {[d.name for d in step_dirs]}")
            chosen = target
        else:
            chosen = step_dirs[-1]
        if (chosen / ".metadata").exists() or any(chosen.glob("*.distcp")):
            return chosen, "dcp"
        if (chosen / "checkpoint.pt").exists():
            return chosen, "ddp"
        return chosen, "dcp"

    if (root / "checkpoint.pt").exists():
        return root, "ddp"
    if (root / ".metadata").exists() or any(root.glob("*.distcp")):
        return root, "dcp"

    raise FileNotFoundError(f"No recognisable checkpoint in {checkpoint_dir}")


# ---------------------------------------------------------------------------
# DCP loading helpers (self-contained, avoids importing checkpoint.py which
# may have TE version-specific imports)
# ---------------------------------------------------------------------------


def _build_app_state(model, optimizer, scheduler):
    """Build a Stateful wrapper compatible with the training checkpoint format."""
    from dataclasses import dataclass

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
    from torch.distributed.checkpoint.stateful import Stateful

    @dataclass
    class _AppState(Stateful):
        model: object
        optimizer: object
        scheduler: object
        step: int = 0
        epoch: int = 0

        def state_dict(self):
            model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
            model_state_dict = {k: v for k, v in model_state_dict.items() if not k.endswith("_extra_state")}
            return {
                "model": model_state_dict,
                "optim": optimizer_state_dict,
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            }

        def load_state_dict(self, state_dict: dict):
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"],
                options=StateDictOptions(strict=False),
            )
            self.scheduler.load_state_dict(state_dict["scheduler"])
            self.step = state_dict["step"]
            self.epoch = state_dict["epoch"]

    return _AppState(model=model, optimizer=optimizer, scheduler=scheduler)


def _get_lenient_load_planner():
    """Return a load planner that skips keys missing from the checkpoint.

    Handles checkpoints saved without TransformerEngine _extra_state keys
    (FP8 metadata). These keys are registered by newer TE versions even when
    FP8 is disabled, but older checkpoints don't contain them.
    """
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    class _LenientLoadPlanner(DefaultLoadPlanner):
        def create_local_plan(self):
            missing_keys = [fqn for fqn in self.state_dict if fqn not in self.metadata.state_dict_metadata]
            if missing_keys:
                logger.warning(
                    "Skipping %d keys not found in checkpoint: %s%s",
                    len(missing_keys),
                    missing_keys[:5],
                    "..." if len(missing_keys) > 5 else "",
                )
                for key in missing_keys:
                    del self.state_dict[key]
            return super().create_local_plan()

    return _LenientLoadPlanner()


# ---------------------------------------------------------------------------
# Distributed checkpoint export
# ---------------------------------------------------------------------------


def export_distributed_checkpoint(
    checkpoint_dir: str,
    model_config: str,
    output_path: str,
    checkpoint_step: int | None = None,
) -> bool:
    """Load a distributed checkpoint and export consolidated weights.

    Auto-detects checkpoint format (safetensors, DCP, DDP). For DCP checkpoints,
    must be called inside a torchrun context. All ranks participate in loading
    and gathering, but only rank 0 saves the exported model.

    Args:
        checkpoint_dir: Root checkpoint directory.
        model_config: Path to model config (e.g. ./model_configs/lingua-1B).
        output_path: Directory to save the consolidated model.
        checkpoint_step: Specific step to load (latest if None).

    Returns:
        True if this is rank 0 (should continue to evaluation), False otherwise.
    """
    import torch
    from safetensors.torch import load_file, save_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from distributed_config import DistributedConfig
    from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
    from scheduler import get_cosine_annealing_schedule_with_warmup

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("dp",))

    ckpt_path, ckpt_type = find_checkpoint_path(checkpoint_dir, checkpoint_step)
    if dist_config.rank == 0:
        logger.info("Resolved checkpoint: %s (type=%s)", ckpt_path, ckpt_type)

    config = NVLlamaConfig.from_pretrained(model_config, dtype=torch.bfloat16, attn_input_format="bshd")
    model = NVLlamaForCausalLM(config)
    if dist_config.rank == 0:
        logger.info("Model created (%s parameters)", f"{sum(p.numel() for p in model.parameters()):,}")

    # For safetensors/DDP: load weights BEFORE FSDP2 wrapping (plain Tensor → plain Parameter)
    if ckpt_type == "safetensors":
        weights = load_file(str(ckpt_path / "model.safetensors"))
        model.load_state_dict(weights, strict=False)
        if dist_config.rank == 0:
            logger.info("Loaded safetensors checkpoint")
    elif ckpt_type == "ddp":
        ckpt = torch.load(ckpt_path / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"], strict=False)
        if dist_config.rank == 0:
            logger.info("Loaded DDP checkpoint (step=%d)", ckpt.get("step", -1))

    # FSDP2 wrapping
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"])
    fully_shard(model, mesh=device_mesh["dp"])

    # For DCP: load AFTER FSDP2 wrapping (DCP handles DTensor resharding)
    if ckpt_type == "dcp":
        if dist_config.rank == 0:
            logger.info("Loading FSDP2 DCP checkpoint from %s", ckpt_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, num_warmup_steps=100, num_decay_steps=1000)
        app_state = _build_app_state(model, optimizer, scheduler)
        dcp_load(
            {"app": app_state},
            checkpoint_id=ckpt_path,
            process_group=device_mesh.get_group("dp"),
            planner=_get_lenient_load_planner(),
        )
        if dist_config.rank == 0:
            logger.info("DCP checkpoint loaded (step=%d, epoch=%d)", app_state.step, app_state.epoch)

    # Gather full model state dict from all ranks
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    if dist_config.is_main_process():
        os.makedirs(output_path, exist_ok=True)
        save_file(model_state_dict, os.path.join(output_path, "model.safetensors"))
        config.save_pretrained(output_path)
        logger.info("Exported consolidated model to %s", output_path)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

    return dist_config.is_main_process()


# ---------------------------------------------------------------------------
# Eval directory preparation
# ---------------------------------------------------------------------------


def prepare_eval_directory(checkpoint_path: str, output_path: str, tokenizer_name: str) -> str:
    """Prepare a checkpoint directory with all files lm-eval needs.

    Copies model files, patches config.json with auto_map and inference-compatible
    attention settings, copies modeling_llama_te.py, and saves the tokenizer.

    Args:
        checkpoint_path: Source directory with model.safetensors + config.json.
        output_path: Destination directory for the eval-ready checkpoint.
        tokenizer_name: HuggingFace tokenizer name or local path.

    Returns:
        The output_path string.
    """
    from transformers import AutoTokenizer

    from modeling_llama_te import AUTO_MAP

    checkpoint_path_obj = Path(checkpoint_path)
    output_path_obj = Path(output_path)

    if output_path_obj.resolve() != checkpoint_path_obj.resolve():
        os.makedirs(output_path, exist_ok=True)
        for f in checkpoint_path_obj.iterdir():
            if f.is_file():
                shutil.copy2(f, output_path_obj / f.name)

    config_file = output_path_obj / "config.json"
    with open(config_file) as f:
        config = json.load(f)

    config["auto_map"] = AUTO_MAP
    config["attn_input_format"] = "bshd"
    config["self_attn_mask_type"] = "causal"

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    script_dir = Path(__file__).parent
    shutil.copy2(script_dir / "modeling_llama_te.py", output_path_obj / "modeling_llama_te.py")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(str(output_path_obj))

    logger.info("Prepared eval directory: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# lm-eval runner
# ---------------------------------------------------------------------------


def run_lm_eval(
    eval_dir: str,
    tasks: str,
    batch_size: str,
    device: str,
    output_path: str | None = None,
    num_fewshot: int | None = None,
) -> float:
    """Run lm-eval on the prepared checkpoint directory.

    Args:
        eval_dir: Path to the prepared eval checkpoint directory.
        tasks: Comma-separated list of lm-eval task names.
        batch_size: Batch size string (integer or "auto").
        device: Device string (e.g. "cuda:0").
        output_path: Optional path to save results JSON.
        num_fewshot: Optional number of few-shot examples.

    Returns:
        Wall-clock time in seconds.
    """
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={eval_dir},tokenizer={eval_dir}",
        "--trust_remote_code",
        "--tasks",
        tasks,
        "--device",
        device,
        "--batch_size",
        batch_size,
    ]

    if output_path:
        cmd.extend(["--output_path", output_path])

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    logger.info("Running lm-eval:\n  %s", " ".join(cmd))
    print("=" * 80)

    start_time = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start_time

    print("=" * 80)
    logger.info("lm-eval completed in %.1fs (%.1f min)", elapsed, elapsed / 60)

    if result.returncode != 0:
        logger.error("lm-eval failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    return elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on downstream NLP tasks with lm-eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint directory. Auto-detects format (final_model, step_N, etc).",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Specific training step to evaluate (latest if omitted).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Tokenizer name or path (default: meta-llama/Meta-Llama-3-8B).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=DOWNSTREAM_TASKS,
        help=f"Comma-separated lm-eval task names (default: {DOWNSTREAM_TASKS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Batch size for lm-eval. Use 'auto' for automatic selection (default: auto).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for lm-eval inference (default: cuda:0).",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Directory to store the prepared eval checkpoint. Uses a temp directory if not set.",
    )
    parser.add_argument(
        "--from-distributed",
        action="store_true",
        help="Export from a distributed checkpoint before evaluating. Requires torchrun.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="./model_configs/lingua-1B",
        help="Model config path for --from-distributed (default: ./model_configs/lingua-1B).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save lm-eval results JSON.",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: lm-eval task default).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point."""
    args = parse_args()

    use_temp = args.eval_dir is None
    eval_dir = args.eval_dir if args.eval_dir else tempfile.mkdtemp(prefix="lm_eval_checkpoint_")

    if use_temp:
        logger.info("Using temporary eval directory: %s", eval_dir)

    try:
        if args.from_distributed:
            is_main = export_distributed_checkpoint(
                checkpoint_dir=args.checkpoint_path,
                model_config=args.model_config,
                output_path=eval_dir,
                checkpoint_step=args.checkpoint_step,
            )
            if not is_main:
                return
            source_dir = eval_dir
        else:
            ckpt_path, ckpt_type = find_checkpoint_path(args.checkpoint_path, args.checkpoint_step)
            if ckpt_type != "safetensors":
                logger.error(
                    "Found %s checkpoint at %s. Non-safetensors checkpoints require "
                    "--from-distributed flag with torchrun.",
                    ckpt_type,
                    ckpt_path,
                )
                sys.exit(1)
            source_dir = str(ckpt_path)

        prepare_eval_directory(
            checkpoint_path=source_dir,
            output_path=eval_dir,
            tokenizer_name=args.tokenizer,
        )

        run_lm_eval(
            eval_dir=eval_dir,
            tasks=args.tasks,
            batch_size=args.batch_size,
            device=args.device,
            output_path=args.output_path,
            num_fewshot=args.num_fewshot,
        )

    finally:
        if use_temp and os.path.exists(eval_dir):
            logger.info("Cleaning up temporary directory: %s", eval_dir)
            shutil.rmtree(eval_dir)


if __name__ == "__main__":
    main()
