# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse

# TODO add back support for TEV logging
# from bionemo.eden.utils.logging.callbacks import TEVCallback
import logging
from pathlib import Path
from typing import List, Optional

import torch
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer, FaultToleranceConfig
from megatron.bridge.training.gpt_step import forward_step as gpt_forward_step
from megatron.bridge.training.mixed_precision import MIXED_PRECISION_RECIPES
from megatron.bridge.training.post_training.checkpointing import has_modelopt_state
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.utils.common_utils import get_rank_safe

from bionemo.eden.models.eden_provider import EDEN_MODEL_OPTIONS
from bionemo.eden.recipes.eden import DEFAULT_HF_TOKENIZER_MODEL_PATH
from bionemo.eden.recipes.eden import eden_pretrain_config as pretrain_config


logger: logging.Logger = logging.getLogger(__name__)

torch._dynamo.config.suppress_errors = True


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments for Eden model training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train an Eden model using NeMo 2.0.\n\n"
            "Choose exactly one data source:\n"
            "  - --mock-data: synthetic mock data for testing/debugging.\n"
            "  - --sharded-eden-data: pre-sharded SQLite sequence DBs + precomputed windows per split\n"
            "      (requires --sequence-db-dir, --train-window-db, --val-window-db, --test-window-db)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = parser.add_mutually_exclusive_group(required=True)

    data_group.add_argument(
        "--mock-data",
        action="store_true",
        help="Use synthetic mock data for quick testing/debugging. Mutually exclusive with --sharded-eden-data.",
    )

    data_group.add_argument(
        "--sharded-eden-data",
        action="store_true",
        help=(
            "Train on pre-sharded SQLite sequence databases with precomputed windows per split "
            "(ShardedEdenDataModule). Requires: --sequence-db-dir, --train-window-db, --val-window-db, --test-window-db. "
            "Mutually exclusive with --mock-data."
        ),
    )

    # Dataset configuration (unified)
    parser.add_argument(
        "--sequence-db-dir",
        type=str,
        help=(
            "Directory containing per-sample SQLite databases with sequences. Required with --sharded-eden-data; "
            "ignored otherwise."
        ),
    )
    parser.add_argument(
        "--train-window-db",
        type=str,
        help=(
            "Path to the precomputed training split windows SQLite database. Required with --sharded-eden-data; "
            "ignored otherwise."
        ),
    )
    parser.add_argument(
        "--val-window-db",
        type=str,
        help=(
            "Path to the precomputed validation split windows SQLite database. Required with --sharded-eden-data; "
            "ignored otherwise."
        ),
    )
    parser.add_argument(
        "--test-window-db",
        type=str,
        help=(
            "Path to the precomputed test split windows SQLite database. Required with --sharded-eden-data; "
            "ignored otherwise."
        ),
    )
    # parser.add_argument(
    #     "--dataset-num-epochs",
    #     type=int,
    #     default=1,
    #     help=(
    #         "When using --sharded-eden-data, wrap each split with a MultiEpochDatasetResampler over this many epochs. "
    #         "Default 1 means each split length equals its base dataset length."
    #     ),
    # )  # TODO implement
    parser.add_argument(
        "--stride",
        type=int,
        default=7992,
        help=(
            "Stride between adjacent windows used by ShardedEdenDataModule. Must match the stride used when "
            "precomputing the windows databases. Ignored for other data modes."
        ),
    )  # DONE
    parser.add_argument(
        "--window-min-length-threshold",
        type=int,
        default=0,
        help=(
            "If > 0, prune windows shorter than this effective length during precomputation and require matching "
            "value in the window DB metadata. Defaults to 0 (disabled)."
        ),
    )  # DONE
    # parser.add_argument(
    #     "--log-windows",
    #     action="store_true",
    #     default=False,
    #     help=("Enable window access logging for ShardedEdenDataset (applies only to --sharded-eden-data)."),
    # )  # TODO implement
    # parser.add_argument(
    #     "--window-log-dir",
    #     type=str,
    #     default=None,
    #     help=("Directory for window-access logging SQLite files (applies only to --sharded-eden-data)."),
    # )  # TODO implement
    parser.add_argument(
        "--rc-aug",
        action="store_true",
        default=False,
        help=("Enable reverse-complement augmentation (applies only to --sharded-eden-data)."),
    )  # DONE
    parser.add_argument("--seq-length", type=int, default=8192, help="Training sequence length")  # DONE
    parser.add_argument(
        "--tensor-model-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1."
    )  # DONE
    parser.add_argument(
        "--pipeline-model-parallel-size", type=int, default=1, help="Order of pipeline parallelism. Defaults to 1."
    )  # DONE
    parser.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )  # DONE
    parser.add_argument(
        "--disable-tensorboard-logger", action="store_true", default=False, help="Create a tensorboard logger."
    )  # DONE
    parser.add_argument("--wandb-entity", type=str, default=None, help="The team posting this run")  # DONE
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name ")  # DONE
    # FIXME wandb tags, group, job type are not supported in megatron.
    # parser.add_argument("--wandb-tags", nargs="+", type=str, default=None, help="Tags associated with this run")
    # parser.add_argument(
    #     "--wandb-group", type=str, default=None, help="A unique string shared by all runs in a given group"
    # )
    # parser.add_argument(
    #     "--wandb-job-type",
    #     type=str,
    #     default=None,
    #     help="A unique string representing a type of run, which is useful when you're grouping runs together into larger experiments using group.",
    # )
    parser.add_argument(
        "--wandb-run-name",  # maps to wandb.experiment.name in megatron.
        type=str,
        default=None,
        help="A unique string representing the name of the wandb run. If not provided, the name will be generated from the model and training specifications.",
    )  # DONE

    # parser.add_argument(
    #     "--wandb-id", type=str, default=None, help="Sets the version, mainly used to resume a previous run"
    # )  # FIXME not supported in megatron
    # parser.add_argument(
    #     "--wandb-anonymous", action="store_true", help="Enable or explicitly disable anonymous logging"
    # )  # FIXME not supported in megatron
    # parser.add_argument(
    #     "--wandb-log-model", action="store_true", help="Save checkpoints in wandb dir to upload on W&B servers"
    # )  # FIXME not supported in megatron
    # parser.add_argument("--wandb-offline", action="store_true", help="Use wandb in offline mode")  # TODO implement
    parser.add_argument("--sequence-parallel", action="store_true", help="Set to enable sequence parallelism.")  # DONE
    parser.add_argument("--no-fp8-wgrad", action="store_true", help="Set to disable fp8 weight gradients.")
    parser.add_argument("--no-fp8-param-gather", action="store_true", help="Set to disable fp8 parameter gathering.")
    parser.add_argument(
        "--mixed-precision-recipe",
        type=str,
        choices=list(MIXED_PRECISION_RECIPES.keys()),
        default="bf16_mixed",
        help="Mixed precision recipe to use for training.",
    )  # DONE
    parser.add_argument(
        "--micro-batch-size", type=int, default=1, help="Micro-batch size for data-parallel training."
    )  # DONE
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=8,
        help="Global batch size for training. "
        "From this and the model parallel sizes, gradient accumulation is inferred.",
    )  # DONE
    # parser.add_argument(
    #     "--grad-acc-batches", type=int, default=1, help="Number of batches to accumulate gradients over. IGNORED FOR NOW."
    # )  # TODO implement
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Number of training optimizer update steps. This controls the total number of steps as well as the "
        "shape of the learning rate curve.",
        default=100_000,
    )  # DONE
    parser.add_argument(
        "--constant-steps",
        type=int,
        help="Number of steps to keep the learning rate constant at minimum after annealing. This controls the "
        "shape of the learning rate curve.",
        default=2_500,
    )  # DONE
    parser.add_argument(
        "--decay-steps",
        type=int,
        help="Number of steps to decay the learning rate to minimum after annealing. If provided, --constant-steps is ignored.",
        default=None,
    )  # DONE
    parser.add_argument(
        "--early-stop-on-step",
        type=int,
        help="Stop training on this step, if set. This may be useful for testing or debugging purposes.",
    )  # TODO implement
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Number of steps between validation measurements and model checkpoints.",
    )  # DONE
    parser.add_argument("--eval-iters", type=int, default=32, help="Number of validation iterations.")  # DONE
    parser.add_argument(
        "--grad-reduce-in-fp32", action="store_true", default=False, help="Gradient reduce in FP32."
    )  # DONE
    parser.add_argument("--use-megatron-comm-overlap-llama3-8k", action="store_true", default=False)  # DONE
    parser.add_argument(
        "--tp-comm-overlap-backend",
        type=str,
        choices=["nccl", "mpi", "gloo"],
        default="nccl",
        help="TP communication backend to use. Defaults to 'nccl'.",
    )  # DONE
    parser.add_argument("--align-param-gather", action="store_true", default=False)  # DONE
    parser.add_argument(
        "--model-size",
        type=str,
        choices=sorted(EDEN_MODEL_OPTIONS.keys()),
        default="eden_7b",
        help="Model size/configuration to use.",
    )  # DONE
    parser.add_argument(
        "--add-bias-output",
        action="store_true",
        default=False,
        help="Add bias to the output layer to enable learning a simple prior.",
    )  # DONE
    parser.add_argument(
        "--result-dir", type=Path, required=False, default=Path("./results"), help="Path to the result directory."
    )  # DONE
    parser.add_argument(
        "--experiment-name", type=str, required=False, default="eden", help="Name of the experiment."
    )  # DONE

    parser.add_argument(
        "--finetune-ckpt-dir",
        type=str,
        default=None,
        help="Directory to restore an initial checkpoint from. Use this for supervised fine-tuning.",
    )  # DONE
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between logging.")  # DONE
    parser.add_argument(
        "--use-precision-aware-optimizer",
        action="store_true",
        default=False,
        help="Use precision aware optimizer that stores main weights in FP32 when doing mixed precision training.",
    )  # DONE
    parser.add_argument(
        "--bf16-main-grads",
        action="store_true",
        default=False,
        help="Use bf16 for main gradients, only use this with --use-precision-aware-optimizer.",
    )  # DONE
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay for optimizer.")  # DONE
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="Adam optimizer beta1 parameter.",
    )  # DONE
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.95,
        help="Adam optimizer beta2 parameter.",
    )  # DONE
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="Adam optimizer epsilon parameter. The inverse of this value (1/eps) represents the maximum adaptive learning rate per parameter.",
    )  # DONE
    parser.add_argument(
        "--average-in-collective",
        action="store_true",
        default=False,
        help="Avaerage optimizer state in collective rather than dividing by dp size and summing.",
    )  # DONE
    parser.add_argument("--seed", type=int, default=1234, help="Set random seed for training.")  # DONE
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help="Set random seed for dataset shuffling. Defaults to training seed if not provided.",
    )  # DONE
    parser.add_argument("--workers", type=int, default=8, help="Number of workers to use for data loading.")  # DONE
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=0,
        help="Set to a value > 0 if you want to synchronize garbage collection, will do gc every gc-interval steps.",
    )  # DONE
    parser.add_argument(
        "--enable-preemption",
        action="store_true",
        default=False,
        help="Enable preemption hooks. If enabled this will save a checkpoint whenever slurm exits.",
    )  # DONE
    parser.add_argument(
        "--ckpt-async-save",
        action="store_true",
        default=False,
    )  # DONE
    parser.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated. Only use if "
        "resuming training from a zarr checkpoint.",
    )  # DONE
    parser.add_argument(
        "--cross-entropy-loss-fusion",
        action="store_true",
        default=False,
        help="Use the faster, but maybe less accurate fused form of cross entropy, "
        "which also has bf16 grads internally.",
    )  # DONE
    parser.add_argument(
        "--no-fp32-residual-connection",
        action="store_true",
        default=False,
        help="If set, turn off fp32 residual connections which may be faster but may impact accuracy.",
    )  # DONE
    parser.add_argument(
        "--debug-ddp-parity-freq",
        type=int,
        default=0,
        help="Set to value > 0 to debug DDP weight parity between ranks.",
    )  # DONE
    parser.add_argument(
        "--num-layers", type=int, help="If set, override the number of layers specified in the requested config."
    )  # DONE

    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")  # DONE
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Min learning rate in cosine annealing.")  # DONE
    parser.add_argument(
        "--warmup-steps", type=int, default=2500, help="Number of warmup steps in cosine annealing"
    )  # DONE
    # NSYS profiling/tooling arguments
    parser.add_argument(
        "--nsys-profiling",
        action="store_true",
        default=False,
        help="Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling"
        " output you must run the whole program with `nsys`. For example: "
        " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true "
        "--capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`",
    )  # DONE
    # start, end, rank
    parser.add_argument(
        "--nsys-start-step",
        type=int,
        required=False,
        default=0,
        help="Start nsys profiling after this step.",
    )  # DONE
    parser.add_argument(
        "--spike-no-more-embedding-init",
        action="store_true",
        default=False,
        help="If set, the embeddings are initialized with a Normal(0, 1.0) distribution rather "
        "than the default Normal(0, 0.02). This may help avoid loss spiking during training. Consider using this with "
        "--no-weight-decay-embeddings to avoid shrinking the embeddings to 0 by skipping weight decay on these layers, "
        "or with --use-targeted-variance-loss to maintain a 1.0 variance during training even with weight decay. This "
        "also turns off shared weights between embeddings and outputs.",
    )  # DONE
    parser.add_argument(
        "--no-weight-decay-embeddings",
        action="store_true",
        default=False,
        help="If set, do not apply weight decay to the embeddings.",
    )  # DONE
    parser.add_argument(
        "--use-targeted-variance-loss",
        action="store_true",
        default=False,
        help="Use targeted variance loss.",
    )  # DONE
    parser.add_argument(
        "--nsys-end-step",
        type=int,
        required=False,
        help="End nsys profiling after this step.",
    )  # DONE
    parser.add_argument(
        "--no-renormalize-loss",
        action="store_true",
        default=False,
        help="Do not renormalize the loss weights.",
    )  # DONE
    # parser.add_argument(
    #     "--mamba-lowercase-loss-weight",
    #     type=float,
    #     default=0.1,
    #     help="Loss weight for the Mamba model for lowercase bases, if you are using a Mamba model. "
    #     "Default is 0.1 like the Eden paper. Set to 1.0 to disable differential loss weighting.",
    # )  # TODO implement
    # rank as list of integers
    parser.add_argument(
        "--nsys-ranks",
        type=int,
        nargs="+",
        required=False,
        default=[0],
        help="Enable nsys profiling for these ranks.",
    )  # DONE
    parser.add_argument(
        "--activation-checkpoint-recompute-num-layers",
        type=int,
        help="If set, override the default value set in the config.",
    )  # DONE
    # parser.add_argument(
    #     "--disable-checkpointing",
    #     action="store_false",
    #     default=True,
    #     dest="create_checkpoint_callback",
    #     help="Disable creating a ModelCheckpoint callback.",
    # )  # TODO implement
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Grad clip value. Note that when using DDP this may need to be inflated.",
    )  # DONE
    parser.add_argument(
        "--seq-len-interpolation-factor",
        type=float,
        help="Adjusts the linear scaling of ROPE (Rotary Position Embedding) for context extension. "
        "Set this factor relative to your base context length e.g., for an original context length of 8192 and "
        "an extended context length of 524288, use 524288/8192 = 64.",
    )  # DONE
    parser.add_argument(
        "--overlap-param-gather",
        action="store_true",
        default=False,
        help="Overlap the parameter gather with the optimizer step. This is currently disabled due to a NeMo bug "
        "when using DDP. Making this an option defaulting to False is a temporary solution until the bug is fixed.",
    )  # DONE
    parser.add_argument(
        "--overlap-grad-reduce",
        action="store_true",
        default=False,
        help="Overlap the gradient reduce with the optimizer step.",
    )  # DONE
    parser.add_argument(
        "--hidden-dropout",
        type=float,
        default=0.0,
        help="Dropout probability for the model layers.",
    )  # DONE
    parser.add_argument(
        "--ffn-hidden-size",
        type=int,
        default=None,
        help="FFN hidden size for the model layers.",
    )  # DONE
    parser.add_argument(
        "--log-num-zeros-in-grad",
        action="store_true",
        default=False,
        help="Log the number of zeros in the gradient.",
    )  # DONE
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.0,
        help="Dropout probability for the attention layers.",
    )  # DONE
    parser.add_argument(
        "--most-recent-k",
        type=int,
        default=5,
        help="Number of most recent checkpoints to keep. Set to -1 to save all checkpoints.",
    )  # DONE
    # parser.add_argument(
    #     "--metric-to-monitor-for-checkpoints",
    #     type=str,
    #     default="val_loss",
    #     help="Metric to monitor for checkpoints.",
    # )  # TODO implement
    # parser.add_argument(
    #     "--save-last-checkpoint",
    #     action="store_true",
    #     default=True,
    #     help="Save the last checkpoint.",
    # )  # TODO implement
    # parser.add_argument(
    #     "--no-save-last-checkpoint",
    #     action="store_false",
    #     dest="save_last_checkpoint",
    #     default=True,
    #     help="Disable saving the last checkpoint.",
    # )  # TODO implement
    # parser.add_argument(
    #     "--lora-finetune", action="store_true", help="Use LoRA fine-tuning", default=False
    # )  # TODO implement
    # parser.add_argument(
    #     "--lora-checkpoint-path", type=str, default=None, help="LoRA checkpoint path"
    # )  # TODO implement
    parser.add_argument(
        "--no-calculate-per-token-loss",
        action="store_true",
        default=False,
        help="Calculate a simpler mean across the microbatch of the loss prior to DDP reduction rather than the global"
        " per-token mean loss. Use this if speed is critical and if you do not need token masking in your loss.",
    )  # DONE
    parser.add_argument(
        "--no-check-for-nan-in-grad",
        action="store_true",
        default=False,
        help="Skip checking for NaNs in gradients. Only use this for debugging purposes.",
    )  # DONE
    parser.add_argument(
        "--garbage-collect-at-inference",
        action="store_true",
        default=False,
        help="Enable CUDA memory cleanup before validation to prevent initialization errors.",
    )  # DONE
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="Alpha parameter for LoRA fine-tuning.",
    )  # TODO implement
    parser.add_argument(
        "--lora-dim",
        type=int,
        default=None,
        help="Dim parameter for LoRA fine-tuning.",
    )  # TODO implement
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug level in logging.",
    )  # DONE
    parser.add_argument(
        "--nvidia-fault-tolerance",
        action="store_true",
        default=False,
        help="Enable NVIDIA fault tolerance. This only works on internal NVIDIA clusters.",
    )  # DONE

    # Optimizer format
    optimizer_fmt_group = parser.add_mutually_exclusive_group(required=False)
    optimizer_fmt_group.add_argument(
        "--optim-fmt-pre-mcore-014",
        action="store_true",
        default=False,
        help="Use the pre-Megatron-Core-v0.14 optimizer format.",
    )
    optimizer_fmt_group.add_argument(
        "--optim-full-reshardable",
        action="store_true",
        default=False,
        help="Enable full optimizer resharding. This is useful for cases where you want to change model parallelism "
        "but keep the same optimizer state.",
    )  # DONE

    recompute_group = parser.add_mutually_exclusive_group(required=False)  # DONE
    recompute_group.add_argument("--no-activation-checkpointing", action="store_true", default=False)  # DONE
    recompute_group.add_argument("--selective-activation-checkpointing", action="store_true", default=False)  # DONE

    mutex_hf_tokenizer_group = parser.add_mutually_exclusive_group(required=False)  # DONE
    mutex_hf_tokenizer_group.add_argument(
        "--hf-tokenizer-model-path", type=Path, help="Path to a local HF tokenizer model."
    )  # DONE
    mutex_hf_tokenizer_group.add_argument(
        "--hf-tokenizer-model-name", type=str, help="Name of a remote HF tokenizer model."
    )  # DONE

    return parser.parse_args(args=args)


def _validate_finetune_ckpt_dir(ckpt_dir: str) -> Path:
    """Validate that a finetune checkpoint directory exists and looks like a valid MBridge checkpoint.

    Args:
        ckpt_dir: Path to the checkpoint directory (may contain ``iter_XXXXXXX`` subdirs
            or be a direct checkpoint directory with ``run_config.yaml``).

    Returns:
        Resolved absolute path to the checkpoint directory.

    Raises:
        FileNotFoundError: If the directory does not exist or is not a valid checkpoint.
    """
    ckpt_path = Path(ckpt_dir).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Finetune checkpoint directory does not exist: {ckpt_path}\n"
            f"  (original path: {ckpt_dir})\n"
            "Please verify the --finetune-ckpt-dir path. If running from a notebook,\n"
            "ensure the path is absolute or relative to the working directory."
        )
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"Finetune checkpoint path is not a directory: {ckpt_path}")

    has_iter_dirs = any(ckpt_path.glob("iter_*"))
    has_run_config = (ckpt_path / "run_config.yaml").exists()
    has_latest_txt = (ckpt_path / "latest_checkpointed_iteration.txt").exists()

    if not (has_iter_dirs or has_run_config or has_latest_txt):
        raise FileNotFoundError(
            f"Finetune checkpoint directory does not look like a valid MBridge checkpoint: {ckpt_path}\n"
            "Expected to find at least one of:\n"
            "  - iter_XXXXXXX/ subdirectories\n"
            "  - run_config.yaml\n"
            "  - latest_checkpointed_iteration.txt"
        )
    return ckpt_path


def main():
    """Parsing args and running Eden training."""
    args = parse_args()
    train(args=args)


def train(args: argparse.Namespace) -> None:
    """Train the Eden model using the Megatron framework."""
    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # 1. Prepare kwargs for the base recipe
    # These kwargs are passed directly to pretrain_config() which handles the core logic
    recipe_kwargs = {}

    # Model
    model_provider = EDEN_MODEL_OPTIONS[args.model_size]
    recipe_kwargs["model_provider"] = model_provider
    logger.info(f"Selected model size: {args.model_size} ({model_provider.__name__})")

    # Tokenizer
    if args.hf_tokenizer_model_path:
        recipe_kwargs["hf_tokenizer_model_or_path"] = args.hf_tokenizer_model_path
    elif args.hf_tokenizer_model_name:
        recipe_kwargs["hf_tokenizer_model_or_path"] = args.hf_tokenizer_model_name
    else:
        recipe_kwargs["hf_tokenizer_model_or_path"] = DEFAULT_HF_TOKENIZER_MODEL_PATH

    # Dataset
    if args.mock_data:
        recipe_kwargs["mock"] = True
    elif args.sharded_eden_data:
        recipe_kwargs["sharded_eden_data"] = True
        recipe_kwargs["sequence_db_dir"] = args.sequence_db_dir
        recipe_kwargs["train_window_db_path"] = args.train_window_db
        recipe_kwargs["val_window_db_path"] = args.val_window_db
        recipe_kwargs["test_window_db_path"] = args.test_window_db
        recipe_kwargs["stride"] = args.stride
        recipe_kwargs["window_min_length_threshold"] = args.window_min_length_threshold
        recipe_kwargs["rc_aug"] = args.rc_aug

    # Parallelism
    recipe_kwargs["tensor_model_parallel_size"] = args.tensor_model_parallel_size
    recipe_kwargs["pipeline_model_parallel_size"] = args.pipeline_model_parallel_size
    recipe_kwargs["context_parallel_size"] = args.context_parallel_size
    recipe_kwargs["sequence_parallel"] = (
        args.tensor_model_parallel_size > 1
    )  # args.sequence_parallel # TODO: remove this arg.

    # Training
    recipe_kwargs["train_iters"] = args.max_steps
    recipe_kwargs["global_batch_size"] = args.global_batch_size
    recipe_kwargs["micro_batch_size"] = args.micro_batch_size
    recipe_kwargs["seq_length"] = args.seq_length
    recipe_kwargs["lr"] = args.lr
    recipe_kwargs["min_lr"] = args.min_lr
    recipe_kwargs["lr_warmup_iters"] = args.warmup_steps
    recipe_kwargs["seed"] = args.seed
    # same as model seed if not provided, but can be overridden.
    recipe_kwargs["dataset_seed"] = args.seed if args.dataset_seed is None else args.dataset_seed
    # Note: weight decay is not in the recipe kwargs signature usually, we set it later.
    recipe_kwargs["precision_config"] = args.mixed_precision_recipe

    if "mxfp8" in args.mixed_precision_recipe or "nvfp4" in args.mixed_precision_recipe:
        # These are required for MXFP8 to work properly.
        args.overlap_param_gather = True
        args.overlap_grad_reduce = True

    # Directories
    if args.result_dir:
        recipe_kwargs["dir"] = args.result_dir
    recipe_kwargs["name"] = args.experiment_name

    if args.no_weight_decay_embeddings:
        recipe_kwargs["no_weight_decay_embeddings"] = True

    # 2. Generate Base Configuration
    cfg: ConfigContainer = pretrain_config(**recipe_kwargs)

    cfg.checkpoint.async_save = args.ckpt_async_save
    cfg.checkpoint.ckpt_format = args.ckpt_format
    cfg.checkpoint.save_interval = args.eval_interval
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.save_rng = True
    cfg.checkpoint.fully_parallel_load = True
    cfg.checkpoint.fully_parallel_save = True
    # cfg.checkpoint.save_tokenizer_assets = True
    cfg.checkpoint.strict_fsdp_dtensor_load = False
    cfg.checkpoint.use_checkpoint_args = False
    cfg.checkpoint.use_persistent_ckpt_worker = True
    cfg.checkpoint.exit_on_missing_checkpoint = False
    cfg.checkpoint.dist_ckpt_strictness = "assume_ok_unexpected"

    if args.no_fp8_wgrad:
        # change if a change is requested to the mixed precision recipe
        cfg.mixed_precision.fp8_wgrad = False
    if args.grad_reduce_in_fp32:
        cfg.mixed_precision.grad_reduce_in_fp32 = True
        cfg.ddp.grad_reduce_in_fp32 = True
    if args.no_fp8_param_gather:
        cfg.mixed_precision.fp8_param_gather = False

    # 3. Apply Manual Overrides (for settings not exposed in recipe kwargs)
    if args.no_renormalize_loss:
        cfg.model.to_upper = "weighted"  # rather than "normalized_weighted"
    if args.seq_len_interpolation_factor is not None:
        cfg.model.seq_len_interpolation_factor = args.seq_len_interpolation_factor
    cfg.model.calculate_per_token_loss = not args.no_calculate_per_token_loss
    # Eden uses TE layers; fp32_residual_connection is not compatible.
    cfg.model.fp32_residual_connection = False
    cfg.model.cross_entropy_loss_fusion = args.cross_entropy_loss_fusion
    # cfg.model.cuda_graph_impl = "local" # or "transformer_engine"
    # cfg.model.cuda_graph_scope = "full_iteration"
    if args.hidden_dropout is not None:
        cfg.model.hidden_dropout = args.hidden_dropout
    if args.attention_dropout is not None:
        cfg.model.attention_dropout = args.attention_dropout
    if args.ffn_hidden_size is not None:
        cfg.model.ffn_hidden_size = args.ffn_hidden_size

    if args.spike_no_more_embedding_init:
        # Spike-no-more-embedding means that the initialization of the embeddings is done with a Normal(0, 1.0)
        #  distribution rather than the default Normal(0, 0.02). This may help avoid loss spiking during training.
        cfg.model.share_embeddings_and_output_weights = False
        cfg.model.embedding_init_method_std = 1.0
    if args.use_targeted_variance_loss:
        cfg.model.use_targeted_variance_loss = True
    if args.num_layers:
        cfg.model.num_layers = args.num_layers

    if args.no_activation_checkpointing:
        cfg.model.recompute_granularity = None
        cfg.model.recompute_method = None
        cfg.model.recompute_num_layers = None
    elif args.selective_activation_checkpointing:
        cfg.model.recompute_granularity = "selective"
        cfg.model.recompute_method = None
        cfg.model.recompute_num_layers = None
    else:
        if args.activation_checkpoint_recompute_num_layers is not None:
            cfg.model.recompute_num_layers = args.activation_checkpoint_recompute_num_layers
    # Optimizer
    if args.wd is not None:
        cfg.optimizer.weight_decay = args.wd
        cfg.scheduler.start_weight_decay = args.wd
        cfg.scheduler.end_weight_decay = args.wd
    cfg.optimizer.adam_beta1 = args.adam_beta1
    cfg.optimizer.adam_beta2 = args.adam_beta2
    cfg.optimizer.adam_eps = args.adam_eps
    cfg.optimizer.use_precision_aware_optimizer = args.use_precision_aware_optimizer
    cfg.optimizer.main_grads_dtype = torch.bfloat16 if args.bf16_main_grads else torch.float32
    cfg.optimizer.log_num_zeros_in_grad = args.log_num_zeros_in_grad
    cfg.optimizer.clip_grad = args.clip_grad
    # Optimizer checkpointing resharding
    if args.optim_fmt_pre_mcore_014:
        cfg.checkpoint.dist_ckpt_save_pre_mcore_014 = True
    elif args.optim_full_reshardable:
        cfg.checkpoint.dist_ckpt_optim_fully_reshardable = True

    cfg.dataset.num_workers = args.workers

    cfg.ddp.average_in_collective = args.average_in_collective
    cfg.ddp.align_param_gather = args.align_param_gather
    cfg.ddp.overlap_param_gather = args.overlap_param_gather
    cfg.ddp.overlap_grad_reduce = args.overlap_grad_reduce
    cfg.ddp.check_for_nan_in_grad = not args.no_check_for_nan_in_grad
    if args.use_megatron_comm_overlap_llama3_8k:
        # Pick the floating point appropriate config.
        fp8 = "fp8" in args.mixed_precision_recipe
        if fp8:
            tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192
        else:
            tp_comm_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
        if cfg.comm_overlap is None:
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True,
                tp_comm_overlap_cfg=tp_comm_overlap_cfg,
                tp_comm_bootstrap_backend=args.tp_comm_overlap_backend,
                wgrad_deferral_limit=22,
                overlap_param_gather_with_optimizer_step=False,
                align_param_gather=args.align_param_gather,
            )
    cfg.train.eval_interval = args.eval_interval
    cfg.train.eval_iters = args.eval_iters

    if args.debug_ddp_parity_freq > 0:
        cfg.train.check_weight_hash_across_dp_replicas_interval = args.debug_ddp_parity_freq
    if args.gc_interval > 0:
        cfg.train.manual_gc = True
        cfg.train.manual_gc_interval = args.gc_interval
    if args.garbage_collect_at_inference:
        cfg.train.manual_gc = True
        cfg.train.manual_gc_eval = True
    if args.enable_preemption:
        cfg.train.exit_signal_handler = True
    # Scheduler
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_warmup_iters = args.warmup_steps
    cfg.scheduler.lr_decay_iters = (
        args.decay_steps if args.decay_steps is not None else args.max_steps - args.warmup_steps - args.constant_steps
    )
    if args.add_bias_output:
        cfg.model.add_bias_output = True
    # Logger & WandB
    if args.log_interval:
        cfg.logger.log_interval = args.log_interval
    if args.disable_tensorboard_logger:
        cfg.logger.tensorboard_dir = None
    if args.wandb_project:
        # Assuming WandbConfig is available in megatron.bridge.training.config
        default_wandb_run_name = (
            f"eden-size-{args.model_size}-TP{args.tensor_model_parallel_size}-"
            f"PP{args.pipeline_model_parallel_size}-CP{args.context_parallel_size}"
            f"-GBS{args.global_batch_size}-MBS{args.micro_batch_size}-SkipLossRenorm{args.no_renormalize_loss}"
            f"-NOAC{args.no_activation_checkpointing}-SELAC{args.selective_activation_checkpointing}"
            f"-ACRNL{cfg.model.recompute_num_layers}"
            f"-F32R{cfg.model.fp32_residual_connection}"
            f"-FCE{cfg.model.cross_entropy_loss_fusion}"
            f"-AIC{cfg.ddp.average_in_collective}"
            f"-PTL{not args.no_calculate_per_token_loss}"
            f"-BO{args.add_bias_output}"
            f"-GCLP{args.clip_grad}"
            f"-HDO{args.hidden_dropout}"
            f"-ADO{args.attention_dropout}"
            f"-LR{args.lr}-MINLR{args.min_lr}-WUSTEPS{args.warmup_steps}-CONSTSTEPS{args.constant_steps}-WD{args.wd}"
            f"-GRFP32{args.grad_reduce_in_fp32}"
            f"-B1{args.adam_beta1}-B2{args.adam_beta2}-EPS{args.adam_eps}"
            f"-PAO{args.use_precision_aware_optimizer}"
            f"-B16MG{args.bf16_main_grads}"
            f"-EWD{args.no_weight_decay_embeddings}-SNI{args.spike_no_more_embedding_init}"
            f"-OGR{args.overlap_grad_reduce}-OPG{args.overlap_param_gather}"
            f"-TVL{args.use_targeted_variance_loss}"
            f"-MPR{args.mixed_precision_recipe}"
        )
        cfg.logger.wandb_project = args.wandb_project
        cfg.logger.wandb_exp_name = args.wandb_run_name or default_wandb_run_name
        cfg.logger.wandb_entity = args.wandb_entity
        # cfg.logger.wandb_save_dir = ...  # FIXME fill this in or decide if the default is ok
        # FIXME consider allowing megatron to specify the run id for regularly restarting slurm jobs.
    # Checkpoint
    # TODO verify that this is the right thing to do here.
    if args.eval_interval:
        cfg.checkpoint.save_interval = args.eval_interval
    cfg.checkpoint.most_recent_k = args.most_recent_k

    if args.finetune_ckpt_dir:
        validated_ckpt_dir = _validate_finetune_ckpt_dir(args.finetune_ckpt_dir)
        cfg.checkpoint.finetune = True
        cfg.checkpoint.pretrained_checkpoint = str(validated_ckpt_dir)
        cfg.checkpoint.dist_ckpt_strictness = "ignore_all"  # necessary unfortunately to avoid extra_state issues.
    if args.nvidia_fault_tolerance:
        cfg.ft = FaultToleranceConfig(
            enable_ft_package=True,
            calc_ft_timeouts=True,
        )
    if args.nsys_profiling:
        """Enable Nsys profiling.
        Example:
            nsys profile -s none -t nvtx,cuda -o <path/to/output_file> --force-overwrite true \
              --capture-range=cudaProfilerApi --capture-range-end=stop
        """
        cfg.profiling.use_nsys_profiler = True
        cfg.profiling.profile_step_start = args.nsys_start_step
        cfg.profiling.profile_step_end = args.nsys_end_step
        cfg.profiling.profile_ranks = args.nsys_ranks
        cfg.profiling.record_memory_history = True
        cfg.profiling.memory_snapshot_path = "memory_snapshot.pickle"
        cfg.profiling.record_shapes = True
        cfg.profiling.nvtx_ranges = True
    # Check for ModelOpt state (restoring from quantized checkpoint)
    if cfg.checkpoint and cfg.checkpoint.load:
        if has_modelopt_state(cfg.checkpoint.load):
            if hasattr(cfg.model, "restore_modelopt_state"):
                cfg.model.restore_modelopt_state = True
                logger.info("Detected ModelOpt state in checkpoint, enabling restore_modelopt_state.")

    # 4. Display or Execute
    if get_rank_safe() == 0:
        logger.info("--- Final Configuration ---")
        cfg.print_yaml()

    logger.info("Starting Eden pretraining...")
    pretrain(cfg, gpt_forward_step)

    if not args.ckpt_async_save:
        # Async checkpoint saving will lazily destroy the process group when the last checkpoint is saved.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
