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


import argparse
import logging
import os

from dotenv import load_dotenv


load_dotenv()

import fiddle as fdl  # noqa: E402

from src.config import get_config  # noqa: E402
from src.tasks import evaluate, finetune, train  # noqa: E402
from src.utils.nemorun_utils import config_to_dict  # noqa: E402


log = logging.getLogger(__name__)


def get_parser():  # noqa: D103
    parser = argparse.ArgumentParser(description="Codon-FM Runner Script")
    parser.add_argument("mode", choices=["pretrain", "finetune", "eval"], help="Mode to run.")
    # General arguments
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dryrun", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default=None, help="Weights & Biases project name.")
    parser.add_argument("--entity", type=str, default=None, help="Weights & Biases entity.")
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging.",
    )

    # Container-like path overrides
    parser.add_argument("--out_dir", type=str, default="results/", help="Base output directory.")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="Checkpoints directory. Defaults to <out_dir>/checkpoints.",
    )
    parser.add_argument(
        "--pretrained_ckpt_path",
        type=str,
        default=None,
        help="Path to pretrained checkpoint. Defaults to --checkpoint_path if set.",
    )

    # Data arguments
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to dataset (not required for SimpleCodonDataset)"
    )
    parser.add_argument(
        "--process_item",
        type=str,
        required=True,
        choices=[
            "mlm_memmap",
            "mutation_pred_mlm",
            "mutation_pred_likelihood",
            "codon_sequence",
        ],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["CodonMemmapDataset", "MutationDataset", "CodonBertDataset", "SimpleCodonDataset"],
    )
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--groups_to_use", type=str, nargs="+", default=[])
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--train_val_test_ratio", type=float, nargs=3, default=[0.9998, 0.0002, 0.00])
    parser.add_argument("--taxid_exclusion_file", type=str, default=None)
    parser.add_argument("--split_name_prefix", type=str, default="")
    parser.add_argument(
        "--collate_fn",
        type=str,
        default=None,
        choices=["bshd", "thd"],
        help="Collate function to use for batching. If None, uses PyTorch's default collate.",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            "encodon_200k",
            "encodon_80m",
            "encodon_600m",
            "encodon_1b",
            "encodon_5b",
            "encodon_10b",
        ],
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=10_000_000)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--mask_replace_prob", type=float, default=0.8)
    parser.add_argument("--random_replace_prob", type=float, default=0.1)
    parser.add_argument("--mask_mutation", action="store_true", default=False)
    parser.add_argument("--warmup_iterations", type=int, default=10_000)
    parser.add_argument("--use_transformer_engine", action="store_true", default=False)
    parser.add_argument(
        "--attn_input_format",
        type=str,
        default="bshd",
        choices=["bshd", "thd"],
        help="Attention input format.",
    )

    # Pretrain specific
    parser.add_argument("--codon_weights_file", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", default=False)

    # Eval specific
    parser.add_argument(
        "--extract-seq",
        action="store_true",
        default=False,
        help="For mutation prediction, whether to extract sequence.",
    )
    parser.add_argument(
        "--predictions_output_dir",
        type=str,
        default=None,
        help="For evaluation, the directory to write predictions to.",
    )
    parser.add_argument("--task_type", type=str, default=None, help="For evaluation, the task type to run.")

    # Finetune specific
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint for finetuning or evaluation.",
    )
    parser.add_argument("--loss_type", choices=["regression", "classification"], default="regression")
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--ref_seq_col", type=str, default="ref_seq")
    parser.add_argument("--resume_trainer_state", action="store_true", default=False)
    parser.add_argument("--checkpoint_every_n_train_steps", type=int, default=2000)
    parser.add_argument(
        "--finetune_strategy",
        type=str,
        default="full",
        choices=["lora", "head_only_random", "head_only_pretrained", "full"],
        help="Finetuning strategy.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes for classification tasks.",
    )
    parser.add_argument(
        "--use_downstream_head",
        action="store_true",
        default=False,
        help="Whether to use downstream cross-attention head.",
    )
    parser.add_argument(
        "--cross_attention_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for cross attention.",
    )
    parser.add_argument(
        "--cross_attention_num_heads",
        type=int,
        default=8,
        help="Number of heads for cross attention.",
    )

    # Common trainer flags
    parser.add_argument("--enable_fsdp", action="store_true", default=False)
    parser.add_argument("--fsdp_cpu_offload", action="store_true", default=False)
    parser.add_argument("--val_check_interval", type=int, default=1000)
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=None,
        help="Run validation every n epochs. Overrides val_check_interval.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before performing a weight update.",
    )

    parser.add_argument("--limit_val_batches", type=int, default=50)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    return parser


def main():  # noqa: D103
    parser = get_parser()
    args = parser.parse_args()
    if args.mode in ["eval"] and not args.checkpoint_path:
        parser.error(f"--checkpoint_path is required for mode '{args.mode}'")

    # Validate data_path requirement based on dataset
    if args.dataset_name != "SimpleCodonDataset" and not args.data_path:
        parser.error(f"--data_path is required for dataset '{args.dataset_name}'")

    if args.enable_wandb:
        missing = []
        if not args.project_name:
            missing.append("--project_name")
        if not args.entity:
            missing.append("--entity")
        if missing:
            parser.error(f"{', '.join(missing)} is required when --enable_wandb is set")

    if (args.attn_input_format == "thd" or args.collate_fn == "thd") and not args.use_transformer_engine:
        raise ValueError("THD format requires transformer engine")
    cfg = get_config(args)

    out_dir = args.out_dir
    checkpoints_dir = args.checkpoints_dir if args.checkpoints_dir else os.path.join(out_dir, "checkpoints")
    pretrained_ckpt_path = args.pretrained_ckpt_path

    # exp_name = args.mode + "_" + args.exp_name
    exp_name = args.exp_name

    cfg_dict = config_to_dict(cfg)
    config_built = fdl.build(cfg)
    # WandB handled directly by Lightning loggers via config; no external plugins needed
    if not (args.enable_wandb and "WANDB_API_KEY" in os.environ):
        log.info("WandB disabled or WANDB_API_KEY not found. Skipping WandB logging.")

    cfg_dict["seed"] = args.seed
    cfg_dict["out_dir"] = out_dir
    # Define task callable and kwargs
    if args.mode == "pretrain":
        ckpt_path = os.path.join(checkpoints_dir, "last.ckpt")
        cfg_dict["ckpt_path"] = ckpt_path
        task_fn = train
        task_kwargs = dict(  # noqa: C408
            config=config_built,
            ckpt_path=ckpt_path,
            seed=args.seed,
            config_dict=cfg_dict,
            out_dir=out_dir,
        )
    elif args.mode == "finetune":
        if args.attn_input_format == "thd" or args.collate_fn == "thd":
            raise ValueError("THD format is not supported for finetuning")
        ckpt_path = os.path.join(checkpoints_dir, "last.ckpt")
        cfg_dict["ckpt_path"] = ckpt_path
        cfg_dict["pretrained_ckpt_path"] = args.checkpoint_path
        cfg_dict["resume_trainer_state"] = args.resume_trainer_state
        task_fn = finetune
        task_kwargs = dict(  # noqa: C408
            config=config_built,
            pretrained_ckpt_path=pretrained_ckpt_path,
            seed=args.seed,
            resume_trainer_state=args.resume_trainer_state,
            config_dict=cfg_dict,
            out_dir=out_dir,
            ckpt_path=ckpt_path,
        )
    elif args.mode == "eval":
        task_fn = evaluate
        task_kwargs = dict(  # noqa: C408
            config=config_built,
            config_dict=cfg_dict,
            model_ckpt_path=args.checkpoint_path,
            seed=args.seed,
            out_dir=out_dir,
        )

    if args.dryrun:
        log.info("Dryrun mode: configuration constructed; skipping execution.")
        return

    log.info(f"Starting job: {exp_name}")
    task_fn(**task_kwargs)


if __name__ == "__main__":
    main()
