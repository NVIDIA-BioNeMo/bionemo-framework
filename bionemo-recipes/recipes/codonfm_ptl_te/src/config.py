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


import importlib
from typing import Any, Dict

import fiddle as fdl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from src.data.collate import thd_collate_fn
from src.data.datamodule import CodonFMDataModule
from src.inference.encodon import EncodonInference
from src.models.encodon_pl import EncodonPL
from src.models.encodon_te_pl import EncodonTEPL
from src.tokenizer import Tokenizer
from src.utils.fsdp_config import get_fsdp_strategy
from src.utils.grad_norm_callback import GradientNormLogger
from src.utils.pred_writer import PredWriter
from src.utils.scheduler import linear_scheduler_with_warmup_lr_lambda
from src.utils.timer import StepTimingCallback


# Datasets
def get_dataset_config(args: Any, process_item_cfg: fdl.Partial) -> fdl.Config:
    """Builds the dataset configuration."""
    class_name = args.dataset_name
    if class_name == "CodonMemmapDataset":
        module_path = "src.data.codon_memmap_dataset"
    elif class_name == "MutationDataset":
        module_path = "src.data.mutation_dataset"
    elif class_name == "CodonBertDataset":
        module_path = "src.data.codon_bert_dataset"
    elif class_name == "SimpleCodonDataset":
        module_path = "src.data.simple_codon_dataset"
    else:
        raise ValueError(f"Unknown dataset name: {class_name}")

    try:
        module = importlib.import_module(module_path)
        dataset_class = getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ValueError(f"Could not import dataset '{args.dataset_name}'. Please check the name.") from e

    tokenizer_cfg = fdl.Config(Tokenizer, model_max_length=args.context_length)

    if args.mode == "eval":
        args.train_val_test_ratio = None

    common_args = {
        "data_path": args.data_path,
        "tokenizer": tokenizer_cfg,
        "context_length": args.context_length,
        "train_val_test_ratio": args.train_val_test_ratio,
        "process_item": process_item_cfg,
    }
    if class_name == "CodonMemmapDataset":
        dataset_cfg = fdl.Partial(
            dataset_class,
            **common_args,
            codon_weights_file=getattr(args, "codon_weights_file", None),
            groups_to_use=args.groups_to_use,
            taxid_exclusion_file=getattr(args, "taxid_exclusion_file", None),
            split_name_prefix=getattr(args, "split_name_prefix", ""),
        )
    elif class_name == "MutationDataset":
        dataset_cfg = fdl.Partial(
            dataset_class,
            **common_args,
            label_col=getattr(args, "label_col", None),
            extract_seq=getattr(args, "extract_seq", False),
            ref_seq_col=getattr(args, "ref_seq_col", "ref_seq"),
        )
    elif class_name == "CodonBertDataset":
        dataset_cfg = fdl.Partial(
            dataset_class,
            data_path=args.data_path,
            tokenizer=tokenizer_cfg,
            process_item=process_item_cfg,
        )
    elif class_name == "SimpleCodonDataset":
        # SimpleCodonDataset doesn't need data_path, tokenizer, or most other args
        dataset_cfg = fdl.Partial(
            dataset_class,
            process_item=process_item_cfg,
        )
    else:
        print(f"Warning: Using generic config for dataset '{args.dataset_name}'.")
        dataset_cfg = fdl.Partial(dataset_class, **common_args)

    return dataset_cfg


# Callbacks
def get_callbacks_config(args: Any) -> Dict[str, fdl.Config]:
    """Builds the callbacks configuration."""
    callbacks = {
        "model_checkpoint": fdl.Config(
            ModelCheckpoint,
            dirpath=args.checkpoints_dir,
            save_last=True,
            every_n_train_steps=getattr(args, "checkpoint_every_n_train_steps", 2000),
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        ),
        "early_stopping": fdl.Config(
            EarlyStopping,
            monitor="val/loss",
            patience=100,
            mode="min",
        ),
        "model_summary": fdl.Config(ModelSummary, max_depth=-1),
        "lr_monitor": fdl.Config(LearningRateMonitor, logging_interval="step", log_weight_decay=True),
        "grad_norm_callback": fdl.Config(GradientNormLogger, log_every_n_steps=args.log_every_n_steps),
        "timer_callback": fdl.Config(StepTimingCallback, log_every_n_steps=args.log_every_n_steps, mode="train"),
    }
    if args.mode == "eval":
        callbacks["pred_writer"] = fdl.Config(
            PredWriter,
            output_dir=args.predictions_output_dir,
            write_interval="batch",
            caching_interval=1,
            merge_on_epoch_end=True,
            delete_after_merge=True,
        )
    return callbacks


# Data
def get_data_config(args: Any) -> fdl.Config:
    """Builds the data configuration."""
    if args.process_item == "mlm_memmap":
        from src.data.preprocess.mlm_memmap import process_item as process_item_fn

        process_item_cfg = fdl.Partial(
            process_item_fn,
            mlm_probability=args.mlm_probability,
            mask_replace_prob=args.mask_replace_prob,
            random_replace_prob=args.random_replace_prob,
        )
    elif args.process_item == "mutation_pred_mlm":
        from src.data.preprocess.mutation_pred import mlm_process_item as process_item_fn

        process_item_cfg = fdl.Partial(process_item_fn, mask_mutation=args.mask_mutation)
    elif args.process_item == "mutation_pred_likelihood":
        from src.data.preprocess.mutation_pred import likelihood_process_item as process_item_fn

        process_item_cfg = fdl.Partial(process_item_fn)
    elif args.process_item == "codon_sequence":
        from src.data.preprocess.codon_sequence import process_item as process_item_fn

        process_item_cfg = fdl.Partial(
            process_item_fn,
            context_length=args.context_length,
        )
    else:
        raise ValueError(f"Unknown process_item: {args.process_item}")

    dataset_cfg = get_dataset_config(args, process_item_cfg)

    # Configure collate function based on collate_fn argument
    collate_fn = None
    if hasattr(args, "collate_fn") and args.collate_fn is not None:
        if args.collate_fn == "bshd":
            collate_fn = None
        elif args.collate_fn == "thd":
            collate_fn = thd_collate_fn
        else:
            raise ValueError(f"Unknown collate_fn: {args.collate_fn}")

    return fdl.Config(
        CodonFMDataModule,
        dataset=dataset_cfg,
        train_iters=args.max_steps,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        process_item=process_item_cfg,
        collate_fn=collate_fn,  # ✅ Now passing collate_fn
        pin_memory=False,
        persistent_workers=False,
        world_size=args.num_nodes * args.num_gpus,
        is_evaluation=args.mode == "eval",
    )


# Logger
def get_logger_config(args: Any) -> fdl.Config:
    """Builds the logger configuration."""
    if not getattr(args, "enable_wandb", False):
        return fdl.Config(
            CSVLogger,
            save_dir=args.out_dir,
            name=args.exp_name,
        )
    return fdl.Config(
        WandbLogger,
        name=args.exp_name,
        project=args.project_name,
        entity=args.entity,
        save_dir=args.out_dir,
    )


# Model
MODEL_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "encodon_200k": {
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
    },
    "encodon_80m": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
    },
    "encodon_600m": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 12,
    },
    "encodon_1b": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 18,
    },
    "encodon_5b": {
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
    },
    "encodon_10b": {
        "hidden_size": 5120,
        "intermediate_size": 20480,
        "num_attention_heads": 40,
        "num_hidden_layers": 34,
    },
}


def get_model_config(args: Any) -> fdl.Config:
    """Return the model or inference configuration.

    For training/finetuning, returns an `EncodonPL` or `EncodonTEPL` configuration with optimizer
    and scheduler. For evaluation, returns an `EncodonInference` configuration.
    If Transformer Engine is used, the model configuration is returned as `EncodonTEPL` otherwise `EncodonPL`.

    Args:
        args: Parsed CLI or configuration arguments namespace.

    Returns:
        A Fiddle config that constructs the selected model/inference object.

    Raises:
        ValueError: If the model name or mode is unrecognized.
    """
    arch = MODEL_ARCHITECTURES.get(args.model_name)
    if arch is None:
        raise ValueError(f"Unknown model name: {args.model_name}")

    if args.mode == "pretrain" or args.mode == "finetune":
        scheduler = fdl.Partial(
            torch.optim.lr_scheduler.LambdaLR,
            lr_lambda=fdl.Partial(
                linear_scheduler_with_warmup_lr_lambda,
                total_iterations=args.max_steps,
                warmup_iterations=args.warmup_iterations,
            ),
        )

        return fdl.Config(
            EncodonTEPL if args.use_transformer_engine else EncodonPL,
            optimizer=fdl.Partial(
                torch.optim.AdamW,
                lr=args.lr,
                weight_decay=args.weight_decay,
            ),
            scheduler=scheduler,
            lora=getattr(args, "lora", False) or args.finetune_strategy == "lora",
            lora_alpha=getattr(args, "lora_alpha", 32.0),
            lora_r=getattr(args, "lora_r", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.1),
            finetune_strategy=args.finetune_strategy,
            loss_type=args.loss_type,
            num_classes=getattr(args, "num_classes", 2),
            use_downstream_head=getattr(args, "use_downstream_head", False),
            cross_attention_hidden_dim=getattr(args, "cross_attention_hidden_dim", 256),
            cross_attention_num_heads=getattr(args, "cross_attention_num_heads", 8),
            max_position_embeddings=getattr(args, "context_length", 2048),
            attn_input_format=args.attn_input_format,
            **arch,
        )
    elif args.mode == "eval":
        return fdl.Config(
            EncodonInference,
            model_path=args.checkpoint_path,
            task_type=args.task_type,
            use_transformer_engine=args.use_transformer_engine,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


# Trainer
def get_trainer_config(args: Any) -> Dict[str, Any]:
    """Builds the trainer configuration arguments."""
    # Configure strategy based on args
    if args.enable_fsdp:
        # Use proper FSDP/FSDP2 strategy with auto-wrap policy
        # This ensures FSDP uses LESS memory than DDP
        strategy = get_fsdp_strategy(
            cpu_offload=getattr(args, "fsdp_cpu_offload", False), activation_checkpointing=False, use_fsdp2=True
        )
    elif args.mode == "finetune":
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"

    trainer_kwargs = dict(  # noqa: C408
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        max_steps=args.max_steps,
        default_root_dir=args.out_dir,
        strategy=strategy,
        precision="bf16-mixed" if getattr(args, "bf16", False) else "32-true",
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=False,
        max_epochs=-1,
        min_epochs=1,
        sync_batchnorm=False,
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )
    if args.check_val_every_n_epoch:
        trainer_kwargs["check_val_every_n_epoch"] = args.check_val_every_n_epoch
    else:
        trainer_kwargs["val_check_interval"] = args.val_check_interval

    return trainer_kwargs


# Main config
def get_config(args: Any) -> fdl.Config:
    """Combines the model, data, and trainer configs into a single config."""
    cfg = fdl.Config(dict)
    cfg.model = get_model_config(args)
    cfg.data = get_data_config(args)
    cfg.trainer = get_trainer_config(args)
    cfg.callbacks = get_callbacks_config(args)
    cfg.log = get_logger_config(args)
    return cfg
