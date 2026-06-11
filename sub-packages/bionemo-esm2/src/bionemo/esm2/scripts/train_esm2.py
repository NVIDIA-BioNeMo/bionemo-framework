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
from pathlib import Path
from typing import List, Optional, Sequence, get_args

from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.llm.data.collate import MLM_LOSS_IGNORE_INDEX
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BiobertSpecOption
from bionemo.llm.model.config import TorchmetricsConfig
from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


__all__: Sequence[str] = ("get_parser", "main", "train_esm2_entrypoint")


def main(
    train_cluster_path: Path,
    train_database_path: Path,
    valid_cluster_path: Path,
    valid_database_path: Path,
    num_nodes: int,
    devices: int,
    min_seq_length: Optional[int],
    max_seq_length: int,
    result_dir: Path,
    num_steps: int,
    scheduler_num_steps: Optional[int],
    warmup_steps: int,
    limit_val_batches: int,
    val_check_interval: int,
    log_every_n_steps: Optional[int],
    num_dataset_workers: int,
    biobert_spec_option: BiobertSpecOption,
    lr: float,
    micro_batch_size: int,
    accumulate_grad_batches: int,
    experiment_name: str,
    resume_if_exists: bool,
    precision: PrecisionTypes,
    weight_decay: float = 0.01,
    early_stop_on_step: Optional[int] = None,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = False,
    wandb_tags: Optional[List[str]] = None,
    wandb_group: Optional[str] = None,
    wandb_job_type: Optional[str] = None,
    wandb_id: Optional[str] = None,
    wandb_anonymous: Optional[bool] = False,
    wandb_log_model: bool = False,
    pipeline_model_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    create_tensorboard_logger: bool = False,
    nemo1_init_path: Optional[Path] = None,
    create_tflops_callback: bool = True,
    create_checkpoint_callback: bool = True,
    restore_from_checkpoint_path: Optional[str] = None,
    save_best_checkpoint: bool = True,
    save_last_checkpoint: bool = True,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    save_top_k: int = 2,
    nsys_profiling: bool = False,
    nsys_start_step: int = 0,
    nsys_end_step: Optional[int] = None,
    nsys_ranks: List[int] = [0],
    random_mask_strategy: RandomMaskStrategy = RandomMaskStrategy.ALL_TOKENS,
    num_layers: int = 33,
    hidden_size: int = 1280,
    num_attention_heads: int = 20,
    ffn_hidden_size: int = 1280 * 4,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    average_in_collective: bool = True,
    grad_reduce_in_fp32: bool = False,
    decoder_first_pipeline_num_layers: Optional[int] = None,
) -> nl.Trainer:
    """Train an ESM2 model on UR data.

    Args:
        train_cluster_path (Path): path to train cluster partquet
        train_database_path (Path): path to train database
        valid_cluster_path (Path): path to validation cluster parquet
        valid_database_path (Path): path to validation database
        num_nodes (int): Number of nodes to run on
        devices (int): number of devices
        min_seq_length (Optional[int]): minimum sequence length
        max_seq_length (int): maximum sequence length
        result_dir (Path): directory to store results, logs and checkpoints
        num_steps (int): number of steps to train the model for
        early_stop_on_step (Optional[int]): Stop training on this step, if set. This may be useful for testing or debugging purposes.
        warmup_steps (int): number of steps for warmup phase
        limit_val_batches (int): limit the number of validation global batches to this many
        val_check_interval (int): number of steps to periodically check the validation loss
        log_every_n_steps (Optional[int]): log every n steps
        num_dataset_workers (int): number of dataset workers
        biobert_spec_option (BiobertSpecOption): the biobert spec option (architecture) to use for this run
        lr (float): learning rate
        scheduler_num_steps (Optional[int]): Number of steps in learning rate scheduler. Use num_steps if not provided.
        micro_batch_size (int): micro batch size, from this and parallelism settings we infer the global batch size
        accumulate_grad_batches (int): number of batches to accumulate gradients for
        experiment_name (str): experiment name, this is the name used for the wandb run, and the sub-directory of the
            result_dir that stores the logs and checkpoints.
        resume_if_exists (bool): attempt to resume if the checkpoint exists [FIXME @skothenhill this doesn't work yet]
        precision (PrecisionTypes): Precision type for training (e.g., float16, float32)
        weight_decay (float): weight decay
        wandb_entity (Optional[str]): The team posting this run (default: your username or your default team)
        wandb_project (Optional[str]): The name of the project to which this run will belong
        wandb_offline (bool): Run offline (data can be streamed later to wandb servers).
        wandb_tags (Optional[List[str]]): Tags associated with this run
        wandb_group (Optional[str]): A unique string shared by all runs in a given group
        wandb_job_type (Optional[str]): Type of run, which is useful when you're grouping runs together into larger experiments using group.
        wandb_id (Optional[str]): Sets the version, mainly used to resume a previous run
        wandb_anonymous (Optional[bool]): Enables or explicitly disables anonymous logging
        wandb_log_model (bool): Save checkpoints in wandb dir to upload on W&B servers
        pipeline_model_parallel_size (int): pipeline model parallel size
        tensor_model_parallel_size (int): tensor model parallel size
        create_tensorboard_logger (bool): create the tensorboard logger
        nemo1_init_path (Optional[Path]): Nemo 1 initialization path
        create_tflops_callback (bool): create the FLOPsMeasurementCallback and attach it to the pytorch lightning trainer to log TFlops per training step
        create_checkpoint_callback (bool): create a ModelCheckpoint callback and attach it to the pytorch lightning trainer
        restore_from_checkpoint_path (Optional[str]): If set, restores the model from the directory passed in. Expects the
            checkpoint to be created by using the ModelCheckpoint class and always_save_context=True.
        save_best_checkpoint (bool): whether to save the best checkpoint
        save_last_checkpoint (bool): whether to save the last checkpoint
        metric_to_monitor_for_checkpoints (str): metric to monitor for checkpoints
        save_top_k (int): number of top checkpoints to save
        nsys_profiling (bool): whether to enable nsys profiling
        nsys_start_step (int): start step for nsys profiling
        nsys_end_step (Optional[int]): end step for nsys profiling
        nsys_ranks (List[int]): ranks for nsys profiling
        random_mask_strategy (RandomMaskStrategy): random mask strategy
        num_layers (int): number of layers
        hidden_size (int): hidden size
        num_attention_heads (int): number of attention heads
        ffn_hidden_size (int): feed forward hidden size
        overlap_grad_reduce (bool): overlap gradient reduction
        overlap_param_gather (bool): overlap parameter gather
        average_in_collective (bool): average in collective
        grad_reduce_in_fp32 (bool): gradient reduction in fp32
        decoder_first_pipeline_num_layers (Optional[int]): number of layers in the decoder first pipeline. Default None is even split of transformer layers across all pipeline stages
    """
    # Create the result directory if it does not exist.
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup the strategy and trainer
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        accumulate_grad_batches=accumulate_grad_batches,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    tokenizer = get_tokenizer()

    # Initialize the data module.
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=train_database_path,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=valid_database_path,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        num_workers=num_dataset_workers,
        random_mask_strategy=random_mask_strategy,
        tokenizer=tokenizer,
    )
    # Configure the model
    train_metric = None
    is_model_parallel = tensor_model_parallel_size * pipeline_model_parallel_size > 1
    if is_model_parallel:
        valid_metric = None  # metric logging under model parallelism is not supported yet
    else:
        valid_metric = TorchmetricsConfig(
            class_path="text.Perplexity",
            task="pretraining",
            kwargs={"ignore_index": MLM_LOSS_IGNORE_INDEX},
            metric_name="val_ppl",
        )

    # Set decoder_first_pipeline_num_layers if needed and not provided
    if num_layers % pipeline_model_parallel_size != 0 and decoder_first_pipeline_num_layers is None:
        decoder_first_pipeline_num_layers = num_layers - int(num_layers / pipeline_model_parallel_size + 0.5) * (
            pipeline_model_parallel_size - 1
        )

    esm2_config = ESM2Config(
        seq_length=max_seq_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=ffn_hidden_size,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=str(nemo1_init_path) if nemo1_init_path is not None else None,
        # handle checkpoint resumption here rather than auto-resume so this supports fine-tuning capabilities
        initial_ckpt_path=str(restore_from_checkpoint_path) if restore_from_checkpoint_path is not None else None,
        variable_seq_lengths=min_seq_length != max_seq_length,
        train_metric=train_metric,
        valid_metric=valid_metric,
        num_layers_in_first_pipeline_stage=decoder_first_pipeline_num_layers,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    if scheduler_num_steps is None:
        scheduler_num_steps = num_steps

    model = biobert_lightning_module(
        esm2_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=lr,
                optimizer="adam",
                use_distributed_optimizer=True,
                weight_decay=weight_decay,
                adam_beta1=0.9,
                adam_beta2=0.98,
            ),
            lr_scheduler=WarmupAnnealDecayHoldScheduler(
                warmup_steps=warmup_steps,
                max_steps=scheduler_num_steps,
                max_lr=lr,
                min_lr=0.0,
                anneal_percentage=0.10,
            ),
        ),
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=get_autocast_dtype(precision),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            average_in_collective=average_in_collective,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            use_distributed_optimizer=True,
        ),
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        num_layers_in_first_pipeline_stage=decoder_first_pipeline_num_layers,
    )

    # for wandb integration
    # Please refer to https://pytorch-lightning.readthedocs.io/en/0.7.6/api/lightning.pytorch.loggers.html"
    wandb_config: Optional[WandbConfig] = (
        None
        if wandb_project is None
        else WandbConfig(
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            tags=wandb_tags,
            group=wandb_group,
            job_type=wandb_job_type,
            id=wandb_id,
            anonymous=wandb_anonymous,
            log_model=wandb_log_model,
        )
    )

    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        nl_callbacks.PreemptionCallback(),
        TimingCallback(),
    ]

    if nsys_profiling:
        if nsys_end_step is None:
            nsys_end_step = num_steps
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=nsys_start_step, end_step=nsys_end_step, ranks=nsys_ranks, gen_shape=True
            )
        )

    if create_tflops_callback:
        # Add callback that logs the tera-FLOPS per second per GPU during training.
        data_module.global_batch_size = (
            global_batch_size  # TODO(dorotat): remove this change after FLOPsMeasurementCallback is refactored
        )
        flop_meas_callback = FLOPsMeasurementCallback(
            esm2_config,
            data_module,
            "bert",
        )
        callbacks.append(flop_meas_callback)

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_config=wandb_config,
    )

    # Configure our custom ModelCheckpointe callback and AutoResume to save at nemo_logger.save_dir/checkpoints
    if create_checkpoint_callback:
        checkpoint_path = str(Path(nemo_logger.save_dir) / "checkpoints")
        checkpoint_callback = nl_callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
            save_last=save_last_checkpoint,
            monitor=metric_to_monitor_for_checkpoints,  # "val_loss",
            save_top_k=save_top_k,
            every_n_train_steps=val_check_interval,
            always_save_context=True,
            # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
            filename="{epoch}-{step}-{consumed_samples}",
            # Including step and consumed_samples in the checkpoint filename prevents duplicate filenames and bugs related to this.
            # Save both the weights and the optimizer state.
            save_weights_only=False,
            save_optim_on_train_end=True,
        )

        callbacks.append(checkpoint_callback)

        auto_resume = resume.AutoResume(
            resume_from_directory=checkpoint_path,
            resume_if_exists=resume_if_exists,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
            resume_past_end=False,
        )
    else:
        auto_resume = None

    trainer = nl.Trainer(
        devices=devices,
        max_steps=num_steps if early_stop_on_step is None else early_stop_on_step,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        num_nodes=num_nodes,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(
            precision=precision,
            params_dtype=get_autocast_dtype(precision),
            pipeline_dtype=get_autocast_dtype(precision),
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            autocast_enabled=False,
        ),
        enable_checkpointing=create_checkpoint_callback,
    )

    llm.train(
        model=model,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=auto_resume,
    )
    return trainer


def train_esm2_entrypoint():
    """Entrypoint for running inference on a geneformer checkpoint and data."""
    # 1. get arguments
    parser = get_parser()
    args = parser.parse_args()
    # 2. Call pretrain with args
    main(
        train_cluster_path=args.train_cluster_path,
        train_database_path=args.train_database_path,
        valid_cluster_path=args.valid_cluster_path,
        valid_database_path=args.valid_database_path,
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        result_dir=args.result_dir,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_id=args.wandb_id,
        wandb_anonymous=args.wandb_anonymous,
        wandb_log_model=args.wandb_log_model,
        wandb_offline=args.wandb_offline,
        num_steps=args.num_steps,
        early_stop_on_step=args.early_stop_on_step,
        warmup_steps=args.warmup_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        num_dataset_workers=args.num_dataset_workers,
        biobert_spec_option=args.biobert_spec_option,
        lr=args.lr,
        scheduler_num_steps=args.scheduler_num_steps,
        micro_batch_size=args.micro_batch_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        experiment_name=args.experiment_name,
        resume_if_exists=args.resume_if_exists,
        nemo1_init_path=args.nemo1_init_path,
        create_tflops_callback=args.create_tflops_callback,
        create_checkpoint_callback=args.create_checkpoint_callback,
        create_tensorboard_logger=args.create_tensorboard_logger,
        restore_from_checkpoint_path=args.restore_from_checkpoint_path,
        save_best_checkpoint=args.save_best_checkpoint,
        save_last_checkpoint=args.save_last_checkpoint,
        metric_to_monitor_for_checkpoints=args.metric_to_monitor_for_checkpoints,
        save_top_k=args.save_top_k,
        nsys_profiling=args.nsys_profiling,
        nsys_start_step=args.nsys_start_step,
        nsys_end_step=args.nsys_end_step,
        nsys_ranks=args.nsys_ranks,
        random_mask_strategy=args.random_mask_strategy,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        overlap_grad_reduce=not args.no_overlap_grad_reduce,
        overlap_param_gather=not args.no_overlap_param_gather,
        average_in_collective=not args.no_average_in_collective,
        grad_reduce_in_fp32=args.grad_reduce_in_fp32,
        decoder_first_pipeline_num_layers=args.decoder_first_pipeline_num_layers,
    )


def get_parser():
    """Return the cli parser for this tool."""
    # TODO migrate to hydra config
    # Parse the arguments and pull them out into local variables for ease of future refactor to a
    #   config management system.
    parser = argparse.ArgumentParser(description="Pretrain ESM2 with UR data.")
    parser.add_argument(
        "--train-cluster-path",
        type=Path,
        required=True,
        help="Path to the train cluster data parquet file",
    )
    parser.add_argument(
        "--train-database-path",
        type=Path,
        required=True,
        help="Path to the train sequence database file",
    )
    parser.add_argument(
        "--valid-cluster-path",
        type=Path,
        required=True,
        help="Path to the valid cluster data parquet file",
    )
    parser.add_argument(
        "--valid-database-path",
        type=Path,
        required=True,
        help="Path to the vali sequence database file",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=get_args(PrecisionTypes),
        required=False,
        default="bf16-mixed",
        help="Precision type to use for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=4e-4,
        help="Learning rate for training. Default is 4e-4",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        required=False,
        default=0.01,
        help="Weight decay for training. Default is 0.01",
    )
    parser.add_argument(
        "--scheduler-num-steps",
        type=int,
        required=False,
        help="Number of steps for learning rate scheduler. Will use --num-steps if not given. Default is None.",
    )
    parser.add_argument(
        "--create-tflops-callback",
        action="store_true",
        default=False,
        help="Enable tflops calculation callback for Hyena / Evo2. Defaults to False.",
    )
    parser.add_argument(
        "--create-tensorboard-logger", action="store_true", default=False, help="Create a tensorboard logger."
    )
    # FIXME (@skothenhill) figure out how checkpointing and resumption should work with the new nemo trainer
    parser.add_argument(
        "--resume-if-exists", action="store_true", default=False, help="Resume training if a checkpoint exists."
    )
    parser.add_argument(
        "--result-dir", type=Path, required=False, default=Path("./results"), help="Path to the result directory."
    )
    parser.add_argument("--experiment-name", type=str, required=False, default="esm2", help="Name of the experiment.")

    parser.add_argument("--wandb-entity", type=str, default=None, help="The team posting this run")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name ")
    parser.add_argument("--wandb-tags", nargs="+", type=str, default=None, help="Tags associated with this run")
    parser.add_argument(
        "--wandb-group", type=str, default=None, help="A unique string shared by all runs in a given group"
    )
    parser.add_argument(
        "--wandb-job-type",
        type=str,
        default=None,
        help="A unique string representing a type of run, which is useful when you're grouping runs together into larger experiments using group.",
    )
    parser.add_argument(
        "--wandb-id", type=str, default=None, help="Sets the version, mainly used to resume a previous run"
    )
    parser.add_argument(
        "--wandb-anonymous", action="store_true", help="Enable or explicitly disable anonymous logging"
    )
    parser.add_argument(
        "--wandb-log-model", action="store_true", help="Save checkpoints in wandb dir to upload on W&B servers"
    )
    parser.add_argument("--wandb-offline", action="store_true", help="Use wandb in offline mode")
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=False,
        default=1,
        help="Number of GPUs to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=False,
        default=1,
        help="Number of nodes to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=500000,
        help="Number of steps to use for training. Default is 500000.",
    )
    parser.add_argument(
        "--early-stop-on-step",
        type=int,
        default=None,
        help="Stop training on this step, if set. This may be useful for testing or debugging purposes.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        required=False,
        default=2000,
        help="Number of warmup steps for WarmupAnnealDecayHold Scheduler. Default is 2000.",
    )
    parser.add_argument(
        "--num-dataset-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to use for training. Default is 1.",
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        required=False,
        default=10000,
        help="Number of steps between validation. Default is 10000.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        required=False,
        help="Number of steps between logging. Default is 50.",
    )
    parser.add_argument(
        "--min-seq-length",
        type=float_or_int_or_none,
        required=False,
        default=1024,
        help="Minimum sequence length. Sampled will be padded if less than this value. Set 'None' to unset minimum.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        required=False,
        default=1024,
        help="Maximum sequence length. Samples will be truncated if exceeds this value.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float_or_int_or_none,
        required=False,
        default=2,
        help="Number of global batches used for validation if int. Fraction of validation dataset if float. Default is 2.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=64,
        help="Micro-batch size. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Pipeline model parallel size. Default is 1.",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Tensor model parallel size. Default is 1.",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        required=False,
        default=1,
        help="Gradient accumulation steps. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--biobert-spec-option",
        type=BiobertSpecOption,
        choices=[e.value for e in BiobertSpecOption],
        required=False,
        default=BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec.value,
        help="Biobert spec option to use for the model. Default is 'esm2_bert_layer_with_transformer_engine_spec'.",
    )
    parser.add_argument(
        "--nemo1-init-path",
        type=Path,
        required=False,
        help="Path to nemo1 file, if desired to load at init time.",
    )
    parser.add_argument(
        "--disable-checkpointing",
        action="store_false",
        default=True,
        dest="create_checkpoint_callback",
        help="Disable creating a ModelCheckpoint callback.",
    )
    parser.add_argument(
        "--save-best-checkpoint",
        action="store_true",
        default=True,
        help="Save the best checkpoint based on the metric to monitor.",
    )
    parser.add_argument(
        "--no-save-best-checkpoint",
        action="store_false",
        default=True,
        dest="save_best_checkpoint",
        help="Disable saving the best checkpoint based on the metric to monitor.",
    )
    parser.add_argument(
        "--save-last-checkpoint",
        action="store_true",
        default=True,
        help="Save the last checkpoint.",
    )
    parser.add_argument(
        "--no-save-last-checkpoint",
        action="store_false",
        dest="save_last_checkpoint",
        default=True,
        help="Disable saving the last checkpoint.",
    )
    parser.add_argument(
        "--metric-to-monitor-for-checkpoints",
        type=str,
        required=False,
        default="val_loss",
        help="The metric to monitor for checkpointing.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        required=False,
        default=2,
        help="Save the top k checkpoints.",
    )
    parser.add_argument(
        "--restore-from-checkpoint-path",
        type=Path,
        required=False,
        default=None,
        help="Path to the checkpoint directory to restore from. Will override `--resume-if-exists` when set.",
    )
    parser.add_argument(
        "--nsys-profiling",
        action="store_true",
        default=False,
        help="Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling output you must run the whole program with `nsys`. For example: "
        " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`",
    )
    # start, end, rank
    parser.add_argument(
        "--nsys-start-step",
        type=int,
        required=False,
        default=0,
        help="Start nsys profiling after this step.",
    )
    parser.add_argument(
        "--nsys-end-step",
        type=int,
        required=False,
        help="End nsys profiling after this step.",
    )
    # rank as list of integers
    parser.add_argument(
        "--nsys-ranks",
        type=int,
        nargs="+",
        required=False,
        default=[0],
        help="Enable nsys profiling for these ranks.",
    )

    # ESM2 specific configuration (default: 650M)
    parser.add_argument(
        "--random-mask-strategy",
        type=RandomMaskStrategy,
        choices=[e.value for e in RandomMaskStrategy],
        default=RandomMaskStrategy.ALL_TOKENS.value,
        help=f"""In ESM2 pretraining, 15%% of all tokens are masked and among which 10%% are replaced with a random token. This class controls the set of random tokens to choose from. Options are: '{"', '".join([e.value for e in RandomMaskStrategy])}'. Note that 'all_token' will introduce non-canonical amino acid tokens as effective mask tokens, and the resultant loss will appear lower than that from 'amino_acids_only'. Note that 'all_token' is the method used in hugging face as well as portions of fairseq.""",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        required=False,
        default=33,
        help="Number of layers in the model. Default is 33.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        required=False,
        default=1280,
        help="Hidden size of the model. Default is 1280.",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        required=False,
        default=20,
        help="Number of attention heads in the model. Default is 20.",
    )
    parser.add_argument(
        "--ffn-hidden-size",
        type=int,
        required=False,
        default=4 * 1280,
        help="FFN hidden size of the model. Default is 4 * 1280.",
    )
    # DDP config
    parser.add_argument(
        "--no-overlap-grad-reduce",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-overlap-param-gather",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-average-in-collective",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--grad-reduce-in-fp32",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--decoder-first-pipeline-num-layers",
        type=int,
        required=False,
        default=None,
        help="The number of transformer layers on the first pipeline stage of the decoder. Default None is even split of transformer layers across all pipeline stages",
    )
    return parser


if __name__ == "__main__":
    train_esm2_entrypoint()
