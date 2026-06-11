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

from pathlib import Path
from typing import List, Optional

import typer
from datasets import load_dataset as hf_load_dataset
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

from bionemo.amplify.datamodule import AMPLIFYDataModule
from bionemo.amplify.model import AMPLIFYConfig
from bionemo.amplify.tokenizer import BioNeMoAMPLIFYTokenizer
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.llm.data.collate import MLM_LOSS_IGNORE_INDEX
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BiobertSpecOption
from bionemo.llm.model.config import TorchmetricsConfig
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


app = typer.Typer()


@app.command()
def main(
    num_nodes: int = 1,
    devices: int = 1,
    min_seq_length: Optional[int] = 512,
    max_seq_length: int = 512,
    result_dir: Path = Path("./results"),
    num_steps: int = 1_000_000,
    early_stop_on_step: Optional[int] = None,
    warmup_steps: int = 1000,
    decay_steps: int = 900_000,
    limit_val_batches: float = 1.0,
    val_check_interval: int = 10000,
    log_every_n_steps: Optional[int] = 100,
    num_dataset_workers: int = 27,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec,
    lr: float = 1e-3,
    micro_batch_size: int = 64,
    accumulate_grad_batches: int = 1,
    experiment_name: str = "amplify",
    resume_if_exists: bool = False,
    precision: str = "bf16-mixed",
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = False,
    wandb_tags: Optional[List[str]] = None,
    wandb_group: Optional[str] = None,
    wandb_id: Optional[str] = None,
    wandb_job_type: Optional[str] = None,
    wandb_anonymous: bool = False,
    wandb_log_model: bool = False,
    pipeline_model_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    create_tensorboard_logger: bool = False,
    create_tflops_callback: bool = True,
    create_checkpoint_callback: bool = True,
    nemo1_init_path: Optional[Path] = None,
    restore_from_checkpoint_path: Optional[str] = None,
    save_last_checkpoint: bool = True,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    save_top_k: int = 2,
    nsys_profiling: bool = False,
    nsys_start_step: int = 0,
    nsys_end_step: Optional[int] = None,
    nsys_ranks: List[int] = [0],
    random_mask_strategy: RandomMaskStrategy = RandomMaskStrategy.ALL_TOKENS,
    num_layers: int = 24,
    hidden_size: int = 640,
    num_attention_heads: int = 10,
    ffn_hidden_size: int = 2560,
    no_overlap_grad_reduce: bool = False,
    overlap_param_gather: bool = False,
    no_average_in_collective: bool = False,
    grad_reduce_in_fp32: bool = False,
    use_sanity_dataset: bool = False,
) -> nl.Trainer:
    """Train an AMPLIFY model on UR100P data.

    Args:
        num_nodes (int): Number of nodes to run on
        devices (int): number of devices
        min_seq_length (Optional[int]): Whether to pad sequences to a minimum length. If None, no extra padding is added
        max_seq_length (int): The maximum sequence length for the AMPLIFY transformer
        result_dir (Path): directory to store results, logs and checkpoints
        num_steps (int): number of steps to train the model for
        early_stop_on_step (Optional[int]): Stop training on this step, if set. This may be useful for testing or debugging purposes.
        warmup_steps (int): number of steps for the learning rate warmup phase
        decay_steps (int): number of steps for the learning rate decay phase
        limit_val_batches (int): limit the number of validation global batches to this many
        val_check_interval (int): number of steps to periodically check the validation loss and save
        log_every_n_steps (Optional[int]): frequency for logging (steps)
        num_dataset_workers (int): num dataset workers
        biobert_spec_option (BiobertSpecOption): the biobert spec option (architecture) to use for this run
        lr (float): learning rate
        micro_batch_size (int): micro batch size, from this and parallelism settings we infer the global batch size
        accumulate_grad_batches (int): number of batches to accumulate before performing a gradient update
        experiment_name (str): experiment name, this is the name used for the wandb run, and the sub-directory of the
            result_dir that stores the logs and checkpoints.
        resume_if_exists (bool): attempt to resume if the checkpoint exists [FIXME @skothenhill this doesn't work yet]
        precision (PrecisionTypes): precision to use for training (bf16-mixed, 16-mixed, 32)
        wandb_entity (str): The team posting this run (default: your username or your default team)
        wandb_project (str): The name of the project to which this run will belong.
        wandb_tags (List[str]): Tags associated with this run.
        wandb_group (str): A unique string shared by all runs in a given group
        wandb_offline (bool): Run offline (data can be streamed later to wandb servers).
        wandb_id (str): Sets the version, mainly used to resume a previous run.
        wandb_job_type (str): A unique string representing a type of run, which is useful when you're grouping runs together into larger experiments using group.
        wandb_anonymous (bool): Enables or explicitly disables anonymous logging.
        wandb_log_model (bool): Save checkpoints in wandb dir to upload on W&B servers.
        pipeline_model_parallel_size (int): degree of pipeline model parallelism
        tensor_model_parallel_size (int): degree of tensor model parallelism
        create_tensorboard_logger (bool): create the tensorboard logger
        create_tflops_callback (bool): create the FLOPsMeasurementCallback and attach it to the pytorch lightning trainer to log TFlops per training step
        create_checkpoint_callback (bool): create a ModelCheckpoint callback and attach it to the pytorch lightning trainer
        nemo1_init_path (Optional[Path]): path to a NeMo v1 checkpoint to initialize from
        restore_from_checkpoint_path (Optional[str]): If set, restores the model from the directory passed in. Expects the
            checkpoint to be created by using the ModelCheckpoint class and always_save_context=True.
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
        no_overlap_grad_reduce (bool): disable overlap gradient reduction
        overlap_param_gather (bool): overlap parameter gather
        no_average_in_collective (bool): disable average in collective
        grad_reduce_in_fp32 (bool): gradient reduction in fp32
        use_sanity_dataset (bool): use a smaller, streaming version of the AMPLIFY dataset for profiling / testing.
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

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=get_autocast_dtype(precision),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=not no_overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            average_in_collective=not no_average_in_collective,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            use_distributed_optimizer=True,
        ),
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
    )

    # for wandb integration
    # Please refer to https://pytorch-lightning.readthedocs.io/en/0.7.6/api/pytorch_lightning.loggers.html"
    wandb_config: Optional[WandbConfig] = (
        None
        if wandb_project is None
        else WandbConfig(
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            tags=wandb_tags,
            group=wandb_group,
            id=wandb_id,
            job_type=wandb_job_type,
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

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_config=wandb_config,
    )

    # Configure our custom Checkpointer
    if create_checkpoint_callback:
        checkpoint_path = str(Path(nemo_logger.save_dir) / "checkpoints")
        checkpoint_callback = nl_callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
            save_last=save_last_checkpoint,
            monitor=metric_to_monitor_for_checkpoints,  # "val_loss",
            save_top_k=save_top_k,
            every_n_train_steps=val_check_interval,
            always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
            filename="{epoch}-{step}-{consumed_samples}",
            # Including step and consumed_samples in the checkpoint filename prevents duplicate filenames and bugs related to this.
        )
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None

    tokenizer = BioNeMoAMPLIFYTokenizer()

    # Initialize the data module. hf_load_dataset loads these datasets from the huggingface hub if they're not available
    # locally, but the download and pre-processing can take a while.
    if use_sanity_dataset:
        train_hf_dataset = hf_load_dataset(
            "chandar-lab/UR100P",
            split="train",
            revision="refs/convert/parquet",
            data_files="default/partial-train/0001.parquet",
        )
        valid_hf_dataset = hf_load_dataset(
            "chandar-lab/UR100P",
            split="test",
            revision="refs/convert/parquet",
        )
    else:
        train_hf_dataset = hf_load_dataset("chandar-lab/UR100P", split="train")
        valid_hf_dataset = hf_load_dataset("chandar-lab/UR100P", data_dir="UniProt", split="test")

    data = AMPLIFYDataModule(
        train_hf_dataset=train_hf_dataset,  # type: ignore
        valid_hf_dataset=valid_hf_dataset,  # type: ignore
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

    amplify_config = AMPLIFYConfig(
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
    )

    if create_tflops_callback:
        # Add callback that logs the tera-FLOPS per second per GPU during training.
        data.global_batch_size = global_batch_size
        flop_meas_callback = FLOPsMeasurementCallback(
            amplify_config,
            data,
            "bert",
        )
        callbacks.append(flop_meas_callback)

    model = biobert_lightning_module(
        amplify_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=lr,
                optimizer="adam",  # fused_adam not supported
                use_distributed_optimizer=True,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.95,
                clip_grad=1.0,
            ),
            lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(
                min_lr=0.1 * lr,
                max_steps=decay_steps,
                warmup_steps=warmup_steps,
                constant_steps=0,
            ),
        ),
    )

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
        enable_checkpointing=create_checkpoint_callback,
        plugins=nl.MegatronMixedPrecision(
            precision=precision,
            params_dtype=get_autocast_dtype(precision),
            pipeline_dtype=get_autocast_dtype(precision),
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            autocast_enabled=False,
        ),
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=resume_if_exists,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )

    return trainer


if __name__ == "__main__":
    app()
