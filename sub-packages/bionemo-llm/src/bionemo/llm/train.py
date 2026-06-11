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


import math
import pathlib
from dataclasses import field
from types import SimpleNamespace
from typing import Optional

from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.utils import logging
from nemo.utils.exp_manager import TimingCallback
from pydantic import BaseModel

from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.llm.lightning import BionemoLightningModule
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
from bionemo.llm.run.config_models import (
    DataConfig,
    DataModuleT,
    ExperimentConfig,
    ExposedModelConfig,
    OptimizerSchedulerConfig,
    ParallelConfig,
    TrainingConfig,
)
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


class NsysConfig(BaseModel):
    """Configuration for nsys profiling."""

    start_step: int = 0
    end_step: Optional[int] = None
    ranks: list[int] = field(default_factory=lambda: [0])


def nemo_logger_factory(experiment_config: ExperimentConfig, wandb_config: Optional[WandbConfig]) -> nl.NeMoLogger:
    """Creates and returns a NeMoLogger instance configured based on the provided experiment and wandb configurations.

    Args:
        experiment_config (ExperimentConfig): Configuration object containing experiment settings such as
            result directory, experiment name, checkpoint settings, and logger preferences.
        wandb_config (Optional[WandbConfig]): Optional configuration object for Weights and Biases logging.

    Returns:
        nl.NeMoLogger: An instance of NeMoLogger configured with the specified settings.
    """
    if experiment_config.create_checkpoint_callback:
        checkpoint_callback = nl_callbacks.ModelCheckpoint(
            save_last=experiment_config.save_last_checkpoint,
            monitor=experiment_config.metric_to_monitor_for_checkpoints,
            save_top_k=experiment_config.save_top_k,
            every_n_train_steps=experiment_config.save_every_n_steps,
            always_save_context=True,
            filename="{epoch}-{val_loss:.2f}-{step}-{consumed_samples}",  # Including step and consumed_samples in the checkpoint filename prevents duplicate filenames and bugs related to this.
        )
    else:
        checkpoint_callback = None

    nemo_logger = setup_nemo_lightning_logger(
        root_dir=experiment_config.result_dir,
        name=experiment_config.experiment_name,
        initialize_tensorboard_logger=experiment_config.create_tensorboard_logger,
        wandb_config=wandb_config,
        ckpt_callback=checkpoint_callback,
    )
    return nemo_logger


def setup_trainer(
    parallel_config: ParallelConfig,
    training_config: TrainingConfig,
    callbacks=None,
    nsys_config: NsysConfig | None = None,
) -> nl.Trainer:
    """Set up the trainer for model training using the specified parallel and training configurations.

    Args:
        parallel_config (ParallelConfig): Configuration for parallelism, including tensor and pipeline model parallel sizes,
                                          number of devices, and number of nodes.
        training_config (TrainingConfig): Configuration for training, including maximum steps, accelerator type,
                                          validation batch limit, validation check interval, and precision.
        callbacks (list, optional): List of callback functions to be used during training. Defaults to None,
                                    in which case default callbacks (RichModelSummary and LearningRateMonitor) are used.
        nsys_config (NsysConfig, optional): Configuration for nsys profiling. If None, is disabled.

    Returns:
        nl.Trainer: Configured trainer object ready for model training.
    """
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        pipeline_dtype=get_autocast_dtype(training_config.precision),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=True,
            overlap_param_gather=False,  # TODO waiting for NeMo fix
            average_in_collective=True,
            use_distributed_optimizer=parallel_config.use_distributed_optimizer,
        ),
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
    )
    if callbacks is None:
        callbacks = [
            RichModelSummary(max_depth=4),
            LearningRateMonitor(),
        ]

    if training_config.gc_interval > 0:
        callbacks.append(
            nl_callbacks.GarbageCollectionCallback(
                gc_interval_train=training_config.gc_interval, gc_interval_val=training_config.gc_interval
            )
        )

    if nsys_config:
        if nsys_config.end_step is None:
            nsys_config.end_step = training_config.max_steps
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=nsys_config.start_step,
                end_step=nsys_config.end_step,
                ranks=nsys_config.ranks,
                gen_shape=True,
            )
        )

    trainer = nl.Trainer(
        devices=parallel_config.num_devices,
        max_steps=training_config.max_steps,
        accelerator=training_config.accelerator,
        strategy=strategy,
        limit_val_batches=training_config.limit_val_batches,
        val_check_interval=training_config.val_check_interval,
        num_nodes=parallel_config.num_nodes,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(
            precision=training_config.precision,
            params_dtype=get_autocast_dtype(training_config.precision),
            pipeline_dtype=get_autocast_dtype(training_config.precision),
            grad_reduce_in_fp32=False,
            autocast_enabled=False,
        ),
        enable_checkpointing=training_config.enable_checkpointing,
    )
    return trainer


def train(
    bionemo_exposed_model_config: ExposedModelConfig,
    data_config: DataConfig[DataModuleT],
    parallel_config: ParallelConfig,
    training_config: TrainingConfig,
    optim_config: OptimizerSchedulerConfig,
    experiment_config: ExperimentConfig,
    wandb_config: Optional[WandbConfig],
    nsys_config: Optional[NsysConfig] = None,
    resume_if_exists: bool = True,
):
    """Train a BioNemo model using the provided configurations. Uses the ExposedModelConfig and DataConfig as the primary variants for this method.

    Args:
        bionemo_exposed_model_config (ExposedModelConfig): Configuration for the exposed BioNemo model.
        data_config (DataConfig[DataModuleT]): Configuration for the data module.
        parallel_config (ParallelConfig): Configuration for parallel training.
        training_config (TrainingConfig): Configuration for training parameters.
        optim_config (OptimizerSchedulerConfig): Configuration for the optimizer and scheduler.
        experiment_config (ExperimentConfig): Configuration for the experiment.
        wandb_config (Optional[WandbConfig]): Configuration for Weights and Biases logging.n
        nsys_config (Optional[NsysConfig], optional): Configuration for nsys profiling. If None, is disabled.
        resume_if_exists (bool, optional): Flag to resume training if a checkpoint exists. Defaults to True.
    """
    bionemo_model_config = bionemo_exposed_model_config.exposed_to_internal_bionemo_model_config()
    pathlib.Path(data_config.result_dir).mkdir(parents=True, exist_ok=True)

    if experiment_config.save_every_n_steps != training_config.val_check_interval:
        logging.warning("Mutating training_config.save_every_n_steps to be equal to val_check_interval.")
        experiment_config.save_every_n_steps = training_config.val_check_interval

    global_batch_size = infer_global_batch_size(
        micro_batch_size=data_config.micro_batch_size,
        num_nodes=parallel_config.num_nodes,
        devices=parallel_config.num_devices,
        accumulate_grad_batches=parallel_config.accumulate_grad_batches,
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
    )

    data: DataModuleT = data_config.construct_data_module(global_batch_size)
    # TODO BioBertDataModule or BioBertTokenizer abstractions. We know all DataModuleT in this case has data.tokenizer,
    # although this constraint is not documented.

    # TODO: need an abstraction for LrSchedulerConfig
    if optim_config.lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingScheduler(
            max_steps=training_config.max_steps if optim_config.max_steps is None else optim_config.max_steps,
            min_lr=optim_config.lr / 100,
            warmup_steps=math.ceil(training_config.max_steps * optim_config.cosine_rampup_frac),
            interval=optim_config.interval,
            monitor=optim_config.monitor,
            constant_steps=math.ceil(training_config.max_steps * optim_config.cosine_hold_frac),
        )
    elif optim_config.lr_scheduler == "warmup_anneal":
        lr_scheduler = WarmupAnnealDecayHoldScheduler(
            warmup_steps=optim_config.warmup_steps,
            max_steps=training_config.max_steps if optim_config.max_steps is None else optim_config.max_steps,
            max_lr=optim_config.lr,
            min_lr=optim_config.lr / 10.0,
            anneal_percentage=0.10,
        )
    else:
        raise NotImplementedError(f"Scheduler {optim_config.lr_scheduler} not implemented.")

    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=optim_config.lr,
            weight_decay=optim_config.weight_decay,
            sgd_momentum=optim_config.sgd_momentum,
            adam_eps=optim_config.adam_eps,
            optimizer=optim_config.optimizer,
            use_distributed_optimizer=parallel_config.use_distributed_optimizer,
            fp16=bionemo_model_config.fp16,
            bf16=bionemo_model_config.bf16,
        ),
        lr_scheduler=lr_scheduler,
    )

    model: BionemoLightningModule = biobert_lightning_module(
        config=bionemo_model_config,
        tokenizer=data.tokenizer,
        optimizer=optimizer,
    )
    # NOTE (SKH): lifted default callbacks out of setup_trainer
    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        TimingCallback(),  # Required for certain plugins such as FLOPsMeasurement
    ]
    if training_config.create_tflops_callback:
        dummy_data_module = SimpleNamespace()
        dummy_data_module.global_batch_size = (
            global_batch_size  # TODO(dorotat): remove this change after FLOPsMeasurementCallback is refactored
        )
        dummy_data_module.tokenizer_vocab_size = data.vocab_size
        flop_meas_callback = FLOPsMeasurementCallback(
            bionemo_model_config,
            dummy_data_module,
            "bert",
        )
        callbacks.append(flop_meas_callback)
    trainer: nl.Trainer = setup_trainer(parallel_config, training_config, nsys_config=nsys_config, callbacks=callbacks)
    nemo_logger: nl.NeMoLogger = nemo_logger_factory(experiment_config, wandb_config=wandb_config)

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=resume_if_exists,
            resume_ignore_no_checkpoint=True,
        ),
    )
