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
from functools import partial
from typing import Callable, List, Optional

import yaml
from nemo.utils import logging
from pydantic import BaseModel

from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.geneformer.run.config_models import (
    ExposedFineTuneSeqLenBioBertConfig,
    ExposedGeneformerPretrainConfig,
    GeneformerPretrainingDataConfig,
)
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.run.config_models import (
    ExperimentConfig,
    MainConfig,
    OptimizerSchedulerConfig,
    ParallelConfig,
    TrainingConfig,
)
from bionemo.llm.utils.logger_utils import WandbConfig


def geneformer_base_parallel_config() -> ParallelConfig:
    """Base parallel config for Geneformer."""
    return ParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        accumulate_grad_batches=1,
        ddp="megatron",
        num_devices=1,
        num_nodes=1,
    )


def geneformer_base_optimizer_scheduler_config() -> OptimizerSchedulerConfig:
    """Base optimizer scheduler config for Geneformer."""
    return OptimizerSchedulerConfig(lr=1e-3, lr_scheduler="cosine")  # Matches bionemo1


def geneformer_1b_optimizer_scheduler_config() -> OptimizerSchedulerConfig:
    """Optimizer scheduler config for Geneformer 1B."""
    return OptimizerSchedulerConfig(
        lr=0.0004,
        lr_scheduler="warmup_anneal",
        cosine_hold_frac=0.05,
        cosine_rampup_frac=0.01,
        interval="step",
        max_steps=123125,
        monitor="val_loss",
        optimizer="adam",
        warmup_steps=500,
    )


def geneformer_base_training_config() -> TrainingConfig:
    """Base training config for Geneformer."""
    return TrainingConfig(
        max_steps=400000, limit_val_batches=8, val_check_interval=100, precision="bf16-mixed"
    )  # matches bionemo1


def geneformer_1b_training_config() -> TrainingConfig:
    """Training config for Geneformer 1B."""
    return TrainingConfig(max_steps=123125, limit_val_batches=8, val_check_interval=100, precision="bf16-mixed")


def geneformer_short_base_training_config() -> TrainingConfig:
    """Base training config for Geneformer."""
    return TrainingConfig(
        max_steps=500, limit_val_batches=8, val_check_interval=100, precision="bf16-mixed"
    )  # matches bionemo1


def geneformer_data_recipe(data_dir) -> GeneformerPretrainingDataConfig:
    """Recipe that produces the base geneformer small data configuration."""
    return GeneformerPretrainingDataConfig(data_dir=data_dir)


# 10m definitions
def geneformer_10m_model_config(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec,
) -> ExposedGeneformerPretrainConfig:
    """Geneformer 10m model config settings."""
    geneformer_config = ExposedGeneformerPretrainConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp32_residual_connection=False,
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        gradient_accumulation_fusion=False,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func="gelu",
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=False,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=True,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
    )
    return geneformer_config


def geneformer_10m_experiment_config(result_dir) -> ExperimentConfig:
    """Experiment config for Geneformer 10m."""
    return ExperimentConfig(
        save_every_n_steps=100,
        result_dir=result_dir,
        experiment_name="geneformer-10m",
        restore_from_checkpoint_path=None,
    )


def geneformer_10m_wandb_config() -> WandbConfig:
    """Wandb config for Geneformer 10m."""
    wandb_config = WandbConfig(
        entity="geneformer-10m_pretraining",
        project="geneformer-10m_pretraining",
        group="geneformer-10m",
        tags=["geneformer-10m"],
        offline=True,
        anonymous=True,
        id="1",
        log_model=False,
    )
    return wandb_config


# 106m definition, model, experiment, wandb, parallel
def geneformer_106m_parallel_config() -> ParallelConfig:
    """Base parallel config for Geneformer."""
    return ParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        accumulate_grad_batches=1,
        ddp="megatron",
        num_devices=8,
        num_nodes=1,
    )


def geneformer_106m_experiment_config(result_dir) -> ExperimentConfig:
    """Experiment config for Geneformer 106m."""
    return ExperimentConfig(
        save_every_n_steps=100,
        result_dir=result_dir,
        experiment_name="geneformer-106m",
        restore_from_checkpoint_path=None,
    )


def geneformer_106m_wandb_config() -> WandbConfig:
    """Wandb config for Geneformer 106m."""
    wandb_config = WandbConfig(
        entity="geneformer-106m_pretraining",
        project="geneformer-106m_pretraining",
        group="geneformer-106m",
        tags=["geneformer-106m"],
        offline=True,
        anonymous=True,
        id="1",
        log_model=False,
    )
    return wandb_config


def geneformer_106m_model_config(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec,
) -> ExposedGeneformerPretrainConfig:
    """Geneformer 106m model config settings."""
    geneformer_config = ExposedGeneformerPretrainConfig(
        num_layers=12,
        hidden_size=768,
        ffn_hidden_size=3072,
        num_attention_heads=12,
        seq_length=seq_length,
        fp32_residual_connection=False,
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        gradient_accumulation_fusion=False,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func="gelu",
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=False,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=True,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
    )
    return geneformer_config


# 1B definition, model, experiment, wandb, parallel
def geneformer_1b_parallel_config() -> ParallelConfig:
    """Base parallel config for Geneformer 1B. Increase devices as needed for DDP."""
    return ParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        accumulate_grad_batches=1,
        ddp="megatron",
        num_devices=1,
        num_nodes=1,
    )


def geneformer_1b_experiment_config(result_dir) -> ExperimentConfig:
    """Experiment config for Geneformer 1B."""
    return ExperimentConfig(
        save_every_n_steps=100,
        result_dir=result_dir,
        experiment_name="geneformer-1b",
        restore_from_checkpoint_path=None,
    )


def geneformer_1b_wandb_config() -> WandbConfig:
    """Wandb config for Geneformer 1B."""
    wandb_config = WandbConfig(
        entity="geneformer-1b_pretraining",
        project="geneformer-1b_pretraining",
        group="geneformer-1b",
        tags=["geneformer-1b"],
        offline=True,
        anonymous=True,
        id="1",
        log_model=False,
    )
    return wandb_config


def geneformer_1b_model_config(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec,
) -> ExposedGeneformerPretrainConfig:
    """Geneformer 1B model config settings."""
    geneformer_config = ExposedGeneformerPretrainConfig(
        num_layers=48,
        hidden_size=1280,
        ffn_hidden_size=5120,
        num_attention_heads=20,
        seq_length=seq_length,
        fp32_residual_connection=False,
        hidden_dropout=0.1,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=True,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        gradient_accumulation_fusion=True,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func="gelu",
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=False,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=True,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
    )
    return geneformer_config


def simple_parallel_recipe(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    num_devices: int = 1,
    accumulate_grad_batches: int = 1,
) -> ParallelConfig:
    """Simple parallel config for Geneformer, only used in testing."""
    assert num_devices >= tensor_model_parallel_size * pipeline_model_parallel_size, (
        "devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size"
    )
    return ParallelConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_devices=num_devices,
    )


def geneformer_finetuning_regression_head_recipe(
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    initial_ckpt_skip_keys_with_these_prefixes: Optional[List[str]] = None,
) -> ExposedFineTuneSeqLenBioBertConfig:
    """Recipe for finetuning a regression head on the masked tokens."""
    partial_finetuning_config = partial(
        ExposedFineTuneSeqLenBioBertConfig,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
        biobert_spec_option=BiobertSpecOption.bert_layer_with_transformer_engine_spec,
    )
    if initial_ckpt_skip_keys_with_these_prefixes:
        finetuning_config = partial_finetuning_config(
            initial_ckpt_skip_keys_with_these_prefixes=initial_ckpt_skip_keys_with_these_prefixes
        )
    else:
        # Use the sensible default when None is passed
        finetuning_config = partial_finetuning_config()
    return finetuning_config


def default_trainer_config_recipe() -> TrainingConfig:
    """Default trainer config for Geneformer."""
    return TrainingConfig(max_steps=55000, limit_val_batches=2, val_check_interval=100)


def geneformer_10m_finetune_config(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option=BiobertSpecOption.bert_layer_with_transformer_engine_spec,
) -> ExposedFineTuneSeqLenBioBertConfig:
    """Geneformer 10m finetuning config settings."""
    geneformer_config = ExposedFineTuneSeqLenBioBertConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp32_residual_connection=False,
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        gradient_accumulation_fusion=False,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func="gelu",
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=False,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=True,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
    )
    return geneformer_config


def geneformer_tiny_config(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec,
) -> ExposedGeneformerPretrainConfig:
    """Geneformer tiny model config settings, used in testing."""
    geneformer_config = ExposedGeneformerPretrainConfig(
        num_layers=2,
        hidden_size=32,
        ffn_hidden_size=4 * 32,
        num_attention_heads=2,
        seq_length=seq_length,
        fp32_residual_connection=False,
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        gradient_accumulation_fusion=False,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func="gelu",
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=False,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=True,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=biobert_spec_option,
        nemo1_ckpt_path=nemo1_init_path,
        initial_ckpt_path=initial_ckpt_path,
    )
    return geneformer_config


def default_adam_optimizer_with_cosine_annealing_recipe() -> OptimizerSchedulerConfig:
    """Default optimizer scheduler config for Geneformer. See OptimizerSchedulerConfig for defaults."""
    return OptimizerSchedulerConfig()


def experiment_config_recipe() -> ExperimentConfig:
    """Default experiment config for Geneformer. Used in testing."""
    return ExperimentConfig(
        save_every_n_steps=100,
        result_dir="./results",
        experiment_name="default_experiment",
        restore_from_checkpoint_path=None,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints="reduced_train_loss",
        save_top_k=2,
        create_tensorboard_logger=False,
    )


def finetune_test_recipe(args) -> MainConfig[ExposedFineTuneSeqLenBioBertConfig, GeneformerPretrainingDataConfig]:
    """Recipe for finetuning a regression head on the masked tokens."""
    data_path = args.data_path
    result_dir = args.result_dir

    parallel_config = ParallelConfig(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, num_devices=1, accumulate_grad_batches=2
    )
    training_config = TrainingConfig(
        max_steps=10, limit_val_batches=2, val_check_interval=2, precision="bf16-mixed", accelerator="gpu"
    )
    data_config = GeneformerPretrainingDataConfig(
        seq_length=128,
        micro_batch_size=2,
        num_dataset_workers=0,
        data_dir=data_path,
    )
    experiment_config = ExperimentConfig(
        save_every_n_steps=training_config.val_check_interval,
        result_dir=result_dir,
        experiment_name="test-experiment",
        restore_from_checkpoint_path=None,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints="reduced_train_loss",
        save_top_k=2,
        create_tensorboard_logger=False,
    )

    optim_config = OptimizerSchedulerConfig(lr_scheduler="cosine")
    geneformer_config = geneformer_10m_finetune_config(
        seq_length=data_config.seq_length, initial_ckpt_path=args.initial_ckpt_path
    )

    return MainConfig(
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=geneformer_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
    )


def pretrain_tiny_test_recipe(args) -> MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]:
    """Recipe for pretraining a tiny model. Used in testing."""
    data_path = args.data_path
    result_dir = args.result_dir

    parallel_config = ParallelConfig(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, num_devices=1, accumulate_grad_batches=2
    )
    training_config = TrainingConfig(
        max_steps=10, limit_val_batches=2, val_check_interval=2, precision="bf16-mixed", accelerator="gpu"
    )
    data_config = GeneformerPretrainingDataConfig(
        seq_length=128,
        micro_batch_size=2,
        num_dataset_workers=0,
        data_dir=data_path,
    )
    experiment_config = ExperimentConfig(
        save_every_n_steps=training_config.val_check_interval,
        result_dir=result_dir,
        experiment_name="test-experiment",
        restore_from_checkpoint_path=None,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints="reduced_train_loss",
        save_top_k=2,
        create_tensorboard_logger=False,
    )

    optim_config = OptimizerSchedulerConfig(lr_scheduler="cosine")
    geneformer_config = geneformer_tiny_config(
        seq_length=data_config.seq_length, initial_ckpt_path=args.initial_ckpt_path
    )

    return MainConfig(
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=geneformer_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
    )


def geneformer_10m_pretrain_recipe(
    args,
) -> MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]:
    """Recipe for pretraining the 10m model."""
    data_config: GeneformerPretrainingDataConfig = geneformer_data_recipe(data_dir=args.data_path)
    parallel_config = simple_parallel_recipe()
    training_config = geneformer_base_training_config()
    bionemo_model_config = geneformer_10m_model_config(initial_ckpt_path=args.initial_ckpt_path)
    optim_config = geneformer_base_optimizer_scheduler_config()
    experiment_config = geneformer_10m_experiment_config(result_dir=args.result_dir)
    wandb_config = geneformer_10m_wandb_config()
    main_config = MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig](
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=bionemo_model_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        wandb_config=wandb_config,
    )
    return main_config


def geneformer_10m_shortpretrain_recipe(
    args,
) -> MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]:
    """Recipe for pretraining the 10m model."""
    data_config: GeneformerPretrainingDataConfig = geneformer_data_recipe(data_dir=args.data_path)
    parallel_config = simple_parallel_recipe()
    training_config = geneformer_short_base_training_config()
    bionemo_model_config = geneformer_10m_model_config(initial_ckpt_path=args.initial_ckpt_path)
    optim_config = geneformer_base_optimizer_scheduler_config()
    experiment_config = geneformer_10m_experiment_config(result_dir=args.result_dir)
    wandb_config = geneformer_10m_wandb_config()
    main_config = MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig](
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=bionemo_model_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        wandb_config=wandb_config,
    )
    return main_config


def geneformer_106m_pretrain_recipe(
    args,
) -> MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]:
    """Recipe for pretraining the 106m model. Uses 8 GPUs for data parallelism."""
    data_config: GeneformerPretrainingDataConfig = geneformer_data_recipe(data_dir=args.data_path)
    parallel_config = geneformer_106m_parallel_config()
    training_config = geneformer_base_training_config()
    bionemo_model_config = geneformer_106m_model_config(initial_ckpt_path=args.initial_ckpt_path)
    optim_config = geneformer_base_optimizer_scheduler_config()
    experiment_config = geneformer_106m_experiment_config(result_dir=args.result_dir)
    wandb_config = geneformer_106m_wandb_config()
    main_config = MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig](
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=bionemo_model_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        wandb_config=wandb_config,
    )
    return main_config


def geneformer_1b_pretrain_recipe(
    args,
) -> MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]:
    """Recipe for pretraining the 1B model. Uses tensor and pipeline parallelism across multiple nodes."""
    data_config: GeneformerPretrainingDataConfig = geneformer_data_recipe(data_dir=args.data_path)
    parallel_config = geneformer_1b_parallel_config()
    training_config = geneformer_1b_training_config()
    bionemo_model_config = geneformer_1b_model_config(initial_ckpt_path=args.initial_ckpt_path)
    optim_config = geneformer_1b_optimizer_scheduler_config()
    experiment_config = geneformer_1b_experiment_config(result_dir=args.result_dir)
    wandb_config = geneformer_1b_wandb_config()
    main_config = MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig](
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=bionemo_model_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        wandb_config=wandb_config,
    )
    return main_config


def geneformer_10m_finetune_recipe(
    args,
) -> MainConfig[ExposedFineTuneSeqLenBioBertConfig, GeneformerPretrainingDataConfig]:
    """Recipe for finetuning the 10m model on a token regression head. Used as an example and for testing."""
    data_config: GeneformerPretrainingDataConfig = geneformer_data_recipe(data_dir=args.data_path)
    parallel_config = simple_parallel_recipe()
    training_config = default_trainer_config_recipe()
    bionemo_model_config = geneformer_finetuning_regression_head_recipe(initial_ckpt_path=args.initial_ckpt_path)
    optim_config = default_adam_optimizer_with_cosine_annealing_recipe()
    experiment_config = experiment_config_recipe()
    wandb_config = WandbConfig(
        project="bionemo2-demo",
        entity="nvidia",
        offline=True,
        tags=[],
        group="dev",
        id="dev",
        log_model=False,
        anonymous=True,
    )
    main_config = MainConfig[ExposedFineTuneSeqLenBioBertConfig, GeneformerPretrainingDataConfig](
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=bionemo_model_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        wandb_config=wandb_config,
    )
    return main_config


class GeneformerRecipes(BaseModel):
    """Pre-baked recipes for Geneformer.

    THIS PYDANTIC MODEL IS NOT MEANT FOR SERIALIZATION. Only used to facilitate argparse. Each recipe should take `args`
    as the only argument. We use partials so we can provide this information at runtime. Add new recipes to this model.
    """

    # Use partials so we can still parameterize the recipes from the CLI (e.g. data paths.)
    geneformer_10m_finetune_recipe: Callable[
        [argparse.Namespace], MainConfig[ExposedFineTuneSeqLenBioBertConfig, GeneformerPretrainingDataConfig]
    ] = partial(geneformer_10m_finetune_recipe)
    geneformer_10m_pretrain_recipe: Callable[
        [argparse.Namespace], MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]
    ] = partial(geneformer_10m_pretrain_recipe)
    geneformer_10m_shortpretrain_recipe: Callable[
        [argparse.Namespace], MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]
    ] = partial(geneformer_10m_shortpretrain_recipe)
    geneformer_106m_pretrain_recipe: Callable[
        [argparse.Namespace], MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]
    ] = partial(geneformer_106m_pretrain_recipe)
    geneformer_1b_pretrain_recipe: Callable[
        [argparse.Namespace], MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]
    ] = partial(geneformer_1b_pretrain_recipe)
    geneformer_tiny_test_recipe: Callable[
        [argparse.Namespace], MainConfig[ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig]
    ] = partial(pretrain_tiny_test_recipe)
    finetune_test_recipe: Callable[
        [argparse.Namespace], MainConfig[ExposedFineTuneSeqLenBioBertConfig, GeneformerPretrainingDataConfig]
    ] = partial(finetune_test_recipe)


def main():  # noqa: D103
    def parse_args():
        parser = argparse.ArgumentParser(description="Create Geneformer configuration YAML.")
        parser.add_argument(
            "--recipe",
            type=str,
            choices=GeneformerRecipes.model_fields.keys(),
            required=True,
            help="Use one of the preconfigured recipes to create a template config file.",
        )

        parser.add_argument(
            "--dest",
            type=str,
            default="./geneformer-recipe.yaml",
            required=True,
            help="Path to the YAML configuration file.",
        )

        parser.add_argument(
            "--data-path", type=str, required=True, help="Path to the directory containing pretraining data."
        )
        parser.add_argument(
            "--result-dir", type=str, required=True, help="Path to the directory used to save results."
        )

        parser.add_argument(
            "--initial-ckpt-path",
            type=str,
            required=False,
            default=None,
            help="Path to an existing to a checkpoint directory to restore an existing checkpoint. Not compatible with all recipes.",
        )

        args = parser.parse_args()
        return args

    """Simple example for creating a YAML from recipes."""
    args = parse_args()
    config_partial: Callable[[argparse.Namespace], MainConfig] = GeneformerRecipes().__getattribute__(args.recipe)
    config = config_partial(args)

    # Save to file
    with open(
        args.dest,
        "w",
    ) as f:
        yaml.dump(config.model_dump(), f, indent=2)

    logging.info(f"Saved configuration to {args.dest=}")


if __name__ == "__main__":
    main()
