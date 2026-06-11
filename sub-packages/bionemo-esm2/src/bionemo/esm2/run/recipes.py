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
from pathlib import Path
from typing import Callable, Optional

import yaml
from nemo.utils import logging
from pydantic import BaseModel

from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.esm2.run.config_models import ESM2DataConfig, ExposedESM2PretrainConfig
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.run.config_models import (
    ExperimentConfig,
    MainConfig,
    OptimizerSchedulerConfig,
    ParallelConfig,
    TrainingConfig,
)
from bionemo.llm.utils.logger_utils import WandbConfig


def esm2_base_training_config(max_steps: int = 500000) -> TrainingConfig:
    """Base training config for ESM2."""
    return TrainingConfig(
        max_steps=max_steps,
        limit_val_batches=1.0,
        val_check_interval=10_000,
        precision="bf16-mixed",
        include_perplexity=True,
    )


def esm2_base_optimizer_scheduler_config(max_steps: Optional[int] = None) -> OptimizerSchedulerConfig:
    """Base optimizer scheduler config for ESM2."""
    return OptimizerSchedulerConfig(
        optimizer="adam",
        lr=4e-4,
        interval="step",
        monitor="val_loss",
        lr_scheduler="warmup_anneal",
        warmup_steps=2000,
        max_steps=max_steps,
    )


def esm2_base_parallel_config() -> ParallelConfig:
    """Base parallel config for ESM2."""
    return ParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        accumulate_grad_batches=1,
        ddp="megatron",
        num_devices=1,
        num_nodes=1,
    )


def esm2_base_data_config(args) -> ESM2DataConfig:
    """Base data config for ESM2."""
    data_config = ESM2DataConfig(
        min_seq_length=1024,
        max_seq_length=1024,
        micro_batch_size=1,
        num_dataset_workers=8,
        train_cluster_path=args.train_cluster_path,
        train_database_path=args.train_database_path,
        valid_cluster_path=args.valid_cluster_path,
        valid_database_path=args.valid_database_path,
    )
    return data_config


def esm2_8m_wandb_config() -> WandbConfig:
    """Wandb config for ESM2 8m."""
    wandb_config = WandbConfig(
        entity="esm2-8m_pretraining",
        project="esm2-8m_pretraining",
        group="esm2-8m",
        tags=["esm2", "pretraining"],
        offline=True,
        anonymous=True,
        id="1",
        log_model=False,
    )
    return wandb_config


def esm2_8m_experiment_config(result_dir) -> ExperimentConfig:
    """Experiment config for ESM2 8m."""
    return ExperimentConfig(
        save_every_n_steps=50,  # default set in previous script.
        result_dir=result_dir,
        experiment_name="esm2-8m-pretraining",
        restore_from_checkpoint_path=None,
    )


def esm2_8m_model_config(initial_ckpt_path=None) -> ExposedESM2PretrainConfig:
    """Model config for ESM2 8m."""
    return ExposedESM2PretrainConfig(
        num_layers=6,
        hidden_size=320,
        ffn_hidden_size=320 * 4,
        num_attention_heads=20,
        seq_length=1024,
        biobert_spec_option=BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec,
        initial_ckpt_path=initial_ckpt_path,
        get_attention_mask_from_fusion=True,
        params_dtype="bf16-mixed",
        pipeline_dtype="bf16-mixed",
        autocast_dtype="bf16-mixed",
    )


def esm2_8m_recipe(args) -> MainConfig[ExposedESM2PretrainConfig, ESM2DataConfig]:
    """Recipe for ESM2 8m."""
    return MainConfig(
        data_config=esm2_base_data_config(args),
        parallel_config=esm2_base_parallel_config(),
        training_config=esm2_base_training_config(max_steps=args.max_steps),  # no changes for 8m
        bionemo_model_config=esm2_8m_model_config(args.initial_ckpt_path),
        optim_config=esm2_base_optimizer_scheduler_config(max_steps=args.scheduler_max_steps),  # no changes for 8m
        experiment_config=esm2_8m_experiment_config(args.result_dir),
        wandb_config=esm2_8m_wandb_config(),
    )


def esm2_650m_model_config(initial_ckpt_path=None) -> ExposedESM2PretrainConfig:
    """Model config for ESM2 650m."""
    return ExposedESM2PretrainConfig(
        num_layers=33,
        hidden_size=1280,
        ffn_hidden_size=1280 * 4,
        seq_length=1024,
        num_attention_heads=20,
        biobert_spec_option=BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec,
        initial_ckpt_path=initial_ckpt_path,
        get_attention_mask_from_fusion=True,
        params_dtype="bf16-mixed",
        pipeline_dtype="bf16-mixed",
        autocast_dtype="bf16-mixed",
    )


def esm2_650m_wandb_config() -> WandbConfig:
    """Wandb config for ESM2 650m."""
    return WandbConfig(
        entity="esm2-650m_pretraining",
        project="esm2-650m_pretraining",
        group="esm2-650m",
        tags=["esm2", "pretraining"],
        offline=True,
        anonymous=True,
        id="1",
        log_model=False,
    )


def esm2_650m_experiment_config(result_dir) -> ExperimentConfig:
    """Experiment config for ESM2 650m."""
    return ExperimentConfig(
        save_every_n_steps=50,
        result_dir=result_dir,
        experiment_name="esm2-650m-pretraining",
        # TODO should this be exposed?
        restore_from_checkpoint_path=None,
    )


def esm2_650m_recipe(args) -> MainConfig[ExposedESM2PretrainConfig, ESM2DataConfig]:
    """Recipe for ESM2 650m."""
    return MainConfig(
        data_config=esm2_base_data_config(args),
        parallel_config=esm2_base_parallel_config(),
        training_config=esm2_base_training_config(max_steps=args.max_steps),  # no changes for 8m
        bionemo_model_config=esm2_650m_model_config(args.initial_ckpt_path),
        optim_config=esm2_base_optimizer_scheduler_config(max_steps=args.scheduler_max_steps),  # no changes for 8m
        experiment_config=esm2_650m_experiment_config(args.result_dir),
        wandb_config=esm2_650m_wandb_config(),
    )


def esm2_3b_parallel_config() -> ParallelConfig:
    """Parallel config for ESM2 3b."""
    return ParallelConfig(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        # TODO: is this correct?
        accumulate_grad_batches=1,
        ddp="megatron",
        # NOTE assumes 8xGPU node. Can always edit the config.
        num_devices=8,
    )


def esm2_3b_model_config(initial_ckpt_path=None) -> ExposedESM2PretrainConfig:
    """Model config for ESM2 3b."""
    return ExposedESM2PretrainConfig(
        num_layers=36,
        hidden_size=2560,
        ffn_hidden_size=2560 * 4,
        num_attention_heads=40,
        seq_length=1024,
        biobert_spec_option=BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec,
        initial_ckpt_path=initial_ckpt_path,
        get_attention_mask_from_fusion=True,
        params_dtype="bf16-mixed",
        pipeline_dtype="bf16-mixed",
        autocast_dtype="bf16-mixed",
    )


def esm2_3b_wandb_config() -> WandbConfig:
    """Wandb config for ESM2 3b."""
    return WandbConfig(
        entity="esm2-3b_pretraining",
        project="esm2-3b_pretraining",
        group="esm2-3b",
        tags=["esm2-650m"],
        offline=True,
        anonymous=True,
        id="1",
        log_model=False,
    )


def esm2_3b_experiment_config(result_dir) -> ExperimentConfig:
    """Experiment config for ESM2 650m."""
    return ExperimentConfig(
        save_every_n_steps=50,
        result_dir=result_dir,
        experiment_name="esm2-3b-pretraining",
        # TODO should this be exposed?
        restore_from_checkpoint_path=None,
    )


def esm2_3b_recipe(args) -> MainConfig[ExposedESM2PretrainConfig, ESM2DataConfig]:
    """Recipe for ESM2 3b."""
    return MainConfig(
        data_config=esm2_base_data_config(args),
        parallel_config=esm2_3b_parallel_config(),
        training_config=esm2_base_training_config(max_steps=args.max_steps),  # no changes for 8m
        bionemo_model_config=esm2_3b_model_config(args.initial_ckpt_path),
        optim_config=esm2_base_optimizer_scheduler_config(max_steps=args.scheduler_max_steps),  # no changes for 8m
        experiment_config=esm2_3b_experiment_config(args.result_dir),
        wandb_config=esm2_3b_wandb_config(),
    )


def simple_parallel_recipe(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    num_devices: int = 1,
    accumulate_grad_batches: int = 1,
) -> ParallelConfig:
    """Simple parallel recipe for ESM2."""
    assert num_devices >= tensor_model_parallel_size * pipeline_model_parallel_size, (
        "devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size"
    )
    return ParallelConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        num_devices=num_devices,
        accumulate_grad_batches=accumulate_grad_batches,
    )


def tiny_train_config_recipe() -> TrainingConfig:
    """Tiny training config for ESM2."""
    return TrainingConfig(max_steps=10, limit_val_batches=2, val_check_interval=2)


def default_adam_optimizer_with_cosine_annealing_recipe(max_steps: Optional[int] = None) -> OptimizerSchedulerConfig:
    """Default optimizer scheduler config for ESM2."""
    return OptimizerSchedulerConfig(max_steps=max_steps)


def experiment_config_recipe(result_dir="./results") -> ExperimentConfig:
    """Experiment config for ESM2."""
    return ExperimentConfig(
        save_every_n_steps=100,
        result_dir=result_dir,
        experiment_name="default_experiment",
        restore_from_checkpoint_path=None,
        save_last_checkpoint=True,
        metric_to_monitor_for_checkpoints="val_loss",
        save_top_k=2,
        create_tensorboard_logger=False,
    )


def esm2_tiny_model_config(
    seq_length: int = 2048,
    precision: PrecisionTypes = "bf16-mixed",
    nemo1_init_path: Optional[str] = None,
    initial_ckpt_path: Optional[str] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec,
    variable_seq_lengths: bool = False,
) -> ExposedESM2PretrainConfig:
    """Model config for ESM2 tiny, used for testing."""
    return ExposedESM2PretrainConfig(
        seq_length=seq_length,
        num_layers=2,
        hidden_size=32,
        num_attention_heads=2,
        ffn_hidden_size=4 * 32,
        params_dtype=precision,
        pipeline_dtype=precision,
        autocast_dtype=precision,
        biobert_spec_option=biobert_spec_option,
        get_attention_mask_from_fusion=True,
        nemo1_ckpt_path=str(nemo1_init_path) if nemo1_init_path is not None else None,
        # handle checkpoint resumption here rather than auto-resume so this supports fine-tuning capabilities
        initial_ckpt_path=str(initial_ckpt_path) if initial_ckpt_path is not None else None,
        variable_seq_lengths=variable_seq_lengths,
    )


def esm2_tiny_test_recipe(args):
    """Test recipe for ESM2 tiny, used for testing."""
    parallel_config = simple_parallel_recipe()
    training_config = tiny_train_config_recipe()

    data_config = ESM2DataConfig(
        min_seq_length=128,
        max_seq_length=128,
        micro_batch_size=2,
        num_dataset_workers=1,
        train_cluster_path=args.train_cluster_path,
        train_database_path=args.train_database_path,
        valid_cluster_path=args.valid_cluster_path,
        valid_database_path=args.valid_database_path,
    )
    bionemo_model_config = esm2_tiny_model_config(
        seq_length=data_config.max_seq_length, initial_ckpt_path=args.initial_ckpt_path
    )

    optim_config = default_adam_optimizer_with_cosine_annealing_recipe(max_steps=args.scheduler_max_steps)
    experiment_config = experiment_config_recipe(args.result_dir)
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
    main_config = MainConfig[ExposedESM2PretrainConfig, ESM2DataConfig](
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        bionemo_model_config=bionemo_model_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        wandb_config=wandb_config,
    )
    return main_config


class ESM2Recipes(BaseModel):
    """Pre-baked recipes for ESM2.

    THIS PYDANTIC MODEL IS NOT MEANT FOR SERIALIZATION. Only used to facilitate argparse. Each recipe should take `args`
    as the only argument. We use partials so we can provide this information at runtime. Add new recipes to this model.
    """

    # Use partials so we can still parameterize the recipes from the CLI (e.g. data paths.)
    esm2_tiny_test_recipe: Callable[[argparse.Namespace], MainConfig[ExposedESM2PretrainConfig, ESM2DataConfig]] = (
        partial(esm2_tiny_test_recipe)
    )
    esm2_8m_recipe: Callable[[argparse.Namespace], MainConfig[ExposedESM2PretrainConfig, ESM2DataConfig]] = partial(
        esm2_8m_recipe
    )
    esm2_650m_recipe: Callable[[argparse.Namespace], MainConfig[ExposedESM2PretrainConfig, ESM2DataConfig]] = partial(
        esm2_650m_recipe
    )
    esm2_3b_recipe: Callable[[argparse.Namespace], MainConfig[ExposedESM2PretrainConfig, ESM2DataConfig]] = partial(
        esm2_3b_recipe
    )


def main():  # noqa: D103
    def parse_args():
        parser = argparse.ArgumentParser(description="Create ESM2 configuration YAML.")
        parser.add_argument(
            "--recipe",
            type=str,
            choices=ESM2Recipes.model_fields.keys(),
            required=True,
            help="Use one of the preconfigured recipes to create a template config file.",
        )

        parser.add_argument(
            "--dest",
            type=str,
            default="./esm2-recipe.yaml",
            required=True,
            help="Path to the YAML configuration file.",
        )

        parser.add_argument(
            "--train-cluster-path", type=Path, required=True, help="Path to the training cluster file."
        )
        parser.add_argument(
            "--train-database-path", type=Path, required=True, help="Path to the training database file."
        )
        parser.add_argument(
            "--valid-cluster-path", type=Path, required=True, help="Path to the validation cluster file."
        )
        parser.add_argument(
            "--valid-database-path", type=Path, required=True, help="Path to the validation database file."
        )

        parser.add_argument("--result-dir", type=Path, required=True, default="results", help="Path to store results")

        # Extra argument.
        parser.add_argument(
            "--initial-ckpt-path",
            type=str,
            required=False,
            default=None,
            help="Path to an existing to a checkpoint directory to restore an existing checkpoint. Not compatible with all recipes.",
        )

        parser.add_argument(
            "--max-steps", type=int, required=False, default=500000, help="Max steps for training. Default to 500000."
        )

        parser.add_argument(
            "--scheduler-max-steps",
            type=int,
            required=False,
            help="Set scheduler max_steps directly. Otherwise default to None, which uses max_steps from training config.",
        )

        args = parser.parse_args()
        return args

    # Simple example for creating a YAML from recipes.
    args = parse_args()
    config_partial: Callable[[argparse.Namespace], MainConfig] = ESM2Recipes().__getattribute__(args.recipe)
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
