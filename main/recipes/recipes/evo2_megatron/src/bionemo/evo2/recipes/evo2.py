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

import os
import warnings
from pathlib import Path

import torch
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, get_mixed_precision_config
from typing_extensions import TypedDict, Unpack

from bionemo.evo2.data.evo2_dataset_provider import Evo2DatasetProvider
from bionemo.evo2.data.evo2_mock_dataset_provider import MockEvo2DatasetProvider
from bionemo.evo2.data.megatron.hyena.evo2_dataset import Evo2Dataset, Evo2DatasetPadEodLossMask
from bionemo.evo2.data.sharded_eden_dataset_provider import ShardedEdenDatasetProvider
from bionemo.evo2.models.evo2_lora import Evo2LoRA
from bionemo.evo2.models.evo2_provider import (
    Hyena1bModelProvider,
    HyenaModelProvider,
    HyenaOptimizerConfigOverrideProvider,
)


class Evo2CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Evo2 recipe helper functions."""

    # Core identifiers
    model_provider: type[HyenaModelProvider]
    hf_tokenizer_model_or_path: str | Path | None
    dir: str | None
    name: str
    # Dataset configuration
    dataset_seed: int
    seed: int
    ## Evo2
    dataset_config_path: str | None
    dataset_dir: str | None
    ## Mock
    mock: bool
    ## Sharded Eden
    sharded_eden_data: bool
    sequence_db_dir: str | None
    train_window_db_path: str | None
    val_window_db_path: str | None
    test_window_db_path: str | None
    rc_aug: bool
    stride: int
    window_min_length_threshold: int
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    sequence_parallel: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None
    pad_eod_loss_mask: bool
    no_weight_decay_embeddings: bool
    lora_finetune: bool
    lora_alpha: int
    lora_dim: int
    lora_dropout: float
    lora_target_modules: list[str]
    lora_skip_freeze_modules: list[str]


def evo2_1b_pretrain_config(**user_kwargs: Unpack[Evo2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Evo2 1B.

    Deprecated:
        This recipe is deprecated and will be removed in a future release.
    """
    warnings.warn(
        "evo2_1b_pretrain_config should only be used for running the original evo2 1b models.",
        DeprecationWarning,
        stacklevel=2,
    )
    recommended: Evo2CommonKwargs = {
        "model_provider": Hyena1bModelProvider,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "no_weight_decay_embeddings": False,
    }
    kwargs: Evo2CommonKwargs = {**recommended, **user_kwargs}
    return _evo2_common(**kwargs)


def _evo2_common(
    model_provider: type[HyenaModelProvider],
    hf_tokenizer_model_or_path: str | Path | None = "EleutherAI/gpt-neox-20b",
    dir: str | None = None,
    name: str = "default",
    # Dataset configuration
    dataset_seed: int = 1234,
    seed: int = 1234,
    ## Evo2
    dataset_config_path: str | None = None,
    dataset_dir: str | None = None,
    ## Mock
    mock: bool = False,
    ## Sharded Eden
    sharded_eden_data: bool = False,
    sequence_db_dir: str | None = None,
    train_window_db_path: str | None = None,
    val_window_db_path: str | None = None,
    test_window_db_path: str | None = None,
    rc_aug: bool = False,
    stride: int = 7992,
    window_min_length_threshold: int = 0,
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool | None = None,
    # Training hyperparameters
    train_iters: int = 1_168_251,
    global_batch_size: int = 8,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int | None = None,
    # TODO spike-no-more-embedding-init
    # Precision recipe
    # TODO fp8 etc
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
    no_weight_decay_embeddings: bool = False,
    pad_eod_loss_mask: bool = False,
    lora_finetune: bool = False,
    lora_alpha: int = 32,
    lora_dim: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: list[str] = ["dense_projection", "linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    lora_skip_freeze_modules: list[str] = [],
) -> ConfigContainer:
    """Create a pre-training configuration for Mamba 2.x models.

    Args mirror the individual recipe helpers; see those functions for recommended defaults.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")
    wandb_save_dir = os.path.join(run_output_dir, "wandb")
    if isinstance(precision_config, str):
        precision_config = get_mixed_precision_config(precision_config)
    if mock:
        dataset_cfg_or_provider = MockEvo2DatasetProvider(
            random_seed=dataset_seed,
            seq_length=seq_length,
            eod_mask_loss=pad_eod_loss_mask,
            overfit_mode=True,  # Does a modulo arange (vocab size) that is rolled by idx so elements are different.
        )
    elif sharded_eden_data:
        assert sequence_db_dir is not None
        assert train_window_db_path is not None
        assert val_window_db_path is not None
        assert test_window_db_path is not None
        dataset_cfg_or_provider = ShardedEdenDatasetProvider(
            random_seed=dataset_seed,
            sequence_db_dir=sequence_db_dir,
            train_window_db_path=train_window_db_path,
            val_window_db_path=val_window_db_path,
            test_window_db_path=test_window_db_path,
            seq_length=seq_length,
            rc_aug=rc_aug,
            stride=stride,
            window_min_length_threshold=window_min_length_threshold,
            use_control_tags=False,
            log_windows=False,
            log_dir=None,
            skip_stats=True,
            create_attention_mask=False,
        )
    elif dataset_config_path:
        dataset_cfg_or_provider = Evo2DatasetProvider(
            random_seed=dataset_seed,
            dataset_config_path=dataset_config_path,
            dataset_dir=dataset_dir,
            seq_length=seq_length,
            eod_mask_loss=pad_eod_loss_mask,
            dataset_cls=Evo2DatasetPadEodLossMask if pad_eod_loss_mask else Evo2Dataset,
        )
    else:
        raise ValueError("TODO unsure how to handle this case")

    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype or torch.bfloat16,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel if sequence_parallel is not None else tensor_model_parallel_size > 1,
        seq_length=seq_length,
        perform_initialization=True,
    )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        weight_decay=0.1,
        max_lr=lr,
        min_lr=min_lr,
    )

    if lora_finetune:
        peft = Evo2LoRA(
            target_modules=lora_target_modules,
            dim=lora_dim,
            alpha=lora_alpha,
            dropout=lora_dropout,
            skip_freeze_modules=lora_skip_freeze_modules,
        )
    else:
        peft = None

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=100,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_config,
        optimizer_config_override_provider=HyenaOptimizerConfigOverrideProvider(
            no_weight_decay_embeddings=no_weight_decay_embeddings,
        ),
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            align_param_gather=False,
            use_distributed_optimizer=True,
        ),
        dataset=dataset_cfg_or_provider,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_params_norm=True,
            log_throughput=True,
            log_progress=True,
            log_timers_to_tensorboard=True,
            log_throughput_to_tensorboard=True,
            log_loss_scale_to_tensorboard=True,
            log_validation_ppl_to_tensorboard=True,
            log_memory_to_tensorboard=True,
            log_l2_norm_grad_to_tensorboard=True,
            log_runtime_to_tensorboard=True,
            log_world_size_to_tensorboard=True,
            wandb_save_dir=wandb_save_dir,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=str(hf_tokenizer_model_or_path)
            if hf_tokenizer_model_or_path is not None
            else "EleutherAI/gpt-neox-20b",
        ),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_load=True,
            dist_ckpt_optim_fully_reshardable=False,
        ),
        rng=RNGConfig(seed=seed),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
        peft=peft,
    )

    return cfg


__all__ = [
    "evo2_1b_pretrain_config",
]
