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

"""Eden pre-training recipe configuration.

Builds a ``ConfigContainer`` for training Eden (Llama 3.1 architecture)
models with the BCR dataloader or mock data.
"""

import os
from pathlib import Path
from typing import Optional

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

from bionemo.eden.data.sharded_eden_dataset_provider import ShardedEdenDatasetProvider
from bionemo.eden.models.eden_provider import EdenModelProvider


_REPO_BASE_DIR = Path(__file__).resolve().parents[4]
DEFAULT_HF_TOKENIZER_MODEL_PATH = str(_REPO_BASE_DIR / "tokenizers" / "nucleotide_fast_tokenizer_256")


def eden_pretrain_config(
    model_provider: type[EdenModelProvider] = EdenModelProvider,
    hf_tokenizer_model_or_path: str | Path | None = None,
    dir: str | None = None,
    name: str = "default",
    dataset_seed: int = 1234,
    seed: int = 1234,
    mock: bool = False,
    sharded_eden_data: bool = False,
    sequence_db_dir: Optional[str] = None,
    train_window_db_path: Optional[str] = None,
    val_window_db_path: Optional[str] = None,
    test_window_db_path: Optional[str] = None,
    rc_aug: bool = False,
    stride: int = 7992,
    window_min_length_threshold: int = 0,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    sequence_parallel: Optional[bool] = None,
    train_iters: int = 100_000,
    global_batch_size: int = 8,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2500,
    lr_decay_iters: Optional[int] = None,
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    no_weight_decay_embeddings: bool = False,
    pad_eod_loss_mask: bool = False,
) -> ConfigContainer:
    """Create a pre-training configuration for Eden models.

    Args:
        model_provider: Eden model provider class.
        hf_tokenizer_model_or_path: HuggingFace tokenizer path. Defaults to
            the bundled 256-token nucleotide tokenizer.
        dir: Base output directory for checkpoints and logs.
        name: Experiment name.
        dataset_seed: Random seed for dataset shuffling.
        seed: Global random seed.
        mock: Use synthetic mock data.
        sharded_eden_data: Use BCR sharded SQLite dataloader.
        sequence_db_dir: Directory of per-sample SQLite databases.
        train_window_db_path: Path to training split window database.
        val_window_db_path: Path to validation split window database.
        test_window_db_path: Path to test split window database.
        rc_aug: Enable reverse-complement augmentation.
        stride: Stride between windows.
        window_min_length_threshold: Minimum window length threshold.
        tensor_model_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        pipeline_dtype: Pipeline dtype override.
        virtual_pipeline_model_parallel_size: Virtual pipeline parallelism.
        context_parallel_size: Context parallelism degree.
        sequence_parallel: Enable sequence parallelism.
        train_iters: Total training iterations.
        global_batch_size: Global batch size.
        micro_batch_size: Micro batch size.
        seq_length: Training sequence length.
        lr: Maximum learning rate.
        min_lr: Minimum learning rate.
        lr_warmup_iters: Warmup iterations.
        lr_decay_iters: Decay iterations.
        precision_config: Mixed precision recipe name or config.
        comm_overlap_config: Communication overlap config.
        no_weight_decay_embeddings: Skip weight decay on embeddings.
        pad_eod_loss_mask: Mask EOD/PAD tokens in loss.

    Returns:
        A ``ConfigContainer`` ready for ``pretrain()``.
    """
    if hf_tokenizer_model_or_path is None:
        hf_tokenizer_model_or_path = DEFAULT_HF_TOKENIZER_MODEL_PATH

    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")
    wandb_save_dir = os.path.join(run_output_dir, "wandb")

    if isinstance(precision_config, str):
        precision_config = get_mixed_precision_config(precision_config)

    if mock:
        from bionemo.eden.data.eden_mock_dataset_provider import MockEdenDatasetProvider

        dataset_cfg_or_provider = MockEdenDatasetProvider(
            random_seed=dataset_seed,
            seq_length=seq_length,
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
    else:
        raise ValueError("Must specify either --mock-data or --sharded-eden-data for Eden training.")

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
        adam_eps=1e-8,
        weight_decay=0.01,
        max_lr=lr,
        min_lr=min_lr,
    )

    return ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=100,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_config,
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
            tokenizer_model=str(hf_tokenizer_model_or_path),
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
    )
