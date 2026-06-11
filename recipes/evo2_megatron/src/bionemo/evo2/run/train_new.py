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
from pathlib import Path

import torch
from megatron.bridge.training.pretrain import pretrain

from bionemo.evo2.models.evo2_provider import hyena_forward_step
from bionemo.evo2.recipes.evo2 import evo2_1b_pretrain_config as pretrain_config


if __name__ == "__main__":
    # Load default Llama 3.1 8B config with custom training parameters
    cfg = pretrain_config(
        train_iters=10,  # Number of training iterations
        micro_batch_size=2,  # Batch size per GPU
        global_batch_size=128,  # Total batch size across all GPUs
        lr_warmup_iters=10,  # Learning rate warmup iterations
        lr_decay_iters=20,  # Learning rate decay iterations
        context_parallel_size=1,  # Context parallelism (1 = disabled)
        hf_tokenizer_model_or_path=Path("./asciitokenizer_512"),
        seq_length=512,  # NOTE: mock data does an arange of each sequence length, so it must be less than vocab size
        mock=True,
    )
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        cfg.to_yaml(yaml_path="base_config.yaml")

    # # Configure dataset paths (train, validation, test)
    # cfg.dataset.split = None
    # cfg.dataset.blend_per_split = [
    #     [["./data/tokenized/train_text_document"], None],
    #     [["./data/tokenized/test_text_document"], None],
    #     [["./data/tokenized/test_text_document"], None]]

    # Logging and checkpoint settings
    cfg.logger.log_interval = 1  # Log every iteration
    cfg.checkpoint.save = None  # Don't save checkpoints (demo only)
    cfg.dataset.num_workers = 2  # Data loading workers
    cfg.train.eval_iters = 0  # Skip evaluation (faster demo)

    # the BF16 format is set by default
    # cfg.mixed_precision = "bf16_mixed"

    # Start training
    pretrain(cfg, hyena_forward_step)
