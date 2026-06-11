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

import logging

import hydra
import torch

from checkpoint import load_torch_checkpoint
from distributed import initialize_distributed
from vit import build_vit_model


logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="vit_base_patch16_224")
def main(cfg) -> None:
    """
    Inference script for ViT. Non-distributed inference.
    """
    with initialize_distributed(**cfg.distributed) as device_mesh:
        # Init ViT.
        model = build_vit_model(cfg, device_mesh).cuda()

        # Load torch.save (non-distributed) model checkpoint trained using (or not using) Megatron-FSDP.
        load_torch_checkpoint(
            cfg.inference.checkpoint.path, model, megatron_fsdp=cfg.inference.checkpoint.megatron_fsdp
        )
        logger.info(f"Model: {model}")

        # Mock input.
        input = torch.randn(1, 3, 224, 224).cuda()
        if cfg.model.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # Infer.
        output = model(input)
        logger.info(f"Output: {output}")


if __name__ == "__main__":
    main()
