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

import pytest
from hydra import compose, initialize_config_dir
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from checkpoint import save_dcp_checkpoint
from distributed import initialize_distributed
from infer import main
from vit import build_vit_model


@pytest.mark.parametrize("config_name", ["vit_base_patch16_224", "vit_te_base_patch16_224"])
def test_infer(monkeypatch, tmp_path, config_name):
    """
    Test inference.
    """
    # Set required environment variables for distributed training
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")

    # Initialize inference config.
    recipe_dir = Path(__file__).parent
    test_ckpt_path = Path(tmp_path) / "test_infer_torch_checkpoint.pt"
    with initialize_config_dir(config_dir=str(recipe_dir / "config"), version_base="1.2"):
        vit_config = compose(
            config_name=config_name,
            overrides=[
                f"++inference.checkpoint.path={test_ckpt_path}",
                # Using a torch.save mock checkpoint for inference.
                "++inference.checkpoint.format=torch",
                # Using a non-Megatron-FSDP mock checkpoint for inference.
                "++inference.checkpoint.megatron_fsdp=false",
            ],
        )

    # Write a test checkpoint.
    with initialize_distributed(**vit_config.distributed) as device_mesh:
        # Init ViT.
        model = build_vit_model(vit_config, device_mesh).cuda()
        # Write checkpoint.
        save_dcp_checkpoint(Path(tmp_path) / "test_infer_dcp_checkpoint", model)
        # Convert checkpoint to Torch format.
        dcp_to_torch_save(Path(tmp_path) / "test_infer_dcp_checkpoint", test_ckpt_path)

    # Run inference.
    main(vit_config)
