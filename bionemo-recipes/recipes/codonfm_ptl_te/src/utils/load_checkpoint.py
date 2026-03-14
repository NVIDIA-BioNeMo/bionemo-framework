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


import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def resolve_model_path(path: str) -> str:
    """Resolve a model path that may be a local path or a HuggingFace Hub repo ID.

    If ``path`` points to an existing local file or directory it is returned
    unchanged.  Otherwise it is treated as a HuggingFace Hub repo ID and
    downloaded via ``snapshot_download``, returning the local cache directory.
    """
    p = Path(path)
    if p.exists():
        return str(p)
    try:
        return snapshot_download(
            repo_id=path,
            allow_patterns=["*.safetensors", "*.json"],
        )
    except Exception as e:
        raise FileNotFoundError(
            f"'{path}' was not found as a local file or directory, and could not "
            f"be resolved as a HuggingFace Hub repo ID: {e}"
        ) from e


def load_checkpoint(checkpoint_path: str, map_location: str = "cpu"):
    """Load checkpoint from a PyTorch .ckpt file, safetensors directory, or HuggingFace Hub repo ID."""
    checkpoint_path = resolve_model_path(checkpoint_path)
    if Path(checkpoint_path).is_dir():
        safetensors_files = list(Path(checkpoint_path).glob("*.safetensors"))

        if not safetensors_files or len(safetensors_files) != 1:
            raise FileNotFoundError(f"Expected single .safetensors files in {checkpoint_path}")

        safetensors_path = safetensors_files[0]
        config_path = Path(checkpoint_path) / "config.json"
        state_dict = load_file(str(safetensors_path))
        with open(config_path, "r") as f:
            hparams = json.load(f)
        return {"state_dict": state_dict, "hyper_parameters": hparams}
    else:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
