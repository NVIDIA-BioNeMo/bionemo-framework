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

"""Create a Qwen3 checkpoint for export.

This script saves a randomly initialized Qwen3 model with TransformerEngine layers.
"""

import json
import shutil
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import convert_qwen3
from modeling_qwen3_te import AUTO_MAP


def export_hf_checkpoint(tag: str, export_path: Path):
    """Export a Hugging Face checkpoint to a Transformer Engine checkpoint.

    Args:
        tag: The tag of the checkpoint to export.
        export_path: The parent path to export the checkpoint to.
    """
    model_hf = AutoConfig.from_pretrained(tag)
    model_hf = AutoModelForCausalLM.from_config(model_hf)

    model_te = convert_qwen3.convert_qwen3_hf_to_te(model_hf)
    model_te.save_pretrained(export_path)

    tokenizer = AutoTokenizer.from_pretrained(tag)
    tokenizer.save_pretrained(export_path)

    # Patch the config
    with open(export_path / "config.json", "r") as f:
        config = json.load(f)

    config["auto_map"] = AUTO_MAP

    with open(export_path / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    shutil.copy(Path(__file__).parent / "modeling_qwen3_te.py", export_path / "modeling_qwen3_te.py")


if __name__ == "__main__":
    export_hf_checkpoint("Qwen/Qwen3-0.6B", Path("checkpoint_export"))
