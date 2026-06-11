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

"""Create a llama3 checkpoint with our nucleotide tokenizer.

We currently don't have a pre-trained Llama3 model to export, so this script currently just saves a randomly initialized
Llama3 model.
"""

import json
import shutil
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import convert
from modeling_llama_te import AUTO_MAP


def export_hf_checkpoint(tag: str, export_path: Path):
    """Export a Hugging Face checkpoint to a Transformer Engine checkpoint.

    Args:
        tag: The tag of the checkpoint to export.
        export_path: The parent path to export the checkpoint to.
    """
    model_hf = AutoConfig.from_pretrained(tag)
    model_hf = AutoModelForCausalLM.from_config(model_hf)

    model_te = convert.convert_llama_hf_to_te(model_hf)
    model_te.save_pretrained(export_path)

    tokenizer = AutoTokenizer.from_pretrained("nucleotide_fast_tokenizer")
    tokenizer.save_pretrained(export_path)

    # Patch the config
    with open(export_path / "config.json", "r") as f:
        config = json.load(f)

    config["auto_map"] = AUTO_MAP

    with open(export_path / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    shutil.copy("modeling_llama_te.py", export_path / "modeling_llama_te.py")


if __name__ == "__main__":
    export_hf_checkpoint("meta-llama/Llama-3.2-1B-Instruct", Path("checkpoint_export"))
