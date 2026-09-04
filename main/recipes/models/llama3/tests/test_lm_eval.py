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
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from modeling_llama_te import AUTO_MAP, NVLlamaConfig, NVLlamaForCausalLM


@pytest.fixture
def model_checkpoint(tmp_path: Path):
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8", revision="42d9515")
    config = NVLlamaConfig.from_pretrained(
        "nvidia/Llama-3.1-8B-Instruct-FP8",
        num_hidden_layers=2,
        attn_input_format="bshd",
        self_attn_mask_type="causal",
        revision="42d9515",
    )
    model = NVLlamaForCausalLM(config)
    model.save_pretrained(tmp_path / "checkpoint")

    tokenizer = AutoTokenizer.from_pretrained("nucleotide_fast_tokenizer")
    tokenizer.save_pretrained(tmp_path / "checkpoint")

    # Patch the config
    with open(tmp_path / "checkpoint" / "config.json", "r") as f:
        config = json.load(f)

    config["auto_map"] = AUTO_MAP

    with open(tmp_path / "checkpoint" / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    shutil.copy("modeling_llama_te.py", tmp_path / "checkpoint" / "modeling_llama_te.py")
    return tmp_path / "checkpoint"


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping slow lm-eval test in CI.")
def test_lm_eval(model_checkpoint: Path):
    # Create a mock model checkpoint

    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_checkpoint},tokenizer={model_checkpoint}",
        "--trust_remote_code",
        "--tasks",
        "arc_easy",  # TODO(BIONEMO-3410): support other tasks that use inference, e.g. coqa
        "--device",
        "cuda:0",
        "--batch_size",
        "8",
    ]

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")
