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

import pytest
from transformer_engine.pytorch import TransformerLayer
from transformers import AutoModelForCausalLM, AutoTokenizer

from export import export_hf_checkpoint


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI not download llama3 model.")
def test_export_llama3_checkpoint(tmp_path):
    export_hf_checkpoint("meta-llama/Llama-3.2-1B-Instruct", tmp_path / "checkpoint_export")

    _ = AutoTokenizer.from_pretrained(tmp_path / "checkpoint_export")
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "checkpoint_export", trust_remote_code=True)
    assert "NVLlamaForCausalLM" in model.__class__.__name__
    assert "NVLlamaConfig" in model.config.__class__.__name__
    assert isinstance(model.model.layers[0], TransformerLayer)
