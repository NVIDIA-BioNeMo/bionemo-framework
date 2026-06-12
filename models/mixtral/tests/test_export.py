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

import os

import pytest
from transformer_engine.pytorch import MultiheadAttention
from transformers import AutoModelForCausalLM, AutoTokenizer

from export import export_hf_checkpoint


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI, requires Mini-Mixtral download.")
def test_export_mixtral_checkpoint(tmp_path):
    export_hf_checkpoint("NeuralNovel/Mini-Mixtral-v0.2", tmp_path / "checkpoint_export")

    _ = AutoTokenizer.from_pretrained(tmp_path / "checkpoint_export")
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "checkpoint_export", trust_remote_code=True)
    assert "NVMixtralForCausalLM" in model.__class__.__name__
    assert "NVMixtralConfig" in model.config.__class__.__name__
    # Mixtral uses custom NVMixtralDecoderLayer with TE MultiheadAttention sub-modules
    assert isinstance(model.model.layers[0].self_attention, MultiheadAttention)
