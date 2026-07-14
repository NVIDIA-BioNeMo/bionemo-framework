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

import sys

import vllm
from transformers import AutoConfig


def test_register_is_lazy_and_idempotent(monkeypatch):
    calls = []
    supported = set()
    monkeypatch.setattr(vllm.ModelRegistry, "get_supported_archs", lambda: supported)

    def record(name, target):
        calls.append((name, target))
        supported.add(name)

    monkeypatch.setattr(vllm.ModelRegistry, "register_model", record)

    from bionemo.evo2.vllm.plugin import register

    register()
    register()

    assert calls == [("Evo2ForCausalLM", "bionemo.evo2.vllm.model:Evo2ForCausalLM")]
    assert "bionemo.evo2.vllm.model" not in sys.modules


def test_register_adds_transformers_config(monkeypatch):
    monkeypatch.setattr(vllm.ModelRegistry, "get_supported_archs", lambda: {"Evo2ForCausalLM"})

    from bionemo.evo2.vllm.config import Evo2Config
    from bionemo.evo2.vllm.plugin import register

    register()

    config = AutoConfig.for_model("evo2", num_hidden_layers=1, hybrid_override_pattern="*")
    assert isinstance(config, Evo2Config)
    assert "bionemo.evo2.vllm.model" not in sys.modules
