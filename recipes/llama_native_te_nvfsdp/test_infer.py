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

import pytest
import torch
from transformers import AutoConfig
from transformers import LlamaForCausalLM

from model import NVLlamaForCausalLM


@pytest.mark.parametrize("model_name", ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"])
def test_hf_and_te_llama_equivalence(model_name: str, tol=0.25):
    device = torch.device("cuda:0")

    config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float16)
    # Dummy input
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, 32), device=device)

    nv_model = NVLlamaForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16).to(device)
    nv_model.eval()

    hf_model = LlamaForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16).to(device)
    hf_model.eval()

    with torch.no_grad():
        nv_logits = nv_model(input_ids).logits
        hf_logits = hf_model(input_ids).logits

    assert hf_logits.shape == nv_logits.shape, \
        f"Shape mismatch: HF {hf_logits.shape} vs NV {nv_logits.shape}"

    assert torch.allclose(hf_logits, nv_logits, atol=tol, rtol=tol), f"Models differ more than tol={tol}"
