# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import torch

from bionemo.amplify.model import AMPLIFYConfig, AMPLIFYModel
from bionemo.amplify.tokenizer import BioNeMoAMPLIFYTokenizer
from bionemo.llm.model.biobert.model import MegatronBioBertModel
from bionemo.testing import megatron_parallel_state_utils


def test_amplify_model_initialized():
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        tokenizer = BioNeMoAMPLIFYTokenizer()
        config = AMPLIFYConfig()
        model = config.configure_model(tokenizer)

        assert isinstance(model, MegatronBioBertModel)
        assert isinstance(model, AMPLIFYModel)


def test_amplify_model_forward_pass():
    tokenizer = BioNeMoAMPLIFYTokenizer()

    test_proteins = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA",
        "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLAGG",
    ]
    tokens = tokenizer(test_proteins, return_tensors="pt", padding=True, truncation=True).to("cuda")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    nemo_config = AMPLIFYConfig(
        num_layers=2,
        num_attention_heads=2,
        hidden_size=4,
        ffn_hidden_size=4 * 4,
    )

    with megatron_parallel_state_utils.distributed_model_parallel_state():
        nemo_model = nemo_config.configure_model(tokenizer).to("cuda").eval()
        nemo_output = nemo_model(input_ids, attention_mask)
        assert isinstance(nemo_output["token_logits"], torch.Tensor)
