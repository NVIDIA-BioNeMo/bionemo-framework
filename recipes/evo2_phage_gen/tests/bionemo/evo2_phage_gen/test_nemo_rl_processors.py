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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from types import SimpleNamespace

import torch

from bionemo.evo2_phage_gen.nemo_rl_processors import phage_prompt_data_processor


class _Tokenizer:
    def __call__(self, text: str, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {"input_ids": torch.tensor([[ord(char) for char in text]], dtype=torch.long)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_phage_prompt_data_processor_tokenizes_openai_user_message_without_chat_template():
    datum = {
        "task_name": "phage",
        "messages": [
            {"role": "user", "content": "ACGT"},
            {"role": "assistant", "content": ""},
        ],
    }
    task_spec = SimpleNamespace(prompt=None)

    output = phage_prompt_data_processor(datum, task_spec, _Tokenizer(), max_seq_length=8, idx=3)

    assert output["idx"] == 3
    assert output["task_name"] == "phage"
    assert output["length"] == 4
    assert len(output["message_log"]) == 1
    message = output["message_log"][0]
    assert message["role"] == "user"
    assert message["content"] == "ACGT"
    assert torch.equal(message["token_ids"], torch.tensor([65, 67, 71, 84], dtype=torch.long))
    assert output["extra_env_info"] == {
        "prompt": "ACGT",
        "prompt_nt_length": 4,
        "prompt_prefix": "ACGT",
        "prompt_index": 3,
    }


def test_phage_prompt_data_processor_preserves_mixed_prompt_identity() -> None:
    datum = {
        "messages": [
            {"role": "user", "content": "+~GAGTTTTA"},
            {"role": "assistant", "content": ""},
        ],
        "prompt_id": "phix174_length_08",
        "length_stratum": 8,
        "rollout_ordinal": 5,
        "order_index": 44,
        "validation_seed": 86,
    }

    output = phage_prompt_data_processor(datum, SimpleNamespace(prompt=None), _Tokenizer(), max_seq_length=32, idx=44)

    extra = output["extra_env_info"]
    _require(extra["prompt_id"] == "phix174_length_08", "prompt ID was dropped")
    _require(extra["length_stratum"] == 8, "length stratum was dropped")
    _require(extra["rollout_ordinal"] == 5, "rollout ordinal was dropped")
    _require(extra["order_index"] == 44, "manifest order was dropped")
    _require(extra["validation_seed"] == 86, "validation seed was dropped")
