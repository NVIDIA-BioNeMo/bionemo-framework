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

from types import SimpleNamespace

import torch

from bionemo.evo2_phage_gen import nemo_rl_processors
from bionemo.evo2_phage_gen.nemo_rl_processors import phage_prompt_data_processor


class _Tokenizer:
    def __call__(self, text: str, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {"input_ids": torch.tensor([[ord(char) for char in text]], dtype=torch.long)}


def test_phage_openai_dataset_uses_stable_task_name_independent_of_path(tmp_path):
    """Moving a prompt bank must not change its environment metric namespace."""
    dataset_path = tmp_path / "arbitrary-run" / "rl" / "validation.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text('{"messages":[{"role":"user","content":"ACGT"},{"role":"assistant","content":""}]}\n')

    dataset_class = getattr(nemo_rl_processors, "PhageOpenAIFormatDataset", None)
    assert dataset_class is not None
    dataset = dataset_class(data_path=str(dataset_path), use_preserving_dataset=True)

    assert dataset.task_name == "phage_qc"
    assert dataset.dataset[0]["task_name"] == "phage_qc"


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
        "prompt_nt_length": 4,
        "prompt_index": 3,
    }
    assert output["loss_multiplier"] == 1.0


def test_phage_prompt_data_processor_recomputes_length_after_truncation():
    output = phage_prompt_data_processor(
        {"prompt": "ACGTACGT"}, SimpleNamespace(prompt=None), _Tokenizer(), max_seq_length=3, idx=0
    )

    assert output["length"] == 3
    assert output["message_log"][0]["token_ids"].shape == (3,)
    assert output["loss_multiplier"] == 0.0


def test_phage_prompt_data_processor_limits_long_prompts_to_four_tokens():
    """Long prompts retain only the four-token generation prefix."""
    output = phage_prompt_data_processor(
        {"prompt": "ACGTACGT"}, SimpleNamespace(prompt=None), _Tokenizer(), max_seq_length=6, idx=7
    )

    assert output["length"] == 4
    assert output["message_log"][0]["token_ids"].tolist() == [65, 67, 71, 84]
    assert output["extra_env_info"] == {"prompt_nt_length": 8, "prompt_index": 7}
    assert output["loss_multiplier"] == 0.0


def test_phage_prompt_data_processor_formats_task_prompt_template():
    """Task prompt templates are applied before tokenization and metadata extraction."""
    output = phage_prompt_data_processor(
        {"prompt": "AC"}, SimpleNamespace(prompt="prefix:{}"), _Tokenizer(), max_seq_length=20, idx=2
    )

    assert output["message_log"][0]["content"] == "prefix:AC"
    assert output["message_log"][0]["token_ids"].tolist() == [ord(char) for char in "prefix:AC"]
    assert output["extra_env_info"] == {"prompt_nt_length": 2, "prompt_index": 2}
    assert output["loss_multiplier"] == 1.0
