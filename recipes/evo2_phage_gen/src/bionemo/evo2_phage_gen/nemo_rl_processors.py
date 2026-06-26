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

"""NeMo-RL data processors for nucleotide-only Evo2 prompts."""

from __future__ import annotations

from typing import Any

from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec


def _extract_prompt(datum_dict: dict[str, Any]) -> str:
    if "prompt" in datum_dict:
        return str(datum_dict["prompt"])

    messages = datum_dict.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))

    raise ValueError("Expected a `prompt` field or an OpenAI-style user message.")


def phage_prompt_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: Any,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Tokenize a raw DNA-design prompt without applying a chat template."""
    prompt = _extract_prompt(datum_dict)
    if task_data_spec.prompt:
        prompt = task_data_spec.prompt.format(prompt)

    token_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    length = int(token_ids.shape[0])

    loss_multiplier = 1.0
    if length >= max_seq_length:
        token_ids = token_ids[: min(4, max_seq_length)]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": [{"role": "user", "content": prompt, "token_ids": token_ids}],
        "length": length,
        "extra_env_info": {"prompt": prompt},
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output
