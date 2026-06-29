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

import warnings

import peft
import pytest
import torch

from modeling_esm_te import NVEsmForMaskedLM


def test_create_peft_model(te_model_checkpoint):
    model = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.TOKEN_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        target_parameters=["layernorm_qkv.weight"],
        bias="none",
    )

    peft_model = peft.get_peft_model(model, peft_config)
    status = peft_model.get_model_status()
    assert status.trainable_params <= 200_000
    assert status.total_params >= 7_500_000


def test_lora_model_forward_pass(te_model_checkpoint, input_data):
    model = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.TOKEN_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        target_parameters=["layernorm_qkv.weight"],
        bias="none",
    )

    peft_model = peft.get_peft_model(model, peft_config)
    peft_model.to("cuda")

    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    outputs = peft_model(**input_data)
    assert outputs.loss is not None


@pytest.mark.xfail(reason="BIONEMO-3136: LoRA model initializes with warnings because of TE layers.")
def test_lora_model_raises_no_warnings(te_model_checkpoint):
    model = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.TOKEN_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        target_parameters=["layernorm_qkv.weight"],
        bias="none",
    )

    with warnings.catch_warnings(record=True) as record:
        # Cause all warnings to be triggered (default behavior may ignore some)
        warnings.simplefilter("always")
        peft.get_peft_model(model, peft_config)

    assert len(record) == 0


@pytest.mark.xfail(reason="BIONEMO-3136: LoRA model initialization fails with target_modules because of TE layers.")
def test_lora_model_with_target_modules(te_model_checkpoint):
    model = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.TOKEN_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        target_modules=["layernorm_qkv"],
        bias="none",
    )

    peft.get_peft_model(model, peft_config)
