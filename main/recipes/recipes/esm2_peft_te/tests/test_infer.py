#!/usr/bin/env python3
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

import peft
import pytest
import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from dataset import SS3_ID2LABEL, SS3_LABEL2ID
from infer import _batched_inference


@pytest.fixture()
def peft_model(recipe_path):
    """Build a real 8M NV-ESM2 model with LoRA adapters (random weights, no checkpoint needed)."""
    model_path = str(recipe_path / "example_8m_checkpoint")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.attn_input_format = "bshd"
    config.id2label = SS3_ID2LABEL
    config.label2id = SS3_LABEL2ID

    base_model = AutoModelForTokenClassification.from_config(config, trust_remote_code=True, dtype=torch.bfloat16)

    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.TOKEN_CLS,
        inference_mode=True,
        r=8,
        lora_alpha=16,
        target_modules=["layernorm_qkv"],
        bias="none",
    )

    model = peft.get_peft_model(base_model, lora_config)
    model.to(device="cuda")
    model.eval()
    return model


@pytest.fixture()
def tokenizer(recipe_path):
    """Load the ESM-2 tokenizer from the local example checkpoint."""
    return AutoTokenizer.from_pretrained(str(recipe_path / "example_8m_checkpoint"))


def test_batched_inference_returns_predictions(peft_model, tokenizer):
    """Test that _batched_inference produces one prediction per input record."""
    records = [
        {"sequence": "MNEAKGVY"},
        {"sequence": "ATSSPSSPADWAKKL"},
    ]

    predictions, mapping = _batched_inference(
        model=peft_model,
        tokenizer=tokenizer,
        records=records,
        batch_size=4,
        max_seq_length=1024,
        stride=16,
        infer_overflowing_aas=False,
    )

    assert len(predictions) == len(records)
    assert len(mapping) == len(records)

    # Each prediction string must only contain valid SS3 label characters
    valid_labels = set(SS3_ID2LABEL.values())
    for pred in predictions:
        assert len(pred) > 0
        assert all(c in valid_labels for c in pred), f"Unexpected character in prediction: {pred}"

    # Mapping indices should cover all input records
    assert sorted(mapping) == list(range(len(records)))


def test_batched_inference_with_overflow(peft_model, tokenizer):
    """Test that long sequences are split into overlapping chunks via overflow."""
    long_seq = "MNEAKGVY" * 20  # 160 amino acids

    records = [{"sequence": long_seq}]

    predictions, mapping = _batched_inference(
        model=peft_model,
        tokenizer=tokenizer,
        records=records,
        batch_size=2,
        max_seq_length=32,  # short window to force multiple chunks
        stride=8,
        infer_overflowing_aas=True,
    )

    # With overflow enabled and a short window, we expect multiple chunks
    assert len(predictions) > 1, "Expected multiple chunks for a long sequence"
    assert all(idx == 0 for idx in mapping), "All chunks should map back to the single input record"


def test_batched_inference_single_record(peft_model, tokenizer):
    """Test _batched_inference with a single short sequence."""
    records = [{"sequence": "ACDE"}]

    predictions, mapping = _batched_inference(
        model=peft_model,
        tokenizer=tokenizer,
        records=records,
        batch_size=1,
        max_seq_length=1024,
        stride=16,
        infer_overflowing_aas=False,
    )

    assert len(predictions) == 1
    assert mapping == [0]


def test_batched_inference_prediction_length(peft_model, tokenizer):
    """Test that each prediction's length equals the number of non-pad tokens.

    The ESM-2 tokenizer prepends <cls> and appends <eos>, so the prediction
    string length should be len(sequence) + 2 for sequences shorter than
    max_seq_length.
    """
    seq = "MNEAKGVY"
    records = [{"sequence": seq}]

    predictions, _ = _batched_inference(
        model=peft_model,
        tokenizer=tokenizer,
        records=records,
        batch_size=1,
        max_seq_length=1024,
        stride=16,
        infer_overflowing_aas=False,
    )

    # +2 for <cls> and <eos> special tokens
    expected_length = len(seq) + 2
    assert len(predictions[0]) == expected_length, (
        f"Prediction length {len(predictions[0])} != expected {expected_length}"
    )
