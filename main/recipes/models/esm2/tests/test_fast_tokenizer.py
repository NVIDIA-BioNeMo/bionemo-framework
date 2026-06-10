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

from transformers import AutoTokenizer


def test_tokenizer_vocab_equivalence():
    original_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", revision="c731040f")
    fast_tokenizer = AutoTokenizer.from_pretrained("esm_fast_tokenizer")
    assert original_tokenizer.get_vocab() == fast_tokenizer.get_vocab()


def test_tokenizer_tokenization_equivalence(test_proteins):
    original_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", revision="c731040f")
    fast_tokenizer = AutoTokenizer.from_pretrained("esm_fast_tokenizer")

    original_output = original_tokenizer(test_proteins)
    fast_output = fast_tokenizer(test_proteins)

    assert len(original_output["input_ids"]) == len(fast_output["input_ids"])
    assert set(original_output.keys()) == set(fast_output.keys())

    for original_input_id, fast_input_id in zip(original_output["input_ids"], fast_output["input_ids"]):
        assert len(original_input_id) == len(fast_input_id)
        assert original_input_id == fast_input_id

    for original_attention_mask, fast_attention_mask in zip(
        original_output["attention_mask"], fast_output["attention_mask"]
    ):
        assert original_attention_mask == fast_attention_mask
