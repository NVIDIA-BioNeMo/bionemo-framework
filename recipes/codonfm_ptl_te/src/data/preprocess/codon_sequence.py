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


import numpy as np


def process_item(seq, context_length, tokenizer):  # noqa: D103
    input_sequence_toks = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq))
    input_sequence_toks = np.asarray(input_sequence_toks)[: (context_length - 2)]

    # - add CLS token
    input_sequence_toks = np.insert(input_sequence_toks, 0, tokenizer.cls_token_id)

    # - add SEP token
    input_sequence_toks = np.append(input_sequence_toks, tokenizer.sep_token_id)

    attention_mask = np.ones(context_length, dtype=np.int64)
    attention_mask[len(input_sequence_toks) :] = 0
    # Pad/truncate to context_length using numpy
    input_sequence_toks = np.pad(
        input_sequence_toks,
        (0, max(0, context_length - len(input_sequence_toks))),
        mode="constant",
        constant_values=tokenizer.pad_token_id,
    )
    # input_sequence_toks = input_sequence_toks[:context_length]
    return {
        "input_ids": np.asarray(input_sequence_toks, dtype=np.int64),
        "attention_mask": np.asarray(attention_mask, dtype=np.int64),
    }
