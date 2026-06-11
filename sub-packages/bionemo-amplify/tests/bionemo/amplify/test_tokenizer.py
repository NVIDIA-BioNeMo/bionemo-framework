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


import pytest
import torch
from nemo.lightning import io

from bionemo.amplify.tokenizer import BioNeMoAMPLIFYTokenizer


@pytest.fixture
def tokenizer():
    return BioNeMoAMPLIFYTokenizer()


def test_tokenizer_serialization(tokenizer, tmp_path):
    tokenizer.io_dump(tmp_path / "tokenizer", yaml_attrs=[])  # BioNeMoESMTokenizer takes no __init__ arguments
    deserialized_tokenizer = io.load(tmp_path / "tokenizer", tokenizer.__class__)

    our_tokens = deserialized_tokenizer.encode("KA<mask>ISQ", add_special_tokens=False)
    amplify_tokens = torch.tensor([17, 7, 2, 14, 10, 18])
    torch.testing.assert_close(torch.tensor(our_tokens), amplify_tokens)
