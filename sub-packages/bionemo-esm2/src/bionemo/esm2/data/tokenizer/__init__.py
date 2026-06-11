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

import functools
from importlib.resources import files

import transformers
from nemo.lightning.io import IOMixin


class BioNeMoESMTokenizer(transformers.EsmTokenizer, IOMixin):  # noqa D101
    def __init__(self):
        """A wrapper to make AutoTokenizer serializable for the ESM2 tokenizer."""
        other = transformers.AutoTokenizer.from_pretrained(str(files("bionemo.esm2.data.tokenizer")), use_fast=True)
        self.__dict__.update(dict(other.__dict__))


@functools.cache
def get_tokenizer() -> BioNeMoESMTokenizer:
    """Get the tokenizer for the ESM2 model."""
    return BioNeMoESMTokenizer()
