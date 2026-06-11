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
"""Utility functions for Evo2 run functions."""

from typing import Literal

from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS

from bionemo.evo2.models.llama import LLAMA_MODEL_OPTIONS
from bionemo.evo2.models.mamba import MAMBA_MODEL_OPTIONS


def patch_eden_tokenizer(tokenizer):
    """Patch the Eden tokenizer to work with the Evo2 tokenizer."""
    bos_id, eos_id, sep_id, pad_id = 1, 2, 3, 0

    # Patch the private attrs so tokenizer.bos_id/.eos_id/.pad_id work
    tokenizer._bos_id = bos_id
    tokenizer._eos_id = eos_id
    tokenizer._sep_id = sep_id
    tokenizer._pad_id = pad_id


def infer_model_type(model_size: str) -> Literal["hyena", "mamba", "llama"]:
    """Infer the model type from the model size."""
    all_keys = set(HYENA_MODEL_OPTIONS.keys()) | set(MAMBA_MODEL_OPTIONS.keys()) | set(LLAMA_MODEL_OPTIONS.keys())
    if len(all_keys) != len(HYENA_MODEL_OPTIONS.keys()) + len(MAMBA_MODEL_OPTIONS.keys()) + len(
        LLAMA_MODEL_OPTIONS.keys()
    ):
        raise ValueError(
            "Duplicate model sizes found in HYENA_MODEL_OPTIONS, MAMBA_MODEL_OPTIONS, and LLAMA_MODEL_OPTIONS."
        )
    if model_size in HYENA_MODEL_OPTIONS:
        return "hyena"
    elif model_size in MAMBA_MODEL_OPTIONS:
        return "mamba"
    elif model_size in LLAMA_MODEL_OPTIONS:
        return "llama"
    else:
        raise ValueError(f"Invalid model size: {model_size}")
