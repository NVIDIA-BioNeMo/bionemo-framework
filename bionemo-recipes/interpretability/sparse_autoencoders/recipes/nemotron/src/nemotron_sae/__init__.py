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

"""Nemotron SAE: Sparse Autoencoders for Nemotron-3-Nano language model.

This package provides tools for training and analyzing sparse autoencoders
on Nemotron-3-Nano activations, including model wrappers, data loading
utilities, and auto-interpretation helpers.
"""

from .data import load_fineweb
from .eval import evaluate_nemotron_loss_recovered
from .interp import TEXT_PROMPT_TEMPLATE, create_text_formatter
from .models import NemotronModel


__all__ = [
    "TEXT_PROMPT_TEMPLATE",
    "NemotronModel",
    "create_text_formatter",
    "evaluate_nemotron_loss_recovered",
    "load_fineweb",
]
