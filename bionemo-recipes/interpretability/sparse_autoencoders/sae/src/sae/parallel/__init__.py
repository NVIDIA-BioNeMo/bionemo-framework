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

"""Tensor-parallel building blocks for latent-sharded SAEs."""

from .checkpoint import load_and_merge, save_sharded
from .comms import all_gather_cat, all_reduce_sum, autograd_all_reduce_sum
from .topk import GlobalTopK, dense_topk_reference, global_topk
from .training import train_tp_loop


__all__ = [
    "GlobalTopK",
    "all_gather_cat",
    "all_reduce_sum",
    "autograd_all_reduce_sum",
    "dense_topk_reference",
    "global_topk",
    "load_and_merge",
    "save_sharded",
    "train_tp_loop",
]
