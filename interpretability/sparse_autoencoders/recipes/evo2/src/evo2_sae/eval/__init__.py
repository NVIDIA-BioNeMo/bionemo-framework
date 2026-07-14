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

"""Evo2 SAE evaluation: probing primitives (eval metrics + ActivationBuffer)."""

from .probing import (
    ActivationBuffer,
    annotate_features,
    auroc_all,
    auroc_vec,
    best_single_train_test,
    decode_eval,
    domain_f1,
    fit_logreg,
    fit_softmax,
    macro_auroc,
    split_indices,
    standardize,
)


__all__ = [
    "ActivationBuffer",
    "annotate_features",
    "auroc_all",
    "auroc_vec",
    "best_single_train_test",
    "decode_eval",
    "domain_f1",
    "fit_logreg",
    "fit_softmax",
    "macro_auroc",
    "split_indices",
    "standardize",
]
