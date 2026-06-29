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


from enum import Enum


class TaskTypes(str, Enum):
    """Enumeration of supported inference task types.

    Values:
        MUTATION_PREDICTION: Predict mutation effects via likelihood ratios.
        MASKED_LANGUAGE_MODELING: Predict masked token identities.
        FITNESS_PREDICTION: Predict per-sequence fitness scores.
        EMBEDDING_PREDICTION: Extract sequence embeddings.
        DOWNSTREAM_PREDICTION: Predict using downstream cross-attention head.
    """

    MUTATION_PREDICTION = "mutation_prediction"
    MASKED_LANGUAGE_MODELING = "masked_language_modeling"
    FITNESS_PREDICTION = "fitness_prediction"
    EMBEDDING_PREDICTION = "embedding_prediction"
    DOWNSTREAM_PREDICTION = "downstream_prediction"
