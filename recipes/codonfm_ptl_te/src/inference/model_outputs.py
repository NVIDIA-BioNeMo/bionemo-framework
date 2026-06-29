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


from dataclasses import dataclass

import numpy as np


@dataclass
class MaskedLMOutput:
    """Output for masked language modeling predictions.

    Attributes:
        preds: Array of predicted logits or probabilities for masked tokens.
        labels: Array of ground-truth labels for masked positions.
        ids: Optional identifier array aligned with predictions.
    """

    preds: np.ndarray
    labels: np.ndarray
    ids: np.ndarray = None


@dataclass
class MutationPredictionOutput:
    """Output for mutation effect predictions.

    Attributes:
        ref_likelihoods: Log-likelihoods for reference codons at mutation sites.
        alt_likelihoods: Log-likelihoods for alternate codons at mutation sites.
        likelihood_ratios: Difference ref - alt indicating predicted effect.
        ids: Optional identifiers per prediction.
    """

    ref_likelihoods: np.ndarray
    alt_likelihoods: np.ndarray
    likelihood_ratios: np.ndarray
    ids: np.ndarray = None


@dataclass
class FitnessPredictionOutput:
    """Output for per-sequence fitness predictions.

    Attributes:
        fitness: Mean log-likelihood or analogous scalar score per sequence.
        ids: Optional identifiers per sequence.
    """

    fitness: np.ndarray
    ids: np.ndarray = None


@dataclass
class EmbeddingOutput:
    """Output for sequence embedding extraction.

    Attributes:
        embeddings: Array of embeddings, typically from [CLS] positions.
        ids: Optional identifiers per embedding.
    """

    embeddings: np.ndarray
    ids: np.ndarray = None


@dataclass
class DownstreamPredictionOutput:
    """Output for downstream task predictions (classification or regression).

    Attributes:
        predictions: Raw predictions from the downstream head.
        probabilities: Class probabilities for classification tasks.
        predicted_classes: Argmax class indices for classification tasks.
        ids: Optional identifiers per prediction.
    """

    predictions: np.ndarray
    probabilities: np.ndarray = None
    predicted_classes: np.ndarray = None
    ids: np.ndarray = None
