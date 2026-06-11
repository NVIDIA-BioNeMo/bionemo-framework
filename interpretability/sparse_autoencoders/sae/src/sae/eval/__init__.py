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

"""Generic SAE evaluation utilities."""

from .dead_latents import DeadLatentStats, DeadLatentTracker
from .evaluate import EvalResults, evaluate_sae
from .loss_recovered import (
    LossRecoveredResult,
    compute_loss_recovered,
    evaluate_loss_recovered,
)
from .reconstruction import (
    ReconstructionMetrics,
    compute_reconstruction_metrics,
    evaluate_reconstruction,
)
from .sparsity import SparsityMetrics, evaluate_sparsity


__all__ = [
    "DeadLatentStats",
    "DeadLatentTracker",
    "EvalResults",
    "LossRecoveredResult",
    "ReconstructionMetrics",
    "SparsityMetrics",
    "compute_loss_recovered",
    "compute_reconstruction_metrics",
    "evaluate_loss_recovered",
    "evaluate_reconstruction",
    "evaluate_sae",
    "evaluate_sparsity",
]
