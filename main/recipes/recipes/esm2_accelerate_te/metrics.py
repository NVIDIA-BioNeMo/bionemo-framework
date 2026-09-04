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

from typing import Mapping

import torch
import transformers
from torchmetrics.text import Perplexity


perplexity = Perplexity(ignore_index=-100, sync_on_compute=False)


def nested_cpu(tensors):
    """Move `tensors` to CPU (even if it's a nested list/tuple/dict of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    return tensors.detach().cpu() if isinstance(tensors, torch.Tensor) else tensors


@torch.no_grad()
def compute_metrics(eval_pred: transformers.EvalPrediction, compute_result: bool):
    """Compute perplexity metrics for the evaluation set.

    Args:
        eval_pred: A tuple containing the logits and labels for the evaluation set.
        compute_result: A boolean indicating whether to compute the perplexity metrics.

    Returns:
        A dictionary containing the perplexity metrics.
    """
    logits, labels = eval_pred
    logits = nested_cpu(logits)
    labels = nested_cpu(labels)
    perplexity(logits, labels)

    if compute_result:
        loss = perplexity.compute()
        perplexity.reset()
        return {"perplexity": loss}
