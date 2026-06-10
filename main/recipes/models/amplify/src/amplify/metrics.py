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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import Mapping, Union

import torch
import transformers
from torchmetrics.text import Perplexity


perplexity = Perplexity(ignore_index=-100, sync_on_compute=False)


def nested_cpu(tensors: Union[list, tuple, Mapping, torch.Tensor]) -> Union[list, tuple, Mapping, torch.Tensor]:
    """Move tensors to the CPU.

    Args:
        tensors (Union[list, tuple, Mapping, torch.Tensor]): The tensors to move to the CPU.

    Returns:
        Union[list, tuple, Mapping, torch.Tensor]: The tensors on the CPU.
    """
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors


@torch.no_grad()
def compute_metrics(eval_pred: transformers.EvalPrediction, compute_result: bool):
    """Compute the metrics.

    Args:
        eval_pred (transformers.EvalPrediction): The evaluation predictions.
        compute_result (bool): Whether to compute the result.

    Returns:
        dict: The metrics.
    """
    # TODO (peter): Is this method even used?
    logits, labels = eval_pred
    logits = nested_cpu(logits)
    labels = nested_cpu(labels)
    perplexity(logits, labels)

    if compute_result:
        loss = perplexity.compute()
        perplexity.reset()
        return {"perplexity": loss}
