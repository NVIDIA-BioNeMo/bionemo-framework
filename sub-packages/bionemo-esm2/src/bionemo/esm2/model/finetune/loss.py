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


from typing import Dict, Sequence, Tuple

import torch
from megatron.core import parallel_state
from nemo.lightning.megatron_parallel import masked_token_loss
from torch import Tensor

from bionemo.llm.model.loss import (
    BERTMLMLossWithReduction,
    unreduced_token_loss_fn,
)


__all__: Sequence[str] = (
    "ClassifierLossReduction",
    "RegressorLossReduction",
)


class RegressorLossReduction(BERTMLMLossWithReduction):
    """A class for calculating the MSE loss of regression output.

    This class used for calculating the loss, and for logging the reduced loss across micro batches.
    """

    def forward(
        self, batch: Dict[str, Tensor], forward_out: Dict[str, Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculates the sum of squared errors within a micro-batch. A micro-batch is a batch of data on a single GPU. The averaging of the loss, i.e. MSE loss, is done in https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py#L304-L314.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside classification head.
        """
        regression_output = forward_out["regression_output"]
        targets = batch["labels"].to(dtype=regression_output.dtype)  # [b, 1]

        num_valid_tokens = torch.tensor(targets.numel(), dtype=torch.int, device=targets.device)

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            loss_sum = ((regression_output - targets) ** 2).sum()  # torch.float
        else:
            raise NotImplementedError("Context Parallel support is not implemented for this loss")

        loss_sum_and_ub_size = torch.cat([loss_sum.clone().detach().view(1), num_valid_tokens.view(1)])
        return loss_sum, num_valid_tokens, {"loss_sum_and_ub_size": loss_sum_and_ub_size}


class ClassifierLossReduction(BERTMLMLossWithReduction):
    """A class for calculating the cross entropy loss of classification output.

    This class used for calculating the loss, and for logging the reduced loss across micro batches.
    """

    def forward(
        self, batch: Dict[str, Tensor], forward_out: Dict[str, Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU. The averaging of the loss is done in https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py#L304-L314.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside classification head.
        """
        targets = batch["labels"].squeeze()  # [b] or [b, s] for sequence-level or token-level classification
        loss_mask = batch["loss_mask"]

        classification_output = forward_out["classification_output"]  # [b, num_class] or [b, s, num_class]
        if classification_output.dim() == 3:
            classification_output = classification_output.permute(1, 0, 2).contiguous()  # change to [s, b, num_class]
        elif classification_output.dim() == 2:
            # NOTE: this is for sequence-level classification, we artificially create a sequence dimension to use the same code path as token-level classification
            classification_output = classification_output.unsqueeze(0)  # change to [1, b, num_class]
            targets = targets.unsqueeze(1)  # change to [b, 1]
            loss_mask = torch.ones((targets.shape[0], 1), dtype=loss_mask.dtype, device=loss_mask.device)
        else:
            raise ValueError(f"Unexpected classification output dimension: {classification_output.dim()}")

        # NOTE: token_logits is [sequence, batch] but labels and other fields, including the loss are [batch, sequence]
        unreduced_token_loss = unreduced_token_loss_fn(classification_output, targets)  # [b s]
        loss_sum, num_valid_tokens = masked_token_loss(unreduced_token_loss, loss_mask)  # torch.float, torch.int

        if self.validation_step and not self.val_drop_last and loss_sum.isnan():
            assert num_valid_tokens == 0, "Got NaN loss with non-empty input"
            if loss_mask.count_nonzero() != 0:
                raise ValueError("Got NaN loss with non-empty input")
            loss_sum = torch.zeros_like(num_valid_tokens)

        num_valid_tokens = num_valid_tokens.clone().detach().to(torch.int)
        loss_sum_and_ub_size = torch.cat([loss_sum.clone().detach().view(1), num_valid_tokens.view(1)])
        return loss_sum, num_valid_tokens, {"loss_sum_and_ub_size": loss_sum_and_ub_size}
