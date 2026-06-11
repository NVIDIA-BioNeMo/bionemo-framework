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

from typing import Dict, Sequence, Tuple, TypedDict

import torch
from megatron.core import tensor_parallel
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from nemo.lightning.megatron_parallel import (
    MegatronLossReduction,
    masked_token_loss,
)
from torch import Tensor


__all__: Sequence[str] = (
    "BERTMLMLossWithReduction",
    "DataParallelGroupLossAndIO",
    "PerTokenLossDict",
    "SameSizeLossDict",
)


# TODO(@sichu) update typing
class PerTokenLossDict(TypedDict):
    """Tensor dictionary for loss.

    This is the return type for a loss that is computed per token in the batch, supporting microbatches of varying sizes.
    """

    loss_sum_and_microbatch_size: Tensor


class SameSizeLossDict(TypedDict):
    """Tensor dictionary for loss.

    This is the return type for a loss that is computed for the entire batch, where all microbatches are the same size.
    """

    avg: Tensor


class DataParallelGroupLossAndIO(TypedDict):
    """Average losses across the data parallel group + the original batch and inference output."""

    avg: Tensor
    batch: dict[str, Tensor]
    forward_out: dict[str, Tensor]


class BERTMLMLossWithReduction(MegatronLossReduction):  # noqa: D101
    def __init__(self, validation_step: bool = False, val_drop_last: bool = True) -> None:  # noqa: D107
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last

    def forward(
        self, batch: Dict[str, Tensor], forward_out: Dict[str, Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward impl.

        https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/megatron_parallel.py#L1733

        Note that Method signature is slightly different from NeMo as the NeMo signature is incorrect.
        """
        # neva returns (logits, loss_mask)
        if isinstance(forward_out, tuple):
            # NOTE(SKH): this comes from NeMo- when does this occur? Directly related to the incorrect method signature.
            forward_out, loss_mask = forward_out
            batch["loss_mask"] = loss_mask

        if "labels" not in batch:
            raise ValueError("Labels not provided in the batch. These are required for this loss computation.")

        # NOTE: token_logits is [sequence, batch] but labels and other fields, including the loss are [batch, sequence]
        unreduced_token_loss = unreduced_token_loss_fn(forward_out["token_logits"], batch["labels"])  # [b s]

        loss_sum, num_valid_tokens = masked_token_loss(unreduced_token_loss, batch["loss_mask"])

        if self.validation_step and not self.val_drop_last and loss_sum.isnan():
            assert num_valid_tokens == 0, "Got NaN loss with non-empty input"
            if batch["loss_mask"].count_nonzero() != 0:
                raise ValueError("Got NaN loss with non-empty input")
            loss_sum = torch.zeros_like(num_valid_tokens)

        num_valid_tokens = num_valid_tokens.clone().detach().to(torch.int)
        loss_sum_and_ub_size = torch.cat([loss_sum.clone().detach().view(1), num_valid_tokens.view(1)])
        # Set to 1 to avoid divide by zero in the megatron scheduler:
        #  https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py#L303-L308
        if num_valid_tokens.item() == 0:
            num_valid_tokens = torch.ones_like(num_valid_tokens)

        return loss_sum, num_valid_tokens, {"loss_sum_and_ub_size": loss_sum_and_ub_size}

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        """Loss reduction impl.

        Taken from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L534-L552 .
        """
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                # legacy behavior, average over the number of microbatches
                avg = [x["avg"] for x in losses_reduced_per_micro_batch]
                loss = torch.cat(avg).mean()
                return loss

            from megatron.core import parallel_state

            loss_sum_and_ub_size = [
                x["loss_sum_and_ub_size"] for x in losses_reduced_per_micro_batch if x["loss_sum_and_ub_size"][1] > 0
            ]
            loss = (
                torch.vstack(loss_sum_and_ub_size).sum(dim=0)
                if len(loss_sum_and_ub_size) > 0
                else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
            )
            torch.distributed.all_reduce(
                loss,
                group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            # average over the total number of tokens across the global batch.
            loss = loss[0] / loss[1]

            return loss

        return torch.tensor(0.0, device=torch.cuda.current_device())


def unreduced_token_loss_fn(logits: Tensor, labels: Tensor, cross_entropy_loss_fusion: bool = False) -> Tensor:
    """Computes the unreduced token loss given the logits and labels without regard to the loss mask.

    WARNING: This function does not apply a loss mask. Also, it does inplace operation on the inputs.

    Args:
        logits (Tensor): The predicted logits of shape [sequence_length, batch_size, num_classes].
        labels (Tensor): The true labels of shape [batch_size, sequence_length].
        cross_entropy_loss_fusion (bool): If True, use the fused kernel version of vocab parallel cross entropy. This
            should generally be preferred for speed as it packs more operations into a single kernel on the GPU. However
            some users have observed reduced training stability when using this method.

    Returns:
        Tensor: The unreduced token loss of shape [batch_size, sequence_length].
    """
    labels = labels.transpose(0, 1).contiguous()  # [b, s] -> [s, b]
    if cross_entropy_loss_fusion:
        loss = fused_vocab_parallel_cross_entropy(logits, labels)
    else:
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)
    # [s b] => [b, s]
    loss = loss.transpose(0, 1).contiguous()
    return loss


def unreduced_sequence_loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
    # TODO (@jstjohn): implement this function to handle the next sequence prediction task
    # TODO (@jstjohn): determine expected shapes of logits/labels in this case and add that to the docstring
    raise NotImplementedError("Sequence loss not implemented yet.")
