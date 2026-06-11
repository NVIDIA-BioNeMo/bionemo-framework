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


from dataclasses import dataclass

import torch

from bionemo.llm.data.types import Tokenizer


@dataclass(frozen=True)
class BertMaskConfig:
    """Configuration for masking tokens in a BERT-style model.

    Attributes:
        mask_prob: Probability of masking a token.
        mask_token_prob: Probability of replacing a masked token with the mask token.
        random_token_prob: Probability of replacing a masked token with a random token.
    """

    tokenizer: Tokenizer
    random_tokens: range
    mask_prob: float = 0.15
    mask_token_prob: float = 0.8
    random_token_prob: float = 0.1

    def __post_init__(self) -> None:
        """Check that the sum of `mask_token_prob` and `random_token_prob` is less than or equal to 1.0.

        Raises:
            ValueError: If the sum of `mask_token_prob` and `random_token_prob` is greater than 1.0.
        """
        if self.random_token_prob + self.mask_token_prob > 1.0:
            raise ValueError("Sum of random_token_prob and mask_token_prob must be less than or equal to 1.0.")


def apply_bert_pretraining_mask(
    tokenized_sequence: torch.Tensor, random_seed: int, mask_config: BertMaskConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies the pretraining mask to a tokenized sequence.

    Args:
        tokenized_sequence: Tokenized protein sequence.
        random_seed: Random seed for reproducibility.
        mask_config: Configuration for masking tokens in a BERT-style model.

    Returns:
        masked_sequence:
            The tokenized sequence with some tokens masked.
        labels:
            A tensor the same shape as `masked_sequence` containing labels for the masked tokens, with -1 for non-masked
            tokens.
        loss_mask:
            A boolean tensor the same shape as `masked_sequence`, where 'True' indicates which tokens should be included
            in the loss.
    """
    if mask_config.tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must have a mask token.")

    if mask_config.random_token_prob + mask_config.mask_token_prob > 1.0:
        raise ValueError("Sum of random_token_prob and mask_token_prob must be less than or equal to 1.0.")

    # Set the seed so that __getitem__(idx) is always deterministic.
    # This is required by Megatron-LM's parallel strategies.
    generator = torch.Generator().manual_seed(random_seed)

    mask_stop_1 = mask_config.mask_prob * mask_config.mask_token_prob
    mask_stop_2 = mask_config.mask_prob * (mask_config.mask_token_prob + mask_config.random_token_prob)

    random_draws = torch.rand(tokenized_sequence.shape, generator=generator)  # Random draws for each token in [0, 1).

    # Overall mask for a token being masked in some capacity - either mask token, random token, or left as-is
    # (identity). We don't want to mask special tokens.
    loss_mask = ~torch.isin(tokenized_sequence, torch.tensor(mask_config.tokenizer.all_special_ids))
    loss_mask &= random_draws < mask_config.mask_prob

    # The first `mask_token_prob` fraction of the `mask_prob` tokens are replaced with the mask token.
    mask_token_mask = (random_draws < mask_stop_1) & loss_mask

    # The next `random_token_prob` fraction of the `mask_prob` tokens are replaced with a random token.
    random_token_mask = ((random_draws >= mask_stop_1) & (random_draws < mask_stop_2)) & loss_mask

    # The remaining tokens are implicitly left as-is, representing an identity mask.

    # Mask the tokens.
    masked_sequence = tokenized_sequence.clone()
    masked_sequence[mask_token_mask] = mask_config.tokenizer.mask_token_id
    num_random_tokens: int = random_token_mask.sum().item()  # type: ignore[assignment]
    masked_sequence[random_token_mask] = torch.randint(
        low=mask_config.random_tokens.start,
        high=mask_config.random_tokens.stop,
        size=(num_random_tokens,),
        dtype=masked_sequence.dtype,
        generator=generator,
    )

    # Create the labels for the masked tokens.
    labels = tokenized_sequence.clone()
    labels[~loss_mask] = -100  # Ignore loss for non-masked tokens.

    return masked_sequence, labels, loss_mask


def add_cls_and_eos_tokens(
    sequence: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    cls_token: int | None = None,
    eos_token: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepends the CLS token and appends the EOS token to the masked sequence, updating the loss mask and labels.

    These labels should never be masked, so this is done after the masking step.

    Args:
        sequence: The input (likely masked) sequence.
        labels: The true values of the input sequence at the mask positions.
        loss_mask: A boolean tensor indicating which tokens should be included in the loss.
        cls_token: The token to use for the CLS token. If None, no CLS token is added.
        eos_token: The token to use for the EOS token. If None, no EOS token is added.

    Returns:
        The same input tensors with the CLS and EOS tokens added, and the labels and loss_mask updated accordingly.
    """
    # Prepend the CLS token and append the EOS token, and update the loss mask and labels accordingly.
    sequence = torch.cat(
        [
            torch.tensor([cls_token], dtype=sequence.dtype)
            if cls_token is not None
            else torch.tensor([], dtype=sequence.dtype),
            sequence,
            torch.tensor([eos_token], dtype=sequence.dtype)
            if eos_token is not None
            else torch.tensor([], dtype=sequence.dtype),
        ]
    )

    labels = torch.cat(
        [
            torch.tensor([-1], dtype=labels.dtype) if cls_token is not None else torch.tensor([], dtype=labels.dtype),
            labels,
            torch.tensor([-1], dtype=labels.dtype) if eos_token is not None else torch.tensor([], dtype=labels.dtype),
        ]
    )

    loss_mask = torch.cat(
        [
            torch.tensor([False]) if cls_token is not None else torch.tensor([], dtype=loss_mask.dtype),
            loss_mask,
            torch.tensor([False]) if eos_token is not None else torch.tensor([], dtype=loss_mask.dtype),
        ]
    )

    return sequence, labels, loss_mask
