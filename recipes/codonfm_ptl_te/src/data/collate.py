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


import torch

from src.data.metadata import MetadataFields


_waste_history: list[float] = []
_expected_padded_length: int | None = None


def thd_collate_fn(batch):
    """Collate function for MLM that mimics PyTorch's default_collate behavior.

    Takes a list of dicts (one per sample) and stacks them into a single dict
    of batched tensors.

    Args:
        batch: List[Dict[str, np.ndarray]] - List of samples from process_item
               Each sample is a dict with keys: INPUT_IDS, LABELS, ATTENTION_MASK, INPUT_MASK

    Returns:
        Dict[str, torch.Tensor] - Dict with same keys, values are batched tensors
                                  with shape [batch_size, seq_len]
    """
    # Get all keys from the first item
    batch = _pt_flatten_collate(features=batch)
    return batch


def _pt_flatten_collate(features: list[dict[str, list[int]]]):
    """Flatten the features into a single batch.

    Args:
        features: List of dictionaries containing the features to flatten.

    Returns:
        Dictionary containing the flattened features.
    """
    global _expected_padded_length  # noqa: PLW0603
    padded_length = len(features[0][MetadataFields.INPUT_IDS])
    if _expected_padded_length is None:
        _expected_padded_length = padded_length
    assert padded_length == _expected_padded_length, (
        f"padded_length changed: expected {_expected_padded_length}, got {padded_length}"
    )

    # Unpad everything.
    for feature in features:
        feature[MetadataFields.INPUT_IDS] = feature[MetadataFields.INPUT_IDS][feature[MetadataFields.ATTENTION_MASK]]
        feature[MetadataFields.LABELS] = feature[MetadataFields.LABELS][feature[MetadataFields.ATTENTION_MASK]]
        feature[MetadataFields.INPUT_MASK] = feature[MetadataFields.INPUT_MASK][feature[MetadataFields.ATTENTION_MASK]]

    is_labels_provided = MetadataFields.LABELS in features[0]
    sample_lengths = [len(sample[MetadataFields.INPUT_IDS]) for sample in features]

    bshd_tokens = len(features) * padded_length
    thd_tokens = sum(sample_lengths)
    batch_waste = 1.0 - thd_tokens / bshd_tokens
    _waste_history.append(batch_waste)
    avg_waste = sum(_waste_history) / len(_waste_history)
    print(
        f"[THD collate] batch waste: {batch_waste:.1%} "
        f"| running avg: {avg_waste:.1%} "
        f"| tokens {thd_tokens}/{bshd_tokens} "
        f"| seq lens: {sample_lengths}"
    )

    batch = {}
    batch[MetadataFields.MAX_LENGTH_Q] = batch[MetadataFields.MAX_LENGTH_K] = max(sample_lengths)

    # CRITICAL: Remove outer brackets - THD expects shape [total_tokens], not [1, total_tokens]
    batch[MetadataFields.INPUT_IDS] = torch.tensor(
        [token for sample in features for token in sample[MetadataFields.INPUT_IDS]], dtype=torch.long
    )

    if is_labels_provided:
        batch[MetadataFields.LABELS] = torch.tensor(
            [label for sample in features for label in sample[MetadataFields.LABELS]], dtype=torch.long
        )

    # MLM loss mask - which tokens to compute loss on
    batch[MetadataFields.INPUT_MASK] = torch.tensor(
        [mask for sample in features for mask in sample[MetadataFields.INPUT_MASK]], dtype=torch.bool
    )

    # Cumulative sequence lengths for THD/packed format
    cu_seq_lens = torch.zeros(len(features) + 1, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(torch.tensor(sample_lengths, dtype=torch.int32), dim=0)
    batch[MetadataFields.CU_SEQ_LENS_Q] = batch[MetadataFields.CU_SEQ_LENS_K] = cu_seq_lens

    return batch
