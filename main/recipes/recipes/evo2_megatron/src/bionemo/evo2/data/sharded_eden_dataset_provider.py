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

"""Megatron-bridge adapter for the BCR ShardedEdenDataset.

The core dataset class and window pre-computation live in ``bionemo-recipeutils``
(``bionemo.recipeutils.data.basecamp``).  This module provides the thin
``DatasetProvider`` wrapper that plugs the dataset into Megatron-bridge's
training loop.

Contributed by BaseCamp Research: https://basecamp-research.com/
"""

from dataclasses import dataclass
from typing import Any, Optional

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from megatron.core.tokenizers.megatron_tokenizer import MegatronTokenizerBase

from bionemo.core.data.multi_epoch_dataset import (
    IdentityMultiEpochDatasetWrapper,
    MultiEpochDatasetResampler,
)
from bionemo.recipeutils.data.basecamp.sharded_dataset import (
    ShardedEdenDataset,
    compute_num_windows,
    precompute_window_database,
)
from bionemo.recipeutils.data.basecamp.utils import extract_sample_id


__all__ = [
    "DatasetBuildContext",
    "ShardedEdenDataset",
    "ShardedEdenDatasetProvider",
    "compute_num_windows",
    "extract_sample_id",
    "precompute_window_database",
]


@dataclass
class ShardedEdenDatasetProvider(DatasetProvider):
    """Dataset provider for ShardedEdenDataset."""

    sequence_db_dir: str | None = None
    train_window_db_path: str | None = None
    val_window_db_path: str | None = None
    test_window_db_path: str | None = None
    rc_aug: bool = False
    random_seed: int = 42
    stride: Optional[int] = 7992
    seq_length: int = 8192
    window_min_length_threshold: Optional[int] = None
    use_control_tags: bool = False
    log_windows: bool = False
    log_dir: Optional[str] = None
    skip_stats: bool = True
    create_attention_mask: bool = False
    skip_getting_attention_mask_from_dataset: bool = True
    dataloader_type: str = "single"

    def _create_epoch_wrapped_sharded_eden_dataset(
        self,
        *,
        window_db_path: str,
        num_samples: int,
        tokenizer: MegatronTokenizerBase,
        split: str,
        shuffle: bool,
    ) -> MultiEpochDatasetResampler:
        """Instantiate ``ShardedEdenDataset`` and wrap it with ``MultiEpochDatasetResampler``."""
        assert self.sequence_db_dir is not None
        assert window_db_path is not None
        base_dataset = ShardedEdenDataset(
            tokenizer=tokenizer,
            sequence_db_dir=self.sequence_db_dir,
            window_db_path=window_db_path,
            seq_length=self.seq_length,
            create_attention_mask=self.create_attention_mask,
            stride=self.stride,
            window_min_length_threshold=self.window_min_length_threshold,
            rc_aug=self.rc_aug,
            use_control_tags=self.use_control_tags,
            split=split,
            log_windows=self.log_windows,
            log_dir=self.log_dir,
        )

        wrapped = MultiEpochDatasetResampler(
            IdentityMultiEpochDatasetWrapper(base_dataset),
            num_samples=num_samples,
            shuffle=shuffle,
            seed=self.random_seed,
        )
        return wrapped

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
        """Build and return the train, validation, and test datasets given the context for this training run."""
        assert context.tokenizer is not None
        assert self.train_window_db_path is not None
        assert self.val_window_db_path is not None
        assert self.test_window_db_path is not None
        train_ds = self._create_epoch_wrapped_sharded_eden_dataset(
            window_db_path=self.train_window_db_path,
            num_samples=context.train_samples,
            tokenizer=context.tokenizer,
            split="train",
            shuffle=True,
        )
        val_ds = self._create_epoch_wrapped_sharded_eden_dataset(
            window_db_path=self.val_window_db_path,
            num_samples=context.valid_samples,
            tokenizer=context.tokenizer,
            split="validation",
            shuffle=False,
        )
        test_ds = self._create_epoch_wrapped_sharded_eden_dataset(
            window_db_path=self.test_window_db_path,
            num_samples=context.test_samples,
            tokenizer=context.tokenizer,
            split="test",
            shuffle=False,
        )
        return train_ds, val_ds, test_ds
