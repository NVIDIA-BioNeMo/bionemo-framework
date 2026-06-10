# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np
import torch
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.tokenizers.megatron_tokenizer import MegatronTokenizerBase
from torch.utils.data import Dataset

from bionemo.evo2.data.megatron.hyena.evo2_dataset import Evo2Dataset


class MockGPTDataset(Dataset):
    """Mock dataset that respects the tokenizer's vocab size and user requested sequence length."""

    def __init__(
        self,
        tokenizer: MegatronTokenizerBase,
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
        create_attention_mask: bool = False,
        overfit_mode: bool = True,
    ) -> None:
        """Initialize the mock dataset."""
        super().__init__()
        self.name = name
        self.seq_length = seq_length
        self.vocab_size = tokenizer.vocab_size
        self.length = num_samples
        self.seed = seed
        self.create_attention_mask = create_attention_mask
        self.overfit_mode = overfit_mode
        if create_attention_mask:
            self.attention_mask = torch.tril(torch.ones((self.seq_length, self.seq_length), device="cpu")).unsqueeze(0)
            self.attention_mask = self.attention_mask < 0.5

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

    def __len__(self) -> int:
        """Get the length of the mock dataset."""
        return self.length

    def _get_text(self, idx: int) -> np.ndarray:
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        return np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get a single item from the mock dataset."""
        # Generate data of the expected size and datatype (based on GPTDataset).
        if self.overfit_mode:
            # Somewhat interesting deterministic function for a model to learn.
            tokens = (
                torch.arange(self.vocab_size, dtype=torch.int64)
                .repeat((self.seq_length + 1) // self.vocab_size + 1)[: self.seq_length + 1]
                .roll(idx)
            )
        else:
            np_gen = np.random.default_rng(seed=(self.seed + idx))
            tokens = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length + 1], dtype=np.int64))

        batch = {
            "tokens": tokens[:-1],
            "labels": tokens[1:],
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
        }

        if self.create_attention_mask:
            batch["attention_mask"] = self.attention_mask

        return batch

    def _collate_fn(self, batch):
        """A default implementation of a collation function.

        Users should override this method to define custom data loaders.
        """
        return torch.utils.data.default_collate(batch)

    def collate_fn(self, batch):
        """Method that user pass as functor to DataLoader.

        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        # Usage:
        dataloader = torch.utils.data.DataLoader(
                ....,
                collate_fn=dataset.collate_fn,
                ....
        )

        Returns:
        -------
            Collated batch, with or without types.
        """
        return self._collate_fn(batch)


@dataclass
class MockEvo2DatasetProvider(DatasetProvider):
    """Dataset provider for Evo2. This is mostly just the megatron gpt dataset, but with a custom dataset class."""

    random_seed: int = 42
    seq_length: int = 8192
    dataset_config_path: Path | None = None  # REQUIRED!
    index_mapping_dir: str | None = None
    dataset_path: Path | None = None
    mmap_bin_files: bool = True
    object_storage_cache_path: str | None = None
    num_dataset_builder_threads: int = 1
    reset_position_ids: bool | None = False
    create_attention_mask: bool = False
    skip_getting_attention_mask_from_dataset: bool = True
    reset_attention_mask: bool | None = False
    eod_mask_loss: bool | None = True
    dataloader_type: str = "single"  # critical
    overfit_mode: bool = False  # if True, return the same batch all the time.
    dataset_cls: Type[MegatronDataset] = Evo2Dataset

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
        """Build and return the train, validation, and test datasets given the context for this training run."""
        num_train_samples = context.train_samples
        num_val_samples = context.valid_samples
        num_test_samples = context.test_samples

        train_ds = MockGPTDataset(
            tokenizer=context.tokenizer,
            name="mock_train",
            num_samples=num_train_samples,
            seq_length=self.seq_length,
            seed=self.random_seed,
            create_attention_mask=self.create_attention_mask,
        )
        val_ds = MockGPTDataset(
            tokenizer=context.tokenizer,
            name="mock_val",
            num_samples=num_val_samples,
            seq_length=self.seq_length,
            seed=self.random_seed + 1,
            create_attention_mask=self.create_attention_mask,
        )
        test_ds = MockGPTDataset(
            tokenizer=context.tokenizer,
            name="mock_test",
            num_samples=num_test_samples,
            seq_length=self.seq_length,
            seed=self.random_seed + 2,
            create_attention_mask=self.create_attention_mask,
        )

        return train_ds, val_ds, test_ds
