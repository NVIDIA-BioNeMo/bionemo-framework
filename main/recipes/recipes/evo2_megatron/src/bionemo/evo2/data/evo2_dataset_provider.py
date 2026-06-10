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
from typing import Any, Type

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.tokenizers.megatron_tokenizer import MegatronTokenizerBase

from bionemo.evo2.data.megatron.hyena.config import parse_dataset_config
from bionemo.evo2.data.megatron.hyena.evo2_dataset import Evo2Dataset


@dataclass
class Evo2DatasetProvider(DatasetProvider):
    """Dataset provider for Evo2. This is mostly just the megatron gpt dataset, but with a custom dataset class."""

    random_seed: int = 42
    seq_length: int = 8192
    dataset_config_path: Path | None = None  # REQUIRED!
    dataset_dir: str | None = None
    index_mapping_dir: str | None = None
    dataset_path: Path | None = None
    mmap_bin_files: bool = True
    object_storage_cache_path: str | None = None
    num_dataset_builder_threads: int = 1
    reset_position_ids: bool | None = False
    reset_attention_mask: bool | None = False
    create_attention_mask: bool = False
    eod_mask_loss: bool | None = False
    dataloader_type: str = "single"  # critical
    dataset_cls: Type[MegatronDataset] = Evo2Dataset

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
        """Build and return the train, validation, and test datasets given the context for this training run."""
        num_train_samples = context.train_samples
        num_val_samples = context.valid_samples
        num_test_samples = context.test_samples

        assert context.tokenizer is not None
        dataset_config = self.get_gpt_dataset_config(context.tokenizer)

        train_valid_test_num_samples = [num_train_samples, num_val_samples, num_test_samples]
        train_ds, val_ds, test_ds = BlendedMegatronDatasetBuilder(
            self.dataset_cls,
            train_valid_test_num_samples,
            is_built_on_rank=lambda: True,
            config=dataset_config,
        ).build()

        return train_ds, val_ds, test_ds

    def get_gpt_dataset_config(self, tokenizer: MegatronTokenizerBase) -> "GPTDatasetConfig":
        """Get the GPT dataset configuration."""
        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

        assert self.dataset_config_path is not None
        paths = parse_dataset_config(
            dataset_config_path=str(self.dataset_config_path),
            dataset_path=str(self.dataset_path) if self.dataset_path is not None else None,
        )
        build_kwargs = {}
        build_kwargs["mmap_bin_files"] = self.mmap_bin_files

        build_kwargs["blend_per_split"] = [
            get_blend_from_list(paths["train"]),
            get_blend_from_list(paths["validation"]),
            get_blend_from_list(paths["test"]),
        ]

        if self.object_storage_cache_path:
            build_kwargs["object_storage_cache_path"] = self.object_storage_cache_path
            build_kwargs["mmap_bin_files"] = False

        return GPTDatasetConfig(
            random_seed=self.random_seed,
            sequence_length=self.seq_length,
            tokenizer=tokenizer,
            path_to_cache=self.index_mapping_dir,
            reset_position_ids=self.reset_position_ids,
            create_attention_mask=self.create_attention_mask,
            reset_attention_mask=self.reset_attention_mask,
            eod_mask_loss=self.eod_mask_loss,
            num_dataset_builder_threads=self.num_dataset_builder_threads,
            **build_kwargs,
        )
