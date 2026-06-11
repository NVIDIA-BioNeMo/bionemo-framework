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


import functools
from typing import Any, Dict, Literal

import lightning.pytorch as pl
from megatron.core.num_microbatches_calculator import update_num_microbatches
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from torch.utils.data import DataLoader, Dataset

from bionemo.llm.data import collate


class MegatronDataModule(pl.LightningDataModule):
    """A mixin that adds a `state_dict` and `load_state_dict` method for datamodule training resumption in NeMo."""

    def __init__(self, *args, **kwargs):
        """Set init_global_step to 0 for datamodule resumption."""
        super().__init__(*args, **kwargs)
        self.init_global_step = 0

    def update_init_global_step(self):
        """Please always call this when you get a new dataloader... if you forget, your resumption will not work."""
        self.init_global_step = self.trainer.global_step  # Update the init_global_step whenever we re-init training
        self.data_sampler.init_global_step = (
            self.init_global_step
        )  # Update the init_global_step whenever we re-init training

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        consumed_samples = self.data_sampler.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        return {"consumed_samples": consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule stat.

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """
        consumed_samples = state_dict["consumed_samples"]
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples

        update_num_microbatches(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        self.data_sampler.if_first_step = 1


class MockDataModule(MegatronDataModule):
    """A simple data module that just wraps input datasets with dataloaders."""

    def __init__(
        self,
        train_dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        predict_dataset: Dataset | None = None,
        pad_token_id: int = 0,
        min_seq_length: int | None = None,
        max_seq_length: int = 512,
        micro_batch_size: int = 16,
        global_batch_size: int = 16,
        num_workers: int = 4,
    ) -> None:
        """Initialize the MockDataModule."""
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.pad_token_id = pad_token_id
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.batch_size = micro_batch_size
        self.num_workers = num_workers
        self.data_sampler = MegatronDataSampler(
            seq_len=max_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type="single",
            output_log=False,
        )

    def setup(self, stage: str | None = None) -> None:  # noqa: D102
        pass

    def _make_dataloader(
        self, dataset: Dataset, mode: Literal["train", "validation", "test", "predict"]
    ) -> WrappedDataLoader:
        if mode not in ["predict", "test"]:
            self.update_init_global_step()

        return WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=self.pad_token_id,
                min_length=self.min_seq_length,
                max_length=self.max_seq_length,
            ),
        )

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        if self.train_dataset is None:
            raise ValueError("No train_dataset was provided")
        return self._make_dataloader(
            self.train_dataset,
            mode="train",
        )

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        if self.valid_dataset is None:
            raise ValueError("No valid_dataset was provided")
        return self._make_dataloader(
            self.valid_dataset,
            mode="validation",
        )

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        if self.test_dataset is None:
            raise ValueError("No test_dataset was provided")
        return self._make_dataloader(
            self.test_dataset,
            mode="test",
        )

    def predict_dataloader(self) -> DataLoader:  # noqa: D102
        if self.predict_dataset is None:
            raise ValueError("No predict_dataset was provided")
        return self._make_dataloader(
            self.predict_dataset,
            mode="predict",
        )
