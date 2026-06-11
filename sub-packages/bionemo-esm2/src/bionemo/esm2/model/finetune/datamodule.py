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
from typing import Literal, Union

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging

from bionemo.core.data.multi_epoch_dataset import IdentityMultiEpochDatasetWrapper, MultiEpochDatasetResampler
from bionemo.esm2.data import tokenizer
from bionemo.esm2.model.finetune.dataset import (
    InMemoryPerTokenValueDataset,
    InMemoryProteinDataset,
    InMemorySingleValueDataset,
)
from bionemo.llm.data import collate
from bionemo.llm.data.datamodule import MegatronDataModule
from bionemo.llm.utils.datamodule_utils import infer_num_samples


Mode = Literal["train", "validation", "test", "predict"]
DATASET_TYPES = Union[InMemoryPerTokenValueDataset, InMemorySingleValueDataset, InMemoryProteinDataset, None]


class ESM2FineTuneDataModule(MegatronDataModule):
    """A PyTorch Lightning DataModule for fine-tuning ESM2 models.

    This DataModule is designed to handle the data preparation and loading for fine-tuning ESM2 models.
    It provides a flexible way to create and manage datasets, data loaders, and sampling strategies.
    """

    def __init__(
        self,
        train_dataset: DATASET_TYPES = None,
        valid_dataset: DATASET_TYPES = None,
        predict_dataset: DATASET_TYPES = None,
        seed: int = 42,
        min_seq_length: int | None = None,
        max_seq_length: int = 1024,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        num_workers: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        rampup_batch_size: list[int] | None = None,
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
    ) -> None:
        """Initialize the ESM2FineTuneDataModule.

        Args:
            train_dataset: The training dataset.
            valid_dataset: The validation dataset.
            predict_dataset: The prediction dataset. Should not be set together with train/valid datasets
            seed: The random seed to use for shuffling the datasets. Defaults to 42.
            min_seq_length: The minimum sequence length for the datasets. Defaults to None.
            max_seq_length: The maximum sequence length for the datasets. Defaults to 1024.
            micro_batch_size: The micro-batch size for the data loader. Defaults to 4.
            global_batch_size: The global batch size for the data loader. Defaults to 8.
            num_workers: The number of worker processes for the data loader. Defaults to 10.
            persistent_workers: Whether to persist the worker processes. Defaults to True.
            pin_memory: Whether to pin the data in memory. Defaults to True.
            rampup_batch_size: The batch size ramp-up schedule. Defaults to None.
            tokenizer: The tokenizer to use for tokenization. Defaults to the BioNeMoESMTokenizer.

        Returns:
            None
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.predict_dataset = predict_dataset
        if predict_dataset is not None:
            assert train_dataset is None, "Datamodule expects either trin/valid dataset or predict dataset"
        self._seed = seed
        self._min_seq_length = min_seq_length
        self._max_seq_length = max_seq_length
        # seq_length needs to be implemented for OneLogger to use
        self.seq_length = max_seq_length
        self._tokenizer = tokenizer

        self._micro_batch_size = micro_batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory

        self.data_sampler = MegatronDataSampler(
            seq_len=max_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type="single",  # `MegatronPretrainingRandomSampler` from "cyclic" is failing.
            rampup_batch_size=rampup_batch_size,
            output_log=predict_dataset is None,  # logging does not work with predict step
        )

    def setup(self, stage: str) -> None:
        """Setup the ESMDataModule.

        Args:
            stage: Unused.

        Raises:
            RuntimeError: If the trainer is not attached, or if the trainer's max_steps is not set.
        """
        del stage  # Unused.

        if not hasattr(self, "trainer") or self.trainer is None:
            raise RuntimeError("Setup should be completed when trainer and config are attached.")

        if self.trainer.max_epochs is not None and self.trainer.max_epochs > 1:
            logging.warning(
                "Trainer is set to run for multiple epochs. This is not recommended due to the same shuffle being used "
                "in each. Instead set max_epochs to 1 and increase the number of max_steps."
            )

        # Create training dataset
        if self.train_dataset is not None:
            # If classification task, ensure consistent label vocabulary across splits
            if hasattr(self.train_dataset, "label_tokenizer"):
                self.valid_dataset.label_tokenizer = self.train_dataset.label_tokenizer

            max_train_steps = self.trainer.max_steps
            if max_train_steps <= 0:
                raise RuntimeError("Please specify trainer.max_steps")

            num_train_samples = int(max_train_steps * self.data_sampler.global_batch_size)
            self._train_ds = self._create_epoch_based_dataset(self.train_dataset, num_train_samples)

        # Create validation dataset
        if self.valid_dataset is not None and self.trainer.limit_val_batches != 0:
            num_val_samples = infer_num_samples(
                limit_batches=self.trainer.limit_val_batches,
                num_samples_in_dataset=len(self.valid_dataset),
                global_batch_size=self.data_sampler.global_batch_size,
                stage="val",
            )
            self._valid_ds = self._create_epoch_based_dataset(self.valid_dataset, num_val_samples)

        assert hasattr(self, "trainer") and self.trainer is not None, (
            "Setup should be completed when trainer and config are attached."
        )

    def _create_epoch_based_dataset(
        self,
        dataset: InMemoryPerTokenValueDataset | InMemorySingleValueDataset,
        total_samples: int,
    ):
        return MultiEpochDatasetResampler(
            IdentityMultiEpochDatasetWrapper(dataset),
            num_samples=total_samples,
            shuffle=self.predict_dataset is None,
            seed=self._seed,
        )

    def _create_dataloader(self, dataset, mode: Mode, **kwargs) -> WrappedDataLoader:
        """Create dataloader for train, validation, and test stages.

        Args:
            dataset: The dataset to create the dataloader for.
            mode: Stage of training, which is used to determined if consumed_samples in MegatronPretrainingSampler should be initialized to 0 (validation/test), or be set to the previous value from state_dict in case of checkpoint resumption (train).
            **kwargs: Additional arguments to pass to the dataloader.
        """
        if mode not in ["predict", "test"]:
            self.update_init_global_step()
        assert self._tokenizer.pad_token_id is not None, "Tokenizer must have a pad token id."

        return WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=self._tokenizer.pad_token_id,
                min_length=self._min_seq_length,
                max_length=self._max_seq_length,
            ),
            **kwargs,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Returns the dataloader for training data."""
        assert self._train_ds is not None, "train_dataset is not provided to ESM2FineTuneDataModule"
        return self._create_dataloader(self._train_ds, mode="train")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Returns the dataloader for validation data."""
        assert self._valid_ds is not None, "valid_dataset is not provided to ESM2FineTuneDataModule"
        return self._create_dataloader(self._valid_ds, mode="validation")

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Returns the dataloader for prediction data."""
        assert self.predict_dataset is not None, "predict_dataset is not provided to ESM2FineTuneDataModule"
        return self._create_dataloader(self.predict_dataset, mode="predict")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Raises a not implemented error."""
        raise NotImplementedError("No test dataset provided for ESM2")
