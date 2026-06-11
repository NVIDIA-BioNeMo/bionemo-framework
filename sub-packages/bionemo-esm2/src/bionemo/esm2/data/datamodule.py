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
import os
from typing import Literal

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging

from bionemo.esm2.data import dataset, tokenizer
from bionemo.llm.data import collate
from bionemo.llm.data.datamodule import MegatronDataModule
from bionemo.llm.utils.datamodule_utils import infer_num_samples


Mode = Literal["train", "validation", "test"]


class ESMDataModule(MegatronDataModule):
    """LightningDataModule wrapper of `ESMDataset`."""

    def __init__(
        self,
        train_cluster_path: str | os.PathLike,
        train_database_path: str | os.PathLike,
        valid_cluster_path: str | os.PathLike,
        valid_database_path: str | os.PathLike,
        seed: int | None = 42,
        min_seq_length: int | None = None,
        max_seq_length: int = 1024,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        num_workers: int = 10,  # TODO(@jomitchell) can this be automatically set?
        persistent_workers: bool = True,
        pin_memory: bool = True,
        rampup_batch_size: list[int] | None = None,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        mask_random_prob: float = 0.1,
        random_mask_strategy: dataset.RandomMaskStrategy = dataset.RandomMaskStrategy.ALL_TOKENS,
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
        dataloader_type: Literal["single", "cyclic"] = "single",
    ) -> None:
        """Initialize the ESMDataModule.

        Args:
            train_cluster_path: A path to the parquet files containing UniRef90 training clusters.
            train_database_path: A path to the sqlite file mapping UniRef90 cluster IDs to sequences.
            valid_cluster_path: A path to the parquet files containing UniRef50 validation clusters.
            valid_database_path: A path to the sqlite file mapping UniRef50 cluster IDs to sequences.
            seed: Input random seed. If None, initializes randomly. Defaults to 42.
            min_seq_length: Whether to pad sequences to a minimum length. If None, sequences are padded to the maximum
                sequence length. Defaults to None.
            max_seq_length: The maximum context length for the ESM transformer. Defaults to 1024.
            micro_batch_size: Passed to MegatronDataSampler. Defaults to 4.
            global_batch_size: Passed to MegatronDataSampler.. Defaults to 8.
            num_workers: The number of workers for the pytorch Dataloaders. Defaults to 10.
            persistent_workers: Whether to keep the workers alive between epochs. Defaults to True.
            pin_memory: Whether to pin GPU memory in the pytorch Dataloaders. Defaults to True.
            rampup_batch_size: Passed to MegatronDataSampler. Defaults to None.
            mask_prob: The overall chance of masking a token and having it appear in the loss fn. Defaults to 0.15.
            mask_token_prob: Percentage of masked tokens that get assigned the <MASK> id. Defaults to 0.8.
            mask_random_prob: Percentage of masked tokens assigned to a random amino acid. Defaults to 0.1.
            random_mask_strategy: Whether to replace random masked tokens with all tokens or amino acids only. Defaults to RandomMaskStrategy.ALL_TOKENS.
            tokenizer: The ESM2 tokenizer. Defaults to the one returned by `tokenizer.get_tokenizer()`.
            dataloader_type: The type of dataloader to use. Defaults to "single".
        """
        super().__init__()
        self._train_cluster_path = train_cluster_path
        self._train_database_path = train_database_path
        self._valid_cluster_path = valid_cluster_path
        self._valid_database_path = valid_database_path
        self._seed = seed
        self._min_seq_length = min_seq_length if min_seq_length is not None else max_seq_length
        self._max_seq_length = max_seq_length
        # seq_length needs to be implemented for OneLogger to use
        self.seq_length = max_seq_length
        self._mask_prob = mask_prob
        self._mask_token_prob = mask_token_prob
        self._mask_random_prob = mask_random_prob
        self._random_mask_strategy = random_mask_strategy
        self._tokenizer = tokenizer

        self._micro_batch_size = micro_batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory

        self.data_sampler = MegatronDataSampler(
            seq_len=max_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type=dataloader_type,  # `MegatronPretrainingRandomSampler` from "cyclic" is failing.
            rampup_batch_size=rampup_batch_size,
        )

    @property
    def tokenizer(self) -> tokenizer.BioNeMoESMTokenizer:
        """Returns the tokenizer."""
        return self._tokenizer

    def setup(self, stage: str = "") -> None:
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

        max_train_steps = self.trainer.max_steps
        if max_train_steps <= 0:
            raise RuntimeError("Please specify trainer.max_steps")

        # Create training dataset
        num_train_samples = int(
            max_train_steps * self.data_sampler.global_batch_size
        )  # training data requires upsampling (multiply by max_train_steps) on single MegatronPretrainingRandomSampler
        self._train_ds = dataset.create_train_dataset(
            cluster_file=self._train_cluster_path,
            db_path=self._train_database_path,
            total_samples=num_train_samples,
            seed=self._seed,
            max_seq_length=self._max_seq_length,
            mask_prob=self._mask_prob,
            mask_token_prob=self._mask_token_prob,
            mask_random_prob=self._mask_random_prob,
            random_mask_strategy=self._random_mask_strategy,
            tokenizer=self._tokenizer,
        )

        # Create validation dataset
        val_clusters = dataset.create_valid_clusters(self._valid_cluster_path)
        if self.trainer.limit_val_batches == 0:  # disable validation
            logging.info("Skip creating validation dataset because trainer.limit_val_batches=0.")
        else:
            num_val_samples = infer_num_samples(
                limit_batches=self.trainer.limit_val_batches,
                num_samples_in_dataset=len(val_clusters),
                global_batch_size=self.data_sampler.global_batch_size,
                stage="val",
            )
            self._valid_ds = dataset.create_valid_dataset(
                clusters=self._valid_cluster_path,
                db_path=self._valid_database_path,
                total_samples=num_val_samples,
                seed=self._seed,
                max_seq_length=self._max_seq_length,
                mask_prob=self._mask_prob,
                mask_token_prob=self._mask_token_prob,
                mask_random_prob=self._mask_random_prob,
                random_mask_strategy=self._random_mask_strategy,
                tokenizer=self._tokenizer,
            )

        assert hasattr(self, "trainer") and self.trainer is not None, (
            "Setup should be completed when trainer and config are attached."
        )

    def _create_dataloader(self, dataset, mode: Mode, **kwargs) -> WrappedDataLoader:
        """Create dataloader for train, validation, and test stages.

        Args:
            dataset: The dataset to create the dataloader for.
            mode: Stage of training, which is used to determined if consumed_samples in MegatronPretrainingSampler should be initialized to 0 (validation/test), or be set to the previous value from state_dict in case of checkpoint resumption (train).
            **kwargs: Additional arguments to pass to the dataloader.
        """
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
        return self._create_dataloader(self._train_ds, mode="train")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Returns the dataloader for validation data."""
        return self._create_dataloader(self._valid_ds, mode="validation")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Raises a not implemented error."""
        raise NotImplementedError("No test dataset provided for ESM2")
