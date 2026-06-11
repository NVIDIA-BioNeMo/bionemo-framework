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
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import numpy as np
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tokenizers import Tokenizer

from bionemo.core.data.multi_epoch_dataset import MultiEpochDatasetResampler
from bionemo.core.utils import random_utils
from bionemo.geneformer.data.singlecell.dataset import SingleCellDataset
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.data import collate
from bionemo.llm.data.datamodule import MegatronDataModule
from bionemo.llm.utils.datamodule_utils import infer_num_samples


Mode = Literal["train", "validation", "test", "predict"]

__all__: Sequence[str] = ("SingleCellDataModule",)


class SingleCellDataModule(MegatronDataModule):
    """LightningDataModule wrapper of `SingleCellDataset`

    Args:
        data_path (Union[str, PosixPath]): Path to preprocessed single-cell data files
        tokenizer (Tokenizer): Maps gene names to ids and vice-versa
        collator: Used to batch samples
        process_item: Function defining how each item should be processed
        num_workers (int): Number of workers to use
        num_mask_per_sample (int): Number of masked versions of a single sample to be returned by each worker
        train_batch_size (int): Batch size for training
        val_batch_size (int): Batch size for validation
        include_unrecognized_vocab_in_dataset (bool, optional): If set to True, a hard-check is performed to verify all gene identifers are in the user supplied tokenizer vocab. Defaults to False which means any gene identifier not in the user supplied tokenizer vocab will be excluded.

    Attributes:
        cfg (Config): Configuration object
        data_path (Union[str, PosixPath]): Path to preprocessed single-cell data files
        median_dict (dict): Dictionary containing median values
        tokenizer (Tokenizer): Tokenizer object
        setup_called (bool): Flag indicating if the setup method has been called
        dataset (SingleCellDataset): Single-cell dataset object

    """  # noqa: D415

    # Nothing says we cant pass in the dataset...
    def __init__(  # noqa: D107
        self,
        tokenizer: Tokenizer,
        median_dict: dict[str, float],
        train_dataset_path: str | Path | None = None,
        val_dataset_path: str | Path | None = None,
        test_dataset_path: str | Path | None = None,
        predict_dataset_path: str | Path | None = None,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,  # 80% mask token
        random_token_prob: float = 0.1,  # 10% random token, remaining 1-(mask+random) will be identity.
        seq_length: int = 2048,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        seed: int = 42,
        num_workers: int = 10,  # TODO can this be automatically set?
        persistent_workers: bool = True,
        pin_memory: bool = True,
        include_unrecognized_vocab_in_dataset: bool = False,
    ) -> None:
        super().__init__()
        if predict_dataset_path is None:
            assert train_dataset_path is not None and val_dataset_path is not None and test_dataset_path is not None, (
                "Provide either predict_dataset_path or (train_dataset_path, val_dataset_path, and test_dataset_path)"
            )
        elif train_dataset_path is None:
            assert val_dataset_path is None and test_dataset_path is None, (
                "Provide either predict_dataset_path or (train_dataset_path, val_dataset_path, and test_dataset_path)"
            )
            assert predict_dataset_path is not None, (
                "Provide either predict_dataset_path or (train_dataset_path, val_dataset_path, and test_dataset_path)"
            )
        self.data_path_predict = predict_dataset_path
        self.data_path_train = train_dataset_path
        self.data_path_val = val_dataset_path
        self.data_path_test = test_dataset_path
        self.tokenizer = tokenizer
        self.median_dict = median_dict
        self.max_len = seq_length
        # We need to define a "seq_length" for OneLogger, but we just set it to max_len
        self.seq_length = seq_length
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.seed = seed
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        rng = np.random.default_rng(seed)
        if self.data_path_train is not None:
            assert self.data_path_val is not None and self.data_path_test is not None
            self._train_dataset_ori = SingleCellDataset(
                self.data_path_train,
                self.tokenizer,
                self.median_dict,
                self.max_len,
                mask_prob=self.mask_prob,
                mask_token_prob=self.mask_token_prob,
                random_token_prob=self.random_token_prob,
                seed=random_utils.get_seed_from_rng(rng),
                include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
            )
            self._val_dataset_ori = SingleCellDataset(
                self.data_path_val,
                self.tokenizer,
                self.median_dict,
                self.max_len,
                mask_prob=self.mask_prob,
                mask_token_prob=self.mask_token_prob,
                random_token_prob=self.random_token_prob,
                seed=random_utils.get_seed_from_rng(rng),
                include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
            )
            self._test_dataset_ori = SingleCellDataset(
                self.data_path_test,
                self.tokenizer,
                self.median_dict,
                self.max_len,
                mask_prob=self.mask_prob,
                mask_token_prob=self.mask_token_prob,
                random_token_prob=self.random_token_prob,
                seed=random_utils.get_seed_from_rng(rng),
                include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
            )
            self._predict_dataset_ori = None
        else:
            assert self.data_path_predict is not None
            self._predict_dataset_ori = SingleCellDataset(
                self.data_path_predict,
                self.tokenizer,
                self.median_dict,
                self.max_len,
                mask_prob=self.mask_prob,
                mask_token_prob=self.mask_token_prob,
                random_token_prob=self.random_token_prob,
                seed=random_utils.get_seed_from_rng(rng),
                include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
            )
            self._train_dataset_ori = None
            self._val_dataset_ori = None
            self._test_dataset_ori = None

        # This is needed here, or you need to specify it in the megatron adapter thing TODO name?
        #  Note that this sampler is sequential, meaning it does not do any shuffling. Let's wrap our data in a shuffler.
        if self.data_path_predict is not None:
            n_predict = len(self._predict_dataset_ori)
            self.data_sampler = MegatronDataSampler(
                seq_len=self.max_len,
                micro_batch_size=min(micro_batch_size, n_predict),
                global_batch_size=min(global_batch_size, n_predict),
                rampup_batch_size=rampup_batch_size,
                output_log=False,  # this is needed for predict step to work
            )
        else:
            self.data_sampler = MegatronDataSampler(
                seq_len=self.max_len,
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                rampup_batch_size=rampup_batch_size,
            )

    def setup(self, stage: str = "") -> None:  # noqa: D102
        assert getattr(self, "trainer", None) is not None, "Please only call setup after trainer is attached."

        if self._train_dataset_ori is not None:
            assert self._val_dataset_ori is not None and self._test_dataset_ori is not None
            # Trainer API
            max_train_steps = self.trainer.max_steps
            if self.trainer.max_epochs > 1:
                logging.warning(
                    "Trainer is set to run for multiple epochs. This is not recommended due to the same shuffle being used in each. Instead set max_epochs to 1 and increase the number of max_steps."
                )
            assert max_train_steps > 0, "Please specify trainer.max_steps"

            num_train_samples = int(max_train_steps * self.data_sampler.global_batch_size)

            # This happens exactly once during setup.
            self._train_ds = MultiEpochDatasetResampler(
                self._train_dataset_ori,
                num_samples=num_train_samples,
                shuffle=True,
                seed=self.seed,
            )
            if self.trainer.limit_val_batches == 0:  # disable validation
                logging.info("Skip creating validation dataset because trainer.limit_val_batches=0.")
            else:
                num_val_samples = infer_num_samples(
                    limit_batches=self.trainer.limit_val_batches,
                    num_samples_in_dataset=len(self._val_dataset_ori),
                    global_batch_size=self.data_sampler.global_batch_size,
                    stage="val",
                )
                self._validation_ds = MultiEpochDatasetResampler(
                    self._val_dataset_ori,
                    num_samples=num_val_samples,
                    shuffle=False,
                    seed=self.seed,
                )
            if self.trainer.limit_test_batches == 0:  # disable testing
                logging.info("Skip creating test dataset because trainer.limit_test_batches=0.")

            else:
                num_test_samples = infer_num_samples(
                    limit_batches=self.trainer.limit_test_batches,
                    num_samples_in_dataset=len(self._test_dataset_ori),
                    global_batch_size=self.data_sampler.global_batch_size,
                    stage="test",
                )
                self._test_ds = MultiEpochDatasetResampler(
                    self._test_dataset_ori,
                    num_samples=num_test_samples,
                    shuffle=False,
                    seed=self.seed,
                )
        else:
            assert self._predict_dataset_ori is not None
            self._predict_ds = MultiEpochDatasetResampler(
                self._predict_dataset_ori,
                shuffle=False,
                seed=self.seed,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._train_ds, mode="train")

    def val_dataloader(self) -> EVAL_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._validation_ds, mode="validation")

    def test_dataloader(self) -> EVAL_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._test_ds, mode="test")

    def predict_dataloader(self) -> EVAL_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._predict_ds, mode="predict", drop_last=False)

    def _create_dataloader(self, dataset, mode: Mode, **kwargs) -> WrappedDataLoader:
        """Create dataloader for train, validation, and test stages.

        Args:
            dataset: The dataset to create the dataloader for.
            mode: Stage of training, which is used to determined if consumed_samples in MegatronPretrainingSampler should be initialized to 0 (validation/test), or be set to the previous value from state_dict in case of checkpoint resumption (train).
            **kwargs: Additional arguments to pass to the dataloader.
        """
        self.update_init_global_step()
        return WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=self.tokenizer.token_to_id(GeneTokenizer.pad_token),
                min_length=self.max_len,
                max_length=self.max_len,
            ),
            **kwargs,
        )

    @property
    def vocab_size(self):  # noqa: D102
        return self.tokenizer.vocab_size
