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


import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from transformers import AutoTokenizer

from src.data.metadata import MetadataConstants, MetadataFields


logger = logging.getLogger(__name__)


class MutationDataset(torch.utils.data.Dataset):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        process_item,
        id_col: str = "id",
        ref_seq_col: str = "ref_seq",
        alt_seq_col: str = "alt_seq",
        ref_codon_col: str = "ref_codon",
        alt_codon_col: str = "alt_codon",
        codon_pos_col: str = "codon_position",
        context_length: int = 2048,
        task: str = "mlm",
        seed: int = 123,
        extract_seq: bool = True,
        train_val_test_ratio: Optional[List[float]] = None,  # - if not provided run through all data
        label_col: Optional[str] = None,
    ):
        is_eval = train_val_test_ratio is None
        data = pl.read_csv(data_path, ignore_errors=True).to_pandas()
        self.tokenizer = tokenizer
        self.variant_id_col = id_col
        self.codon_pos_col = codon_pos_col
        self.ref_seq_col = ref_seq_col
        self.alt_seq_col = alt_seq_col
        self.ref_codon_col = ref_codon_col
        self.alt_codon_col = alt_codon_col
        self.context_length = context_length
        self.adjusted_context_length = (
            context_length - MetadataConstants.MLM_TOK_ADJUST
            if task == "mlm"
            else context_length - MetadataConstants.NEXT_TOKEN_PREDICTION_TOK_ADJUST
        )
        self.sequence_extractor = FastRefAlt(context_length=self.adjusted_context_length)
        self.process_item = process_item
        self.seed = seed
        self.extract_seq = extract_seq
        self.label_col = label_col

        if not extract_seq:
            self.data = self.pre_process(data)
        else:
            self.data = data

        self.train_idx, self.val_idx, self.test_idx = None, None, None
        if not is_eval:
            assert train_val_test_ratio is not None, "train_val_test_ratio must be provided for training"
            assert sum(train_val_test_ratio) == 1.0, "train_val_test_ratio must sum to 1.0"
            self.train_idx, self.val_idx, self.test_idx = self.load_train_val_test_indices(
                data_path=data_path,
                train_val_test_ratio=train_val_test_ratio,
                seed=self.seed,
            )
            self.idxs = self.train_idx
        else:
            self.idxs = np.arange(len(self.data))
            self.train_idx, self.val_idx, self.test_idx = self.idxs, self.idxs, self.idxs

    def __len__(self) -> int:  # noqa: D105
        return len(self.idxs)

    def pre_process(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        mask = data[self.codon_pos_col] < self.adjusted_context_length
        data = data[mask]
        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # noqa: D105
        idx = self.idxs[idx]
        row = self.data.iloc[idx]
        if self.extract_seq:
            ref_seq, codon_pos, ref_codon, alt_codon = self.sequence_extractor(
                row[self.ref_seq_col], row[self.codon_pos_col], row[self.ref_codon_col], row[self.alt_codon_col]
            )
        else:
            ref_seq = row[self.ref_seq_col]
            codon_pos = row[self.codon_pos_col]
            ref_codon = row[self.ref_codon_col]
            alt_codon = row[self.alt_codon_col]

        item = self.process_item(
            ref_seq, codon_pos, ref_codon, alt_codon, context_length=self.context_length, tokenizer=self.tokenizer
        )
        item[MetadataFields.ID] = row[self.variant_id_col]
        if self.label_col is not None:
            item[MetadataFields.LABELS] = row[self.label_col]
        return item

    def load_train_val_test_indices(  # noqa: D102
        self,
        data_path: str,
        train_val_test_ratio: List[float],
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_path = Path(data_path)
        dir_path = data_path if data_path.is_dir() else data_path.parent
        post_fix = ""
        train_idx_path = dir_path / f"train_idx{post_fix}.npy"
        val_idx_path = dir_path / f"val_idx{post_fix}.npy"
        test_idx_path = dir_path / f"test_idx{post_fix}.npy"
        if train_idx_path.exists() and val_idx_path.exists() and test_idx_path.exists():
            train_idx = np.load(train_idx_path)
            val_idx = np.load(val_idx_path)
            test_idx = np.load(test_idx_path)
        else:
            num_samples = len(self.data)
            if len(train_val_test_ratio) != 3:
                raise ValueError("train_val_test_ratio must have 3 values")
                # normalize to sum to 1
            train_val_test_ratio = np.array(train_val_test_ratio)
            if train_val_test_ratio.sum() <= 0:
                raise ValueError("train_val_test_ratio must sum to a positive number")
            train_val_test_ratio = train_val_test_ratio / train_val_test_ratio.sum()
            logger.info(f"train_val_test_ratio: {train_val_test_ratio}")
            rng = np.random.RandomState(seed)
            idx = rng.permutation(num_samples)
            train_size = int(train_val_test_ratio[0] * num_samples)
            val_size = int(train_val_test_ratio[1] * num_samples)

            train_idx = idx[:train_size]
            val_idx = idx[train_size : train_size + val_size]
            test_idx = idx[train_size + val_size :]

            np.save(train_idx_path, train_idx)
            np.save(val_idx_path, val_idx)
            np.save(test_idx_path, test_idx)

        return train_idx, val_idx, test_idx

    def copy(self):  # noqa: D102
        return copy.copy(self)

    def get_num_samples(self, split):  # noqa: D102
        if split == "train":
            return self.get_train_num_samples()
        elif split == "valid":
            return self.get_val_num_samples()
        elif split == "test":
            return self.get_test_num_samples()
        else:
            raise ValueError(f"Invalid split: {split}")

    def get_train_num_samples(self):  # noqa: D102
        l = len(self.data) if self.train_idx is None else len(self.train_idx)  # noqa: E741
        return l

    def get_val_num_samples(self):  # noqa: D102
        l = len(self.data) if self.val_idx is None else len(self.val_idx)  # noqa: E741
        return l

    def get_test_num_samples(self):  # noqa: D102
        l = len(self.data) if self.test_idx is None else len(self.test_idx)  # noqa: E741
        return l

    def get_train(self, process_item):
        """Modifies indices to correspond to `train` split"""  # noqa: D415
        assert self.train_idx is not None, "train_idx is not loaded"
        self.process_item = process_item
        self.idxs = self.train_idx
        return self

    def get_validation(self, process_item):
        """Modifies indices to correspond to `valid` split"""  # noqa: D415
        assert self.val_idx is not None, "val_idx is not loaded"
        copy = self.copy()
        copy.process_item = process_item
        copy.idxs = self.val_idx
        return copy

    def get_test(self, process_item):
        """Modifies indices to correspond to `test` split"""  # noqa: D415
        assert self.test_idx is not None, "test_idx is not loaded"
        copy = self.copy()
        copy.process_item = process_item
        copy.idxs = self.test_idx
        return copy


def collate_fn(batch):  # noqa: D103
    collated_batch = {}
    for key in batch[0].keys():
        if key == MetadataFields.ID:
            collated_batch[key] = [item[key] for item in batch]
        else:
            collated_batch[key] = torch.stack([torch.tensor(item[key]) for item in batch])

    return collated_batch


class FastRefAlt:
    """Returns subsequence described by location (coding sequence start position, mutation relative distance)."""

    def __init__(self, context_length=None, nan_codon="NA"):  # noqa: D417
        """Initialize the FastRefAlt.

        Args:
            context_length (int, optional): The length of the output sequences. Defaults to None.
        """
        self.context_length = context_length
        self.nan_codon = nan_codon

    def __call__(self, ref_cds, codon_position, ref_codon, alt_codon) -> Tuple[str, int, str, str]:
        """Get the subsequence described by the location, centered around the variant codon.

        Args:
            ref_cds (str): The reference coding sequence.
            codon_position (int): The position of the mutated codon.
            ref_codon (str): The reference codon.
            alt_codon (str): The alternate codon.

        Returns:
            Tuple[str, str, int, str, str]: The DNA sequence context for reference and altered samples, the position of the mutated codon, and the reference and alternate codons.
        """
        total_codons = len(ref_cds) // MetadataConstants.CODON_LENGTH
        context_len_codons = self.context_length

        if total_codons <= context_len_codons:
            start_codon = 0
            end_codon = total_codons
        else:
            half_context = context_len_codons // 2
            start_codon = codon_position - half_context
            end_codon = start_codon + context_len_codons

            if start_codon < 0:
                start_codon = 0
                end_codon = context_len_codons
            elif end_codon > total_codons:
                end_codon = total_codons
                start_codon = total_codons - context_len_codons

        context_start = start_codon * MetadataConstants.CODON_LENGTH
        context_end = end_codon * MetadataConstants.CODON_LENGTH

        ref_seq = ref_cds[context_start:context_end]
        rel_codon_position = codon_position - start_codon
        ref_codon_start = codon_position * MetadataConstants.CODON_LENGTH
        alt_codon_start = codon_position * MetadataConstants.CODON_LENGTH  # noqa: F841

        ref_codon_new = ref_cds[ref_codon_start : ref_codon_start + MetadataConstants.CODON_LENGTH]
        assert ref_codon_new == ref_codon, f"ref_codon_new: {ref_codon_new} != ref_codon: {ref_codon}"

        if len(ref_codon) != MetadataConstants.CODON_LENGTH or len(alt_codon) != MetadataConstants.CODON_LENGTH:
            ref_codon = self.nan_codon
            alt_codon = self.nan_codon

        return ref_seq, rel_codon_position, ref_codon, alt_codon
