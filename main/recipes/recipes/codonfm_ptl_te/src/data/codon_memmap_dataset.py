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


import json
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .utils import load_train_val_test_indices_proportional


def get_group_codon_weights(codon_weights_file, tokenizer):  # noqa: D103
    with open(codon_weights_file, "r") as f:
        group_counts = json.load(f)
    for group in group_counts:
        group_counts[group] = np.array(group_counts[group])
        non_zero_mask = np.zeros(group_counts[group].shape[0], dtype=bool)
        non_zero_mask[len(tokenizer.special_tokens) : tokenizer.vocab_size] = True
        non_zero_mask = non_zero_mask & (group_counts[group] > 0)
        group_counts[group][non_zero_mask] = (
            group_counts[group][non_zero_mask].mean() / group_counts[group][non_zero_mask]
        )
    return group_counts


def get_taxids_to_exclude(taxid_exclusion_file):  # noqa: D103
    with open(taxid_exclusion_file, "r") as f:
        exclusion_content = json.load(f)
        taxids_to_exclude = []
        for _, v in exclusion_content.items():  # noqa: PERF102
            taxids_to_exclude.extend(v)
    return set(taxids_to_exclude)


class CodonMemmapDataset(torch.utils.data.Dataset):
    """Dataset for loading the codon memmap data.

    Args:
        data_path: Path to the data directory.
        tokenizer: Tokenizer to use for tokenizing the data.
        context_length: Length of the context to use for the data.
        context_overlap: Overlap between the context and the data.

    Note: get_train, get_validation, get_test return a new shallow copy of the dataset with the split set to train, validation, or test.
    This is done to avoid modifying the original dataset instance.
    **IMPORTANT** this works because we aren't mutating the underlying data between these splits, and only the indices are modified.
    """

    def __init__(  # noqa: D107
        self,
        data_path: str,
        tokenizer: Callable,
        context_length: int = 2048,
        context_overlap: int = 0,
        pretraining_task: str = "mlm",
        train_val_test_ratio: List[float] = [0.9998, 0.0002, 0.00],
        process_item: Callable = lambda *x, **kwargs: (x, kwargs),
        min_seq_length: int = 100,
        max_seq_length: int = 150_000,
        codon_weights_file: str = None,  # noqa: RUF013
        groups_to_use: List[str] = None,  # noqa: RUF013
        taxid_exclusion_file: str = None,  # noqa: RUF013
        split_name_prefix: str = "",
        seed: int = 123,
    ):
        self.data_path = Path(data_path)
        self.metadata_path = self.data_path / "metadata.json"
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.context_overlap = context_overlap
        self.pretraining_task = pretraining_task
        self.process_item = process_item
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.groups_to_use = groups_to_use
        self.codon_weights = None
        self.taxids_to_exclude = None
        if codon_weights_file:
            self.codon_weights = get_group_codon_weights(codon_weights_file, tokenizer)
            print(f"Codon weights: {codon_weights_file}")
        if taxid_exclusion_file:
            assert split_name_prefix != "", "split_name_prefix cannot be empty if taxid_exclusion_file is provided"
            self.taxids_to_exclude = get_taxids_to_exclude(taxid_exclusion_file)
            print(f"loaded {len(self.taxids_to_exclude)} taxids to exclude")

        if self.pretraining_task == "mlm":
            self.tok_adjust = 2
        elif self.pretraining_task == "next_token_prediction":
            self.tok_adjust = 1
        else:
            raise ValueError(f"Invalid pretraining_task '{pretraining_task}'")

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.chunks_metadata = self.metadata["chunks"]

        self.sequences_mmaps = []
        self.indices_mmaps = []
        for chunk in self.chunks_metadata:
            seq_mmap_path = self.data_path / chunk["sequences"]["path"]
            idx_mmap_path = self.data_path / chunk["index"]["path"]

            seq_mmap = np.memmap(
                seq_mmap_path, dtype=chunk["sequences"]["dtype"], mode="r", shape=tuple(chunk["sequences"]["shape"])
            )
            idx_mmap = np.memmap(
                idx_mmap_path, dtype=chunk["index"]["dtype"], mode="r", shape=tuple(chunk["index"]["shape"])
            )

            self.sequences_mmaps.append(seq_mmap)
            self.indices_mmaps.append(idx_mmap)

        cache_suffix = f"_{split_name_prefix}" if split_name_prefix else ""
        cache_path = self.metadata_path.with_suffix(f".cache{cache_suffix}.npy")
        print(f"cache_path: {cache_path}")
        key_cache_path = self.metadata_path.with_suffix(f".key.cache{cache_suffix}.npy")
        seq_cluster_path = self.data_path / "allSeqClusterIdx.npy"

        if cache_path.exists():
            print("Loading cached global indices...")
            self.global_indices = np.load(cache_path, allow_pickle=True)
            if key_cache_path.exists():  # - added for backward compatability
                self.global_keys = np.load(key_cache_path, allow_pickle=True)
            else:
                self.global_keys = None
        else:
            print("Computing global indices for subsequences...")
            global_indices_list = []
            total_sequences = sum(len(idx_mmap) for idx_mmap in self.indices_mmaps)
            if seq_cluster_path.exists():
                seq_clusters = np.load(seq_cluster_path, allow_pickle=True)
                assert seq_clusters.shape[0] == total_sequences, "Mismatch in sequence clusters and total sequences."
            else:
                seq_clusters = np.arange(
                    total_sequences, dtype=int
                )  # NOTE: this is a placeholder for the case where the sequence clusters are not available

            global_seq_idx = 0
            global_keys = []
            with tqdm(total=total_sequences, desc="Processing sequences") as pbar:
                for chunk_id, idx_mmap in enumerate(self.indices_mmaps):
                    for seq_idx in range(len(idx_mmap)):
                        seq_cluster_idx = seq_clusters[global_seq_idx]
                        global_seq_idx += 1
                        seq_start, seq_end, taxid = idx_mmap[seq_idx]
                        if self.taxids_to_exclude and taxid in self.taxids_to_exclude:
                            pbar.update(1)
                            continue
                        seq_len_tokens = seq_end - seq_start
                        if seq_len_tokens < self.min_seq_length or seq_len_tokens > self.max_seq_length:
                            pbar.update(1)
                            continue
                        step_size = self.context_length - self.tok_adjust - self.context_overlap
                        num_subsequences = max(1, (seq_len_tokens - self.context_overlap) // step_size)
                        for sub_seq_idx in range(num_subsequences):
                            start_token_idx = seq_start + sub_seq_idx * step_size
                            end_token_idx = min(start_token_idx + (self.context_length - self.tok_adjust), seq_end)
                            if end_token_idx > start_token_idx:
                                global_indices_list.append([chunk_id, start_token_idx, end_token_idx])
                                global_keys.append(seq_cluster_idx)
                        pbar.update(1)

            self.global_indices = np.array(global_indices_list, dtype=int)
            self.global_keys = np.array(global_keys, dtype=int)
            np.save(cache_path, self.global_indices)
            np.save(key_cache_path, self.global_keys)
            print(f"Cached global indices saved at {cache_path}")

        self.train_indices, self.val_indices, self.test_indices = self.load_train_val_test_indices(
            metadata_path=self.metadata_path,
            train_val_test_ratio=train_val_test_ratio,
            seed=self.seed,
            split_name_prefix=split_name_prefix,
        )
        assert len(self.train_indices) + len(self.val_indices) + len(self.test_indices) == len(self.global_indices), (
            f"Mismatch in train, val, and test indices: {len(self.train_indices)} + {len(self.val_indices)} + {len(self.test_indices)} != {len(self.global_indices)}"
        )
        if self.groups_to_use:
            groups = [
                self.metadata["file_metadata"][i]["file_name"].split(".")[0]
                for i in range(len(self.metadata["file_metadata"]))
            ]
            assert all([group in groups for group in self.groups_to_use]), (  # noqa: C419
                f"Some groups in 'groups_to_use' are not present in the dataset: {self.groups_to_use} vs {groups}"
            )
            global_groups = [groups[i] for i in self.global_indices[:, 0]]
            indices_to_use = np.where(np.isin(global_groups, self.groups_to_use))[0]
            self.train_indices = np.intersect1d(self.train_indices, indices_to_use)
            self.val_indices = np.intersect1d(self.val_indices, indices_to_use)
            self.test_indices = np.intersect1d(self.test_indices, indices_to_use)

        self._validate_splits()  # validate that train and val splits are not empty and do not overlap
        self.current_split_indices = self.train_indices

    def load_train_val_test_indices(  # noqa: D102
        self,
        metadata_path: Path,
        train_val_test_ratio: List[float],
        seed: int = 42,
        split_name_prefix: str = "",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if split_name_prefix:
            split_name_prefix = "_" + split_name_prefix
        post_fix = "" if self.context_overlap == 0 else f"_overlap_{self.context_overlap}"
        post_fix = split_name_prefix + post_fix
        train_idx, val_idx, test_idx = load_train_val_test_indices_proportional(
            metadata_path=metadata_path,
            global_keys=self.global_keys,
            global_indices=self.global_indices,
            train_val_test_ratio=train_val_test_ratio,
            post_fix=post_fix,
            seed=seed,
        )
        return train_idx, val_idx, test_idx

    def __len__(self):  # noqa: D105
        return len(self.current_split_indices)

    def set_split(self, split: str):  # noqa: D102
        if split == "train":
            self.current_split_indices = self.train_indices
        elif split == "valid":
            self.current_split_indices = self.val_indices
        elif split == "test":
            self.current_split_indices = self.test_indices
        else:
            raise ValueError(f"Invalid split '{split}'")

    def _validate_splits(self):
        if len(self.train_indices) == 0 or len(self.val_indices) == 0:
            raise ValueError("Empty train or validation split")
        if len(set(self.train_indices) & set(self.val_indices)) > 0:
            raise ValueError("Train and validation splits overlap")

    def get_sequence_length(self, idx: int) -> int:
        """Return the unpadded token count for sample *idx* without loading data.

        The length accounts for CLS and SEP tokens that ``process_item`` prepends/appends.
        """
        global_idx = self.current_split_indices[idx]
        _, start_token_idx, end_token_idx = self.global_indices[global_idx]
        raw_len = end_token_idx - start_token_idx
        return min(raw_len, self.context_length - 2) + 2  # +2 for CLS / SEP

    def __getitem__(self, idx):  # noqa: D105
        global_idx = self.current_split_indices[idx]
        chunk_id, start_token_idx, end_token_idx = self.global_indices[global_idx]
        if self.codon_weights:
            group = self.metadata["file_metadata"][chunk_id]["file_name"].split(".")[0]
            codon_weights = self.codon_weights[group]
        else:
            codon_weights = None

        sequence_tokens = self.sequences_mmaps[chunk_id][start_token_idx:end_token_idx]

        return self.process_item(
            tokenizer=self.tokenizer,
            sequence_tokens=sequence_tokens,
            context_length=self.context_length,
            codon_weights=codon_weights,
        )

    def get_train(self, process_item: Callable) -> "CodonMemmapDataset":  # noqa: D102
        dataset_copy = CodonMemmapDataset.__new__(CodonMemmapDataset)
        dataset_copy.__dict__ = {**self.__dict__}
        dataset_copy.set_split("train")
        dataset_copy.process_item = process_item
        return dataset_copy

    def get_validation(self, process_item: Callable) -> "CodonMemmapDataset":  # noqa: D102
        dataset_copy = CodonMemmapDataset.__new__(CodonMemmapDataset)
        dataset_copy.__dict__ = {**self.__dict__}
        dataset_copy.set_split("valid")
        dataset_copy.process_item = process_item
        return dataset_copy

    def get_test(self, process_item: Callable) -> "CodonMemmapDataset":  # noqa: D102
        dataset_copy = CodonMemmapDataset.__new__(CodonMemmapDataset)
        dataset_copy.__dict__ = {**self.__dict__}
        dataset_copy.set_split("test")
        dataset_copy.process_item = process_item
        return dataset_copy

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
        return len(self.train_indices)

    def get_val_num_samples(self):  # noqa: D102
        return len(self.val_indices)

    def get_test_num_samples(self):  # noqa: D102
        return len(self.test_indices)
