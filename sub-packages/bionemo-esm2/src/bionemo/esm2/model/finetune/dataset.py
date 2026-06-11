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


import os
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import Tensor
from torch.utils.data import Dataset

from bionemo.esm2.data import tokenizer
from bionemo.llm.data.collate import MLM_LOSS_IGNORE_INDEX
from bionemo.llm.data.label2id_tokenizer import Label2IDTokenizer
from bionemo.llm.data.types import BertSample


__all__: Sequence[str] = (
    "InMemoryPerTokenValueDataset",
    "InMemoryProteinDataset",
    "InMemorySingleValueDataset",
)


class InMemoryProteinDataset(Dataset):
    """An in-memory dataset that tokenize strings into BertSample instances."""

    def __init__(
        self,
        sequences: pd.Series,
        labels: pd.Series | None = None,
        labels_mask: pd.Series | None = None,
        task_type: Literal["classification", "regression", None] = None,
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
        seed: int = np.random.SeedSequence().entropy,  # type: ignore
    ):
        """Initializes a dataset of protein sequences.

        This is an in-memory dataset that does not apply masking to the sequence. But keeps track of <mask> in the
        dataset sequences provided.

        Args:
            sequences (pd.Series): A pandas Series containing protein sequences.
            labels (pd.Series, optional): A pandas Series containing labels. Defaults to None.
            labels_mask (pd.Series, optional): A pandas Series containing loss mask, i.e. which tokens to keep for loss calculation. Defaults to None. can be 0 or 1.
            task_type (str, optional): Fine-tuning task type. Defaults to None.
            tokenizer (tokenizer.BioNeMoESMTokenizer, optional): The tokenizer to use. Defaults to tokenizer.get_tokenizer().
            seed: Random seed for reproducibility. This seed is mixed with the index of the sample to retrieve to ensure
                that __getitem__ is deterministic, but can be random across different runs. If None, a random seed is
                generated.
        """
        self.sequences = sequences
        self.labels = labels
        self.labels_mask = labels_mask
        self.task_type = task_type

        self.seed = seed
        self._len = len(self.sequences)
        self.tokenizer = tokenizer

    @classmethod
    def from_csv(
        cls,
        csv_path: str | os.PathLike,
        task_type: Literal["classification", "regression", None] = None,
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
        ignore_labels: bool = False,
        label_column: str = "labels",
        labels_mask_column: Optional[str] = None,
    ):
        """Class method to create a ProteinDataset instance from a CSV file.

        Args:
            csv_path: path to CSV file containing sequences and optionally labels column.
            task_type (str, optional): Fine-tuning task type. Defaults to None.
            tokenizer (tokenizer.BioNeMoESMTokenizer, optional): The tokenizer to use. Defaults to tokenizer.get_tokenizer().
            ignore_labels (bool): ignore labels column if exist (to avoid reading labels during inference)
            label_column (str): label column name in CSV file. Defaults to `labels`.
            labels_mask_column (str, optional): labels mask column name in CSV file. Defaults to None.
        """
        df = pd.read_csv(csv_path)

        # Validate presence of required columns
        if "sequences" not in df.columns and "sequence" not in df.columns:
            raise KeyError("The CSV must contain a 'sequences' or 'sequence' column.")

        sequences = df["sequences"] if "sequences" in df.columns else df["sequence"]

        if labels_mask_column is not None and cls is not InMemoryPerTokenValueDataset:
            raise ValueError("labels_mask_column is only supported for InMemoryPerTokenValueDataset")
        labels = None
        if not ignore_labels:
            labels = df[label_column]
        labels_mask = None
        if labels_mask_column is not None:
            labels_mask = df[labels_mask_column]

        if labels_mask_column is None:
            return cls(sequences, labels=labels, task_type=task_type, tokenizer=tokenizer)
        else:
            return cls(sequences, labels=labels, labels_mask=labels_mask, task_type=task_type, tokenizer=tokenizer)

    def __len__(self) -> int:
        """The size of the dataset."""
        return self._len

    def __getitem__(self, index: int) -> BertSample:
        """Obtains the BertSample at the given index."""
        sequence = self.sequences[index]

        # Tokenize sequence and track special token positions
        tokenized_sequence, special_positions = self._tokenize_with_special_tracking(sequence)

        # Handle labels_mask if it exists
        if self.labels_mask is not None:
            # Get the original labels_mask for this sequence
            labels_mask = list(self.labels_mask.iloc[index])

            # Insert 0 tokens at special token positions to pad to tokenized_sequence length
            idx = 0
            while idx < len(special_positions):
                labels_mask.insert(special_positions[idx], "0")
                idx += 1
            labels_mask = "".join(labels_mask)
            assert len(labels_mask) == len(tokenized_sequence), (
                f"labels_mask length {len(labels_mask)} != tokenized_sequence length {len(tokenized_sequence)}"
            )
        else:
            labels_mask = None

        label = tokenized_sequence if self.labels is None else self.transform_label(self.labels.iloc[index])
        # Overall mask for a token being masked in some capacity - either mask token, random token, or left as-is
        loss_mask = ~torch.isin(tokenized_sequence, Tensor(self.tokenizer.all_special_ids))

        # Combine with labels_mask if it exists
        if labels_mask is not None:
            # Convert string labels_mask to boolean tensor
            labels_mask_tensor = torch.from_numpy(np.fromiter(labels_mask, dtype="U1") == "1")
            loss_mask &= labels_mask_tensor

        return {
            "text": tokenized_sequence,
            "types": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
            "attention_mask": torch.ones_like(tokenized_sequence, dtype=torch.int64),
            "labels": label,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
        }

    def _tokenize_with_special_tracking(self, sequence: str) -> tuple[Tensor, list[int]]:
        """Tokenize sequence and return positions where special tokens were added.

        Args:
            sequence: The protein sequence to tokenize

        Returns:
            Tuple of (tokenized_sequence, special_token_positions)
        """
        # Tokenize without special tokens to get original length
        tokens_no_special = self.tokenizer.encode(sequence, add_special_tokens=False, return_tensors="pt")
        if isinstance(tokens_no_special, list):
            tokens_no_special = torch.tensor(tokens_no_special)
        tokens_no_special = tokens_no_special.flatten()

        # Tokenize with special tokens
        tokens_with_special = self.tokenizer.encode(sequence, add_special_tokens=True, return_tensors="pt")
        if isinstance(tokens_with_special, list):
            tokens_with_special = torch.tensor(tokens_with_special)
        tokens_with_special = tokens_with_special.flatten()

        mask = torch.isin(tokens_with_special, torch.tensor(self.tokenizer.all_special_ids))
        # Find positions where special tokens were inserted
        special_positions = mask.nonzero(as_tuple=False).squeeze(1).tolist()

        return tokens_with_special, special_positions

    def _tokenize(self, sequence: str) -> Tensor:
        """Tokenize a protein sequence.

        Args:
            sequence: The protein sequence.

        Returns:
            The tokenized sequence.
        """
        tensor = self.tokenizer.encode(sequence, add_special_tokens=True, return_tensors="pt")
        if isinstance(tensor, list):
            tensor = torch.tensor(tensor)
        return tensor.flatten()  # type: ignore

    def transform_label(self, label):
        """Transform the label.

        This method should be implemented by subclass if label needs additional transformation.

        Args:
            label: label to be transformed

        Returns:
            transformed_label
        """
        return label


class InMemorySingleValueDataset(InMemoryProteinDataset):
    """An in-memory dataset that tokenizes strings into BertSample instances."""

    def __init__(
        self,
        sequences: pd.Series,
        labels: pd.Series,
        task_type: Literal["classification", "regression"] = "regression",
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
        seed: int = np.random.SeedSequence().entropy,  # type: ignore
    ):
        """Initializes a dataset for single-value fine-tuning.

        This is an in-memory dataset that does not apply masking to the sequence. But keeps track of <mask> in the
        dataset sequences provided.

        Args:
            sequences (pd.Series): A pandas Series containing protein sequences.
            labels (pd.Series, optional): A pandas Series containing labels.
            task_type (str): Fine-tuning task type. Defaults to regression.
            tokenizer (tokenizer.BioNeMoESMTokenizer, optional): The tokenizer to use. Defaults to tokenizer.get_tokenizer().
            seed: Random seed for reproducibility. This seed is mixed with the index of the sample to retrieve to ensure
                that __getitem__ is deterministic, but can be random across different runs. If None, a random seed is
                generated.
        """
        super().__init__(sequences=sequences, labels=labels, task_type=task_type, tokenizer=tokenizer, seed=seed)

        self.task_type = task_type
        if self.task_type == "classification":
            label_tokenizer = Label2IDTokenizer()
            self.label_tokenizer = label_tokenizer.build_vocab(
                self.labels.sort_values(inplace=False).values.reshape(-1, 1)
            )

    def transform_label(self, label: float | str) -> Tensor:
        """Transform the regression label.

        Args:
            label: single regression/classification value

        Returns:
            tokenized label
        """
        if self.task_type == "regression":
            return torch.tensor([label], dtype=torch.float)
        elif self.task_type == "classification":
            tokenized_label = torch.tensor(self.label_tokenizer.text_to_ids([label]))
            return tokenized_label
        else:
            raise ValueError(f"{self.task_type} task type is not supported with {self.__class__.__name__}")


class InMemoryPerTokenValueDataset(InMemoryProteinDataset):
    """An in-memory dataset of labeled strings, which are tokenized on demand."""

    def __init__(
        self,
        sequences: pd.Series,
        labels: pd.Series,
        labels_mask: Optional[pd.Series] = None,
        task_type: Literal["classification", "regression"] = "classification",
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
        seed: int = np.random.SeedSequence().entropy,  # type: ignore
    ):
        """Initializes a dataset for per-token classification fine-tuning.

        This is an in-memory dataset that does not apply masking to the sequence. But keeps track of <mask> in the
        dataset sequences provided.

        Args:
            sequences (pd.Series): A pandas Series containing protein sequences.
            labels (pd.Series, optional): A pandas Series containing labels. Defaults to None.
            labels_mask (pd.Series, optional): A pandas Series containing loss mask, i.e. which tokens to keep for loss calculation. Defaults to None. can be 0 or 1.
            task_type (str): Fine-tuning task type. Defaults to classification. Regression per-token values are not supported.
            tokenizer (tokenizer.BioNeMoESMTokenizer, optional): The tokenizer to use. Defaults to tokenizer.get_tokenizer().
            seed: Random seed for reproducibility. This seed is mixed with the index of the sample to retrieve to ensure
                that __getitem__ is deterministic, but can be random across different runs. If None, a random seed is
                generated.
        """
        super().__init__(
            sequences=sequences,
            labels=labels,
            labels_mask=labels_mask,
            task_type=task_type,
            tokenizer=tokenizer,
            seed=seed,
        )

        self.task_type = task_type
        if not task_type == "classification":
            raise ValueError(f"{task_type} task type is not supported with {self.__class__.__name__}")

        label_tokenizer = Label2IDTokenizer()
        self.label_tokenizer = label_tokenizer.build_vocab(self.labels.sort_values(inplace=False).values)
        self.label_cls_eos_id = MLM_LOSS_IGNORE_INDEX

    def transform_label(self, label: str) -> Tensor:
        """Transform the sequence label by tokenizing them.

        This method tokenizes a sequence of labels into a tensor of tokens and adds CLS/EOS tokens.

        Args:
            label: label sequence to be transformed

        Returns:
            tokenized label
        """
        tokenized_labels = torch.tensor(self.label_tokenizer.text_to_ids(label))

        # for multi-class (mutually exclusive) classification with CrossEntropyLoss
        cls_eos = torch.tensor([self.label_cls_eos_id], dtype=tokenized_labels.dtype)

        # add cls / eos label ids with padding value -100 to have the same shape as tokenized_sequence
        labels = torch.cat((cls_eos, tokenized_labels, cls_eos))
        return labels
