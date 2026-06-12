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

import csv
from collections import defaultdict
from pathlib import Path

import datasets
import datasets.distributed
import torch
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithFlattening,
)
from transformers.trainer_pt_utils import get_parameter_names

from collator import TokenPackingDataset
from distributed_config import DistributedConfig


SS3_ID2LABEL = {0: "H", 1: "E", 2: "C"}

SS3_LABEL2ID = {
    "H": 0,
    "I": 0,
    "G": 0,
    "E": 1,
    "B": 1,
    "S": 2,
    "T": 2,
    "~": 2,
    "C": 2,
    "L": 2,
}  # '~' denotes coil / unstructured

SS8_ID2LABEL = {0: "H", 1: "I", 2: "G", 3: "E", 4: "B", 5: "S", 6: "T", 7: "C"}

SS8_LABEL2ID = {
    "H": 0,
    "I": 1,
    "G": 2,
    "E": 3,
    "B": 4,
    "S": 5,
    "T": 6,
    "~": 7,
    "C": 7,
    "L": 7,
}  # '~' denotes coil / unstructured


def create_dataloader(
    distributed_config: DistributedConfig,
    use_sequence_packing: bool,
    tokenizer_name: str,
    micro_batch_size: int,
    val_micro_batch_size: int,
    num_workers: int,
    max_seq_length: int,
    stride: int,
    seed: int,
    ss3_classification: bool,
    load_dataset_kwargs: dict,
) -> tuple[DataLoader, DataLoader | None, IterableDataset | DistributedSampler]:
    """Create a dataloader for the secondary structure dataset."""
    dataset_or_dataset_dict = load_dataset(**load_dataset_kwargs)

    if isinstance(dataset_or_dataset_dict, dict):
        train_dataset = dataset_or_dataset_dict.get("train")
        assert train_dataset, "'train' split must be specified."
        val_dataset = dataset_or_dataset_dict.get("validation")
    else:
        train_dataset = dataset_or_dataset_dict
        val_dataset = None

    print(
        f"Loading dataset: path: '{load_dataset_kwargs['path']}' | data_files: '{load_dataset_kwargs['data_files']}'."
    )

    perform_validation = val_dataset is not None

    if isinstance(train_dataset, IterableDataset):
        train_dataset = datasets.distributed.split_dataset_by_node(
            train_dataset,
            rank=distributed_config.rank,
            world_size=distributed_config.world_size,
        )
        train_dataset = train_dataset.shuffle(seed=seed, buffer_size=10_000)

        if perform_validation:
            val_dataset = datasets.distributed.split_dataset_by_node(
                val_dataset,
                rank=distributed_config.rank,
                world_size=distributed_config.world_size,
            )

    if ss3_classification:
        ss_token_map = SS3_LABEL2ID
    else:
        ss_token_map = SS8_LABEL2ID

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenize_args = {
        "max_length": max_seq_length,
        "truncation": True,
        "stride": stride,
        "return_overflowing_tokens": True,
        "return_offsets_mapping": True,
    }

    def tokenize(example):
        """Tokenize both the input protein sequence and the secondary structure labels."""
        result = tokenizer(example["Sequence"], **tokenize_args)

        # While we can use the rust-based tokenizer for the protein sequence, we manually encode the secondary structure
        # labels. Our goal is to return a list of integer labels with the same shape as the input_ids.
        labels = []
        for batch_idx in range(len(result["input_ids"])):
            sequence_labels = []

            # This array maps the possibly-chunked result["input_ids"] to the original sequence. Because of
            # `return_overflowing_tokens`, each input sequence may be split into multiple input rows.
            offsets = result["offset_mapping"][batch_idx]

            # This gets the original secondary structure sequence for the current chunk.
            ss_sequence = example["Secondary_structure"][result["overflow_to_sample_mapping"][batch_idx]]

            for offset_start, offset_end in offsets:
                if offset_start == offset_end:
                    sequence_labels.append(-100)  # Start and end of the sequence tokens can be ignored.
                elif offset_end == offset_start + 1:  # All tokens are single-character.
                    ss_char = ss_sequence[offset_start]
                    ss_label_value = ss_token_map[ss_char]  # Encode the secondary structure character
                    sequence_labels.append(ss_label_value)
                else:
                    raise ValueError(f"Invalid offset: {offset_start} {offset_end}")

            labels.append(sequence_labels)

        return {"input_ids": result["input_ids"], "labels": labels}

    train_tokenized_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[col for col in train_dataset.features if col not in ["input_ids", "labels"]],
    )

    if isinstance(train_tokenized_dataset, IterableDataset):
        train_sampler = None
    else:
        train_sampler = DistributedSampler(
            train_tokenized_dataset,
            rank=distributed_config.rank,
            num_replicas=distributed_config.world_size,
            seed=seed,
        )

    if use_sequence_packing:
        assert isinstance(train_tokenized_dataset, datasets.IterableDataset), (
            "THD token packing requires a streaming dataset."
        )
        collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
        train_tokenized_dataset = TokenPackingDataset(
            train_tokenized_dataset, max_tokens_per_batch=micro_batch_size * max_seq_length
        )
        batch_size = None  # The TokenPackingDataset will handle the batching.
    else:
        collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer, padding="max_length", max_length=max_seq_length
        )
        batch_size = micro_batch_size

    train_dataloader = DataLoader(
        train_tokenized_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    if perform_validation:
        val_tokenized_dataset = val_dataset.map(
            tokenize,
            batched=True,
            remove_columns=[col for col in val_dataset.features if col not in ["input_ids", "labels"]],
        )

        if isinstance(val_tokenized_dataset, IterableDataset):
            val_sampler = None
        else:
            val_sampler = DistributedSampler(
                val_tokenized_dataset,
                rank=distributed_config.rank,
                num_replicas=distributed_config.world_size,
                seed=seed,
            )

        if use_sequence_packing:
            assert isinstance(val_tokenized_dataset, datasets.IterableDataset), (
                "THD token packing requires a streaming dataset."
            )
            collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
            val_tokenized_dataset = TokenPackingDataset(
                val_tokenized_dataset, max_tokens_per_batch=micro_batch_size * max_seq_length
            )
            val_batch_size = None  # The TokenPackingDataset will handle the batching.
        else:
            collator = DataCollatorForTokenClassification(
                tokenizer=tokenizer, padding="max_length", max_length=max_seq_length
            )
            val_batch_size = val_micro_batch_size

        val_dataloader = DataLoader(
            val_tokenized_dataset,
            sampler=val_sampler,
            batch_size=val_batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader, train_tokenized_dataset if train_sampler is None else train_sampler


def compute_accuracy(preds, labels, ignore_index=-100) -> tuple[int, int]:
    """Calculate the accuracy."""
    preds_labels = torch.argmax(preds, dim=-1)
    mask = labels != ignore_index
    correct = (preds_labels == labels) & mask

    return correct.sum().item(), mask.sum().item()


def get_parameter_names_with_lora(model):
    """Get layers with non-zero weight decay.

    This function reuses the Transformers' library function
    to list all the layers that should have weight decay.
    """
    forbidden_name_patterns = [
        r"bias",
        r"layernorm",
        r"rmsnorm",
        r"(?:^|\.)norm(?:$|\.)",
        r"_norm(?:$|\.)",
        r"\.lora_[AB]\.",
    ]

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns)

    return decay_parameters


def load_fasta(path: Path) -> list[dict]:
    """Read FASTA file and return input sequences."""
    records = []
    seq, pdb_id = [], None

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith(">"):
                if seq:
                    records.append({"pdb_id": pdb_id, "sequence": "".join(seq)})
                pdb_id = line[1:] or None
                seq = []
            else:
                seq.append(line)

        if seq:
            records.append({"pdb_id": pdb_id, "sequence": "".join(seq)})

    return records


def load_csv(path: Path) -> list[dict]:
    """Read input CSV file for inference.

    It is assumed that the input CSV file contains:
    - Optional column named 'pdb_id' of the sequence.
    - Aminoacid sequence.
    """
    with open(path) as f:
        reader = csv.DictReader(f)
        has_pdb_id = "pdb_id" in reader.fieldnames

        return [
            {
                "pdb_id": row["pdb_id"] if has_pdb_id else None,
                "sequence": row["sequence"],
            }
            for row in reader
        ]


def load_input(path: Path) -> list[dict]:
    """Read the input sequences from FASTA or CSV file."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_csv(path)
    elif suffix in {".fa", ".fasta", ".faa"}:
        return load_fasta(path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")


def format_output_rows(records, predictions, sequences_to_sample_mapping):
    """Format the output into CSV-type lines.

    Returns:
      header: list[str]
      rows: list[tuple[str, str]]
    """
    has_pdb_id = any(r.get("pdb_id") for r in records)
    header = ["pdb_id", "prediction"] if has_pdb_id else ["id", "prediction"]

    counts = defaultdict(int)
    rows = []

    for pred, orig_idx in zip(predictions, sequences_to_sample_mapping):
        counts[orig_idx] += 1
        suffix = counts[orig_idx]

        base = records[orig_idx]["pdb_id"] if has_pdb_id else str(orig_idx)

        out_id = base if suffix == 1 else f"{base}_{suffix}"
        rows.append((out_id, pred))

    return header, rows


def write_output(records, predictions, sequences_to_sample_mapping: list[int], output_path: Path):
    """Write the predictions to an output file."""
    header, rows = format_output_rows(records, predictions, sequences_to_sample_mapping)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
