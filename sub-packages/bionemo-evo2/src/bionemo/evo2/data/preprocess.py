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


"""Module containing data preprocessing and splitting functions for Evo2 in BioNeMo.

It can also be utilized as a script to dump pre-processed data to JSON.
"""

import argparse
import multiprocessing as mp
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Semaphore
from typing import Optional

import numpy as np
import torch
import yaml
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
from nemo.utils import logging

from bionemo.evo2.data.tokenizer import Evo2Tokenizer
from bionemo.evo2.utils.config import Evo2PreprocessingConfig, Evo2TaxonomyLineage
from bionemo.noodles import back_transcribe_sequence, complement_sequence, reverse_sequence, transcribe_sequence
from bionemo.noodles.nvfaidx import NvFaidx


class Evo2Preprocessor:
    """Data preprocessing class for Evo2."""

    BIN = ".bin"
    IDX = ".idx"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __init__(self, params: Evo2PreprocessingConfig | None = None):
        """Initialize Evo2Preprocessor.

        Args:
            params (Evo2PreprocessingConfig | None): Configuration parameters for preprocessing.
        """
        self.tokenizer: Evo2Tokenizer = Evo2Tokenizer(params)

    @staticmethod
    @contextmanager
    def preprocessing_context_manager(seed: Optional[int] = None):
        """Context manager for setting and restoring the random number generator state.

        Args:
            seed (int | None): Seed for the random number generator. Defaults to None.
        """
        # Track current state.
        current_state = random.getstate()
        try:
            # Set random seed.
            random.seed(seed)
            yield seed
        finally:
            # Restore random state.
            random.setstate(current_state)

    @staticmethod
    def _get_output_filename(
        config: Evo2PreprocessingConfig, ext: Optional[str] = None, split: Optional[str] = None, temp: bool = False
    ) -> Path:
        """Generate the output filename for the preprocessed data.

        Args:
            config (Evo2PreprocessingConfig): Configuration object containing preprocessing settings.
            ext (Optional[str]): File extension for the output file. Defaults to None.
            split (Optional[str]): Data split type (e.g., 'train', 'val', 'test'). Defaults to None.
            temp (bool): Flag indicating whether the file is temporary. Defaults to False.

        Returns:
            Path: The constructed output file path.
        """
        # Get output directory. Defaults to CWD.
        output_dir = config.output_dir
        if output_dir is None:
            output_dir = Path.cwd()
        # Pickup output file prefix.
        config_prefix = "{}_{}".format(config.output_prefix, config.tokenizer_type.lower().replace(" ", ""))
        output_filepath = Path(output_dir) / (
            config_prefix
            + (f"_{split}" if split is not None else "")
            + (ext if ext is not None else "")
            + (".tmp" if temp else "")
        )
        return output_filepath

    @staticmethod
    def _subsequence_generator(sequence: str, subsequence_length: Optional[int] = None, offset: Optional[int] = None):
        """Generate subsequences from a given sequence.

        Args:
            sequence (str): The input sequence.
            subsequence_length (int | None): Length of each subsequence. Defaults to the length of the sequence.
            offset (int | None): Step size for generating subsequences. Defaults to subsequence_length.

        Yields:
            str: Subsequences of the input sequence.
        """
        subsequence_length = subsequence_length if subsequence_length is not None else len(sequence)
        step_size = offset if offset is not None else subsequence_length
        for i in range(0, len(sequence), step_size):
            yield sequence[i : i + subsequence_length]

    @staticmethod
    def _random_reverse_complement(seq: str, prob: float = 0.0, seed: Optional[int] = None):
        """Randomly reverse complements a DNA sequence based on a given probability.

        Args:
            seq (str): The DNA sequence to potentially reverse complement.
            prob (float): The probability of reverse complementing the sequence. Defaults to 0.0.
            seed (Optional[int]): The seed for the random number generator. Defaults to None.

        Returns:
            str: The original or reverse complemented DNA sequence based on the probability.
        """
        with Evo2Preprocessor.preprocessing_context_manager(seed):
            if random.random() < prob:
                return complement_sequence(reverse_sequence(seq))
            else:
                return seq

    @staticmethod
    def _reverse_complement_expansion(seq: str):
        """Generate a list containing the original and reverse complemented sequence.

        Args:
            seq (str): The input DNA sequence.

        Returns:
            list[str]: List containing the original and reverse complemented sequence.
        """
        return [seq, complement_sequence(reverse_sequence(seq))]

    @staticmethod
    def _train_val_test_split(train_weight: float, val_weight: float, test_weight: float, seed: Optional[int] = None):
        """Randomly assign a data point to train, validation, or test split based on provided weights.

        Args:
            train_weight (float): The weight for the training split.
            val_weight (float): The weight for the validation split.
            test_weight (float): The weight for the test split.
            seed (Optional[int]): The seed for the random number generator. Defaults to None.

        Returns:
            str: The split assignment ('train', 'val', or 'test').

        Raises:
            ValueError: If the sum of the weights is zero or negative.
        """
        with Evo2Preprocessor.preprocessing_context_manager(seed if seed is not None else None):
            # Generate random number.
            roll = random.random()
            # Rectify and normalize split ratios.
            total_weight = abs(train_weight) + abs(val_weight) + abs(test_weight)
            if total_weight <= 0:
                raise ValueError("Train-validation-test split proportions cannot be zero.")
            train_split = abs(train_weight) / total_weight
            test_split = abs(test_weight) / total_weight
            split = "train"
            if roll > train_split:
                if roll < 1 - test_split:
                    split = "val"
                else:
                    split = "test"
            return split

    @staticmethod
    def _construct_taxonomy_token(
        lineage: Evo2TaxonomyLineage, dropout: float = 0.0, seed: Optional[int] = None
    ) -> Optional[str]:
        """Construct a special Taxonomy token for natural language prompting of DNA generation models.

        Args:
            lineage (Evo2TaxonomyLineage): The taxonomy lineage information.
            dropout (float): The probability of dropping out segments of the lineage. Defaults to 0.0.
            seed (Optional[int]): The seed for the random number generator. Defaults to None.

        Returns:
            Optional[str]: The constructed taxonomy token or None if lineage is None.
        """
        # If dropout > 0, randomly drop out segments of the lineage for training on incomplete lineages.
        with Evo2Preprocessor.preprocessing_context_manager(seed if seed is not None else None):
            return (
                "|d__{};p__{};c__{};o__{};f__{};g__{};s__{}|".format(
                    lineage.domain if random.random() >= dropout else None,
                    lineage.phylum if random.random() >= dropout else None,
                    lineage.clazz if random.random() >= dropout else None,
                    lineage.order if random.random() >= dropout else None,
                    lineage.family if random.random() >= dropout else None,
                    lineage.genus if random.random() >= dropout else None,
                    lineage.species if random.random() >= dropout else None,
                )
                if lineage is not None
                else None
            )

    def preprocess_data(self, filepath: str, seqid: str, seq: str, seq_idx: int, config: Evo2PreprocessingConfig):
        """Preprocess fasta datapaths.

        Args:
            filepath (str): Path to the .fasta file.
            seqid (str): Sequence ID.
            seq (str): DNA sequence.
            seq_idx (int): Sequence index.
            config (Evo2PreprocessingConfig): Configuration object containing preprocessing settings.

        Returns:
            tuple[list[dict], float]: Preprocessed data and the time taken for preprocessing.
        """
        # Timing.
        start = time.time()
        # Retrieve taxonomy lineage string if SeqID has associated taxonomy data.
        # Note: Better implemented as a suffix tree substring dictionary, but convenient
        # for identifying a large amount of sequences with identical lineages.
        # Slow for extremely large dictionaries of (SeqID Substr, Taxonomy) pairs.
        lineage = None
        for id, tax in config.taxonomy_data.items():
            # Taxonomy ID is a substring of Seq ID.
            if id in seqid:
                lineage = tax
                break

        # Preprocess data.
        preproc_data = []
        with self.preprocessing_context_manager(
            config.seed + hash(filepath) + seq_idx if config.seed is not None else None
        ):
            # Randomly reverse complement the sequence.
            seq = self._random_reverse_complement(seq, prob=config.random_reverse_complement)
            seqs_to_parse = self._reverse_complement_expansion(seq) if config.embed_reverse_complement else [seq]
            for seq in seqs_to_parse:
                # Sequence Modifiers
                if config.force_uppercase:
                    seq = seq.upper()
                if config.transcribe == "transcribe":
                    seq = transcribe_sequence(seq)
                elif config.transcribe == "back_transcribe":
                    seq = back_transcribe_sequence(seq)
                if config.drop_empty_sequences and len(seq) == 0:
                    continue
                if config.nnn_filter and "NNN" in seq.upper():
                    continue

                # Construct taxonomy token with random dropout on the lineage categories per sequence.
                taxonomy_token = self._construct_taxonomy_token(lineage, dropout=config.random_lineage_dropout)

                # Inject taxonomy lineage tokens every prompt_spacer_length tokens in the sequence.
                # If the taxonomy lineage token is not provided, then just take the original sequence.
                target_length = (
                    config.prompt_spacer_length - len(taxonomy_token) if taxonomy_token is not None else None
                )
                taxonomy_injected_sequence = [
                    taxonomy_token + str(subseq) if taxonomy_token is not None else str(subseq)
                    for subseq in self._subsequence_generator(seq, target_length, target_length)
                ]

                # Wrap and tokenize.
                preproc_data_record = {
                    "text": "".join(taxonomy_injected_sequence),
                }
                preproc_data_record["tokens"] = self.tokenizer.tokenize(
                    preproc_data_record["text"],
                    use_ftfy=config.ftfy,
                    enforce_sample_length=config.enforce_sample_length,
                    append_eod=config.append_eod,
                    drop_empty_sequences=config.drop_empty_sequences,
                )
                preproc_data.append(preproc_data_record)
        end = time.time()
        return preproc_data, end - start

    def preprocess_data_task(self, file_sequence_config):
        """Wrapper function to unpack args for preprocess_data.

        Args:
            file_sequence_config (tuple): Tuple containing arguments for preprocess_data.

        Returns:
            tuple[list[dict], float]: Preprocessed data and the time taken for preprocessing.
        """
        return self.preprocess_data(*file_sequence_config)

    @staticmethod
    def _yield_sequences_from_files(config: Evo2PreprocessingConfig, semaphore: Semaphore):
        """Iterator over sequences within multiple input documents. Arguments for multiprocessing tasks.

        Utilized to limit the amount of sequences streamed into memory.

        Args:
            config (Evo2PreprocessingConfig): Configuration object containing preprocessing settings.
            semaphore (Semaphore): Semaphore to limit the number of sequences in memory.

        Yields:
            tuple: Arguments for preprocess_data.
        """

        def yielder(fname, semaphore):
            # Read FASTA.
            index = NvFaidx(fname)
            for i, (seqid, sequence) in enumerate(index.items()):
                semaphore.acquire()
                # Yield filename and sequence within fasta.
                yield str(fname), seqid, sequence, i, config

        for fname in config.datapaths:
            semaphore.acquire()
            yield from yielder(fname, semaphore)

    def preprocess_generator(self, preproc_config: Evo2PreprocessingConfig):
        """Main function to preprocess data for Evo2.

        Args:
            preproc_config (Evo2PreprocessingConfig): Configuration object containing preprocessing settings.

        Yields:
            tuple[dict, float]: Preprocessed sequence data and the time taken for preprocessing.
        """
        # Track which splits have been assigned
        split_assignments = {
            "train": preproc_config.train_split > 0,
            "val": preproc_config.valid_split > 0,
            "test": preproc_config.test_split > 0,
        }
        splits_needed = {k for k, v in split_assignments.items() if v}

        # Instantiate multiprocessing pool. Use semaphore to limit the amount of sequences to read into memory.
        semaphore = Semaphore(preproc_config.preproc_concurrency + preproc_config.workers)
        if preproc_config.workers > 1:
            pool = mp.Pool(preproc_config.workers)
            # Ordered imap for downstream seeded splitting.
            preproc_tasks = pool.imap(
                self.preprocess_data_task,
                self._yield_sequences_from_files(preproc_config, semaphore),
                chunksize=preproc_config.chunksize,
            )
        else:
            preproc_tasks = (
                self.preprocess_data_task(x) for x in self._yield_sequences_from_files(preproc_config, semaphore)
            )

        # Preprocess data and split results into train, test, and split.
        with self.preprocessing_context_manager(preproc_config.seed if preproc_config.seed is not None else None):
            for result, elapsed_time in preproc_tasks:
                # Release semaphore for the task associated with the result.
                semaphore.release()
                # If we still need to ensure splits are assigned
                if splits_needed:
                    # Force assign to a needed split
                    split = splits_needed.pop()
                else:
                    # Regular random assignment
                    split = self._train_val_test_split(
                        preproc_config.train_split, preproc_config.valid_split, preproc_config.test_split
                    )
                for sequence in result:
                    sequence["split"] = split
                    yield sequence, elapsed_time

    def preprocess_offline(self, preproc_config: Evo2PreprocessingConfig):
        """Offline data preprocessing script for Evo2.

        Args:
            preproc_config (Evo2PreprocessingConfig): Configuration object containing preprocessing settings.
        """
        # Validate if binaries have already been produced for the given config and overwrite is set to False.
        if any(
            self._get_output_filename(preproc_config, ext, split).is_file()
            for ext, split in zip([self.BIN, self.IDX], [self.TRAIN, self.VAL, self.TEST])
        ):
            if not preproc_config.overwrite:
                # Skip this dataset!
                logging.info(
                    f"Skipped overwriting (overwrite: False) existing preprocessed data: {preproc_config.output_prefix}"
                )
                return
            else:
                logging.info(
                    f"Overwriting (overwrite: True) existing preprocessed data: {preproc_config.output_prefix}"
                )

        # Instantiate indexed data builders.
        dataset_dtype = getattr(np, preproc_config.indexed_dataset_dtype)
        temp_train_bin = self._get_output_filename(preproc_config, self.BIN, self.TRAIN, temp=True)
        temp_val_bin = self._get_output_filename(preproc_config, self.BIN, self.VAL, temp=True)
        temp_test_bin = self._get_output_filename(preproc_config, self.BIN, self.TEST, temp=True)
        train_builder: IndexedDatasetBuilder = IndexedDatasetBuilder(bin_path=str(temp_train_bin), dtype=dataset_dtype)
        val_builder: IndexedDatasetBuilder = IndexedDatasetBuilder(bin_path=str(temp_val_bin), dtype=dataset_dtype)
        test_builder: IndexedDatasetBuilder = IndexedDatasetBuilder(bin_path=str(temp_test_bin), dtype=dataset_dtype)
        logging.info(f"Created temporary binary datasets: {temp_train_bin} {temp_val_bin} {temp_test_bin}")

        # Preprocess data and split results into train, validation, or test.
        avg_preproc_time = 0.0
        avg_index_time = 0.0
        count = 0
        for sequence, elapsed_time in self.preprocess_generator(preproc_config):
            index_start_time = time.time()
            if sequence["split"] == "train":
                train_builder.add_item(torch.Tensor(sequence["tokens"]))
                train_builder.end_document()
            elif sequence["split"] == "val":
                val_builder.add_item(torch.Tensor(sequence["tokens"]))
                val_builder.end_document()
            elif sequence["split"] == "test":
                test_builder.add_item(torch.Tensor(sequence["tokens"]))
                test_builder.end_document()
            index_end_time = time.time()
            # Update average preprocessing and indexing time.
            avg_preproc_time = (avg_preproc_time * count + elapsed_time) / (count + 1)
            avg_index_time = (avg_index_time * count + index_end_time - index_start_time) / (count + 1)
            count += 1

        # Report timing.
        logging.info(f"Average preprocessing time per sequence: {avg_preproc_time}")
        logging.info(f"Average indexing time per sequence: {avg_index_time}")
        logging.info(f"Number of sequences processed: {count}")

        # Write preprocessed index data to disk. Rename temporary binaries to denote preprocessing completion.
        train_builder.finalize(idx_path=str(self._get_output_filename(preproc_config, self.IDX, self.TRAIN)))
        val_builder.finalize(idx_path=str(self._get_output_filename(preproc_config, self.IDX, self.VAL)))
        test_builder.finalize(idx_path=str(self._get_output_filename(preproc_config, self.IDX, self.TEST)))
        os.rename(temp_train_bin, self._get_output_filename(preproc_config, self.BIN, self.TRAIN))
        os.rename(temp_val_bin, self._get_output_filename(preproc_config, self.BIN, self.VAL))
        os.rename(temp_test_bin, self._get_output_filename(preproc_config, self.BIN, self.TEST))


def parse_args():
    """Parse arguments for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess FASTA files for training Evo2.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to data preprocessing config JSON.")
    return parser.parse_args()


def main():
    """Main function to execute the preprocessing script.

    This function parses command-line arguments, reads the configuration file,
    and initiates the preprocessing of data as specified in the configuration.
    """
    # Parse arguments.
    args = parse_args()
    # Read config YAML.
    with open(args.config, "r") as yaml_fs:
        evo2_preproc_config_batch = yaml.safe_load(yaml_fs)
    for config in evo2_preproc_config_batch:
        start = time.time()
        # Convert into Evo2PreprocessingConfig.
        evo2_preproc_config = Evo2PreprocessingConfig(**config)
        if evo2_preproc_config.output_dir is not None:
            evo2_preproc_config.output_dir.mkdir(parents=True, exist_ok=True)
        # Instantiate Evo2Preprocessor.
        evo2_preprocessor = Evo2Preprocessor(evo2_preproc_config)
        # Preprocess data specified in config.
        evo2_preprocessor.preprocess_offline(evo2_preproc_config)
        end = time.time()
        logging.info(
            f"Finished preprocessing {evo2_preproc_config.output_prefix} ({evo2_preproc_config.datapaths}) in {end - start:.3f} seconds with {evo2_preproc_config.workers} workers."
        )


if __name__ == "__main__":
    main()
