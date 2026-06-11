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


import random
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from nemo.utils import logging
from torch.utils.data import Dataset
from tqdm import tqdm

from bionemo.core.data.load import load
from bionemo.core.data.multi_epoch_dataset import EpochIndex
from bionemo.core.utils import random_utils
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.data.singlecell.utils import sample_or_truncate
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.data import masking, types
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset


__all__: Sequence[str] = (
    "SingleCellDataset",
    "process_item",
)


class SingleCellDataset(Dataset):
    """A dataset class for single-cell pre-training. These can be generated using the sc_memmap.py script. Future
    updates will contain more comprehensive workflows for generating a Sparse Memmap from scRNA-seq.

    Args:
        data_path (str): Path where the single cell files are stored in SingleCell Memmap format. It should contain the following files:
            - `metadata.json`: Path containing the number of rows int he dataset.
            - Gene expression matrix stored in CSR format as `numpy.memmap`:
                - `data.npy`: Non-zero gene expression values.
                - `col_ptr.npy`: Indices of the corresponding genes for each entry in data.npy.
                - `row_ptr.npy`: Column index pointers for each cell sample.
        tokenizer: The tokenizer to use for tokenizing the input data.
        median_dict (dict, optional): A dictionary containing median values for each gene. Defaults to None.
        max_len (int, optional): The maximum length of the input sequence. Defaults to 1024.
        include_unrecognized_vocab_in_dataset (bool, optional): If set to True, a hard-check is performed to verify all gene identifers are in the user supplied tokenizer vocab. Defaults to False which means any gene identifier not in the user supplied tokenizer vocab will be excluded.

    Attributes:
        data_path (str): Path where the single cell files are stored in SCDL memmap format.
        max_len (int): The maximum length of the input sequence.
        metadata (dict): Metadata loaded from `metadata.json`.
        gene_medians (dict): A dictionary containing median values for each gene. If None, a median of '1' is assumed for all genes.
        num_train (int): The number of samples in the training split.
        num_val (int): The number of samples in the validation split.
        num_test (int): The number of samples in the test split.
        index_offset (int): The offset to apply to the indices.
        length (int): The total number of samples in the dataset.
        gene_data (numpy.memmap): Gene expression values stored in CSR format.
        gene_data_indices (numpy.memmap): Gene indices associated with gene values.
        gene_data_ptr (numpy.memmap): Column indices for each sample.
        tokenizer: The tokenizer used for tokenizing the input data.
        dataset_ccum (numpy.ndarray): Cumulative sum of row counts to map row indices to dataset id.
        dataset_map (dict): Mapping of dataset id to dataset name.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    See Also:
        bionemo/data/singlecell/sc_memmap.py - creates the artifacts required for instantiating a singlecell dataset from hdf5 files.
    """  # noqa: D205

    def __init__(  # noqa: D107
        self,
        data_path: str | Path,
        tokenizer: Any,
        median_dict: Optional[dict] = None,
        max_len: int = 1024,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        prepend_cls_token: bool = True,
        eos_token: int | None = None,
        include_unrecognized_vocab_in_dataset: bool = False,
        seed: int = np.random.SeedSequence().entropy,  # type: ignore
    ):
        super().__init__()

        self.data_path = data_path
        self.max_len = max_len
        self.random_token_prob = random_token_prob
        self.mask_token_prob = mask_token_prob
        self.mask_prob = mask_prob
        self.prepend_cls_token = prepend_cls_token
        self._seed = seed
        self.eos_token = eos_token

        self.scdl = SingleCellMemMapDataset(str(data_path))
        self.length = len(self.scdl)
        # - median dict
        self.gene_medians = median_dict
        self.tokenizer = tokenizer
        self.include_unrecognized_vocab_in_dataset = include_unrecognized_vocab_in_dataset

    def __len__(self):  # noqa: D105
        return self.length

    def __getitem__(self, index: EpochIndex) -> types.BertSample:
        """Performs a lookup and the required transformation for the model."""
        rng = np.random.default_rng([self._seed, index.epoch, index.idx])
        values, feature_ids, _ = self.scdl.get_row(
            index.idx, return_var_features=True, var_feature_names=["feature_id"]
        )
        assert (
            len(feature_ids) == 1
        )  # we expect feature_ids to be a list containing one np.array with the row's feature ids
        gene_data, col_idxs = np.array(values[0]), np.array(values[1])
        if len(gene_data) == 0:
            raise ValueError(
                "SingleCellMemap data provided is invalid; the gene expression data parsed for the specified index is empty."
            )
        return process_item(
            gene_data,
            col_idxs,
            feature_ids[0],
            self.tokenizer,
            gene_median=self.gene_medians,
            rng=rng,
            max_len=self.max_len,
            mask_token_prob=self.mask_token_prob,
            mask_prob=self.mask_prob,
            random_token_prob=self.random_token_prob,
            prepend_cls_token=self.prepend_cls_token,
            eos_token=self.eos_token,
            include_unrecognized_vocab_in_dataset=self.include_unrecognized_vocab_in_dataset,
        )


def _gather_medians(
    gene_names: np.ndarray,
    gene_data: np.ndarray,
    normalize: bool,
    vocab: dict[str, int],
    gene_median: dict[str, float],
    include_unrecognized_vocab_in_dataset: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter out genes that are not in the provided tokenizer vocab, and tokenize the gene names."""
    genes, tokens, medians = [], [], []
    for tok, gene in zip(gene_names, gene_data):
        if tok in vocab:
            tokens.append(vocab[tok])
            genes.append(gene)
            if normalize:
                med = gene_median[tok]  # If not in the dictionary we default to no normalization (1)
                medians.append(med)
        elif include_unrecognized_vocab_in_dataset:
            raise ValueError(f"Provided gene identifier, {str(tok)}, is not in the tokenizer vocab.")
    return np.asarray(genes), np.asarray(tokens), np.asarray(medians)


def process_item(  # noqa: D417
    gene_data: np.ndarray,
    gene_idxs: np.ndarray,
    feature_ids: np.ndarray,
    tokenizer: GeneTokenizer,
    gene_median: dict,
    rng: np.random.Generator,
    max_len: int = 1024,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    target_sum: int = 10000,
    normalize: bool = True,
    prepend_cls_token: bool = True,
    eos_token: None | int = None,
    include_unrecognized_vocab_in_dataset: bool = False,
) -> types.BertSample:
    """Process a single item in the dataset.

    Optionally performs median normalization and rank ordering. The tokenizers CLS token is added to the beginning
    of every sample. Converts gene names to ensemble ids before tokenizing. Expects gene_medians to contain ensembl ids as keys.

    Args:
        gene_data (list): List of gene data, these are expression counts.
        gene_idxs (list): List of gene indices, these are keys in 'metadata['feature_ids']' and corresponding the CSR entry.
        feature_ids (list): Feature ids for the full dataset.
        tokenizer (Tokenizer): Tokenizer object.
        gene_median (optional(dict)): Dictionary of gene medians. Defaults to None. Expects ensembl IDs to be keys.
        rng: Random number generator to ensure deterministic results.
        max_len (int): Maximum length of the item. Defaults to 1024. Applies padding to any sequence shorter than max_len and truncates any sequence longer than max_len.
        mask_prob (float): Probability of masking a token. Defaults to 0.15.
        target_sum (int): Target sum for normalization. Defaults to 10000.
        normalize (bool): Flag to normalize the gene data. Defaults to True.
            When set, this re-orders the gene tokens by their median expression value.
        probabilistic_dirichlet_sampling (bool): Flag to enable probabilistic dirichlet sampling. Defaults to False.
        dirichlet_alpha (float): Alpha value for dirichlet sampling if set by `probabilistic_dirichlet_sampling`. Defaults to 0.5.
        same_length (bool): when true, sample the same length of genes as you originally had before the dirichlet sampler.
        recompute_globals (bool): when true, global arrays are always recomputed. this is only useful for testing.
        include_unrecognized_vocab_in_dataset (bool, optional): If set to True, a hard-check is performed to verify all gene identifers are in the user supplied tokenizer vocab. Defaults to False which means any gene identifier not in the user supplied tokenizer vocab will be excluded.

    Returns:
        dict: Processed item dictionary.

    NOTE: this method is very important and very useful. To generalize thiswwe should add an abstraction for
        Datasets that have some kind of functor transformation.
    """
    if max_len < 1:
        raise ValueError(f"max_len must be greater than 1, {max_len=}")

    if gene_median is None:
        raise ValueError("gene_median must be provided for this tokenizer")

    if prepend_cls_token:
        max_len = max_len - 1  # - minus 1 for [CLS] token
    if eos_token is not None:
        max_len = max_len - 1  # - minus 1 for [EOS] token

    gene_names = feature_ids[gene_idxs]

    gene_expression_cell, token_ids, gene_expression_medians = _gather_medians(
        gene_names,
        gene_data,
        normalize,
        tokenizer.vocab,
        gene_median,
        include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
    )

    if normalize:
        # re-order according to expression median normalized rank. descending order.

        gene_expression_cell = gene_expression_cell / gene_expression_cell.sum() * target_sum
        gene_expression_cell = gene_expression_cell / gene_expression_medians.astype(float)
        idxs = np.argsort(
            -gene_expression_cell
        )  # sort in descending order so that the 0th position is the highest value.
        gene_expression_cell = gene_expression_cell[idxs]
        token_ids = token_ids[idxs]

    # - select max_len subset, set sample to false so it doesnt permute the already rank ordered expression values.
    token_ids = sample_or_truncate(token_ids, max_len, sample=False)
    with torch.no_grad(), torch.device("cpu"):
        masked_tokens, labels, loss_mask = masking.apply_bert_pretraining_mask(
            tokenized_sequence=torch.from_numpy(token_ids),
            random_seed=int(random_utils.get_seed_from_rng(rng)),
            mask_config=masking.BertMaskConfig(
                tokenizer=tokenizer,
                random_tokens=range(len(tokenizer.special_tokens), len(tokenizer.vocab)),
                mask_prob=mask_prob,
                mask_token_prob=mask_token_prob,
                random_token_prob=random_token_prob,
            ),
        )
        cls_token = tokenizer.token_to_id(tokenizer.cls_token) if prepend_cls_token else None
        if cls_token is not None or eos_token is not None:
            masked_tokens, labels, loss_mask = masking.add_cls_and_eos_tokens(
                sequence=masked_tokens,
                labels=labels,
                loss_mask=loss_mask,
                cls_token=cls_token,
                eos_token=eos_token,
            )

        # NeMo megatron assumes this return structure.
        return {
            "text": masked_tokens,
            "types": torch.zeros_like(masked_tokens, dtype=torch.int64),
            "attention_mask": torch.ones_like(masked_tokens, dtype=torch.int64),
            "labels": labels,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(masked_tokens, dtype=torch.int64),
        }


def _profile_sc_dataset():
    data_path = load("single_cell/testdata-20241203") / "cellxgene_2023-12-15_small_processed_scdl" / "train"
    preprocessor = GeneformerPreprocess(
        download_directory=data_path,
        medians_file_path=data_path / "medians.json",
        tokenizer_vocab_path=data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")
    scd = SingleCellDataset(data_path=data_path, tokenizer=tokenizer, median_dict=median_dict, max_len=2048, seed=321)
    n_epochs = 1
    len_dataset: int = len(scd)
    idxs = list(range(len_dataset * n_epochs))
    random.seed(315)
    random.shuffle(idxs)
    start = time.monotonic()  # Like time.time() but uses the CPU clock rather so subsequent calls will progress.
    for i in tqdm(idxs):
        _ = scd[EpochIndex(idx=i % len_dataset, epoch=i // len_dataset)]
    stop = time.monotonic()
    print(f"Processed {len_dataset * n_epochs} rows in {stop - start} seconds")


if __name__ == "__main__":
    # python -m bionemo.geneformer.data.singlecell.dataset will run this profile.
    _profile_sc_dataset()
