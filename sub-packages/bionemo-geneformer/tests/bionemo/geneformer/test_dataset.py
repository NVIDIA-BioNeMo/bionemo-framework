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


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pytest
import torch
from nemo.utils import logging

from bionemo.core.data.multi_epoch_dataset import EpochIndex
from bionemo.core.utils import random_utils
from bionemo.geneformer.data.singlecell.dataset import SingleCellDataset
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.testing.megatron_dataset_compatibility import assert_dataset_elements_not_equal


def test_load_sc_datasets(tmp_path, test_directory_feat_ids):
    tokenizer = MagicMock()
    sc_memmap_dataset_path0 = tmp_path / "test_data_0"
    ds_0 = SingleCellMemMapDataset(
        str(sc_memmap_dataset_path0), h5ad_path=str(test_directory_feat_ids / "adata_sample0.h5ad"), use_X_not_raw=True
    )  # create the memmap dataset format from h5ad for testing purposes
    dataset0 = SingleCellDataset(str(sc_memmap_dataset_path0), tokenizer)
    assert len(dataset0) == len(ds_0) == 8
    sc_memmap_dataset_path1 = tmp_path / "test_data_1"
    ds_1 = SingleCellMemMapDataset(
        str(sc_memmap_dataset_path1), h5ad_path=str(test_directory_feat_ids / "adata_sample1.h5ad"), use_X_not_raw=True
    )  # create the memmap dataset format from h5ad for testing purposes
    dataset1 = SingleCellDataset(str(sc_memmap_dataset_path1), tokenizer)
    assert len(dataset1) == len(ds_1) == 6
    sc_memmap_dataset_path2 = tmp_path / "test_data_2"
    ds_2 = SingleCellMemMapDataset(
        str(sc_memmap_dataset_path2), h5ad_path=str(test_directory_feat_ids / "adata_sample2.h5ad"), use_X_not_raw=True
    )  # create the memmap dataset format from h5ad for testing purposes
    dataset2 = SingleCellDataset(str(sc_memmap_dataset_path2), tokenizer)
    assert len(dataset2) == len(ds_2) == 100


def test_gene_not_in_tok_vocab(tmp_path, test_directory_feat_ids):
    sc_memmap_dataset_path0 = tmp_path / "test_data_0_sc_memmap"
    sc_h5ad_dataset_path0 = tmp_path / "test_data_0.h5ad"

    adata = ad.read_h5ad(test_directory_feat_ids / "adata_sample0.h5ad")
    synthetic_ids = [
        "ENSG00000243485",
        "ENSG00000186092",
        "ENSG00000238009",
        "ENSG00000239945",
        "ENSG00000241860",
        "ENSG00000241599",
        "ENSG00000286448",
        "ENSG00000236601",
        "ENSG00000235146",
        "ENSG00000229905",
    ]
    adata.var["feature_id"] = synthetic_ids
    adata.write(sc_h5ad_dataset_path0)
    SingleCellMemMapDataset(
        str(sc_memmap_dataset_path0), h5ad_path=str(sc_h5ad_dataset_path0), use_X_not_raw=True
    )  # create the memmap dataset format from h5ad for testing purposes
    preprocessor = GeneformerPreprocess(
        download_directory=str(sc_memmap_dataset_path0),
        medians_file_path=str(sc_memmap_dataset_path0 / "medians.json"),
        tokenizer_vocab_path=str(sc_memmap_dataset_path0 / "geneformer.vocab"),
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")

    dataset0 = SingleCellDataset(
        str(sc_memmap_dataset_path0), tokenizer, median_dict=median_dict, include_unrecognized_vocab_in_dataset=True
    )  # type: ignore
    index = EpochIndex(epoch=0, idx=3)
    with pytest.raises(ValueError) as error_info:
        dataset0.__getitem__(index)
    assert "not in the tokenizer vocab." in str(error_info.value)
    dataset0 = SingleCellDataset(
        str(sc_memmap_dataset_path0),
        tokenizer,
        median_dict=median_dict,
    )  # type: ignore

    item = dataset0.__getitem__(index)
    assert np.array(item["text"].tolist()) == [0]


def test_empty_gene_data_input(tmp_path, test_directory_feat_ids):
    sc_memmap_dataset_path0 = tmp_path / "test_data_0"
    SingleCellMemMapDataset(
        str(sc_memmap_dataset_path0), h5ad_path=str(test_directory_feat_ids / "adata_sample0.h5ad"), use_X_not_raw=True
    )  # create the memmap dataset format from h5ad for testing purposes
    preprocessor = GeneformerPreprocess(
        download_directory=str(sc_memmap_dataset_path0),
        medians_file_path=str(sc_memmap_dataset_path0 / "medians.json"),
        tokenizer_vocab_path=str(sc_memmap_dataset_path0 / "geneformer.vocab"),
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")
    dataset0 = SingleCellDataset(sc_memmap_dataset_path0, tokenizer, median_dict=median_dict)  # type: ignore
    index = EpochIndex(epoch=0, idx=1)
    with pytest.raises(ValueError) as error_info:
        dataset0.__getitem__(index)
    assert (
        "SingleCellMemap data provided is invalid; the gene expression data parsed for the specified index is empty."
        == str(error_info.value)
    )


def test_lookup_row(tmp_path, cellx_small_directory):
    tokenizer = MagicMock()
    dataset = SingleCellDataset(str(tmp_path / cellx_small_directory / "val"), tokenizer)
    values, feature_ids, _ = dataset.scdl.get_row(0, return_var_features=True, var_feature_names=["feature_id"])
    gene_data, col_idxs = values[0], values[1]
    assert len(gene_data) == 440
    assert len(col_idxs) == 440
    assert len(feature_ids[0]) == 60664

    values, feature_ids, _ = dataset.scdl.get_row(
        len(dataset) - 1, return_var_features=True, var_feature_names=["feature_id"]
    )
    gene_data, col_idxs = values[0], values[1]
    assert len(gene_data) == 1147
    assert len(col_idxs) == 1147
    assert len(feature_ids[0]) == 60664


def test_get_item_synthetic(tmp_path, test_directory_feat_ids):
    sc_memmap_dataset_path0 = tmp_path / "test_data_0"
    SingleCellMemMapDataset(
        sc_memmap_dataset_path0, h5ad_path=test_directory_feat_ids / "adata_sample0.h5ad", use_X_not_raw=True
    )  # create the memmap dataset format from h5ad for testing purposes
    preprocessor = GeneformerPreprocess(
        download_directory=sc_memmap_dataset_path0,
        medians_file_path=sc_memmap_dataset_path0 / "medians.json",
        tokenizer_vocab_path=sc_memmap_dataset_path0 / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")
    dataset0 = SingleCellDataset(
        str(sc_memmap_dataset_path0),
        tokenizer,
        median_dict=median_dict,
        mask_token_prob=0,
        mask_prob=0,
        random_token_prob=0,
    )  # type: ignore
    index = EpochIndex(epoch=0, idx=0)
    item = dataset0.__getitem__(index)
    assert np.all(np.array(item["text"]) == np.array([0, 10]))
    assert np.all(np.array(item["types"]) == np.array([0, 0]))
    assert np.all(np.array(item["attention_mask"]) == np.array([1, 1]))
    assert np.all(np.array(item["labels"]) == np.array([-1, -100]))
    assert np.all(np.array(item["loss_mask"]) == np.array([False, False]))
    assert np.all(np.array(item["is_random"]) == np.array([0, 0]))


def test_GeneformerDataset_changes_with_epoch(tmp_path, cellx_small_directory):
    preprocessor = GeneformerPreprocess(
        download_directory=str(tmp_path / cellx_small_directory / "val"),
        medians_file_path=str(tmp_path / cellx_small_directory / "val" / "medians.json"),
        tokenizer_vocab_path=str(tmp_path / cellx_small_directory / "val" / "geneformer.vocab"),
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")
    genformer_ds = SingleCellDataset(
        str(tmp_path / cellx_small_directory / "val"),
        tokenizer,  # type: ignore
        median_dict=median_dict,  # type: ignore
    )  # type: ignore

    index_0 = EpochIndex(epoch=0, idx=0)
    index_1 = EpochIndex(epoch=1, idx=0)

    # Tests megatron validity (subsequent calls to the same index produce the same result) and epoch non-determinism
    assert_dataset_elements_not_equal(genformer_ds, index_0, index_1)


def test_get_item_cellx(tmp_path, cellx_small_directory):
    preprocessor = GeneformerPreprocess(
        download_directory=str(tmp_path / cellx_small_directory / "val"),
        medians_file_path=str(tmp_path / cellx_small_directory / "val" / "medians.json"),
        tokenizer_vocab_path=str(tmp_path / cellx_small_directory / "val" / "geneformer.vocab"),
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")
    ds = SingleCellDataset(
        str(tmp_path / cellx_small_directory / "val"),
        tokenizer,  # type: ignore
        median_dict=median_dict,  # type: ignore
        mask_prob=0,
        mask_token_prob=0,
        random_token_prob=0,
    )  # type: ignore
    index = EpochIndex(epoch=0, idx=2)
    item = ds.__getitem__(index)
    expected_output_first = np.array(
        [
            0,
            20502,
            15942,
            8191,
            2701,
            16227,
            8932,
            14368,
            5209,
            11346,
            10122,
            8806,
            530,
            8016,
            7788,
            6755,
            10695,
            5767,
            12231,
            3813,
            8639,
            11447,
            17704,
            20034,
            16715,
            3141,
            12632,
            18986,
            8715,
            16351,
            11897,
            3672,
            3364,
            2453,
            3833,
            6925,
            12089,
            6396,
            257,
            3951,
            14400,
            9758,
            6860,
            6267,
            467,
            11899,
            5070,
            8870,
            3974,
            3084,
            10804,
            2187,
            2346,
            17722,
            11845,
            11551,
            16387,
            12822,
            18577,
            10201,
            1955,
            2744,
            10991,
            11911,
            7822,
            20491,
            1078,
            2552,
            12177,
            6716,
            9503,
            10404,
            12220,
            8298,
            8471,
            4092,
            6885,
            2386,
            16454,
            5641,
            8417,
            12754,
            18000,
            154,
            15484,
            8458,
            2964,
            4217,
            469,
            3058,
            19800,
            5816,
            8309,
            17681,
            16909,
            9566,
            18037,
            17578,
            1634,
            11592,
        ]
    )
    expected_output_last = np.array(
        [
            4502,
            1145,
            12212,
            3667,
            14669,
            811,
            8670,
            2291,
            1986,
            10551,
            4544,
            15361,
            7906,
            12532,
            4719,
            1336,
            12062,
            16414,
            3438,
            12258,
            10295,
            3008,
            14606,
            19632,
            12418,
            12655,
            12185,
            235,
            12018,
            7505,
            11927,
            653,
            887,
            12533,
            1686,
            7289,
            103,
            17298,
            5611,
            20504,
            6552,
            8305,
            1436,
            4883,
            5578,
            708,
            20343,
            4390,
            6241,
            2563,
            16300,
            20888,
            1873,
            10956,
            4491,
            9515,
            2403,
            6269,
            14978,
            4828,
            12412,
            16728,
            9665,
            5084,
            3781,
            6255,
            8568,
            14059,
            6564,
            1629,
            758,
            14814,
            9749,
            15807,
            17317,
            6657,
            3829,
            7196,
            7329,
            2347,
            4812,
            1052,
            3615,
            13011,
            12175,
            10948,
            611,
            13008,
            8255,
            13747,
            8519,
            4764,
            13814,
            10324,
            14631,
            6182,
            7248,
            16740,
            6386,
            11411,
        ]
    )
    assert all(np.array(item["text"][:100]) == expected_output_first)
    assert all(np.array(item["text"][-100:]) == expected_output_last)
    assert np.array(item["labels"])[0] == -1
    assert np.all(np.array(item["labels"][1:]) == -100)


def test_dataset_process_item():
    tokenizer = MagicMock()

    tokenizer.pad_token = "pad"
    tokenizer.cls_token = "cls"
    tokenizer.mask_token = "mask"
    tokenizer.ukw_token = "ukn"
    tokenizer.gene_tok_to_ens = lambda x: x
    tokenizer.mask_token_id = 6

    # Need this to mock the underlying dictionary behavior with arbitrary keys
    class gene_to_ens:
        @staticmethod
        def get(x, other):
            return x

    tokenizer.gene_to_ens = gene_to_ens
    tokenizer.vocab = {"GENE0": 1, "GENE1": 2, "GENE2": 3, "ukn": 7, "mask": 6, "cls": 5, "pad": 4}

    def tok_to_id(tok):
        if tok == tokenizer.pad_token:
            return 4
        if tok == tokenizer.cls_token:
            return 5
        if tok == tokenizer.mask_token:
            return 6
        if tok == tokenizer.ukw_token:
            return 7
        if tok == "GENE0":
            return 1
        if tok == "GENE1":
            return 2
        if tok == "GENE2":
            return 3

    tokenizer.token_to_id = tok_to_id
    # Create a sample input item
    input_item = {
        "expression": np.array([1, 2, 3]),
        "indices": np.array([0, 1, 2]),
        "metadata": np.array([f"GENE{i}" for i in range(3)]),
    }

    # Process the input item
    from bionemo.geneformer.data.singlecell.dataset import process_item

    seed = 42
    rng = np.random.default_rng(seed)
    seed = random_utils.get_seed_from_rng(rng)
    idx = 0
    rng = np.random.default_rng([seed, idx])

    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        max_len=5,
        mask_prob=0,
        rng=rng,
    )
    assert all(processed_item["text"] == torch.tensor([5, 3, 2, 1]))  # CLS, 1, 2, 3, but in reverse order
    # The following is used as 'attention_mask' in NeMo, so it's probably the opposite of what you think it should be.
    assert all(processed_item["attention_mask"] == torch.tensor([1, 1, 1, 1]))  # this is all 1s

    ###### Check median rank norm, sorts in ascending order. ######

    # 1/6/1=1/6 , 2/3/6 =2/18=1/9, 3/6/6 =3/36=1/12 => 3, 2, 1
    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 3, "GENE2": 6},
        max_len=4,
        mask_prob=0,
        target_sum=1,
        rng=rng,
    )
    assert all(processed_item["text"] == torch.tensor([5, 1, 2, 3]))

    # Checks median norm, should change the order due to medians.
    # 1/6/.5=1/3, 2/6/1=2/6=1/3, 3/6/2=3/12=1/4
    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 0.5, "GENE1": 1, "GENE2": 2},
        max_len=4,
        mask_prob=0,
        target_sum=1,
        rng=rng,
    )
    assert all(processed_item["text"] == torch.tensor([5, 1, 2, 3]))

    #    Masking - test that no special tokens are masked, all when 100, none when 0
    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        random_token_prob=0,
        max_len=5,
        mask_prob=1.0,
        mask_token_prob=1.0,
        target_sum=1,
        rng=rng,
    )
    # NOTE: we need to set masked tokens to MASK so that they are decoded.
    assert all(processed_item["text"] == torch.tensor([5, 6, 6, 6]))  # CLS, MASK, MASK, MASK
    # NOTE: MASKed tokens are the only ones used by loss
    assert all(processed_item["loss_mask"] == torch.tensor([False, True, True, True]))  # NO, MASK, MASK, MASK, NO
    # the ARBITRARY labels should be ignored due to loss mask.
    assert all(processed_item["labels"] == torch.tensor([-1, 3, 2, 1]))  # ARBITRARY, 3, 2, 1, ARBITRARY
    assert all(processed_item["is_random"] == 0)  # For now we don't support random masking.

    # checks sequence is truncated for a long sequence
    processed_item = process_item(
        input_item["expression"],
        input_item["indices"],
        input_item["metadata"],
        tokenizer,
        gene_median={"GENE0": 1, "GENE1": 1, "GENE2": 1},
        max_len=3,
        mask_prob=0,
        target_sum=1,
        rng=rng,
    )
    # Randomly permutes the other values, no fixed order
    assert processed_item["text"][0] == torch.tensor([5])
    # Truncate to exactly three items
    assert len(processed_item["text"]) == 3
    assert all(processed_item["loss_mask"] == torch.tensor([False, False, False]))  # No mask applied
