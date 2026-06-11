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


import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Sequence

from nemo.utils import logging

from bionemo.geneformer.data.preprocess import ResourcePreprocessor
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.utils.remote import RemoteResource


__all__: Sequence[str] = (
    "GeneformerPreprocess",
    "GeneformerResourcePreprocessor",
)


@dataclass
class GeneformerResourcePreprocessor(ResourcePreprocessor):
    """ResourcePreprocessor for the Geneformer model. Downloads the gene_name_id_dict.pkl and gene_median_dictionary.pkl files."""

    dest_directory: str = "geneformer"

    def get_remote_resources(self) -> List[RemoteResource]:  # noqa: D102
        url_fn = {
            "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl?download=true": "gene_name_id_dict.pkl",
            "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl?download=true": "gene_median_dictionary.pkl",
        }

        resources = []
        for url, filename in url_fn.items():
            resource = RemoteResource(
                dest_directory=self.dest_directory,
                dest_filename=filename,
                root_directory=self.root_directory,
                checksum=None,
                url=url,
            )
            resources.append(resource)
        return resources

    def prepare_resource(self, resource: RemoteResource) -> str:
        """Logs and downloads the passed resource.

        resource: RemoteResource - Resource to be prepared.

        Returns - the absolute destination path for the downloaded resource
        """
        return resource.download_resource()

    def prepare(self):  # noqa: D102
        return [self.prepare_resource(resource) for resource in self.get_remote_resources()]


class GeneformerPreprocess:  # noqa: D101
    def __init__(self, download_directory: Path, medians_file_path: Path, tokenizer_vocab_path: Path):
        """Downloads HGNC symbols

        preproc_dir (str): Directory to store the reference preproc in
        tokenizer_vocab_path (str): Filepath to store the tokenizer vocab
        dataset_conf (OmegaConf): has 'train', 'val', 'test' keys containing
            the names of preprocessed train/val/test files to use for training.
        """  # noqa: D415
        self.download_directory = download_directory
        self.medians_file_path = medians_file_path
        self.tokenizer_vocab_path = tokenizer_vocab_path
        self._validate_tokenizer_args(
            self.tokenizer_vocab_path,
        )

    def build_and_save_tokenizer(self, median_dict, gene_to_ens, vocab_output_name):
        """Builds the GeneTokenizer using the median dictionary
        then serializes and saves the dictionary to disk.
        """  # noqa: D205
        tokenizer = GeneTokenizer.from_medians_and_genes_dicts(median_dict, gene_to_ens)
        tokenizer.save_vocab(vocab_output_name)
        return tokenizer

    def _validate_tokenizer_args(self, vocab_output_name):
        vocab_exists = os.path.exists(vocab_output_name)
        if vocab_exists:
            logging.warning(f"Tokenizer vocab file: {vocab_output_name} already exists. Overwriting...")

    def preprocess(self) -> dict[Literal["tokenizer", "median_dict"], Any]:
        """Preprocesses for the Geneformer model"""  # noqa: D415
        gene_name_dict_fn, gene_median_dict_fn = GeneformerResourcePreprocessor(
            dest_directory=self.download_directory,
        ).prepare()

        # Load artifacts
        with open(gene_name_dict_fn, "rb") as fd:
            gene_ens = pickle.load(fd)

        with open(gene_median_dict_fn, "rb") as fd:
            median_dict = pickle.load(fd)

        # Save converted artifacts to JSON to prevent pickle issues.
        medians_dir = os.path.dirname(self.medians_file_path)
        if not os.path.exists(medians_dir):
            os.makedirs(medians_dir, exist_ok=True)  # ensure the dir exists but be ok with race conditions.
        with open(self.medians_file_path, "w") as fp:
            json.dump(median_dict, fp)

        if self.tokenizer_vocab_path is not None:
            tokenizer = self.build_and_save_tokenizer(
                median_dict,
                gene_ens,
                self.tokenizer_vocab_path,
            )
        else:
            tokenizer = None

        return {"tokenizer": tokenizer, "median_dict": median_dict}
