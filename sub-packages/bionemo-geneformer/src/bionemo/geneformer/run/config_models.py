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

import pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Type

from nemo.utils import logging
from pydantic import field_serializer, field_validator
from tokenizers import Tokenizer

from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.model.finetune_token_regressor import FineTuneSeqLenBioBertConfig
from bionemo.llm.run.config_models import (
    DataConfig,
    ExposedModelConfig,
    deserialize_str_to_path,
    serialize_path_or_str,
)


@dataclass
class GeneformerDataArtifacts:
    """Data artifacts produced by the geneformer preprocess."""

    tokenizer: Tokenizer
    median_dict: dict


class GeneformerPretrainingDataConfig(DataConfig[SingleCellDataModule]):
    """Configuration class for Geneformer pretraining data.

    Expects train/test/val to be prior split by directory and processed by `sub-packages/bionemo-scdl/src/bionemo/scdl/scripts/convert_h5ad_to_scdl.py`.

    Attributes:
        data_dir (str): Directory where the data is stored.
        result_dir (str | pathlib.Path): Directory where the results will be stored. Defaults to "./results".
        micro_batch_size (int): Size of the micro-batch. Defaults to 8.
        seq_length (int): Sequence length for the data. Defaults to 2048.
        num_dataset_workers (int): Number of workers for data loading. Defaults to 0.

    Properties:
        train_data_path (str): Path to the training data.
        val_data_path (str): Path to the validation data.
        test_data_path (str): Path to the test data.

    Methods:
        geneformer_preprocess() -> GeneformerDataArtifacts:
            Preprocesses the data using a legacy preprocessor from BioNeMo 1 and returns the necessary artifacts.
        construct_data_module(global_batch_size: int) -> SingleCellDataModule:
            Constructs and returns a SingleCellDataModule using the preprocessed data artifacts.
    """

    # Shadow two attributes from the parent for visibility.
    data_dir: str
    result_dir: str | pathlib.Path = "./results"
    micro_batch_size: int = 8

    seq_length: int = 2048
    num_dataset_workers: int = 0

    @field_serializer("result_dir")
    def serialize_paths(self, value: pathlib.Path) -> str:  # noqa: D102
        return serialize_path_or_str(value)

    @field_validator("result_dir")
    def deserialize_paths(cls, value: str) -> pathlib.Path:  # noqa: D102
        return deserialize_str_to_path(value)

    @property
    def train_data_path(self) -> str:  # noqa: D102
        return self.data_dir + "/train"

    @property
    def val_data_path(self) -> str:  # noqa: D102
        return self.data_dir + "/val"

    @property
    def test_data_path(self) -> str:  # noqa: D102
        return self.data_dir + "/test"

    def geneformer_preprocess(self) -> GeneformerDataArtifacts:
        """Geneformer datamodule expects certain artifacts to be present in the data directory.

        This method uses a legacy 'preprocessor' from BioNeMo 1 to acquire the associated artifacts.
        """
        preprocessor = GeneformerPreprocess(
            download_directory=pathlib.Path(self.train_data_path),
            medians_file_path=pathlib.Path(self.train_data_path + "/medians.json"),
            tokenizer_vocab_path=pathlib.Path(self.train_data_path + "/geneformer.vocab"),
        )
        result = preprocessor.preprocess()
        if "tokenizer" in result and "median_dict" in result:
            logging.info("*************** Preprocessing Finished ************")
            return GeneformerDataArtifacts(tokenizer=result["tokenizer"], median_dict=result["median_dict"])
        else:
            logging.error("Preprocessing failed.")
            raise ValueError("Preprocessing failed to create tokenizer and/or median dictionary.")

    def construct_data_module(self, global_batch_size: int) -> SingleCellDataModule:
        """Downloads the requisite data artifacts and instantiates the DataModule."""
        geneformer_data_artifacts: GeneformerDataArtifacts = self.geneformer_preprocess()
        data = SingleCellDataModule(
            seq_length=self.seq_length,
            tokenizer=geneformer_data_artifacts.tokenizer,
            train_dataset_path=self.train_data_path,
            val_dataset_path=self.val_data_path,
            test_dataset_path=self.test_data_path,
            random_token_prob=0.02,
            median_dict=geneformer_data_artifacts.median_dict,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=global_batch_size,
            persistent_workers=self.num_dataset_workers > 0,
            pin_memory=False,
            num_workers=self.num_dataset_workers,
        )
        return data


class ExposedGeneformerPretrainConfig(ExposedModelConfig[GeneformerConfig]):
    """Exposes custom parameters for pretraining and binds the class to GeneformerConfig.

    Attributes:
        initial_ckpt_path (str): Path to a directory containing checkpoint files for initializing the model. This is only
        initial_ckpt_skip_keys_with_these_prefixes (List[str]): Skip any layer that contains this key during restoration. Useful for finetuning, set the names of the task heads so checkpoint restoration does not errorniously try to restore these.
    """

    # Custom parameters for FineTuning
    initial_ckpt_path: Optional[str] = None
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=list)

    def model_class(self) -> Type[GeneformerConfig]:  # noqa: D102
        return GeneformerConfig


class ExposedFineTuneSeqLenBioBertConfig(ExposedModelConfig[FineTuneSeqLenBioBertConfig]):
    """Config for models that fine-tune a BioBERT model from a pre-trained checkpoint.

    Parameters:
        initial_ckpt_path - path to a directory containing checkpoint files for initializing the model. This is only
            required on the first execution of the model, any restored checkpoints should skip this step.
        initial_ckpt_skip_keys_with_these_prefixes - skip any layer that contains this key during restoration. Useful
            for ignoring extra additional layers used for finetuning. Layers with these keys are then randomly initialized.
    """

    # Custom parameters for FineTuning
    initial_ckpt_path: Optional[str] = None
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: ["regression_head"])

    def model_class(self) -> Type[FineTuneSeqLenBioBertConfig]:
        """Binds the class to FineTuneSeqLenBioBertConfig."""
        return FineTuneSeqLenBioBertConfig
