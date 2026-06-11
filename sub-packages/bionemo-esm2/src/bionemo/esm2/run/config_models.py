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


import importlib
from pathlib import Path
from typing import Optional, Type

import torch
from nemo.utils import logging
from pydantic import field_serializer, field_validator, model_validator

from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.model import ESM2Config
from bionemo.llm.model.biobert.model import BiobertSpecOption
from bionemo.llm.run.config_models import (
    DataConfig,
    ExposedModelConfig,
    MainConfig,
    deserialize_str_to_path,
    serialize_path_or_str,
)


class ESM2DataConfig(DataConfig[ESMDataModule]):
    """ESM2DataConfig is a configuration class for setting up the pre-training data module for ESM2.

    The ESM2DataModule implements the cluster oriented sampling method defined in the ESM2 publication.

    Attributes:
        train_cluster_path (Path): Path to the training cluster data.
        train_database_path (Path): Path to the training database.
        valid_cluster_path (Path): Path to the validation cluster data.
        valid_database_path (Path): Path to the validation database.
        micro_batch_size (int): Size of the micro-batch. Default is 8.
        result_dir (str): Directory to store results. Default is "./results".
        min_seq_length (int): Minimum sequence length. Default is 128.
        max_seq_length (int): Maximum sequence length. Default is 128.
        random_mask_strategy (RandomMaskStrategy): Strategy for random masking. Default is RandomMaskStrategy.ALL_TOKENS.
        num_dataset_workers (int): Number of workers for the dataset. Default is 0.

    Methods:
        construct_data_module(global_batch_size: int) -> ESMDataModule:
            Constructs and returns an ESMDataModule instance with the provided global batch size.
    """

    train_cluster_path: Path
    train_database_path: Path
    valid_cluster_path: Path
    valid_database_path: Path

    micro_batch_size: int = 8
    result_dir: str | Path = "./results"
    min_seq_length: int = 128
    max_seq_length: int = 128
    random_mask_strategy: RandomMaskStrategy = RandomMaskStrategy.ALL_TOKENS
    num_dataset_workers: int = 0

    @field_serializer(
        "train_cluster_path", "train_database_path", "valid_cluster_path", "valid_database_path", "result_dir"
    )
    def serialize_paths(self, value: Path) -> str:  # noqa: D102
        return serialize_path_or_str(value)

    @field_validator(
        "train_cluster_path", "train_database_path", "valid_cluster_path", "valid_database_path", "result_dir"
    )
    def deserialize_paths(cls, value: str) -> Path:  # noqa: D102
        return deserialize_str_to_path(value)

    @field_serializer("random_mask_strategy")
    def serialize_spec_option(self, value: RandomMaskStrategy) -> str:  # noqa: D102
        return value.value

    @field_validator("random_mask_strategy", mode="before")
    def deserialize_spec_option(cls, value: str) -> RandomMaskStrategy:  # noqa: D102
        return RandomMaskStrategy(value)

    def construct_data_module(self, global_batch_size: int) -> ESMDataModule:
        """Constructs and returns an ESMDataModule instance with the provided global batch size.

        This method provides means for constructing the datamodule, any pre-requisites for the DataModule should be
        aquired here. For example, tokenizers, preprocessing, may want to live in this method.

        Args:
            global_batch_size (int): Global batch size for the data module. Global batch size must be a function of
                parallelism settings and the `micro_batch_size` attribute. Since the DataConfig has no ownership over
                parallelism configuration, we expect someone higher up on the ownership chain to provide the value to
                this method.

        """
        tokenizer = get_tokenizer()
        data = ESMDataModule(
            train_cluster_path=self.train_cluster_path,
            train_database_path=self.train_database_path,
            valid_cluster_path=self.valid_cluster_path,
            valid_database_path=self.valid_database_path,
            global_batch_size=global_batch_size,
            micro_batch_size=self.micro_batch_size,
            min_seq_length=self.min_seq_length,
            max_seq_length=self.max_seq_length,
            num_workers=self.num_dataset_workers,
            random_mask_strategy=self.random_mask_strategy,
            tokenizer=tokenizer,
        )
        return data


class ExposedESM2PretrainConfig(ExposedModelConfig[ESM2Config]):
    """Configuration class for ESM2 pretraining with select exposed parameters.

    See the inherited ExposedModelConfig for attributes and methods from the base class. Use this class either
    as a template or extension for custom configurations. Importantly, these kinds of classes should do two things,
    select attributes to expose to the user, and provide validation and serialization any attributes.

    Attributes:
        use_esm_attention (bool): Flag to skip ESM2 custom attention for TE acceleration. Defaults to False.
        token_dropout (bool): Flag to enable token dropout. Defaults to True.
        normalize_attention_scores (bool): Flag to normalize attention scores. Defaults to False.
        variable_seq_lengths (bool): Flag to enable variable sequence lengths. Defaults to False.
        core_attention_override (Optional[Type[torch.nn.Module]]): Optional override for core attention module. Defaults to None.

    Methods:
        restrict_biobert_spec_to_esm2(cls, biobert_spec_option: BiobertSpecOption) -> BiobertSpecOption:
            Validates the BiobertSpecOption to ensure it is compatible with ESM2.
        serialize_core_attention_override(self, value: Optional[Type[torch.nn.Module]]) -> Optional[str]:
            Serializes the core attention override module to a string.
        validate_core_attention_override(cls, value):
            Validates the core attention override module, ensuring it is a subclass of torch.nn.Module.
        validate_and_set_attention_and_scaling(self):
            Validates and sets the attention and scaling parameters based on the biobert_spec_option.
        model_validator(self, global_cfg: MainConfig) -> MainConfig:
            Validates the global configuration, ensuring compatibility with ESM2DataConfig and parallel settings.
        model_class(self) -> Type[ESM2Config]:
            Returns the model class associated with this configuration.
    """

    use_esm_attention: bool = False  # Skip ESM2 custom attention for TE acceleration. Still passes golden value test.
    token_dropout: bool = True
    normalize_attention_scores: bool = False
    variable_seq_lengths: bool = False
    core_attention_override: Type[torch.nn.Module] | None = None

    @field_serializer("core_attention_override")
    def serialize_core_attention_override(self, value: Optional[Type[torch.nn.Module]]) -> Optional[str]:
        """Serializes the core attention override module to a string."""
        if value is None:
            return None
        return f"{value.__module__}.{value.__name__}"

    @field_validator("core_attention_override", mode="before")
    def validate_core_attention_override(cls, value):
        """Validates the core attention override module, ensuring it is a subclass of torch.nn.Module."""
        if value is None:
            return None
        if isinstance(value, str):
            module_name, class_name = value.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                if not issubclass(cls, torch.nn.Module):
                    raise ValueError(f"{cls} is not a subclass of torch.nn.Module")
                return cls
            except (ImportError, AttributeError):
                raise ValueError(f"Cannot import {value}")
        return value

    @model_validator(mode="after")
    def validate_and_set_attention_and_scaling(self):
        """Validates and sets the attention and scaling parameters based on the biobert_spec_option."""
        logging.info(
            "Mutating apply_query_key_layer_scaling and core_attention_override based on biobert_spec_option.."
        )
        if self.biobert_spec_option == BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec:
            self.apply_query_key_layer_scaling = False
        elif self.biobert_spec_option == BiobertSpecOption.esm2_bert_layer_local_spec:
            logging.warning(
                "BiobertSpecOption.esm2_bert_layer_local_spec is deprecated. "
                "Use BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec instead."
            )
            self.apply_query_key_layer_scaling = True
        return self

    def model_validator(self, global_cfg: MainConfig) -> MainConfig:
        """Validates the global configuration, ensuring compatibility with ESM2DataConfig and parallel settings.

        The global validator acts on the MainConfig, this couples together the ESM2DataConfig with ESM2PretrainingConfig.
        Additionally, it provides validation for sequence length and parallelism settings.

        Args:
            global_cfg (MainConfig): The global configuration object.
        """
        global_cfg = super().model_validator(global_cfg)
        # Need to ensure that at the least we have access to min_seq_length and max_seq_length
        if not isinstance(global_cfg.data_config, ESM2DataConfig):
            raise TypeError(f"ESM2PretrainConfig requires ESM2DataConfig, got {global_cfg.data_config=}")

        pipeline_model_parallel_size, tensor_model_parallel_size = (
            global_cfg.parallel_config.pipeline_model_parallel_size,
            global_cfg.parallel_config.tensor_model_parallel_size,
        )
        min_seq_length, max_seq_length = global_cfg.data_config.min_seq_length, global_cfg.data_config.max_seq_length
        assert self.variable_seq_lengths == (
            pipeline_model_parallel_size * tensor_model_parallel_size > 1 and min_seq_length != max_seq_length
        ), (
            "Must set variable_seq_lengths to True when min_seq_length != max_seq_length under pipeline or tensor parallelism."
        )
        return global_cfg

    def model_class(self) -> Type[ESM2Config]:
        """Returns the model class associated with this configuration."""
        return ESM2Config
