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
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, Callable, Dict, Generic, List, Literal, Optional, Type, TypeVar

import lightning.pytorch as pl
import torch
from pydantic import BaseModel, field_serializer, field_validator, model_validator
from torch.nn import functional as F

from bionemo.core.utils import dtypes
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.utils.logger_utils import WandbConfig


ModelConfigT = TypeVar("ModelConfigT", bound=BioBertConfig)
DataModuleT = TypeVar("DataModuleT", bound=pl.LightningDataModule)

# Activation functions not available in torch.nn.functional require custom serialization/validation. Add them here with a lookup key.
CUSTOM_ACTIVATION_FNS: Dict[str, Callable[[torch.Tensor, Any], torch.Tensor]] = {}

# DO NOT use keys that already exist in torch.nn.functional, as the torch.nn.functional functions are selected first.
for key in CUSTOM_ACTIVATION_FNS:
    assert key not in dir(torch.nn.functional), f"Key {key} already exists in torch.nn.functional"

# It does not matter if values are duplicated as the key=>value mapping still does the right thing. Repeat values should be considered aliases.
REVERSE_CUSTOM_ACTIVATION_FNS: Dict[Callable[[torch.Tensor, Any], torch.Tensor], str] = {
    v: k for k, v in CUSTOM_ACTIVATION_FNS.items()
}


def deserialize_str_to_path(path: str) -> pathlib.Path:
    """General purpose deserialize for string/path objects. Since YAML has no native representation for pathlib.Path, we serialize to strings. Import this method as a @field_validator."""
    return pathlib.Path(path)


def serialize_path_or_str(path: str | pathlib.Path) -> str:
    """General purpose serialization for string/path objects. Since YAML has no native representation for pathlib.Path, we serialize to strings. Import this method as a @field_serializer."""
    if isinstance(path, pathlib.Path):
        return str(path)
    elif isinstance(path, str):
        return path
    else:
        raise ValueError(f"Expected str or pathlib.Path, got {type(path)}")


class DataConfig(BaseModel, Generic[DataModuleT], ABC):
    """Base class for all data configurations.

    This class is used to define the interface for all data configurations. It is used to define the data module that
    will be used in the training loop.
    """

    micro_batch_size: int = 8
    result_dir: str | pathlib.Path = "./results"
    num_dataset_workers: int = 0
    seq_length: int = 128

    @field_serializer("result_dir")
    def serialize_paths(self, value: pathlib.Path) -> str:  # noqa: D102
        return serialize_path_or_str(value)

    @field_validator("result_dir")
    def deserialize_paths(cls, value: str) -> pathlib.Path:  # noqa: D102
        return deserialize_str_to_path(value)

    @abstractmethod
    def construct_data_module(self, global_batch_size: int) -> DataModuleT:
        """Construct the data module from the configuration. Cannot be defined generically."""
        ...

    def custom_model_validator(self, global_cfg: "MainConfig") -> "MainConfig":
        """Use custom implementation of this method to define the things inside global_config.

        The following expression will always be true:

        global_cfg.data_config == self
        """
        return global_cfg


class ExposedModelConfig(BaseModel, Generic[ModelConfigT], ABC):
    """BioNeMo model configuration class, wraps TransformerConfig and friends.

    This class is used to define the interface for all model configurations. It is **Exposed** to guard against ill-typed
    or poorly defined fields in the underlying configuration objects. `ModelConfigT` declares the associated type of the
    underlying config (most commonly a BioBertGenericConfig, but could also be a TransformerConfig or something similar).
    Children should try to expose the minimal set of fields necessary for the user to configure the model while keeping
    the more esoteric configuration private to the underlying ModelConfigT.
    """

    # Restores weights from a pretrained checkpoint
    initial_ckpt_path: Optional[str] = None
    # Does not attempt to load keys with these prefixes (useful if you attached extra parameters and still want to load a set of weights)
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=list)

    # Pydantic stuff to allow arbitrary types + validators + serializers
    class Config:  # noqa: D106
        arbitrary_types_allowed = True

    def model_class(self) -> Type[ModelConfigT]:
        """Returns the underlying model class that this config wraps."""
        raise NotImplementedError

    def custom_model_validator(self, global_cfg: "MainConfig") -> "MainConfig":
        """Use custom implementation of this method to define the things inside global_config.

        The following expression will always be true:

        global_cfg.bionemo_model_config == self
        """
        return global_cfg

    def exposed_to_internal_bionemo_model_config(self) -> ModelConfigT:
        """Converts the exposed dataclass to the underlying Transformer config.

        The underlying ModelConfigT may both be incomplete and unserializable. We use this transformation as a way to
        hide fields that are either not serializable by Pydantic or that we do not want to expose.
        """
        cls: Type[ModelConfigT] = self.model_class()
        model_dict = {}
        for attr in self.model_fields:
            if attr not in model_dict and attr in cls.__dataclass_fields__:
                model_dict[attr] = getattr(self, attr)

        # Now set fp16 and bf16 based on the precision for the underlying TransformerConfig=>ParallelConfig
        #   the only constraint is that both must not be true.
        model_dict["bf16"] = self.pipeline_dtype == dtypes.precision_to_dtype["bf16-mixed"]
        model_dict["fp16"] = self.pipeline_dtype == dtypes.precision_to_dtype["16-mixed"]
        result = cls(**model_dict)

        return result

    # NOTE: See PrecisionTypes for a list of valid literals that may be deserialized.
    params_dtype: torch.dtype
    pipeline_dtype: torch.dtype
    autocast_dtype: torch.dtype

    num_layers: int = 6
    hidden_size: int = 256
    ffn_hidden_size: int = 512
    num_attention_heads: int = 4
    seq_length: int = 512
    fp32_residual_connection: bool = False
    hidden_dropout: float = 0.02
    init_method_std: float = 0.02
    kv_channels: Optional[int] = None
    apply_query_key_layer_scaling: bool = False
    make_vocab_size_divisible_by: int = 128
    masked_softmax_fusion: bool = True
    fp16_lm_cross_entropy: bool = False
    gradient_accumulation_fusion: bool = False
    layernorm_zero_centered_gamma: bool = False
    layernorm_epsilon: float = 1.0e-12
    activation_func: Callable[[torch.Tensor, Any], torch.Tensor] = F.gelu
    qk_layernorm: bool = False
    apply_residual_connection_post_layernorm: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    get_attention_mask_from_fusion: bool = False
    attention_dropout: float = 0.1
    share_embeddings_and_output_weights: bool = True
    enable_autocast: bool = False
    nemo1_ckpt_path: Optional[str] = None
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec

    @field_serializer("biobert_spec_option")
    def serialize_spec_option(self, value: BiobertSpecOption) -> str:  # noqa: D102
        return value.value

    @field_validator("biobert_spec_option", mode="before")
    def deserialize_spec_option(cls, value: str) -> BiobertSpecOption:  # noqa: D102
        return BiobertSpecOption(value)

    @field_validator("activation_func", mode="before")
    @classmethod
    def validate_activation_func(cls, activation_func: str) -> Callable:
        """Validates the activation function, assumes this function exists in torch.nn.functional.

        For custom activation functions, use the CUSTOM_ACTIVATION_FUNCTIONS dictionary in the module. This method
        validates the provided activation function string and returns a callable function based on the validation
        context using the provided validator in the base class.

        Args:
            activation_func (str): The activation function to be validated.
            context (ValidationInfo): The context for validation.

        Returns:
            Callable: A callable function after validation.

        See Also:
            CUSTOM_ACTIVATION_FNS
        """
        func = getattr(torch.nn.functional, activation_func.lower(), None)
        if func is None and activation_func in CUSTOM_ACTIVATION_FNS:
            func = CUSTOM_ACTIVATION_FNS[activation_func]
            return func
        elif func is None:
            raise ValueError(
                f"activation_func must be a valid function in `torch.nn.functional`, got {activation_func=}"
            )
        else:
            return func

    @field_serializer("activation_func")
    def serialize_activation_func(self, v: Callable[[torch.Tensor, Any], torch.Tensor]) -> str:
        """Serializes a given activation function to its corresponding string representation.

        By default, all activation functions from `torch.nn.functional` are serialized to their name. User defined
        activation functions should also be defined here with a custom mapping in CUSTOM_ACTIVATION_FNS defined at the
        top of this file. This allows our Pydantic model to serialize and deserialize the activation function.

        Args:
            v (Callable[[torch.Tensor, Any], torch.Tensor]): The activation function to serialize.

        Returns:
            str: The name of the activation function if it is a standard PyTorch function,
                 or the corresponding serialization key if it is a custom activation function.

        Raises:
            ValueError: If the activation function is not supported.
        """
        func_name = v.__name__
        func = getattr(torch.nn.functional, func_name, None)
        if func is not None:
            return func_name
        elif func in REVERSE_CUSTOM_ACTIVATION_FNS:
            return REVERSE_CUSTOM_ACTIVATION_FNS[func]  # Get the serialization key
        else:
            raise ValueError(f"Unsupported activation function: {v}")

    @field_validator("params_dtype", "pipeline_dtype", "autocast_dtype", mode="before")
    @classmethod
    def precision_validator(cls, v: dtypes.PrecisionTypes) -> torch.dtype:
        """Validates the precision type and returns the corresponding torch dtype."""
        return dtypes.get_autocast_dtype(v)

    @field_serializer("params_dtype", "pipeline_dtype", "autocast_dtype")
    def serialize_dtypes(self, v: torch.dtype) -> dtypes.PrecisionTypes:
        """Serializes the torch dtype to the corresponding precision type."""
        return dtypes.dtype_to_precision[v]


class ParallelConfig(BaseModel):
    """ParallelConfig is a configuration class for setting up parallelism in model training.

    Attributes:
        tensor_model_parallel_size (int): The size of the tensor model parallelism. Default is 1.
        pipeline_model_parallel_size (int): The size of the pipeline model parallelism. Default is 1.
        accumulate_grad_batches (int): The number of batches to accumulate gradients over. Default is 1.
        ddp (Literal["megatron"]): The distributed data parallel method to use. Default is "megatron".
        remove_unused_parameters (bool): Whether to remove unused parameters. Default is True.
        num_devices (int): The number of devices to use. Default is 1.
        num_nodes (int): The number of nodes to use. Default is 1.

    Methods:
        validate_devices(): Validates the number of devices based on the tensor and pipeline model parallel sizes.
    """

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    accumulate_grad_batches: int = 1
    ddp: Literal["megatron"] = "megatron"
    remove_unused_parameters: bool = True
    use_distributed_optimizer: bool = True
    num_devices: int = 1
    num_nodes: int = 1

    @model_validator(mode="after")
    def validate_devices(self):
        """Validates the number of devices based on the tensor and pipeline model parallel sizes."""
        if self.num_devices < self.tensor_model_parallel_size * self.pipeline_model_parallel_size:
            raise ValueError("devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size")
        return self


class TrainingConfig(BaseModel):
    """TrainingConfig is a configuration class for training models.

    Attributes:
        max_steps (int): The maximum number of training steps.
        limit_val_batches (int | float): The number of validation batches to use. Can be a fraction or a count.
        val_check_interval (int): The interval (in steps) at which to check validation.
        precision (Literal["32", "bf16-mixed", "16-mixed"], optional): The precision to use for training. Defaults to "bf16-mixed".
        accelerator (str, optional): The type of accelerator to use for training. Defaults to "gpu".
        gc_interval (int, optional): The interval of global steps at which to run synchronized garbage collection. Useful for synchronizing garbage collection when performing distributed training. Defaults to 0.
        include_perplexity (bool, optional): Whether to include perplexity in the validation logs. Defaults to False.
        enable_checkpointing (bool, optional): Whether to enable checkpointing and configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint. Corresponds to the same parameter name in pl.Trainer
    """

    max_steps: int
    limit_val_batches: int | float  # Because this can be a fraction or a count...
    val_check_interval: int
    precision: Literal["32", "bf16-mixed", "16-mixed"] = "bf16-mixed"
    accelerator: str = "gpu"
    # NOTE: VERY important for distributed training performance.
    gc_interval: int = 0
    log_train_ppl: bool = False
    log_val_ppl: bool = True
    enable_checkpointing: bool = True
    create_tflops_callback: bool = False


class OptimizerSchedulerConfig(BaseModel):
    """Configuration for the optimizer and learning rate scheduler.

    Attributes:
        lr (float): Learning rate for the optimizer. Default is 1e-4.
        optimizer (str): Type of optimizer to use. Default is "adam".
        interval (str): Interval for updating the learning rate scheduler. Default is "step".
        monitor (str): Metric to monitor for learning rate adjustments. Default is "val_loss".
        interval (str): Interval for updating the learning rate scheduler. Default is "step".
        monitor (str): Metric to monitor for learning rate adjustments. Default is "val_loss".
        warmup_steps (int): Number of warmup steps for use with the warmup annealing learning rate scheduler. Default is 0.
        lr_scheduler (Literal['warmup_anneal', 'cosine']): Type of learning rate scheduler to use. Default is 'warmup_anneal'. NOTE this is likely to change.
        max_steps (Optional[int]): max_steps used in optimizer. Default to None which uses max_steps from TrainingConfig.
    """

    lr: float = 1e-4
    sgd_momentum: float = 0.9
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    use_distributed_optimizer: bool = True
    optimizer: str = "adam"
    interval: str = "step"
    monitor: str = "val_loss"
    cosine_rampup_frac: float = 0.01
    cosine_hold_frac: float = 0.05
    warmup_steps: int = 0
    lr_scheduler: Literal["warmup_anneal", "cosine"] = "warmup_anneal"
    max_steps: Optional[int] = None


class ExperimentConfig(BaseModel):
    """Configuration class for setting up and managing experiment parameters.

    Attributes:
        save_every_n_steps (int): Number of steps between saving checkpoints.
        result_dir (str | pathlib.Path): Directory where results will be saved.
        experiment_name (str): Name of the experiment.
        restore_from_checkpoint_path (Optional[str]): Path to restore from a checkpoint. Note: This does not invoke the checkpoint callback as expected.
        save_last_checkpoint (bool): Flag to save the last checkpoint. Default is True.
        metric_to_monitor_for_checkpoints (str): Metric to monitor for saving top-k checkpoints. Default is "reduced_train_loss".
        save_top_k (int): Number of top checkpoints to save based on the monitored metric. Default is 2.
        create_tensorboard_logger (bool): Flag to create a TensorBoard logger. Default is False.
        create_checkpoint_callback (bool): Flag to create a ModelCheckpoint callback
    """

    save_every_n_steps: int
    result_dir: str | pathlib.Path
    experiment_name: str
    # NOTE: restore_from_checkpoint_path does not invoke the checkpoint callback in the way we'd like. Avoid using.
    restore_from_checkpoint_path: Optional[str]
    save_last_checkpoint: bool = True
    metric_to_monitor_for_checkpoints: str = "reduced_train_loss"
    save_top_k: int = 2
    create_tensorboard_logger: bool = False
    create_checkpoint_callback: bool = True

    @field_serializer("result_dir")
    def serialize_paths(self, value: pathlib.Path) -> str:  # noqa: D102
        return serialize_path_or_str(value)

    @field_validator("result_dir")
    def deserialize_paths(cls, value: str) -> pathlib.Path:  # noqa: D102
        return deserialize_str_to_path(value)


# DataConfig -> some config that can make a data module (see ABC definition.)
DataConfigT = TypeVar("DataConfigT", bound=DataConfig)
# ExposedModelConfig -> some config that can make a non-exposed model config (see ABC definition.)
ExModelConfigT = TypeVar("ExModelConfigT", bound=ExposedModelConfig)


class MainConfig(BaseModel, Generic[ExModelConfigT, DataConfigT]):
    """Main configuration class for BioNeMo. All serialized configs that are a valid MainConfig should be Runnable.

    This class is used to define the main configuration for BioNeMo. It defines the minimal pieces of configuration
    to execution a training job with the NeMo2 training api. It accepts two generic type parameters which users
    must define in their own environment for execution.

    Additionally, this class assumes that the configs for ExposedModelConfig and DataConfig may have custom validators
    implemented that operate on the entire MainConfig. This prevents the need from type based conditionals inside this
    class while still allowing for custom validation global logic to be implemented in the underlying classes. For example,
    some models may want to restrict their Datamodules seq_length to a certain value.


    Args:
        data_config: Generic config type that contains instructions on instantiating the required DataModule.
        parallel_config: The parallel configuration for the model.
        training_config: The training configuration for the model.
        bionemo_model_config: Generic ExposedModelConfig type. This class hides extra configuration parameters in the
            underlying model configuration as well as providing
        optim_config: The optimizer/scheduler configuration for the model.
        experiment_config: The experiment configuration for the model.
        wandb_config: Optional, the wandb configuration for the model.
    """

    data_config: DataConfigT
    parallel_config: ParallelConfig
    training_config: TrainingConfig
    bionemo_model_config: ExModelConfigT
    optim_config: OptimizerSchedulerConfig
    experiment_config: ExperimentConfig
    wandb_config: Optional[WandbConfig] = None

    @model_validator(mode="after")
    def validate_master_config(self) -> "MainConfig":
        """Validates the master configuration object."""
        self.bionemo_model_config.seq_length = self.data_config.seq_length
        return self

    @model_validator(mode="after")
    def run_bionemo_model_config_model_validators(self) -> "MainConfig":
        """Runs the model validators on the bionemo_model_config."""
        return self.bionemo_model_config.custom_model_validator(self)

    @model_validator(mode="after")
    def run_data_config_model_validators(self) -> "MainConfig":
        """Runs the model validators on the data_config."""
        return self.data_config.custom_model_validator(self)

    @model_validator(mode="after")
    def validate_checkpointing_setting(self) -> "MainConfig":
        """Validates the master configuration object."""
        self.training_config.enable_checkpointing = self.experiment_config.create_checkpoint_callback
        return self
