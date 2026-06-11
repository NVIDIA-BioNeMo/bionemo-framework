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
import logging
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Generic, List, Literal, Optional, Protocol, Sequence, Type

import torchmetrics
from megatron.core.transformer import TransformerConfig
from nemo.lightning import io

from bionemo.core.model.config import BionemoModelConfig, BionemoTrainableModelConfig
from bionemo.llm.api import MegatronLossType, MegatronModelType
from bionemo.llm.utils import iomixin_utils as iom
from bionemo.llm.utils.weight_utils import load_weights_sharded_inplace_nemo2_to_mcore


__all__: Sequence[str] = ("MegatronBioNeMoModelConfig", "TorchmetricsConfig")

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
)

logger = logging.getLogger(__file__)

_OVERRIDE_BIONEMO_CONFIG_DEFAULTS: List[str] = [
    "initial_ckpt_skip_keys_with_these_prefixes",
    "override_parent_fields",
    "initial_ckpt_path_ignore_weights",
    "initial_ckpt_path",
    "model_cls",
    "bf16",
    "fp16",
    "train_metric",
    "valid_metric",
]

OVERRIDE_BIONEMO_CONFIG_DEFAULTS = deepcopy(_OVERRIDE_BIONEMO_CONFIG_DEFAULTS)  # copy for export


class MegatronBioNeMoModelConfig(BionemoModelConfig[MegatronModelType], TransformerConfig, iom.WillHaveGetSetHparam):
    """A ModelConfig class for bionemo that supports usage with Megatron models, for example as NeMo2 requires."""

    model_cls: Type[MegatronModelType]


@dataclass
class MegatronBioNeMoTrainableModelConfig(
    MegatronBioNeMoModelConfig[MegatronModelType],
    BionemoTrainableModelConfig[MegatronModelType, MegatronLossType],
    Generic[MegatronModelType, MegatronLossType],
):
    """A TrainableModelConfig class for bionemo that supports usage with Megatron models, for example as NeMo2 requires."""

    initial_ckpt_path: str | None = None
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=list)
    override_parent_fields: List[str] = field(default_factory=lambda: _OVERRIDE_BIONEMO_CONFIG_DEFAULTS)

    def load_settings_from_checkpoint(self, initial_ckpt_path: str) -> None:
        """Load settings into self from the checkpoint saved in self.

        Any setting in self.override_parent_fields is not overriden. Note that this function will also update the hyper
        parameters in this config, as well as the associated attributes in self in case they were modified post-init.

        Args:
            initial_ckpt_path: The path to the checkpoint to load, note that everything is loaded from this checkpoint
                other than the settings in self.override_parent_fields.

        Returns:
            None, the settings are loaded into self in place, and the hyper-parameters that will later be saved into
                a checkpoint are updated.
        """
        logger.warning(f"Loading {self.initial_ckpt_path}")
        # 1. get the config from the trainer io context by querying the `model.config` subpath of the trainer.
        initial_config: MegatronBioNeMoTrainableModelConfig = io.load_context(
            path=Path(initial_ckpt_path) / "context", subpath="model.config"
        )  # type: ignore
        initial_fields = {f.name for f in fields(initial_config)}
        my_fields = [f.name for f in fields(self)]
        skip_fields = set(self.override_parent_fields)
        override_fields = [f for f in my_fields if f in initial_fields and f not in skip_fields]
        override_mutate_possibly_extra_mutated_fiddle(self, initial_config, override_fields)

    def update_model_from_checkpoint(self, model: MegatronModelType, initial_ckpt_path: str) -> None:
        """Utility function to standardize how to load a megatron model from a checkpoint ignoring user-specified keys.

        Update the model with the weights from the provided checkpoint path, skipping the keys with the prefixes in
            self.initial_ckpt_skip_keys_with_these_prefixes.

        Args:
            model: The Megatron model to update.
            initial_ckpt_path: The path to the megatron checkpoint to load.

        Returns:
            None, the model is updated in place, supporting megatron model parallelism abstractions, and ignoring
                any extra keys that are provided in self.initial_ckpt_skip_keys_with_these_prefixes.
        """
        load_weights_sharded_inplace_nemo2_to_mcore(
            model=model,  # type: ignore
            distributed_checkpoint_dir=initial_ckpt_path,
            skip_keys_with_these_prefixes=set(self.initial_ckpt_skip_keys_with_these_prefixes),
        )


class IOMixinProto(Protocol):
    """A Protocol for the get/set hparam functions of the IOMixin class from NeMo."""

    def set_hparam(self, attribute: str, value: Any, also_change_value: bool = True) -> None:
        """Set the value of an attribute in the config attached to the class by the IOMixin."""
        ...

    def get_hparam(self, attribute: str) -> Any:
        """Get the value of an attribute in the config attached to the class by the IOMixin."""
        ...


def override_mutate_possibly_extra_mutated_fiddle(
    target_cfg: IOMixinProto, source_cfg: IOMixinProto, maybe_mutated_elements_to_clone: List[str]
) -> None:
    """Override the values of the target config with the values of the source config for the given elements.

    This will modify the tracked init hyper-parameter values, as well as modifying the associated attributes in
        self incase they were modified later by post_init code.

    Args:
        target_cfg: The config to update.
        source_cfg: The config to copy values from.
        maybe_mutated_elements_to_clone: The list of elements to copy from the source config to the target config.

    Returns:
        None, the target config is updated in place.
    """
    for f in maybe_mutated_elements_to_clone:
        # 1. Update the tracked config values. Note that the associated attribute in self may have been modified
        #  post-init, so we don't want to change the value in self here. We do that separately next.
        target_cfg.set_hparam(f, source_cfg.get_hparam(f), also_change_value=False)
        # 2. Update the lazily untracked values (if the same variable name is used post-init)
        setattr(target_cfg, f, getattr(source_cfg, f))


@dataclass
class TorchmetricsConfig:
    """TorchmetricsConfig to instantiate torchmetrics.Metric class.

    Fiddle requires all objects in config serializable and torchmetric.Metric is not. Its instantiation must be deferred into BionemoLightningModule.__init__.
    Only support torchmetrics currently, e.g. users can provide 'text.Perplexity' to 'class_path' to use 'torchmetrics.text.Perplexity'.
    """

    class_path: str
    task: Literal["lm", "classification", "regression"]
    metric_name: str
    kwargs: Optional[dict[str, Any]] = None

    def __post_init__(self):
        """__post_init__ in dataclass."""
        self.kwargs = {} if self.kwargs is None else self.kwargs

    def get_instance(self) -> torchmetrics.Metric:
        """Dynamically imports and instantiates the metric class."""
        if "." in self.class_path:
            module_path, class_name = self.class_path.rsplit(".", 1)
            module = importlib.import_module(f"torchmetrics.{module_path}")
        else:
            class_name = self.class_path
            module = importlib.import_module("torchmetrics")

        cls_ = getattr(module, class_name)
        return cls_(**self.kwargs)
