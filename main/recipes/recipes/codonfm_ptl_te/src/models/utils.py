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


import math
from functools import partial

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .components.encodon_config import EnCodonConfig


def construct_pretrained_config(config, pretrained_config_class=EnCodonConfig):
    """Constructs a configuration object from a given config object with dot accessible attributes.
    Excludes any attributes that start with '_' or are callable.
    Ensures that all required attributes are present in the config_dict by checking against the default config.
    If an attribute is missing, it is set to the default value from the default config.

    Args:
        config: A config object with dot accessible attributes.
        pretrained_config_class: An optional config class to use for constructing the config object.

    Returns:
        A configuration object.
    """  # noqa: D205
    # - convert config object to dictionary

    config_dict = {
        key: (getattr(config, key) if hasattr(config, key) else config[key])
        for key in (config.keys() if isinstance(config, dict) else dir(config))
        if not key.startswith("_") and not callable(getattr(config, key, None))
    }

    default_config = pretrained_config_class()

    encodon_config_dict = {}
    # - ensure all required attributes are present in the config_dict
    for key in default_config.to_dict().keys():
        if key not in config_dict:
            encodon_config_dict[key] = getattr(default_config, key)
        else:
            encodon_config_dict[key] = config_dict[key]

    # - construct config object
    return pretrained_config_class(**encodon_config_dict)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """Taken from: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """  # noqa: D415
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_decay_parameter_names(
    model: nn.Module,
    none_applicable_layer_types: list = None,  # noqa: RUF013
    disallowed_layer_names: list = None,  # noqa: RUF013
):
    """Recursively retrieves the names of model parameters that are subject to weight decay.

    This function traverses the model's module hierarchy and identifies parameters
    eligible for weight decay based on inclusion/exclusion rules for layer types and parameter names.
    Typically, normalization layers, embedding layers, and biases are excluded from weight decay.

    The logic is adopted from Hugging Face Transformers:
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py

    Args:
        model (nn.Module): The model to inspect.
        none_applicable_layer_types (list, optional): A list of `nn.Module` subclasses whose parameters
            should be excluded from decay. Defaults to `[nn.LayerNorm, nn.Embedding]`.
        disallowed_layer_names (list, optional): A list of substrings. Parameter names
            containing any of these substrings (case-insensitive) will be excluded from decay.
            Defaults to `['bias', 'layernorm', 'rmsnorm']`.

    Returns:
        list[str]: A list of fully qualified parameter names to apply weight decay to.
    """
    if none_applicable_layer_types is None:
        none_applicable_layer_types = [nn.LayerNorm, nn.Embedding]
    if disallowed_layer_names is None:
        disallowed_layer_names = ["bias", "layernorm", "rmsnorm"]

    result = []
    for name, child in model.named_children():
        child_params = get_decay_parameter_names(child, none_applicable_layer_types, disallowed_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(none_applicable_layer_types))
            and not any(forbidden in f"{name}.{n}".lower() for forbidden in disallowed_layer_names)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k for k in model._parameters.keys() if not any(forbidden in k.lower() for forbidden in disallowed_layer_names)
    ]
    return result
