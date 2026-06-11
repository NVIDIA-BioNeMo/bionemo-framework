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


from typing import Any, Dict, List, Union


def float_or_int_or_none(value: Union[str, float, int, None]) -> Union[float, int, None]:
    """Converts a given value into a float, int, or None.

    Args:
        value (Union[str, float, int, None]): A value that can be either a string, float, int, or None.

    Returns:
        Union[float, int, None]: A float, int, or None based on the input value.

    If the input value is None or "None", it returns None.
    If the input value is an int or float, it returns the same value.
    If the input value is a string, it tries to convert it into an int if possible, otherwise into a float.
    """
    if value is None or value == "None":
        return
    if isinstance(value, (int, float)):
        return value
    if value.isdigit():
        return int(value)
    return float(value)


def parse_kwargs_to_arglist(kwargs: Dict[str, Any]) -> List[str]:
    """Converts a dictionary of keyword arguments into a list of command-line arguments.

    Args:
        kwargs (Dict[str, Any]): A dictionary where keys are argument names and values are argument values.

    Returns:
        A list of strings, where each string is a command-line argument in the format '--argument-name value'.
    """
    arglist = []
    for k, v in kwargs.items():
        arglist.extend([f"--{k.replace('_', '-')}", str(v)])
    return arglist


def infer_global_batch_size(
    micro_batch_size: int,
    num_nodes: int,
    devices: int,
    accumulate_grad_batches: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_model_parallel_size: int = 1,
) -> int:
    """Infers the global batch size based on the micro batch size, number of nodes, devices, accumulation of gradient batches, and model parallel sizes.

    Args:
        micro_batch_size (int): The micro batch size.
        num_nodes (int): The number of nodes.
        devices (int): The number of devices.
        accumulate_grad_batches (int): The accumulation of gradient batches. Defaults to 1.
        tensor_model_parallel_size (int): The tensor model parallel size. Defaults to 1.
        pipeline_model_parallel_size (int): The pipeline model parallel size. Defaults to 1.
        context_model_parallel_size (int): The context model parallel size. Defaults to 1.

    Returns:
        int: The global batch size.
    """
    if not all(
        isinstance(arg, int)
        for arg in [
            micro_batch_size,
            num_nodes,
            devices,
            accumulate_grad_batches,
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_model_parallel_size,
        ]
    ):
        raise ValueError(
            f"All arguments must be of type int, got {type(micro_batch_size)}, {type(num_nodes)}, {type(devices)}, "
            f"{type(accumulate_grad_batches)}, {type(tensor_model_parallel_size)}, {type(pipeline_model_parallel_size)}, and {type(context_model_parallel_size)}"
        )
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be greater than 0, got {micro_batch_size}")
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be greater than 0, got {num_nodes}")
    if devices <= 0:
        raise ValueError(f"devices must be greater than 0, got {devices}")
    if accumulate_grad_batches <= 0:
        raise ValueError(f"accumulate_grad_batches must be greater than 0, got {accumulate_grad_batches}")
    if tensor_model_parallel_size <= 0:
        raise ValueError(f"tensor_model_parallel_size must be greater than 0, got {tensor_model_parallel_size}")
    if pipeline_model_parallel_size <= 0:
        raise ValueError(f"pipeline_model_parallel_size must be greater than 0, got {pipeline_model_parallel_size}")
    if context_model_parallel_size <= 0:
        raise ValueError(f"context_model_parallel_size must be greater than 0, got {context_model_parallel_size}")

    world_size = num_nodes * devices
    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size * context_model_parallel_size) != 0:
        raise ValueError(
            f"world_size must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size * context_model_parallel_size, "
            f"got {world_size} and TP{tensor_model_parallel_size} * PP{pipeline_model_parallel_size} * CP{context_model_parallel_size}"
        )

    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_model_parallel_size
    data_parallel_size = world_size // model_parallel_size
    global_batch_size = micro_batch_size * data_parallel_size * accumulate_grad_batches
    return global_batch_size


def infer_num_samples(
    limit_batches: Union[float, int, str, None], num_samples_in_dataset: int, global_batch_size: int, stage: str
):
    """Infers the number of samples based on the limit_batches parameter, the length of the dataset, and the global batch size.

    Args:
        limit_batches (Union[float, int, str, None]): The limit on the number of batches. Can be a float
            between 0 and 1, an integer, a string, or None. If None, defaults to 1.0.
        num_samples_in_dataset (int): The number of samples in the dataset.
        global_batch_size (int): The global batch size.
        stage (str): The stage of the training.

    Returns:
        int: The number of samples from the limit.

    Raises:
        ValueError: If the limited number of samples is less than the global batch size, or if the
            limit_batches parameter is invalid.

    If limit_batches is a float between 0 and 1, the number of samples is inferred as a fraction of the number of samples
    in the dataset. If limit_batches is an integer greater than or equal to 1, the number of limited samples is inferred
    as the product of limit_batches and global batch size. If limit_batches is None, it defaults to 1.0, indicating that
    all dataset samples should be used.
    """
    limit_batches = 1.0 if limit_batches is None else limit_batches  # validation data does not require upsampling
    if 0 < limit_batches <= 1.0 and isinstance(limit_batches, float):
        num_limited_samples = int(num_samples_in_dataset * limit_batches)
        if num_limited_samples < global_batch_size:
            raise ValueError(
                "The limited number of %s samples %s is less than the global batch size %s"
                % (stage, num_limited_samples, global_batch_size)
            )
    elif limit_batches >= 1 and isinstance(limit_batches, int):
        num_limited_samples = int(limit_batches * global_batch_size)
    else:
        raise ValueError("Invalid choice of limit_%s_batches size: %s" % (stage, limit_batches))

    return num_limited_samples
