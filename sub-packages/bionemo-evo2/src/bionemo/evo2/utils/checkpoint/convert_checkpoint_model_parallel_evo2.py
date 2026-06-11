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

"""This script converts (potentially sharded) ZeRo1 checkpoint parameters to the desired level of model tensor parallelism for the Evo 2 architecture.

It only supports Zero-1 checkpoints and does not convert any optimizer state,
only the parameters.

Usage:
    python convert_checkpoint_model_parallel_evo2.py \
        --input-checkpoint-dir /path/to/input/checkpoint/global_step1000 \
        --output-checkpoint-dir /path/to/output/checkpoint_mp2/global_step1000 \
        --output-model-parallelism 2
"""

import argparse
import os
import re
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import List, Optional, Set, Union

import torch
from nemo.utils import logging
from params import EVO2_PARAMS, Param


DEVICE = "cpu"
DEFAULT_PARAM_PATTERN = r"sequential\.\d+\.(.+)"


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert checkpoint parameters to desired model parallelism.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the input checkpoint directory containing ZeRo1 checkpoint shards, i.e. mp_rank_*_model_states.pt.",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="mp_rank_*_model_states.pt",
        required=False,
        help="Filename pattern to glob for ZeRo1 checkpoint shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output checkpoint directory to dump the --mp_size converted model checkpoint (ZeRo1).",
    )
    parser.add_argument("--mp_size", type=int, required=True, help="Desired output model parallelism to convert to.")
    parser.add_argument(
        "--exclude-extra",
        action="store_true",
        help="Exclude extra states in the conversion. Default to False, i.e. include extra states.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print more information about the conversion.")
    args = parser.parse_args()
    return args


def concatenate_tensors_across_shards(
    tensor_name: str,
    data_shards: List[OrderedDict[str, torch.Tensor]],
    partition_dim: int,
    hidden_dim: Optional[int] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Concatenate tensor shards across multiple shards.

    Args:
        tensor_name (str): Name of the tensor to concatenate.
        data_shards (List[OrderedDict[str, torch.Tensor]]): List of data shards containing tensors.
        partition_dim (int): Dimension along which to partition the tensor.
        hidden_dim (int, optional): Hidden dimension of the tensor. Defaults to None.
        verbose (bool, optional): Whether to print detailed information. Defaults to False.

    Returns:
        torch.Tensor: Concatenated tensor.
    """
    # Retrieve tensor shards.
    tensors = [shard["module"][tensor_name] for shard in data_shards]

    # Check shape of tensors without tensor parallelism, i.e. stored in all shards of the checkpoint.
    if partition_dim is None:
        for i, tensor in enumerate(tensors):
            if not torch.allclose(tensors[0], tensor):
                logging.info(
                    f"WARNING: Synchronized params differ for param {tensor_name}: abs max diff = {(tensors[0] - tensor).abs().max()}."
                )
                # Get the distribution of tensors[0] and tensor.
                if verbose:
                    ref_tensor = tensors[0].flatten().to(torch.float32)
                    ref_min, ref_max = ref_tensor.min(), ref_tensor.max()

                    q = torch.tensor([0.25, 0.5, 0.75], device=ref_tensor.device)
                    ref_quantiles = ref_tensor.quantile(q)
                    logging.info(f"rank0 tensor: min={ref_min}, max={ref_max} quantiles={ref_quantiles}")

                    target_tensor = tensor.flatten().to(torch.float32)
                    target_min, target_max = target_tensor.min(), target_tensor.max()
                    target_quantiles = target_tensor.quantile(q)
                    logging.info(f"rank{i} tensor: min={target_min}, max={target_max} quantiles={target_quantiles}")

                    logging.info(f"rank0 tensor distribution:\n {ref_tensor.histc(100, min=ref_min, max=ref_max)}")
                    logging.info(f"rank{i} distribution:\n {target_tensor.histc(100, min=ref_min, max=ref_max)}")

        logging.info(f"tensor {tensor_name} not partitioned, returning rank0 tensor {tensors[0].shape}")
        return tensors[0]
    # Check for sharding across the hidden dimension.
    elif partition_dim == hidden_dim:
        raise ValueError(f"Detected sharding for {tensor_name} across hidden dimension at index {hidden_dim}.")

    # Check that the tensors have a consistent hidden dimension.
    expected_dim = None
    if hidden_dim is not None:
        for tensor in tensors:
            if expected_dim is None:
                # Store expected hidden dimension for all tensors.
                expected_dim = tensor.shape[hidden_dim]
            if not tensor.shape[hidden_dim] == expected_dim:
                raise ValueError(f"Tensor {tensor_name} has invalid hidden shape {tensor.shape}.")

    # Concatenate shards.
    return torch.cat(tensors, dim=partition_dim)


def split_tensor_across_shards(
    data_shards: List[OrderedDict],
    tensor: torch.Tensor,
    tensor_name: str,
    partition_dim: int,
) -> None:
    """Split a tensor across multiple shards.

    Args:
        data_shards (List[OrderedDict]): List of data shards to store the split tensors.
        tensor (torch.Tensor): Tensor to split.
        tensor_name (str): Name of the tensor.
        partition_dim (int): Dimension along which to partition the tensor.
    """
    if partition_dim is None:
        # No sharding. Synchronize weights across all shards.
        for data_shard in data_shards:
            data_shard["module"][tensor_name] = tensor
            data_shard["param_shapes"][tensor_name] = tensor.shape
    else:
        # Split the tensor along the partition dimension across shards.
        n_shards = len(data_shards)
        if tensor.shape[partition_dim] % n_shards != 0:
            raise ValueError(
                f"Cannot shard {tensor_name} of dimension {tensor.shape[partition_dim]} across {n_shards} evenly."
            )
        for chunk, data_shard in zip(
            torch.chunk(tensor, chunks=n_shards, dim=partition_dim),
            data_shards,
        ):
            data_shard["module"][tensor_name] = chunk.clone()
            data_shard["param_shapes"][tensor_name] = chunk.shape


def format_output_filename(shard: int) -> str:
    """Format the output filename for a given shard index.

    Args:
        shard (int): Shard index.

    Returns:
        str: Formatted output filename.
    """
    return f"mp_rank_{str(shard).zfill(2)}_model_states.pt"


def check_params(
    detected: List[str],
    expected: Union[Set[str], List[str]],
    buffers: Set[str],
    param_pattern: str = DEFAULT_PARAM_PATTERN,
    verbose: bool = False,
):
    """Check that all model parameters are expected.

    Args:
        detected (List[str]): Detected model parameters names.
        expected (Set[str]): Expected model parameters names.
        buffers (Set[str]): Set of buffer names.
        param_pattern (str, optional): Regex pattern to match parameter names. Defaults to DEFAULT_PARAM_PATTERN.
        verbose (bool, optional): Whether to print detailed information. Defaults to False.
    """
    # Expected model parameters.
    expected = set(expected) if not isinstance(expected, set) else expected
    # Detected model parameters.
    model_param_names = []
    for k in detected:
        match = re.search(param_pattern, k)
        if match is not None:
            model_param_names.append(match.group(1))
        else:
            logging.info(f"Could not match {k}")
    detected_param_set = set(model_param_names)
    if verbose:
        logging.info("Detected Params:\n  {detected_params}".format(detected_params="\n  ".join(detected_param_set)))

    # Log unexpected model parameters.
    missing_params = expected - detected_param_set
    extra_params = detected_param_set - expected
    extra_params = [param for param in extra_params if param not in buffers]
    extra_params = [param for param in extra_params if not param.endswith("._extra_state")]
    if len(extra_params) > 0:
        logging.info(f"WARNING: detected extra params: {extra_params}")
    if len(missing_params) > 0:
        logging.info(f"WARNING: missing params: {missing_params}")
    if not (extra_params or missing_params):
        logging.info("No missing or extra params detected!")


def convert_model_weights(
    input_data_shards: List[OrderedDict],
    output_data_shards: List[OrderedDict],
    model_parameter_names: List[str],
    param_list: List[Param],
    verbose: bool = False,
    exclude_extra: bool = False,
):
    """Convert model weights from input model parallelism to output model parallelism.

    Args:
        input_data_shards (List[OrderedDict]): List of input data shards.
        output_data_shards (List[OrderedDict]): List of output data shards.
        model_parameter_names (List[str]): List of model parameter names.
        param_list (List[Param]): List of parameter information.
        verbose (bool, optional): Whether to print detailed information. Defaults to False.
        exclude_extra (bool, optional): Whether to exclude extra states in the conversion. Defaults to False.
    """
    logging.info(
        f"Converting {len(model_parameter_names)} parameters from {len(input_data_shards)} input shards to {len(output_data_shards)} output shards..."
    )
    converted = 0
    skipped = 0
    for model_parameter in model_parameter_names:
        if args.verbose:
            logging.info(f"Processing {model_parameter}...")

        # Ignore FP8 extra state.
        if model_parameter.endswith("._extra_state"):
            if "extra_state" in model_parameter:
                logging.info(f"Ignoring {model_parameter} -> contains extra state.")
            skipped += 1
            continue

        # Get the partition dimension and hidden dimension of each parameter.
        param_info = None
        for param in param_list:
            if ".".join(model_parameter.split(".")[2:]) == param.name:
                if param_info is None:
                    param_info = param
                else:
                    raise ValueError(
                        f"Found more than one matching model parallelism parameter for {model_parameter}: {param_info}, {param}"
                    )
        if param_info is None:
            raise ValueError(f"Could not find {model_parameter} among known parameters.")

        # Concatenate shards.
        concatenated_tensor = concatenate_tensors_across_shards(
            model_parameter, input_data_shards, param_info.partition_dim, param_info.hidden_dim, verbose=verbose
        )
        # Split into shards.
        split_tensor_across_shards(
            output_data_shards,
            concatenated_tensor,
            model_parameter,
            param_info.partition_dim,
        )
        converted += 1
    logging.info(f"Converted {converted} of {len(model_parameter_names)} parameters (skipped {skipped} params).")
    num_params = len(output_data_shards[0]["module"])
    logging.info(f"Total Params: {num_params}")
    if not all(num_params == len(shard["module"]) for shard in output_data_shards):
        raise ValueError("Shards have different number of parameters, which is not permitted in model parallelism.")

    if not exclude_extra:
        logging.info("Adding extra states from rank0 input shard...")
        rank0_model = input_data_shards[0]["module"]
        for k in rank0_model.keys():
            for i, output_shard in enumerate(output_data_shards):
                if k not in output_shard["module"]:
                    if i == 0:
                        logging.info(f"Adding {k} to output shards.")
                    output_shard["module"][k] = rank0_model[k]
        new_params = len(output_data_shards[0]["module"]) - num_params
        logging.info(f"Added {new_params} extra states, total params: {num_params + new_params}")
        if not all(num_params + new_params == len(shard["module"]) for shard in output_data_shards):
            raise ValueError("Shards have different number of parameters after adding extra states.")

    for shard_idx, output_data_shard in enumerate(output_data_shards):
        output_path = Path(output_data_shard["output_dir"]) / format_output_filename(shard_idx)
        torch.save(
            output_data_shard,
            output_path,
        )
        logging.info(f"Converted checkpoint saved to: {output_path}")


def convert_zero1_model_parallel_checkpoint(
    source_dir: str,
    output_dir: str,
    glob_pattern: str = "mp_rank_*_model_states.pt",
    model_parallel: int = 8,
    param_list: List[Param] = EVO2_PARAMS,
    exclude_extra_params: bool = False,
    verbose: bool = False,
):
    """Convert sharded ZeRo1 checkpoint to desired model parallelism.

    Args:
        source_dir (str): Path to the input checkpoint directory.
        output_dir (str): Path to the output checkpoint directory.
        glob_pattern (str): Filename pattern to glob for ZeRo1 checkpoint shards. Defaults to "mp_rank_*_model_states.pt".
        model_parallel (int): Desired output model parallelism. Defaults to 8.
        param_list (List[Param]): List of parameter information. Defaults to EVO2_PARAMS.
        exclude_extra_params (bool): Whether to exclude extra states in the conversion. Defaults to False.
        verbose (bool): Whether to print detailed information. Defaults to False.
    """
    # Argument validation.
    if not os.path.exists(source_dir):
        raise ValueError(f"Input checkpoint dir ({source_dir}) not found.")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Converting checkpoint from {source_dir} to {output_dir}")

    # Identify all checkpoint model path files.
    parameter_paths = sorted(glob(f"{source_dir}/{glob_pattern}"))
    if len(parameter_paths) == 0:
        raise ValueError(f"No parameter files found in {source_dir}")

    # Load all shards from the ZeRo1 checkpoint.
    input_data_shards = [torch.load(path, map_location=DEVICE) for path in parameter_paths]
    buffers = {buf for x in input_data_shards for buf in x.get("buffer_names", [])}

    # Initialize output MP shards.
    output_data_shards = [
        {
            "module": OrderedDict(),
            "param_shapes": OrderedDict(),
            "dp_world_size": input_data_shards[0]["dp_world_size"],
            "output_dir": output_dir,
        }
        for _ in range(model_parallel)
    ]
    model_parameter_names = input_data_shards[0]["module"].keys()

    # Check no missing or extra params
    check_params(
        detected=list(model_parameter_names),
        expected={param.name for param in param_list},
        buffers=buffers,
        verbose=verbose,
    )
    # Convert the checkpoint
    convert_model_weights(
        input_data_shards,
        output_data_shards,
        model_parameter_names,
        param_list,
        verbose=verbose,
        exclude_extra=exclude_extra_params,
    )
    logging.info("Done!")


if __name__ == "__main__":
    args = get_args()
    convert_zero1_model_parallel_checkpoint(
        args.source_dir,
        args.output_dir,
        args.glob_pattern,
        args.mp_size,
        EVO2_PARAMS,
        args.exclude_extra,
        args.verbose,
    )
