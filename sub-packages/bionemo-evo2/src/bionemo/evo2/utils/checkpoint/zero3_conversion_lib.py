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


"""Helper utility for converting ZeRO3 and ZeRO2 checkpoints to PyTorch."""

import glob
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set

import psutil
import torch
from tqdm import tqdm


BUFFER_NAMES = "buffer_names"
DS_VERSION = "ds_version"
FP32_FLAT_GROUPS = "fp32_flat_groups"
FROZEN_PARAM_FRAGMENTS = "frozen_param_fragments"
FROZEN_PARAM_SHAPES = "frozen_param_shapes"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
PARAM_SHAPES = "param_shapes"
PARTITION_COUNT = "partition_count"
SINGLE_PARTITION_OF_FP32_GROUPS = "single_partition_of_fp32_groups"
ZERO_STAGE = "zero_stage"
EXTRA_STATE = "._extra_state"


@dataclass
class ZeroModelState:
    """A dataclass representing the state of a ZeRO model.

    Attributes:
        buffers (Dict): Buffers in the model state.
        extra_states (Dict): Extra states in the model state.
        param_shapes (List): Shapes of the parameters.
        shared_params (List): Shared parameters in the model state.
        ds_version (int): Version of the DeepSpeed checkpoint.
        frozen_param_shapes (Dict): Shapes of the frozen parameters.
        frozen_param_fragments (Dict): Fragments of the frozen parameters.
    """

    buffers: Dict
    extra_states: Dict
    param_shapes: List
    shared_params: List
    ds_version: int
    frozen_param_shapes: Dict
    frozen_param_fragments: Dict


debug = 0
device = torch.device("cpu")


def profile_memory_decorator(func: Iterable):
    """A decorator to profile memory usage of a function.

    Args:
        func (Iterable): The function to be decorated.

    Returns:
        wrapper: The decorated function with memory profiling.
    """

    def profile_memory():
        pid = os.getpid()
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        print_pid(f"{pid}: RSS = {memory_info.rss / 1024**2:.2f} MB")

    def wrapper(*args, **kwargs):
        profile_memory()
        func(*args, **kwargs)
        profile_memory()

    return wrapper


def print_pid(msg: str):
    """Prints the process ID along with a message.

    Args:
        msg (str): The message to be printed.
    """
    pid = os.getpid()
    print(f"{pid=}:{msg}")


def atoi(text: str):
    """Converts a string to an integer if it is a digit, otherwise returns the string.

    Args:
        text (str): The text to be converted.

    Returns:
        int or str: The converted integer or the original string.
    """
    return int(text) if text.isdigit() else text


def natural_keys(text: str):
    """Sorts a list in human order.

    Args:
        text (str): The text to be sorted.

    Returns:
        list: The sorted list.

    Note:
        alist.sort(key=natural_keys) sorts in human order.
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def get_checkpoint_files(checkpoint_dir: str, glob_pattern: str):
    """Retrieves checkpoint files from a directory based on a glob pattern.

    Args:
        checkpoint_dir (str): The directory to search for checkpoint files.
        glob_pattern (str): The glob pattern to match files.

    Returns:
        list: A sorted list of checkpoint files.

    Raises:
        FileNotFoundError: If no files matching the glob pattern are found.
    """
    # XXX: need to test that this simple glob rule works for multi-node setup too
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern)), key=natural_keys)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"can't find {glob_pattern} files in directory '{checkpoint_dir}'")

    return ckpt_files


def get_model_files_by_rank(checkpoint_dir: str, rank: int):
    """Retrieves model files for a specific rank from a checkpoint directory.

    Args:
        checkpoint_dir (str): The directory to search for model files.
        rank (int): The rank to search for.

    Returns:
        list: A list of model files for the specified rank.
    """
    return get_checkpoint_files(checkpoint_dir, f"*mp_rank_{rank:02}_model_states.pt")


def get_optim_files_by_rank(checkpoint_dir: str, rank: int):
    """Retrieves optimizer files for a specific rank from a checkpoint directory.

    Args:
        checkpoint_dir (str): The directory to search for optimizer files.
        rank (int): The rank to search for.

    Returns:
        list: A list of optimizer files for the specified rank.
    """
    return get_checkpoint_files(checkpoint_dir, f"*mp_rank_{rank:02}_optim_states.pt")


def create_ds_output_path(rank: int):
    """Creates the output path for a DeepSpeed checkpoint.

    Args:
        rank (int): The rank to create the output path for.

    Returns:
        str: The output path for the DeepSpeed checkpoint.
    """
    return f"mp_rank_{rank:02}_model_states.pt"


def create_zero3_model_state_path(dp_rank: int, mp_rank: int):
    """Creates the path for a ZeRO3 model state file.

    Args:
        dp_rank (int): The data parallel rank.
        mp_rank (int): The model parallel rank.

    Returns:
        str: The path for the ZeRO3 model state file.
    """
    return f"zero_pp_rank_{dp_rank}_mp_rank_{mp_rank:02}_model_states.pt"


def create_zero3_optim_state_path(dp_rank: int, mp_rank: int):
    """Creates the path for a ZeRO3 optimizer state file.

    Args:
        dp_rank (int): The data parallel rank.
        mp_rank (int): The model parallel rank.

    Returns:
        str: The path for the ZeRO3 optimizer state file.
    """
    return f"bf16_zero_pp_rank_{dp_rank}_mp_rank_{mp_rank:02}_optim_states.pt"


def get_model_state_file(checkpoint_dir: str, zero_stage: int):
    """Retrieves the model state file from a checkpoint directory based on the ZeRO stage.

    Args:
        checkpoint_dir (str): The directory to search for the model state file.
        zero_stage (int): The ZeRO stage to search for.

    Returns:
        str: The path to the model state file.

    Raises:
        FileNotFoundError: If the directory or model state file is not found.
        ValueError: If the ZeRO stage is not supported.
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")

    # there should be only one file
    if zero_stage <= 2:
        file = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    elif zero_stage == 3:
        file = os.path.join(checkpoint_dir, "zero_pp_rank_0_mp_rank_00_model_states.pt")
    else:
        raise ValueError(f"Unsupported zero stage {zero_stage}. Expected 1, 2, or 3")

    if not os.path.exists(file):
        raise FileNotFoundError(f"can't find model states file at '{file}'")

    return file


def parse_model_states(files: Set[str]):
    """Parses model state files and returns a list of ZeroModelState objects.

    Args:
        files (Set[str]): A set of file paths to parse.

    Returns:
        List[ZeroModelState]: A list of parsed ZeroModelState objects.

    Raises:
        ValueError: If a file is not a model state checkpoint.
    """
    zero_model_states = []
    for file in files:
        state_dict = torch.load(file, map_location=device)

        if BUFFER_NAMES not in state_dict:
            raise ValueError(f"{file} is not a model state checkpoint")
        buffer_names = state_dict[BUFFER_NAMES]
        if debug:
            print_pid("Found buffers:", buffer_names)

        # recover just the buffers while restoring them to fp32 if they were saved in fp16
        buffers = {k: v.float() for k, v in state_dict["module"].items() if k in buffer_names}

        extra_states = {k: v for k, v in state_dict["module"].items() if k.endswith(EXTRA_STATE)}

        # collect parameters that are included in param_shapes
        param_shapes = state_dict[PARAM_SHAPES]
        param_names = []
        for s in param_shapes:
            for name in s.keys():
                param_names.append(name)

        # update with frozen parameters
        frozen_param_shapes = state_dict.get(FROZEN_PARAM_SHAPES, None)
        if frozen_param_shapes is not None:
            if debug:
                print_pid(f"Found frozen_param_shapes: {frozen_param_shapes}")
            param_names += list(frozen_param_shapes.keys())

        # handle shared params
        shared_params = [[k, v] for k, v in state_dict["shared_params"].items()]

        ds_version = state_dict.get(DS_VERSION, None)

        frozen_param_fragments = state_dict.get(FROZEN_PARAM_FRAGMENTS, None)

        z_model_state = ZeroModelState(
            buffers=buffers,
            extra_states=extra_states,
            param_shapes=param_shapes,
            shared_params=shared_params,
            ds_version=ds_version,
            frozen_param_shapes=frozen_param_shapes,
            frozen_param_fragments=frozen_param_fragments,
        )
        zero_model_states.append(z_model_state)

    return zero_model_states


def parse_optim_states(files: Set[str], ds_checkpoint_dir: str):
    """Parses optimizer state files and returns the ZeRO stage, world size, and fp32 flat groups.

    Args:
        files (Set[str]): A set of file paths to parse.
        ds_checkpoint_dir (str): The directory containing the DeepSpeed checkpoint.

    Returns:
        tuple: A tuple containing the ZeRO stage, world size, and fp32 flat groups.

    Raises:
        ValueError: If a file is not a ZeRO checkpoint or if the number of files does not match the expected world size.
    """
    total_files = len(files)
    state_dicts = []
    for f in files:
        state_dict = torch.load(f, map_location=device)
        # immediately discard the potentially huge 2 optimizer states as we only care for fp32 master weights
        # and also handle the case where it was already removed by another helper script
        state_dict["optimizer_state_dict"].pop("optimizer_state_dict", None)
        state_dict[OPTIMIZER_STATE_DICT] = {
            FP32_FLAT_GROUPS: state_dict[OPTIMIZER_STATE_DICT][FP32_FLAT_GROUPS],
            ZERO_STAGE: state_dict[OPTIMIZER_STATE_DICT][ZERO_STAGE],
            PARTITION_COUNT: state_dict[OPTIMIZER_STATE_DICT][PARTITION_COUNT],
        }
        state_dicts.append(state_dict)

    if ZERO_STAGE not in state_dicts[0][OPTIMIZER_STATE_DICT]:
        raise ValueError(f"{files[0]} is not a zero checkpoint")
    zero_stage = state_dicts[0][OPTIMIZER_STATE_DICT][ZERO_STAGE]
    world_size = state_dicts[0][OPTIMIZER_STATE_DICT][PARTITION_COUNT]

    # For ZeRO-2 each param group can have different partition_count as data parallelism for expert
    # parameters can be different from data parallelism for non-expert parameters. So we can just
    # use the max of the partition_count to get the dp world_size.

    if type(world_size) is list:
        world_size = max(world_size)

    if world_size != total_files:
        raise ValueError(
            f"Expected {world_size} of '*_optim_states.pt' under '{ds_checkpoint_dir}' but found {total_files} files. "
            "Possibly due to an overwrite of an old checkpoint, or a checkpoint didn't get saved by one or more processes."
        )

    # the groups are named differently in each stage
    if zero_stage <= 2:
        fp32_groups_key = SINGLE_PARTITION_OF_FP32_GROUPS
    elif zero_stage == 3:
        fp32_groups_key = FP32_FLAT_GROUPS
    else:
        raise ValueError(f"unknown zero stage {zero_stage}")

    if zero_stage <= 2:
        fp32_flat_groups = [state_dicts[i][OPTIMIZER_STATE_DICT][fp32_groups_key] for i in range(len(state_dicts))]
    elif zero_stage == 3:
        # if there is more than one param group, there will be multiple flattened tensors - one
        # flattened tensor per group - for simplicity merge them into a single tensor
        #
        # XXX: could make the script more memory efficient for when there are multiple groups - it
        # will require matching the sub-lists of param_shapes for each param group flattened tensor

        fp32_flat_groups = [
            torch.cat(state_dicts[i][OPTIMIZER_STATE_DICT][fp32_groups_key], 0) for i in range(len(state_dicts))
        ]

    return zero_stage, world_size, fp32_flat_groups


def _get_fp32_state_dict_from_zero_checkpoint(
    ds_checkpoint_dir: str, rank: int, exclude_frozen_parameters: bool = False
):
    """Returns the fp32 state dictionary reconstructed from a ZeRO checkpoint.

    Args:
        ds_checkpoint_dir (str): Path to the DeepSpeed checkpoint folder.
        rank (int): The rank to process.
        exclude_frozen_parameters (bool): Whether to exclude frozen parameters.

    Returns:
        OrderedDict: The reconstructed fp32 state dictionary.
    """
    print_pid(f"Processing zero checkpoint '{ds_checkpoint_dir}'")

    # optim_files = get_optim_files(ds_checkpoint_dir)
    # zero_stage, world_size, fp32_flat_groups = parse_optim_states(optim_files, ds_checkpoint_dir)

    optim_files = get_optim_files_by_rank(ds_checkpoint_dir, rank=rank)
    optim_files_check = get_checkpoint_files(ds_checkpoint_dir, f"bf16*_{rank:02d}_optim_states.pt")
    assert set(optim_files) == set(optim_files_check), f"Expected {optim_files_check}, got {optim_files}"
    # check ordering as well
    for f1, f2 in zip(optim_files, optim_files_check):
        assert os.path.basename(f1) == os.path.basename(f2), (
            f"Found mismatching optim files for rank {rank}: {os.path.basename(f1)} != {os.path.basename(f2)}"
        )
    print_pid(f" -> Optim files for rank {rank}: {len(optim_files)}")

    if debug:
        print_pid(f"{optim_files=}")

    if os.environ.get("ZERO3_CONVERSION_DEBUG", "0") == "1":
        breakpoint()

    zero_stage, world_size, fp32_flat_groups = parse_optim_states(optim_files, ds_checkpoint_dir)
    assert len(optim_files) == world_size, f"Expected {world_size} optim files, got {len(optim_files)}"
    if debug:
        print_pid(
            f" -> rank{rank} stage: {zero_stage} {world_size=} {len(fp32_flat_groups)=} {fp32_flat_groups.shape=}"
        )

    model_files = get_model_files_by_rank(ds_checkpoint_dir, rank=rank)
    model_files_check = get_checkpoint_files(ds_checkpoint_dir, f"zero_*_mp_rank_{rank:02d}_model_states.pt")
    assert set(model_files) == set(model_files_check), f"Expected {model_files_check}, got {model_files}"

    for f1, f2 in zip(model_files, model_files_check):
        assert os.path.basename(f1) == os.path.basename(f2), (
            f"Found mismatching optim files for rank {rank}: {os.path.basename(f1)} != {os.path.basename(f2)}"
        )
    print_pid(f" -> Model files for rank {rank}: {len(model_files)}")

    assert len(optim_files) == len(model_files), (
        f"Expected same number of optim and model files: {len(optim_files)} != {len(model_files)}"
    )
    assert len(optim_files) > 0, f"Expected at least one optim file, got {len(optim_files)}"

    zero_model_states = parse_model_states(model_files)
    print_pid(f"Parsing checkpoint created by deepspeed=={zero_model_states[0].ds_version}")

    return _get_fp32_state_dict_from_zero3_checkpoint(
        world_size, fp32_flat_groups, zero_model_states, exclude_frozen_parameters
    )


def zero3_partitioned_param_info(unpartitioned_numel: int, world_size: int):
    """Returns the partitioned and padding number of elements for a parameter.

    Args:
        unpartitioned_numel (int): The number of elements in the unpartitioned parameter.
        world_size (int): The world size.

    Returns:
        tuple: A tuple containing the partitioned number of elements and the padding number of elements.
    """
    remainder = unpartitioned_numel % world_size
    padding_numel = (world_size - remainder) if remainder else 0
    partitioned_numel = math.ceil(unpartitioned_numel / world_size)
    return partitioned_numel, padding_numel


def _zero3_merge_frozen_params(state_dict: Dict[str, Any], world_size: int, zero_model_states: List[ZeroModelState]):
    """Merges frozen parameters into the state dictionary.

    Args:
        state_dict (Dict[str, Any]): The state dictionary to update.
        world_size (int): The world size.
        zero_model_states (List[ZeroModelState]): The list of ZeroModelState objects.

    Returns:
        None
    """
    if zero_model_states[0].frozen_param_shapes is None or len(zero_model_states[0].frozen_param_shapes) == 0:
        return

    if debug:
        for i in range(world_size):
            num_elem = sum(s.numel() for s in zero_model_states[i].frozen_param_fragments.values())
            print_pid(f"rank {i}: {FROZEN_PARAM_SHAPES}.numel = {num_elem}")

        frozen_param_shapes = zero_model_states[0].frozen_param_shapes
        wanted_params = len(frozen_param_shapes)
        wanted_numel = sum(s.numel() for s in frozen_param_shapes.values())
        avail_numel = sum([p.numel() for p in zero_model_states[0].frozen_param_fragments.values()]) * world_size
        print_pid(f"Frozen params: Have {avail_numel} numels to process.")
        print_pid(f"Frozen params: Need {wanted_numel} numels in {wanted_params} params")

    total_params = 0
    total_numel = 0
    for name, shape in zero_model_states[0].frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        param_frags = tuple(model_state.frozen_param_fragments[name] for model_state in zero_model_states)
        state_dict[name] = torch.cat(param_frags, 0).narrow(0, 0, unpartitioned_numel).view(shape)

        partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)

        if debug:
            print_pid(
                f"Frozen params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}"
            )

    print_pid(f"Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements")


# @profile_memory_decorator
def _zero3_merge_trainable_params(
    state_dict: Dict[str, Any],
    world_size: int,
    fp32_flat_groups: List[torch.Tensor],
    zero_model_states: List[ZeroModelState],
):
    """Merges trainable parameters into the state dictionary.

    Args:
        state_dict (Dict[str, Any]): The state dictionary to update.
        world_size (int): The world size.
        fp32_flat_groups (List[torch.Tensor]): The list of fp32 flat groups.
        zero_model_states (List[ZeroModelState]): The list of ZeroModelState objects.

    Returns:
        None
    """
    if os.environ.get("ZERO3_CONVERSION_DEBUG", "0") == "1":
        breakpoint()

    param_shapes = zero_model_states[0].param_shapes
    avail_numel = fp32_flat_groups[0].numel() * world_size
    # Reconstruction protocol: For zero3 we need to zip the partitions together at boundary of each
    # param, re-consolidating each param, while dealing with padding if any

    # merge list of dicts, preserving order
    param_shapes = {k: v for d in param_shapes for k, v in d.items()}

    if debug:
        for i in range(world_size):
            print_pid(f"{FP32_FLAT_GROUPS}[{i}].shape={fp32_flat_groups[i].shape}")

        wanted_params = len(param_shapes)
        wanted_numel = sum(shape.numel() for shape in param_shapes.values())
        # not asserting if there is a mismatch due to possible padding
        avail_numel = fp32_flat_groups[0].numel() * world_size
        print_pid(f"Trainable params: Have {avail_numel} numels to process.")
        print_pid(f"Trainable params: Need {wanted_numel} numels in {wanted_params} params.")

    # params
    # XXX: for huge models that can't fit into the host's RAM we will have to recode this to support
    # out-of-core computing solution
    offset = 0
    total_numel = 0
    total_params = 0
    pid = os.getpid()
    for name, shape in tqdm(param_shapes.items(), desc=f"{pid=}: Gathering Sharded Weights"):
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel
        total_params += 1
        # NOTE: partitioned_numel includes padding, padding applies if unpartitioned_numel is not divisible by world_size
        partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)

        if debug:
            print_pid(
                f"Trainable params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}"
            )

        # XXX: memory usage doubles here
        state_dict[name] = (
            torch.cat(tuple(fp32_flat_groups[i].narrow(0, offset, partitioned_numel) for i in range(world_size)), 0)
            .narrow(0, 0, unpartitioned_numel)
            .view(shape)
        )
        offset += partitioned_numel

    offset *= world_size

    # Sanity check
    if offset != avail_numel:
        raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")

    print_pid(f"Reconstructed Trainable fp32 state dict with {total_params} params {total_numel} elements")


def _get_fp32_state_dict_from_zero3_checkpoint(
    world_size: int,
    fp32_flat_groups: List[torch.Tensor],
    zero_model_states: List[ZeroModelState],
    exclude_frozen_parameters: bool,
):
    """Returns the fp32 state dictionary reconstructed from a ZeRO3 checkpoint.

    Args:
        world_size (int): The world size.
        fp32_flat_groups (List[torch.Tensor]): The list of fp32 flat groups.
        zero_model_states (List[ZeroModelState]): The list of ZeroModelState objects.
        exclude_frozen_parameters (bool): Whether to exclude frozen parameters.

    Returns:
        OrderedDict: The reconstructed fp32 state dictionary.
    """
    state_dict = OrderedDict()

    # buffers
    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)
    if debug:
        print_pid(f"added {len(buffers)} buffers")

    # extra state (e.g., fp8)
    extra_states = zero_model_states[0].extra_states
    state_dict.update(extra_states)
    if debug:
        print_pid(f"added {len(extra_states)} extra_states")

    if not exclude_frozen_parameters:
        _zero3_merge_frozen_params(state_dict, world_size, zero_model_states)

    _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states)

    # recover shared parameters
    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]

    return state_dict


def get_elapsed(t: float):
    """Converts elapsed time in seconds to a formatted string.

    Args:
        t (float): The elapsed time in seconds.

    Returns:
        str: The formatted elapsed time as a string.
    """
    minutes = t // 60
    seconds = t % 60
    if minutes > 0:
        total_time = f"{minutes:.0f}min{seconds:.0f}s"
    else:
        total_time = f"{seconds:.1f}s"
    return total_time


def process_single_rank(
    rank: int,
    ds_checkpoint_dir: str,
    output_dir: str,
    overwrite: bool = False,
    exclude_frozen_parameters: bool = False,
):
    """Processes a single rank to gather and save the state dictionary.

    Args:
        rank (int): The rank to process.
        ds_checkpoint_dir (str): Path to the DeepSpeed checkpoint folder.
        output_dir (str): Directory to save the output.
        overwrite (bool): Whether to overwrite existing files. Default is False.
        exclude_frozen_parameters (bool): Whether to exclude frozen parameters. Default is False.
    """
    print_pid(f"Gathering rank {rank} state_dict...")

    start = time.time()
    output_path = os.path.join(output_dir, create_ds_output_path(rank))
    if os.path.exists(output_path) and not overwrite:
        print_pid(f"Output path {output_path} exists, skipping")
        return

    print_pid(f" -> Gathering data parallel partitions for mp rank {rank}...")

    if os.environ.get("ZERO3_CONVERSION_DEBUG", "0") == "1":
        breakpoint()

    state_dict = _get_fp32_state_dict_from_zero_checkpoint(
        ds_checkpoint_dir=ds_checkpoint_dir, rank=rank, exclude_frozen_parameters=exclude_frozen_parameters
    )
    print_pid(f" -> Done processing rank {rank} state_dict, gathered {len(state_dict)} params")

    checkpoint = {
        "module": state_dict,
        "param_shapes": OrderedDict(),
        "dp_world_size": 1,
    }

    for param, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            checkpoint["param_shapes"][param] = value.shape

    print_pid(f" -> Saving mp rank {rank} checkpoint to {output_path}")
    torch.save(checkpoint, f"{output_path}")

    total_time = get_elapsed(time.time() - start)
    print_pid(f" -> rank {rank} took {total_time}")
