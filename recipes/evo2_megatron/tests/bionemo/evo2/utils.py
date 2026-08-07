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

"""Shared test utilities for evo2 tests."""

import gc
import socket
from contextlib import contextmanager

import megatron.core.num_microbatches_calculator
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel import random as tp_random
from pytest import MonkeyPatch


DEFAULT_MASTER_ADDR = "localhost"
DEFAULT_MASTER_PORT = "29500"
DEFAULT_NCCL_TIMEOUT = "30"  # in seconds


def find_free_network_port(address: str = "localhost") -> int:
    """Find a free port on localhost for distributed testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((address, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_compute_capability() -> tuple[int, int]:
    """Get the compute capability of the current device."""
    if not torch.cuda.is_available():
        return (0, 0)
    # Returns a tuple, e.g., (9, 0) for H100
    return torch.cuda.get_device_capability()


def is_fp8_supported() -> bool:
    """Check if FP8 is supported on the current device.

    FP8 is supported on Ada Lovelace (8.9) and Hopper (9.0+).
    """
    cc = get_compute_capability()
    return cc >= (8, 9)


def is_fp4_supported() -> bool:
    """Check if FP4 is supported on the current device.

    Native support requires Blackwell (10.0+).
    """
    cc = get_compute_capability()
    return (10, 0) <= cc < (12, 0)


def is_mxfp8_supported() -> bool:
    """Check if MXFP8 is supported on the current device.

    Native support requires Blackwell (10.0+).
    """
    cc = get_compute_capability()
    return (10, 0) <= cc < (12, 0)


def check_fp8_support(device_id: int = 0) -> tuple[bool, str, str]:
    """Check if FP8 is supported on the current GPU.

    FP8 requires compute capability 8.9+ (Ada Lovelace/Hopper architecture or newer).

    Returns:
        Tuple of (is_supported, compute_capability_string, device_info_message).
    """
    if not torch.cuda.is_available():
        return False, "0.0", "CUDA not available"
    device_props = torch.cuda.get_device_properties(device_id)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    device_name = device_props.name
    # FP8 is supported on compute capability 8.9+ (Ada Lovelace/Hopper architecture)
    is_supported = (device_props.major > 8) or (device_props.major == 8 and device_props.minor >= 9)
    return is_supported, compute_capability, f"Device: {device_name}, Compute Capability: {compute_capability}"


def is_a6000_gpu() -> bool:
    """Check if any of the visible GPUs is an A6000."""
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        if "A6000" in device_name:
            return True
    return False


def _reset_microbatch_calculator():
    """Resets _GLOBAL_NUM_MICROBATCHES_CALCULATOR in megatron.

    This is used in NeMo to initialize model parallel in
    nemo.collections.nlp.modules.common.megatron.megatron_init.initialize_model_parallel_for_nemo
    """
    megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


def clean_up_distributed_and_parallel_states(verify_distributed_state: bool = False):
    """Clean up parallel states, torch.distributed and torch cuda cache."""
    _reset_microbatch_calculator()
    # Destroy Megatron distributed/parallel state environment.
    parallel_state.destroy_model_parallel()
    # Destroy the torch default / world process group.
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # Clear torch.compile/dynamo cache
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        if hasattr(torch, "compiler"):
            torch.compiler.reset()
    except Exception as e:
        print(f"Failed to reset torch compile: {e}")
    # Free unused CPU memory.
    gc.collect()
    # Free reserved / cached GPU memory allocated by Torch / CUDA.
    torch.cuda.empty_cache()
    if verify_distributed_state:
        # Utilize to debug OOM or orphaned processes in GPU.
        allocated_vram = torch.cuda.memory_allocated() / 1024**3
        reserved_vram = torch.cuda.memory_reserved() / 1024**3
        print(
            "\n--------------------------------\n"
            f"Memory Profile for Device: {torch.cuda.current_device()}\n"
            f"Allocated: {allocated_vram} GB\n"
            f"Reserved: {reserved_vram} GB\n"
            f"GPU Processes:\n{torch.cuda.list_gpu_processes()}\n"
            "--------------------------------\n"
        )


@contextmanager
def clean_parallel_state_context():
    """Puts you into a clean parallel state, and again tears it down at the end."""
    try:
        clean_up_distributed_and_parallel_states()
        yield
    finally:
        clean_up_distributed_and_parallel_states()


@contextmanager
def distributed_model_parallel_state(
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    backend: str = "nccl",
    **initialize_model_parallel_kwargs,
):
    """Context manager for torch distributed and parallel state testing.

    This context manager properly initializes and tears down torch.distributed
    and Megatron's parallel state for testing. It uses MonkeyPatch to scope
    environment variable changes, avoiding stale state between tests.

    Args:
        seed: Random seed to be passed into tensor_parallel.random. Default 42.
        rank: Global rank of the current cuda device. Default 0.
        world_size: World size or number of devices. Default 1.
        backend: Backend to torch.distributed.init_process_group. Default 'nccl'.
        **initialize_model_parallel_kwargs: Kwargs passed to initialize_model_parallel.
    """
    with MonkeyPatch.context() as context:
        initial_states = None
        try:
            clean_up_distributed_and_parallel_states()

            # distributed and parallel state set up
            if not torch.distributed.is_initialized():
                context.setenv("MASTER_ADDR", DEFAULT_MASTER_ADDR)
                free_network_port = find_free_network_port()
                context.setenv(
                    "MASTER_PORT", str(free_network_port) if free_network_port is not None else DEFAULT_MASTER_PORT
                )
                context.setenv("NCCL_TIMEOUT", DEFAULT_NCCL_TIMEOUT)
                context.setenv("RANK", str(rank))

                torch.distributed.init_process_group(backend=backend, world_size=world_size)
            parallel_state.initialize_model_parallel(**initialize_model_parallel_kwargs)

            # tensor parallel random seed set up
            # do not call torch.cuda.manual_seed after so!
            if tp_random.get_cuda_rng_tracker().is_initialized():
                initial_states = tp_random.get_cuda_rng_tracker().get_states()
            if seed is not None:
                tp_random.model_parallel_cuda_manual_seed(seed)

            yield
        finally:
            # restore/unset tensor parallel random seed
            if initial_states is not None:
                tp_random.get_cuda_rng_tracker().set_states(initial_states)
            else:
                # Reset to the unset state
                tp_random.get_cuda_rng_tracker().reset()

            clean_up_distributed_and_parallel_states()
