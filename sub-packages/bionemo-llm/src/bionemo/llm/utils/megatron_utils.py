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

import torch.distributed
from megatron.core import parallel_state


def is_only_data_parallel() -> bool:
    """Checks to see if you are in a distributed megatron environment with only data parallelism active.

    This is useful if you are working on a model, loss, etc and you know that you do not yet support megatron model
    parallelism. You can test that the only kind of parallelism in use is data parallelism.

    Returns:
        True if data parallel is the only parallel mode, False otherwise.
    """
    if not (torch.distributed.is_available() and parallel_state.is_initialized()):
        raise RuntimeError("This function is only defined within an initialized megatron parallel environment.")
    # Idea: when world_size == data_parallel_world_size, then you know that you are fully DDP, which means you are not
    #  using model parallelism (meaning virtual GPUs composed of several underlying GPUs that you need to reduce over).

    world_size: int = torch.distributed.get_world_size()
    dp_world_size: int = parallel_state.get_data_parallel_world_size()
    return world_size == dp_world_size


def average_losses_across_data_parallel_group(losses, with_context_parallel: bool = False):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    # Reduce across the DP (or optionally, the flattened DP + CP) group.
    # Refer to the ring attention algorithm on why we always must reduce across the CP group.
    torch.distributed.all_reduce(
        averaged_losses, group=parallel_state.get_data_parallel_group(with_context_parallel=with_context_parallel)
    )
    averaged_losses = averaged_losses / torch.distributed.get_world_size(
        # Only average losses across the data parallel group, not the context parallel group!
        group=parallel_state.get_data_parallel_group()
    )
    return averaged_losses
