# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import torch
import torch.distributed as dist

from sae.process_group_manager import ProcessGroupManager


def setup_dist():
    """Initialize distributed process group and return local rank."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    """Run a demo of the ProcessGroupManager with distributed setup."""
    local_rank = setup_dist()
    world_size = dist.get_world_size()
    print(f"World size: {world_size}")
    print(f"Local rank: {local_rank}")
    _is_on_local_rank = local_rank == 0
    _device = torch.device(f"cuda:{local_rank}")
    _dtype = torch.bfloat16
    pg = ProcessGroupManager(dp_size=2, tp_size=2)
    print(pg)


if __name__ == "__main__":
    main()
