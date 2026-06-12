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


class ProcessGroupManager:
    """Manages data-parallel and tensor-parallel process groups for distributed training."""

    def __init__(self, dp_size: int = 1, tp_size: int = 1):
        """Initialize process groups with the given data-parallel and tensor-parallel sizes."""
        print("=" * 100)
        self.world_size = dist.get_world_size()
        print(f"World size: {self.world_size}")
        print(f"DP size: {dp_size}")
        print(f"TP size: {tp_size}")
        # ensure world size is divisible by dp_size and tp_size
        assert self.world_size == dp_size * tp_size, (
            f"World size ({self.world_size}) != DP ({dp_size}) * TP ({tp_size})"
        )

        self.global_rank = dist.get_rank()  # e.g. 1-16
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % self.world_size))  # e.g. 1-8

        self.grid = torch.arange(self.world_size).view(dp_size, tp_size)  # dp * tp

        # position of curent rank in grid
        self.dp_rank, self.tp_rank = (self.grid == self.global_rank).nonzero().flatten().tolist()
        print(f"Global rank: {self.global_rank}")
        print(f"DP rank: {self.dp_rank}")
        print(f"TP rank: {self.tp_rank}")

        # create process groups
        self.dp_group = dist.new_subgroups_by_enumeration([self.grid[:, t].tolist() for t in range(tp_size)])[0]
        self.tp_group = dist.new_subgroups_by_enumeration([self.grid[d, :].tolist() for d in range(dp_size)])[0]

        self.world_group = dist.group.WORLD

        # group ids
        self.dp_group_ids = self.grid[:, self.tp_rank].tolist()
        self.tp_group_ids = self.grid[self.dp_rank, :].tolist()

        # tensor parallel
        self.tp_world_size = dist.get_world_size(group=self.tp_group)
        self.tp_first_rank = self.tp_group_ids[0]
        self.tp_last_rank = self.tp_group_ids[-1]

        # data parallel
        self.dp_world_size = dist.get_world_size(group=self.dp_group)
        self.dp_first_rank = self.dp_group_ids[0]
        self.dp_last_rank = self.dp_group_ids[-1]

        print(f"DP group ids: {self.dp_group_ids}")
        print(f"TP group ids: {self.tp_group_ids}")
        print(f"DP first rank: {self.dp_first_rank}")
        print(f"DP last rank: {self.dp_last_rank}")
        print(f"TP first rank: {self.tp_first_rank}")
        print(f"TP last rank: {self.tp_last_rank}")

        print(f"DP rank: {self.dp_rank}")
        print(f"TP rank: {self.tp_rank}")

        # # setup dp
        # self.dp_group = dist.new_group(ranks=list(range(self.world_size)))
        # # get this rank's position in the dp group
        # self.dp_rank = dist.get_rank(group=self.dp_group)
        # self.dp_world_size = dist.get_world_size(group=self.dp_group)

    def is_main_process(self) -> bool:
        """Return True if this is the main process (global rank 0)."""
        return self.global_rank == 0

    def __str__(self):
        """Return string representation of the process group configuration."""
        return f"DP:{self.dp_world_size} | TP:{self.tp_world_size}"
