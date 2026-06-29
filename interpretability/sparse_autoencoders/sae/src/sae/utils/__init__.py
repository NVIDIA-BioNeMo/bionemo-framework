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

# Seed and device utilities
import random
import subprocess

import numpy as np
import torch


def get_device() -> str:
    """Get available device.

    Returns:
        str: Available device
    """
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def set_seed(seed: int):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_file_limit() -> int:
    """Determine the system's maximum number of open files limit.

    Uses the 'ulimit -n' command to get the system limit and falls back to a
    conservative default if the command fails.

    Returns:
        int: Maximum number of files that can be opened simultaneously
    """
    try:
        result = subprocess.run(["ulimit", "-n"], capture_output=True, text=True, shell=True)
        return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return 1024  # Default to a conservative value


# Memory utilities
from .memory import (
    sae_backward_memory,
    sae_forward_memory,
    sae_total_memory,
    sae_weight_memory,
)


__all__ = [
    "get_device",
    "get_file_limit",
    "sae_backward_memory",
    "sae_forward_memory",
    "sae_total_memory",
    "sae_weight_memory",
    "set_seed",
]
