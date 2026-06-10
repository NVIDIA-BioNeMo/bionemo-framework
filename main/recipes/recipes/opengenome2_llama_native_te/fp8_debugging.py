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

import logging
import os
from pathlib import Path

import nvdlfw_inspect.api as debug_api
import transformer_engine

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize_fp8_debugging(
    dist_config: DistributedConfig,
    enabled: bool,
    fp8_stats_file: str,
    fp8_log_dir: str | os.PathLike,
    fp8_enabled: bool,
) -> None:
    """Initialize FP8 debugging.

    Args:
        dist_config: The distributed configuration.
        enabled: Whether to enable FP8 debugging.
        fp8_stats_file: The file containing the FP8 stats.
        fp8_log_dir: The directory to log the FP8 stats to.
        fp8_enabled: Whether FP8 autocast is enabled.
    """
    if not enabled:
        return

    if not fp8_enabled:
        raise ValueError(
            "fp8_stats_config.enabled is true but fp8_config.enabled is false, "
            "please set fp8_config.enabled to true in the config if you wish to collect FP8 stats"
        )

    fp8_log_dir = Path(fp8_log_dir) / f"rank_{dist_config.rank}"
    fp8_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logging FP8 stats to {fp8_log_dir}")
    te_features_dir = str(Path(transformer_engine.__file__).parent / "debug" / "features")
    debug_api.initialize(
        config_file=fp8_stats_file,
        feature_dirs=[te_features_dir],
        log_dir=fp8_log_dir.as_posix(),
        default_logging_enabled=True,
    )
