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

from pathlib import Path
from typing import Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def verify_tensorboard_logs(tb_log_dir: Path, expected_metrics: list[str], min_steps: int = 1) -> Optional[str]:
    """Verify that TensorBoard logs exist and contain expected metrics.

    Args:
        tb_log_dir: Path to the TensorBoard log directory
        expected_metrics: List of metric names expected in the logs
        min_steps: Minimum number of steps expected in the logs

    Returns:
        None if verification succeeds, error message string if it fails
    """
    # Find event files in the log directory
    event_files = list(tb_log_dir.glob("events.out.tfevents.*"))
    if len(event_files) == 0:
        return f"No TensorBoard event files found in {tb_log_dir}"

    # Load the event file
    event_acc = EventAccumulator(str(tb_log_dir))
    event_acc.Reload()

    # Get available scalar tags
    scalar_tags = event_acc.Tags()["scalars"]

    # Check that expected metrics are present
    for metric in expected_metrics:
        # Check if metric exists in any form (might have prefixes like "train/" or suffixes)
        metric_found = any(metric in tag for tag in scalar_tags)
        if not metric_found:
            return f"Expected metric '{metric}' not found in TensorBoard logs. Available tags: {scalar_tags}"

    # Verify we have logged data for at least min_steps
    if scalar_tags:
        # Get the first available metric to check step count
        first_metric = scalar_tags[0]
        events = event_acc.Scalars(first_metric)
        if len(events) < min_steps:
            return f"Expected at least {min_steps} steps logged, but found {len(events)}"

    return None
