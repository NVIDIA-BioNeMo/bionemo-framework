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

"""Structured JSONL metric emitter for low-latency agent monitoring.

Writes machine-readable metric records to a JSONL file on shared NFS so
the agent daemon can tail them with sub-second latency (bypassing the
wandb API round-trip). Designed to be called from PerfLogger alongside
the existing wandb logging path.
"""

import json
import logging
import math
import os
import time
from pathlib import Path


logger = logging.getLogger(__name__)


class AgentMetricWriter:
    """Appends one JSON line per logging window to an NFS-backed JSONL file.

    Only the global-rank-0 process writes; other ranks are no-ops.

    Args:
        output_path: JSONL file path on shared storage.
        enabled: Master switch.  When False every method is a no-op.
        is_main_process: True only on global rank 0.
    """

    def __init__(
        self,
        output_path: str | os.PathLike = "/data/agent/metrics.jsonl",
        enabled: bool = False,
        is_main_process: bool = False,
    ):
        """Initialize the metric writer."""
        self._enabled = enabled and is_main_process
        self._path = Path(output_path)
        self._prev_loss: float | None = None

        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("AgentMetricWriter enabled → %s", self._path)

    def write_step(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        lr: float,
        step_time: float,
        tokens_per_sec: float,
        gpu_mem_gb: float,
    ) -> None:
        """Append a single metric record for the current logging window.

        Args:
            step: Global optimizer step.
            loss: Average loss over the logging window.
            grad_norm: Gradient norm after clipping.
            lr: Current learning rate.
            step_time: Wall-clock seconds per step (averaged over window).
            tokens_per_sec: Tokens/sec/GPU throughput.
            gpu_mem_gb: GPU memory allocated in GiB.
        """
        if not self._enabled:
            return

        loss_delta_pct = 0.0
        if self._prev_loss is not None and self._prev_loss != 0.0:
            loss_delta_pct = ((loss - self._prev_loss) / abs(self._prev_loss)) * 100.0
        self._prev_loss = loss

        record = {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "step_time": step_time,
            "tokens_per_sec": tokens_per_sec,
            "gpu_mem_gb": round(gpu_mem_gb, 3),
            "loss_delta_pct": round(loss_delta_pct, 2),
            "nan_detected": math.isnan(loss) or math.isnan(grad_norm),
            "grad_norm_spike": False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            logger.exception("Failed to write agent metric record")

    def write_intervention(self, step: int, interventions: dict) -> None:
        """Log an agent intervention event alongside normal metrics.

        Args:
            step: Global optimizer step at which the intervention was applied.
            interventions: The dict of parameter overrides that were applied.
        """
        if not self._enabled:
            return

        record = {
            "step": step,
            "type": "intervention",
            "interventions": interventions,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            logger.exception("Failed to write agent intervention record")
