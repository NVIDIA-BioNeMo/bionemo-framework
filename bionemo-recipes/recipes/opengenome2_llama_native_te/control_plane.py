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

"""NFS-based control plane for live hot-reload interventions during training.

Polls a YAML control file on shared storage every N training steps. Only rank 0
reads the file; updates are broadcast to all ranks via torch.distributed. A
monotonic version number prevents re-applying the same intervention.

Tier 1 (hot-reload) changes — learning rate, grad clip, FP8 recipe knobs,
logging toggles — flow through this module. Tier 2 changes (checkpoint-and-
restart) are signaled via the ``request_checkpoint_and_stop`` field.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.distributed as dist
import yaml


logger = logging.getLogger(__name__)


# Hard safety bounds for Tier 1 hot-reload parameters
_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (0.0, 1.0),
    "grad_clip_norm": (0.1, 100.0),
    "logging_frequency": (1, 10_000),
}


@dataclass
class ControlPlaneConfig:
    """Configuration for the control plane.

    Attributes:
        enabled: Master switch for the control plane.
        control_file: Path to the YAML control file on shared NFS.
        poll_every_n_steps: How often (in optimizer steps) to check the file.
        bounds: Per-parameter (min, max) safety bounds for Tier 1 interventions.
        cooldown_steps: Minimum steps between successive interventions.
    """

    enabled: bool = False
    control_file: str = "/data/agent/control.yaml"
    poll_every_n_steps: int = 1
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(_DEFAULT_BOUNDS))
    cooldown_steps: int = 500


class ControlPlane:
    """Reads a shared YAML control file and distributes interventions to all ranks.

    The control file must contain a monotonically increasing ``version`` field.
    When the version changes the new ``interventions`` block is consumed.

    Args:
        config: ControlPlaneConfig instance.
        rank: The global rank of this process.
    """

    def __init__(self, config: ControlPlaneConfig, rank: int = 0):
        """Initialize the control plane."""
        self._config = config
        self._rank = rank
        self._last_version: int = -1
        self._last_mtime: float = 0.0
        self._last_poll_step: int = -999_999
        self._last_intervention_step: int = -999_999
        self._pending: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public API called from the training loop
    # ------------------------------------------------------------------

    def poll(self, step: int) -> None:
        """Check for a new control file version (respects poll frequency).

        Call once per optimizer step. Actual I/O only happens every
        ``poll_every_n_steps`` steps on rank 0.

        Args:
            step: Current global training step.
        """
        if not self._config.enabled:
            return

        if step - self._last_poll_step < self._config.poll_every_n_steps:
            return

        self._last_poll_step = step
        interventions = self._read_and_broadcast()

        if interventions is not None:
            if step - self._last_intervention_step < self._config.cooldown_steps:
                logger.warning(
                    "Control plane update at step %d ignored — cooldown (%d steps) not elapsed since last "
                    "intervention at step %d",
                    step,
                    self._config.cooldown_steps,
                    self._last_intervention_step,
                )
                return
            self._pending = interventions
            self._last_intervention_step = step

    def has_update(self) -> bool:
        """Return True if there is an unconsumed intervention."""
        return self._pending is not None

    def consume(self) -> dict[str, Any]:
        """Return and clear the pending intervention dict.

        Raises:
            RuntimeError: If there is no pending update.
        """
        if self._pending is None:
            raise RuntimeError("No pending control-plane update to consume")
        result = self._pending
        self._pending = None
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_and_broadcast(self) -> dict[str, Any] | None:
        """Rank 0 reads the file; result is broadcast to all ranks."""
        payload: list[dict[str, Any] | None] = [None]

        if self._rank == 0:
            payload[0] = self._try_read_file()

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast_object_list(payload, src=0)

        return payload[0]

    def _try_read_file(self) -> dict[str, Any] | None:
        """Read the control YAML if it changed on disk. Returns validated interventions or None."""
        path = Path(self._config.control_file)
        if not path.exists():
            return None

        try:
            st = path.stat()
        except OSError:
            return None

        if st.st_mtime == self._last_mtime:
            return None

        self._last_mtime = st.st_mtime

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except Exception:
            logger.exception("Failed to parse control file %s", path)
            return None

        if not isinstance(data, dict):
            return None

        version = data.get("version", 0)
        if version <= self._last_version:
            return None

        self._last_version = version
        interventions = data.get("interventions", {})
        if not interventions:
            return None

        validated = self._validate(interventions)
        logger.info("Control plane v%d consumed: %s", version, validated)
        return validated

    def _validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Clamp numeric parameters to configured safety bounds."""
        result: dict[str, Any] = {}
        for key, value in raw.items():
            if key in self._config.bounds and isinstance(value, (int, float)):
                lo, hi = self._config.bounds[key]
                clamped = max(lo, min(hi, float(value)))
                if clamped != float(value):
                    logger.warning("Clamped %s from %s to %s (bounds [%s, %s])", key, value, clamped, lo, hi)
                result[key] = clamped
            else:
                result[key] = value
        return result


def write_control_file(
    path: str | os.PathLike,
    version: int,
    interventions: dict[str, Any],
    agent_id: str = "manual",
) -> None:
    """Write a control file atomically (for use by the agent or manual debugging).

    Args:
        path: Destination path for the YAML control file.
        version: Monotonically increasing version number.
        interventions: Dict of parameter overrides.
        agent_id: Identifier for the writing agent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent_id": agent_id,
        "interventions": interventions,
    }

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False)
    tmp.rename(path)
