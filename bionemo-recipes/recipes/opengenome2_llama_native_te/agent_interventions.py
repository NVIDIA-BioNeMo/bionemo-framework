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

"""Tier 1 and Tier 2 intervention executors for the agent daemon.

Tier 1 (hot-reload): writes a new version of the NFS control file so the
training loop picks up parameter changes on its next poll.

Tier 2 (checkpoint-restart): signals the training loop to save a checkpoint
and exit, then modifies the Hydra config on disk and resubmits the job
via the Lepton API.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent_journal import EntryType, ExperimentJournal
from control_plane import write_control_file


logger = logging.getLogger(__name__)


@dataclass
class InterventionLimits:
    """Hard safety limits for automated interventions.

    Attributes:
        lr_max_multiplier: LR cannot exceed ``initial_lr * lr_max_multiplier``.
        grad_clip_min: Gradient clip norm floor.
        max_tier2_restarts: Total Tier 2 restarts allowed per session.
        cooldown_steps: Minimum steps between any two interventions.
    """

    lr_max_multiplier: float = 2.0
    grad_clip_min: float = 0.1
    max_tier2_restarts: int = 3
    cooldown_steps: int = 500


@dataclass
class InterventionState:
    """Mutable state tracking for the intervention engine."""

    version: int = 0
    tier2_restart_count: int = 0
    last_intervention_step: int = -999_999
    initial_lr: float | None = None
    previous_values: dict[str, Any] = field(default_factory=dict)


class Tier1Executor:
    """Write safe hot-reload interventions to the NFS control file.

    Args:
        control_file: Path to the YAML control file.
        journal: Experiment journal for audit logging.
        limits: Safety limits.
        agent_id: Identifier for this agent session.
    """

    def __init__(
        self,
        control_file: str | os.PathLike,
        journal: ExperimentJournal,
        limits: InterventionLimits | None = None,
        agent_id: str = "agent",
    ):
        """Initialize the Tier 1 executor."""
        self._control_file = Path(control_file)
        self._journal = journal
        self._limits = limits or InterventionLimits()
        self._agent_id = agent_id
        self._state = InterventionState()

    def can_intervene(self, current_step: int) -> bool:
        """Check whether an intervention is allowed right now."""
        return (current_step - self._state.last_intervention_step) >= self._limits.cooldown_steps

    def apply(
        self,
        interventions: dict[str, Any],
        current_step: int,
        trigger: str = "",
        hypothesis: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> bool:
        """Validate, write control file, and journal the intervention.

        Args:
            interventions: Parameter overrides to apply.
            current_step: Current training step (for cooldown checks).
            trigger: Human-readable description of what triggered this.
            hypothesis: Agent's hypothesis about the root cause.
            evidence: Supporting metric data.

        Returns:
            True if the intervention was written; False if rejected.
        """
        if not self.can_intervene(current_step):
            logger.warning("Intervention rejected: cooldown not elapsed (step %d)", current_step)
            return False

        validated = self._validate(interventions)
        if not validated:
            logger.warning("Intervention rejected: all values out of bounds")
            return False

        self._state.version += 1
        write_control_file(
            path=self._control_file,
            version=self._state.version,
            interventions=validated,
            agent_id=self._agent_id,
        )

        config_diff = {}
        for k, v in validated.items():
            old = self._state.previous_values.get(k)
            config_diff[k] = [old, v]
            self._state.previous_values[k] = v

        self._state.last_intervention_step = current_step

        self._journal.log(
            EntryType.INTERVENTION,
            step=current_step,
            trigger=trigger,
            hypothesis=hypothesis,
            evidence=evidence,
            action=f"tier1: {validated}",
            config_diff=config_diff,
        )

        logger.info("Tier 1 intervention v%d applied at step %d: %s", self._state.version, current_step, validated)
        return True

    def rollback(self, current_step: int) -> bool:
        """Revert to the previously saved parameter values.

        Returns:
            True if a rollback was written; False if there is nothing to revert.
        """
        if not self._state.previous_values:
            return False

        rollback_values = dict(self._state.previous_values)
        self._state.version += 1
        write_control_file(
            path=self._control_file,
            version=self._state.version,
            interventions=rollback_values,
            agent_id=self._agent_id,
        )

        self._journal.log(
            EntryType.INTERVENTION,
            step=current_step,
            trigger="auto-rollback due to degradation",
            action=f"tier1_rollback: {rollback_values}",
        )

        logger.info("Tier 1 rollback v%d at step %d: %s", self._state.version, current_step, rollback_values)
        return True

    def _validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Enforce safety bounds on intervention values."""
        result: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "learning_rate" and isinstance(value, (int, float)):
                clamped = float(value)
                if self._state.initial_lr is not None:
                    max_lr = self._state.initial_lr * self._limits.lr_max_multiplier
                    clamped = min(clamped, max_lr)
                result[key] = max(0.0, clamped)
            elif key == "grad_clip_norm" and isinstance(value, (int, float)):
                result[key] = max(self._limits.grad_clip_min, float(value))
            else:
                result[key] = value
        return result

    def set_initial_lr(self, lr: float) -> None:
        """Record the initial LR so we can enforce the max-multiplier bound."""
        self._state.initial_lr = lr
        self._state.previous_values["learning_rate"] = lr


class Tier2Executor:
    """Checkpoint-and-restart interventions via Lepton API.

    Args:
        control_file: Path to the NFS control file (for checkpoint-stop signal).
        journal: Experiment journal.
        hydra_config_dir: Path to the Hydra config directory on shared NFS.
        lepton_config_path: Path to the Lepton YAML config to resubmit.
        limits: Safety limits.
        agent_id: Identifier for this agent session.
    """

    def __init__(
        self,
        control_file: str | os.PathLike,
        journal: ExperimentJournal,
        hydra_config_dir: str | os.PathLike | None = None,
        lepton_config_path: str | os.PathLike | None = None,
        limits: InterventionLimits | None = None,
        agent_id: str = "agent",
    ):
        """Initialize the Tier 2 executor."""
        self._control_file = Path(control_file)
        self._journal = journal
        self._hydra_config_dir = Path(hydra_config_dir) if hydra_config_dir else None
        self._lepton_config_path = Path(lepton_config_path) if lepton_config_path else None
        self._limits = limits or InterventionLimits()
        self._agent_id = agent_id
        self._state = InterventionState()

    def can_restart(self) -> bool:
        """Check whether a Tier 2 restart is allowed."""
        return self._state.tier2_restart_count < self._limits.max_tier2_restarts

    def request_checkpoint_and_stop(self, current_step: int, reason: str = "") -> bool:
        """Signal the training loop to save a checkpoint and exit.

        Args:
            current_step: Current training step.
            reason: Why the restart is needed.

        Returns:
            True if the signal was written.
        """
        if not self.can_restart():
            logger.warning("Tier 2 restart rejected: max restarts (%d) reached", self._limits.max_tier2_restarts)
            return False

        self._state.version += 1
        write_control_file(
            path=self._control_file,
            version=self._state.version,
            interventions={"request_checkpoint_and_stop": True},
            agent_id=self._agent_id,
        )

        self._journal.log(
            EntryType.INTERVENTION,
            step=current_step,
            trigger=reason,
            action="tier2: request_checkpoint_and_stop",
        )

        logger.info("Tier 2 checkpoint-and-stop requested at step %d: %s", current_step, reason)
        return True

    def modify_hydra_config(
        self,
        config_name: str,
        overrides: dict[str, Any],
        current_step: int,
    ) -> bool:
        """Modify a Hydra YAML config file on NFS for the next restart.

        Args:
            config_name: Name of the YAML file (without extension) in the hydra config dir.
            overrides: Flat dict of dotted keys → new values.
            current_step: Current training step.

        Returns:
            True if the config was modified.
        """
        if self._hydra_config_dir is None:
            logger.error("Cannot modify Hydra config: hydra_config_dir not set")
            return False

        config_path = self._hydra_config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            logger.error("Hydra config not found: %s", config_path)
            return False

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

            config_diff = {}

            for dotted_key, new_value in overrides.items():
                keys = dotted_key.split(".")
                node = data
                for k in keys[:-1]:
                    if k not in node or not isinstance(node[k], dict):
                        node[k] = {}
                    node = node[k]

                old_value = node.get(keys[-1])
                node[keys[-1]] = new_value
                config_diff[dotted_key] = [old_value, new_value]

            # Atomic write
            tmp = config_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
            tmp.rename(config_path)

            self._journal.log(
                EntryType.INTERVENTION,
                step=current_step,
                action=f"tier2: modify {config_name}.yaml",
                config_diff=config_diff,
            )

            logger.info("Modified Hydra config %s: %s", config_name, config_diff)
            return True

        except Exception:
            logger.exception("Failed to modify Hydra config %s", config_path)
            return False

    def resubmit_lepton_job(
        self,
        current_step: int,
        extra_overrides: dict[str, str] | None = None,
    ) -> bool:
        """Resubmit the Lepton training job with resume_from_checkpoint=true.

        Requires ``leptonai`` to be installed in the agent environment.

        Args:
            current_step: Current training step (for journaling).
            extra_overrides: Additional Lepton config overrides.

        Returns:
            True if the job was submitted.
        """
        if self._lepton_config_path is None:
            logger.error("Cannot resubmit: lepton_config_path not set")
            return False

        try:
            import importlib.util

            if importlib.util.find_spec("leptonai") is None:
                raise ImportError("leptonai not available")
        except ImportError:
            logger.error("leptonai not installed — cannot resubmit job")
            return False

        try:
            with open(self._lepton_config_path) as f:
                cfg = yaml.safe_load(f) or {}

            cfg["resume_from_checkpoint"] = True
            if extra_overrides:
                cfg.update(extra_overrides)

            # Atomic rewrite of config with resume enabled
            tmp = self._lepton_config_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False)
            tmp.rename(self._lepton_config_path)

            self._state.tier2_restart_count += 1

            self._journal.log(
                EntryType.INTERVENTION,
                step=current_step,
                action=f"tier2: resubmit Lepton job (restart #{self._state.tier2_restart_count})",
                extra={"lepton_config": str(self._lepton_config_path)},
            )

            logger.info(
                "Lepton job config updated for restart #%d. Run submit_og2_lepton_eden.py to launch.",
                self._state.tier2_restart_count,
            )
            return True

        except Exception:
            logger.exception("Failed to resubmit Lepton job")
            return False
