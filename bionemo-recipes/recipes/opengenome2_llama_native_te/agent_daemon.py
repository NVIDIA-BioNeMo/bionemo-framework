#!/usr/bin/env python3

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

r"""AI-in-the-loop FP8 precision debugging daemon.

Runs as a sidecar Lepton job on a CPU-only node with shared NFS access.
Monitors training metrics and FP8 stats, detects anomalies, and either
applies Tier 1 hot-reload fixes (via the NFS control file) or proposes
Tier 2 checkpoint-and-restart changes.

Usage (standalone)::

    python agent_daemon.py --config /data/agent/agent_config.yaml

Usage (as Lepton sidecar):
    Launched automatically by ``submit_og2_lepton_eden.py`` when
    ``agent.enabled=true`` is set in the Lepton config.
"""

import argparse
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent_fp8_analyzer import FP8StatsAnalyzer
from agent_interventions import InterventionLimits, Tier1Executor, Tier2Executor
from agent_journal import EntryType, ExperimentJournal


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("agent_daemon")


@dataclass
class AgentConfig:
    """Daemon configuration loaded from YAML or CLI args."""

    metrics_file: str = "/data/agent/metrics.jsonl"
    control_file: str = "/data/agent/control.yaml"
    journal_file: str = "/data/agent/journal.jsonl"
    fp8_stats_dir: str = "./log_fp8_stats"

    monitor_interval_seconds: float = 30.0
    loss_spike_threshold_pct: float = 50.0
    grad_norm_spike_multiplier: float = 3.0
    nan_halt: bool = True
    observation_only: bool = True
    max_tier2_restarts: int = 3
    cooldown_steps: int = 500

    hydra_config_dir: str | None = None
    hydra_config_name: str | None = None
    lepton_config_path: str | None = None

    rolling_window_size: int = 20

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "AgentConfig":
        """Load config from a YAML file, falling back to defaults for missing keys."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MetricSnapshot:
    """A single training metric record parsed from the JSONL stream."""

    step: int = 0
    loss: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    loss_delta_pct: float = 0.0
    nan_detected: bool = False
    timestamp: str = ""


class AgentDaemon:
    """Main monitoring and intervention loop.

    Args:
        config: Agent configuration.
    """

    def __init__(self, config: AgentConfig):
        """Initialize the agent daemon."""
        self._config = config
        self._session_id = f"agent-{uuid.uuid4().hex[:8]}"

        self._journal = ExperimentJournal(
            journal_path=config.journal_file,
            session_id=self._session_id,
        )

        limits = InterventionLimits(
            max_tier2_restarts=config.max_tier2_restarts,
            cooldown_steps=config.cooldown_steps,
        )

        self._tier1 = Tier1Executor(
            control_file=config.control_file,
            journal=self._journal,
            limits=limits,
            agent_id=self._session_id,
        )
        self._tier2 = Tier2Executor(
            control_file=config.control_file,
            journal=self._journal,
            hydra_config_dir=config.hydra_config_dir,
            lepton_config_path=config.lepton_config_path,
            limits=limits,
            agent_id=self._session_id,
        )

        self._fp8_analyzer = FP8StatsAnalyzer(log_dir=config.fp8_stats_dir)

        self._metrics_read_pos: int = 0
        self._loss_history: deque[float] = deque(maxlen=config.rolling_window_size)
        self._grad_norm_history: deque[float] = deque(maxlen=config.rolling_window_size)
        self._last_step: int = 0
        self._last_intervention_entry_id: int | None = None

    def run(self) -> None:
        """Start the blocking monitoring loop."""
        self._journal.log(
            EntryType.SESSION_START,
            extra={
                "session_id": self._session_id,
                "config": dict(self._config.__dict__),
                "observation_only": self._config.observation_only,
            },
        )

        logger.info(
            "Agent daemon started (session=%s, observation_only=%s)", self._session_id, self._config.observation_only
        )

        prior = self._journal.load_history()
        if prior:
            logger.info("Loaded %d prior journal entries for context", len(prior))

        try:
            while True:
                self._monitor_tick()
                time.sleep(self._config.monitor_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Agent daemon interrupted")
        finally:
            self._journal.log(EntryType.SESSION_END, extra={"session_id": self._session_id})

    def _monitor_tick(self) -> None:
        """One iteration of the monitoring loop."""
        snapshots = self._read_new_metrics()
        fp8_report = self._fp8_analyzer.analyze()

        for snap in snapshots:
            self._analyze_snapshot(snap, fp8_report)

        if not fp8_report.healthy:
            self._handle_fp8_anomalies(fp8_report)

    # ------------------------------------------------------------------
    # Metric reading
    # ------------------------------------------------------------------

    def _read_new_metrics(self) -> list[MetricSnapshot]:
        """Tail-read new JSONL records since the last read position."""
        path = Path(self._config.metrics_file)
        if not path.exists():
            return []

        snapshots: list[MetricSnapshot] = []
        try:
            with open(path) as f:
                f.seek(self._metrics_read_pos)
                for raw_line in f:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") == "intervention":
                        continue

                    snap = MetricSnapshot(
                        step=data.get("step", 0),
                        loss=data.get("loss", 0.0),
                        grad_norm=data.get("grad_norm", 0.0),
                        lr=data.get("lr", 0.0),
                        loss_delta_pct=data.get("loss_delta_pct", 0.0),
                        nan_detected=data.get("nan_detected", False),
                        timestamp=data.get("timestamp", ""),
                    )
                    snapshots.append(snap)

                self._metrics_read_pos = f.tell()
        except OSError:
            logger.debug("Could not read metrics file", exc_info=True)

        return snapshots

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def _analyze_snapshot(self, snap: MetricSnapshot, fp8_report: Any) -> None:
        """Check a single metric snapshot for anomalies and decide on interventions."""
        self._last_step = snap.step

        if snap.nan_detected:
            self._handle_nan(snap)
            return

        self._loss_history.append(snap.loss)
        self._grad_norm_history.append(snap.grad_norm)

        if self._detect_loss_spike(snap):
            self._handle_loss_spike(snap)

        if self._detect_grad_norm_spike(snap):
            self._handle_grad_norm_spike(snap)

        self._journal.log(
            EntryType.OBSERVATION,
            step=snap.step,
            extra={
                "loss": snap.loss,
                "grad_norm": snap.grad_norm,
                "lr": snap.lr,
                "loss_delta_pct": snap.loss_delta_pct,
                "fp8_healthy": fp8_report.healthy if hasattr(fp8_report, "healthy") else True,
            },
        )

    def _detect_loss_spike(self, snap: MetricSnapshot) -> bool:
        """True if loss jumped significantly vs rolling average."""
        if len(self._loss_history) < 3:
            return False
        recent = list(self._loss_history)[-5:]
        avg = sum(recent[:-1]) / len(recent[:-1])
        if avg == 0:
            return False
        pct_change = ((snap.loss - avg) / abs(avg)) * 100.0
        return pct_change > self._config.loss_spike_threshold_pct

    def _detect_grad_norm_spike(self, snap: MetricSnapshot) -> bool:
        """True if grad norm exceeds N * rolling average."""
        if len(self._grad_norm_history) < 3:
            return False
        recent = list(self._grad_norm_history)[-5:]
        avg = sum(recent[:-1]) / len(recent[:-1])
        if avg == 0:
            return False
        return snap.grad_norm > avg * self._config.grad_norm_spike_multiplier

    # ------------------------------------------------------------------
    # Intervention handlers
    # ------------------------------------------------------------------

    def _handle_nan(self, snap: MetricSnapshot) -> None:
        """React to NaN loss or gradient norm."""
        logger.error("NaN detected at step %d", snap.step)
        self._journal.log(
            EntryType.OBSERVATION,
            step=snap.step,
            trigger=f"NaN detected at step {snap.step}",
            evidence={"loss": snap.loss, "grad_norm": snap.grad_norm},
        )

        if self._config.nan_halt and not self._config.observation_only:
            logger.info("Requesting checkpoint-and-stop due to NaN")
            self._tier2.request_checkpoint_and_stop(
                snap.step,
                reason=f"NaN detected at step {snap.step}",
            )

    def _handle_loss_spike(self, snap: MetricSnapshot) -> None:
        """React to a sudden loss spike."""
        recent = list(self._loss_history)[-5:]
        avg = sum(recent[:-1]) / len(recent[:-1]) if len(recent) > 1 else snap.loss
        pct = ((snap.loss - avg) / abs(avg)) * 100.0 if avg != 0 else 0

        trigger = f"Loss spike at step {snap.step}: {snap.loss:.4f} ({pct:+.1f}% vs rolling avg {avg:.4f})"
        logger.warning(trigger)

        self._journal.log(
            EntryType.OBSERVATION,
            step=snap.step,
            trigger=trigger,
            evidence={"loss": snap.loss, "rolling_avg": avg, "pct_change": pct},
        )

        if self._config.observation_only:
            return

        if self._tier1.can_intervene(snap.step):
            new_clip = max(0.1, snap.grad_norm * 0.5)
            self._journal.log(
                EntryType.HYPOTHESIS,
                step=snap.step,
                hypothesis="Loss spike likely from gradient explosion; reducing grad clip",
            )
            self._tier1.apply(
                {"grad_clip_norm": new_clip},
                current_step=snap.step,
                trigger=trigger,
                hypothesis="Reducing grad clip to stabilize",
            )

    def _handle_grad_norm_spike(self, snap: MetricSnapshot) -> None:
        """React to a gradient norm spike."""
        recent = list(self._grad_norm_history)[-5:]
        avg = sum(recent[:-1]) / len(recent[:-1]) if len(recent) > 1 else snap.grad_norm

        trigger = (
            f"Grad norm spike at step {snap.step}: {snap.grad_norm:.4f} "
            f"({snap.grad_norm / avg:.1f}x rolling avg {avg:.4f})"
        )
        logger.warning(trigger)

        self._journal.log(
            EntryType.OBSERVATION,
            step=snap.step,
            trigger=trigger,
            evidence={"grad_norm": snap.grad_norm, "rolling_avg": avg},
        )

    def _handle_fp8_anomalies(self, fp8_report: Any) -> None:
        """React to FP8 stat anomalies (overflow, underflow, instability)."""
        summary = fp8_report.summary()
        logger.warning("FP8 anomalies detected: %s", summary)

        self._journal.log(
            EntryType.OBSERVATION,
            step=self._last_step,
            trigger=f"FP8 anomalies: {len(summary.get('anomalies', []))} issues",
            evidence=summary,
        )

        if self._config.observation_only:
            return

        overflow_layers = summary.get("overflow_layers", [])
        if overflow_layers and self._tier2.can_restart():
            layer_indices = self._extract_layer_indices(overflow_layers)
            if layer_indices:
                min_idx = min(layer_indices)

                hypothesis = (
                    f"FP8 overflow in layers {overflow_layers} — proposing to extend BF16 buffer to cover these layers"
                )
                self._journal.log(
                    EntryType.HYPOTHESIS,
                    step=self._last_step,
                    hypothesis=hypothesis,
                    evidence=summary,
                )

                n_start_bf16 = min_idx + 1 if min_idx < 4 else 1
                logger.info(
                    "FP8 overflow: would propose num_layers_at_start_in_bf16=%d (requires human approval for Tier 2)",
                    n_start_bf16,
                )

    @staticmethod
    def _extract_layer_indices(layer_names: list[str]) -> list[int]:
        """Extract numeric layer indices from layer name strings like 'model.layers.3.self_attn'."""
        import re

        indices = []
        for name in layer_names:
            match = re.search(r"layers?[._](\d+)", name)
            if match:
                indices.append(int(match.group(1)))
        return indices


def main() -> None:
    """CLI entry point for the agent daemon."""
    parser = argparse.ArgumentParser(description="AI-in-the-loop FP8 debugging daemon")
    parser.add_argument("--config", type=str, default=None, help="Path to agent config YAML")
    parser.add_argument("--metrics-file", type=str, default=None)
    parser.add_argument("--control-file", type=str, default=None)
    parser.add_argument("--journal-file", type=str, default=None)
    parser.add_argument("--fp8-stats-dir", type=str, default=None)
    parser.add_argument("--monitor-interval", type=float, default=None)
    parser.add_argument("--observe-only", action="store_true", default=None)
    parser.add_argument("--intervene", action="store_true", help="Enable active interventions (disable observe-only)")
    args = parser.parse_args()

    if args.config:
        config = AgentConfig.from_yaml(args.config)
    else:
        config = AgentConfig()

    if args.metrics_file:
        config.metrics_file = args.metrics_file
    if args.control_file:
        config.control_file = args.control_file
    if args.journal_file:
        config.journal_file = args.journal_file
    if args.fp8_stats_dir:
        config.fp8_stats_dir = args.fp8_stats_dir
    if args.monitor_interval is not None:
        config.monitor_interval_seconds = args.monitor_interval
    if args.observe_only:
        config.observation_only = True
    if args.intervene:
        config.observation_only = False

    daemon = AgentDaemon(config)
    daemon.run()


if __name__ == "__main__":
    main()
