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

"""Experiment journal for cross-session learning and audit trail.

Records every observation, hypothesis, intervention, and outcome as a
JSONL log on shared NFS.  The journal serves three purposes:

1. **Audit trail** — reproducible record of what the agent did and why.
2. **Cross-session context** — the agent loads prior entries on startup
   so it can reason about patterns across multiple training runs.
3. **Knowledge accumulation** — over time the journal becomes a model-
   and dataset-specific knowledge base for FP8 debugging heuristics.
"""

import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class EntryType(str, Enum):
    """Journal entry categories."""

    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    INTERVENTION = "intervention"
    OUTCOME = "outcome"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ERROR = "error"


class ExperimentJournal:
    """Append-only JSONL journal on shared storage.

    Args:
        journal_path: Path to the JSONL file.
        session_id: Unique identifier for the current agent session.
    """

    def __init__(
        self,
        journal_path: str | os.PathLike = "/data/agent/journal.jsonl",
        session_id: str = "default",
    ):
        """Initialize the experiment journal."""
        self._path = Path(journal_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id
        self._entry_counter = 0

    def log(
        self,
        entry_type: EntryType | str,
        *,
        step: int | None = None,
        trigger: str | None = None,
        hypothesis: str | None = None,
        evidence: dict[str, Any] | None = None,
        action: str | None = None,
        config_diff: dict[str, Any] | None = None,
        outcome: str | None = None,
        outcome_step_range: tuple[int, int] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one journal entry and return it.

        All keyword arguments are optional; only relevant fields need to be
        supplied for each entry type.

        Args:
            entry_type: Category tag (observation / hypothesis / intervention / …).
            step: Training step associated with the event.
            trigger: What triggered this entry (e.g. metric anomaly description).
            hypothesis: Agent's hypothesis for the root cause.
            evidence: Supporting data (metric snapshots, per-layer stats, …).
            action: Description of the action taken.
            config_diff: Mapping of config keys to ``[old, new]`` value pairs.
            outcome: Textual summary filled in after observing the result.
            outcome_step_range: ``(start, end)`` step range over which the
                outcome was measured.
            extra: Arbitrary extra data to attach.

        Returns:
            The full entry dict that was written.
        """
        self._entry_counter += 1
        entry: dict[str, Any] = {
            "id": self._entry_counter,
            "session_id": self._session_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": str(entry_type),
        }
        if step is not None:
            entry["step"] = step
        if trigger is not None:
            entry["trigger"] = trigger
        if hypothesis is not None:
            entry["hypothesis"] = hypothesis
        if evidence is not None:
            entry["evidence"] = evidence
        if action is not None:
            entry["action"] = action
        if config_diff is not None:
            entry["config_diff"] = config_diff
        if outcome is not None:
            entry["outcome"] = outcome
        if outcome_step_range is not None:
            entry["outcome_step_range"] = list(outcome_step_range)
        if extra is not None:
            entry["extra"] = extra

        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.exception("Failed to write journal entry")

        return entry

    def load_history(self, max_entries: int = 500) -> list[dict[str, Any]]:
        """Load the most recent journal entries (for agent context on startup).

        Args:
            max_entries: Maximum number of entries to return (from the tail).

        Returns:
            List of entry dicts, oldest first.
        """
        if not self._path.exists():
            return []

        entries: list[dict[str, Any]] = []
        try:
            with open(self._path) as f:
                for raw_line in f:
                    stripped = raw_line.strip()
                    if stripped:
                        try:
                            entries.append(json.loads(stripped))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            logger.exception("Failed to read journal")
            return []

        return entries[-max_entries:]

    def update_outcome(
        self,
        entry_id: int,
        outcome: str,
        outcome_step_range: tuple[int, int] | None = None,
    ) -> None:
        """Append a follow-up outcome entry referencing an earlier intervention.

        Rather than mutating the original JSONL line (which is fragile), we
        append a new ``outcome`` entry that references the original ``entry_id``.

        Args:
            entry_id: The ``id`` of the original intervention entry.
            outcome: Textual summary of the observed result.
            outcome_step_range: Step range over which the outcome was measured.
        """
        self.log(
            EntryType.OUTCOME,
            action=f"outcome for entry {entry_id}",
            outcome=outcome,
            outcome_step_range=outcome_step_range,
            extra={"references_entry_id": entry_id},
        )
