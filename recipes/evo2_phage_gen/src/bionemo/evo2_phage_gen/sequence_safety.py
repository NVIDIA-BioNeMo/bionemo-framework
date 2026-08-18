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

"""Result types and policy loading for phage sequence-safety screens."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml


class SafetyState(StrEnum):
    """Represent a sequence-screen decision."""

    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class SafetyFinding:
    """Describe one normalized safety finding."""

    safety_class: str
    state: SafetyState
    reason_codes: tuple[str, ...] = ()
    finding_id: str | None = None

    def __post_init__(self) -> None:
        """Normalize enum and tuple fields."""
        object.__setattr__(self, "state", SafetyState(self.state))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, object]:
        """Serialize the finding."""
        return {
            "safety_class": self.safety_class,
            "state": self.state.value,
            "reason_codes": list(self.reason_codes),
            "finding_id": self.finding_id,
        }


@dataclass(frozen=True)
class SafetyClassResult:
    """Summarize one required or optional safety class."""

    safety_class: str
    state: SafetyState
    required: bool
    findings: tuple[SafetyFinding, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize enum and tuple fields."""
        object.__setattr__(self, "state", SafetyState(self.state))
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, object]:
        """Serialize the class result."""
        return {
            "safety_class": self.safety_class,
            "state": self.state.value,
            "required": self.required,
            "findings": [finding.to_dict() for finding in self.findings],
            "reason_codes": list(self.reason_codes),
        }


def aggregate_safety_state(class_results: tuple[SafetyClassResult, ...]) -> SafetyState:
    """Combine required class results conservatively."""
    states = [result.state for result in class_results if result.required]
    if not states:
        return SafetyState.INDETERMINATE
    if SafetyState.FAIL in states:
        return SafetyState.FAIL
    if SafetyState.INDETERMINATE in states:
        return SafetyState.INDETERMINATE
    return SafetyState.PASS


@dataclass(frozen=True)
class GenomeSafetyResult:
    """Summarize all safety classes for one genome."""

    state: SafetyState
    class_results: tuple[SafetyClassResult, ...]

    def __post_init__(self) -> None:
        """Normalize enum and tuple fields."""
        object.__setattr__(self, "state", SafetyState(self.state))
        object.__setattr__(self, "class_results", tuple(self.class_results))

    @classmethod
    def from_class_results(cls, class_results: tuple[SafetyClassResult, ...]) -> "GenomeSafetyResult":
        """Build a genome result from class results."""
        results = tuple(class_results)
        return cls(aggregate_safety_state(results), results)

    def to_dict(self) -> dict[str, object]:
        """Serialize the genome result."""
        return {
            "state": self.state.value,
            "class_results": [result.to_dict() for result in self.class_results],
        }


@dataclass(frozen=True)
class PhageSafetyPolicy:
    """Define the sequence classes required by the safety screen."""

    policy_id: str
    required_sequence_classes: tuple[str, ...]
    bacterial_required_sequence_classes: tuple[str, ...]
    failure_policy: Mapping[str, str]


def _classes(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a nonempty list")
    unknown = set(value) - {"amr", "toxin", "lysogeny"}
    if unknown or len(value) != len(set(value)):
        raise ValueError(f"{label} contains unknown or duplicate classes")
    return tuple(value)


def load_phage_safety_policy(path: str | Path) -> PhageSafetyPolicy:
    """Load and validate a phage safety policy."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("safety policy must be a mapping")
    policy_id = raw.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("safety policy requires policy_id")
    bacterial = raw.get("bacterial_replication_profile")
    failures = raw.get("failure_policy")
    if not isinstance(bacterial, dict) or not isinstance(failures, dict):
        raise ValueError("safety policy lacks bacterial or failure behavior")
    required = _classes(raw.get("required_sequence_classes"), "required_sequence_classes")
    bacterial_required = _classes(
        bacterial.get("required_sequence_classes"),
        "bacterial_replication_profile.required_sequence_classes",
    )
    if not {"amr", "toxin"} <= set(required) or not {"amr", "toxin", "lysogeny"} <= set(bacterial_required):
        raise ValueError("safety policy omits a required safety class")
    if any(value != SafetyState.INDETERMINATE.value for value in failures.values()):
        raise ValueError("missing evidence must remain INDETERMINATE")
    return PhageSafetyPolicy(policy_id, required, bacterial_required, failures)
