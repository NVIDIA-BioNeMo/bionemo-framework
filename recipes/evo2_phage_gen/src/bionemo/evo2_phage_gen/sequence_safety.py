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

"""Immutable sequence-safety results and strict version-one policy loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml


class SafetyState(StrEnum):
    """The three-state outcome for a safety check."""

    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class SafetyFinding:
    """One immutable scanner finding."""

    safety_class: str
    state: SafetyState
    reason_codes: tuple[str, ...] = ()
    finding_id: str | None = None

    def __post_init__(self) -> None:
        """Normalize reason codes to an immutable sequence."""
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, object]:
        """Serialize the scanner finding."""
        return {
            "safety_class": self.safety_class,
            "state": self.state.value,
            "reason_codes": list(self.reason_codes),
            "finding_id": self.finding_id,
        }


@dataclass(frozen=True)
class SafetyClassResult:
    """Aggregated result for a single required or informational safety class."""

    safety_class: str
    state: SafetyState
    required: bool
    findings: tuple[SafetyFinding, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze nested findings and reason codes."""
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, object]:
        """Serialize the class-level safety result."""
        return {
            "safety_class": self.safety_class,
            "state": self.state.value,
            "required": self.required,
            "findings": [finding.to_dict() for finding in self.findings],
            "reason_codes": list(self.reason_codes),
        }


def aggregate_safety_state(class_results: tuple[SafetyClassResult, ...]) -> SafetyState:
    """Apply the required-class precedence: FAIL, then INDETERMINATE, then PASS."""
    required_states = [result.state for result in class_results if result.required]
    if SafetyState.FAIL in required_states:
        return SafetyState.FAIL
    if SafetyState.INDETERMINATE in required_states:
        return SafetyState.INDETERMINATE
    return SafetyState.PASS


@dataclass(frozen=True)
class GenomeSafetyResult:
    """Deterministic safety result for a genome over all configured safety classes."""

    state: SafetyState
    class_results: tuple[SafetyClassResult, ...]

    def __post_init__(self) -> None:
        """Freeze the complete set of class results."""
        object.__setattr__(self, "class_results", tuple(self.class_results))

    @classmethod
    def from_class_results(cls, class_results: tuple[SafetyClassResult, ...]) -> GenomeSafetyResult:
        """Construct a result whose state is derived only from required class outcomes."""
        frozen_results = tuple(class_results)
        return cls(state=aggregate_safety_state(frozen_results), class_results=frozen_results)

    def to_dict(self) -> dict[str, object]:
        """Serialize the genome-level safety result."""
        return {
            "state": self.state.value,
            "class_results": [result.to_dict() for result in self.class_results],
        }


_KNOWN_SEQUENCE_CLASSES = frozenset({"amr", "toxin", "lysogeny"})
_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "regulatory_basis",
        "host_scope",
        "required_sequence_classes",
        "bacterial_replication_profile",
        "archaeal_only_profile",
        "failure_policy",
    }
)
_REGULATORY_BASIS_KEYS = frozenset(
    {"label", "source", "source_status", "source_status_as_of", "regulatory_compliance_claimed"}
)
_HOST_SCOPE_KEYS = frozenset({"allowed_replication_host_domains", "disallowed_endpoint"})
_BACTERIAL_PROFILE_KEYS = frozenset({"required_sequence_classes", "strict_lytic_required"})
_ARCHAEAL_PROFILE_KEYS = frozenset({"required_sequence_classes", "lysogeny"})
_FAILURE_POLICY_KEYS = frozenset(
    {
        "missing_required_tool",
        "missing_required_database",
        "parser_schema_mismatch",
        "incomplete_host_evidence",
    }
)


def _strict_mapping(value: object, *, name: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    actual_keys = frozenset(value)
    if actual_keys != keys:
        unknown = sorted(actual_keys - keys)
        missing = sorted(keys - actual_keys)
        raise ValueError(f"{name} keys do not match schema; unknown={unknown}, missing={missing}")
    return value


def _parse_required_sequence_classes(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of sequence classes")
    unknown = sorted(set(value) - _KNOWN_SEQUENCE_CLASSES)
    if unknown:
        raise ValueError(f"unknown required sequence class: {', '.join(unknown)}")
    return tuple(sorted(value))


def _json_safe(value: object) -> object:
    """Recursively make parsed YAML values immutable and JSON serializable."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return MappingProxyType({str(key): _json_safe(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_json_safe(item) for item in value)
    return value


@dataclass(frozen=True)
class PhageSafetyPolicy:
    """Validated schema-version-one policy with deterministic serialization."""

    schema_version: int
    policy_id: str
    regulatory_basis: Mapping[str, object]
    host_scope: Mapping[str, object]
    required_sequence_classes: tuple[str, ...]
    bacterial_replication_profile: Mapping[str, object]
    archaeal_only_profile: Mapping[str, object]
    failure_policy: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze policy mappings and sort its required classes."""
        for name in (
            "regulatory_basis",
            "host_scope",
            "bacterial_replication_profile",
            "archaeal_only_profile",
            "failure_policy",
        ):
            object.__setattr__(self, name, _json_safe(dict(getattr(self, name))))
        object.__setattr__(self, "required_sequence_classes", tuple(sorted(self.required_sequence_classes)))

    def to_dict(self) -> dict[str, object]:
        """Return the normalized schema-version-one policy mapping."""
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "regulatory_basis": dict(self.regulatory_basis),
            "host_scope": {
                **dict(self.host_scope),
                "allowed_replication_host_domains": sorted(self.host_scope["allowed_replication_host_domains"]),
            },
            "required_sequence_classes": list(self.required_sequence_classes),
            "bacterial_replication_profile": {
                **dict(self.bacterial_replication_profile),
                "required_sequence_classes": sorted(self.bacterial_replication_profile["required_sequence_classes"]),
            },
            "archaeal_only_profile": {
                **dict(self.archaeal_only_profile),
                "required_sequence_classes": sorted(self.archaeal_only_profile["required_sequence_classes"]),
            },
            "failure_policy": dict(self.failure_policy),
        }

    @property
    def canonical_json(self) -> str:
        """Return a stable representation suitable for provenance records."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        """Return the SHA-256 digest of :attr:`canonical_json`."""
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def load_phage_safety_policy(path: str | Path) -> PhageSafetyPolicy:
    """Load and strictly validate a schema-version-one phage safety policy."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("policy must be a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported policy schema version: {raw.get('schema_version')}")
    policy = _strict_mapping(raw, name="policy", keys=_POLICY_KEYS)

    regulatory_basis = _strict_mapping(
        policy["regulatory_basis"], name="regulatory_basis", keys=_REGULATORY_BASIS_KEYS
    )
    host_scope = _strict_mapping(policy["host_scope"], name="host_scope", keys=_HOST_SCOPE_KEYS)
    bacterial_profile = _strict_mapping(
        policy["bacterial_replication_profile"],
        name="bacterial_replication_profile",
        keys=_BACTERIAL_PROFILE_KEYS,
    )
    archaeal_profile = _strict_mapping(
        policy["archaeal_only_profile"], name="archaeal_only_profile", keys=_ARCHAEAL_PROFILE_KEYS
    )
    failure_policy = _strict_mapping(policy["failure_policy"], name="failure_policy", keys=_FAILURE_POLICY_KEYS)

    if not isinstance(host_scope["allowed_replication_host_domains"], list) or not all(
        isinstance(domain, str) for domain in host_scope["allowed_replication_host_domains"]
    ):
        raise ValueError("allowed_replication_host_domains must be a list of host domains")
    if set(host_scope["allowed_replication_host_domains"]) != {
        "BACTERIA",
        "ARCHAEA",
        "BACTERIA_AND_ARCHAEA",
    }:
        raise ValueError("allowed_replication_host_domains must match schema version 1")
    if any(value != SafetyState.INDETERMINATE.value for value in failure_policy.values()):
        raise ValueError("failure_policy values must be INDETERMINATE")

    return PhageSafetyPolicy(
        schema_version=policy["schema_version"],
        policy_id=policy["policy_id"],
        regulatory_basis=regulatory_basis,
        host_scope=host_scope,
        required_sequence_classes=_parse_required_sequence_classes(
            policy["required_sequence_classes"], name="required_sequence_classes"
        ),
        bacterial_replication_profile={
            **bacterial_profile,
            "required_sequence_classes": _parse_required_sequence_classes(
                bacterial_profile["required_sequence_classes"],
                name="bacterial_replication_profile.required_sequence_classes",
            ),
        },
        archaeal_only_profile={
            **archaeal_profile,
            "required_sequence_classes": _parse_required_sequence_classes(
                archaeal_profile["required_sequence_classes"],
                name="archaeal_only_profile.required_sequence_classes",
            ),
        },
        failure_policy=failure_policy,
    )
