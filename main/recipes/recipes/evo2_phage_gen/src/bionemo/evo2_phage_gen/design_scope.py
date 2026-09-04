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

"""Declarative host-scope decisions for phage sequence design."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class HostDomain(StrEnum):
    """Replication-host domains represented by verified host evidence."""

    BACTERIA = "BACTERIA"
    ARCHAEA = "ARCHAEA"
    BACTERIA_AND_ARCHAEA = "BACTERIA_AND_ARCHAEA"
    EUKARYOTA = "EUKARYOTA"
    UNKNOWN = "UNKNOWN"


class ObjectiveKind(StrEnum):
    """Structured purpose of a design objective."""

    PRODUCTIVE_INFECTION = "productive_infection"
    PRODUCTIVE_REPLICATION = "productive_replication"
    ENTRY = "entry"
    PHARMACOKINETICS = "pharmacokinetics"
    BIODISTRIBUTION = "biodistribution"
    PERSISTENCE = "persistence"
    CIRCULATION_HALF_LIFE = "circulation_half_life"
    PREMATURE_CLEARANCE = "premature_clearance"
    NEUTRALIZATION = "neutralization"
    DEGRADATION = "degradation"
    NONINFECTIVITY_ASSESSMENT = "noninfectivity_assessment"
    CYTOTOXICITY_ASSESSMENT = "cytotoxicity_assessment"


class ObjectiveDirection(StrEnum):
    """Requested direction of a design objective."""

    INCREASE = "increase"
    DECREASE = "decrease"
    EVALUATE = "evaluate"


class ObjectiveEndpoint(StrEnum):
    """Validated endpoint semantics for a structured design objective."""

    PRODUCTIVE_INFECTION = "productive_infection"
    PRODUCTIVE_REPLICATION = "productive_replication"
    INCREASED_EUKARYOTIC_REPLICATION = "increased_eukaryotic_replication"
    PHARMACOKINETICS = "pharmacokinetics"
    BIODISTRIBUTION = "biodistribution"
    PERSISTENCE = "persistence"
    CIRCULATION_HALF_LIFE = "circulation_half_life"
    REDUCED_PREMATURE_CLEARANCE = "reduced_premature_clearance"
    REDUCED_NEUTRALIZATION = "reduced_neutralization"
    REDUCED_DEGRADATION = "reduced_degradation"
    MAMMALIAN_NONINFECTIVITY = "mammalian_noninfectivity"
    MAMMALIAN_CYTOTOXICITY = "mammalian_cytotoxicity"


@dataclass(frozen=True)
class DesignObjective:
    """A structured objective, deliberately separate from free-text annotations."""

    kind: ObjectiveKind
    direction: ObjectiveDirection
    replication_host_domains: frozenset[HostDomain]
    endpoint: ObjectiveEndpoint

    def __post_init__(self) -> None:
        """Normalize serialized enum values and replication-host domains."""
        try:
            object.__setattr__(self, "kind", ObjectiveKind(self.kind))
        except ValueError as error:
            raise ValueError(f"unsupported objective kind: {self.kind}") from error
        try:
            object.__setattr__(self, "direction", ObjectiveDirection(self.direction))
        except ValueError as error:
            raise ValueError(f"unsupported objective direction: {self.direction}") from error
        try:
            object.__setattr__(self, "endpoint", ObjectiveEndpoint(self.endpoint))
        except ValueError as error:
            raise ValueError(f"unsupported design endpoint: {self.endpoint}") from error
        try:
            object.__setattr__(
                self,
                "replication_host_domains",
                frozenset(HostDomain(domain) for domain in self.replication_host_domains),
            )
        except ValueError as error:
            raise ValueError(f"unsupported replication host domain in {self.replication_host_domains}") from error

    def to_dict(self) -> dict[str, object]:
        """Serialize the structured objective for a saved result."""
        return {
            "kind": self.kind.value,
            "direction": self.direction.value,
            "replication_host_domains": sorted(domain.value for domain in self.replication_host_domains),
            "endpoint": self.endpoint.value,
        }


@dataclass(frozen=True)
class ScopeDecision:
    """Deterministic scope decision with machine-stable reason codes."""

    allowed: bool
    reason_codes: tuple[str, ...]
    quarantined: bool = False

    def __post_init__(self) -> None:
        """Normalize reason codes to a stable sequence."""
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, object]:
        """Serialize the decision and stable reason codes."""
        return {
            "allowed": self.allowed,
            "reason_codes": list(self.reason_codes),
            "quarantined": self.quarantined,
        }


@dataclass(frozen=True)
class HostEvidence:
    """Versioned evidence limited to replication-host attribution."""

    source: str
    source_version: str | None
    replication_host_domains: frozenset[HostDomain]
    confirmed: bool
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze host domains and metadata at construction time."""
        try:
            domains = frozenset(HostDomain(domain) for domain in self.replication_host_domains)
        except ValueError as error:
            raise ValueError(f"unsupported replication host domain in {self.replication_host_domains}") from error
        object.__setattr__(self, "replication_host_domains", domains)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Serialize host evidence without mutating its frozen metadata."""
        return {
            "source": self.source,
            "source_version": self.source_version,
            "replication_host_domains": sorted(domain.value for domain in self.replication_host_domains),
            "confirmed": self.confirmed,
            "metadata": _serialize_metadata(self.metadata),
        }


_PROKARYOTIC_DOMAINS = frozenset({HostDomain.BACTERIA, HostDomain.ARCHAEA, HostDomain.BACTERIA_AND_ARCHAEA})
_REPLICATION_ENDPOINTS = frozenset(
    {
        ObjectiveEndpoint.PRODUCTIVE_INFECTION,
        ObjectiveEndpoint.PRODUCTIVE_REPLICATION,
        ObjectiveEndpoint.INCREASED_EUKARYOTIC_REPLICATION,
    }
)
_REPLICATION_OBJECTIVE_KINDS = frozenset({ObjectiveKind.PRODUCTIVE_INFECTION, ObjectiveKind.PRODUCTIVE_REPLICATION})


def _freeze_metadata(value: object) -> object:
    """Freeze nested metadata while retaining a JSON-compatible representation."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("unsupported host-evidence metadata value: mapping keys must be strings")
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported host-evidence metadata value: {type(value).__name__}")


def _serialize_metadata(value: object) -> object:
    """Convert frozen metadata back to ordinary JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {key: _serialize_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize_metadata(item) for item in value]
    return value


def validate_design_scope(objective: DesignObjective) -> ScopeDecision:
    """Reject only increased objectives toward replication within eukaryotic cells."""
    has_eukaryotic_replication_host = HostDomain.EUKARYOTA in objective.replication_host_domains
    replication_kind = objective.kind in _REPLICATION_OBJECTIVE_KINDS
    replication_entry = objective.kind is ObjectiveKind.ENTRY and objective.endpoint in _REPLICATION_ENDPOINTS
    explicit_eukaryotic_replication_endpoint = objective.endpoint is ObjectiveEndpoint.INCREASED_EUKARYOTIC_REPLICATION
    if objective.direction is ObjectiveDirection.INCREASE and (
        explicit_eukaryotic_replication_endpoint
        or (has_eukaryotic_replication_host and (replication_kind or replication_entry))
    ):
        return ScopeDecision(False, ("EUKARYOTIC_REPLICATION_OBJECTIVE",))
    return ScopeDecision(True, ("OBJECTIVE_WITHIN_HOST_SCOPE",))


def evaluate_host_evidence(evidence: HostEvidence) -> ScopeDecision:
    """Determine eligibility from confirmed, versioned replication-host evidence only."""
    domains = evidence.replication_host_domains
    has_prokaryotic_host = bool(domains & _PROKARYOTIC_DOMAINS)
    has_eukaryotic_host = HostDomain.EUKARYOTA in domains

    if has_prokaryotic_host and has_eukaryotic_host:
        return ScopeDecision(False, ("CONFLICTING_REPLICATION_HOST_EVIDENCE",), quarantined=True)
    if not evidence.confirmed or not evidence.source or not evidence.source_version:
        return ScopeDecision(False, ("INCOMPLETE_HOST_EVIDENCE",))
    if has_eukaryotic_host:
        return ScopeDecision(False, ("EUKARYOTIC_REPLICATION_HOST",))
    if has_prokaryotic_host:
        return ScopeDecision(True, ("CONFIRMED_PROKARYOTIC_REPLICATION_HOST",))
    return ScopeDecision(False, ("INCOMPLETE_HOST_EVIDENCE",))
