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

"""Behavioral tests for declarative phage host scope."""

import json

import pytest

from bionemo.evo2_phage_gen.design_scope import (
    DesignObjective,
    HostDomain,
    HostEvidence,
    ObjectiveDirection,
    ObjectiveKind,
    evaluate_host_evidence,
    validate_design_scope,
)


def test_productive_prokaryotic_host_objectives_are_allowed():
    """A scope gate must retain bacterial, archaeal, and mixed-host designs."""
    bacterial_or_archaeal_host_range = DesignObjective(
        kind=ObjectiveKind.PRODUCTIVE_INFECTION,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.BACTERIA}),
        endpoint="productive_replication",
    )
    archaeal_host_range = DesignObjective(
        kind=ObjectiveKind.PRODUCTIVE_REPLICATION,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.ARCHAEA}),
        endpoint="productive_replication",
    )
    bacteria_and_archaea_host_range = DesignObjective(
        kind=ObjectiveKind.PRODUCTIVE_REPLICATION,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.BACTERIA_AND_ARCHAEA}),
        endpoint="productive_replication",
    )

    assert validate_design_scope(bacterial_or_archaeal_host_range).allowed
    assert validate_design_scope(archaeal_host_range).allowed
    assert validate_design_scope(bacteria_and_archaea_host_range).allowed


def test_eukaryotic_replication_objectives_are_rejected():
    """Increasing replication within eukaryotic cells must be blocked."""
    increase_eukaryotic_replication = DesignObjective(
        kind=ObjectiveKind.PRODUCTIVE_REPLICATION,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.EUKARYOTA}),
        endpoint="productive_replication",
    )
    increase_eukaryotic_entry_for_replication = DesignObjective(
        kind=ObjectiveKind.ENTRY,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.EUKARYOTA}),
        endpoint="productive_replication",
    )

    replication_decision = validate_design_scope(increase_eukaryotic_replication)
    entry_decision = validate_design_scope(increase_eukaryotic_entry_for_replication)

    assert not replication_decision.allowed
    assert replication_decision.reason_codes == ("EUKARYOTIC_REPLICATION_OBJECTIVE",)
    assert not entry_decision.allowed
    assert entry_decision.reason_codes == ("EUKARYOTIC_REPLICATION_OBJECTIVE",)


def test_policy_replication_endpoint_cannot_bypass_entry_scope():
    """Entry must reject the policy's declared eukaryotic replication endpoint."""
    policy_replication_endpoint = DesignObjective(
        kind=ObjectiveKind.ENTRY,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.EUKARYOTA}),
        endpoint="increased_eukaryotic_replication",
    )

    decision = validate_design_scope(policy_replication_endpoint)

    assert not decision.allowed
    assert decision.reason_codes == ("EUKARYOTIC_REPLICATION_OBJECTIVE",)


def test_policy_replication_endpoint_cannot_hide_behind_inconsistent_fields():
    """The explicit eukaryotic-replication endpoint is sufficient to reject an increase."""
    disguised = DesignObjective(
        kind=ObjectiveKind.PERSISTENCE,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.BACTERIA}),
        endpoint="increased_eukaryotic_replication",
    )

    decision = validate_design_scope(disguised)

    assert not decision.allowed
    assert decision.reason_codes == ("EUKARYOTIC_REPLICATION_OBJECTIVE",)


def test_objective_coerces_raw_kind_and_direction_before_validation_and_serialization():
    """Deserialized enum values must retain the same safety semantics as enum instances."""
    objective = DesignObjective(
        kind="productive_replication",
        direction="increase",
        replication_host_domains=frozenset({HostDomain.EUKARYOTA}),
        endpoint="productive_replication",
    )

    assert objective.to_dict()["kind"] == "productive_replication"
    assert objective.to_dict()["direction"] == "increase"
    assert not validate_design_scope(objective).allowed


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "unsupported_kind", "unsupported objective kind"),
        ("direction", "sideways", "unsupported objective direction"),
    ],
)
def test_objective_rejects_unsupported_raw_enum_values(field, value, message):
    values = {
        "kind": ObjectiveKind.ENTRY,
        "direction": ObjectiveDirection.EVALUATE,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        DesignObjective(
            **values,
            replication_host_domains=frozenset({HostDomain.BACTERIA}),
            endpoint="mammalian_noninfectivity",
        )


def test_unknown_entry_endpoint_is_rejected_at_objective_construction():
    """Unvalidated endpoint strings cannot silently acquire scope semantics."""
    with pytest.raises(ValueError, match="unsupported design endpoint"):
        DesignObjective(
            kind=ObjectiveKind.ENTRY,
            direction=ObjectiveDirection.INCREASE,
            replication_host_domains=frozenset({HostDomain.EUKARYOTA}),
            endpoint="unvalidated_productive_endpoint",
        )


def test_eukaryotic_pharmacokinetic_and_safety_assessment_objectives_are_allowed():
    """Host-cell PK and noninfectivity assessment are not productive-host objectives."""
    human_pk_persistence_for_bacterial_replication = DesignObjective(
        kind=ObjectiveKind.PERSISTENCE,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.BACTERIA}),
        endpoint="circulation_half_life",
    )
    mammalian_noninfectivity_assay = DesignObjective(
        kind=ObjectiveKind.NONINFECTIVITY_ASSESSMENT,
        direction=ObjectiveDirection.EVALUATE,
        replication_host_domains=frozenset({HostDomain.ARCHAEA}),
        endpoint="mammalian_noninfectivity",
    )

    assert validate_design_scope(human_pk_persistence_for_bacterial_replication).allowed
    assert validate_design_scope(mammalian_noninfectivity_assay).allowed


@pytest.mark.parametrize(
    ("kind", "direction", "endpoint"),
    [
        (ObjectiveKind.PHARMACOKINETICS, ObjectiveDirection.INCREASE, "pharmacokinetics"),
        (ObjectiveKind.BIODISTRIBUTION, ObjectiveDirection.INCREASE, "biodistribution"),
        (ObjectiveKind.PERSISTENCE, ObjectiveDirection.INCREASE, "persistence"),
        (ObjectiveKind.CIRCULATION_HALF_LIFE, ObjectiveDirection.INCREASE, "circulation_half_life"),
        (ObjectiveKind.PREMATURE_CLEARANCE, ObjectiveDirection.DECREASE, "reduced_premature_clearance"),
        (ObjectiveKind.NEUTRALIZATION, ObjectiveDirection.DECREASE, "reduced_neutralization"),
        (ObjectiveKind.DEGRADATION, ObjectiveDirection.DECREASE, "reduced_degradation"),
        (ObjectiveKind.NONINFECTIVITY_ASSESSMENT, ObjectiveDirection.EVALUATE, "mammalian_noninfectivity"),
        (ObjectiveKind.CYTOTOXICITY_ASSESSMENT, ObjectiveDirection.EVALUATE, "mammalian_cytotoxicity"),
    ],
)
def test_validated_nonproductive_endpoints_remain_allowed_for_prokaryotic_replication(kind, direction, endpoint):
    """Endpoint validation must preserve every explicitly permitted nonproductive objective."""
    objective = DesignObjective(
        kind=kind,
        direction=direction,
        replication_host_domains=frozenset({HostDomain.BACTERIA}),
        endpoint=endpoint,
    )

    assert validate_design_scope(objective).allowed


def test_only_confirmed_versioned_prokaryotic_host_evidence_is_eligible():
    """Eligibility must depend on replication-host evidence, not clinical metadata."""
    bacteria = HostEvidence(
        source="ncbi",
        source_version="2026-08-07",
        replication_host_domains=frozenset({HostDomain.BACTERIA}),
        confirmed=True,
    )
    archaea = HostEvidence(
        source="ncbi",
        source_version="2026-08-07",
        replication_host_domains=frozenset({HostDomain.ARCHAEA}),
        confirmed=True,
    )
    mixed = HostEvidence(
        source="ncbi",
        source_version="2026-08-07",
        replication_host_domains=frozenset({HostDomain.BACTERIA_AND_ARCHAEA}),
        confirmed=True,
    )

    assert evaluate_host_evidence(bacteria).allowed
    assert evaluate_host_evidence(archaea).allowed
    assert evaluate_host_evidence(mixed).allowed


def test_host_evidence_normalizes_string_domain_values():
    evidence = HostEvidence(
        source="ncbi",
        source_version="v2",
        replication_host_domains=frozenset({"BACTERIA", "ARCHAEA"}),
        confirmed=True,
    )

    assert evidence.replication_host_domains == frozenset({HostDomain.BACTERIA, HostDomain.ARCHAEA})
    assert evidence.to_dict()["replication_host_domains"] == ["ARCHAEA", "BACTERIA"]


@pytest.mark.parametrize(
    "source",
    ("RefSeq", "GenBank", "PhagesDB", "metagenomic collection", "model-generated candidate"),
)
def test_prokaryotic_host_evidence_is_source_neutral(source):
    """An origin label alone must not determine biological eligibility."""
    evidence = HostEvidence(
        source=source,
        source_version="user-recorded-origin",
        replication_host_domains=frozenset({HostDomain.BACTERIA}),
        confirmed=True,
    )

    decision = evaluate_host_evidence(evidence)

    assert decision.allowed
    assert decision.reason_codes == ("CONFIRMED_PROKARYOTIC_REPLICATION_HOST",)


def test_eukaryotic_or_conflicting_host_evidence_is_not_eligible():
    """A confirmed eukaryotic host or conflict must not enter the design scope."""
    eukaryotic = HostEvidence(
        source="ncbi",
        source_version="2026-08-07",
        replication_host_domains=frozenset({HostDomain.EUKARYOTA}),
        confirmed=True,
    )
    conflicting = HostEvidence(
        source="ncbi",
        source_version="2026-08-07",
        replication_host_domains=frozenset({HostDomain.BACTERIA, HostDomain.EUKARYOTA}),
        confirmed=True,
    )

    assert not evaluate_host_evidence(eukaryotic).allowed
    quarantined = evaluate_host_evidence(conflicting)
    assert not quarantined.allowed
    assert quarantined.quarantined
    assert quarantined.reason_codes == ("CONFLICTING_REPLICATION_HOST_EVIDENCE",)


def test_unknown_host_stays_ineligible_and_non_host_metadata_is_ignored():
    """Human isolate, indication, formulation, and immune fields are not host evidence."""
    unknown = HostEvidence(
        source="ncbi",
        source_version=None,
        replication_host_domains=frozenset({HostDomain.UNKNOWN}),
        confirmed=False,
        metadata={
            "isolation_host": "Homo sapiens",
            "indication": "human infection",
            "formulation": "intravenous",
            "immune_interaction": "neutralizing antibody",
        },
    )

    decision = evaluate_host_evidence(unknown)

    assert not decision.allowed
    assert not decision.quarantined
    assert decision.reason_codes == ("INCOMPLETE_HOST_EVIDENCE",)


def test_host_evidence_deep_freezes_metadata_for_serializable_records():
    """Host-evidence metadata must not remain mutable through nested input values."""
    evidence = HostEvidence(
        source="ncbi",
        source_version="2026-08-07",
        replication_host_domains=frozenset({HostDomain.BACTERIA}),
        confirmed=True,
        metadata={"nested": {"evidence_ids": ["host-record-1"]}},
    )

    assert evidence.metadata["nested"]["evidence_ids"] == ("host-record-1",)
    with pytest.raises(TypeError):
        evidence.metadata["nested"]["evidence_ids"] = ()
    assert json.dumps(evidence.to_dict())


@pytest.mark.parametrize("unsupported_value", [{"record-1"}, object()])
def test_host_evidence_rejects_non_json_metadata_values(unsupported_value):
    """Sets and arbitrary objects cannot enter serializable metadata."""
    with pytest.raises(TypeError, match="unsupported host-evidence metadata value"):
        HostEvidence(
            source="ncbi",
            source_version="2026-08-07",
            replication_host_domains=frozenset({HostDomain.BACTERIA}),
            confirmed=True,
            metadata={"unsupported": unsupported_value},
        )
