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

"""Behavioral tests for the declarative phage host-scope contract."""

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
    bacteria_and_archaea_host_range = DesignObjective(
        kind=ObjectiveKind.PRODUCTIVE_REPLICATION,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.BACTERIA_AND_ARCHAEA}),
        endpoint="productive_replication",
    )

    assert validate_design_scope(bacterial_or_archaeal_host_range).allowed
    assert validate_design_scope(bacteria_and_archaea_host_range).allowed


def test_productive_eukaryotic_endpoints_are_rejected():
    """Increasing eukaryotic productive infection or replication must be blocked."""
    increase_productive_eukaryotic_infection = DesignObjective(
        kind=ObjectiveKind.PRODUCTIVE_INFECTION,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.EUKARYOTA}),
        endpoint="productive_infection",
    )
    increase_eukaryotic_entry_for_productive_replication = DesignObjective(
        kind=ObjectiveKind.ENTRY,
        direction=ObjectiveDirection.INCREASE,
        replication_host_domains=frozenset({HostDomain.EUKARYOTA}),
        endpoint="productive_replication",
    )

    infection_decision = validate_design_scope(increase_productive_eukaryotic_infection)
    entry_decision = validate_design_scope(increase_eukaryotic_entry_for_productive_replication)

    assert not infection_decision.allowed
    assert infection_decision.reason_codes == ("EUKARYOTIC_PRODUCTIVE_ENDPOINT",)
    assert not entry_decision.allowed
    assert entry_decision.reason_codes == ("EUKARYOTIC_PRODUCTIVE_ENDPOINT",)


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
        metadata={"evidence_ids": ["host-record-1"]},
    )

    assert evidence.metadata["evidence_ids"] == ("host-record-1",)
