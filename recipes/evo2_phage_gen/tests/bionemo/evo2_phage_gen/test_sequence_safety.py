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

"""Behavioral tests for deterministic sequence-safety aggregation and policy loading."""

import hashlib
import json

import pytest

from bionemo.evo2_phage_gen.sequence_safety import (
    GenomeSafetyResult,
    SafetyClassResult,
    SafetyState,
    load_phage_safety_policy,
)


def _class_result(name: str, state: SafetyState, *, required: bool = True) -> SafetyClassResult:
    return SafetyClassResult(safety_class=name, state=state, required=required)


def test_required_failure_dominates_genome_safety_state():
    """Any required failed class must fail the complete genome result."""
    result = GenomeSafetyResult.from_class_results(
        (_class_result("amr", SafetyState.PASS), _class_result("toxin", SafetyState.FAIL))
    )

    assert result.state is SafetyState.FAIL


def test_required_indeterminate_dominates_when_no_required_failure_exists():
    """Incomplete required evidence must remain indeterminate rather than pass."""
    result = GenomeSafetyResult.from_class_results(
        (_class_result("amr", SafetyState.PASS), _class_result("toxin", SafetyState.INDETERMINATE))
    )

    assert result.state is SafetyState.INDETERMINATE


def test_all_required_passes_yield_pass():
    """Optional failures cannot override an all-required-pass outcome."""
    result = GenomeSafetyResult.from_class_results(
        (
            _class_result("amr", SafetyState.PASS),
            _class_result("toxin", SafetyState.PASS),
            _class_result("lysogeny", SafetyState.FAIL, required=False),
        )
    )

    assert result.state is SafetyState.PASS


def test_missing_required_results_are_indeterminate():
    """An empty or informational-only result set cannot pass a required safety gate."""
    empty = GenomeSafetyResult.from_class_results(())
    only_informational = GenomeSafetyResult.from_class_results(
        (_class_result("lysogeny", SafetyState.PASS, required=False),)
    )

    assert empty.state is SafetyState.INDETERMINATE
    assert only_informational.state is SafetyState.INDETERMINATE


def test_policy_load_is_strict_and_digest_is_canonical(tmp_path):
    """Policy input must reject unknown classes and hash its sorted JSON representation."""
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """
schema_version: 1
policy_id: phage-sequence-safety-v1
regulatory_basis:
  label: EMA-derived sequence-design safety gate
  source: EMA/CHMP/BWP/1/2024
  source_status: draft
  source_status_as_of: 2026-08-07
  regulatory_compliance_claimed: false
host_scope:
  allowed_replication_host_domains: [BACTERIA, ARCHAEA, BACTERIA_AND_ARCHAEA]
  disallowed_endpoint: increased_eukaryotic_replication
required_sequence_classes: [toxin, amr]
bacterial_replication_profile:
  required_sequence_classes: [amr, toxin, lysogeny]
  strict_lytic_required: true
archaeal_only_profile:
  required_sequence_classes: [amr, toxin]
  lysogeny: informational
failure_policy:
  missing_required_tool: INDETERMINATE
  missing_required_database: INDETERMINATE
  parser_schema_mismatch: INDETERMINATE
  incomplete_host_evidence: INDETERMINATE
""".lstrip()
    )

    policy = load_phage_safety_policy(policy_path)

    expected_json = json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"))
    assert policy.canonical_json == expected_json
    assert policy.sha256 == hashlib.sha256(expected_json.encode()).hexdigest()

    policy_path.write_text(policy_path.read_text().replace("[toxin, amr]", "[toxin, novel_class]"))
    with pytest.raises(ValueError, match="unknown required sequence class"):
        load_phage_safety_policy(policy_path)


def test_policy_rejects_unknown_schema_version(tmp_path):
    """Unsupported policy versions must not silently acquire new semantics."""
    policy_path = tmp_path / "unknown-version.yaml"
    policy_path.write_text("schema_version: 2\n")

    with pytest.raises(ValueError, match="unsupported policy schema version"):
        load_phage_safety_policy(policy_path)


@pytest.mark.parametrize(
    ("original", "replacement", "expected_message"),
    [
        (
            "regulatory_compliance_claimed: false",
            "regulatory_compliance_claimed: true",
            "regulatory_compliance_claimed",
        ),
        ("strict_lytic_required: true", "strict_lytic_required: false", "strict_lytic_required"),
        ("label: EMA-derived sequence-design safety gate", "label: non-regulatory label", "regulatory_basis"),
        ("source_status: draft", "source_status: final", "regulatory_basis"),
        ("lysogeny: informational", "lysogeny: required", "archaeal_only_profile"),
        ("required_sequence_classes: [amr, toxin]", "required_sequence_classes: [amr]", "required_sequence_classes"),
        (
            "required_sequence_classes: [amr, toxin]",
            "required_sequence_classes: [amr, amr, toxin]",
            "duplicate required sequence class",
        ),
        (
            "required_sequence_classes: [amr, toxin, lysogeny]",
            "required_sequence_classes: [amr, toxin]",
            "bacterial_replication_profile",
        ),
        (
            "archaeal_only_profile:\n  required_sequence_classes: [amr, toxin]",
            "archaeal_only_profile:\n  required_sequence_classes: [amr]",
            "archaeal_only_profile",
        ),
    ],
)
def test_policy_rejects_invalid_schema_v1_scalar_and_profile_values(tmp_path, original, replacement, expected_message):
    """Schema v1 requires its fixed scalars and exact profile class semantics."""
    policy_path = tmp_path / "invalid-policy.yaml"
    policy_path.write_text(
        """
schema_version: 1
policy_id: phage-sequence-safety-v1
regulatory_basis:
  label: EMA-derived sequence-design safety gate
  source: EMA/CHMP/BWP/1/2024
  source_status: draft
  source_status_as_of: 2026-08-07
  regulatory_compliance_claimed: false
host_scope:
  allowed_replication_host_domains: [BACTERIA, ARCHAEA, BACTERIA_AND_ARCHAEA]
  disallowed_endpoint: increased_eukaryotic_replication
required_sequence_classes: [amr, toxin]
bacterial_replication_profile:
  required_sequence_classes: [amr, toxin, lysogeny]
  strict_lytic_required: true
archaeal_only_profile:
  required_sequence_classes: [amr, toxin]
  lysogeny: informational
failure_policy:
  missing_required_tool: INDETERMINATE
  missing_required_database: INDETERMINATE
  parser_schema_mismatch: INDETERMINATE
  incomplete_host_evidence: INDETERMINATE
""".lstrip()
    )

    policy_path.write_text(policy_path.read_text().replace(original, replacement, 1))

    with pytest.raises(ValueError, match=expected_message):
        load_phage_safety_policy(policy_path)
