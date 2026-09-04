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

from pathlib import Path

import pytest

from bionemo.evo2_phage_gen.sequence_safety import (
    GenomeSafetyResult,
    SafetyClassResult,
    SafetyState,
    load_phage_safety_policy,
)


def _result(name: str, state: SafetyState, *, required: bool = True) -> SafetyClassResult:
    return SafetyClassResult(name, state, required)


def test_required_failure_wins() -> None:
    result = GenomeSafetyResult.from_class_results(
        (_result("amr", SafetyState.PASS), _result("toxin", SafetyState.FAIL))
    )
    assert result.state is SafetyState.FAIL


def test_missing_or_indeterminate_evidence_never_passes() -> None:
    assert GenomeSafetyResult.from_class_results(()).state is SafetyState.INDETERMINATE
    result = GenomeSafetyResult.from_class_results(
        (_result("amr", SafetyState.PASS), _result("toxin", SafetyState.INDETERMINATE))
    )
    assert result.state is SafetyState.INDETERMINATE


def test_informational_result_does_not_block_required_passes() -> None:
    result = GenomeSafetyResult.from_class_results(
        (
            _result("amr", SafetyState.PASS),
            _result("toxin", SafetyState.PASS),
            _result("lysogeny", SafetyState.INDETERMINATE, required=False),
        )
    )
    assert result.state is SafetyState.PASS


def test_policy_keeps_bacterial_safety_classes_and_failure_semantics(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        """
policy_id: lab-policy
required_sequence_classes: [amr, toxin]
bacterial_replication_profile:
  required_sequence_classes: [amr, toxin, lysogeny]
failure_policy:
  missing_tool: INDETERMINATE
  parser_error: INDETERMINATE
"""
    )
    loaded = load_phage_safety_policy(policy)
    assert loaded.required_sequence_classes == ("amr", "toxin")
    assert loaded.bacterial_required_sequence_classes == ("amr", "toxin", "lysogeny")


@pytest.mark.parametrize(
    "policy_text",
    (
        "policy_id: x\nrequired_sequence_classes: [amr]\nbacterial_replication_profile: {}\nfailure_policy: {}\n",
        "policy_id: x\nrequired_sequence_classes: [amr, toxin]\nbacterial_replication_profile:\n  required_sequence_classes: [amr, toxin]\nfailure_policy: {}\n",
        "policy_id: x\nrequired_sequence_classes: [amr, toxin]\nbacterial_replication_profile:\n  required_sequence_classes: [amr, toxin, lysogeny]\nfailure_policy:\n  missing_tool: PASS\n",
    ),
)
def test_policy_rejects_unsafe_omissions(tmp_path: Path, policy_text: str) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(policy_text)
    with pytest.raises(ValueError):
        load_phage_safety_policy(policy)
