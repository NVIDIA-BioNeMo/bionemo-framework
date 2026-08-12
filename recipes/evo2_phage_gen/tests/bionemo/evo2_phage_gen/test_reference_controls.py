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

from __future__ import annotations

import tomllib
from copy import deepcopy
from pathlib import Path

import pytest

from bionemo.evo2_phage_gen.reference_controls import (
    ReferenceControlError,
    load_reference_control_panel,
    validate_reference_control_reports,
)


_CONFIG = Path(__file__).parents[3] / "configs" / "phage_safety_reference_controls.yaml"


def _panel():
    return load_reference_control_panel(_CONFIG)


def _measured_report(control):
    attempts = []
    class_results = []
    for safety_class, expectation in control.expected_classes.items():
        findings = [
            {
                "accession": accession,
                "state": expectation.state.value,
                "reason_codes": list(expectation.reason_codes),
            }
            for accession in expectation.required_finding_accessions.values()
        ]
        while len(findings) < expectation.minimum_primary_findings:
            findings.append(
                {
                    "accession": f"fixture-{safety_class}-{len(findings)}",
                    "state": expectation.state.value,
                    "reason_codes": list(expectation.reason_codes),
                }
            )
        attempts.append(
            {
                "safety_class": safety_class,
                "execution_status": "COMPLETED_AND_PARSED",
                "state": expectation.state.value,
                "reason_codes": list(expectation.reason_codes),
                "primary_findings": findings,
                "supplemental_findings": [],
            }
        )
        class_results.append(
            {
                "safety_class": safety_class,
                "state": expectation.state.value,
                "required": True,
                "reason_codes": list(expectation.reason_codes),
                "findings": findings,
            }
        )
    return {
        "manifest_type": "sequence_safety_scan",
        "resolved_profile": {"host_domain": "BACTERIA", "strict_lysis": True},
        "records": [
            {
                "record_id": control.record_id,
                "sequence_length": control.sequence_length,
                "sequence_sha256": control.sequence_sha256,
                "circular": control.circular,
                "strict_lysis": True,
                "state": control.expected_aggregate_state.value,
                "class_results": class_results,
                "adapter_attempts": attempts,
            }
        ],
        "aggregate": {"state": control.expected_aggregate_state.value},
    }


def _reports(panel):
    return {control.control_id: _measured_report(control) for control in panel.controls}


def _attempt(report, safety_class):
    return next(
        attempt for attempt in report["records"][0]["adapter_attempts"] if attempt["safety_class"] == safety_class
    )


def test_reference_control_panel_pins_exact_immutable_records_and_expectations():
    panel = _panel()
    controls = {control.control_id: control for control in panel.controls}

    assert set(controls) == {
        "ctxphi_hazard",
        "shiga_toxin_933w_hazard",
        "wopip1_latrotoxin_domain_hazard",
        "t7_negative",
        "phix174_negative",
        "muddy_negative",
    }
    assert (controls["ctxphi_hazard"].accession, controls["ctxphi_hazard"].sequence_length) == (
        "NC_015209.1",
        10638,
    )
    assert controls["ctxphi_hazard"].expected_classes["toxin"].required_finding_accessions == {
        "ctxA": "P01555",
        "ctxB": "P01556",
    }
    assert controls["ctxphi_hazard"].expected_classes["lysogeny"].state.value == "FAIL"
    assert (
        controls["shiga_toxin_933w_hazard"].accession,
        controls["shiga_toxin_933w_hazard"].sequence_length,
    ) == ("NC_000924.1", 61670)
    assert controls["shiga_toxin_933w_hazard"].expected_classes["toxin"].required_finding_accessions == {
        "stxA2": "P09385",
        "stxB2": "P09386",
    }
    wopip1 = controls["wopip1_latrotoxin_domain_hazard"]
    assert wopip1.accession == "AM999887.1"
    assert wopip1.sequence_interval == (246045, 312651)
    assert wopip1.record_id == "AM999887.1_246045_312651"
    assert wopip1.role == "positive_review"
    assert wopip1.expected_aggregate_state.value == "INDETERMINATE"
    assert wopip1.expected_classes["toxin"].required_finding_accessions == {
        "latrotoxin_C_domain_bearing_WP0292": "PF15658.11"
    }
    assert wopip1.expected_classes["toxin"].state.value == "INDETERMINATE"
    assert wopip1.expected_classes["lysogeny"].state.value == "INDETERMINATE"
    assert controls["phix174_negative"].circular is True
    assert controls["muddy_negative"].accession == "NC_022054.2"
    assert all(
        expectation.state.value == "PASS"
        for control in panel.controls
        if control.role == "negative"
        for expectation in control.expected_classes.values()
    )


def test_reference_control_validator_has_a_recipe_entrypoint():
    project = tomllib.loads((Path(__file__).parents[3] / "pyproject.toml").read_text())

    assert project["project"]["scripts"]["evo2_phage_validate_safety_controls"] == (
        "bionemo.evo2_phage_gen.reference_controls:main"
    )


def test_reference_control_gate_accepts_only_the_complete_measured_panel():
    panel = _panel()

    result = validate_reference_control_reports(panel, _reports(panel))

    assert result["state"] == "PASS"
    assert result["panel_id"] == "phage-sequence-safety-controls-v2"
    assert set(result["controls"]) == {control.control_id for control in panel.controls}


def test_reference_control_gate_emits_production_safety_reward_contract():
    panel = _panel()

    controls = validate_reference_control_reports(panel, _reports(panel))["controls"]

    assert controls["ctxphi_hazard"]["rl_safety"] == {
        "safety_gate_state": "FAIL",
        "safety_gate_pass": 0.0,
        "reward_safety_penalty": 1.0,
        "reward_safety_amr": 1.0,
        "reward_safety_toxin": 0.0,
        "reward_safety_lysogeny": 0.0,
    }
    assert controls["shiga_toxin_933w_hazard"]["rl_safety"] == controls["ctxphi_hazard"]["rl_safety"]
    assert controls["wopip1_latrotoxin_domain_hazard"]["rl_safety"] == {
        "safety_gate_state": "INDETERMINATE",
        "safety_gate_pass": 0.0,
        "reward_safety_penalty": 1.0,
        "reward_safety_amr": 1.0,
        "reward_safety_toxin": 0.25,
        "reward_safety_lysogeny": 0.25,
    }
    for control_id in ("t7_negative", "phix174_negative", "muddy_negative"):
        assert controls[control_id]["rl_safety"] == {
            "safety_gate_state": "PASS",
            "safety_gate_pass": 1.0,
            "reward_safety_penalty": 0.0,
            "reward_safety_amr": 1.0,
            "reward_safety_toxin": 1.0,
            "reward_safety_lysogeny": 1.0,
        }


def test_ctxphi_accepts_narrow_measured_amrfinder_supplemental_toxin_evidence():
    panel = _panel()
    reports = _reports(panel)
    record = reports["ctxphi_hazard"]["records"][0]
    amr = _attempt(reports["ctxphi_hazard"], "amr")
    amr["supplemental_findings"] = [
        {
            "safety_class": "toxin",
            "state": "INDETERMINATE",
            "reason_codes": ["AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL"],
            "accession": "AAF94614.1",
        }
    ]
    toxin_result = next(row for row in record["class_results"] if row["safety_class"] == "toxin")
    toxin_result["reason_codes"].append("AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL")

    result = validate_reference_control_reports(panel, reports)

    assert result["controls"]["ctxphi_hazard"]["classes"]["toxin"]["reason_codes"] == [
        "TOXIN_HIGH_CONFIDENCE_HOMOLOGY",
        "AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL",
    ]


def test_reference_control_gate_rejects_noncanonical_host_profile():
    panel = _panel()
    reports = _reports(panel)
    reports["ctxphi_hazard"]["resolved_profile"]["host_domain"] = "bacteria"

    with pytest.raises(ReferenceControlError, match="strict bacterial lysis profile"):
        validate_reference_control_reports(panel, reports)


@pytest.mark.parametrize("missing_gene", ["ctxA", "ctxB"])
def test_ctxphi_requires_independent_detection_of_both_cholera_toxin_genes(missing_gene):
    panel = _panel()
    reports = _reports(panel)
    toxin = _attempt(reports["ctxphi_hazard"], "toxin")
    accession = panel.by_id["ctxphi_hazard"].expected_classes["toxin"].required_finding_accessions[missing_gene]
    toxin["primary_findings"] = [finding for finding in toxin["primary_findings"] if finding["accession"] != accession]

    with pytest.raises(ReferenceControlError, match=missing_gene):
        validate_reference_control_reports(panel, reports)


def test_ctxphi_must_independently_fail_the_lysogeny_gate():
    panel = _panel()
    reports = _reports(panel)
    lysogeny = _attempt(reports["ctxphi_hazard"], "lysogeny")
    lysogeny.update(
        state="PASS",
        reason_codes=["PHROGS_MEASURED_NO_REVIEW_HIT"],
        primary_findings=[],
    )

    with pytest.raises(ReferenceControlError, match="ctxphi_hazard.*lysogeny"):
        validate_reference_control_reports(panel, reports)


@pytest.mark.parametrize("control_id", ["t7_negative", "phix174_negative", "muddy_negative"])
def test_negative_controls_reject_toxin_or_lysogeny_false_positives(control_id):
    panel = _panel()
    for safety_class in ("toxin", "lysogeny"):
        reports = _reports(panel)
        attempt = _attempt(reports[control_id], safety_class)
        attempt["primary_findings"] = [{"accession": "false-positive", "state": "FAIL", "reason_codes": ["fixture"]}]

        with pytest.raises(ReferenceControlError, match=f"{control_id}.*{safety_class}"):
            validate_reference_control_reports(panel, reports)


def test_reference_control_gate_rejects_identity_drift_and_incomplete_panels():
    panel = _panel()
    drifted = _reports(panel)
    drifted["muddy_negative"]["records"][0]["sequence_sha256"] = "0" * 64
    with pytest.raises(ReferenceControlError, match="muddy_negative.*sequence SHA-256"):
        validate_reference_control_reports(panel, drifted)

    incomplete = deepcopy(_reports(panel))
    incomplete.pop("phix174_negative")
    with pytest.raises(ReferenceControlError, match="control report set"):
        validate_reference_control_reports(panel, incomplete)
