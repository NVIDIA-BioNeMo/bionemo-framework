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

import json
from pathlib import Path

import pytest
import yaml

from bionemo.evo2_phage_gen import reference_controls
from bionemo.evo2_phage_gen.reference_controls import (
    ReferenceControlError,
    load_reference_control_panel,
    validate_reference_control_reports,
)


CONFIG = Path(__file__).parents[3] / "configs" / "phage_safety_reference_controls.yaml"


def _report(control) -> dict[str, object]:
    class_results = []
    attempts = []
    for name, expectation in control.expected_classes.items():
        accessions = list(expectation.required_finding_accessions.values())
        while len(accessions) < expectation.minimum_primary_findings:
            accessions.append(f"observed-{name}-{len(accessions)}")
        findings = [
            {
                "safety_class": name,
                "state": expectation.state.value,
                "reason_codes": [f"{name.upper()}_CONTROL_SIGNAL"],
                "finding_id": f"{name}-{index}",
                "accession": accession,
            }
            for index, accession in enumerate(accessions)
        ]
        class_results.append(
            {
                "safety_class": name,
                "state": expectation.state.value,
                "required": True,
                "reason_codes": [f"{name.upper()}_CONTROL_RESULT"],
                "findings": findings,
            }
        )
        attempts.append(
            {
                "safety_class": name,
                "execution_status": "COMPLETED_AND_PARSED",
                "policy_id": f"{name}-policy",
            }
        )
    state = control.expected_aggregate_state.value
    return {
        "schema_version": 2,
        "manifest_type": "sequence_safety_scan",
        "resolved_profile": {
            "host_domain": "BACTERIA",
            "strict_lysis": True,
            "circular": control.circular,
        },
        "records": [
            {
                "input_index": 0,
                "record_id": control.record_id,
                "sequence_length": control.sequence_length,
                "state": state,
                "reason_codes": [],
                "class_results": class_results,
                "adapter_attempts": attempts,
            }
        ],
        "aggregate": {
            "state": state,
            "counts": {
                "PASS": int(state == "PASS"),
                "FAIL": int(state == "FAIL"),
                "INDETERMINATE": int(state == "INDETERMINATE"),
            },
        },
    }


def test_panel_covers_positive_review_and_negative_controls() -> None:
    panel = load_reference_control_panel(CONFIG)
    roles = {control.role for control in panel.controls}
    assert roles == {"positive_hazard", "positive_review", "negative"}
    assert panel.by_id["phix174_negative"].accession == "NC_001422.1"


def test_panel_uses_scientific_identifiers_and_expected_states() -> None:
    payload = yaml.safe_load(CONFIG.read_text())
    assert "sequence_identity" not in payload
    assert all(control["sequence_length"] > 0 for control in payload["controls"])


def test_complete_measured_control_panel_passes() -> None:
    panel = load_reference_control_panel(CONFIG)
    reports = {control.control_id: _report(control) for control in panel.controls}
    result = validate_reference_control_reports(panel, reports)
    assert result["state"] == "PASS"
    assert set(result["controls"]) == set(panel.by_id)
    assert "tool versions" in result["claim_boundary"]


def test_missing_detector_or_failed_attempt_rejects_control() -> None:
    panel = load_reference_control_panel(CONFIG)
    reports = {control.control_id: _report(control) for control in panel.controls}
    first = panel.controls[0]
    reports[first.control_id]["records"][0]["adapter_attempts"][1]["execution_status"] = "FAILED"
    with pytest.raises(ReferenceControlError, match="measured outcome"):
        validate_reference_control_reports(panel, reports)


@pytest.mark.parametrize("field", ("record_id", "sequence_length"))
def test_accession_and_length_mismatch_rejects_control(field: str) -> None:
    panel = load_reference_control_panel(CONFIG)
    reports = {control.control_id: _report(control) for control in panel.controls}
    first = panel.controls[0]
    reports[first.control_id]["records"][0][field] = "wrong" if field == "record_id" else 1
    with pytest.raises(ReferenceControlError, match="accession or length"):
        validate_reference_control_reports(panel, reports)


def test_topology_mismatch_rejects_control() -> None:
    panel = load_reference_control_panel(CONFIG)
    reports = {control.control_id: _report(control) for control in panel.controls}
    control = panel.by_id["phix174_negative"]
    reports[control.control_id]["resolved_profile"]["circular"] = False
    with pytest.raises(ReferenceControlError, match="topology"):
        validate_reference_control_reports(panel, reports)


def test_changed_positive_signal_is_reported_for_review() -> None:
    panel = load_reference_control_panel(CONFIG)
    reports = {control.control_id: _report(control) for control in panel.controls}
    control = panel.by_id["ctxphi_hazard"]
    toxin = next(
        result
        for result in reports[control.control_id]["records"][0]["class_results"]
        if result["safety_class"] == "toxin"
    )
    toxin["findings"] = []

    result = validate_reference_control_reports(panel, reports)

    assert result["state"] == "REVIEW"
    assert control.control_id in result["changed_controls"]
    assert any(
        "missed reference findings" in reason for reason in result["controls"][control.control_id]["review_reasons"]
    )


def test_new_database_finding_is_logged_without_requiring_an_older_database() -> None:
    panel = load_reference_control_panel(CONFIG)
    reports = {control.control_id: _report(control) for control in panel.controls}
    control = panel.by_id["phix174_negative"]
    toxin = next(
        result
        for result in reports[control.control_id]["records"][0]["class_results"]
        if result["safety_class"] == "toxin"
    )
    toxin["findings"].append(
        {
            "safety_class": "toxin",
            "state": "FAIL",
            "reason_codes": ["NEW_DATABASE_HIT"],
            "finding_id": "new-hit",
            "accession": "NEW0001",
        }
    )
    toxin["state"] = "FAIL"
    reports[control.control_id]["records"][0]["state"] = "FAIL"
    reports[control.control_id]["aggregate"]["state"] = "FAIL"
    reports[control.control_id]["aggregate"]["counts"] = {"PASS": 0, "FAIL": 1, "INDETERMINATE": 0}

    result = validate_reference_control_reports(panel, reports)

    assert result["state"] == "REVIEW"
    assert "older database" in result["claim_boundary"]
    assert result["controls"][control.control_id]["observed_filter_state"] == "FAIL"


def test_reports_must_cover_the_whole_panel() -> None:
    panel = load_reference_control_panel(CONFIG)
    reports = {control.control_id: _report(control) for control in panel.controls[:-1]}
    with pytest.raises(ReferenceControlError, match="complete control panel"):
        validate_reference_control_reports(panel, reports)


def test_cli_writes_small_summary(tmp_path: Path, monkeypatch) -> None:
    panel = load_reference_control_panel(CONFIG)
    reports = {control.control_id: _report(control) for control in panel.controls}
    report_paths = {}
    for control_id, report in reports.items():
        path = tmp_path / f"{control_id}.json"
        path.write_text(json.dumps(report))
        report_paths[control_id] = path
    monkeypatch.setattr(
        reference_controls,
        "load_reference_control_panel",
        lambda _path: panel,
    )
    output = tmp_path / "summary.json"
    args = ["--config", str(CONFIG), "--output", str(output)]
    for control_id, path in report_paths.items():
        args.extend(("--report", f"{control_id}={path}"))
    assert reference_controls.main(args) == 0
    assert json.loads(output.read_text())["state"] == "PASS"
