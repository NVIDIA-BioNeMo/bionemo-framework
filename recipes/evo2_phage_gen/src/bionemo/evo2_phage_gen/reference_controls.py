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

"""Validate positive, review, and negative controls for sequence-safety scans."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from bionemo.evo2_phage_gen.sequence_safety import SafetyState


SAFETY_CLASSES = ("amr", "toxin", "lysogeny")
_ROLES = {"positive_hazard", "positive_review", "negative"}
_ACCESSION = re.compile(r"^(?:[A-Z]{2}_[0-9]+|[A-Z]{1,4}[0-9]{5,9})\.[1-9][0-9]*$")


class ReferenceControlError(ValueError):
    """Report an invalid reference-control definition or result."""

    pass


@dataclass(frozen=True)
class ReferenceClassExpectation:
    """Describe the expected result for one safety class."""

    state: SafetyState
    required_finding_accessions: Mapping[str, str]
    minimum_primary_findings: int
    allow_additional_findings: bool


@dataclass(frozen=True)
class ReferenceControl:
    """Describe one accession-based safety control."""

    control_id: str
    accession: str
    sequence_interval: tuple[int, int] | None
    display_name: str
    role: str
    topology: str
    sequence_length: int
    source_url: str
    evidence_urls: tuple[str, ...]
    expected_aggregate_state: SafetyState
    expected_classes: Mapping[str, ReferenceClassExpectation]

    @property
    def record_id(self) -> str:
        """Return the FASTA identifier expected for this control."""
        if self.sequence_interval is None:
            return self.accession
        start, end = self.sequence_interval
        return f"{self.accession}_{start}_{end}"

    @property
    def circular(self) -> bool:
        """Return whether the control sequence is circular."""
        return self.topology == "circular"


@dataclass(frozen=True)
class ReferenceControlPanel:
    """Collect the safety controls used for a comparison."""

    panel_id: str
    controls: tuple[ReferenceControl, ...]
    config_path: Path

    @property
    def by_id(self) -> dict[str, ReferenceControl]:
        """Index controls by their short identifiers."""
        return {control.control_id: control for control in self.controls}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReferenceControlError(f"{label} must be a mapping")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReferenceControlError(f"{label} must be a nonempty string")
    return value


def _expectation(value: object, label: str) -> ReferenceClassExpectation:
    payload = _mapping(value, label)
    try:
        state = SafetyState(payload["state"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReferenceControlError(f"{label}.state is invalid") from error
    accessions = _mapping(payload.get("required_finding_accessions", {}), f"{label}.required_finding_accessions")
    expected_accessions = {
        _text(gene, f"{label}.gene"): _text(accession, f"{label}.{gene}") for gene, accession in accessions.items()
    }
    minimum = payload.get("minimum_primary_findings", 0)
    allow_additional = payload.get("allow_additional_findings", False)
    if type(minimum) is not int or minimum < len(expected_accessions):
        raise ReferenceControlError(f"{label}.minimum_primary_findings is invalid")
    if type(allow_additional) is not bool:
        raise ReferenceControlError(f"{label}.allow_additional_findings must be boolean")
    if state is SafetyState.PASS and minimum:
        raise ReferenceControlError(f"{label} cannot require findings for PASS")
    return ReferenceClassExpectation(state, expected_accessions, minimum, allow_additional)


def _control(value: object, index: int) -> ReferenceControl:
    payload = _mapping(value, f"controls[{index}]")
    accession = _text(payload.get("accession"), f"controls[{index}].accession")
    if not _ACCESSION.fullmatch(accession):
        raise ReferenceControlError(f"{accession!r} is not an accession.version")
    interval_value = payload.get("sequence_interval")
    interval = None
    if interval_value is not None:
        raw_interval = _mapping(interval_value, f"{accession}.sequence_interval")
        start, end = raw_interval.get("start"), raw_interval.get("end")
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise ReferenceControlError(f"{accession} has an invalid interval")
        interval = (start, end)
    length = payload.get("sequence_length")
    if type(length) is not int or length < 1:
        raise ReferenceControlError(f"{accession} has an invalid sequence length")
    if interval is not None and length != interval[1] - interval[0] + 1:
        raise ReferenceControlError(f"{accession} interval and length disagree")
    role = _text(payload.get("role"), f"{accession}.role")
    topology = _text(payload.get("topology"), f"{accession}.topology")
    if role not in _ROLES or topology not in {"linear", "circular"}:
        raise ReferenceControlError(f"{accession} has an unsupported role or topology")
    classes = _mapping(payload.get("expected_classes"), f"{accession}.expected_classes")
    if set(classes) != set(SAFETY_CLASSES):
        raise ReferenceControlError(f"{accession} must define all safety classes")
    expected_classes = {name: _expectation(classes[name], f"{accession}.{name}") for name in SAFETY_CLASSES}
    try:
        aggregate = SafetyState(payload["expected_aggregate_state"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReferenceControlError(f"{accession} aggregate state is invalid") from error
    states = [item.state for item in expected_classes.values()]
    derived = (
        SafetyState.FAIL
        if SafetyState.FAIL in states
        else SafetyState.INDETERMINATE
        if SafetyState.INDETERMINATE in states
        else SafetyState.PASS
    )
    expected_for_role = {
        "positive_hazard": SafetyState.FAIL,
        "positive_review": SafetyState.INDETERMINATE,
        "negative": SafetyState.PASS,
    }[role]
    if aggregate is not derived or aggregate is not expected_for_role:
        raise ReferenceControlError(f"{accession} role and expected outcomes disagree")
    evidence = payload.get("evidence_urls", [])
    if not isinstance(evidence, list) or not all(isinstance(url, str) and url for url in evidence):
        raise ReferenceControlError(f"{accession} evidence_urls is invalid")
    return ReferenceControl(
        control_id=_text(payload.get("control_id"), f"{accession}.control_id"),
        accession=accession,
        sequence_interval=interval,
        display_name=_text(payload.get("display_name"), f"{accession}.display_name"),
        role=role,
        topology=topology,
        sequence_length=length,
        source_url=_text(payload.get("source_url"), f"{accession}.source_url"),
        evidence_urls=tuple(evidence),
        expected_aggregate_state=aggregate,
        expected_classes=expected_classes,
    )


def load_reference_control_panel(path: Path) -> ReferenceControlPanel:
    """Load and validate a safety-control panel."""
    config_path = Path(path).resolve()
    payload = _mapping(yaml.safe_load(config_path.read_text()), "reference control config")
    if payload.get("schema_version") != 1:
        raise ReferenceControlError("unsupported reference control schema")
    values = payload.get("controls")
    if not isinstance(values, list) or not values:
        raise ReferenceControlError("reference control panel is empty")
    controls = tuple(_control(value, index) for index, value in enumerate(values))
    ids = [control.control_id for control in controls]
    accessions = [control.accession for control in controls]
    if len(ids) != len(set(ids)) or len(accessions) != len(set(accessions)):
        raise ReferenceControlError("control IDs and accessions must be unique")
    roles = [control.role for control in controls]
    if "positive_hazard" not in roles or roles.count("negative") < 2:
        raise ReferenceControlError("panel needs positive-hazard and multiple negative controls")
    return ReferenceControlPanel(
        panel_id=_text(payload.get("panel_id"), "panel_id"),
        controls=controls,
        config_path=config_path,
    )


def _index(values: object, key: str, label: str) -> dict[str, Mapping[str, object]]:
    if not isinstance(values, list):
        raise ReferenceControlError(f"{label} must be a list")
    indexed: dict[str, Mapping[str, object]] = {}
    for value in values:
        row = _mapping(value, label)
        name = row.get(key)
        if not isinstance(name, str) or not name or name in indexed:
            raise ReferenceControlError(f"{label} contains a missing or duplicate {key}")
        indexed[name] = row
    return indexed


def _validate_report(control: ReferenceControl, report: Mapping[str, object]) -> dict[str, object]:
    """Check that a control was measured and compare its result with the reference outcome."""
    if report.get("manifest_type") != "sequence_safety_scan":
        raise ReferenceControlError(f"{control.control_id} is not a scan")
    profile = _mapping(report.get("resolved_profile"), f"{control.control_id}.resolved_profile")
    if (
        profile.get("host_domain") != "BACTERIA"
        or profile.get("strict_lysis") is not True
        or profile.get("circular") is not control.circular
    ):
        raise ReferenceControlError(f"{control.control_id} used the wrong host, lysis, or topology profile")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ReferenceControlError(f"{control.control_id} must contain one record")
    record = _mapping(records[0], f"{control.control_id}.record")
    if record.get("record_id") != control.record_id or record.get("sequence_length") != control.sequence_length:
        raise ReferenceControlError(f"{control.control_id} accession or length does not match")

    classes = _index(record.get("class_results"), "safety_class", f"{control.control_id}.class_results")
    attempts = _index(record.get("adapter_attempts"), "safety_class", f"{control.control_id}.attempts")
    if set(classes) != set(SAFETY_CLASSES) or set(attempts) != set(SAFETY_CLASSES):
        raise ReferenceControlError(f"{control.control_id} did not run every detector")

    review_reasons: list[str] = []
    observed_state = _text(record.get("state"), f"{control.control_id}.state")
    expected_state = control.expected_aggregate_state.value
    if observed_state != expected_state:
        review_reasons.append(f"aggregate changed from {expected_state} to {observed_state}")

    summary: dict[str, object] = {}
    for name, expectation in control.expected_classes.items():
        result = classes[name]
        attempt = attempts[name]
        findings = result.get("findings")
        if (
            result.get("required") is not True
            or attempt.get("execution_status") != "COMPLETED_AND_PARSED"
            or not isinstance(findings, list)
        ):
            raise ReferenceControlError(f"{control.control_id}.{name} did not produce a measured outcome")
        observed_class_state = _text(result.get("state"), f"{control.control_id}.{name}.state")
        accessions = {
            finding.get("accession")
            for finding in findings
            if isinstance(finding, Mapping) and isinstance(finding.get("accession"), str)
        }
        missing = sorted(set(expectation.required_finding_accessions.values()) - accessions)
        if observed_class_state != expectation.state.value:
            review_reasons.append(f"{name} changed from {expectation.state.value} to {observed_class_state}")
        if missing:
            review_reasons.append(f"{name} missed reference findings: {', '.join(missing)}")
        if len(findings) < expectation.minimum_primary_findings:
            review_reasons.append(f"{name} findings fell below {expectation.minimum_primary_findings}")
        if not expectation.allow_additional_findings and len(findings) > expectation.minimum_primary_findings:
            review_reasons.append(f"{name} has new findings")
        summary[name] = {
            "reference_state": expectation.state.value,
            "observed_state": observed_class_state,
            "finding_count": len(findings),
            "missing_reference_findings": missing,
        }

    return {
        "accession": control.accession,
        "role": control.role,
        "reference_filter_state": expected_state,
        "observed_filter_state": observed_state,
        "matches_reference": not review_reasons,
        "review_reasons": review_reasons,
        "classes": summary,
    }


def validate_reference_control_reports(
    panel: ReferenceControlPanel,
    reports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Summarize current control behavior without requiring an older database release."""
    if set(reports) != set(panel.by_id):
        raise ReferenceControlError("reports must cover the complete control panel")
    controls = {
        control.control_id: _validate_report(control, reports[control.control_id]) for control in panel.controls
    }
    changed = [control_id for control_id, result in controls.items() if not result["matches_reference"]]
    return {
        "schema_version": 1,
        "panel_id": panel.panel_id,
        "panel_config": str(panel.config_path),
        "state": "REVIEW" if changed else "PASS",
        "changed_controls": changed,
        "controls": controls,
        "claim_boundary": (
            "These controls record how the current software, tool versions, and database releases behaved. "
            "A changed result needs scientific review, not automatic rollback to an older database; "
            "the controls are not wet-lab, clinical, or regulatory validation."
        ),
    }


def _report_arguments(values: Sequence[str]) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for value in values:
        control_id, separator, path = value.partition("=")
        if not separator or not control_id or not path or control_id in reports:
            raise ReferenceControlError("--report values must be unique CONTROL_ID=MANIFEST_PATH pairs")
        reports[control_id] = Path(path)
    return reports


def build_parser() -> argparse.ArgumentParser:
    """Build the reference-control validation parser."""
    parser = argparse.ArgumentParser(prog="evo2_phage_validate_safety_controls")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="CONTROL_ID=MANIFEST_PATH",
        help="Repeat for every control in the panel.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate current control scans against the reference outcomes."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        panel = load_reference_control_panel(args.config)
        paths = _report_arguments(args.report)
        from bionemo.evo2_phage_gen.sequence_safety_cli import validate_manifest_file

        reports = {
            control_id: validate_manifest_file(path, expected_type="sequence_safety_scan")
            for control_id, path in paths.items()
        }
        result = validate_reference_control_reports(panel, reports)
        serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(serialized)
        else:
            print(serialized, end="")
        return 0 if result["state"] == "PASS" else 2
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser._print_message(f"{parser.prog}: error: {error}\n", sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
