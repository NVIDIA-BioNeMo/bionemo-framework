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

"""Reference controls that must all match their expected sequence-safety outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from bionemo.evo2_phage_gen.sequence_safety import SafetyState


_SAFETY_CLASSES = ("amr", "toxin", "lysogeny")
_CONTROL_ROLES = frozenset({"positive_hazard", "positive_review", "negative"})
_TOPOLOGIES = frozenset({"linear", "circular"})
_ACCESSION_PATTERN = re.compile(r"^(?:[A-Z]{2}_[0-9]+|[A-Z]{1,4}[0-9]{5,9})\.[1-9][0-9]*$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReferenceControlError(ValueError):
    """The control definition or measured report cannot establish filter fitness."""


def _strict_mapping(value: object, *, name: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReferenceControlError(f"{name} must be a mapping")
    actual = frozenset(value)
    if actual != keys:
        raise ReferenceControlError(
            f"{name} keys do not match schema; unknown={sorted(actual - keys)}, missing={sorted(keys - actual)}"
        )
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReferenceControlError(f"{name} must be a non-empty string")
    return value


def _string_tuple(value: object, *, name: str, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ReferenceControlError(f"{name} must be a string list")
    if not allow_empty and not value:
        raise ReferenceControlError(f"{name} must not be empty")
    if len(value) != len(set(value)):
        raise ReferenceControlError(f"{name} contains duplicates")
    return tuple(value)


@dataclass(frozen=True)
class ReferenceClassExpectation:
    """Expected measured outcome for one safety class in one reference control."""

    state: SafetyState
    reason_codes: tuple[str, ...]
    required_finding_accessions: Mapping[str, str]
    minimum_primary_findings: int
    allow_additional_findings: bool

    def __post_init__(self) -> None:
        """Freeze nested values and reject expectations that could accept an unmeasured outcome."""
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(
            self,
            "required_finding_accessions",
            MappingProxyType(dict(self.required_finding_accessions)),
        )
        if not self.reason_codes:
            raise ReferenceControlError("reference class expectations require reason codes")
        if self.minimum_primary_findings < len(self.required_finding_accessions):
            raise ReferenceControlError("minimum findings cannot be smaller than required accessions")
        if self.state is SafetyState.PASS and self.minimum_primary_findings:
            raise ReferenceControlError("a PASS expectation cannot require findings")


@dataclass(frozen=True)
class ReferenceControl:
    """One immutable genome identity and its expected safety-filter behavior."""

    control_id: str
    accession: str
    sequence_interval: tuple[int, int] | None
    display_name: str
    role: str
    topology: str
    sequence_length: int
    sequence_sha256: str
    source_url: str
    evidence_urls: tuple[str, ...]
    expected_aggregate_state: SafetyState
    expected_classes: Mapping[str, ReferenceClassExpectation]

    def __post_init__(self) -> None:
        """Freeze nested values."""
        object.__setattr__(
            self, "sequence_interval", None if self.sequence_interval is None else tuple(self.sequence_interval)
        )
        object.__setattr__(self, "evidence_urls", tuple(self.evidence_urls))
        object.__setattr__(self, "expected_classes", MappingProxyType(dict(self.expected_classes)))

    @property
    def record_id(self) -> str:
        """Return the scanner-safe ID for a complete record or exact source interval."""
        if self.sequence_interval is None:
            return self.accession
        start, end = self.sequence_interval
        return f"{self.accession}_{start}_{end}"

    @property
    def circular(self) -> bool:
        """Return the exact topology flag consumed by the scanner."""
        return self.topology == "circular"


@dataclass(frozen=True)
class ReferenceControlPanel:
    """Versioned complete control panel and its source identity."""

    panel_id: str
    sequence_identity: Mapping[str, str]
    controls: tuple[ReferenceControl, ...]
    config_path: Path
    config_sha256: str

    def __post_init__(self) -> None:
        """Freeze the panel."""
        object.__setattr__(self, "sequence_identity", MappingProxyType(dict(self.sequence_identity)))
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "config_path", Path(self.config_path))

    @property
    def by_id(self) -> Mapping[str, ReferenceControl]:
        """Index controls by stable identifier."""
        return MappingProxyType({control.control_id: control for control in self.controls})


def _parse_class_expectation(value: object, *, name: str) -> ReferenceClassExpectation:
    payload = _strict_mapping(
        value,
        name=name,
        keys=frozenset(
            {
                "state",
                "reason_codes",
                "required_finding_accessions",
                "minimum_primary_findings",
                "allow_additional_findings",
            }
        ),
    )
    try:
        state = SafetyState(payload["state"])
    except (TypeError, ValueError) as error:
        raise ReferenceControlError(f"{name}.state is invalid") from error
    required = payload["required_finding_accessions"]
    if not isinstance(required, Mapping):
        raise ReferenceControlError(f"{name}.required_finding_accessions must be a mapping")
    required_accessions: dict[str, str] = {}
    for gene, accession in required.items():
        gene_name = _nonempty_string(gene, name=f"{name}.required_finding_accessions key")
        required_accessions[gene_name] = _nonempty_string(
            accession,
            name=f"{name}.required_finding_accessions.{gene_name}",
        )
    if len(required_accessions.values()) != len(set(required_accessions.values())):
        raise ReferenceControlError(f"{name}.required_finding_accessions must be independently identifiable")
    minimum = payload["minimum_primary_findings"]
    allow_additional = payload["allow_additional_findings"]
    if type(minimum) is not int or minimum < 0:
        raise ReferenceControlError(f"{name}.minimum_primary_findings must be a non-negative integer")
    if type(allow_additional) is not bool:
        raise ReferenceControlError(f"{name}.allow_additional_findings must be boolean")
    return ReferenceClassExpectation(
        state=state,
        reason_codes=_string_tuple(payload["reason_codes"], name=f"{name}.reason_codes", allow_empty=False),
        required_finding_accessions=required_accessions,
        minimum_primary_findings=minimum,
        allow_additional_findings=allow_additional,
    )


def _parse_control(value: object, *, index: int) -> ReferenceControl:
    name = f"controls[{index}]"
    payload = _strict_mapping(
        value,
        name=name,
        keys=frozenset(
            {
                "control_id",
                "accession",
                "sequence_interval",
                "display_name",
                "role",
                "topology",
                "sequence_length",
                "sequence_sha256",
                "source_url",
                "evidence_urls",
                "expected_aggregate_state",
                "expected_classes",
            }
        ),
    )
    control_id = _nonempty_string(payload["control_id"], name=f"{name}.control_id")
    accession = _nonempty_string(payload["accession"], name=f"{name}.accession")
    role = _nonempty_string(payload["role"], name=f"{name}.role")
    topology = _nonempty_string(payload["topology"], name=f"{name}.topology")
    digest = _nonempty_string(payload["sequence_sha256"], name=f"{name}.sequence_sha256")
    length = payload["sequence_length"]
    if not _ACCESSION_PATTERN.fullmatch(accession):
        raise ReferenceControlError(f"{name}.accession must be an exact INSDC accession.version")
    interval_value = payload["sequence_interval"]
    sequence_interval: tuple[int, int] | None
    if interval_value is None:
        sequence_interval = None
    else:
        interval = _strict_mapping(
            interval_value,
            name=f"{name}.sequence_interval",
            keys=frozenset({"start", "end"}),
        )
        start, end = interval["start"], interval["end"]
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise ReferenceControlError(f"{name}.sequence_interval must be a valid one-based closed interval")
        sequence_interval = (start, end)
    if role not in _CONTROL_ROLES:
        raise ReferenceControlError(f"{name}.role is unsupported")
    if topology not in _TOPOLOGIES:
        raise ReferenceControlError(f"{name}.topology is unsupported")
    if type(length) is not int or length < 1:
        raise ReferenceControlError(f"{name}.sequence_length must be a positive integer")
    if sequence_interval is not None and length != sequence_interval[1] - sequence_interval[0] + 1:
        raise ReferenceControlError(f"{name}.sequence_length does not match its exact interval")
    if not _DIGEST_PATTERN.fullmatch(digest):
        raise ReferenceControlError(f"{name}.sequence_sha256 must be a lowercase SHA-256 digest")
    try:
        aggregate_state = SafetyState(payload["expected_aggregate_state"])
    except (TypeError, ValueError) as error:
        raise ReferenceControlError(f"{name}.expected_aggregate_state is invalid") from error
    classes = payload["expected_classes"]
    if not isinstance(classes, Mapping) or set(classes) != set(_SAFETY_CLASSES):
        raise ReferenceControlError(f"{name}.expected_classes must contain exactly {_SAFETY_CLASSES}")
    expected_classes = {
        safety_class: _parse_class_expectation(
            classes[safety_class],
            name=f"{name}.expected_classes.{safety_class}",
        )
        for safety_class in _SAFETY_CLASSES
    }
    if any(expectation.state is SafetyState.FAIL for expectation in expected_classes.values()):
        derived_aggregate = SafetyState.FAIL
    elif any(expectation.state is SafetyState.INDETERMINATE for expectation in expected_classes.values()):
        derived_aggregate = SafetyState.INDETERMINATE
    else:
        derived_aggregate = SafetyState.PASS
    if aggregate_state is not derived_aggregate:
        raise ReferenceControlError(f"{name}.expected_aggregate_state conflicts with class expectations")
    if role == "negative" and aggregate_state is not SafetyState.PASS:
        raise ReferenceControlError(f"{name} negative controls must expect PASS")
    if role == "positive_hazard" and aggregate_state is not SafetyState.FAIL:
        raise ReferenceControlError(f"{name} positive hazard controls must independently establish FAIL")
    if role == "positive_review" and aggregate_state is not SafetyState.INDETERMINATE:
        raise ReferenceControlError(f"{name} positive review controls must establish INDETERMINATE")
    return ReferenceControl(
        control_id=control_id,
        accession=accession,
        sequence_interval=sequence_interval,
        display_name=_nonempty_string(payload["display_name"], name=f"{name}.display_name"),
        role=role,
        topology=topology,
        sequence_length=length,
        sequence_sha256=digest,
        source_url=_nonempty_string(payload["source_url"], name=f"{name}.source_url"),
        evidence_urls=_string_tuple(payload["evidence_urls"], name=f"{name}.evidence_urls"),
        expected_aggregate_state=aggregate_state,
        expected_classes=expected_classes,
    )


def load_reference_control_panel(path: Path) -> ReferenceControlPanel:
    """Load the exact schema-v2 reference panel without implicit latest-version resolution."""
    supplied_path = Path(path)
    if supplied_path.is_symlink() or not supplied_path.is_file():
        raise ReferenceControlError("reference control config must be a non-symlink regular file")
    config_path = supplied_path.resolve()
    payload_bytes = config_path.read_bytes()
    try:
        loaded = yaml.safe_load(payload_bytes)
    except yaml.YAMLError as error:
        raise ReferenceControlError(f"cannot parse reference control config: {error}") from error
    payload = _strict_mapping(
        loaded,
        name="reference control config",
        keys=frozenset({"schema_version", "panel_id", "sequence_identity", "controls"}),
    )
    if payload["schema_version"] != 2:
        raise ReferenceControlError("unsupported reference control schema version")
    identity = _strict_mapping(
        payload["sequence_identity"],
        name="sequence_identity",
        keys=frozenset({"retrieval", "normalization", "digest", "payload_distribution"}),
    )
    expected_identity = {
        "retrieval": "exact_insdc_accession_version_with_optional_1_based_closed_interval",
        "normalization": "uppercase_sequence_without_whitespace",
        "digest": "sha256",
        "payload_distribution": "accession_and_digest_only",
    }
    if dict(identity) != expected_identity:
        raise ReferenceControlError("sequence identity policy does not match schema v2")
    raw_controls = payload["controls"]
    if not isinstance(raw_controls, list) or not raw_controls:
        raise ReferenceControlError("controls must be a non-empty list")
    controls = tuple(_parse_control(value, index=index) for index, value in enumerate(raw_controls))
    ids = [control.control_id for control in controls]
    accessions = [control.accession for control in controls]
    if len(ids) != len(set(ids)) or len(accessions) != len(set(accessions)):
        raise ReferenceControlError("control IDs and accessions must be unique")
    if sum(control.role == "positive_hazard" for control in controls) < 1:
        raise ReferenceControlError("the panel requires at least one positive hazard control")
    if sum(control.role == "negative" for control in controls) < 2:
        raise ReferenceControlError("the panel requires at least two negative controls")
    return ReferenceControlPanel(
        panel_id=_nonempty_string(payload["panel_id"], name="panel_id"),
        sequence_identity=expected_identity,
        controls=controls,
        config_path=config_path,
        config_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def _indexed_rows(value: object, *, name: str, key: str) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(value, list):
        raise ReferenceControlError(f"{name} must be a list")
    indexed: dict[str, Mapping[str, object]] = {}
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ReferenceControlError(f"{name}[{index}] must be a mapping")
        row_key = row.get(key)
        if not isinstance(row_key, str) or not row_key or row_key in indexed:
            raise ReferenceControlError(f"{name} contains a missing or duplicate {key}")
        indexed[row_key] = row
    return indexed


def _validate_control_report(control: ReferenceControl, report: Mapping[str, object]) -> dict[str, object]:
    label = control.control_id
    if report.get("manifest_type") != "sequence_safety_scan":
        raise ReferenceControlError(f"{label} is not a sequence-safety scan manifest")
    profile = report.get("resolved_profile")
    if (
        not isinstance(profile, Mapping)
        or profile.get("host_domain") != "BACTERIA"
        or profile.get("strict_lysis") is not True
    ):
        raise ReferenceControlError(f"{label} must use the strict bacterial lysis profile")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
        raise ReferenceControlError(f"{label} must contain exactly one control record")
    record = records[0]
    identity_checks = (
        ("record ID", record.get("record_id"), control.record_id),
        ("sequence length", record.get("sequence_length"), control.sequence_length),
        ("sequence SHA-256", record.get("sequence_sha256"), control.sequence_sha256),
        ("topology", record.get("circular"), control.circular),
    )
    for field, observed, expected in identity_checks:
        if observed != expected:
            raise ReferenceControlError(f"{label} {field} does not match the pinned control")
    if record.get("strict_lysis") is not True:
        raise ReferenceControlError(f"{label} record did not use strict lysis")
    expected_aggregate = control.expected_aggregate_state.value
    aggregate = report.get("aggregate")
    if (
        record.get("state") != expected_aggregate
        or not isinstance(aggregate, Mapping)
        or aggregate.get("state") != expected_aggregate
    ):
        raise ReferenceControlError(f"{label} aggregate outcome does not match the control expectation")
    class_results = _indexed_rows(record.get("class_results"), name=f"{label}.class_results", key="safety_class")
    attempts = _indexed_rows(record.get("adapter_attempts"), name=f"{label}.adapter_attempts", key="safety_class")
    if set(class_results) != set(_SAFETY_CLASSES) or set(attempts) != set(_SAFETY_CLASSES):
        raise ReferenceControlError(f"{label} must report every required safety class exactly once")
    supplemental_reasons: dict[str, list[str]] = {safety_class: [] for safety_class in _SAFETY_CLASSES}
    for source_class, attempt in attempts.items():
        supplemental = attempt.get("supplemental_findings")
        if not isinstance(supplemental, list) or not all(isinstance(finding, Mapping) for finding in supplemental):
            raise ReferenceControlError(f"{label} {source_class} supplemental findings are malformed")
        if control.role == "negative" and supplemental:
            raise ReferenceControlError(f"{label} negative control produced unexpected supplemental evidence")
        for finding in supplemental:
            reason_codes = finding.get("reason_codes")
            if (
                source_class != "amr"
                or finding.get("safety_class") != "toxin"
                or finding.get("state") != "INDETERMINATE"
                or reason_codes != ["AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL"]
            ):
                raise ReferenceControlError(f"{label} contains unsupported supplemental safety evidence")
            for reason_code in reason_codes:
                if reason_code not in supplemental_reasons["toxin"]:
                    supplemental_reasons["toxin"].append(reason_code)
    class_summary: dict[str, object] = {}
    for safety_class, expectation in control.expected_classes.items():
        expected_state = expectation.state.value
        expected_reasons = list(expectation.reason_codes)
        expected_class_reasons = [*expected_reasons, *supplemental_reasons[safety_class]]
        class_result = class_results[safety_class]
        attempt = attempts[safety_class]
        if (
            class_result.get("required") is not True
            or class_result.get("state") != expected_state
            or class_result.get("reason_codes") != expected_class_reasons
            or attempt.get("execution_status") != "COMPLETED_AND_PARSED"
            or attempt.get("state") != expected_state
            or attempt.get("reason_codes") != expected_reasons
        ):
            raise ReferenceControlError(f"{label} {safety_class} outcome is not the required measured result")
        findings = attempt.get("primary_findings")
        if not isinstance(findings, list) or not all(isinstance(finding, Mapping) for finding in findings):
            raise ReferenceControlError(f"{label} {safety_class} primary findings are malformed")
        accessions = [finding.get("accession") for finding in findings]
        if not all(isinstance(accession, str) and accession for accession in accessions):
            raise ReferenceControlError(f"{label} {safety_class} finding accessions are malformed")
        for gene, accession in expectation.required_finding_accessions.items():
            if accession not in accessions:
                raise ReferenceControlError(
                    f"{label} {safety_class} did not independently detect {gene} ({accession})"
                )
        if len(findings) < expectation.minimum_primary_findings:
            raise ReferenceControlError(f"{label} {safety_class} has too few primary findings")
        if not expectation.allow_additional_findings and len(findings) != expectation.minimum_primary_findings:
            raise ReferenceControlError(f"{label} {safety_class} produced an unexpected finding")
        class_summary[safety_class] = {
            "state": expected_state,
            "reason_codes": expected_class_reasons,
            "primary_finding_count": len(findings),
        }
    from bionemo.evo2_phage_gen.reward import sequence_safety_reward_contract

    rl_safety = sequence_safety_reward_contract(
        class_states={name: expectation.state.value for name, expectation in control.expected_classes.items()},
        required_by_class=dict.fromkeys(_SAFETY_CLASSES, True),
        review_eligible_by_class={
            name: expectation.state is SafetyState.INDETERMINATE and class_summary[name]["primary_finding_count"] > 0
            for name, expectation in control.expected_classes.items()
        },
    )
    if rl_safety["safety_gate_state"] != expected_aggregate:
        raise ReferenceControlError(f"{label} RL safety gate conflicts with the hard-filter expectation")
    return {
        "accession": control.accession,
        "role": control.role,
        "state": "PASS",
        "expected_filter_state": expected_aggregate,
        "classes": class_summary,
        "rl_safety": rl_safety,
    }


def validate_reference_control_reports(
    panel: ReferenceControlPanel,
    reports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Require the complete panel to exhibit its versioned measured outcomes."""
    expected_ids = set(panel.by_id)
    if set(reports) != expected_ids:
        raise ReferenceControlError(
            f"control report set does not match the panel; missing={sorted(expected_ids - set(reports))}, "
            f"unknown={sorted(set(reports) - expected_ids)}"
        )
    results = {
        control.control_id: _validate_control_report(control, reports[control.control_id])
        for control in panel.controls
    }
    return {
        "schema_version": 1,
        "panel_id": panel.panel_id,
        "panel_config": {"path": str(panel.config_path), "sha256": panel.config_sha256},
        "state": "PASS",
        "controls": results,
        "claim_boundary": (
            "Passing controls establishes software sensitivity/specificity for this pinned panel only; "
            "it is not biological, therapeutic, clinical, or regulatory validation."
        ),
    }


def _report_arguments(values: Sequence[str]) -> Mapping[str, Path]:
    reports: dict[str, Path] = {}
    for value in values:
        control_id, separator, path_value = value.partition("=")
        if not separator or not control_id or not path_value or control_id in reports:
            raise ReferenceControlError("--report must be a unique CONTROL_ID=MANIFEST_PATH value")
        reports[control_id] = Path(path_value)
    return reports


def build_parser() -> argparse.ArgumentParser:
    """Build the reference-control validation CLI."""
    parser = argparse.ArgumentParser(prog="evo2_phage_validate_safety_controls")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True, help="CONTROL_ID=scan/manifest.json")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Revalidate scan manifests and accept only when the whole panel matches."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        panel = load_reference_control_panel(args.config)
        report_paths = _report_arguments(args.report)
        from bionemo.evo2_phage_gen.sequence_safety_cli import validate_manifest_file

        reports = {
            control_id: validate_manifest_file(path, expected_type="sequence_safety_scan")
            for control_id, path in report_paths.items()
        }
        result = validate_reference_control_reports(panel, reports)
        serialized = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(serialized, end="")
        else:
            args.output.write_text(serialized, encoding="utf-8")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        sys.stderr.write(f"{parser.prog}: error: {error}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
