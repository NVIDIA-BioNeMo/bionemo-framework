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

"""Render a bounded report from authenticated SFT sequence-safety manifests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from bionemo.evo2_phage_gen import sft as sft_module


@dataclass(frozen=True)
class _SafetyReportRecord:
    record_id: str
    severity: str
    failing_classes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    detectors: tuple[str, ...]
    finding_counts: tuple[tuple[str, int], ...]

    @property
    def finding_count(self) -> int:
        """Return the number of FAIL findings summarized for this record."""
        return sum(count for _safety_class, count in self.finding_counts)


_SEVERITY_ORDER = {"Higher": 0, "Elevated": 1, "Unclassified": 2}
_WATERFALL_COMBINATIONS = (
    (frozenset({"lysogeny"}), "Lysogeny only"),
    (frozenset({"amr"}), "Antibiotic resistance only"),
    (frozenset({"toxin"}), "Toxin only"),
    (frozenset({"toxin", "lysogeny"}), "Toxin + lysogeny"),
    (frozenset({"amr", "lysogeny"}), "Antibiotic resistance + lysogeny"),
    (frozenset({"amr", "toxin"}), "Antibiotic resistance + toxin"),
    (
        frozenset({"amr", "toxin", "lysogeny"}),
        "Antibiotic resistance + toxin + lysogeny",
    ),
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise sft_module.SFTSafetyError(f"{label} must be a mapping")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise sft_module.SFTSafetyError(f"{label} must be a list")
    return value


def _count(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise sft_module.SFTSafetyError(f"{label} must be a nonnegative integer")
    return value


def _markdown_cell(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _summarize_failed_record(decision: Mapping[str, object], scan_record: Mapping[str, object]) -> _SafetyReportRecord:
    record_id = decision.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise sft_module.SFTSafetyError("failed SFT decision has no stable record ID")
    if scan_record.get("state") != "FAIL":
        raise sft_module.SFTSafetyError(f"failed SFT decision does not match its trusted scan record: {record_id}")

    failing_classes: list[str] = []
    reason_codes: list[str] = []
    detectors: list[str] = []
    finding_counts: list[tuple[str, int]] = []
    class_values = _list(scan_record.get("class_results"), label=f"{record_id} class_results")
    for class_value in class_values:
        class_result = _mapping(class_value, label=f"{record_id} class result")
        if class_result.get("required") is not True or class_result.get("state") != "FAIL":
            continue
        safety_class = class_result.get("safety_class")
        if not isinstance(safety_class, str) or not safety_class:
            raise sft_module.SFTSafetyError(f"{record_id} failing class has no name")
        failing_classes.append(safety_class)
        class_reasons = _list(
            class_result.get("reason_codes"),
            label=f"{record_id}.{safety_class} reason_codes",
        )
        for reason in class_reasons:
            if not isinstance(reason, str) or not reason:
                raise sft_module.SFTSafetyError(f"{record_id}.{safety_class} reason code is malformed")
            reason_codes.append(reason)
        class_finding_count = 0
        findings = _list(
            class_result.get("findings"),
            label=f"{record_id}.{safety_class} findings",
        )
        for finding_value in findings:
            finding = _mapping(
                finding_value,
                label=f"{record_id}.{safety_class} finding",
            )
            if finding.get("state") != "FAIL":
                continue
            class_finding_count += 1
            detector = finding.get("detector")
            if isinstance(detector, str) and detector:
                detectors.append(detector)
            finding_reasons = _list(
                finding.get("reason_codes"),
                label=f"{record_id}.{safety_class} finding reason_codes",
            )
            for reason in finding_reasons:
                if not isinstance(reason, str) or not reason:
                    raise sft_module.SFTSafetyError(f"{record_id}.{safety_class} finding reason code is malformed")
                reason_codes.append(reason)
        finding_counts.append((safety_class, class_finding_count))

    unique_classes = tuple(dict.fromkeys(failing_classes))
    if {"amr", "toxin"}.intersection(unique_classes):
        severity = "Higher"
    elif "lysogeny" in unique_classes:
        severity = "Elevated"
    else:
        severity = "Unclassified"
    return _SafetyReportRecord(
        record_id=record_id,
        severity=severity,
        failing_classes=unique_classes,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        detectors=tuple(dict.fromkeys(detectors)),
        finding_counts=tuple(finding_counts),
    )


def _render(
    *,
    total: int,
    passed: int,
    retained: int,
    removed: Sequence[_SafetyReportRecord],
    unresolved: int,
    other_exclusions: int,
    max_highlights: int,
) -> bytes:
    severity_counts = dict.fromkeys(_SEVERITY_ORDER, 0)
    class_record_counts: dict[str, int] = {}
    class_finding_counts: dict[str, int] = {}
    combination_counts = {combination: 0 for combination, _label in _WATERFALL_COMBINATIONS}
    other_combination_count = 0
    for record in removed:
        severity_counts[record.severity] += 1
        combination = frozenset(record.failing_classes)
        if combination in combination_counts:
            combination_counts[combination] += 1
        else:
            other_combination_count += 1
        for safety_class in record.failing_classes:
            class_record_counts[safety_class] = class_record_counts.get(safety_class, 0) + 1
        for safety_class, count in record.finding_counts:
            class_finding_counts[safety_class] = class_finding_counts.get(safety_class, 0) + count

    lines = [
        "# SFT sequence-safety filter report",
        "",
        "## High-level outcome",
        "",
        "| Measure | Count |",
        "| --- | ---: |",
        f"| Total source records examined | {total} |",
        f"| Retained for SFT | {retained} |",
        f"| Removed after potential sequence-safety signals | {len(removed)} |",
        f"| Unresolved sequence-safety assessments | {unresolved} |",
        f"| Other data-preparation exclusions | {other_exclusions} |",
        "",
        "## Safety-filter waterfall",
        "",
        "| Count | Stage |",
        "| ---: | --- |",
        f"| {total} | Starting set of phage genomes |",
        f"| {passed} | No observed or known safety signals from configured filters |",
        f"| {len(removed)} | Phage genomes failing the sequence-safety filter |",
    ]
    lines.extend(
        f"| {combination_counts[combination]} | &emsp;↳ {label} |" for combination, label in _WATERFALL_COMBINATIONS
    )
    if other_combination_count:
        lines.append(f"| {other_combination_count} | &emsp;↳ Other safety-signal combinations |")
    lines.extend(
        [
            f"| {unresolved} | Safety assessments unresolved by configured filters |",
            "",
            "## Interpretation boundary",
            "",
            "This report summarizes potential sequence-safety signals from the configured computational filters. "
            "It is not a clinical safety conclusion, risk assessment, or statement that a source organism is dangerous.",
            "",
            "## Removed records by potential severity",
            "",
            "| Potential severity | Removed records |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| {severity} | {severity_counts[severity]} |" for severity in _SEVERITY_ORDER)
    lines.extend(
        [
            "",
            "These are transparent report-triage categories: **Higher** means at least one required AMR or toxin "
            "class failed; **Elevated** means lysogeny was the only recognized required class that failed; "
            "**Unclassified** means a trusted FAIL lacked one of those recognized class assignments.",
            "",
            "## Potential-signal classes",
            "",
            "| Finding class | Removed records | FAIL findings |",
            "| --- | ---: | ---: |",
        ]
    )
    if class_record_counts:
        lines.extend(
            (
                f"| {_markdown_cell(safety_class)} | {class_record_counts[safety_class]} | "
                f"{class_finding_counts.get(safety_class, 0)} |"
            )
            for safety_class in sorted(class_record_counts)
        )
    else:
        lines.append("| None | 0 | 0 |")

    lines.extend(["", "## Highest-priority removed-record highlights", ""])
    if not removed:
        lines.append("No records were removed for potential sequence-safety signals.")
    else:
        lines.extend(
            [
                f"Showing at most {max_highlights} records, ordered by the report-triage categories above.",
                "",
                "| Stable record ID | Potential severity | Finding classes | Reason codes | Tool evidence |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        ordered = sorted(
            removed,
            key=lambda record: (
                _SEVERITY_ORDER[record.severity],
                -len(record.failing_classes),
                -record.finding_count,
                record.record_id,
            ),
        )
        for record in ordered[:max_highlights]:
            classes = ", ".join(record.failing_classes) or "unclassified"
            reasons = ", ".join(record.reason_codes) or "trusted record-level FAIL"
            detectors = ", ".join(record.detectors) or "validated class result"
            evidence = f"{detectors}; {record.finding_count} FAIL finding(s)"
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        record.record_id,
                        record.severity,
                        classes,
                        reasons,
                        evidence,
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Generated only from the validated SFT SAFETY_MANIFEST and its digest-authenticated child scan "
            "manifests. Raw sequence and evidence paths are intentionally omitted.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def write_safety_filter_report(
    safety_manifest: str | Path,
    output: str | Path,
    *,
    safety_manifest_validator: Callable[..., Mapping[str, object]],
    max_highlights: int = 10,
) -> Path:
    """Write a bounded post-SFT report from authenticated safety-filter manifests."""
    if type(max_highlights) is not int or not 1 <= max_highlights <= 100:
        raise sft_module.SFTSafetyError("max_highlights must be an integer from 1 through 100")
    validated = safety_manifest_validator(
        Path(safety_manifest).absolute(),
        require_ready=False,
    )
    manifest = _mapping(validated, label="validated SFT safety manifest")
    source = _mapping(manifest.get("source"), label="validated SFT source")
    curated = _mapping(manifest.get("curated_output"), label="validated curated output")
    total = _count(source.get("record_count"), label="source record_count")
    retained = _count(curated.get("count"), label="curated output count")
    decisions = _list(manifest.get("record_decisions"), label="record_decisions")
    if len(decisions) != total:
        raise sft_module.SFTSafetyError("record decision count does not match the validated source count")

    scan_records: dict[str, Mapping[str, object]] = {}
    children = _list(manifest.get("domain_children"), label="domain_children")
    for index, child_value in enumerate(children):
        child = _mapping(child_value, label=f"domain child {index}")
        snapshot = sft_module._validate_artifact_snapshot(
            child.get("scan_manifest"),
            label=f"domain child {index} scan",
        )
        scan = sft_module._parse_task4_manifest_snapshot(
            snapshot,
            label=f"domain child {index} scan",
        )
        if scan.get("manifest_type") != "sequence_safety_scan":
            raise sft_module.SFTSafetyError("post-SFT report requires completed sequence-safety scan manifests")
        rows = _list(
            scan.get("records"),
            label=f"domain child {index} scan records",
        )
        for row_value in rows:
            row = _mapping(row_value, label=f"domain child {index} scan record")
            scanner_id = row.get("record_id")
            if not isinstance(scanner_id, str) or not scanner_id or scanner_id in scan_records:
                raise sft_module.SFTSafetyError("child scans contain a missing or duplicate record ID")
            scan_records[scanner_id] = row

    removed: list[_SafetyReportRecord] = []
    passed = 0
    unresolved = 0
    other_exclusions = 0
    eligible_count = 0
    for decision_value in decisions:
        decision = _mapping(decision_value, label="record decision")
        if decision.get("eligible_for_sft") is True:
            eligible_count += 1
        state = decision.get("safety_state")
        if state == "PASS":
            passed += 1
        elif state == "INDETERMINATE":
            unresolved += 1
        elif state is None:
            other_exclusions += 1
        elif state != "FAIL":
            raise sft_module.SFTSafetyError("record decision has an invalid safety state")
        if state != "FAIL":
            continue
        scanner_id = decision.get("scanner_record_id")
        if not isinstance(scanner_id, str) or scanner_id not in scan_records:
            raise sft_module.SFTSafetyError("failed SFT decision lacks its authenticated child scan record")
        removed.append(_summarize_failed_record(decision, scan_records[scanner_id]))
    if eligible_count != retained:
        raise sft_module.SFTSafetyError("retained decision count does not match the validated curated output")
    if passed != retained:
        raise sft_module.SFTSafetyError("safety PASS count does not match the validated curated output")
    if passed + len(removed) + unresolved + other_exclusions != total:
        raise sft_module.SFTSafetyError("SFT safety-state counts are inconsistent")

    output_path = Path(output).absolute()
    if output_path.exists() or output_path.is_symlink():
        raise sft_module.SFTSafetyError("SFT safety-filter report already exists")
    published = sft_module._publish_owned_bytes(
        output_path,
        _render(
            total=total,
            passed=passed,
            retained=retained,
            removed=removed,
            unresolved=unresolved,
            other_exclusions=other_exclusions,
            max_highlights=max_highlights,
        ),
        label="SFT safety-filter report",
    )
    os.close(published.parent_descriptor)
    return published.path
