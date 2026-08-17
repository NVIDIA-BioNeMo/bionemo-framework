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

"""Validate and summarize a sequence-safety scan manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from bionemo.evo2_phage_gen.sequence_safety_cli import validate_manifest_file


SAFETY_CLASSES = ("amr", "toxin", "lysogeny")
STATES = ("PASS", "FAIL", "INDETERMINATE")
REASON_COMBINATIONS = (
    "amr",
    "toxin",
    "lysogeny",
    "amr+toxin",
    "amr+lysogeny",
    "toxin+lysogeny",
    "amr+toxin+lysogeny",
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _state(value: object, *, label: str) -> str:
    if value not in STATES:
        raise ValueError(f"{label} has an unsupported state")
    return str(value)


def _aggregate_state(states: Sequence[str]) -> str:
    if "FAIL" in states:
        return "FAIL"
    if "INDETERMINATE" in states:
        return "INDETERMINATE"
    return "PASS"


def _reason_combination(class_states: Mapping[str, str], target: str) -> str:
    combination = "+".join(safety_class for safety_class in SAFETY_CLASSES if class_states[safety_class] == target)
    if combination not in REASON_COMBINATIONS:
        raise ValueError(f"{target} record lacks a recognized class combination")
    return combination


def summarize_scan_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    """Return mutually exclusive record and safety-class counts from a validated scan."""
    if manifest.get("schema_version") != 2 or manifest.get("manifest_type") != "sequence_safety_scan":
        raise ValueError("expected a sequence-safety scan manifest with schema version 2")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("scan manifest records must be a non-empty list")

    record_counts = Counter(dict.fromkeys(STATES, 0))
    class_state_counts = {safety_class: dict.fromkeys(STATES, 0) for safety_class in SAFETY_CLASSES}
    fail_combinations = dict.fromkeys(REASON_COMBINATIONS, 0)
    indeterminate_combinations = dict.fromkeys(REASON_COMBINATIONS, 0)
    reason_code_counts: Counter[str] = Counter()
    seen_record_ids: set[str] = set()

    for expected_index, value in enumerate(records):
        record = _mapping(value, label=f"record {expected_index}")
        record_id = record.get("record_id")
        if (
            record.get("input_index") != expected_index
            or not isinstance(record_id, str)
            or not record_id
            or record_id in seen_record_ids
        ):
            raise ValueError("scan records must preserve unique IDs and input order")
        seen_record_ids.add(record_id)

        class_results = record.get("class_results")
        if not isinstance(class_results, list):
            raise ValueError(f"record {record_id} lacks class results")
        class_states: dict[str, str] = {}
        for result_value in class_results:
            result = _mapping(result_value, label=f"record {record_id} class result")
            safety_class = result.get("safety_class")
            if (
                safety_class not in SAFETY_CLASSES
                or safety_class in class_states
                or result.get("required") is not True
            ):
                raise ValueError(f"record {record_id} must contain each required safety class once")
            state = _state(result.get("state"), label=f"record {record_id} class {safety_class}")
            class_states[str(safety_class)] = state
            class_state_counts[str(safety_class)][state] += 1
            reason_codes = result.get("reason_codes")
            if not isinstance(reason_codes, list) or any(
                not isinstance(reason, str) or not reason for reason in reason_codes
            ):
                raise ValueError(f"record {record_id} has invalid class reason codes")
            reason_code_counts.update(reason_codes)
        if tuple(class_states) != SAFETY_CLASSES:
            raise ValueError(f"record {record_id} must preserve the canonical safety-class order")

        state = _state(record.get("state"), label=f"record {record_id}")
        if state != _aggregate_state(tuple(class_states.values())):
            raise ValueError(f"record {record_id} state does not reconcile with its class results")
        record_counts[state] += 1
        if state == "FAIL":
            fail_combinations[_reason_combination(class_states, state)] += 1
        elif state == "INDETERMINATE":
            indeterminate_combinations[_reason_combination(class_states, state)] += 1

    aggregate = _mapping(manifest.get("aggregate"), label="aggregate")
    aggregate_counts = _mapping(aggregate.get("counts"), label="aggregate counts")
    expected_counts = {state: record_counts[state] for state in STATES}
    if dict(aggregate_counts) != expected_counts:
        raise ValueError("aggregate counts do not reconcile with scan records")
    if aggregate.get("state") != _aggregate_state(tuple(state for state, count in expected_counts.items() if count)):
        raise ValueError("aggregate state does not reconcile with scan records")
    if sum(fail_combinations.values()) != record_counts["FAIL"]:
        raise ValueError("FAIL combinations do not partition failed records")
    if sum(indeterminate_combinations.values()) != record_counts["INDETERMINATE"]:
        raise ValueError("INDETERMINATE combinations do not partition review-required records")

    total = len(records)
    return {
        "schema_version": 1,
        "summary_type": "sequence_safety_manifest_tally",
        "records": {
            "total": total,
            "pass": record_counts["PASS"],
            "fail": record_counts["FAIL"],
            "indeterminate": record_counts["INDETERMINATE"],
            "excluded": record_counts["FAIL"] + record_counts["INDETERMINATE"],
        },
        "fail_combinations": fail_combinations,
        "indeterminate_combinations": indeterminate_combinations,
        "class_state_counts": class_state_counts,
        "reason_code_counts": dict(sorted(reason_code_counts.items())),
        "claim_boundary": (
            "Computational sequence-safety screening signals from the validated manifest; "
            "not clinical, therapeutic, or organism-level safety conclusions."
        ),
    }


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    """Build the safety-manifest summary command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the safety-manifest summary command."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        manifest_path = args.manifest.absolute()
        manifest = validate_manifest_file(args.manifest)
        summary = summarize_scan_manifest(manifest)
        summary["source_manifest"] = {
            "path": str(manifest_path),
            "schema_version": manifest["schema_version"],
            "manifest_type": manifest["manifest_type"],
        }
        serialized = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            print(serialized, end="")
        else:
            _write_json_atomic(args.output, summary)
        return 0
    except (OSError, TypeError, ValueError) as error:
        parser._print_message(f"{parser.prog}: error: {error}\n", sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
