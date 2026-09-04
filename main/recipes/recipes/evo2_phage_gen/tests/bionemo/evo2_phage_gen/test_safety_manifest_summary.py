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

import json
from pathlib import Path

import pytest

from bionemo.evo2_phage_gen import safety_manifest_summary


def _class_result(safety_class: str, state: str) -> dict[str, object]:
    return {
        "safety_class": safety_class,
        "state": state,
        "required": True,
        "reason_codes": [f"{safety_class.upper()}_{state}"],
        "findings": [],
    }


def _record(record_id: str, states: dict[str, str], index: int) -> dict[str, object]:
    class_states = {name: states.get(name, "PASS") for name in ("amr", "toxin", "lysogeny")}
    state = (
        "FAIL"
        if "FAIL" in class_states.values()
        else "INDETERMINATE"
        if "INDETERMINATE" in class_states.values()
        else "PASS"
    )
    return {
        "record_id": record_id,
        "input_index": index,
        "state": state,
        "reason_codes": [f"{name.upper()}_{value}" for name, value in class_states.items()],
        "class_results": [_class_result(name, value) for name, value in class_states.items()],
    }


def _manifest() -> dict[str, object]:
    records = [
        _record("pass", {}, 0),
        _record("toxin-fail", {"toxin": "FAIL"}, 1),
        _record("toxin-lysogeny-fail", {"toxin": "FAIL", "lysogeny": "FAIL"}, 2),
        _record("lysogeny-review", {"lysogeny": "INDETERMINATE"}, 3),
        _record("amr-fail-toxin-review", {"amr": "FAIL", "toxin": "INDETERMINATE"}, 4),
    ]
    return {
        "schema_version": 2,
        "manifest_type": "sequence_safety_scan",
        "records": records,
        "aggregate": {
            "state": "FAIL",
            "counts": {"PASS": 1, "FAIL": 3, "INDETERMINATE": 1},
        },
    }


def test_summarize_scan_manifest_emits_mutually_exclusive_reason_combinations() -> None:
    summary = safety_manifest_summary.summarize_scan_manifest(_manifest())

    assert summary["records"] == {
        "total": 5,
        "pass": 1,
        "fail": 3,
        "indeterminate": 1,
        "excluded": 4,
    }
    assert summary["fail_combinations"] == {
        "amr": 1,
        "toxin": 1,
        "lysogeny": 0,
        "amr+toxin": 0,
        "amr+lysogeny": 0,
        "toxin+lysogeny": 1,
        "amr+toxin+lysogeny": 0,
    }
    assert summary["indeterminate_combinations"]["lysogeny"] == 1
    assert sum(summary["fail_combinations"].values()) == summary["records"]["fail"]
    assert sum(summary["indeterminate_combinations"].values()) == summary["records"]["indeterminate"]
    assert summary["class_state_counts"]["toxin"] == {
        "PASS": 2,
        "FAIL": 2,
        "INDETERMINATE": 1,
    }


def test_summarize_scan_manifest_rejects_aggregate_count_drift() -> None:
    manifest = _manifest()
    manifest["aggregate"]["counts"]["FAIL"] = 2

    with pytest.raises(ValueError, match="aggregate counts"):
        safety_manifest_summary.summarize_scan_manifest(manifest)


def test_main_validates_manifest_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"placeholder": True}), encoding="utf-8")
    output = tmp_path / "summary.json"
    calls: list[Path] = []

    def validate(path: Path):
        calls.append(Path(path))
        return _manifest()

    monkeypatch.setattr(safety_manifest_summary, "validate_manifest_file", validate)

    exit_code = safety_manifest_summary.main(["--manifest", str(manifest_path), "--output", str(output)])

    assert exit_code == 0
    assert calls == [manifest_path]
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["source_manifest"]["path"] == str(manifest_path.absolute())
    assert written["records"]["total"] == 5
