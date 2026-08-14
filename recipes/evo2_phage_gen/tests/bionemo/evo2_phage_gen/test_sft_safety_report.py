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

"""Tests for the post-SFT sequence-safety report renderer."""

import hashlib
import json
from pathlib import Path

from bionemo.evo2_phage_gen import sft_safety_report


def test_write_safety_filter_report_summarizes_signals_without_sequence_details(tmp_path: Path) -> None:
    """The post-SFT report must lead with counts and bound evidence-only highlights."""
    scan_path = tmp_path / "scan" / "manifest.json"
    scan_path.parent.mkdir()
    scan = {
        "manifest_type": "sequence_safety_scan",
        "records": [
            {
                "record_id": "scanner_high",
                "input_index": 0,
                "state": "FAIL",
                "class_results": [
                    {
                        "safety_class": "amr",
                        "state": "FAIL",
                        "required": True,
                        "reason_codes": ["AMR_DETERMINANT_DETECTED"],
                        "findings": [
                            {
                                "state": "FAIL",
                                "reason_codes": ["AMR_DETERMINANT_DETECTED"],
                                "detector": "amrfinder-plus",
                                "source_path": "/private/raw/evidence.tsv",
                            }
                        ],
                    }
                ],
            },
            {
                "record_id": "scanner_elevated",
                "input_index": 1,
                "state": "FAIL",
                "class_results": [
                    {
                        "safety_class": "lysogeny",
                        "state": "FAIL",
                        "required": True,
                        "reason_codes": ["LYSOGENY_HIGH_CONFIDENCE_PROFILE"],
                        "findings": [
                            {
                                "state": "FAIL",
                                "reason_codes": ["LYSOGENY_HIGH_CONFIDENCE_PROFILE"],
                                "detector": "mmseqs-phrogs-v4",
                                "evidence_path": "/private/generated/protein.faa",
                            }
                        ],
                    }
                ],
            },
        ],
    }
    scan_path.write_text(json.dumps(scan), encoding="utf-8")
    validated_manifest = {
        "source": {"record_count": 3},
        "curated_output": {"count": 1},
        "domain_children": [
            {
                "label": "bacteria",
                "scan_manifest": {
                    "path": str(scan_path),
                    "sha256": hashlib.sha256(scan_path.read_bytes()).hexdigest(),
                },
            }
        ],
        "record_decisions": [
            {
                "record_id": "record_high",
                "scanner_record_id": "scanner_high",
                "safety_state": "FAIL",
                "eligible_for_sft": False,
            },
            {
                "record_id": "record_elevated",
                "scanner_record_id": "scanner_elevated",
                "safety_state": "FAIL",
                "eligible_for_sft": False,
            },
            {
                "record_id": "record_pass",
                "scanner_record_id": "scanner_pass",
                "safety_state": "PASS",
                "eligible_for_sft": True,
            },
        ],
    }
    report_path = tmp_path / "artifacts" / "SAFETY_FILTER_REPORT.md"

    result = sft_safety_report.write_safety_filter_report(
        tmp_path / "SAFETY_MANIFEST.yaml",
        report_path,
        safety_manifest_validator=lambda _path, **_kwargs: validated_manifest,
        max_highlights=1,
    )

    report = result.read_text(encoding="utf-8")
    assert report.startswith("# SFT sequence-safety filter report\n\n## High-level outcome")
    assert "| Total source records examined | 3 |" in report
    assert "| Retained for SFT | 1 |" in report
    assert "| Removed after potential sequence-safety signals | 2 |" in report
    assert "## Safety-filter waterfall" in report
    assert "| 3 | Starting set of phage genomes |" in report
    assert "| 1 | No observed or known safety signals from configured filters |" in report
    assert "| 2 | Phage genomes failing the sequence-safety filter |" in report
    assert "| 1 | &emsp;↳ Lysogeny only |" in report
    assert "| 1 | &emsp;↳ Antibiotic resistance only |" in report
    assert "| 0 | &emsp;↳ Toxin only |" in report
    assert "| 0 | &emsp;↳ Toxin + lysogeny |" in report
    assert "| 0 | &emsp;↳ Antibiotic resistance + lysogeny |" in report
    assert "| 0 | &emsp;↳ Antibiotic resistance + toxin |" in report
    assert "| 0 | &emsp;↳ Antibiotic resistance + toxin + lysogeny |" in report
    assert "| Higher | 1 |" in report
    assert "| Elevated | 1 |" in report
    assert "record_high" in report
    assert "record_elevated" not in report
    assert "AMR_DETERMINANT_DETECTED" in report
    assert "amrfinder-plus" in report
    assert "/private/" not in report
    assert "not a clinical safety conclusion" in report
