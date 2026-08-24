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
from types import SimpleNamespace

import pytest

from bionemo.evo2_phage_gen import sequence_safety_cli as cli
from bionemo.evo2_phage_gen.design_scope import HostDomain
from bionemo.evo2_phage_gen.sequence_safety import SafetyClassResult, SafetyState
from bionemo.evo2_phage_gen.sequence_safety_adapters import AdapterResult


def _adapter(name: str, state: SafetyState = SafetyState.PASS) -> AdapterResult:
    return AdapterResult(
        SafetyClassResult(name, state, True, reason_codes=(f"{name.upper()}_MEASURED",)),
        "COMPLETED_AND_PARSED",
        (name, "scan"),
    )


def _policy(path: Path) -> None:
    path.write_text(
        """
policy_id: test-policy
required_sequence_classes: [amr, toxin]
bacterial_replication_profile:
  required_sequence_classes: [amr, toxin, lysogeny]
failure_policy:
  missing_tool: INDETERMINATE
"""
    )


def _host() -> str:
    return json.dumps(
        {
            "source": "lab notebook",
            "source_version": "2026-08-17",
            "replication_host_domains": ["BACTERIA"],
            "confirmed": True,
            "metadata": {},
        }
    )


def test_fasta_parser_normalizes_sequences_and_preserves_order(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">a note\natg n\n>b\nCCCC\n")
    records = cli.parse_fasta_records(fasta)
    assert [(record.sequence_id, record.sequence) for record in records] == [
        ("a", "ATGN"),
        ("b", "CCCC"),
    ]


def test_fasta_parser_rejects_duplicate_ids(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">a\nATG\n>a other\nCCC\n")
    with pytest.raises(cli.CLIValidationError):
        cli.parse_fasta_records(fasta)


def test_missing_detector_result_is_indeterminate() -> None:
    result = cli.aggregate_adapter_results(
        {"amr": _adapter("amr"), "toxin": _adapter("toxin")},
        host_domain=HostDomain.BACTERIA,
        strict_lysis=True,
    )
    assert result.state is SafetyState.INDETERMINATE
    assert result.class_results[-1].reason_codes == ("ADAPTER_RESULT_MISSING",)


def test_failure_dominates_and_all_classes_remain_visible() -> None:
    result = cli.aggregate_adapter_results(
        {
            "amr": _adapter("amr"),
            "toxin": _adapter("toxin", SafetyState.FAIL),
            "lysogeny": _adapter("lysogeny"),
        },
        host_domain=HostDomain.BACTERIA,
    )
    assert result.state is SafetyState.FAIL
    assert [item.safety_class for item in result.class_results] == list(cli.SAFETY_CLASSES)


def test_scope_rejects_eukaryotic_replication() -> None:
    result = cli.validate_design_scope_payload(
        objective={
            "kind": "productive_replication",
            "direction": "increase",
            "replication_host_domains": ["EUKARYOTA"],
            "endpoint": "increased_eukaryotic_replication",
        },
        host_evidence={
            "source": "catalog",
            "source_version": "v1",
            "replication_host_domains": ["EUKARYOTA"],
            "confirmed": True,
            "metadata": {},
        },
    )
    assert result["allowed"] is False


def test_scan_runs_every_detector_and_logs_observed_state(tmp_path: Path, monkeypatch) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">phix\n" + "ATG" * 30 + "\n")
    policy = tmp_path / "policy.yaml"
    _policy(policy)
    assets = tmp_path / "assets.yaml"
    assets.write_text("schema_version: 1\ntools: {}\ndatabases: {}\n")
    output = tmp_path / "scan"
    calls: list[str] = []

    monkeypatch.setattr(cli, "_load_asset_state", lambda _path: {})
    monkeypatch.setattr(
        cli,
        "_runtime_state",
        lambda _args, _assets: (
            {
                "selected": {"amrfinder": object(), "diamond": object(), "mmseqs": object()},
                "observed": {
                    "amrfinder": {"path": "/tools/amrfinder", "version": "4.2.7"},
                    "diamond": {"path": "/tools/diamond", "version": "2.1.24"},
                    "mmseqs": {"path": "/tools/mmseqs", "version": "18"},
                },
            },
            {
                "amrfinder": {"path": "/db/amr", "version": "2026-08"},
                "toxins": {"path": "/db/toxins", "version": "UniProt 2026_03"},
                "phrogs": {"path": "/db/phrogs", "lookup": "/db/lookup.tsv", "version": "v4"},
            },
        ),
    )
    monkeypatch.setattr(cli, "prepare_orf_artifacts", lambda *_args, **_kwargs: SimpleNamespace())

    def detector(name: str):
        def run(genomes, _artifacts, **_kwargs):
            calls.append(name)
            return {genome.sequence_id: _adapter(name) for genome in genomes}

        return run

    monkeypatch.setattr(cli, "run_amrfinder_batch", detector("amr"))
    monkeypatch.setattr(cli, "run_toxin_batch", detector("toxin"))
    monkeypatch.setattr(cli, "run_phrogs_batch", detector("lysogeny"))

    exit_code = cli.main(
        [
            "scan",
            "--input-fasta",
            str(fasta),
            "--output-dir",
            str(output),
            "--policy",
            str(policy),
            "--asset-manifest",
            str(assets),
            "--host-domain",
            "BACTERIA",
            "--host-evidence-json",
            _host(),
            "--strict-lysis",
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text())

    assert exit_code == 0
    assert calls == ["amr", "toxin", "lysogeny"]
    assert manifest["aggregate"]["counts"] == {"FAIL": 0, "INDETERMINATE": 0, "PASS": 1}
    assert manifest["tools"]["diamond"]["version"] == "2.1.24"
    assert (output / "RUNLOG.md").is_file()


def test_scan_batches_records_with_bounded_workers(tmp_path: Path, monkeypatch, capsys) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text("".join(f">g{index}\n" + "ATG" * 30 + "\n" for index in range(5)))
    policy = tmp_path / "policy.yaml"
    _policy(policy)
    assets = tmp_path / "assets.yaml"
    assets.write_text("schema_version: 1\ntools: {}\ndatabases: {}\n")
    output = tmp_path / "scan"
    prepared: list[tuple[list[str], int]] = []
    detector_calls: list[tuple[str, list[str], int]] = []

    monkeypatch.setattr(cli, "_load_asset_state", lambda _path: {})
    monkeypatch.setattr(
        cli,
        "_runtime_state",
        lambda _args, _assets: (
            {
                "selected": {"amrfinder": object(), "diamond": object(), "mmseqs": object()},
                "observed": {},
            },
            {
                "amrfinder": {"path": "x", "version": "x"},
                "toxins": {"path": "x", "version": "x"},
                "phrogs": {"path": "x", "lookup": "x", "version": "x"},
            },
        ),
    )

    def prepare(genomes, _work_dir, *, workers=1, **_kwargs):
        prepared.append(([genome.sequence_id for genome in genomes], workers))
        return SimpleNamespace()

    def detector(name: str):
        def scan(genomes, _artifacts, *, threads, **_kwargs):
            detector_calls.append((name, [genome.sequence_id for genome in genomes], threads))
            return {genome.sequence_id: _adapter(name) for genome in genomes}

        return scan

    monkeypatch.setattr(cli, "prepare_orf_artifacts", prepare)
    monkeypatch.setattr(cli, "run_amrfinder_batch", detector("amr"))
    monkeypatch.setattr(cli, "run_toxin_batch", detector("toxin"))
    monkeypatch.setattr(cli, "run_phrogs_batch", detector("lysogeny"))

    exit_code = cli.main(
        [
            "scan",
            "--input-fasta",
            str(fasta),
            "--output-dir",
            str(output),
            "--policy",
            str(policy),
            "--asset-manifest",
            str(assets),
            "--host-domain",
            "BACTERIA",
            "--host-evidence-json",
            _host(),
            "--strict-lysis",
            "--batch-size",
            "2",
            "--orf-workers",
            "4",
            "--threads",
            "7",
            "--phrogs-threads",
            "11",
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    progress = capsys.readouterr().out

    assert exit_code == 0
    assert prepared == [(["g0", "g1"], 4), (["g2", "g3"], 4), (["g4"], 4)]
    assert detector_calls == [
        (name, records, 11 if name == "lysogeny" else 7)
        for records in (["g0", "g1"], ["g2", "g3"], ["g4"])
        for name in ("amr", "toxin", "lysogeny")
    ]
    assert [record["record_id"] for record in manifest["records"]] == [f"g{index}" for index in range(5)]
    assert manifest["execution"] == {
        "batch_count": 3,
        "batch_size": 2,
        "orf_workers": 4,
        "phrogs_threads": 11,
        "threads": 7,
    }
    assert "safety scan batch 1/3: predicting ORFs for 2 records" in progress
    assert "safety scan batch 1/3: AMR screen" in progress
    assert "safety scan batch 1/3: toxin screen" in progress
    assert "safety scan batch 1/3: lysogeny screen" in progress
    assert "safety scan batch 1/3: complete in " in progress


def test_scan_marks_every_class_indeterminate_when_orf_prediction_fails(tmp_path: Path, monkeypatch) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">tiny\nATG\n")
    policy = tmp_path / "policy.yaml"
    _policy(policy)
    assets = tmp_path / "assets.yaml"
    assets.write_text("schema_version: 1\ntools: {}\ndatabases: {}\n")
    output = tmp_path / "scan"

    monkeypatch.setattr(cli, "_load_asset_state", lambda _path: {})
    monkeypatch.setattr(
        cli,
        "_runtime_state",
        lambda _args, _assets: (
            {"selected": {"amrfinder": object(), "diamond": object(), "mmseqs": object()}, "observed": {}},
            {
                "amrfinder": {"path": "x", "version": "x"},
                "toxins": {"path": "x", "version": "x"},
                "phrogs": {"path": "x", "lookup": "x", "version": "x"},
            },
        ),
    )
    monkeypatch.setattr(
        cli,
        "prepare_orf_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no genes")),
    )

    exit_code = cli.main(
        [
            "scan",
            "--input-fasta",
            str(fasta),
            "--output-dir",
            str(output),
            "--policy",
            str(policy),
            "--asset-manifest",
            str(assets),
            "--host-domain",
            "BACTERIA",
            "--host-evidence-json",
            _host(),
            "--strict-lysis",
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert exit_code == 3
    assert [result["state"] for result in manifest["records"][0]["class_results"]] == [
        "INDETERMINATE",
        "INDETERMINATE",
        "INDETERMINATE",
    ]


def test_filter_partitions_records_without_reordering(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">a\n+!ATG\n>b\n+$CCC\n")
    scan = tmp_path / "scan.json"
    scan.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_type": "sequence_safety_scan",
                "records": [
                    {
                        "input_index": 0,
                        "record_id": "a",
                        "state": "PASS",
                        "class_results": [
                            {"safety_class": name, "state": "PASS", "required": True} for name in cli.SAFETY_CLASSES
                        ],
                        "adapter_attempts": [],
                    },
                    {
                        "input_index": 1,
                        "record_id": "b",
                        "state": "FAIL",
                        "class_results": [
                            {
                                "safety_class": name,
                                "state": "FAIL" if name == "toxin" else "PASS",
                                "required": True,
                            }
                            for name in cli.SAFETY_CLASSES
                        ],
                        "adapter_attempts": [],
                    },
                ],
                "aggregate": {"state": "FAIL", "counts": {"PASS": 1, "FAIL": 1, "INDETERMINATE": 0}},
            }
        )
    )
    output = tmp_path / "filtered"
    assert (
        cli.main(
            [
                "filter-fasta",
                "--input-fasta",
                str(fasta),
                "--scan-manifest",
                str(scan),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert (output / "pass.fasta").read_text() == ">a\n+!ATG\n"
    assert (output / "fail.fasta").read_text() == ">b\n+$CCC\n"


def test_cli_error_is_distinct_from_indeterminate(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    assert cli.main(["validate-manifest", "--manifest", str(missing)]) == 4


def test_detector_execution_tolerates_only_unscorable_invalid_candidates() -> None:
    manifest = {
        "records": [
            {
                "record_id": "no-primary-genes",
                "state": "INDETERMINATE",
                "class_results": [
                    {
                        "safety_class": "amr",
                        "state": "PASS",
                        "required": True,
                        "findings": [],
                        "reason_codes": ["AMRFINDER_MEASURED_NO_AMR_HIT"],
                    },
                    {
                        "safety_class": "toxin",
                        "state": "PASS",
                        "required": True,
                        "findings": [],
                        "reason_codes": ["TOXIN_MEASURED_NO_REVIEW_HIT"],
                    },
                    {
                        "safety_class": "lysogeny",
                        "state": "INDETERMINATE",
                        "required": True,
                        "findings": [],
                        "reason_codes": ["PHROGS_NO_PREDICTED_GENES"],
                    },
                ],
                "adapter_attempts": [
                    {"safety_class": "amr", "execution_status": "COMPLETED_AND_PARSED"},
                    {"safety_class": "toxin", "execution_status": "COMPLETED_AND_PARSED"},
                    {"safety_class": "lysogeny", "execution_status": "NOT_RUN"},
                ],
            }
        ]
    }

    with pytest.raises(cli.CLIValidationError, match="incomplete safety detector execution"):
        cli.validate_detector_execution(manifest)

    assert cli.validate_detector_execution(manifest, allow_no_primary_gene_candidates=True) == (
        ("no-primary-genes", "lysogeny", "PHROGS_NO_PREDICTED_GENES"),
    )


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("FAILED", "PHROGS_EXECUTION_TIMEOUT"),
        ("PARSER_ERROR", "PHROGS_PARSER_ERROR"),
        ("NOT_RUN", "PHROGS_DATABASE_UNAVAILABLE"),
    ],
)
def test_detector_execution_never_tolerates_detector_failures(status: str, reason: str) -> None:
    manifest = {
        "records": [
            {
                "record_id": "candidate",
                "state": "INDETERMINATE",
                "class_results": [
                    {
                        "safety_class": "lysogeny",
                        "state": "INDETERMINATE",
                        "required": True,
                        "findings": [],
                        "reason_codes": [reason],
                    }
                ],
                "adapter_attempts": [{"safety_class": "lysogeny", "execution_status": status}],
            }
        ]
    }

    with pytest.raises(cli.CLIValidationError, match="incomplete safety detector execution"):
        cli.validate_detector_execution(manifest, allow_no_primary_gene_candidates=True)
