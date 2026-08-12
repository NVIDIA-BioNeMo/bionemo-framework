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

"""Behavioral tests for the strict sequence-safety CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bionemo.evo2_phage_gen.design_scope import HostDomain, ObjectiveKind
from bionemo.evo2_phage_gen.sequence_safety import SafetyClassResult, SafetyFinding, SafetyState
from bionemo.evo2_phage_gen.sequence_safety_adapters import AdapterResult


_ADAPTER_POLICIES = {
    "amr": (
        "amrfinder-curated-thresholds-v4.2.7",
        "871fdcba5b14c69bb159be508da014eea62f1e1f9dbd395f1c8a31d8797a790b",
    ),
    "toxin": (
        "toxin-homology-v2",
        "faa488a383c36b28695a0e590dade72699f5e84f3c03e7563a2ce470a988de3e",
    ),
    "lysogeny": (
        "phrogs-homology-v1",
        "de323894ac6dab6b295ed3a8f27a8e038b0f88b642e10a1e6c8f8ee491cc7f1f",
    ),
}


def _write_policy(path: Path) -> Path:
    path.write_text(
        """\
schema_version: 1
policy_id: phage-sequence-safety-v1
regulatory_basis:
  label: EMA-derived sequence-design safety gate
  source: EMA/CHMP/BWP/1/2024
  source_status: draft
  source_status_as_of: 2026-08-07
  regulatory_compliance_claimed: false
host_scope:
  allowed_replication_host_domains: [BACTERIA, ARCHAEA, BACTERIA_AND_ARCHAEA]
  disallowed_endpoint: increased_eukaryotic_replication
required_sequence_classes: [amr, toxin]
bacterial_replication_profile:
  required_sequence_classes: [amr, toxin, lysogeny]
  strict_lytic_required: true
archaeal_only_profile:
  required_sequence_classes: [amr, toxin]
  lysogeny: informational
failure_policy:
  missing_required_tool: INDETERMINATE
  missing_required_database: INDETERMINATE
  parser_schema_mismatch: INDETERMINATE
  incomplete_host_evidence: INDETERMINATE
"""
    )
    return path


def _class_adapter(
    safety_class: str,
    *,
    supplemental: tuple[SafetyFinding, ...] = (),
) -> AdapterResult:
    policy_id, policy_sha256 = _ADAPTER_POLICIES[safety_class]
    return AdapterResult(
        class_result=SafetyClassResult(
            safety_class=safety_class,
            state=SafetyState.PASS,
            required=True,
            reason_codes=("PINNED_SEARCH_COMPLETED_NO_QUALIFYING_HIT",),
        ),
        supplemental_findings=supplemental,
        command=("trusted-tool", "search"),
        raw_output_path=f"/{safety_class}.tsv",
        raw_output_sha256="1" * 64,
        policy_id=policy_id,
        policy_sha256=policy_sha256,
    )


def test_cli_module_exposes_an_argv_entrypoint():
    """Removing the argv-aware CLI entrypoint must break a real help invocation."""
    from bionemo.evo2_phage_gen import sequence_safety_cli

    with pytest.raises(SystemExit) as stopped:
        sequence_safety_cli.main(["--help"])

    assert stopped.value.code == 0


def test_recipe_registers_sequence_safety_console_entry_point():
    recipe_root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((recipe_root / "pyproject.toml").read_text())

    assert (
        project["project"]["scripts"]["evo2_phage_sequence_safety"]
        == "bionemo.evo2_phage_gen.sequence_safety_cli:main"
    )


def test_scope_payload_coerces_explicit_enums_before_typed_validation():
    """Bypassing enum coercion must not let raw JSON strings masquerade as typed scope evidence."""
    from bionemo.evo2_phage_gen.sequence_safety_cli import validate_design_scope_payload

    result = validate_design_scope_payload(
        objective={
            "kind": "persistence",
            "direction": "increase",
            "replication_host_domains": ["BACTERIA"],
            "endpoint": "circulation_half_life",
        },
        host_evidence={
            "source": "curated-host-catalog",
            "source_version": "2026-08-08",
            "replication_host_domains": ["BACTERIA"],
            "confirmed": True,
            "metadata": {},
        },
    )

    assert result.allowed
    assert result.objective.kind is ObjectiveKind.PERSISTENCE
    assert result.host_evidence.replication_host_domains == frozenset({HostDomain.BACTERIA})


def test_amrfinder_supplemental_virulence_prevents_toxin_pass():
    """Dropping AMRFinderPlus virulence evidence must make an unsafe aggregate test fail."""
    from bionemo.evo2_phage_gen.sequence_safety_cli import aggregate_adapter_results

    virulence = SafetyFinding(
        safety_class="toxin",
        state=SafetyState.INDETERMINATE,
        reason_codes=("AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL",),
        finding_id="amrfinder-plus-virulence-1",
    )
    result = aggregate_adapter_results(
        {
            "amr": _class_adapter("amr", supplemental=(virulence,)),
            "toxin": _class_adapter("toxin"),
            "lysogeny": _class_adapter("lysogeny"),
        },
        host_domain=HostDomain.BACTERIA,
    )

    toxin = next(item for item in result.class_results if item.safety_class == "toxin")
    assert toxin.state is SafetyState.INDETERMINATE
    assert toxin.findings == (virulence,)
    assert "AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL" in toxin.reason_codes
    assert result.state is SafetyState.INDETERMINATE


def test_aggregation_uses_fail_precedence_and_ignores_informational_archaeal_lysis():
    from bionemo.evo2_phage_gen.sequence_safety_cli import aggregate_adapter_results

    failed_amr = AdapterResult(
        class_result=SafetyClassResult(
            safety_class="amr",
            state=SafetyState.FAIL,
            required=True,
            reason_codes=("AMR_HIT",),
        ),
        policy_id=_ADAPTER_POLICIES["amr"][0],
        policy_sha256=_ADAPTER_POLICIES["amr"][1],
    )
    uncertain_toxin = AdapterResult(
        class_result=SafetyClassResult(
            safety_class="toxin",
            state=SafetyState.INDETERMINATE,
            required=True,
            reason_codes=("TOXIN_REVIEW_HIT",),
        ),
        policy_id=_ADAPTER_POLICIES["toxin"][0],
        policy_sha256=_ADAPTER_POLICIES["toxin"][1],
    )
    lysogeny_fail = AdapterResult(
        class_result=SafetyClassResult(
            safety_class="lysogeny",
            state=SafetyState.FAIL,
            required=True,
            reason_codes=("LYSOGENY_HIT",),
        ),
        policy_id=_ADAPTER_POLICIES["lysogeny"][0],
        policy_sha256=_ADAPTER_POLICIES["lysogeny"][1],
    )

    bacterial = aggregate_adapter_results(
        {"amr": failed_amr, "toxin": uncertain_toxin, "lysogeny": _class_adapter("lysogeny")},
        host_domain=HostDomain.BACTERIA,
    )
    archaeal = aggregate_adapter_results(
        {"amr": _class_adapter("amr"), "toxin": _class_adapter("toxin"), "lysogeny": lysogeny_fail},
        host_domain=HostDomain.ARCHAEA,
    )

    assert bacterial.state is SafetyState.FAIL
    assert archaeal.state is SafetyState.PASS
    archaeal_lysis = next(item for item in archaeal.class_results if item.safety_class == "lysogeny")
    assert archaeal_lysis.required is False
    assert archaeal_lysis.state is SafetyState.FAIL


def test_normalized_amrfinder_virulence_remains_bound_to_toxin_manifest_evidence(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_adapters import NormalizedSafetyFinding
    from bionemo.evo2_phage_gen.sequence_safety_cli import _validate_class_results, aggregate_adapter_results

    virulence = NormalizedSafetyFinding(
        safety_class="toxin",
        state=SafetyState.INDETERMINATE,
        reason_codes=("AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL",),
        finding_id="virulence-1",
        detector="amrfinderplus",
        accession="VIR-1",
        query_id="phage-a_orf_1",
        sequence_id="phage-a",
        start=1,
        end=9,
        strand="+",
        frame=1,
        scores={"identity": 90.0},
        thresholds={},
        source_path=str(tmp_path / "amr-db"),
        source_sha256="1" * 64,
        tool_version="4.2.7",
        database_version="2026-08-08",
        evidence_path="primary_orf",
        evidence_method="amrfinderplus_combined",
        threshold_policy=_ADAPTER_POLICIES["amr"][0],
        threshold_policy_sha256=_ADAPTER_POLICIES["amr"][1],
        tool_path=str(tmp_path / "amrfinder"),
        tool_sha256="2" * 64,
    )
    aggregate = aggregate_adapter_results(
        {
            "amr": _class_adapter("amr", supplemental=(virulence,)),
            "toxin": _class_adapter("toxin"),
            "lysogeny": _class_adapter("lysogeny"),
        },
        host_domain=HostDomain.BACTERIA,
    )

    restored, finding_ids = _validate_class_results(
        [result.to_dict() for result in aggregate.class_results],
        record_id="phage-a",
        applicability={"amr": True, "toxin": True, "lysogeny": True},
    )

    toxin = next(result for result in restored if result.safety_class == "toxin")
    assert toxin.state is SafetyState.INDETERMINATE
    assert toxin.findings == (virulence,)
    assert finding_ids == {"virulence-1"}


def test_trust_boundary_rejects_unknown_adapter_classes(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import CLIValidationError, _trusted_adapter_bundle

    with pytest.raises(CLIValidationError, match="unknown adapter"):
        _trusted_adapter_bundle(
            {
                "amr": _class_adapter("amr"),
                "toxin": _class_adapter("toxin"),
                "lysogeny": _class_adapter("lysogeny"),
                "unreviewed": _class_adapter("amr"),
            },
            record_root=tmp_path,
        )


def test_default_scan_without_trusted_homology_tool_pins_exits_three(tmp_path: Path):
    """Default scanning must not certify a record when DIAMOND or MMseqs identity is untrusted."""
    from bionemo.evo2_phage_gen.sequence_safety_cli import main

    input_fasta = tmp_path / "input.fna"
    input_fasta.write_bytes(b">phage-1\nATGAAATAG\n")
    policy_path = _write_policy(tmp_path / "policy.yaml")
    asset_manifest = tmp_path / "assets.json"
    asset_manifest.write_text("{}\n")
    output_dir = tmp_path / "scan"
    host_evidence = {
        "source": "curated-host-catalog",
        "source_version": "2026-08-08",
        "replication_host_domains": ["BACTERIA"],
        "confirmed": True,
        "metadata": {},
    }

    exit_code = main(
        [
            "scan",
            "--input-fasta",
            str(input_fasta),
            "--output-dir",
            str(output_dir),
            "--policy",
            str(policy_path),
            "--asset-manifest",
            str(asset_manifest),
            "--host-domain",
            "BACTERIA",
            "--host-evidence-json",
            json.dumps(host_evidence),
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert exit_code == 3
    assert manifest["manifest_type"] == "sequence_safety_diagnostic"
    assert manifest["aggregate"]["state"] == "INDETERMINATE"
    assert {
        "MISSING_TRUSTED_DIAMOND_TOOL_PIN",
        "MISSING_TRUSTED_MMSEQS_TOOL_PIN",
    }.issubset(manifest["records"][0]["reason_codes"])
    assert main(["validate-manifest", "--manifest", str(output_dir / "manifest.json")]) == 3

    filter_dir = tmp_path / "filter"
    assert (
        main(
            [
                "filter-fasta",
                "--input-fasta",
                str(input_fasta),
                "--scan-manifest",
                str(output_dir / "manifest.json"),
                "--output-dir",
                str(filter_dir),
            ]
        )
        == 3
    )
    assert not filter_dir.exists()


@pytest.mark.parametrize(
    "payload",
    [
        (
            '{"objective":{"kind":"persistence","direction":"increase",'
            '"replication_host_domains":["BACTERIA"],"endpoint":"circulation_half_life"},'
            '"host_evidence":{"source":"catalog","source":"shadowed","source_version":"v1",'
            '"replication_host_domains":["BACTERIA"],"confirmed":true,"metadata":{}}}'
        ),
        (
            '{"objective":{"kind":"persistence","direction":"increase",'
            '"replication_host_domains":["BACTERIA"],"endpoint":"circulation_half_life"},'
            '"host_evidence":{"source":"catalog","source_version":"v1",'
            '"replication_host_domains":["BACTERIA"],"confirmed":true,"metadata":{"score":NaN}}}'
        ),
    ],
)
def test_scope_cli_rejects_duplicate_keys_and_nonfinite_metadata(tmp_path: Path, payload: str):
    from bionemo.evo2_phage_gen.sequence_safety_cli import main

    request = tmp_path / "scope.json"
    output = tmp_path / "result.json"
    request.write_text(payload)

    assert main(["validate-design-scope", "--input", str(request), "--output", str(output)]) == 3
    assert not output.exists()


def test_scan_rejects_duplicate_policy_yaml_before_writing_a_diagnostic(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import main

    input_fasta = tmp_path / "input.fna"
    input_fasta.write_bytes(b">phage-1\nATGAAATAG\n")
    policy = _write_policy(tmp_path / "policy.yaml")
    policy.write_text(policy.read_text() + "\npolicy_id: shadowed-policy\n")
    output_dir = tmp_path / "scan"

    assert (
        main(
            [
                "scan",
                "--input-fasta",
                str(input_fasta),
                "--output-dir",
                str(output_dir),
                "--policy",
                str(policy),
                "--asset-manifest",
                str(tmp_path / "unused-assets.yaml"),
                "--host-domain",
                "BACTERIA",
                "--host-evidence-json",
                json.dumps(
                    {
                        "source": "catalog",
                        "source_version": "v1",
                        "replication_host_domains": ["BACTERIA"],
                        "confirmed": True,
                        "metadata": {},
                    }
                ),
            ]
        )
        == 3
    )
    assert not output_dir.exists()


def test_adapter_downgrade_preserves_supplemental_virulence_evidence(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_adapters import NormalizedSafetyFinding
    from bionemo.evo2_phage_gen.sequence_safety_cli import _trusted_adapter_bundle

    supplemental = NormalizedSafetyFinding(
        safety_class="toxin",
        state=SafetyState.INDETERMINATE,
        reason_codes=("AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL",),
        finding_id="toxin:q1:VIR-1",
        detector="amrfinder-plus",
        accession="VIR-1",
        query_id="q1",
        sequence_id="phage-1",
        start=1,
        end=9,
        strand="+",
        frame=1,
        scores={"identity": 90.0},
        thresholds={},
        source_path=str(tmp_path / "amr-db"),
        source_sha256="1" * 64,
        tool_version="4.2.7",
        database_version="db-v1",
        evidence_path="pyrodigal-gv",
        evidence_method="combined",
        threshold_policy=_ADAPTER_POLICIES["amr"][0],
        threshold_policy_sha256=_ADAPTER_POLICIES["amr"][1],
        tool_path=str(tmp_path / "amrfinder"),
        tool_sha256="2" * 64,
    )
    amr = replace(_class_adapter("amr", supplemental=(supplemental,)), command=())

    trusted = _trusted_adapter_bundle({"amr": amr}, record_root=tmp_path)

    assert trusted["amr"].class_result.state is SafetyState.INDETERMINATE
    assert trusted["amr"].supplemental_findings == (supplemental,)


def test_adapter_trust_downgrade_is_a_failed_not_replayable_attempt(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import _adapter_execution_status, _trusted_adapter_bundle

    normalized = tmp_path / "amrfinder.tsv"
    normalized.write_text("header\n")
    adapter = replace(
        _class_adapter("amr"),
        command=("/reviewed/amrfinder", "--plus"),
        raw_output_path=str(normalized),
        raw_output_sha256="0" * 64,
    )

    trusted = _trusted_adapter_bundle({"amr": adapter}, record_root=tmp_path)["amr"]

    assert trusted.class_result.state is SafetyState.INDETERMINATE
    assert trusted.class_result.reason_codes == ("ADAPTER_OUTPUT_DIGEST_DRIFT",)
    assert _adapter_execution_status(trusted) == "FAILED"


def test_eukaryotic_replication_objective_is_biological_fail_exit_two(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import main

    request = tmp_path / "scope.json"
    output = tmp_path / "scope-result.json"
    request.write_text(
        json.dumps(
            {
                "objective": {
                    "kind": "productive_replication",
                    "direction": "increase",
                    "replication_host_domains": ["EUKARYOTA"],
                    "endpoint": "productive_replication",
                },
                "host_evidence": {
                    "source": "curated-host-catalog",
                    "source_version": "2026-08-08",
                    "replication_host_domains": ["EUKARYOTA"],
                    "confirmed": True,
                    "metadata": {},
                },
            }
        )
    )

    exit_code = main(["validate-design-scope", "--input", str(request), "--output", str(output)])
    result = json.loads(output.read_text())

    assert exit_code == 2
    assert result["state"] == "FAIL"
    assert result["reason_codes"][0] == "EUKARYOTIC_REPLICATION_OBJECTIVE"


def test_conflicting_host_evidence_is_indeterminate_exit_three(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import main

    request = tmp_path / "scope.json"
    output = tmp_path / "scope-result.json"
    request.write_text(
        json.dumps(
            {
                "objective": {
                    "kind": "productive_replication",
                    "direction": "increase",
                    "replication_host_domains": ["BACTERIA"],
                    "endpoint": "productive_replication",
                },
                "host_evidence": {
                    "source": "curated-host-catalog",
                    "source_version": "2026-08-08",
                    "replication_host_domains": ["BACTERIA", "EUKARYOTA"],
                    "confirmed": True,
                    "metadata": {},
                },
            }
        )
    )

    exit_code = main(["validate-design-scope", "--input", str(request), "--output", str(output)])
    result = json.loads(output.read_text())

    assert exit_code == 3
    assert result["state"] == "INDETERMINATE"
    assert result["host_evidence_decision"]["quarantined"] is True


def test_tool_pin_file_binds_expected_bytes_and_exact_version(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import CLIValidationError, load_tool_pin_file

    executable = tmp_path / "diamond"
    executable.write_bytes(b"reviewed-diamond-binary")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    pin_file = tmp_path / "diamond-pin.json"
    pin_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "diamond",
                "path": str(executable),
                "sha256": digest,
                "version": "diamond version 2.1.11",
                "version_args": ["version"],
            }
        )
    )
    calls: list[list[str]] = []

    def version_runner(command, **kwargs):
        calls.append(command)
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 17.0,
        }
        return subprocess.CompletedProcess(command, 0, stdout="diamond version 2.1.11\n", stderr="")

    loaded = load_tool_pin_file(
        pin_file,
        expected_tool="diamond",
        runner=version_runner,
        timeout=17.0,
    )

    assert loaded.pin.path == executable
    assert loaded.pin.sha256 == digest
    assert loaded.pin.version_args == ("version",)
    assert loaded.pin_file_sha256 == hashlib.sha256(pin_file.read_bytes()).hexdigest()
    assert calls == [[str(executable), "version"]]

    executable.write_bytes(b"changed-after-review")
    with pytest.raises(CLIValidationError, match="pinned tool"):
        load_tool_pin_file(pin_file, expected_tool="diamond", runner=version_runner, timeout=17.0)


def test_fasta_parser_preserves_original_record_bytes_and_normalizes_only_scan_copy(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import CLIValidationError, parse_fasta_records

    fasta = tmp_path / "input.fna"
    fasta.write_bytes(b">phage-a description\r\natg aa\r\n>phage-b\nNnN\n")

    records = parse_fasta_records(fasta)

    assert [record.sequence_id for record in records] == ["phage-a", "phage-b"]
    assert [record.normalized_sequence for record in records] == ["ATGAA", "NNN"]
    assert records[0].original_bytes == b">phage-a description\r\natg aa\r\n"
    assert records[1].original_bytes == b">phage-b\nNnN\n"

    fasta.write_bytes(b">duplicate\nATG\n>duplicate second\nTAG\n")
    with pytest.raises(CLIValidationError, match="duplicate FASTA record ID"):
        parse_fasta_records(fasta)


def test_partition_fasta_records_is_disjoint_complete_ordered_and_byte_exact(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import parse_fasta_records, partition_fasta_records

    fasta = tmp_path / "input.fna"
    fasta.write_bytes(b">pass-a details\r\natg aa\r\n>fail-b\nCCCTAG\n>pass-c\nNnN\n>review-d extra\nATG TAG\n")
    records = parse_fasta_records(fasta)

    partitions = partition_fasta_records(
        records,
        {
            "pass-a": SafetyState.PASS,
            "fail-b": SafetyState.FAIL,
            "pass-c": SafetyState.PASS,
            "review-d": SafetyState.INDETERMINATE,
        },
    )

    assert partitions[SafetyState.PASS] == records[0].original_bytes + records[2].original_bytes
    assert partitions[SafetyState.FAIL] == records[1].original_bytes
    assert partitions[SafetyState.INDETERMINATE] == records[3].original_bytes
    assert b"atg aa\r\n" in partitions[SafetyState.PASS]


def test_scan_records_invokes_adapter_bundle_once_per_record_and_preserves_state_order(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import parse_fasta_records, scan_records

    fasta = tmp_path / "input.fna"
    fasta.write_bytes(b">pass-record\nATGAAATAG\n>review-record\nATGCCCTAG\n")
    records = parse_fasta_records(fasta)
    calls: list[str] = []

    def scanner(record, input_index):
        calls.append(f"{input_index}:{record.sequence_id}")
        supplemental = ()
        if record.sequence_id == "review-record":
            supplemental = (
                SafetyFinding(
                    safety_class="toxin",
                    state=SafetyState.INDETERMINATE,
                    reason_codes=("AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL",),
                    finding_id="review-record-virulence",
                ),
            )
        return {
            "amr": _class_adapter("amr", supplemental=supplemental),
            "toxin": _class_adapter("toxin"),
            "lysogeny": _class_adapter("lysogeny"),
        }

    batch = scan_records(records, scanner=scanner, host_domain=HostDomain.BACTERIA)

    assert calls == ["0:pass-record", "1:review-record"]
    assert [record.sequence_id for record in batch.records] == ["pass-record", "review-record"]
    assert [record.result.state for record in batch.records] == [SafetyState.PASS, SafetyState.INDETERMINATE]
    assert batch.state is SafetyState.INDETERMINATE


def test_scan_records_bounds_parallel_workers_and_preserves_input_order(tmp_path: Path):
    import threading

    from bionemo.evo2_phage_gen.sequence_safety_cli import parse_fasta_records, scan_records

    fasta = tmp_path / "input.fna"
    fasta.write_bytes(b"".join(f">record-{index}\nATGAAATAG\n".encode() for index in range(4)))
    records = parse_fasta_records(fasta)
    lock = threading.Lock()
    two_workers_entered = threading.Event()
    active = 0
    maximum_active = 0
    entered = 0

    def scanner(record, _input_index):
        nonlocal active, maximum_active, entered
        with lock:
            active += 1
            entered += 1
            maximum_active = max(maximum_active, active)
            if entered == 2:
                two_workers_entered.set()
        assert two_workers_entered.wait(timeout=2.0)
        try:
            return {
                "amr": _class_adapter("amr"),
                "toxin": _class_adapter("toxin"),
                "lysogeny": _class_adapter("lysogeny"),
            }
        finally:
            with lock:
                active -= 1

    batch = scan_records(
        records,
        scanner=scanner,
        host_domain=HostDomain.BACTERIA,
        max_workers=2,
    )

    assert maximum_active == 2
    assert [record.sequence_id for record in batch.records] == [f"record-{index}" for index in range(4)]


def test_record_worker_budget_rejects_nested_cpu_oversubscription():
    from bionemo.evo2_phage_gen.sequence_safety_cli import CLIValidationError, _resolve_record_workers

    assert _resolve_record_workers(requested=4, record_count=20, tool_threads=4, cpu_slots=16) == 4
    assert _resolve_record_workers(requested=4, record_count=2, tool_threads=4, cpu_slots=16) == 2
    with pytest.raises(CLIValidationError, match="CPU slots"):
        _resolve_record_workers(requested=5, record_count=20, tool_threads=4, cpu_slots=16)


def test_serialized_record_materializes_every_adapter_attempt(tmp_path: Path):
    from bionemo.evo2_phage_gen.design_scope import HostEvidence
    from bionemo.evo2_phage_gen.sequence_safety_cli import (
        _serialize_scanned_record,
        parse_fasta_records,
        scan_records,
    )

    fasta = tmp_path / "input.fna"
    fasta.write_bytes(b">phage-a\nATGAAATAG\n")
    record = parse_fasta_records(fasta)[0]
    batch = scan_records(
        (record,),
        scanner=lambda *_: {"amr": _class_adapter("amr"), "toxin": _class_adapter("toxin")},
        host_domain=HostDomain.BACTERIA,
    )
    evidence = HostEvidence(
        source="catalog",
        source_version="v1",
        replication_host_domains=frozenset({HostDomain.BACTERIA}),
        confirmed=True,
        metadata={},
    )

    serialized = _serialize_scanned_record(
        batch.records[0],
        record,
        root=tmp_path,
        evidence=evidence,
        host_domain=HostDomain.BACTERIA,
        strict_lysis=False,
        circular=True,
    )

    assert [attempt["safety_class"] for attempt in serialized["adapter_attempts"]] == [
        "amr",
        "toxin",
        "lysogeny",
    ]
    assert serialized["adapter_attempts"][2]["state"] == "INDETERMINATE"
    assert serialized["adapter_attempts"][2]["reason_codes"] == ["REQUIRED_ADAPTER_RESULT_MISSING"]


def test_validator_rejects_toxin_adapter_and_class_state_mismatch(tmp_path: Path):
    from bionemo.evo2_phage_gen import sequence_safety_adapters as safety_adapters
    from bionemo.evo2_phage_gen.sequence_safety_cli import CLIValidationError, _validate_adapter_attempts

    record_root = tmp_path / "records" / "000000-phage-a"
    record_root.mkdir(parents=True)
    artifacts = {}
    raw_artifacts = {}
    for safety_class, columns in (
        ("toxin", safety_adapters._DIAMOND_COLUMNS),
        ("lysogeny", safety_adapters._PHROGS_COLUMNS),
    ):
        raw = record_root / f"{safety_class}.raw.tsv"
        raw.write_text("")
        normalized = record_root / f"{safety_class}.tsv"
        safety_adapters._write_normalized_header(normalized, columns, raw)
        artifacts[safety_class] = {
            "path": str(normalized.relative_to(tmp_path)),
            "sha256": hashlib.sha256(normalized.read_bytes()).hexdigest(),
            "owned": True,
        }
        raw_artifacts[safety_class] = {
            "path": str(raw.relative_to(tmp_path)),
            "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "owned": True,
        }
    amr = record_root / "amrfinder.tsv"
    amr.write_text("header\n")
    artifacts["amr"] = raw_artifacts["amr"] = {
        "path": str(amr.relative_to(tmp_path)),
        "sha256": hashlib.sha256(amr.read_bytes()).hexdigest(),
        "owned": True,
    }
    attempts = [
        {
            "safety_class": safety_class,
            "execution_status": "COMPLETED_AND_PARSED",
            "state": "PASS",
            "reason_codes": ["MEASURED_NO_HIT"],
            "policy_id": _ADAPTER_POLICIES[safety_class][0],
            "policy_sha256": _ADAPTER_POLICIES[safety_class][1],
            "command_cwd": "@OUTPUT_ROOT",
            "command": ["tool", "search"],
            "normalized_output": artifacts[safety_class],
            "raw_command_output": raw_artifacts[safety_class],
            "primary_findings": [],
            "supplemental_findings": [],
        }
        for safety_class in ("amr", "toxin", "lysogeny")
    ]
    class_results = (
        SafetyClassResult("amr", SafetyState.PASS, True, reason_codes=("MEASURED_NO_HIT",)),
        SafetyClassResult("toxin", SafetyState.INDETERMINATE, True, reason_codes=("REVIEW_HIT",)),
        SafetyClassResult("lysogeny", SafetyState.PASS, True, reason_codes=("MEASURED_NO_HIT",)),
    )

    with pytest.raises(CLIValidationError, match="adapter and class result state mismatch"):
        _validate_adapter_attempts(
            attempts,
            root=tmp_path,
            record_id="phage-a",
            input_index=0,
            class_results=class_results,
            expected_tool_paths={"amr": "tool", "toxin": "tool", "lysogeny": "tool"},
        )


def _normalized_toxin_finding(
    tmp_path: Path,
    *,
    finding_id: str,
    state: SafetyState = SafetyState.FAIL,
    start: int = 1,
    end: int = 9,
):
    from bionemo.evo2_phage_gen.sequence_safety_adapters import NormalizedSafetyFinding

    return NormalizedSafetyFinding(
        safety_class="toxin",
        state=state,
        reason_codes=("TOXIN_HIGH_CONFIDENCE_HOMOLOGY",),
        finding_id=finding_id,
        detector="diamond-reviewed-toxin",
        accession="P12345",
        query_id="phage-a__orf0001",
        sequence_id="phage-a",
        start=start,
        end=end,
        strand="+",
        frame=1,
        scores={"identity": 90.0},
        thresholds={
            "evalue": 1e-10,
            "identity": 80.0,
            "query_coverage": 80.0,
            "reference_coverage": 80.0,
        },
        source_path=str(tmp_path / "toxin.dmnd"),
        source_sha256="1" * 64,
        tool_version="diamond-v1",
        database_version="UniProt 2026_01",
        evidence_path="pyrodigal-gv",
        evidence_method="diamond-blastp",
        threshold_policy=_ADAPTER_POLICIES["toxin"][0],
        threshold_policy_sha256=_ADAPTER_POLICIES["toxin"][1],
        tool_path=str(tmp_path / "diamond"),
        tool_sha256="2" * 64,
    )


def test_curated_domain_profile_provenance_compares_string_values(tmp_path: Path):
    """Equivalent profile strings from JSON and the asset manifest need not be the same object."""
    from bionemo.evo2_phage_gen.sequence_safety_adapters import TOXIN_HOMOLOGY_POLICY_V2
    from bionemo.evo2_phage_gen.sequence_safety_cli import _validate_finding

    accession = "PF15658.11"
    observed_profile = ("x" + accession)[1:]
    expected_profile = ("y" + accession)[1:]
    assert observed_profile == expected_profile and observed_profile is not expected_profile
    finding = replace(
        _normalized_toxin_finding(
            tmp_path,
            finding_id="toxin:phage-a__orf0001:PF15658.11",
            state=SafetyState.INDETERMINATE,
        ),
        reason_codes=("TOXIN_LATROTOXIN_C_DOMAIN_HOMOLOGY_REVIEW",),
        detector="diamond-curated-toxin-domain",
        accession=accession,
        profile=observed_profile,
        thresholds=TOXIN_HOMOLOGY_POLICY_V2.curated_domain_review.to_dict(),
    )
    provenance = {
        "detector": "diamond-reviewed-toxin",
        "detector_by_accession": {accession: "diamond-curated-toxin-domain"},
        "profile": None,
        "profile_by_accession": {accession: expected_profile},
        "source_path": finding.source_path,
        "source_sha256": finding.source_sha256,
        "tool_version": finding.tool_version,
        "database_version": finding.database_version,
        "tool_path": finding.tool_path,
        "tool_sha256": finding.tool_sha256,
        "evidence_method": finding.evidence_method,
        "policy_descriptor": TOXIN_HOMOLOGY_POLICY_V2.to_dict(),
    }

    assert (
        _validate_finding(
            finding.to_dict(),
            record_id="phage-a",
            safety_class="toxin",
            expected_policy=(TOXIN_HOMOLOGY_POLICY_V2.policy_id, TOXIN_HOMOLOGY_POLICY_V2.sha256),
            provenance=provenance,
        ).profile
        == accession
    )


def test_class_result_state_must_follow_finding_precedence(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import CLIValidationError, _validate_class_results

    finding = _normalized_toxin_finding(tmp_path, finding_id="toxin:one")
    rows = [
        SafetyClassResult("amr", SafetyState.PASS, True, reason_codes=("NO_HIT",)).to_dict(),
        SafetyClassResult(
            "toxin",
            SafetyState.PASS,
            True,
            findings=(finding,),
            reason_codes=("NO_HIT",),
        ).to_dict(),
        SafetyClassResult("lysogeny", SafetyState.PASS, True, reason_codes=("NO_HIT",)).to_dict(),
    ]

    with pytest.raises(CLIValidationError, match="finding precedence"):
        _validate_class_results(
            rows,
            record_id="phage-a",
            applicability={"amr": True, "toxin": True, "lysogeny": True},
        )


def test_class_results_reject_semantically_duplicate_findings_and_invalid_coordinates(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import CLIValidationError, _validate_class_results

    first = _normalized_toxin_finding(tmp_path, finding_id="toxin:one")
    duplicate = replace(first, finding_id="toxin:two")
    duplicate_rows = [
        SafetyClassResult("amr", SafetyState.PASS, True, reason_codes=("NO_HIT",)).to_dict(),
        SafetyClassResult(
            "toxin",
            SafetyState.FAIL,
            True,
            findings=(first, duplicate),
            reason_codes=("TOXIN_HIGH_CONFIDENCE_HOMOLOGY",),
        ).to_dict(),
        SafetyClassResult("lysogeny", SafetyState.PASS, True, reason_codes=("NO_HIT",)).to_dict(),
    ]
    with pytest.raises(CLIValidationError, match="semantically duplicate"):
        _validate_class_results(
            duplicate_rows,
            record_id="phage-a",
            applicability={"amr": True, "toxin": True, "lysogeny": True},
        )

    invalid = _normalized_toxin_finding(tmp_path, finding_id="toxin:invalid", start=9, end=1)
    invalid_rows = [
        SafetyClassResult("amr", SafetyState.PASS, True, reason_codes=("NO_HIT",)).to_dict(),
        SafetyClassResult(
            "toxin",
            SafetyState.FAIL,
            True,
            findings=(invalid,),
            reason_codes=("TOXIN_HIGH_CONFIDENCE_HOMOLOGY",),
        ).to_dict(),
        SafetyClassResult("lysogeny", SafetyState.PASS, True, reason_codes=("NO_HIT",)).to_dict(),
    ]
    with pytest.raises(CLIValidationError, match="coordinates"):
        _validate_class_results(
            invalid_rows,
            record_id="phage-a",
            applicability={"amr": True, "toxin": True, "lysogeny": True},
        )


def test_asset_manifest_wrapper_binds_recipe_digest_and_rejects_unknown_top_level_keys(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import CLIValidationError, load_safety_asset_manifest

    recipe = tmp_path / "phage_safety_assets.yaml"
    recipe.write_text("schema_version: 2\n")
    recipe_digest = hashlib.sha256(recipe.read_bytes()).hexdigest()
    payload = {
        "schema_version": 2,
        "recipe": {"path": str(recipe), "sha256": recipe_digest},
        "amrfinder_plus": {},
        "toxin_reference": {},
        "phrogs_v4": {},
    }
    manifest_path = tmp_path / "asset_manifest.yaml"
    manifest_path.write_text(json.dumps(payload))
    validations: list[tuple[dict, bool]] = []

    def validator(manifest, *, verify_asset_paths):
        validations.append((manifest, verify_asset_paths))

    loaded = load_safety_asset_manifest(manifest_path, validator=validator)

    assert loaded.manifest == payload
    assert loaded.manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert validations == [(payload, True)]

    recipe.write_text("schema_version: 1\n")
    with pytest.raises(CLIValidationError, match="recipe digest drift"):
        load_safety_asset_manifest(manifest_path, validator=validator)

    recipe.write_text("schema_version: 2\n")
    payload["unexpected"] = {}
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(CLIValidationError, match="top-level keys"):
        load_safety_asset_manifest(manifest_path, validator=validator)


def test_safety_asset_loader_rejects_manifest_mutation_between_parse_and_return(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_cli import CLIValidationError, load_safety_asset_manifest

    recipe = tmp_path / "phage_safety_assets.yaml"
    recipe.write_text("schema_version: 2\n")
    payload = {
        "schema_version": 2,
        "recipe": {"path": str(recipe), "sha256": hashlib.sha256(recipe.read_bytes()).hexdigest()},
        "amrfinder_plus": {},
        "toxin_reference": {},
        "phrogs_v4": {},
    }
    manifest_path = tmp_path / "asset_manifest.yaml"
    manifest_path.write_text(json.dumps(payload))

    def mutating_validator(manifest, *, verify_asset_paths):
        assert manifest == payload
        assert verify_asset_paths is True
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(CLIValidationError, match="safety asset manifest changed during validation"):
        load_safety_asset_manifest(manifest_path, validator=mutating_validator)


def test_cli_identity_rejects_an_arbitrary_matching_path_and_digest():
    from bionemo.evo2_phage_gen.sequence_safety_cli import (
        CLI_ID,
        CLI_VERSION,
        CLIValidationError,
        _validate_cli_identity,
    )

    arbitrary = Path(__file__).parents[3] / "pyproject.toml"
    with pytest.raises(CLIValidationError, match="CLI source path mismatch"):
        _validate_cli_identity(
            {
                "name": CLI_ID,
                "version": CLI_VERSION,
                "entry_point": "bionemo.evo2_phage_gen.sequence_safety_cli:main",
                "source_path": str(arbitrary),
                "source_sha256": hashlib.sha256(arbitrary.read_bytes()).hexdigest(),
            }
        )


def test_default_adapter_bundle_prepares_one_genome_and_calls_all_pinned_adapters(tmp_path: Path):
    from bionemo.evo2_phage_gen.sequence_safety_adapters import (
        ORFArtifacts,
        ORFPreparationResult,
        ORFQueryRecord,
        ToolPin,
    )
    from bionemo.evo2_phage_gen.sequence_safety_cli import FastaRecord, run_default_adapter_bundle

    executable = tmp_path / "tool"
    executable.write_bytes(b"tool")
    tool_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    diamond_pin = ToolPin(executable, tool_digest, "diamond-v1")
    mmseqs_pin = ToolPin(executable, tool_digest, "mmseqs-v1")
    record = FastaRecord(
        sequence_id="phage-a",
        original_bytes=b">phage-a\nATGAAATAG\n",
        normalized_sequence="ATGAAATAG",
    )
    artifacts: ORFArtifacts | None = None
    calls: list[tuple[str, object]] = []

    def prepare(genomes, work_dir, **kwargs):
        nonlocal artifacts
        calls.append(("prepare", (genomes, work_dir, kwargs)))
        assert len(genomes) == 1
        assert genomes[0].sequence_id == "phage-a"
        assert genomes[0].sequence == "ATGAAATAG"
        artifact_paths = {
            "genomes_fna": work_dir / "genomes.fna",
            "proteins_faa": work_dir / "proteins.faa",
            "proteins_fna": work_dir / "proteins.fna",
            "proteins_gff": work_dir / "proteins.gff",
            "all_queries_faa": work_dir / "all_queries.faa",
        }
        artifact_paths["genomes_fna"].write_text(">phage-a\nATGAAATAG\n")
        artifact_paths["proteins_faa"].write_text(">phage-a__orf0001\nMK\n")
        artifact_paths["proteins_fna"].write_text(">phage-a__orf0001\nATGAAA\n")
        artifact_paths["proteins_gff"].write_text("##gff-version 3\n")
        artifact_paths["all_queries_faa"].write_text(">phage-a__orf0001\nMK\n")
        artifacts = ORFArtifacts(
            **artifact_paths,
            query_records=(
                ORFQueryRecord(
                    query_id="phage-a__orf0001",
                    sequence_id="phage-a",
                    start=1,
                    end=6,
                    strand="+",
                    frame=1,
                    nucleotide="ATGAAA",
                    protein="MK",
                    evidence_path="pyrodigal-gv",
                ),
            ),
        )
        return ORFPreparationResult(SafetyState.PASS, artifacts)

    def amr(received_artifacts, **kwargs):
        calls.append(("amr", kwargs))
        assert received_artifacts is artifacts
        assert kwargs["required"] is True
        assert kwargs["manifest_section"] == {"amr": "assets"}
        return _class_adapter("amr")

    def toxin(received_artifacts, **kwargs):
        calls.append(("toxin", kwargs))
        assert received_artifacts is artifacts
        assert kwargs["required"] is True
        assert kwargs["tool_pin"] is diamond_pin
        assert kwargs["manifest_section"] == {"toxin": "assets"}
        return _class_adapter("toxin")

    def phrogs(received_artifacts, **kwargs):
        calls.append(("lysogeny", kwargs))
        assert received_artifacts is artifacts
        assert kwargs["host_domain"] is HostDomain.BACTERIA
        assert kwargs["strict_lysis"] is False
        assert kwargs["tool_pin"] is mmseqs_pin
        assert kwargs["manifest_section"] == {"phrogs": "assets"}
        return _class_adapter("lysogeny")

    result = run_default_adapter_bundle(
        record,
        3,
        work_root=tmp_path / "records",
        asset_manifest={
            "amrfinder_plus": {"amr": "assets"},
            "toxin_reference": {"toxin": "assets"},
            "phrogs_v4": {"phrogs": "assets"},
        },
        diamond_pin=diamond_pin,
        mmseqs_pin=mmseqs_pin,
        host_domain=HostDomain.BACTERIA,
        prepare_orfs=prepare,
        amr_adapter=amr,
        toxin_adapter=toxin,
        phrogs_adapter=phrogs,
    )

    assert list(result) == ["amr", "toxin", "lysogeny"]
    assert [name for name, _ in calls] == ["prepare", "amr", "toxin", "lysogeny"]
    assert (tmp_path / "records" / "000003-phage-a").is_dir()


def test_scan_cli_runs_each_record_and_atomically_publishes_a_provenance_manifest(tmp_path: Path):
    from bionemo.evo2_phage_gen import sequence_safety_adapters as safety_adapters
    from bionemo.evo2_phage_gen.sequence_safety_adapters import (
        ORFArtifacts,
        ORFQueryRecord,
        ToolPin,
        build_amrfinder_command,
        build_diamond_command,
        build_phrogs_command,
    )
    from bionemo.evo2_phage_gen.sequence_safety_cli import (
        CLIRuntime,
        LoadedSafetyAssetManifest,
        LoadedToolPin,
        ReplayedORFEvidence,
        _write_orf_provenance,
        main,
    )

    input_fasta = tmp_path / "input.fna"
    input_fasta.write_bytes(b">phage-a\nATGAAATAG\n>phage-b\natgccctag\n>phage-c details\r\nNnN\r\n")
    policy_path = _write_policy(tmp_path / "policy.yaml")
    asset_path = tmp_path / "assets.yaml"
    asset_path.write_text("asset-manifest\n")
    recipe = tmp_path / "asset-recipe.yaml"
    recipe.write_text("recipe\n")
    executable = tmp_path / "tool"
    executable.write_bytes(b"tool")
    tool_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    amr_database = tmp_path / "amr-database"
    amr_database.mkdir()
    toxin_database = tmp_path / "toxins.dmnd"
    toxin_database.write_bytes(b"toxin-db")
    phrogs_database = tmp_path / "phrogs-profile-db"
    phrogs_database.write_bytes(b"phrogs-db")
    asset_payload = {
        "schema_version": 1,
        "recipe": {"path": str(recipe), "sha256": hashlib.sha256(recipe.read_bytes()).hexdigest()},
        "amrfinder_plus": {
            "binary_path": str(executable),
            "binary_sha256": tool_digest,
            "amrfinder_version": "amrfinder-v1",
            "database_path": str(amr_database),
            "database_sha256": "a" * 64,
            "database_version": "amr-db-v1",
        },
        "toxin_reference": {
            "uniprot_release": "2026_01",
            "files": {
                "diamond_database": {
                    "path": str(toxin_database),
                    "sha256": hashlib.sha256(toxin_database.read_bytes()).hexdigest(),
                }
            },
        },
        "phrogs_v4": {
            "profile_database": {
                "path": str(phrogs_database),
                "sha256": hashlib.sha256(phrogs_database.read_bytes()).hexdigest(),
            }
        },
    }
    loaded_assets = LoadedSafetyAssetManifest(
        manifest=asset_payload,
        manifest_path=asset_path,
        manifest_sha256=hashlib.sha256(asset_path.read_bytes()).hexdigest(),
        recipe_path=recipe,
        recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
    )
    pin_file = tmp_path / "tool-pin.json"
    pin_file.write_text("{}\n")
    loaded_pins = {
        "diamond": LoadedToolPin(
            "diamond",
            ToolPin(executable, tool_digest, "diamond-v1"),
            pin_file,
            hashlib.sha256(pin_file.read_bytes()).hexdigest(),
        ),
        "mmseqs": LoadedToolPin(
            "mmseqs",
            ToolPin(executable, tool_digest, "mmseqs-v1"),
            pin_file,
            hashlib.sha256(pin_file.read_bytes()).hexdigest(),
        ),
    }
    calls: list[str] = []
    orf_identity = {
        "predictor": "injected-deterministic-predictor",
        "predictor_version": "test-v1",
        "entry_point": "tests:record_scanner",
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    def orf_replayer(record, *, circular):
        query_id = f"{record.sequence_id}__orf0001"
        proteins = {
            "ATGAAATAG": "MK",
            "ATGCCCTAG": "MP",
            "NNN": "X",
        }
        protein = proteins[record.normalized_sequence]
        query = ORFQueryRecord(
            query_id=query_id,
            sequence_id=record.sequence_id,
            start=1,
            end=len(record.normalized_sequence),
            strand="+",
            frame=1,
            nucleotide=record.normalized_sequence,
            protein=protein,
            evidence_path="pyrodigal-gv",
        )
        return ReplayedORFEvidence(
            artifact_bytes={
                "genomes_fna": (
                    f">{record.sequence_id} circular={'true' if circular else 'false'}\n{record.normalized_sequence}\n"
                ).encode(),
                "proteins_faa": f">{query_id}\n{protein}\n".encode(),
                "proteins_fna": f">{query_id}\n{record.normalized_sequence}\n".encode(),
                "proteins_gff": b"##gff-version 3\n",
                "all_queries_faa": f">{query_id}\n{protein}\n".encode(),
            },
            query_records=(query,),
        )

    def record_scanner(
        record,
        input_index,
        *,
        work_root,
        diamond_pin,
        mmseqs_pin,
        asset_manifest,
        threads,
        orf_generation_identity,
        circular,
        **kwargs,
    ):
        calls.append(record.sequence_id)
        record_dir = work_root / f"{input_index:06d}-{record.sequence_id}"
        record_dir.mkdir(parents=True)
        artifact_paths = {
            "genomes_fna": record_dir / "genomes.fna",
            "proteins_faa": record_dir / "proteins.faa",
            "proteins_fna": record_dir / "proteins.fna",
            "proteins_gff": record_dir / "proteins.gff",
            "all_queries_faa": record_dir / "all_queries.faa",
        }
        replayed_orfs = orf_replayer(record, circular=circular)
        for role, payload in replayed_orfs.artifact_bytes.items():
            artifact_paths[role].write_bytes(payload)
        artifacts = ORFArtifacts(
            **artifact_paths,
            query_records=replayed_orfs.query_records,
        )
        _write_orf_provenance(
            record_dir,
            artifacts=artifacts,
            generation_identity=orf_generation_identity,
        )
        normalized_paths = {
            "amr": record_dir / "amrfinder.tsv",
            "toxin": record_dir / "toxin_diamond.tsv",
            "lysogeny": record_dir / "phrogs.tsv",
        }
        raw_paths = {
            "amr": normalized_paths["amr"],
            "toxin": record_dir / "toxin_diamond.raw.tsv",
            "lysogeny": record_dir / "phrogs.raw.tsv",
        }
        raw_paths["amr"].write_text("header\n")
        raw_paths["toxin"].write_text("")
        raw_paths["lysogeny"].write_text("")
        safety_adapters._write_normalized_header(
            normalized_paths["toxin"], safety_adapters._DIAMOND_COLUMNS, raw_paths["toxin"]
        )
        safety_adapters._write_normalized_header(
            normalized_paths["lysogeny"], safety_adapters._PHROGS_COLUMNS, raw_paths["lysogeny"]
        )
        commands = {
            "amr": build_amrfinder_command(
                amrfinder=Path(asset_manifest["amrfinder_plus"]["binary_path"]),
                genomes_fna=artifacts.genomes_fna,
                proteins_faa=artifacts.proteins_faa,
                proteins_gff=artifacts.proteins_gff,
                database_dir=Path(asset_manifest["amrfinder_plus"]["database_path"]),
                threads=threads,
                output_tsv=raw_paths["amr"],
            ),
            "toxin": build_diamond_command(
                diamond=diamond_pin.path,
                queries_faa=artifacts.all_queries_faa,
                database=Path(asset_manifest["toxin_reference"]["files"]["diamond_database"]["path"]),
                output_tsv=raw_paths["toxin"],
                threads=threads,
            ),
            "lysogeny": build_phrogs_command(
                mmseqs=mmseqs_pin.path,
                profile_database=Path(asset_manifest["phrogs_v4"]["profile_database"]["path"]),
                proteins_faa=artifacts.proteins_faa,
                output_tsv=raw_paths["lysogeny"],
                temporary_dir=record_dir / "tmp",
                threads=threads,
            ),
        }
        adapters = {}
        for safety_class in ("amr", "toxin", "lysogeny"):
            normalized = normalized_paths[safety_class]
            policy_id, policy_sha256 = _ADAPTER_POLICIES[safety_class]
            adapters[safety_class] = AdapterResult(
                class_result=SafetyClassResult(
                    safety_class=safety_class,
                    state=SafetyState.PASS,
                    required=True,
                    reason_codes=(f"{safety_class.upper()}_PINNED_SEARCH_PASS",),
                ),
                command=tuple(commands[safety_class]),
                raw_output_path=str(normalized),
                raw_output_sha256=hashlib.sha256(normalized.read_bytes()).hexdigest(),
                policy_id=policy_id,
                policy_sha256=policy_sha256,
            )
        return adapters

    def adapter_replayer(safety_class, *, normalized_output, required, **kwargs):
        policy_id, policy_sha256 = _ADAPTER_POLICIES[safety_class]
        payload = normalized_output.read_bytes()
        if payload == b"FAIL\n":
            state, reason = SafetyState.FAIL, "AMR_HIT"
        elif payload.endswith(b"INDET\n"):
            state, reason = SafetyState.INDETERMINATE, "TOXIN_REVIEW_HIT"
        elif payload in {
            b"header\n",
            ("\t".join(safety_adapters._DIAMOND_COLUMNS) + "\n").encode(),
            ("\t".join(safety_adapters._PHROGS_COLUMNS) + "\n").encode(),
        }:
            state, reason = SafetyState.PASS, f"{safety_class.upper()}_PINNED_SEARCH_PASS"
        else:
            state, reason = SafetyState.INDETERMINATE, "PARSER_REPLAY_MISMATCH"
        return AdapterResult(
            class_result=SafetyClassResult(
                safety_class=safety_class,
                state=state,
                required=required,
                reason_codes=(reason,),
            ),
            raw_output_path=str(normalized_output),
            raw_output_sha256=hashlib.sha256(normalized_output.read_bytes()).hexdigest(),
            policy_id=policy_id,
            policy_sha256=policy_sha256,
        )

    runtime = CLIRuntime(
        record_scanner=record_scanner,
        asset_loader=lambda path: loaded_assets,
        tool_pin_loader=lambda path, *, expected_tool, **kwargs: loaded_pins[expected_tool],
        clock=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        environment_collector=lambda: {"python": "3.test", "platform": "test"},
        orf_identity_collector=lambda: orf_identity,
        orf_replayer=orf_replayer,
        adapter_replayer=adapter_replayer,
    )
    output_dir = tmp_path / "scan-output"
    host_evidence = {
        "source": "curated-host-catalog",
        "source_version": "2026-08-08",
        "replication_host_domains": ["BACTERIA"],
        "confirmed": True,
        "metadata": {},
    }

    exit_code = main(
        [
            "scan",
            "--input-fasta",
            str(input_fasta),
            "--output-dir",
            str(output_dir),
            "--policy",
            str(policy_path),
            "--asset-manifest",
            str(asset_path),
            "--diamond-tool-pin",
            str(pin_file),
            "--mmseqs-tool-pin",
            str(pin_file),
            "--host-domain",
            "BACTERIA",
            "--host-evidence-json",
            json.dumps(host_evidence),
        ],
        runtime=runtime,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert exit_code == 0
    assert calls == ["phage-a", "phage-b", "phage-c"]
    assert manifest["manifest_type"] == "sequence_safety_scan"
    assert manifest["aggregate"]["state"] == "PASS"
    assert manifest["policy"]["policy_id"] == "phage-sequence-safety-v1"
    assert manifest["policy"]["canonical_sha256"]
    assert all(policy["descriptor"]["policy_id"] == policy["policy_id"] for policy in manifest["adapter_policies"])
    assert manifest["safety_asset_manifest"]["sha256"] == loaded_assets.manifest_sha256
    assert [record["record_id"] for record in manifest["records"]] == ["phage-a", "phage-b", "phage-c"]
    assert all(record["state"] == "PASS" for record in manifest["records"])
    assert all(record["orf_provenance"]["artifact"]["owned"] is True for record in manifest["records"])
    assert all(record["orf_provenance"]["query_inventory_sha256"] for record in manifest["records"])
    for record in manifest["records"]:
        attempts = {attempt["safety_class"]: attempt for attempt in record["adapter_attempts"]}
        assert all(attempt["execution_status"] == "COMPLETED_AND_PARSED" for attempt in attempts.values())
        assert all(attempt["command_cwd"] == "@OUTPUT_ROOT" for attempt in attempts.values())
        assert all(
            not any(".scan-output." in argument for argument in attempt["command"]) for attempt in attempts.values()
        )
        assert attempts["amr"]["raw_command_output"] == attempts["amr"]["normalized_output"]
        assert attempts["toxin"]["raw_command_output"] != attempts["toxin"]["normalized_output"]
        assert attempts["lysogeny"]["raw_command_output"] != attempts["lysogeny"]["normalized_output"]
    assert "not proof of strict lysis" in manifest["claim_boundary"]["pass_meaning"]
    assert not any(path.name.startswith(".scan-output.") for path in tmp_path.iterdir())

    from bionemo.evo2_phage_gen.sequence_safety_cli import (
        CLIValidationError,
        parse_fasta_records,
        validate_manifest_file,
    )

    validated = validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    assert validated["aggregate"]["state"] == "PASS"

    normalized = output_dir / manifest["records"][0]["adapter_attempts"][0]["normalized_output"]["path"]
    original_normalized = normalized.read_bytes()
    normalized.write_bytes(b"tampered\n")
    with pytest.raises(CLIValidationError, match="digest drift"):
        validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    normalized.write_bytes(original_normalized)

    coherently_tampered = json.loads((output_dir / "manifest.json").read_bytes())
    normalized.write_bytes(b"coherently-tampered-output\n")
    tampered_digest = hashlib.sha256(normalized.read_bytes()).hexdigest()
    coherently_tampered["records"][0]["adapter_attempts"][0]["normalized_output"]["sha256"] = tampered_digest
    coherently_tampered["records"][0]["adapter_attempts"][0]["raw_command_output"]["sha256"] = tampered_digest
    (output_dir / "manifest.json").write_text(json.dumps(coherently_tampered))
    with pytest.raises(CLIValidationError, match="parser replay"):
        validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    normalized.write_bytes(original_normalized)
    (output_dir / "manifest.json").write_bytes(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")

    original_input = input_fasta.read_bytes()
    input_fasta.write_bytes(original_input + b">late-change\nATG\n")
    with pytest.raises(CLIValidationError, match="input FASTA digest drift"):
        validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    input_fasta.write_bytes(original_input)

    filter_dir = tmp_path / "filtered"
    filter_exit = main(
        [
            "filter-fasta",
            "--input-fasta",
            str(input_fasta),
            "--scan-manifest",
            str(output_dir / "manifest.json"),
            "--output-dir",
            str(filter_dir),
        ],
        runtime=runtime,
    )
    assert filter_exit == 0
    assert (filter_dir / "pass.fna").read_bytes() == input_fasta.read_bytes()
    assert (filter_dir / "fail.fna").read_bytes() == b""
    assert (filter_dir / "indeterminate.fna").read_bytes() == b""
    filter_manifest = json.loads((filter_dir / "manifest.json").read_text())
    assert filter_manifest["manifest_type"] == "sequence_safety_filter"
    assert filter_manifest["aggregate"]["state"] == "PASS"
    validate_manifest_file(filter_dir / "manifest.json", runtime=runtime)

    assert main(["validate-manifest", "--manifest", str(output_dir / "manifest.json")], runtime=runtime) == 0

    original_manifest_bytes = (output_dir / "manifest.json").read_bytes()
    false_pass_cases: list[str] = []

    substituted_orf_manifest = json.loads(original_manifest_bytes)
    substituted_orf_wrapper = substituted_orf_manifest["records"][0]["orf_provenance"]
    substituted_orf_path = output_dir / substituted_orf_wrapper["artifact"]["path"]
    original_orf_bytes = substituted_orf_path.read_bytes()
    substituted_orf = json.loads(original_orf_bytes)
    substituted_genome_path = substituted_orf_path.parent / substituted_orf["artifacts"]["genomes_fna"]["path"]
    original_genome_bytes = substituted_genome_path.read_bytes()
    substituted_genome_path.write_text(">phage-a circular=true\nTTTTTTTTT\n")
    substituted_orf["artifacts"]["genomes_fna"]["sha256"] = hashlib.sha256(
        substituted_genome_path.read_bytes()
    ).hexdigest()
    substituted_orf["artifacts"]["genomes_fna"]["size_bytes"] = substituted_genome_path.stat().st_size
    substituted_orf_path.write_text(json.dumps(substituted_orf))
    substituted_orf_wrapper["artifact"]["sha256"] = hashlib.sha256(substituted_orf_path.read_bytes()).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(substituted_orf_manifest))
    try:
        validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    except CLIValidationError:
        pass
    else:
        false_pass_cases.append("substituted ORF genome")
    substituted_genome_path.write_bytes(original_genome_bytes)
    substituted_orf_path.write_bytes(original_orf_bytes)
    (output_dir / "manifest.json").write_bytes(original_manifest_bytes)

    for attempt_index, safety_class in ((1, "toxin"), (2, "lysogeny")):
        hidden_hit_manifest = json.loads(original_manifest_bytes)
        attempt = hidden_hit_manifest["records"][0]["adapter_attempts"][attempt_index]
        raw_path = output_dir / attempt["raw_command_output"]["path"]
        original_raw_bytes = raw_path.read_bytes()
        raw_path.write_text("hidden-safety-hit-without-final-newline")
        attempt["raw_command_output"]["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        (output_dir / "manifest.json").write_text(json.dumps(hidden_hit_manifest))
        try:
            validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
        except CLIValidationError:
            pass
        else:
            false_pass_cases.append(f"hidden {safety_class} raw hit")
        raw_path.write_bytes(original_raw_bytes)
        (output_dir / "manifest.json").write_bytes(original_manifest_bytes)

    assert not false_pass_cases, f"validator accepted false-PASS evidence: {false_pass_cases}"

    tampered = json.loads(original_manifest_bytes)
    unbound_finding = _normalized_toxin_finding(tmp_path, finding_id="toxin:unbound").to_dict()
    toxin_class = tampered["records"][0]["class_results"][1]
    toxin_attempt = tampered["records"][0]["adapter_attempts"][1]
    toxin_class["state"] = "FAIL"
    toxin_class["reason_codes"] = ["TOXIN_HIGH_CONFIDENCE_HOMOLOGY"]
    toxin_class["findings"] = [unbound_finding]
    toxin_attempt["state"] = "FAIL"
    toxin_attempt["reason_codes"] = ["TOXIN_HIGH_CONFIDENCE_HOMOLOGY"]
    toxin_attempt["primary_findings"] = [unbound_finding]
    tampered["records"][0]["state"] = "FAIL"
    tampered["records"][0]["reason_codes"] = [
        "AMR_PINNED_SEARCH_PASS",
        "TOXIN_HIGH_CONFIDENCE_HOMOLOGY",
        "LYSOGENY_PINNED_SEARCH_PASS",
    ]
    tampered["aggregate"]["state"] = "FAIL"
    tampered["aggregate"]["counts"] = {"PASS": 2, "FAIL": 1, "INDETERMINATE": 0}
    tampered["aggregate"]["reason_codes"] = list(
        dict.fromkeys(reason for record in tampered["records"] for reason in record["reason_codes"])
    )
    (output_dir / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(CLIValidationError, match="finding (tool|source) provenance"):
        validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    (output_dir / "manifest.json").write_bytes(original_manifest_bytes)

    tampered = json.loads(original_manifest_bytes)
    tampered["records"][0]["adapter_attempts"][1]["command"][0] = str(tmp_path / "unreviewed-diamond")
    (output_dir / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(CLIValidationError, match="exact Task 3 replay command"):
        validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    (output_dir / "manifest.json").write_bytes(original_manifest_bytes)

    tampered = json.loads(original_manifest_bytes)
    tampered["aggregate"]["state"] = "FAIL"
    (output_dir / "manifest.json").write_text(json.dumps(tampered))
    assert main(["validate-manifest", "--manifest", str(output_dir / "manifest.json")], runtime=runtime) == 3
    (output_dir / "manifest.json").write_bytes(original_manifest_bytes)

    tampered = json.loads(original_manifest_bytes)
    tampered["records"][0]["unexpected"] = True
    (output_dir / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(CLIValidationError, match=r"record\[0\] keys"):
        validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    (output_dir / "manifest.json").write_bytes(original_manifest_bytes)

    raw_nonfinite = original_manifest_bytes.decode().replace('"environment": {', '"environment": {"nonfinite": NaN,')
    (output_dir / "manifest.json").write_text(raw_nonfinite)
    with pytest.raises(CLIValidationError, match="non-finite"):
        validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    (output_dir / "manifest.json").write_bytes(original_manifest_bytes)

    tampered = json.loads(original_manifest_bytes)
    tampered["records"][0]["input_index"] = False
    (output_dir / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(CLIValidationError, match="input_index must be an integer"):
        validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    (output_dir / "manifest.json").write_bytes(original_manifest_bytes)

    tampered = json.loads(original_manifest_bytes)
    tampered["aggregate"]["counts"]["FAIL"] = False
    (output_dir / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(CLIValidationError, match="aggregate counts must be integers"):
        validate_manifest_file(output_dir / "manifest.json", runtime=runtime)
    (output_dir / "manifest.json").write_bytes(original_manifest_bytes)

    mixed = json.loads(original_manifest_bytes)
    mixed_amr_path = output_dir / mixed["records"][0]["adapter_attempts"][0]["normalized_output"]["path"]
    mixed_amr_path.write_bytes(b"FAIL\n")
    mixed_amr_sha256 = hashlib.sha256(mixed_amr_path.read_bytes()).hexdigest()
    mixed["records"][0]["adapter_attempts"][0]["normalized_output"]["sha256"] = mixed_amr_sha256
    mixed["records"][0]["adapter_attempts"][0]["raw_command_output"]["sha256"] = mixed_amr_sha256
    mixed_toxin_path = output_dir / mixed["records"][1]["adapter_attempts"][1]["normalized_output"]["path"]
    mixed_toxin_raw_path = output_dir / mixed["records"][1]["adapter_attempts"][1]["raw_command_output"]["path"]
    mixed_toxin_raw_path.write_bytes(b"INDET")
    safety_adapters._write_normalized_header(mixed_toxin_path, safety_adapters._DIAMOND_COLUMNS, mixed_toxin_raw_path)
    mixed_toxin_sha256 = hashlib.sha256(mixed_toxin_path.read_bytes()).hexdigest()
    mixed["records"][1]["adapter_attempts"][1]["normalized_output"]["sha256"] = mixed_toxin_sha256
    mixed["records"][1]["adapter_attempts"][1]["raw_command_output"]["sha256"] = hashlib.sha256(
        mixed_toxin_raw_path.read_bytes()
    ).hexdigest()
    mixed["records"][0]["class_results"][0]["state"] = "FAIL"
    mixed["records"][0]["class_results"][0]["reason_codes"] = ["AMR_HIT"]
    mixed["records"][0]["adapter_attempts"][0]["state"] = "FAIL"
    mixed["records"][0]["adapter_attempts"][0]["reason_codes"] = ["AMR_HIT"]
    mixed["records"][0]["state"] = "FAIL"
    mixed["records"][0]["reason_codes"] = [
        "AMR_HIT",
        "TOXIN_PINNED_SEARCH_PASS",
        "LYSOGENY_PINNED_SEARCH_PASS",
    ]
    mixed["records"][1]["class_results"][1]["state"] = "INDETERMINATE"
    mixed["records"][1]["class_results"][1]["reason_codes"] = ["TOXIN_REVIEW_HIT"]
    mixed["records"][1]["adapter_attempts"][1]["state"] = "INDETERMINATE"
    mixed["records"][1]["adapter_attempts"][1]["reason_codes"] = ["TOXIN_REVIEW_HIT"]
    mixed["records"][1]["state"] = "INDETERMINATE"
    mixed["records"][1]["reason_codes"] = [
        "AMR_PINNED_SEARCH_PASS",
        "TOXIN_REVIEW_HIT",
        "LYSOGENY_PINNED_SEARCH_PASS",
    ]
    mixed["aggregate"]["state"] = "FAIL"
    mixed["aggregate"]["counts"] = {"PASS": 1, "FAIL": 1, "INDETERMINATE": 1}
    mixed["aggregate"]["reason_codes"] = list(
        dict.fromkeys(reason for record in mixed["records"] for reason in record["reason_codes"])
    )
    (output_dir / "manifest.json").write_text(json.dumps(mixed))
    validate_manifest_file(output_dir / "manifest.json", runtime=runtime)

    mixed_filter_dir = tmp_path / "mixed-filter"
    mixed_filter_exit = main(
        [
            "filter-fasta",
            "--input-fasta",
            str(input_fasta),
            "--scan-manifest",
            str(output_dir / "manifest.json"),
            "--output-dir",
            str(mixed_filter_dir),
        ],
        runtime=runtime,
    )
    assert mixed_filter_exit == 2
    source_records = parse_fasta_records(input_fasta)
    assert (mixed_filter_dir / "fail.fna").read_bytes() == source_records[0].original_bytes
    assert (mixed_filter_dir / "indeterminate.fna").read_bytes() == source_records[1].original_bytes
    assert (mixed_filter_dir / "pass.fna").read_bytes() == source_records[2].original_bytes
    validate_manifest_file(mixed_filter_dir / "manifest.json", runtime=runtime)

    mixed_filter_manifest_bytes = (mixed_filter_dir / "manifest.json").read_bytes()
    tampered_filter = json.loads(mixed_filter_manifest_bytes)
    tampered_filter["derivatives"]["fail"]["count"] = True
    (mixed_filter_dir / "manifest.json").write_text(json.dumps(tampered_filter))
    with pytest.raises(CLIValidationError, match="derivative count must be an integer"):
        validate_manifest_file(mixed_filter_dir / "manifest.json", runtime=runtime)
    (mixed_filter_dir / "manifest.json").write_bytes(mixed_filter_manifest_bytes)

    existing_filter_dir = tmp_path / "existing-filter"
    existing_filter_dir.mkdir()
    marker = existing_filter_dir / "keep.txt"
    marker.write_text("keep\n")
    assert (
        main(
            [
                "filter-fasta",
                "--input-fasta",
                str(input_fasta),
                "--scan-manifest",
                str(output_dir / "manifest.json"),
                "--output-dir",
                str(existing_filter_dir),
            ],
            runtime=runtime,
        )
        == 3
    )
    assert marker.read_text() == "keep\n"

    def fail_filter_replace(source: Path, destination: Path):
        raise OSError("injected filter rename failure")

    failed_filter_dir = tmp_path / "failed-filter"
    assert (
        main(
            [
                "filter-fasta",
                "--input-fasta",
                str(input_fasta),
                "--scan-manifest",
                str(output_dir / "manifest.json"),
                "--output-dir",
                str(failed_filter_dir),
            ],
            runtime=replace(runtime, replace=fail_filter_replace),
        )
        == 3
    )
    assert not failed_filter_dir.exists()
    assert not any(path.name.startswith(".failed-filter.") for path in tmp_path.iterdir())

    (mixed_filter_dir / "pass.fna").write_bytes(b"tampered\n")
    with pytest.raises(CLIValidationError, match="digest drift"):
        validate_manifest_file(mixed_filter_dir / "manifest.json", runtime=runtime)

    def fail_replace(source: Path, destination: Path):
        raise OSError("injected rename failure")

    failed_dir = tmp_path / "failed-scan"
    failed_runtime = replace(runtime, replace=fail_replace)
    failed_exit = main(
        [
            "scan",
            "--input-fasta",
            str(input_fasta),
            "--output-dir",
            str(failed_dir),
            "--policy",
            str(policy_path),
            "--asset-manifest",
            str(asset_path),
            "--diamond-tool-pin",
            str(pin_file),
            "--mmseqs-tool-pin",
            str(pin_file),
            "--host-domain",
            "BACTERIA",
            "--host-evidence-json",
            json.dumps(host_evidence),
        ],
        runtime=failed_runtime,
    )
    assert failed_exit == 3
    assert not failed_dir.exists()
    assert not any(path.name.startswith(".failed-scan.") for path in tmp_path.iterdir())

    race_policy = _write_policy(tmp_path / "race-policy.yaml")
    race_mutated = False

    def policy_mutating_scanner(*scanner_args, **scanner_kwargs):
        nonlocal race_mutated
        result = record_scanner(*scanner_args, **scanner_kwargs)
        if not race_mutated:
            race_policy.write_text(race_policy.read_text() + "\n# changed during scan\n")
            race_mutated = True
        return result

    race_output = tmp_path / "race-scan"
    race_exit = main(
        [
            "scan",
            "--input-fasta",
            str(input_fasta),
            "--output-dir",
            str(race_output),
            "--policy",
            str(race_policy),
            "--asset-manifest",
            str(asset_path),
            "--diamond-tool-pin",
            str(pin_file),
            "--mmseqs-tool-pin",
            str(pin_file),
            "--host-domain",
            "BACTERIA",
            "--host-evidence-json",
            json.dumps(host_evidence),
        ],
        runtime=replace(runtime, record_scanner=policy_mutating_scanner),
    )
    assert race_exit == 3
    assert not race_output.exists()
    assert not any(path.name.startswith(".race-scan.") for path in tmp_path.iterdir())

    mutable_input = tmp_path / "mutable-input.fna"
    mutable_input.write_bytes(input_fasta.read_bytes())
    input_mutated = False

    def input_mutating_scanner(*scanner_args, **scanner_kwargs):
        nonlocal input_mutated
        result = record_scanner(*scanner_args, **scanner_kwargs)
        if not input_mutated:
            mutable_input.write_bytes(mutable_input.read_bytes() + b"\n")
            input_mutated = True
        return result

    input_race_output = tmp_path / "input-race-scan"
    input_race_exit = main(
        [
            "scan",
            "--input-fasta",
            str(mutable_input),
            "--output-dir",
            str(input_race_output),
            "--policy",
            str(policy_path),
            "--asset-manifest",
            str(asset_path),
            "--diamond-tool-pin",
            str(pin_file),
            "--mmseqs-tool-pin",
            str(pin_file),
            "--host-domain",
            "BACTERIA",
            "--host-evidence-json",
            json.dumps(host_evidence),
        ],
        runtime=replace(runtime, record_scanner=input_mutating_scanner),
    )
    assert input_race_exit == 3
    assert not input_race_output.exists()
    assert not any(path.name.startswith(".input-race-scan.") for path in tmp_path.iterdir())

    original_asset_bytes = asset_path.read_bytes()
    asset_mutated = False

    def asset_mutating_scanner(*scanner_args, **scanner_kwargs):
        nonlocal asset_mutated
        result = record_scanner(*scanner_args, **scanner_kwargs)
        if not asset_mutated:
            asset_path.write_bytes(original_asset_bytes + b"changed during scan\n")
            asset_mutated = True
        return result

    asset_race_output = tmp_path / "asset-race-scan"
    asset_race_exit = main(
        [
            "scan",
            "--input-fasta",
            str(input_fasta),
            "--output-dir",
            str(asset_race_output),
            "--policy",
            str(policy_path),
            "--asset-manifest",
            str(asset_path),
            "--diamond-tool-pin",
            str(pin_file),
            "--mmseqs-tool-pin",
            str(pin_file),
            "--host-domain",
            "BACTERIA",
            "--host-evidence-json",
            json.dumps(host_evidence),
        ],
        runtime=replace(runtime, record_scanner=asset_mutating_scanner),
    )
    assert asset_race_exit == 3
    assert not asset_race_output.exists()
    assert not any(path.name.startswith(".asset-race-scan.") for path in tmp_path.iterdir())
    asset_path.write_bytes(original_asset_bytes)
