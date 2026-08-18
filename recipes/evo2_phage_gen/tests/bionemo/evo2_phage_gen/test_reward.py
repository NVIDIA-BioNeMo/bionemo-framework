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

"""Tests for ``bionemo.evo2_phage_gen.reward``."""

import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bionemo.evo2_phage_gen import sequence_safety_cli
from bionemo.evo2_phage_gen.design_scope import HostDomain, HostEvidence
from bionemo.evo2_phage_gen.qc import NucleotideQCConfig
from bionemo.evo2_phage_gen.reward import (
    REWARD_COMPONENTS,
    TIMING_COLUMN_PREFIX,
    ExternalQCRewardConfig,
    MMseqsClusterDiversityConfig,
    RewardWeights,
    SequenceSafetyRewardConfig,
    _aai_evidence_score,
    _aai_novelty_score,
    _add_average_protein_identity_rewards,
    _add_full_synteny_rewards,
    _add_mmseqs_hit_rewards,
    _add_required_gene_rewards,
    _aggregate_reward,
    _bounded_range_score,
    _external_qc_env,
    _lower_bound_ratio_score,
    _spike_identity_score,
    _synteny_distance_score,
    _upper_bound_ratio_score,
    _write_external_qc_config,
    score_fasta,
    score_nucleotide_metrics,
)


def _deterministic_dna(length: int) -> str:
    """Build reproducible DNA for dustmask reward tests."""
    rng = random.Random(11)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _bacterial_safety_config(tmp_path: Path, *, enabled: bool = True) -> SequenceSafetyRewardConfig:
    """Build typed synthetic bacterial scan configuration without live safety assets."""
    return SequenceSafetyRewardConfig(
        host_domain=HostDomain.BACTERIA,
        host_evidence=HostEvidence(
            source="synthetic-test-catalog",
            source_version="v1",
            replication_host_domains=frozenset({HostDomain.BACTERIA}),
            confirmed=True,
        ),
        asset_manifest_path=tmp_path / "asset-manifest.yaml",
        diamond_bin=tmp_path / "diamond.json",
        mmseqs_bin=tmp_path / "mmseqs.json",
        work_dir=tmp_path / "safety-work",
        enabled=enabled,
    )


def _archaeal_safety_config(tmp_path: Path, *, strict_lysis: bool) -> SequenceSafetyRewardConfig:
    """Build typed synthetic archaeal scan configuration without live safety assets."""
    return SequenceSafetyRewardConfig(
        host_domain=HostDomain.ARCHAEA,
        host_evidence=HostEvidence(
            source="synthetic-test-catalog",
            source_version="v1",
            replication_host_domains=frozenset({HostDomain.ARCHAEA}),
            confirmed=True,
        ),
        asset_manifest_path=tmp_path / "asset-manifest.yaml",
        diamond_bin=tmp_path / "diamond.json",
        mmseqs_bin=tmp_path / "mmseqs.json",
        work_dir=tmp_path / ("strict-safety-work" if strict_lysis else "informational-safety-work"),
        strict_lysis=strict_lysis,
    )


def _install_synthetic_safety_scan(
    monkeypatch,
    *,
    class_states_by_record: list[dict[str, str]],
    required_by_class: dict[str, bool] | None = None,
    review_classes_by_record: list[set[str]] | None = None,
) -> dict[str, object]:
    """Install a compact synthetic scan at the reward/CLI boundary."""
    required = required_by_class or {"amr": True, "toxin": True, "lysogeny": True}
    review_classes = review_classes_by_record or [set() for _ in class_states_by_record]
    capture: dict[str, object] = {}

    def argv_value(argv: list[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    def fake_main(argv, **_kwargs):
        argv = list(argv)
        capture["argv"] = argv
        input_fasta = Path(argv_value(argv, "--input-fasta"))
        output_dir = Path(argv_value(argv, "--output-dir"))
        headers = [
            line[1:].split(maxsplit=1)[0] for line in input_fasta.read_text().splitlines() if line.startswith(">")
        ]
        capture["input_fasta_bytes"] = input_fasta.read_bytes()
        assert len(headers) == len(class_states_by_record)
        strict_lysis = "--strict-lysis" in argv

        records: list[dict[str, object]] = []
        record_states: list[str] = []
        for input_index, (record_id, states) in enumerate(zip(headers, class_states_by_record, strict=True)):
            class_results: list[dict[str, object]] = []
            attempts: list[dict[str, object]] = []
            reasons: list[str] = []
            for safety_class in ("amr", "toxin", "lysogeny"):
                state = states[safety_class]
                measured_review = safety_class in review_classes[input_index]
                reason = (
                    f"{safety_class.upper()}_MEASURED_NO_HIT"
                    if state == "PASS"
                    else f"{safety_class.upper()}_HIT"
                    if state == "FAIL"
                    else f"{safety_class.upper()}_REVIEW"
                    if measured_review
                    else f"{safety_class.upper()}_TOOL_FAILED"
                )
                findings = (
                    [
                        {
                            "safety_class": safety_class,
                            "state": state,
                            "reason_codes": [reason],
                            "finding_id": f"{safety_class}-{input_index}",
                        }
                    ]
                    if state == "FAIL" or measured_review
                    else []
                )
                reasons.append(reason)
                class_results.append(
                    {
                        "safety_class": safety_class,
                        "state": state,
                        "required": required[safety_class],
                        "findings": findings,
                        "reason_codes": [reason],
                    }
                )
                attempts.append(
                    {
                        "safety_class": safety_class,
                        "execution_status": (
                            "FAILED" if state == "INDETERMINATE" and not measured_review else "COMPLETED_AND_PARSED"
                        ),
                        "policy_id": f"{safety_class}-policy",
                    }
                )
            required_states = [states[name] for name in required if required[name]]
            record_state = (
                "FAIL"
                if "FAIL" in required_states
                else "INDETERMINATE"
                if "INDETERMINATE" in required_states
                else "PASS"
            )
            record_states.append(record_state)
            records.append(
                {
                    "input_index": input_index,
                    "record_id": record_id,
                    "state": record_state,
                    "reason_codes": list(dict.fromkeys(reasons)),
                    "class_results": class_results,
                    "adapter_attempts": attempts,
                }
            )

        batch_state = (
            "FAIL" if "FAIL" in record_states else "INDETERMINATE" if "INDETERMINATE" in record_states else "PASS"
        )
        manifest = {
            "schema_version": 2,
            "manifest_type": "sequence_safety_scan",
            "input": {"path": str(input_fasta), "count": len(records)},
            "policy": {"policy_id": "synthetic-policy"},
            "asset_state": {"path": argv_value(argv, "--asset-manifest")},
            "resolved_profile": {
                "host_domain": argv_value(argv, "--host-domain"),
                "strict_lysis": strict_lysis,
            },
            "tools": {
                "amrfinder": {"path": "/tools/amrfinder", "version": "4.2.7"},
                "diamond": {"path": argv_value(argv, "--diamond-bin"), "version": "2.1.24"},
                "mmseqs": {"path": argv_value(argv, "--mmseqs-bin"), "version": "18"},
            },
            "databases": {},
            "records": records,
            "aggregate": {
                "state": batch_state,
                "counts": {state: record_states.count(state) for state in ("PASS", "FAIL", "INDETERMINATE")},
            },
        }
        output_dir.mkdir(parents=True)
        (output_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")
        capture["manifest"] = manifest
        return {"PASS": 0, "FAIL": 2, "INDETERMINATE": 3}[batch_state]

    def fake_validate_manifest_file(path, *, expected_type=None, **_kwargs):
        payload = json.loads(Path(path).read_text())
        if expected_type is not None and payload["manifest_type"] != expected_type:
            raise sequence_safety_cli.CLIValidationError(f"expected {expected_type} manifest")
        capture["validated_manifest_path"] = Path(path)
        return payload

    monkeypatch.setattr(sequence_safety_cli, "main", fake_main)
    monkeypatch.setattr(sequence_safety_cli, "validate_manifest_file", fake_validate_manifest_file)
    return capture


def test_sequence_safety_reward_fields_partial_credit_requires_explicit_review_evidence():
    from bionemo.evo2_phage_gen.reward import sequence_safety_reward_fields

    class_states = {"amr": "PASS", "toxin": "INDETERMINATE", "lysogeny": "PASS"}
    required = {"amr": True, "toxin": True, "lysogeny": True}

    review = sequence_safety_reward_fields(
        class_states=class_states,
        required_by_class=required,
        review_eligible_by_class={"amr": False, "toxin": True, "lysogeny": False},
    )
    unavailable = sequence_safety_reward_fields(
        class_states=class_states,
        required_by_class=required,
        review_eligible_by_class={"amr": False, "toxin": False, "lysogeny": False},
    )

    assert review["reward_safety_toxin"] == 0.25
    assert unavailable["reward_safety_toxin"] == 0.0
    assert review["safety_gate_state"] == "INDETERMINATE"
    assert review["safety_gate_pass"] == 0.0
    assert review["reward_safety_penalty"] == 1.0


def test_measured_review_gets_bounded_class_credit_but_cannot_pass_the_safety_gate(tmp_path, monkeypatch):
    _install_synthetic_safety_scan(
        monkeypatch,
        class_states_by_record=[{"amr": "PASS", "toxin": "INDETERMINATE", "lysogeny": "PASS"}],
        review_classes_by_record=[{"toxin"}],
    )

    scored = score_nucleotide_metrics(
        pd.DataFrame({"id_prompt": ["review"], "sequence": ["ACGT" * 1000]}),
        sequence_safety=_bacterial_safety_config(tmp_path),
    )

    assert scored["safety_toxin_measurement_available"].tolist() == [1.0]
    assert scored["safety_toxin_finding_count"].tolist() == [1]
    assert scored["reward_safety_toxin"].tolist() == [0.25]
    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_gate_pass"].tolist() == [0.0]
    assert scored["reward_safety_penalty"].tolist() == [1.0]
    assert scored["reward"].tolist() == [0.0]


def test_score_nucleotide_metrics_rewards_passing_sequence():
    """Missing mandatory sequence-safety evidence must zero eligibility but retain the historical score."""
    df = pd.DataFrame({"id_prompt": ["pass"], "sequence": ["ACGT" * 1000]})

    scored = score_nucleotide_metrics(df)

    assert scored.loc[0, "reward_historical"] == 1.0
    assert scored.loc[0, "reward_safety_penalty"] == 1.0
    assert scored.loc[0, "reward"] == 0.0
    assert scored.loc[0, "reward_valid_nt_chars"] == 1.0
    assert scored.loc[0, "safety_gate_state"] == "INDETERMINATE"
    assert scored.loc[0, "safety_gate_pass"] == 0.0
    assert scored.loc[0, "safety_environment_healthy"] == 0.0
    assert scored.loc[0, "safety_gate_reason_codes"] == '["SEQUENCE_SAFETY_CONFIG_MISSING"]'
    assert scored.loc[0, "reward_binary_historical_core_pass"] == 1.0
    assert scored.loc[0, "reward_binary_core_pass"] == 0.0
    for safety_class in ("amr", "toxin", "lysogeny"):
        assert scored.loc[0, f"safety_{safety_class}_state"] == "INDETERMINATE"
        assert scored.loc[0, f"safety_{safety_class}_required"] == 1.0
        assert scored.loc[0, f"safety_{safety_class}_measurement_available"] == 0.0
        assert scored.loc[0, f"reward_safety_{safety_class}"] == 0.0


def test_disabled_sequence_safety_config_is_explicitly_indeterminate(tmp_path):
    """Disabling the mandatory scanner must not restore historical reward or eligibility."""
    scored = score_nucleotide_metrics(
        pd.DataFrame({"id_prompt": ["disabled"], "sequence": ["ACGT" * 1000]}),
        sequence_safety=_bacterial_safety_config(tmp_path, enabled=False),
    )

    assert scored["reward_historical"].tolist() == [1.0]
    assert scored["reward"].tolist() == [0.0]
    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_gate_reason_codes"].tolist() == ['["SEQUENCE_SAFETY_DISABLED"]']


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("host_domain", "BACTERIA"),
        ("host_evidence", {}),
        ("asset_manifest_path", "asset-manifest.yaml"),
        ("diamond_bin", "diamond.json"),
        ("mmseqs_bin", "mmseqs.json"),
        ("policy_path", "phage-safety-policy.yaml"),
        ("work_dir", "safety-work"),
        ("enabled", "false"),
        ("strict_lysis", 1),
        ("circular", 1),
        ("threads", True),
        ("threads", 0),
        ("timeout_seconds", "300"),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", 0.0),
    ],
)
def test_malformed_sequence_safety_config_is_explicitly_indeterminate(
    tmp_path,
    monkeypatch,
    field_name,
    invalid_value,
):
    """Malformed runtime configuration must produce no acceptance credit before Task 4 runs."""
    malformed = replace(_bacterial_safety_config(tmp_path), **{field_name: invalid_value})

    def unexpected_scan(_argv, *, runtime=None):
        pytest.fail(f"malformed sequence-safety config reached scanner with runtime={runtime!r}")

    monkeypatch.setattr(sequence_safety_cli, "main", unexpected_scan)

    scored = score_nucleotide_metrics(
        pd.DataFrame({"id_prompt": ["malformed-config"], "sequence": ["ACGT" * 1000]}),
        sequence_safety=malformed,
    )

    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_gate_pass"].tolist() == [0.0]
    assert scored["safety_gate_reason_codes"].tolist() == ['["SEQUENCE_SAFETY_CONFIG_INVALID"]']
    assert scored["reward_safety_penalty"].tolist() == [1.0]
    assert scored["reward"].tolist() == [0.0]
    assert scored["reward_binary_core_pass"].tolist() == [0.0]
    assert scored["safety_strict_lysis"].tolist() == [False]


def test_disabled_archaeal_config_keeps_lysogeny_informational_but_gate_ineligible(tmp_path):
    """Unavailable informational lysogeny is neutral, while missing required AMR/toxin still blocks reward."""
    config = _archaeal_safety_config(tmp_path, strict_lysis=False)
    config = SequenceSafetyRewardConfig(
        host_domain=config.host_domain,
        host_evidence=config.host_evidence,
        asset_manifest_path=config.asset_manifest_path,
        diamond_bin=config.diamond_bin,
        mmseqs_bin=config.mmseqs_bin,
        work_dir=config.work_dir,
        enabled=False,
        strict_lysis=False,
    )

    scored = score_nucleotide_metrics(
        pd.DataFrame({"id_prompt": ["archaeal-disabled"], "sequence": ["ACGT" * 1000]}),
        sequence_safety=config,
    )

    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_required_class_count"].tolist() == [2]
    assert scored["safety_lysogeny_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_lysogeny_required"].tolist() == [0.0]
    assert scored["reward_safety_lysogeny"].tolist() == [1.0]
    assert scored["reward"].tolist() == [0.0]


def test_valid_unavailable_config_preserves_strict_lysis_telemetry(tmp_path):
    """Unavailable-scan records should retain a structurally valid strict-lysis request."""
    config = replace(_archaeal_safety_config(tmp_path, strict_lysis=True), enabled=False)

    scored = score_nucleotide_metrics(
        pd.DataFrame({"id_prompt": ["archaeal-strict-disabled"], "sequence": ["ACGT" * 1000]}),
        sequence_safety=config,
    )

    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_gate_reason_codes"].tolist() == ['["SEQUENCE_SAFETY_DISABLED"]']
    assert scored["safety_strict_lysis"].tolist() == [True]
    assert scored["safety_lysogeny_required"].tolist() == [1.0]
    assert scored["reward"].tolist() == [0.0]


def test_clean_scan_emits_independent_safety_rewards(tmp_path, monkeypatch):
    """An all-PASS scan qualifies the ordinary whole-genome reward."""
    capture = _install_synthetic_safety_scan(
        monkeypatch,
        class_states_by_record=[{"amr": "PASS", "toxin": "PASS", "lysogeny": "PASS"}],
    )
    source = pd.DataFrame({"id_prompt": ["original-id"], "sequence": ["ACGT" * 1000]})

    config = replace(
        _bacterial_safety_config(tmp_path),
        threads=7,
        batch_size=5,
        orf_workers=3,
        phrogs_threads=11,
    )
    scored = score_nucleotide_metrics(source, sequence_safety=config)

    assert source.to_dict("records") == [{"id_prompt": "original-id", "sequence": "ACGT" * 1000}]
    assert capture["input_fasta_bytes"].startswith(b">safety_record_000000\n")
    argv = capture["argv"]
    assert argv[argv.index("--threads") + 1] == "7"
    assert argv[argv.index("--batch-size") + 1] == "5"
    assert argv[argv.index("--orf-workers") + 1] == "3"
    assert argv[argv.index("--phrogs-threads") + 1] == "11"
    assert scored["safety_scan_record_id"].tolist() == ["safety_record_000000"]
    assert scored["safety_gate_state"].tolist() == ["PASS"]
    assert scored["safety_environment_healthy"].tolist() == [1.0]
    assert scored["safety_required_class_pass_count"].tolist() == [3]
    assert scored["reward"].tolist() == [1.0]
    assert scored["reward_binary_core_pass"].tolist() == [1.0]
    for safety_class in ("amr", "toxin", "lysogeny"):
        assert scored[f"safety_{safety_class}_state"].tolist() == ["PASS"]
        assert scored[f"safety_{safety_class}_measurement_available"].tolist() == [1.0]
        assert scored[f"reward_safety_{safety_class}"].tolist() == [1.0]
    assert scored["safety_policy_id"].tolist() == ["synthetic-policy"]
    assert Path(scored.loc[0, "safety_scan_manifest_path"]) == capture["validated_manifest_path"]
    assert {"safety_amr", "safety_toxin", "safety_lysogeny"}.issubset(
        {component.name for component in REWARD_COMPONENTS}
    )


def test_mixed_batch_isolates_each_required_failure_and_unscannable_record(tmp_path, monkeypatch):
    """One unsafe or malformed generation must not contaminate another record's manifest-backed result."""
    _install_synthetic_safety_scan(
        monkeypatch,
        class_states_by_record=[
            {"amr": "PASS", "toxin": "PASS", "lysogeny": "PASS"},
            {"amr": "FAIL", "toxin": "PASS", "lysogeny": "PASS"},
            {"amr": "PASS", "toxin": "FAIL", "lysogeny": "PASS"},
            {"amr": "PASS", "toxin": "PASS", "lysogeny": "FAIL"},
            {"amr": "PASS", "toxin": "INDETERMINATE", "lysogeny": "PASS"},
        ],
    )
    scored = score_nucleotide_metrics(
        pd.DataFrame(
            {
                "id_prompt": ["clean", "amr-hit", "toxin-hit", "lysogeny-hit", "detector-failed", "malformed"],
                "sequence": ["ACGT" * 1000] * 5 + ["ACGT!"],
            }
        ),
        sequence_safety=_bacterial_safety_config(tmp_path),
    )

    assert scored["safety_scan_record_id"].tolist() == [
        "safety_record_000000",
        "safety_record_000001",
        "safety_record_000002",
        "safety_record_000003",
        "safety_record_000004",
        "",
    ]
    assert scored["safety_gate_state"].tolist() == [
        "PASS",
        "FAIL",
        "FAIL",
        "FAIL",
        "INDETERMINATE",
        "INDETERMINATE",
    ]
    assert scored["reward_safety_amr"].tolist() == [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert scored["reward_safety_toxin"].tolist() == [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    assert scored["reward_safety_lysogeny"].tolist() == [1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    assert scored["safety_amr_finding_count"].tolist() == [0, 1, 0, 0, 0, 0]
    assert scored["safety_toxin_finding_count"].tolist() == [0, 0, 1, 0, 0, 0]
    assert scored["safety_lysogeny_finding_count"].tolist() == [0, 0, 0, 1, 0, 0]
    assert scored["safety_environment_healthy"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert scored["reward_safety_penalty"].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert scored["reward"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert scored.loc[4, "safety_toxin_reason_codes"] == '["TOXIN_TOOL_FAILED"]'
    assert scored.loc[5, "safety_gate_reason_codes"] == '["SEQUENCE_SAFETY_RECORD_UNSCANNABLE"]'


def test_invalid_safety_result_has_zero_credit(tmp_path, monkeypatch):
    """A rejected safety result must not award credit."""
    _install_synthetic_safety_scan(
        monkeypatch,
        class_states_by_record=[{"amr": "PASS", "toxin": "PASS", "lysogeny": "PASS"}],
    )

    def reject_manifest(*_args, **_kwargs):
        raise sequence_safety_cli.CLIValidationError("synthetic result mismatch")

    monkeypatch.setattr(sequence_safety_cli, "validate_manifest_file", reject_manifest)

    scored = score_nucleotide_metrics(
        pd.DataFrame({"id_prompt": ["untrusted"], "sequence": ["ACGT" * 1000]}),
        sequence_safety=_bacterial_safety_config(tmp_path),
    )

    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_gate_pass"].tolist() == [0.0]
    assert scored["safety_gate_reason_codes"].tolist() == ['["SEQUENCE_SAFETY_MANIFEST_REJECTED"]']
    assert scored["reward_safety_amr"].tolist() == [0.0]
    assert scored["reward_safety_toxin"].tolist() == [0.0]
    assert scored["reward_safety_lysogeny"].tolist() == [0.0]
    assert scored["reward_historical"].tolist() == [1.0]
    assert scored["reward"].tolist() == [0.0]
    assert scored["safety_scan_manifest_path"].tolist() == [""]


def test_diagnostic_result_has_zero_credit(tmp_path, monkeypatch):
    """A diagnostic summary is not per-record safety evidence."""
    capture = _install_synthetic_safety_scan(
        monkeypatch,
        class_states_by_record=[{"amr": "PASS", "toxin": "PASS", "lysogeny": "PASS"}],
    )
    write_scan = sequence_safety_cli.main

    def write_diagnostic(argv, *, runtime=None):
        write_scan(argv, runtime=runtime)
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        manifest_path = output_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text())
        payload["manifest_type"] = "sequence_safety_diagnostic"
        manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        capture["manifest"] = payload
        return 3

    monkeypatch.setattr(sequence_safety_cli, "main", write_diagnostic)

    scored = score_nucleotide_metrics(
        pd.DataFrame({"id_prompt": ["diagnostic-only"], "sequence": ["ACGT" * 1000]}),
        sequence_safety=_bacterial_safety_config(tmp_path),
    )

    assert capture["manifest"]["manifest_type"] == "sequence_safety_diagnostic"
    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_gate_pass"].tolist() == [0.0]
    assert scored["safety_gate_reason_codes"].tolist() == ['["SEQUENCE_SAFETY_MANIFEST_REJECTED"]']
    assert scored["reward"].tolist() == [0.0]


def test_archaeal_lysogeny_retains_raw_state_but_is_neutral_unless_strict(tmp_path, monkeypatch):
    """Informational archaeal lysogeny must stay visible without penalizing an otherwise required-class PASS."""
    states = [{"amr": "PASS", "toxin": "PASS", "lysogeny": "FAIL"}]
    _install_synthetic_safety_scan(
        monkeypatch,
        class_states_by_record=states,
        required_by_class={"amr": True, "toxin": True, "lysogeny": False},
    )

    informational = score_nucleotide_metrics(
        pd.DataFrame({"id_prompt": ["archaeal"], "sequence": ["ACGT" * 1000]}),
        sequence_safety=_archaeal_safety_config(tmp_path, strict_lysis=False),
    )

    assert informational["safety_resolved_profile"].tolist() == ["ARCHAEA"]
    assert informational["safety_strict_lysis"].tolist() == [False]
    assert informational["safety_lysogeny_state"].tolist() == ["FAIL"]
    assert informational["safety_lysogeny_required"].tolist() == [0.0]
    assert informational["safety_lysogeny_finding_count"].tolist() == [1]
    assert informational["reward_safety_lysogeny"].tolist() == [1.0]
    assert informational["safety_required_class_count"].tolist() == [2]
    assert informational["safety_gate_state"].tolist() == ["PASS"]
    assert informational["reward"].tolist() == [1.0]

    strict_capture = _install_synthetic_safety_scan(
        monkeypatch,
        class_states_by_record=states,
        required_by_class={"amr": True, "toxin": True, "lysogeny": True},
    )
    strict = score_nucleotide_metrics(
        pd.DataFrame({"id_prompt": ["archaeal-strict"], "sequence": ["ACGT" * 1000]}),
        sequence_safety=_archaeal_safety_config(tmp_path, strict_lysis=True),
    )

    assert "--strict-lysis" in strict_capture["argv"]
    assert strict["safety_lysogeny_state"].tolist() == ["FAIL"]
    assert strict["safety_lysogeny_required"].tolist() == [1.0]
    assert strict["reward_safety_lysogeny"].tolist() == [0.0]
    assert strict["safety_gate_state"].tolist() == ["FAIL"]
    assert strict["reward"].tolist() == [0.0]


def test_mismatched_record_mapping_rejects_batch(tmp_path, monkeypatch):
    """A mismatched later index must invalidate the complete measured batch."""
    _install_synthetic_safety_scan(
        monkeypatch,
        class_states_by_record=[
            {"amr": "PASS", "toxin": "PASS", "lysogeny": "PASS"},
            {"amr": "PASS", "toxin": "PASS", "lysogeny": "PASS"},
        ],
    )
    validate = sequence_safety_cli.validate_manifest_file

    def return_mismatched_result(*args, **kwargs):
        payload = validate(*args, **kwargs)
        payload["records"][1]["input_index"] = 0
        return payload

    monkeypatch.setattr(sequence_safety_cli, "validate_manifest_file", return_mismatched_result)

    scored = score_nucleotide_metrics(
        pd.DataFrame(
            {
                "id_prompt": ["first", "second"],
                "sequence": ["ACGT" * 1000, "TGCA" * 1000],
            }
        ),
        sequence_safety=_bacterial_safety_config(tmp_path),
    )

    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE", "INDETERMINATE"]
    assert scored["safety_gate_pass"].tolist() == [0.0, 0.0]
    assert scored["safety_gate_reason_codes"].tolist() == [
        '["SEQUENCE_SAFETY_MANIFEST_REJECTED"]',
        '["SEQUENCE_SAFETY_MANIFEST_REJECTED"]',
    ]
    assert scored["reward"].tolist() == [0.0, 0.0]
    assert scored["safety_scan_manifest_path"].tolist() == ["", ""]


def test_inconsistent_class_state_has_zero_credit(tmp_path, monkeypatch):
    """A class result inconsistent with its aggregate must not award credit."""
    _install_synthetic_safety_scan(
        monkeypatch,
        class_states_by_record=[{"amr": "PASS", "toxin": "PASS", "lysogeny": "PASS"}],
    )
    validate = sequence_safety_cli.validate_manifest_file

    def return_mismatched_result(*args, **kwargs):
        payload = validate(*args, **kwargs)
        payload["records"][0]["class_results"][0]["state"] = "FAIL"
        return payload

    monkeypatch.setattr(sequence_safety_cli, "validate_manifest_file", return_mismatched_result)

    scored = score_nucleotide_metrics(
        pd.DataFrame({"id_prompt": ["state-drift"], "sequence": ["ACGT" * 1000]}),
        sequence_safety=_bacterial_safety_config(tmp_path),
    )

    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_gate_pass"].tolist() == [0.0]
    assert scored["safety_gate_reason_codes"].tolist() == ['["SEQUENCE_SAFETY_MANIFEST_REJECTED"]']
    assert scored["reward"].tolist() == [0.0]


def test_score_nucleotide_metrics_reports_reward_timing_columns():
    """Reward timing columns should be present for NeMo-RL timing metric routing."""
    df = pd.DataFrame({"id_prompt": ["pass"], "sequence": ["ACGT" * 1000]})

    scored = score_nucleotide_metrics(df)

    begin = scored.loc[0, f"{TIMING_COLUMN_PREFIX}reward/begin_unix_s"]
    end = scored.loc[0, f"{TIMING_COLUMN_PREFIX}reward/end_unix_s"]
    assert end >= begin
    assert scored.loc[0, f"{TIMING_COLUMN_PREFIX}reward/total_s"] >= 0.0
    assert scored.loc[0, f"{TIMING_COLUMN_PREFIX}reward/nucleotide_qc_s"] >= 0.0
    assert scored.loc[0, f"{TIMING_COLUMN_PREFIX}reward/nucleotide_reward_scores_s"] >= 0.0
    assert scored.loc[0, f"{TIMING_COLUMN_PREFIX}reward/aggregate_s"] >= 0.0


def test_score_nucleotide_metrics_penalizes_invalid_sequence():
    """Invalid characters and out-of-range metrics should reduce the reward."""
    df = pd.DataFrame({"id_prompt": ["bad"], "sequence": ["NNNN"]})

    scored = score_nucleotide_metrics(df)

    assert scored.loc[0, "reward_historical"] < 1.0
    assert scored.loc[0, "reward"] == 0.0
    assert scored.loc[0, "reward_valid_nt_chars"] == 0.0


def test_score_nucleotide_metrics_homopolymer_reward_stays_dense():
    """Oversized homopolymers should retain an optimization signal instead of saturating at zero."""
    df = pd.DataFrame(
        {
            "id_prompt": ["short_run", "long_run"],
            "sequence": ["A" * 20 + "CGT" * 1500, "A" * 2000 + "CGT" * 1000],
        }
    )

    scored = score_nucleotide_metrics(df)

    assert 0.0 < scored.loc[1, "reward_nt_homopolymer"] < scored.loc[0, "reward_nt_homopolymer"] < 1.0


def test_score_nucleotide_metrics_can_weight_nucleotide_pass_bonus():
    """A pass-gated term should increase pressure on satisfying all online nucleotide filters."""
    df = pd.DataFrame(
        {
            "id_prompt": ["pass", "long_homopolymer"],
            "sequence": ["ACGT" * 1000, "A" * 200 + "CGT" * 1267],
        }
    )

    scored = score_nucleotide_metrics(
        df,
        weights=RewardWeights(
            valid_nt_chars=0.0,
            genome_length=0.0,
            gc_content=0.0,
            nt_homopolymer=0.0,
            nucleotide_pass=1.0,
        ),
    )

    assert scored.loc[0, "reward_nucleotide_pass"] == 1.0
    assert scored.loc[1, "reward_nucleotide_pass"] == 0.0
    assert scored["reward_historical"].tolist() == [1.0, 0.0]
    assert scored["reward"].tolist() == [0.0, 0.0]


def test_score_nucleotide_metrics_penalizes_low_complexity_sequence_ends():
    """The dustmask reward should give low-complexity sequence ends less credit."""
    good_sequence = _deterministic_dna(4200)
    bad_sequence = _deterministic_dna(4000) + "A" * 200
    df = pd.DataFrame({"id_prompt": ["good", "bad_tail"], "sequence": [good_sequence, bad_sequence]})

    scored = score_nucleotide_metrics(
        df,
        config=NucleotideQCConfig(
            dustmask_filter=True,
            dustmask_use_external=False,
            dustmask_window=64,
            dustmask_level=20.0,
            dustmask_end_window=200,
            dustmask_max_end_fraction=0.9,
        ),
        weights=RewardWeights(
            valid_nt_chars=0.0,
            genome_length=0.0,
            gc_content=0.0,
            nt_homopolymer=0.0,
            dustmask_end=1.0,
        ),
    )

    assert scored.loc[0, "reward_dustmask_end"] > scored.loc[1, "reward_dustmask_end"]
    assert scored.loc[1, "dustmask_max_end_masked_fraction"] > 0.9
    assert scored.loc[1, "reward_nucleotide_pass"] == 0.0
    assert scored["reward_historical"].tolist() == scored["reward_dustmask_end"].tolist()
    assert scored["reward"].tolist() == [0.0, 0.0]


def test_reward_components_are_registered_and_clipped_to_unit_interval():
    """The aggregate RL score should be easy to reweight and stay in [0, 1]."""
    component_names = {component.name for component in REWARD_COMPONENTS}
    assert {"valid_nt_chars", "genome_length", "gc_content", "protein_hit_count", "tropism"}.issubset(component_names)
    assert "mmseqs_cluster_diversity" in component_names
    assert "dustmask_end" in component_names
    removed_components = {
        "checkv",
        "training_data_identity",
        "reference_genome_identity",
        "mmseqs_clustering",
        "diversity",
    }
    assert removed_components.isdisjoint(component_names)

    df = pd.DataFrame(
        {
            "reward_valid_nt_chars": [2.0, -1.0],
            "reward_gc_content": [0.5, 0.25],
        }
    )
    scored = _aggregate_reward(
        df,
        RewardWeights(valid_nt_chars=1.0, genome_length=0.0, gc_content=1.0, nt_homopolymer=0.0),
    )

    assert scored["reward_valid_nt_chars"].tolist() == [1.0, 0.0]
    assert scored["reward_historical"].tolist() == [0.75, 0.125]
    assert scored["reward"].tolist() == [0.0, 0.0]
    assert scored["reward_binary_core_pass"].tolist() == [0.0, 0.0]
    assert scored["reward_active_components"].tolist() == ["valid_nt_chars,gc_content"] * 2


def test_only_exact_safety_gate_pass_one_can_qualify_reward():
    """Only a numeric scalar one may satisfy the binary safety eligibility rule."""
    scored = _aggregate_reward(
        pd.DataFrame(
            {
                "reward_valid_nt_chars": [1.0] * 5,
                "safety_gate_pass": [1.0, 2.0, True, "1", "1.0"],
            }
        ),
        RewardWeights(valid_nt_chars=1.0, genome_length=0.0, gc_content=0.0, nt_homopolymer=0.0),
    )

    assert scored["reward_historical"].tolist() == [1.0] * 5
    assert scored["reward_safety_penalty"].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert scored["reward"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert scored["reward_binary_core_pass"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_cluster_deduplication_uses_row_positions_with_duplicate_dataframe_indexes():
    """Duplicate caller labels must not select every row in one passing cluster."""
    scored = _aggregate_reward(
        pd.DataFrame(
            {
                "reward_valid_nt_chars": [1.0, 1.0],
                "safety_gate_pass": [1.0, 1.0],
                "mmseqs_cluster_id": ["shared-cluster", "shared-cluster"],
                "mmseqs_cluster_size": [2, 2],
                "mmseqs_cluster_valid_for_clustering": [1.0, 1.0],
            },
            index=[7, 7],
        ),
        RewardWeights(valid_nt_chars=1.0, genome_length=0.0, gc_content=0.0, nt_homopolymer=0.0),
    )

    assert scored["reward_binary_historical_core_pass"].tolist() == [1.0, 1.0]
    assert scored["reward_binary_core_pass"].tolist() == [1.0, 1.0]
    assert scored["reward_binary_historical_core_cluster_deduplicated_pass"].tolist() == [1.0, 0.0]
    assert scored["reward_binary_core_cluster_deduplicated_pass"].tolist() == [1.0, 0.0]


def test_mmseqs_cluster_diversity_reward_uses_inverse_cluster_size(tmp_path, monkeypatch):
    """Batch-local MMseqs clusters should give each valid member 1 / cluster_size."""
    commands = []

    def fake_run(args, check):
        commands.append(args)
        assert check is True
        assert args[:2] == ["fake-mmseqs", "easy-cluster"]
        assert args[args.index("--min-seq-id") + 1] == "0.99"
        assert args[args.index("-c") + 1] == "0"
        assert args[args.index("--cov-mode") + 1] == "0"
        assert args[args.index("--seq-id-mode") + 1] == "0"
        assert args[args.index("--cluster-mode") + 1] == "0"
        assert args[args.index("-v") + 1] == "0"
        Path(f"{args[3]}_cluster.tsv").write_text("seq_0\tseq_0\nseq_0\tseq_1\nseq_2\tseq_2\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    df = pd.DataFrame(
        {
            "id_prompt": ["seq0", "seq1", "seq2", "invalid"],
            "sequence": ["ACGT" * 1000, "ACGT" * 1000, "TGCA" * 1000, "NNNN"],
        }
    )

    scored = score_nucleotide_metrics(
        df,
        weights=RewardWeights(
            valid_nt_chars=0.0,
            genome_length=0.0,
            gc_content=0.0,
            nt_homopolymer=0.0,
            mmseqs_cluster_diversity=1.0,
        ),
        mmseqs_cluster_diversity=MMseqsClusterDiversityConfig(
            enabled=True,
            mmseqs_bin="fake-mmseqs",
            work_dir=tmp_path,
        ),
    )

    assert len(commands) == 1
    assert scored["reward_mmseqs_cluster_diversity"].tolist() == [0.5, 0.5, 1.0, 0.0]
    assert scored["reward_historical"].tolist() == [0.5, 0.5, 1.0, 0.0]
    assert scored["reward"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert scored["mmseqs_cluster_size"].tolist() == [2, 2, 1, 0]
    assert scored["mmseqs_cluster_valid_for_clustering"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert scored["mmseqs_cluster_missing_from_output"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_mmseqs_cluster_diversity_missing_output_gets_zero_reward(tmp_path, monkeypatch):
    """A valid sequence omitted from MMseqs output should get no cluster-diversity credit."""
    commands = []

    def fake_run(args, check):
        commands.append(args)
        assert check is True
        Path(f"{args[3]}_cluster.tsv").write_text("seq_0\tseq_0\nseq_0\tseq_1\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    df = pd.DataFrame(
        {
            "id_prompt": ["seq0", "seq1", "seq2"],
            "sequence": ["ACGT" * 1000, "TGCA" * 1000, "GTAC" * 1000],
        }
    )

    scored = score_nucleotide_metrics(
        df,
        weights=RewardWeights(
            valid_nt_chars=0.0,
            genome_length=0.0,
            gc_content=0.0,
            nt_homopolymer=0.0,
            mmseqs_cluster_diversity=1.0,
        ),
        mmseqs_cluster_diversity=MMseqsClusterDiversityConfig(
            enabled=True,
            mmseqs_bin="fake-mmseqs",
            work_dir=tmp_path,
        ),
    )

    assert len(commands) == 1
    assert scored["reward_mmseqs_cluster_diversity"].tolist() == [0.5, 0.5, 0.0]
    assert scored["reward_historical"].tolist() == [0.5, 0.5, 0.0]
    assert scored["reward"].tolist() == [0.0, 0.0, 0.0]
    assert scored["mmseqs_cluster_id"].tolist() == ["group0:seq_0", "group0:seq_0", ""]
    assert scored["mmseqs_cluster_size"].tolist() == [2, 2, 0]
    assert scored["mmseqs_cluster_missing_from_output"].tolist() == [0.0, 0.0, 1.0]
    assert scored["mmseqs_cluster_num_missing_from_output"].tolist() == [1, 1, 1]


def test_mmseqs_cluster_diversity_singleton_group_is_not_missing(tmp_path, monkeypatch):
    """A singleton prompt group should be a valid one-member cluster, not missing MMseqs output."""

    def fake_run(*_args, **_kwargs):
        raise AssertionError("singleton groups should not invoke MMseqs")

    monkeypatch.setattr("subprocess.run", fake_run)
    df = pd.DataFrame(
        {
            "id_prompt": ["seq0"],
            "prompt_group": ["prompt-a"],
            "sequence": ["ACGT" * 1000],
        }
    )

    scored = score_nucleotide_metrics(
        df,
        weights=RewardWeights(
            valid_nt_chars=0.0,
            genome_length=0.0,
            gc_content=0.0,
            nt_homopolymer=0.0,
            mmseqs_cluster_diversity=1.0,
        ),
        mmseqs_cluster_diversity=MMseqsClusterDiversityConfig(
            enabled=True,
            mmseqs_bin="fake-mmseqs",
            work_dir=tmp_path,
        ),
    )

    assert scored["reward_mmseqs_cluster_diversity"].tolist() == [1.0]
    assert scored["reward_historical"].tolist() == [1.0]
    assert scored["reward"].tolist() == [0.0]
    assert scored["mmseqs_cluster_id"].tolist() == ["group0:seq_0"]
    assert scored["mmseqs_cluster_size"].tolist() == [1]
    assert scored["mmseqs_cluster_num_clusters"].tolist() == [1]
    assert scored["mmseqs_cluster_num_missing_from_output"].tolist() == [0]
    assert scored["mmseqs_cluster_missing_from_output"].tolist() == [0.0]


def test_mmseqs_cluster_diversity_is_prompt_group_local(tmp_path, monkeypatch):
    """Cluster sizes and inverse-size rewards should be computed within each prompt group."""
    commands = []

    def fake_run(args, check):
        commands.append(args)
        assert check is True
        result_prefix = Path(args[3])
        if result_prefix.parent.name == "prompt_group_0000":
            Path(f"{result_prefix}_cluster.tsv").write_text("seq_0\tseq_0\nseq_0\tseq_1\n")
        elif result_prefix.parent.name == "prompt_group_0001":
            Path(f"{result_prefix}_cluster.tsv").write_text("seq_0\tseq_0\nseq_1\tseq_1\n")
        else:
            raise AssertionError(f"unexpected prompt group directory: {result_prefix.parent}")

    monkeypatch.setattr("subprocess.run", fake_run)
    df = pd.DataFrame(
        {
            "id_prompt": ["a0", "a1", "b0", "b1"],
            "prompt_group": ["prompt-a", "prompt-a", "prompt-b", "prompt-b"],
            "sequence": ["ACGT" * 1000, "TGCA" * 1000, "GTAC" * 1000, "CAGT" * 1000],
        }
    )

    scored = score_nucleotide_metrics(
        df,
        weights=RewardWeights(
            valid_nt_chars=0.0,
            genome_length=0.0,
            gc_content=0.0,
            nt_homopolymer=0.0,
            mmseqs_cluster_diversity=1.0,
        ),
        mmseqs_cluster_diversity=MMseqsClusterDiversityConfig(
            enabled=True,
            mmseqs_bin="fake-mmseqs",
            work_dir=tmp_path,
        ),
    )

    assert len(commands) == 2
    assert scored["reward_mmseqs_cluster_diversity"].tolist() == [0.5, 0.5, 1.0, 1.0]
    assert scored["reward_historical"].tolist() == [0.5, 0.5, 1.0, 1.0]
    assert scored["reward"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert scored["mmseqs_cluster_id"].tolist() == [
        "group0:seq_0",
        "group0:seq_0",
        "group1:seq_0",
        "group1:seq_1",
    ]
    assert scored["mmseqs_cluster_size"].tolist() == [2, 2, 1, 1]
    assert scored["mmseqs_cluster_num_clusters"].tolist() == [3, 3, 3, 3]
    assert scored["mmseqs_cluster_num_missing_from_output"].tolist() == [0, 0, 0, 0]


def test_threshold_reward_helpers_plateau_at_pass_criteria():
    """Continuous threshold scores should not prefer over-matching acceptable criteria."""
    assert _lower_bound_ratio_score(7, 7) == 1.0
    assert _lower_bound_ratio_score(8, 7) == 1.0
    assert _lower_bound_ratio_score(20, 7) == 1.0
    assert _lower_bound_ratio_score(3.5, 7) == 0.5

    assert _upper_bound_ratio_score(10, 10) == 1.0
    assert _upper_bound_ratio_score(8, 10) == 1.0
    assert _upper_bound_ratio_score(20, 10) == 0.5

    assert _bounded_range_score(7, 7, 9) == 1.0
    assert _bounded_range_score(8, 7, 9) == 1.0
    assert _bounded_range_score(9, 7, 9) == 1.0
    assert _bounded_range_score(3.5, 7, 9) == 0.5
    assert _bounded_range_score(18, 7, 9) == 0.5


def test_spike_identity_score_plateaus_at_paper_threshold():
    """Spike/tropism score should stop increasing after the paper threshold."""
    assert _spike_identity_score(95.0, measured_hit=False) == 0.0
    assert _spike_identity_score(0.0, measured_hit=True) == 0.0
    assert _spike_identity_score(30.0, measured_hit=True) == 0.5
    assert _spike_identity_score(60.0, measured_hit=True) == 1.0
    assert _spike_identity_score(95.0, measured_hit=True) == 1.0


def test_soft_preference_components_do_not_gate_binary_pass():
    """Global pass should not reject known-viable-like designs for soft novelty preferences."""
    df = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0],
            "reward_external_synteny": [0.0],
            "reward_external_average_protein_identity": [0.0],
        }
    )

    scored = _aggregate_reward(
        df,
        RewardWeights(valid_nt_chars=1.0, synteny=1.0, average_protein_identity=1.0),
    )

    assert scored["reward_binary_historical_core_pass"].tolist() == [1.0]
    assert scored["reward_binary_core_pass"].tolist() == [0.0]


def test_score_fasta_writes_reward_csv(tmp_path, monkeypatch):
    """The FASTA scorer must forward Task 4 safety configuration into its CSV diagnostics."""
    input_fasta = tmp_path / "input.fasta"
    output_csv = tmp_path / "rewards.csv"
    input_fasta.write_text(">seq1\n" + "ACGT" * 1000 + "\n")
    _install_synthetic_safety_scan(
        monkeypatch,
        class_states_by_record=[{"amr": "PASS", "toxin": "PASS", "lysogeny": "PASS"}],
    )

    score_fasta(
        input_fasta,
        output_csv,
        sequence_safety=_bacterial_safety_config(tmp_path),
    )

    scored = pd.read_csv(output_csv)
    assert scored["reward"].tolist() == [1.0]
    assert scored["reward_historical"].tolist() == [1.0]
    assert scored["safety_gate_state"].tolist() == ["PASS"]


def test_external_qc_config_enables_paper_ready_validation_filters(tmp_path):
    """AAI and required-gene rewards should make Arc run the final paper-stage filters."""
    base_config = {
        "results_save_dir": "unused",
        "current_config_file": "unused",
        "evo_gen_seqs_fasta_file_save_location": "unused",
        "reference_genome_fasta": "",
        "genetic_architecture_reference_genome": "",
        "reference_tropism_protein": "",
        "mmseqs_db_protein_database": "",
        "training_data_genomes_fasta": "",
        "mmseqs_db_tropism_protein": "",
        "genetic_architecture_visualization_script": "",
        "protein_annotation_file": "",
        "reference_genome_gff_file_save_location": "reference.gff",
    }
    base_config_path = tmp_path / "arc_config.yaml"
    base_config_path.write_text(yaml.safe_dump(base_config))
    input_fasta = tmp_path / "input.fasta"
    input_fasta.write_text(">umi1\nACGT\n")

    run_config_path = _write_external_qc_config(
        base_config_path,
        tmp_path / "run",
        input_fasta,
        ExternalQCRewardConfig(
            enable_synteny=True,
            synteny_mode="full",
            enable_average_protein_identity=True,
            enable_required_genes=True,
        ),
    )

    run_config = yaml.safe_load(run_config_path.read_text())
    assert run_config["training_data_sequence_identity_filter"] is False
    assert run_config["diversification_filtering"] is False
    assert run_config["mmseqs_reference_genome_sequence_identity_remove_filter"] is False
    assert run_config["genetic_architecture_visualization_and_synteny_filtering"] is True
    assert run_config["syntenic_gene_count_filter"] is True
    assert run_config["average_protein_sequence_identity_filter"] is True
    assert run_config["required_genes_filter"] is True
    assert run_config["lovis4u_parallel_jobs"] == 12
    assert run_config["n_parallel_jobs"] == 12
    assert run_config["lovis4u_chunk_size"] == 12
    assert run_config["chunk_size"] == 12
    assert "lovis4u_mmseqs_threads" not in run_config
    assert run_config["lovis4u_metrics_only"] is False
    assert run_config["lovis4u_collect_pdfs"] is False
    assert run_config["use_reference_genome"] is True
    assert run_config["reference_genome_gff_file_save_location"].endswith("reference.gff")
    assert run_config["online_measurement_mode"] is True


def test_external_qc_env_prepends_run_specific_tool_directory(tmp_path):
    """External objectives must use the fresh run's tools instead of ambient PATH."""
    tool_bin_dir = tmp_path / "fresh-external" / "bin"
    tool_bin_dir.mkdir(parents=True)

    env = _external_qc_env(ExternalQCRewardConfig(tool_bin_dir=tool_bin_dir))

    assert env["PATH"].split(os.pathsep)[0] == str(tool_bin_dir.resolve())


def test_successful_tropism_search_without_hits_is_a_measured_zero(tmp_path):
    """A completed no-hit search is biological zero evidence, not missing telemetry."""
    tropism_dir = tmp_path / "tropism"
    tropism_dir.mkdir()
    pd.DataFrame(
        columns=[
            "id_prompt",
            "tropism_protein_mmseqs_target",
            "tropism_protein_mmseqs_percent_identity",
        ]
    ).to_csv(tropism_dir / "mmseqs2_hits.csv", index=False)
    scored = pd.DataFrame({"id_prompt": ["umi1"], "reward_external_tropism": [0.0]})

    observed = _add_mmseqs_hit_rewards(
        scored,
        tmp_path,
        {
            "mmseqs_tropism_protein_results_dir_save_location": "tropism",
            "tropism_protein_sequence_identity_range": [60, 100],
        },
    )

    assert observed["tropism_stage_reached"].tolist() == [1.0]
    assert observed["tropism_measurement_available"].tolist() == [1.0]
    assert observed["tropism_hit_present"].tolist() == [0.0]
    assert observed["reward_external_tropism"].tolist() == [0.0]


def test_score_nucleotide_metrics_can_fold_in_external_qc_rewards(tmp_path, monkeypatch):
    """The external Arc wrapper should map staged outputs back to per-sequence rewards."""
    annotation_file = tmp_path / "phrog_annot_v4.tsv"
    annotation_file.write_text(
        "phrog\tannot\tcategory\nphrog_1\tterminase\tpackaging\nphrog_2\tendolysin\tlysis\nphrog_3\tnan\tunknown\n"
    )
    base_config = {
        "results_save_dir": "unused",
        "current_config_file": "unused",
        "evo_gen_seqs_fasta_file_save_location": "unused",
        "overwrite_sequence_ids": False,
        "orf_filter_seqs_csv_file_save_location": "qc3_orf_filter_seqs.csv",
        "homology_filter_seqs_csv_file_save_location": "qc4_homology_filter_seqs.csv",
        "orfipy_proteins_file_save_location": "qc4_orfipy_proteins.fasta",
        "mmseqs_protein_database_results_dir_save_location": "qc4_mmseqs_results_protein_database",
        "mmseqs_tropism_protein_results_dir_save_location": "qc4_mmseqs_results_tropism_protein",
        "synteny_filter_seqs_csv_file_save_location": "qc6_synteny_filter_seqs.csv",
        "average_protein_sequence_identity_metrics_file_save_location": "qc6_average_protein_sequence_identity_metrics.csv",
        "required_genes_metrics_file_save_location": "qc6_required_genes_metrics.csv",
        "synteny_metrics_file_save_location": "qc6_synteny_filter_metrics.csv",
        "protein_database_hit_count": 2,
        "protein_annotation_file": str(annotation_file),
        "required_genes_list": ["terminase", "endolysin"],
        "total_gene_count_range": [2, 2],
        "tropism_protein_sequence_identity_range": [60, 100],
    }
    config_path = tmp_path / "arc_config.yaml"
    config_path.write_text(yaml.safe_dump(base_config))
    pipeline_script = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_script.write_text("print('mock pipeline')\n")

    def fake_run(args, check, cwd, env, timeout):
        assert args[0] == sys.executable
        assert timeout == 1800.0
        run_config_path = Path(args[-1])
        run_config = yaml.safe_load(run_config_path.read_text())
        run_dir = Path(run_config["results_save_dir"])
        pd.DataFrame({"id_prompt": ["umi1"], "sequence": ["ACGT"]}).to_csv(
            run_dir / "qc3_orf_filter_seqs.csv", index=False
        )
        pd.DataFrame({"id_prompt": ["umi1", "umi2"], "sequence": ["ACGT", "ACGT"]}).to_csv(
            run_dir / "qc4_homology_filter_seqs.csv",
            index=False,
        )
        (run_dir / "qc4_orfipy_proteins.fasta").write_text(
            ">umi1_ORF.1\nM\n>umi1_ORF.2\nM\n>umi1_ORF.3\nM\n>umi2_ORF.1\nM\n"
        )
        phrogs_dir = run_dir / "qc4_mmseqs_results_protein_database"
        phrogs_dir.mkdir()
        pd.DataFrame(
            {
                "id_prompt": ["umi1_ORF.1", "umi1_ORF.2", "umi2_ORF.1"],
                "sequence": ["M", "M", "M"],
                "protein_database_mmseqs_target": ["phrog_1", "phrog_2", "phrog_3"],
                "protein_database_mmseqs_e_value": [1e-5, 1e-6, 1e-4],
                "protein_database_mmseqs_percent_identity": [80.0, 75.0, 70.0],
            }
        ).to_csv(phrogs_dir / "mmseqs2_hits.csv", index=False)
        tropism_dir = run_dir / "qc4_mmseqs_results_tropism_protein"
        tropism_dir.mkdir()
        pd.DataFrame(
            {
                "id_prompt": ["umi1_ORF.1", "umi2_ORF.1"],
                "sequence": ["M", "M"],
                "tropism_protein_mmseqs_target": ["G", "G"],
                "tropism_protein_mmseqs_e_value": [1e-5, 1e-4],
                "tropism_protein_mmseqs_percent_identity": [90.0, 30.0],
            }
        ).to_csv(tropism_dir / "mmseqs2_hits.csv", index=False)
        pd.DataFrame(
            {
                "id_prompt": ["umi1", "umi2"],
                "average_protein_percent_identity": [90.0, 100.0],
                "average_protein_identity_gene_count": [10, 1],
            }
        ).to_csv(run_dir / "qc6_average_protein_sequence_identity_metrics.csv", index=False)
        pd.DataFrame(
            {
                "id_prompt": ["umi1", "umi2"],
                "required_genes_matched_count": [2, 1],
                "required_genes_total_count": [2, 2],
            }
        ).to_csv(run_dir / "qc6_required_genes_metrics.csv", index=False)
        pd.DataFrame(
            {
                "id_prompt": ["umi1", "umi2"],
                "num_syntenic_genes": [10, 11],
                "total_num_genes": [10, 11],
            }
        ).to_csv(run_dir / "qc6_synteny_filter_metrics.csv", index=False)

    monkeypatch.setattr("subprocess.run", fake_run)

    df = pd.DataFrame({"id_prompt": ["seq0", "seq1"], "sequence": ["ACGT" * 1000, "ACGT" * 1000]})
    scored = score_nucleotide_metrics(
        df,
        weights=RewardWeights(
            valid_nt_chars=0,
            genome_length=0,
            gc_content=0,
            nt_homopolymer=0,
            protein_hit_count=1,
            tropism=1,
            synteny=1,
            average_protein_identity=1,
            required_genes=1,
        ),
        external_qc=ExternalQCRewardConfig(
            enabled=True,
            config_path=config_path,
            pipeline_script=pipeline_script,
            work_dir=tmp_path / "work",
            keep_artifacts=True,
            enable_synteny=True,
            synteny_mode="full",
            enable_average_protein_identity=True,
            enable_required_genes=True,
            required_genes_evidence_target=2,
        ),
    )

    assert scored.loc[0, "reward_historical"] == 1.0
    assert 0.0 < scored.loc[1, "reward_historical"] < 1.0
    assert scored["reward"].tolist() == [0.0, 0.0]
    assert scored["reward_binary_historical_core_pass"].tolist() == [1.0, 0.0]
    assert scored["reward_binary_core_pass"].tolist() == [0.0, 0.0]
    assert scored["reward_binary_historical_full_qc_pass"].tolist() == [1.0, 0.0]
    assert scored["reward_binary_full_qc_pass"].tolist() == [0.0, 0.0]
    assert scored["reward_binary_historical_full_qc_cluster_deduplicated_pass"].tolist() == [1.0, 0.0]
    assert scored["reward_binary_full_qc_cluster_deduplicated_pass"].tolist() == [0.0, 0.0]
    assert scored.loc[0, "reward_external_synteny"] == 1.0
    assert scored.loc[0, "reward_external_average_protein_identity"] == 1.0
    assert scored.loc[0, "reward_external_required_genes"] == 1.0
    assert scored.loc[1, "reward_external_protein_hit_count"] == 0.5
    assert scored.loc[1, "reward_external_tropism"] == 0.5
    assert scored.loc[1, "reward_external_synteny"] == 0.5
    assert scored["predicted_orf_count"].tolist() == [3, 1]
    assert scored["phrogs_hit_orf_count"].tolist() == [2, 1]
    assert scored["phrogs_annotated_orf_count"].tolist() == [2, 0]
    assert scored["unique_phrog_family_count"].tolist() == [2, 1]
    assert scored["unique_canonical_function_count"].tolist() == [2, 0]
    assert scored["phrogs_hit_fraction"].tolist() == [2 / 3, 1.0]
    assert scored["average_protein_identity_measurement_available"].tolist() == [1.0, 1.0]
    assert scored["required_genes_measurement_available"].tolist() == [1.0, 1.0]


def test_external_qc_subprocess_failure_zeros_external_rewards(tmp_path, monkeypatch):
    """An external-QC subprocess failure should zero its rewards without stopping offline scoring."""
    config_path = tmp_path / "arc_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "results_save_dir": "unused",
                "current_config_file": "unused",
                "evo_gen_seqs_fasta_file_save_location": "unused",
            }
        )
    )
    pipeline_script = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_script.write_text("raise SystemExit(1)\n")

    calls = []

    def fail_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise subprocess.CalledProcessError(1, [sys.executable, str(pipeline_script)])

    monkeypatch.setattr("subprocess.run", fail_run)

    with pytest.warns(RuntimeWarning, match="Arc external QC failed"):
        scored = score_nucleotide_metrics(
            pd.DataFrame({"id_prompt": ["seq0"], "sequence": ["ACGT" * 1000]}),
            weights=RewardWeights(
                valid_nt_chars=0,
                genome_length=0,
                gc_content=0,
                nt_homopolymer=0,
                protein_hit_count=1,
            ),
            external_qc=ExternalQCRewardConfig(
                enabled=True,
                config_path=config_path,
                pipeline_script=pipeline_script,
                work_dir=tmp_path / "work",
                fail_on_error=False,
                timeout_seconds=123,
            ),
        )

    assert scored["reward_external_protein_hit_count"].tolist() == [0.0]
    assert scored["reward"].tolist() == [0.0]
    assert scored["external_qc_tool_succeeded"].tolist() == [0.0]
    assert scored["external_qc_measurement_available"].tolist() == [0.0]
    assert calls[0][0][0][0] == sys.executable
    assert calls[0][1]["timeout"] == 123
    assert any((tmp_path / "work").glob("batch_*"))


def test_external_qc_subprocess_failure_raises_by_default_and_retains_artifacts(tmp_path, monkeypatch):
    """Full-QC RL should fail fast if Arc itself fails instead of turning that into biological zero."""
    config_path = tmp_path / "arc_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "results_save_dir": "unused",
                "current_config_file": "unused",
                "evo_gen_seqs_fasta_file_save_location": "unused",
            }
        )
    )
    pipeline_script = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_script.write_text("raise SystemExit(1)\n")

    def fail_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, [sys.executable, str(pipeline_script)])

    monkeypatch.setattr("subprocess.run", fail_run)

    with pytest.raises(RuntimeError, match="Arc external QC failed"):
        score_nucleotide_metrics(
            pd.DataFrame({"id_prompt": ["seq0"], "sequence": ["ACGT" * 1000]}),
            weights=RewardWeights(
                valid_nt_chars=0,
                genome_length=0,
                gc_content=0,
                nt_homopolymer=0,
                protein_hit_count=1,
            ),
            external_qc=ExternalQCRewardConfig(
                enabled=True,
                config_path=config_path,
                pipeline_script=pipeline_script,
                work_dir=tmp_path / "work",
            ),
        )

    assert any((tmp_path / "work").glob("batch_*"))


def test_full_synteny_reward_uses_arc_valid_pair_distance_metric(tmp_path):
    """Full synteny mode should score distance to Arc-valid gene-count pairs."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["valid_a", "valid_b", "reference_like", "low_neighbor", "high_neighbor", "invalid"],
            "num_syntenic_genes": [10, 11, 11, 9, 12, 13],
            "total_num_genes": [10, 12, 11, 10, 13, 12],
        }
    ).to_csv(run_dir / "qc6_synteny_filter_metrics.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["valid_a", "valid_b", "reference_like", "low_neighbor", "high_neighbor", "invalid"],
            "reward_external_synteny": [0.0] * 6,
        }
    )

    scored = _add_full_synteny_rewards(
        df,
        run_dir,
        {
            "synteny_metrics_file_save_location": "qc6_synteny_filter_metrics.csv",
            "synteny_filter_seqs_csv_file_save_location": "qc6_synteny_filter_seqs.csv",
        },
    )

    assert scored["reward_external_synteny"].tolist() == [1.0, 1.0, 0.5, 0.5, 0.25, 0.0]
    assert scored["synteny_pair_distance"].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]


def test_online_synteny_pass_uses_metrics_not_measurement_survivor_csv(tmp_path):
    """Measurement mode retains all rows, so hard pass must come from raw synteny pairs."""
    pd.DataFrame(
        {
            "id_prompt": ["valid", "invalid"],
            "num_syntenic_genes": [10, 0],
            "total_num_genes": [10, 5],
            "missing_synteny_output": [False, False],
        }
    ).to_csv(tmp_path / "metrics.csv", index=False)
    pd.DataFrame({"id_prompt": ["valid", "invalid"]}).to_csv(tmp_path / "survivors.csv", index=False)

    scored = _add_full_synteny_rewards(
        pd.DataFrame({"arc_qc_id": ["valid", "invalid"], "reward_external_synteny": [0.0, 0.0]}),
        tmp_path,
        {
            "online_measurement_mode": True,
            "synteny_metrics_file_save_location": "metrics.csv",
            "synteny_filter_seqs_csv_file_save_location": "survivors.csv",
        },
    )

    assert scored["reward_external_synteny_pass"].tolist() == [1.0, 0.0]


def test_full_synteny_reward_does_not_score_unmeasured_rows(tmp_path):
    """Missing Arc/LoVis4u measurement rows should be unavailable, not partial biological scores."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["measured", "artifact_missing"],
            "num_syntenic_genes": [10, 0],
            "total_num_genes": [10, 0],
            "missing_synteny_output": [False, True],
        }
    ).to_csv(run_dir / "qc6_synteny_filter_metrics.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["measured", "not_reached", "artifact_missing"],
            "reward_external_synteny": [0.0, 0.0, 0.0],
        }
    )

    scored = _add_full_synteny_rewards(
        df,
        run_dir,
        {
            "synteny_metrics_file_save_location": "qc6_synteny_filter_metrics.csv",
            "synteny_filter_seqs_csv_file_save_location": "qc6_synteny_filter_seqs.csv",
        },
    )

    assert scored["reward_external_synteny"].tolist() == [1.0, 0.0, 0.0]
    assert scored["synteny_stage_reached"].tolist() == [1.0, 0.0, 1.0]
    assert scored["synteny_measurement_available"].tolist() == [1.0, 0.0, 0.0]
    assert scored["synteny_missing_artifact"].tolist() == [0.0, 0.0, 1.0]
    assert pd.isna(scored.loc[1, "num_syntenic_genes"])
    assert pd.isna(scored.loc[1, "synteny_pair_score"])


def test_synteny_distance_score_matches_planned_examples():
    """The standalone synteny score should match the reviewed table examples."""
    assert _synteny_distance_score(10, 11)[0] == 1.0
    assert _synteny_distance_score(11, 12)[0] == 1.0
    assert _synteny_distance_score(11, 11)[0] == 0.5
    assert _synteny_distance_score(9, 10)[0] == 0.5
    assert _synteny_distance_score(12, 13)[0] == 0.25
    assert _synteny_distance_score(13, 12)[0] == 0.0


def test_average_protein_identity_reward_uses_prefilter_metrics(tmp_path):
    """Average protein identity should combine novelty and gene evidence."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["umi1", "umi2", "umi3", "umi4"],
            "average_protein_percent_identity": [80.0, 95.0, 97.5, 100.0],
            "average_protein_identity_gene_count": [10, 10, 10, 9],
        }
    ).to_csv(run_dir / "qc6_average_protein_sequence_identity_metrics.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi2", "umi3", "umi4"],
            "reward_external_average_protein_identity": [0.0, 0.0, 0.0, 0.0],
        }
    )

    scored = _add_average_protein_identity_rewards(
        df,
        run_dir,
        {
            "average_protein_sequence_identity_metrics_file_save_location": (
                "qc6_average_protein_sequence_identity_metrics.csv"
            ),
            "average_protein_sequence_identity_range": [0, 95],
        },
    )

    assert scored["reward_external_average_protein_identity_pass"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert scored["reward_external_average_protein_identity"].tolist() == [1.0, 1.0, 0.5, 0.225]
    assert _aai_novelty_score(100.0) == 0.25
    assert _aai_evidence_score(9.0) == 0.9


def test_average_protein_identity_reward_requires_evidence(tmp_path):
    """High AAI with missing or weak gene evidence should receive zero or fractional credit."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["umi1", "umi2"],
            "average_protein_percent_identity": [99.8, 80.0],
            "average_protein_identity_gene_count": [1, 0],
        }
    ).to_csv(run_dir / "qc6_average_protein_sequence_identity_metrics.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi2", "umi_missing"],
            "reward_external_average_protein_identity": [0.0, 0.0, 0.0],
        }
    )

    scored = _add_average_protein_identity_rewards(
        df,
        run_dir,
        {
            "average_protein_sequence_identity_metrics_file_save_location": (
                "qc6_average_protein_sequence_identity_metrics.csv"
            ),
            "average_protein_sequence_identity_range": [0, 95],
        },
    )

    assert 0.0 < scored.loc[0, "reward_external_average_protein_identity"] < 0.1
    assert scored["reward_external_average_protein_identity"].tolist()[1:] == [0.0, 0.0]
    assert scored["reward_external_average_protein_identity_pass"].tolist() == [0.0, 0.0, 0.0]


def test_required_gene_reward_is_fractional_and_evidence_weighted(tmp_path):
    """Required-gene reward should stay gradual and give no credit without evidence."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["umi1", "umi2", "umi3", "umi4"],
            "required_genes_matched_count": [9, 6, 4, 0],
            "required_genes_total_count": [9, 9, 6, 0],
        }
    ).to_csv(run_dir / "qc6_required_genes_metrics.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi2", "umi3", "umi4"],
            "reward_external_required_genes": [0.0, 0.0, 0.0, 0.0],
        }
    )

    scored = _add_required_gene_rewards(
        df,
        run_dir,
        {"required_genes_metrics_file_save_location": "qc6_required_genes_metrics.csv"},
        evidence_target=9.0,
    )

    assert scored["reward_external_required_genes_pass"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert scored.loc[0, "reward_external_required_genes"] == 1.0
    assert round(scored.loc[1, "reward_external_required_genes"], 6) == round(6 / 9, 6)
    assert round(scored.loc[2, "reward_external_required_genes"], 6) == round((4 / 6) * (6 / 9), 6)
    assert scored.loc[3, "reward_external_required_genes"] == 0.0
