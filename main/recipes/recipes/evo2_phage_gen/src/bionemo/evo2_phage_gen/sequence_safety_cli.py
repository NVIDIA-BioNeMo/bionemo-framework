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

"""Run and summarize computational phage sequence-safety screens."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from bionemo.evo2_phage_gen.design_scope import (
    DesignObjective,
    HostDomain,
    HostEvidence,
    evaluate_host_evidence,
    validate_design_scope,
)
from bionemo.evo2_phage_gen.sequence_safety import (
    GenomeSafetyResult,
    SafetyClassResult,
    SafetyState,
    load_phage_safety_policy,
)
from bionemo.evo2_phage_gen.sequence_safety_adapters import (
    AdapterResult,
    GenomeInput,
    ToolRuntime,
    observe_tool_version,
    prepare_orf_artifacts,
    run_amrfinder_batch,
    run_phrogs_batch,
    run_toxin_batch,
)


SAFETY_CLASSES = ("amr", "toxin", "lysogeny")


class CLIValidationError(ValueError):
    """Report invalid sequence-screen input or output."""

    pass


@dataclass(frozen=True)
class FastaRecord:
    """Represent one parsed FASTA record."""

    sequence_id: str
    header: str
    sequence: str


def parse_fasta_records(path: Path, *, validate_sequence: bool = True) -> tuple[FastaRecord, ...]:
    """Parse and validate FASTA records."""
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_lines: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(_fasta_record(header, sequence_lines, validate_sequence=validate_sequence))
            header = line[1:]
            sequence_lines = []
        else:
            if header is None:
                raise CLIValidationError("FASTA sequence appears before its first header")
            sequence_lines.append(line)
    if header is not None:
        records.append(_fasta_record(header, sequence_lines, validate_sequence=validate_sequence))
    if not records:
        raise CLIValidationError("input FASTA is empty")
    identifiers = [record.sequence_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise CLIValidationError("input FASTA contains duplicate record IDs")
    return tuple(records)


def _fasta_record(header: str, sequence_lines: list[str], *, validate_sequence: bool) -> FastaRecord:
    sequence_id = header.split(maxsplit=1)[0]
    sequence = "".join(sequence_lines).upper()
    if not validate_sequence:
        if not sequence_id or not sequence:
            raise CLIValidationError("FASTA records require a nonempty ID and sequence")
        return FastaRecord(sequence_id, header, sequence)
    try:
        genome = GenomeInput(sequence_id, sequence)
    except ValueError as error:
        raise CLIValidationError(str(error)) from error
    return FastaRecord(genome.sequence_id, header, genome.sequence)


def _host_evidence(value: str) -> HostEvidence:
    try:
        payload = json.loads(value)
        domains = frozenset(HostDomain(item) for item in payload["replication_host_domains"])
        return HostEvidence(
            source=payload["source"],
            source_version=payload.get("source_version"),
            replication_host_domains=domains,
            confirmed=payload["confirmed"],
            metadata=payload.get("metadata", {}),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CLIValidationError("host evidence must be valid HostEvidence JSON") from error


def validate_design_scope_payload(*, objective: object, host_evidence: object) -> dict[str, object]:
    """Validate a serialized design objective and host-evidence pair."""
    try:
        if not isinstance(objective, Mapping) or not isinstance(host_evidence, Mapping):
            raise TypeError("objective and host evidence must be mappings")
        parsed_objective = DesignObjective(
            kind=objective["kind"],
            direction=objective["direction"],
            replication_host_domains=frozenset(objective["replication_host_domains"]),
            endpoint=objective["endpoint"],
        )
        parsed_evidence = HostEvidence(
            source=host_evidence["source"],
            source_version=host_evidence.get("source_version"),
            replication_host_domains=frozenset(host_evidence["replication_host_domains"]),
            confirmed=host_evidence["confirmed"],
            metadata=host_evidence.get("metadata", {}),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CLIValidationError(str(error)) from error
    objective_decision = validate_design_scope(parsed_objective)
    evidence_decision = evaluate_host_evidence(parsed_evidence)
    allowed = objective_decision.allowed and evidence_decision.allowed
    return {
        "state": "PASS" if allowed else "FAIL",
        "allowed": allowed,
        "reason_codes": list(dict.fromkeys((*objective_decision.reason_codes, *evidence_decision.reason_codes))),
        "objective": parsed_objective.to_dict(),
        "host_evidence": parsed_evidence.to_dict(),
    }


def aggregate_adapter_results(
    adapters: Mapping[str, AdapterResult],
    *,
    host_domain: HostDomain,
    strict_lysis: bool = False,
) -> GenomeSafetyResult:
    """Combine detector outputs for one genome."""
    required = {
        "amr": True,
        "toxin": True,
        "lysogeny": host_domain in {HostDomain.BACTERIA, HostDomain.BACTERIA_AND_ARCHAEA} or strict_lysis,
    }
    results: list[SafetyClassResult] = []
    for safety_class in SAFETY_CLASSES:
        adapter = adapters.get(safety_class)
        if adapter is None:
            results.append(
                SafetyClassResult(
                    safety_class,
                    SafetyState.INDETERMINATE,
                    required[safety_class],
                    reason_codes=("ADAPTER_RESULT_MISSING",),
                )
            )
        else:
            results.append(replace(adapter.class_result, required=required[safety_class]))

    supplemental = adapters.get("amr")
    virulence = () if supplemental is None else supplemental.supplemental_findings
    if virulence:
        toxin = results[1]
        state = SafetyState.FAIL if toxin.state is SafetyState.FAIL else SafetyState.INDETERMINATE
        results[1] = replace(
            toxin,
            state=state,
            findings=(*toxin.findings, *virulence),
            reason_codes=tuple(dict.fromkeys((*toxin.reason_codes, "AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL"))),
        )
    return GenomeSafetyResult.from_class_results(tuple(results))


def _load_asset_state(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise CLIValidationError("safety asset state is missing or unsupported")
    tools = payload.get("tools")
    databases = payload.get("databases")
    if not isinstance(tools, dict) or not isinstance(databases, dict):
        raise CLIValidationError("safety asset state lacks tools or databases")
    return payload


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CLIValidationError(f"{label} must be a mapping")
    return value


def _path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CLIValidationError(f"{label} path is missing")
    return Path(value)


def _version(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CLIValidationError(f"{label} version is missing")
    return value


def _runtime_state(args: argparse.Namespace, assets: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    tools = _mapping(assets["tools"], "tools")
    databases = _mapping(assets["databases"], "databases")
    amr_tool = _mapping(tools.get("amrfinder"), "AMRFinder tool")
    diamond_tool = _mapping(tools.get("diamond"), "DIAMOND tool")
    mmseqs_tool = _mapping(tools.get("mmseqs"), "MMseqs tool")
    amr_database = _mapping(databases.get("amrfinder"), "AMRFinder database")
    toxin_database = _mapping(databases.get("toxins"), "toxin database")
    phrogs_database = _mapping(databases.get("phrogs"), "PHROGs database")

    selected = {
        "amrfinder": ToolRuntime(_path(amr_tool.get("path"), "AMRFinder")),
        "diamond": ToolRuntime(
            Path(args.diamond_bin) if args.diamond_bin else _path(diamond_tool.get("path"), "DIAMOND"),
            ("version",),
        ),
        "mmseqs": ToolRuntime(
            Path(args.mmseqs_bin) if args.mmseqs_bin else _path(mmseqs_tool.get("path"), "MMseqs"),
            ("version",),
        ),
    }
    observed: dict[str, object] = {}
    for name, runtime in selected.items():
        try:
            version = observe_tool_version(runtime, timeout=args.timeout)
        except (OSError, subprocess.SubprocessError):
            version = "unavailable"
        observed[name] = {"path": str(runtime.path), "version": version}
    database_state = {
        "amrfinder": {
            "path": str(_path(amr_database.get("path"), "AMRFinder database")),
            "version": _version(amr_database.get("version"), "AMRFinder database"),
        },
        "toxins": {
            "path": str(_path(toxin_database.get("diamond_database_path"), "toxin database")),
            "version": _version(toxin_database.get("release"), "toxin database"),
        },
        "phrogs": {
            "path": str(_path(phrogs_database.get("profile_database_path"), "PHROGs profile database")),
            "lookup": str(_path(phrogs_database.get("lookup_path"), "PHROGs lookup")),
            "version": _version(phrogs_database.get("release"), "PHROGs database"),
        },
    }
    return {"selected": selected, "observed": observed}, database_state


def _run_scan(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise CLIValidationError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    policy = load_phage_safety_policy(args.policy)
    evidence = _host_evidence(args.host_evidence_json)
    evidence_decision = evaluate_host_evidence(evidence)
    if not evidence_decision.allowed:
        raise CLIValidationError("host evidence is not eligible for phage design")
    host_domain = HostDomain(args.host_domain)
    if host_domain not in evidence.replication_host_domains:
        raise CLIValidationError("host domain does not match the supplied evidence")

    records = parse_fasta_records(args.input_fasta)
    genomes = tuple(GenomeInput(record.sequence_id, record.sequence, circular=not args.linear) for record in records)
    assets = _load_asset_state(args.asset_manifest)
    runtime_state, databases = _runtime_state(args, assets)
    selected = runtime_state.pop("selected")
    assert isinstance(selected, dict)
    batches = tuple(genomes[index : index + args.batch_size] for index in range(0, len(genomes), args.batch_size))
    phrogs_threads = args.threads if args.phrogs_threads is None else args.phrogs_threads
    adapter_results: dict[str, dict[str, AdapterResult]] = {}
    commands: dict[str, list[str]] = {}
    for batch_index, batch in enumerate(batches):
        batch_root = output_dir if len(batches) == 1 else output_dir / "batches" / f"{batch_index:06d}"
        batch_label = f"safety scan batch {batch_index + 1}/{len(batches)}"
        batch_started = time.perf_counter()
        print(
            f"{batch_label}: predicting ORFs for {len(batch)} records",
            flush=True,
        )
        try:
            artifacts = prepare_orf_artifacts(batch, batch_root / "orfs", workers=args.orf_workers)
        except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as error:
            reason = f"ORF_PREDICTION_FAILED:{type(error).__name__}"
            for genome in batch:
                adapter_results[genome.sequence_id] = {
                    safety_class: AdapterResult(
                        SafetyClassResult(
                            safety_class,
                            SafetyState.INDETERMINATE,
                            safety_class != "lysogeny" or host_domain is not HostDomain.ARCHAEA or args.strict_lysis,
                            reason_codes=(reason,),
                        ),
                        "FAILED",
                    )
                    for safety_class in SAFETY_CLASSES
                }
            print(
                f"{batch_label}: ORF preparation failed after {time.perf_counter() - batch_started:.1f}s",
                flush=True,
            )
            continue

        orf_finished = time.perf_counter()
        amr_database = databases["amrfinder"]
        toxin_database = databases["toxins"]
        phrogs_database = databases["phrogs"]
        print(f"{batch_label}: AMR screen", flush=True)
        amr = run_amrfinder_batch(
            batch,
            artifacts,
            runtime=selected["amrfinder"],
            database=Path(amr_database["path"]),
            database_version=str(amr_database["version"]),
            work_dir=batch_root / "amrfinder",
            threads=args.threads,
            timeout=args.timeout,
        )
        amr_finished = time.perf_counter()
        print(f"{batch_label}: toxin screen", flush=True)
        toxin = run_toxin_batch(
            batch,
            artifacts,
            runtime=selected["diamond"],
            database=Path(toxin_database["path"]),
            database_version=str(toxin_database["version"]),
            work_dir=batch_root / "toxins",
            threads=args.threads,
            timeout=args.timeout,
        )
        toxin_finished = time.perf_counter()
        print(f"{batch_label}: lysogeny screen", flush=True)
        phrogs = run_phrogs_batch(
            batch,
            artifacts,
            runtime=selected["mmseqs"],
            database=Path(phrogs_database["path"]),
            lookup=Path(phrogs_database["lookup"]),
            database_version=str(phrogs_database["version"]),
            host_domain=host_domain,
            strict_lysis=args.strict_lysis,
            work_dir=batch_root / "phrogs",
            threads=phrogs_threads,
            timeout=args.timeout,
        )
        for genome in batch:
            adapter_results[genome.sequence_id] = {
                "amr": amr[genome.sequence_id],
                "toxin": toxin[genome.sequence_id],
                "lysogeny": phrogs[genome.sequence_id],
            }
        for safety_class, result in (("amr", amr), ("toxin", toxin), ("lysogeny", phrogs)):
            if safety_class not in commands and result:
                commands[safety_class] = list(next(iter(result.values())).command)
        finished = time.perf_counter()
        print(
            f"{batch_label}: complete in {finished - batch_started:.1f}s "
            f"(ORFs {orf_finished - batch_started:.1f}s, AMR {amr_finished - orf_finished:.1f}s, "
            f"toxin {toxin_finished - amr_finished:.1f}s, lysogeny {finished - toxin_finished:.1f}s)",
            flush=True,
        )

    serialized_records: list[dict[str, object]] = []
    for index, record in enumerate(records):
        adapters = adapter_results[record.sequence_id]
        aggregate = aggregate_adapter_results(adapters, host_domain=host_domain, strict_lysis=args.strict_lysis)
        reasons = list(dict.fromkeys(reason for result in aggregate.class_results for reason in result.reason_codes))
        serialized_records.append(
            {
                "input_index": index,
                "record_id": record.sequence_id,
                "sequence_length": len(record.sequence),
                "state": aggregate.state.value,
                "reason_codes": reasons,
                "class_results": [result.to_dict() for result in aggregate.class_results],
                "adapter_attempts": [
                    {
                        "safety_class": safety_class,
                        "execution_status": adapters[safety_class].execution_status,
                        "policy_id": (
                            "amrfinder-curated-calls"
                            if safety_class == "amr"
                            else "toxin-homology-v2"
                            if safety_class == "toxin"
                            else "phrogs-homology-v1"
                        ),
                    }
                    for safety_class in SAFETY_CLASSES
                ],
            }
        )
    counts = Counter(record["state"] for record in serialized_records)
    states = {record["state"] for record in serialized_records}
    aggregate_state = "FAIL" if "FAIL" in states else "INDETERMINATE" if "INDETERMINATE" in states else "PASS"
    manifest = {
        "schema_version": 2,
        "manifest_type": "sequence_safety_scan",
        "input": {"path": str(Path(args.input_fasta).absolute()), "count": len(records)},
        "policy": {"policy_id": policy.policy_id},
        "asset_state": {"path": str(Path(args.asset_manifest).absolute())},
        "resolved_profile": {
            "host_domain": host_domain.value,
            "strict_lysis": args.strict_lysis,
            "circular": not args.linear,
        },
        "tools": runtime_state["observed"],
        "databases": databases,
        "execution": {
            "batch_count": len(batches),
            "batch_size": args.batch_size,
            "orf_workers": args.orf_workers,
            "threads": args.threads,
            "phrogs_threads": phrogs_threads,
        },
        "commands": commands,
        "records": serialized_records,
        "aggregate": {
            "state": aggregate_state,
            "counts": {state: counts.get(state, 0) for state in ("PASS", "FAIL", "INDETERMINATE")},
        },
        "claim_boundary": "Computational screening only; not evidence of wet-lab viability or clinical safety.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "RUNLOG.md").write_text(
        "\n".join(
            (
                "# Sequence safety scan",
                "",
                f"- Input: {Path(args.input_fasta).absolute()} ({len(records)} records)",
                f"- Host profile: {host_domain.value}; strict lysis: {args.strict_lysis}",
                f"- Tools: {json.dumps(runtime_state['observed'], sort_keys=True)}",
                f"- Databases: {json.dumps(databases, sort_keys=True)}",
                (
                    f"- Execution: {len(batches)} batches of up to {args.batch_size}; "
                    f"ORF workers {args.orf_workers}; detector threads {args.threads}; "
                    f"PHROGs threads {phrogs_threads}"
                ),
                f"- Results: {dict(manifest['aggregate']['counts'])}",
                "",
            )
        )
    )
    return _state_exit_code(aggregate_state)


def _state_exit_code(state: str) -> int:
    return 0 if state == "PASS" else 2 if state == "FAIL" else 3


def validate_manifest_file(path: Path, *, expected_type: str | None = None, **_: object) -> dict[str, object]:
    """Load and validate a sequence-screen manifest."""
    try:
        manifest = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CLIValidationError(f"cannot read scan manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise CLIValidationError("manifest must be a mapping")
    manifest_type = manifest.get("manifest_type")
    if expected_type is not None and manifest_type != expected_type:
        raise CLIValidationError(f"expected {expected_type} manifest")
    if manifest_type == "sequence_safety_scan":
        _validate_scan_manifest(manifest)
    elif manifest_type == "sequence_safety_filter":
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("counts"), dict):
            raise CLIValidationError("filter manifest is invalid")
    else:
        raise CLIValidationError("unsupported manifest type")
    return manifest


def validate_detector_execution(
    manifest: Mapping[str, object],
    *,
    allow_no_primary_gene_candidates: bool = False,
) -> tuple[tuple[str, str, str], ...]:
    """Require completed detectors, optionally retaining invalid no-gene candidates as non-passers."""
    records = manifest.get("records")
    if not isinstance(records, list):
        raise CLIValidationError("scan manifest has no records")
    tolerated: list[tuple[str, str, str]] = []
    incomplete: list[tuple[object, object, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise CLIValidationError("scan manifest contains an invalid record")
        record_id = record.get("record_id")
        class_results = record.get("class_results")
        attempts = record.get("adapter_attempts")
        if not isinstance(class_results, list) or not isinstance(attempts, list):
            raise CLIValidationError("scan record lacks detector execution details")
        class_by_name = {result.get("safety_class"): result for result in class_results if isinstance(result, Mapping)}
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                incomplete.append((record_id, None, None))
                continue
            safety_class = attempt.get("safety_class")
            status = attempt.get("execution_status")
            if status == "COMPLETED_AND_PARSED":
                continue
            result = class_by_name.get(safety_class)
            no_primary_genes = (
                allow_no_primary_gene_candidates
                and safety_class == "lysogeny"
                and status == "NOT_RUN"
                and record.get("state") != "PASS"
                and isinstance(result, Mapping)
                and result.get("state") == "INDETERMINATE"
                and result.get("findings") == []
                and result.get("reason_codes") == ["PHROGS_NO_PREDICTED_GENES"]
            )
            if no_primary_genes and isinstance(record_id, str):
                tolerated.append((record_id, "lysogeny", "PHROGS_NO_PREDICTED_GENES"))
            else:
                incomplete.append((record_id, safety_class, status))
    if incomplete:
        raise CLIValidationError(f"incomplete safety detector execution: {incomplete[:10]}")
    return tuple(tolerated)


def _validate_scan_manifest(manifest: dict[str, object]) -> None:
    if manifest.get("schema_version") != 2:
        raise CLIValidationError("unsupported scan schema")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise CLIValidationError("scan manifest has no records")
    counts = Counter()
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("input_index") != index:
            raise CLIValidationError("scan record order is invalid")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise CLIValidationError("scan record IDs are invalid")
        seen.add(record_id)
        class_results = record.get("class_results")
        attempts = record.get("adapter_attempts")
        if not isinstance(class_results, list) or not isinstance(attempts, list):
            raise CLIValidationError("scan record lacks class results")
        classes = [result.get("safety_class") for result in class_results if isinstance(result, dict)]
        if classes != list(SAFETY_CLASSES):
            raise CLIValidationError("scan record safety classes are invalid")
        derived = GenomeSafetyResult.from_class_results(
            tuple(
                SafetyClassResult(
                    str(result["safety_class"]),
                    SafetyState(result["state"]),
                    bool(result["required"]),
                    reason_codes=tuple(result.get("reason_codes", [])),
                )
                for result in class_results
            )
        )
        if record.get("state") != derived.state.value:
            raise CLIValidationError("scan record aggregate is inconsistent")
        counts[derived.state.value] += 1
    aggregate = manifest.get("aggregate")
    expected = {state: counts.get(state, 0) for state in ("PASS", "FAIL", "INDETERMINATE")}
    if not isinstance(aggregate, dict) or aggregate.get("counts") != expected:
        raise CLIValidationError("scan aggregate counts are inconsistent")


def _write_fasta(path: Path, records: list[FastaRecord]) -> None:
    path.write_text("".join(f">{record.header}\n{record.sequence}\n" for record in records))


def _run_filter(args: argparse.Namespace) -> int:
    manifest = validate_manifest_file(args.scan_manifest, expected_type="sequence_safety_scan")
    records = parse_fasta_records(args.input_fasta, validate_sequence=False)
    scan_records = manifest["records"]
    if len(records) != len(scan_records):
        raise CLIValidationError("FASTA and scan manifest record counts differ")
    output = Path(args.output_dir)
    if output.exists():
        if not args.overwrite:
            raise CLIValidationError(f"output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    partitions: dict[str, list[FastaRecord]] = {"PASS": [], "FAIL": [], "INDETERMINATE": []}
    for fasta, scan in zip(records, scan_records, strict=True):
        if fasta.sequence_id != scan["record_id"]:
            raise CLIValidationError("FASTA and scan manifest IDs differ")
        partitions[scan["state"]].append(fasta)
    for state, selected in partitions.items():
        _write_fasta(output / f"{state.lower()}.fasta", selected)
    result = {
        "schema_version": 1,
        "manifest_type": "sequence_safety_filter",
        "source_scan": str(Path(args.scan_manifest).absolute()),
        "counts": {state: len(selected) for state, selected in partitions.items()},
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return _state_exit_code(str(manifest["aggregate"]["state"]))


def _run_validate(args: argparse.Namespace) -> int:
    manifest = validate_manifest_file(args.manifest)
    state = manifest.get("aggregate", {}).get("state", "PASS")
    return _state_exit_code(str(state))


def _run_scope(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text())
    result = validate_design_scope_payload(objective=payload["objective"], host_evidence=payload["host_evidence"])
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(serialized)
    else:
        print(serialized, end="")
    return 0 if result["allowed"] else 2


def build_parser() -> argparse.ArgumentParser:
    """Build the sequence-safety command-line parser."""
    parser = argparse.ArgumentParser(prog="evo2_phage_sequence_safety")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="scan every FASTA record for AMR, toxin, and lysogeny evidence")
    scan.add_argument("--input-fasta", type=Path, required=True)
    scan.add_argument("--output-dir", type=Path, required=True)
    scan.add_argument("--policy", type=Path, required=True)
    scan.add_argument("--asset-manifest", type=Path, required=True)
    scan.add_argument("--host-domain", choices=tuple(domain.value for domain in HostDomain), required=True)
    scan.add_argument("--host-evidence-json", required=True)
    scan.add_argument("--diamond-bin", type=Path)
    scan.add_argument("--mmseqs-bin", type=Path)
    scan.add_argument("--strict-lysis", action="store_true")
    scan.add_argument("--linear", action="store_true")
    scan.add_argument("--threads", type=int, default=1)
    scan.add_argument("--batch-size", type=int, default=1, help="records per detector batch")
    scan.add_argument("--orf-workers", type=int, default=1, help="parallel ORF predictions within each batch")
    scan.add_argument(
        "--phrogs-threads",
        type=int,
        help="PHROGs MMseqs threads per batch (defaults to --threads)",
    )
    scan.add_argument("--timeout", type=float, default=300.0)
    scan.add_argument("--overwrite", action="store_true")
    scan.set_defaults(handler=_run_scan)

    filter_parser = subparsers.add_parser("filter-fasta", help="partition FASTA records by scan state")
    filter_parser.add_argument("--input-fasta", type=Path, required=True)
    filter_parser.add_argument("--scan-manifest", type=Path, required=True)
    filter_parser.add_argument("--output-dir", type=Path, required=True)
    filter_parser.add_argument("--overwrite", action="store_true")
    filter_parser.set_defaults(handler=_run_filter)

    validate_parser = subparsers.add_parser("validate-manifest", help="check manifest consistency")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.set_defaults(handler=_run_validate)

    scope = subparsers.add_parser("validate-design-scope")
    scope.add_argument("--input", type=Path, required=True)
    scope.add_argument("--output", type=Path)
    scope.set_defaults(handler=_run_scope)
    return parser


def main(argv: Sequence[str] | None = None, **_: object) -> int:
    """Run a sequence-safety subcommand."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "scan" and (
            args.threads < 1
            or args.batch_size < 1
            or args.orf_workers < 1
            or (args.phrogs_threads is not None and args.phrogs_threads < 1)
            or args.timeout <= 0
        ):
            raise CLIValidationError("batch size, workers, threads, and timeout must be positive")
        return int(args.handler(args))
    except (CLIValidationError, OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError) as error:
        parser._print_message(f"{parser.prog}: error: {error}\n", sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
