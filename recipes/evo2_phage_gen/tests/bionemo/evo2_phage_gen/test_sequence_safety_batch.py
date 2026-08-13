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

import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from bionemo.evo2_phage_gen import sequence_safety_adapters, sequence_safety_batch
from bionemo.evo2_phage_gen.sequence_safety import SafetyState
from bionemo.evo2_phage_gen.sequence_safety_adapters import (
    AdapterResult,
    GenomeInput,
    ORFPreparationResult,
    PredictedGene,
    ToolPin,
    prepare_orf_artifacts,
)
from bionemo.evo2_phage_gen.sequence_safety_batch import (
    AMRFINDER_SPLIT_POLICY_ID,
    AMRFINDER_SPLIT_POLICY_SHA256,
    DIAMOND_SPLIT_POLICY_ID,
    DIAMOND_SPLIT_POLICY_SHA256,
    BatchAdapterExecution,
    BatchSafetyError,
    materialize_batched_orf_inputs,
    run_amrfinder_batch,
    run_toxin_diamond_batch,
    split_amrfinder_batch_output,
    split_diamond_batch_output,
)


_AMRFINDER_HEADER = (
    "Protein id",
    "Contig id",
    "Start",
    "Stop",
    "Strand",
    "Element symbol",
    "Element name",
    "Scope",
    "Type",
    "Subtype",
    "Class",
    "Subclass",
    "Method",
    "Target length",
    "Reference sequence length",
    "% Coverage of reference",
    "% Identity to reference",
    "Alignment length",
    "Closest reference accession",
    "Closest reference name",
    "HMM accession",
    "HMM description",
    "Hierarchy node",
)


class _Predictor:
    def predict(self, sequence: str, *, circular: bool):
        del circular
        return (
            PredictedGene(
                start=1,
                end=9,
                strand="+",
                nucleotide=sequence[:9],
                protein="MK",
            ),
        )


def _artifacts(tmp_path: Path, record_id: str, sequence: str):
    return prepare_orf_artifacts(
        (GenomeInput(sequence_id=record_id, sequence=sequence),),
        tmp_path / record_id,
        predictor=_Predictor(),
        minimum_fallback_amino_acids=3,
    )


def _amrfinder_row(*, protein_id: str, contig_id: str, method: str = "EXACTP") -> str:
    values = {
        "Protein id": protein_id,
        "Contig id": contig_id,
        "Start": "1",
        "Stop": "9",
        "Strand": "+",
        "Element symbol": "blaSYN",
        "Element name": "synthetic control",
        "Scope": "core",
        "Type": "AMR",
        "Subtype": "AMR",
        "Class": "BETA-LACTAM",
        "Subclass": "CLASS A",
        "Method": method,
        "Target length": "3",
        "Reference sequence length": "3",
        "% Coverage of reference": "100.00",
        "% Identity to reference": "100.00",
        "Alignment length": "3",
        "Closest reference accession": "SYN1",
        "Closest reference name": "synthetic control",
        "HMM accession": "NA",
        "HMM description": "NA",
        "Hierarchy node": "SYN",
    }
    return "\t".join(values[column] for column in _AMRFINDER_HEADER)


def test_materialize_batched_orf_inputs_preserves_order_and_one_global_gff_header(tmp_path: Path):
    first = _artifacts(tmp_path, "record-a", "ATGAAATAG")
    second = _artifacts(tmp_path, "record-b", "ATGCCCTAG")

    batched = materialize_batched_orf_inputs(
        (("record-a", first), ("record-b", second)),
        tmp_path / "batch",
    )

    assert batched.record_ids == ("record-a", "record-b")
    assert batched.query_owners == tuple(
        (query.query_id, record_id)
        for record_id, artifacts in (("record-a", first), ("record-b", second))
        for query in artifacts.query_records
    )
    assert batched.artifacts.genomes_fna.read_bytes() == (
        first.genomes_fna.read_bytes() + second.genomes_fna.read_bytes()
    )
    assert batched.artifacts.proteins_faa.read_bytes() == (
        first.proteins_faa.read_bytes() + second.proteins_faa.read_bytes()
    )
    gff_lines = batched.artifacts.proteins_gff.read_text().splitlines()
    assert gff_lines.count("##gff-version 3") == 1
    assert [line for line in gff_lines if line.startswith("##sequence-region ")] == [
        "##sequence-region record-a 1 9",
        "##sequence-region record-b 1 9",
    ]


def test_materialize_batched_orf_inputs_rejects_record_and_query_identity_drift(tmp_path: Path):
    first = _artifacts(tmp_path, "record-a", "ATGAAATAG")
    drifted = replace(first, query_records=(replace(first.query_records[0], sequence_id="record-b"),))

    with pytest.raises(BatchSafetyError, match="record identity"):
        materialize_batched_orf_inputs((("record-a", drifted),), tmp_path / "batch")


def test_split_amrfinder_batch_output_is_byte_exact_for_protein_and_nucleotide_rows(tmp_path: Path):
    first = _artifacts(tmp_path, "record-a", "ATGAAATAG")
    second = _artifacts(tmp_path, "record-b", "ATGCCCTAG")
    batched = materialize_batched_orf_inputs(
        (("record-a", first), ("record-b", second)),
        tmp_path / "batch",
    )
    header = "\t".join(_AMRFINDER_HEADER)
    protein_row = _amrfinder_row(protein_id="record-a__orf0001", contig_id="record-a")
    nucleotide_row = _amrfinder_row(protein_id="NA", contig_id="record-b", method="BLASTX")
    raw = tmp_path / "amrfinder.tsv"
    raw.write_text(f"{header}\n{protein_row}\n{nucleotide_row}\n")

    split = split_amrfinder_batch_output(raw, batched=batched, output_root=tmp_path / "split")

    assert split["record-a"].read_text() == f"{header}\n{protein_row}\n"
    assert split["record-b"].read_text() == f"{header}\n{nucleotide_row}\n"


@pytest.mark.parametrize(
    "protein_id,contig_id",
    (("unknown", "unknown"), ("record-a__orf0001", "record-b")),
)
def test_split_amrfinder_batch_output_rejects_unmapped_or_conflicting_owners(
    tmp_path: Path, protein_id: str, contig_id: str
):
    first = _artifacts(tmp_path, "record-a", "ATGAAATAG")
    second = _artifacts(tmp_path, "record-b", "ATGCCCTAG")
    batched = materialize_batched_orf_inputs(
        (("record-a", first), ("record-b", second)),
        tmp_path / "batch",
    )
    raw = tmp_path / "amrfinder.tsv"
    raw.write_text(
        "\t".join(_AMRFINDER_HEADER) + "\n" + _amrfinder_row(protein_id=protein_id, contig_id=contig_id) + "\n"
    )

    with pytest.raises(BatchSafetyError, match="owner"):
        split_amrfinder_batch_output(raw, batched=batched, output_root=tmp_path / "split")


def test_split_diamond_batch_output_preserves_per_record_row_order_and_rejects_unknown_queries(tmp_path: Path):
    first = _artifacts(tmp_path, "record-a", "ATGAAATAG")
    second = _artifacts(tmp_path, "record-b", "ATGCCCTAG")
    batched = materialize_batched_orf_inputs(
        (("record-a", first), ("record-b", second)),
        tmp_path / "batch",
    )
    row_a1 = "record-a__orf0001\ttoxin-a\t90\t2\t2\t2\t100\t100\t1e-20\t50"
    row_b = "record-b__orf0001\ttoxin-b\t80\t2\t2\t2\t100\t100\t1e-10\t40"
    row_a2 = "record-a__orf0001\ttoxin-c\t70\t2\t2\t2\t100\t100\t1e-8\t30"
    raw = tmp_path / "diamond.tsv"
    raw.write_text(f"{row_a1}\n{row_b}\n{row_a2}\n")

    split = split_diamond_batch_output(raw, batched=batched, output_root=tmp_path / "split")

    assert split["record-a"].read_text() == f"{row_a1}\n{row_a2}\n"
    assert split["record-b"].read_text() == f"{row_b}\n"
    raw.write_text("unknown\ttoxin\t90\t2\t2\t2\t100\t100\t1e-20\t50\n")
    with pytest.raises(BatchSafetyError, match="unknown DIAMOND query"):
        split_diamond_batch_output(raw, batched=batched, output_root=tmp_path / "other-split")


def test_run_amrfinder_batch_executes_once_then_parses_independent_record_outputs(tmp_path: Path, monkeypatch):
    from bionemo.evo2_phage_gen.sequence_safety_cli import (
        _serialize_adapter_attempt,
        _serialize_shared_execution,
    )

    first = _artifacts(tmp_path, "record-a", "ATGAAATAG")
    second = _artifacts(tmp_path, "record-b", "ATGCCCTAG")
    tool = tmp_path / "amrfinder"
    tool.write_bytes(b"amrfinder")
    tool.chmod(0o755)
    blast = tmp_path / "blast"
    hmmer = tmp_path / "hmmer"
    blast.mkdir()
    hmmer.mkdir()
    database = tmp_path / "database"
    database.mkdir()
    pin = ToolPin(
        path=tool,
        sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
        version="AMRFinderPlus version 4.2.7",
    )
    manifest = {
        "binary_path": str(tool),
        "binary_sha256": pin.sha256,
        "amrfinder_version": pin.version,
        "database_path": str(database),
        "database_sha256": "a" * 64,
        "database_version": "2026-08-07.1",
    }
    monkeypatch.setattr(
        sequence_safety_batch,
        "_validate_amrfinder_manifest_section",
        lambda section: (pin, database, "2026-08-07.1", blast, hmmer),
    )
    monkeypatch.setattr(sequence_safety_batch, "validate_tool_pin", lambda *args, **kwargs: pin.version)
    commands = []

    def runner(command, **kwargs):
        commands.append(tuple(command))
        if command[-1] == "--database_version":
            return subprocess.CompletedProcess(command, 0, stdout="2026-08-07.1\n", stderr="")
        output = Path(command[command.index("-o") + 1])
        output.write_text(
            "\t".join(_AMRFINDER_HEADER)
            + "\n"
            + _amrfinder_row(protein_id="record-a__orf0001", contig_id="record-a")
            + "\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    record_roots = {record_id: tmp_path / "records" / record_id for record_id in ("record-a", "record-b")}
    for root in record_roots.values():
        root.mkdir(parents=True)

    execution = run_amrfinder_batch(
        (("record-a", first), ("record-b", second)),
        manifest_section=manifest,
        work_dir=tmp_path / "batch-execution",
        record_output_roots=record_roots,
        threads=8,
        runner=runner,
    )

    assert execution.batch_id == "batch-execution"
    assert execution.record_ids == ("record-a", "record-b")
    assert execution.raw_output_path == tmp_path / "batch-execution" / "amrfinder.raw.tsv"
    assert len(commands) == 2  # database version check plus one combined search
    assert commands[-1][commands[-1].index("--threads") + 1] == "8"
    results = dict(execution.record_results)
    assert results["record-a"].class_result.state is SafetyState.FAIL
    assert results["record-b"].class_result.state is SafetyState.PASS
    assert results["record-a"].raw_output_path == str(record_roots["record-a"] / "amrfinder.tsv")
    assert results["record-b"].raw_output_path == str(record_roots["record-b"] / "amrfinder.tsv")
    assert results["record-a"].command == execution.command == results["record-b"].command
    assert results["record-a"].shared_execution_id == execution.batch_id
    assert results["record-b"].shared_execution_id == execution.batch_id
    assert results["record-a"].command_output_path == str(execution.raw_output_path)
    assert results["record-b"].command_output_path == str(execution.raw_output_path)

    shared = _serialize_shared_execution(
        execution,
        root=tmp_path,
        record_indices={"record-a": 0, "record-b": 1},
    )
    attempt = _serialize_adapter_attempt(
        "amr",
        results["record-a"],
        root=tmp_path,
        record_root=record_roots["record-a"],
    )
    assert shared["execution_id"] == "batch-execution"
    assert shared["record_ids"] == ["record-a", "record-b"]
    assert shared["record_indices"] == [0, 1]
    assert shared["split_policy"] == {
        "policy_id": execution.split_policy_id,
        "policy_sha256": execution.split_policy_sha256,
    }
    assert set(shared["inputs"]) == {
        "all_queries_faa",
        "genomes_fna",
        "proteins_faa",
        "proteins_fna",
        "proteins_gff",
    }
    assert shared["raw_command_output"]["path"] == "batch-execution/amrfinder.raw.tsv"
    assert attempt["shared_execution_id"] == "batch-execution"
    assert attempt["raw_command_output"]["path"] == "records/record-a/amrfinder.tsv"
    assert attempt["raw_command_output"] != shared["raw_command_output"]


def test_run_toxin_batch_executes_once_then_parses_independent_record_outputs(tmp_path: Path, monkeypatch):
    """One exact DIAMOND search can be split without changing per-record toxin decisions."""
    first = _artifacts(tmp_path, "record-a", "ATGAAATAG")
    second = _artifacts(tmp_path, "record-b", "ATGCCCTAG")
    tool = tmp_path / "diamond"
    tool.write_bytes(b"diamond-2.1.24")
    tool.chmod(0o755)
    pin = ToolPin(
        path=tool,
        sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
        version="diamond version 2.1.24",
        version_args=("version",),
    )
    annotations = tmp_path / "reviewed_toxins.tsv"
    annotations.write_text(
        "Entry\tEntry Name\tProtein names\tGene Names\tOrganism\tOrganism (ID)\tKeyword ID\t"
        "Taxonomic lineage (IDs)\tFunction [CC]\n"
        "P0C1\tSYN_TOX\tSynthetic toxin control\tsyn\tSynthetic virus\t10239\tKW-0800; KW-0843\t"
        "1; 10239\tSynthetic annotation\n"
    )
    fasta = tmp_path / "reviewed_toxins.faa"
    fasta.write_text(">sp|P0C1|SYN_TOX Synthetic toxin control\nMPEPTIDE\n")
    database = tmp_path / "reviewed_toxins.dmnd"
    database.write_bytes(b"reviewed-toxin-diamond-db")
    manifest = {
        "uniprot_release": "2026_03",
        "classification_policy": dict(sequence_safety_adapters.TOXIN_REFERENCE_CLASSIFICATION_POLICY),
        "files": {
            "annotations": {"path": str(annotations), "sha256": hashlib.sha256(annotations.read_bytes()).hexdigest()},
            "fasta": {"path": str(fasta), "sha256": hashlib.sha256(fasta.read_bytes()).hexdigest()},
            "diamond_database": {
                "path": str(database),
                "sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
            },
        },
    }
    monkeypatch.setattr(sequence_safety_batch, "validate_tool_pin", lambda *args, **kwargs: pin.version)
    commands = []

    def runner(command, **kwargs):
        commands.append(tuple(command))
        output = Path(command[command.index("--out") + 1])
        output.write_text("record-a__orf0001\tsp|P0C1|SYN_TOX\t85.0\t2\t2\t2\t100\t100\t1e-30\t150\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    record_roots = {record_id: tmp_path / "records" / record_id for record_id in ("record-a", "record-b")}
    for root in record_roots.values():
        root.mkdir(parents=True)

    execution = run_toxin_diamond_batch(
        (("record-a", first), ("record-b", second)),
        manifest_section=manifest,
        tool_pin=pin,
        work_dir=tmp_path / "toxin-batch",
        record_output_roots=record_roots,
        threads=8,
        runner=runner,
    )

    assert len(commands) == 1
    assert execution.safety_class == "toxin"
    assert execution.raw_output_path == tmp_path / "toxin-batch" / "toxin_diamond.raw.tsv"
    results = dict(execution.record_results)
    assert results["record-a"].class_result.state is SafetyState.FAIL
    assert results["record-b"].class_result.state is SafetyState.PASS
    assert results["record-a"].shared_execution_id == execution.batch_id
    assert results["record-b"].shared_execution_id == execution.batch_id
    assert (record_roots["record-a"] / "toxin_diamond.raw.tsv").read_text().startswith("record-a__orf0001")
    assert (record_roots["record-b"] / "toxin_diamond.raw.tsv").read_text() == ""


def test_default_batched_scan_preserves_record_order_and_runs_independent_phrogs(tmp_path: Path):
    """The high-level scheduler shares only AMR/toxin and keeps one PHROGs target per record."""
    from bionemo.evo2_phage_gen.sequence_safety_cli import (
        _ADAPTER_POLICIES,
        FastaRecord,
        run_default_batched_adapter_bundles,
    )

    records = tuple(
        FastaRecord(
            sequence_id=f"record-{index}",
            original_bytes=f">record-{index}\nATGAAATAG\n".encode(),
            normalized_sequence="ATGAAATAG",
        )
        for index in range(4)
    )

    def prepare_orfs(genomes, output_dir, **_kwargs):
        artifacts = prepare_orf_artifacts(
            genomes,
            output_dir,
            predictor=_Predictor(),
            minimum_fallback_amino_acids=3,
        )
        return ORFPreparationResult(state=SafetyState.PASS, artifacts=artifacts)

    def batch_adapter(record_artifacts, *, work_dir, record_output_roots, **kwargs):
        safety_class = "amr" if "tool_pin" not in kwargs else "toxin"
        work_dir.mkdir(parents=True)
        batched = materialize_batched_orf_inputs(record_artifacts, work_dir / "inputs")
        raw = work_dir / f"{safety_class}.raw.tsv"
        raw.write_text("\t".join(_AMRFINDER_HEADER) + "\n" if safety_class == "amr" else "")
        results = []
        for record_id, _artifacts_for_record in record_artifacts:
            root = Path(record_output_roots[record_id])
            if safety_class == "amr":
                normalized = root / "amrfinder.tsv"
                normalized.write_bytes(raw.read_bytes())
            else:
                (root / "toxin_diamond.raw.tsv").write_text("")
                normalized = root / "toxin_diamond.tsv"
                normalized.write_text("\t".join(sequence_safety_adapters._DIAMOND_COLUMNS) + "\n")
            policy_id, policy_sha256 = _ADAPTER_POLICIES[safety_class]
            results.append(
                (
                    record_id,
                    AdapterResult(
                        class_result=sequence_safety_adapters._class_result(
                            safety_class,
                            SafetyState.PASS,
                            required=True,
                            reason_codes=(f"{safety_class.upper()}_MEASURED_PASS",),
                        ),
                        command=(f"{safety_class}-tool",),
                        raw_output_path=str(normalized),
                        raw_output_sha256=hashlib.sha256(normalized.read_bytes()).hexdigest(),
                        policy_id=policy_id,
                        policy_sha256=policy_sha256,
                        shared_execution_id=work_dir.name,
                        command_output_path=str(raw.absolute()),
                    ),
                )
            )
        split_policy = (
            (AMRFINDER_SPLIT_POLICY_ID, AMRFINDER_SPLIT_POLICY_SHA256)
            if safety_class == "amr"
            else (DIAMOND_SPLIT_POLICY_ID, DIAMOND_SPLIT_POLICY_SHA256)
        )
        return BatchAdapterExecution(
            batch_id=work_dir.name,
            safety_class=safety_class,
            record_ids=batched.record_ids,
            inputs=batched,
            command=(f"{safety_class}-tool",),
            raw_output_path=raw,
            raw_output_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            split_policy_id=split_policy[0],
            split_policy_sha256=split_policy[1],
            record_results=tuple(results),
        )

    phrogs_calls = []
    phrogs_validation_calls = []
    validated_context = object()

    def validate_phrogs(section):
        phrogs_validation_calls.append(("before", section))
        return validated_context

    def revalidate_phrogs(section, context):
        phrogs_validation_calls.append(("after", section, context))

    def phrogs_adapter(artifacts, *, work_dir, _validated_assets, **_kwargs):
        assert _validated_assets is validated_context
        phrogs_calls.append(artifacts.query_records[0].sequence_id)
        raw = Path(work_dir) / "phrogs.raw.tsv"
        normalized = Path(work_dir) / "phrogs.tsv"
        raw.write_text("")
        normalized.write_text("\t".join(sequence_safety_adapters._PHROGS_COLUMNS) + "\n")
        policy_id, policy_sha256 = _ADAPTER_POLICIES["lysogeny"]
        return AdapterResult(
            class_result=sequence_safety_adapters._class_result(
                "lysogeny",
                SafetyState.PASS,
                required=True,
                reason_codes=("PHROGS_MEASURED_PASS",),
            ),
            command=("mmseqs",),
            raw_output_path=str(normalized),
            raw_output_sha256=hashlib.sha256(normalized.read_bytes()).hexdigest(),
            policy_id=policy_id,
            policy_sha256=policy_sha256,
        )

    execution = run_default_batched_adapter_bundles(
        records,
        work_root=tmp_path / "records",
        shared_root=tmp_path / "shared-executions",
        asset_manifest={"amrfinder_plus": {}, "toxin_reference": {}, "phrogs_v4": {}},
        diamond_pin=ToolPin(tmp_path / "diamond", "a" * 64, "diamond"),
        mmseqs_pin=ToolPin(tmp_path / "mmseqs", "b" * 64, "mmseqs"),
        host_domain=sequence_safety_adapters.HostDomain.BACTERIA,
        batch_size=2,
        batch_workers=2,
        phrogs_workers=2,
        prepare_orfs=prepare_orfs,
        amr_batch_adapter=batch_adapter,
        toxin_batch_adapter=batch_adapter,
        phrogs_adapter=phrogs_adapter,
        phrogs_asset_validator=validate_phrogs,
        phrogs_asset_revalidator=revalidate_phrogs,
    )

    assert [record.sequence_id for record in execution.batch.records] == [record.sequence_id for record in records]
    assert len(execution.shared_executions) == 4
    assert {item.safety_class for item in execution.shared_executions} == {"amr", "toxin"}
    assert sorted(phrogs_calls) == sorted(record.sequence_id for record in records)
    assert phrogs_validation_calls == [
        ("before", {}),
        ("after", {}, validated_context),
    ]


def test_asset_failure_shared_execution_serializes_as_not_started(tmp_path: Path):
    """An INDETERMINATE asset error must remain publishable without fabricated command output."""
    from bionemo.evo2_phage_gen.sequence_safety_cli import _serialize_shared_execution

    first = _artifacts(tmp_path, "record-a", "ATGAAATAG")
    second = _artifacts(tmp_path, "record-b", "ATGCCCTAG")
    record_roots = {
        record_id: tmp_path / "records" / f"{index:06d}-{record_id}"
        for index, record_id in enumerate(("record-a", "record-b"))
    }
    for root in record_roots.values():
        root.mkdir(parents=True)

    execution = run_amrfinder_batch(
        (("record-a", first), ("record-b", second)),
        manifest_section={},
        work_dir=tmp_path / "shared-executions" / "amr-0000",
        record_output_roots=record_roots,
    )

    assert execution.execution_status == "NOT_STARTED"
    assert all(result.shared_execution_id == execution.batch_id for _, result in execution.record_results)
    payload = _serialize_shared_execution(
        execution,
        root=tmp_path,
        record_indices={"record-a": 0, "record-b": 1},
    )
    assert payload["execution_status"] == "NOT_STARTED"
    assert payload["command"] == []
    assert payload["raw_command_output"] is None


def test_started_batch_failure_without_output_serializes_as_failed(tmp_path: Path, monkeypatch):
    """A failed command with no output must retain its command and a nullable raw artifact."""
    from bionemo.evo2_phage_gen.sequence_safety_cli import _serialize_shared_execution

    first = _artifacts(tmp_path, "record-a", "ATGAAATAG")
    second = _artifacts(tmp_path, "record-b", "ATGCCCTAG")
    record_roots = {record_id: tmp_path / "records" / record_id for record_id in ("record-a", "record-b")}
    for root in record_roots.values():
        root.mkdir(parents=True)
    pin = ToolPin(tmp_path / "amrfinder", "a" * 64, "4.2.7")
    monkeypatch.setattr(
        sequence_safety_batch,
        "_validate_amrfinder_manifest_section",
        lambda _section: (pin, tmp_path / "database", "2026-08-08", tmp_path / "blast", tmp_path / "hmmer"),
    )
    monkeypatch.setattr(sequence_safety_batch, "validate_tool_pin", lambda *_args, **_kwargs: pin.version)
    monkeypatch.setattr(sequence_safety_batch, "_parse_amrfinder_database_version", lambda _value: "2026-08-08")

    def runner(command, **_kwargs):
        if "--database_version" in command:
            return subprocess.CompletedProcess(command, 0, stdout="2026-08-08", stderr="")
        raise subprocess.CalledProcessError(1, command)

    execution = run_amrfinder_batch(
        (("record-a", first), ("record-b", second)),
        manifest_section={},
        work_dir=tmp_path / "shared-executions" / "amr-0000",
        record_output_roots=record_roots,
        runner=runner,
    )
    payload = _serialize_shared_execution(
        execution,
        root=tmp_path,
        record_indices={"record-a": 0, "record-b": 1},
    )

    assert execution.execution_status == "FAILED"
    assert payload["execution_status"] == "FAILED"
    assert payload["command"]
    assert payload["raw_command_output"] is None


def test_amrfinder_batch_parse_failure_promotes_no_record_outputs(tmp_path: Path, monkeypatch):
    """Parsing must succeed for every split before any record artifact is published."""
    first = _artifacts(tmp_path, "record-a", "ATGAAATAG")
    second = _artifacts(tmp_path, "record-b", "ATGCCCTAG")
    record_roots = {record_id: tmp_path / "records" / record_id for record_id in ("record-a", "record-b")}
    for root in record_roots.values():
        root.mkdir(parents=True)
    pin = ToolPin(tmp_path / "amrfinder", "a" * 64, "4.2.7")
    monkeypatch.setattr(
        sequence_safety_batch,
        "_validate_amrfinder_manifest_section",
        lambda _section: (pin, tmp_path / "database", "2026-08-08", tmp_path / "blast", tmp_path / "hmmer"),
    )
    monkeypatch.setattr(sequence_safety_batch, "validate_tool_pin", lambda *_args, **_kwargs: pin.version)
    monkeypatch.setattr(sequence_safety_batch, "_parse_amrfinder_database_version", lambda _value: "2026-08-08")

    def runner(command, **_kwargs):
        if "--database_version" in command:
            return subprocess.CompletedProcess(command, 0, stdout="2026-08-08", stderr="")
        output = Path(command[command.index("-o") + 1])
        output.write_text(
            "\t".join(_AMRFINDER_HEADER)
            + "\n"
            + _amrfinder_row(protein_id="record-a__orf0001", contig_id="record-a")
            + "\n"
            + _amrfinder_row(protein_id="record-b__orf0001", contig_id="record-b", method="UNSUPPORTED")
            + "\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    execution = run_amrfinder_batch(
        (("record-a", first), ("record-b", second)),
        manifest_section={},
        work_dir=tmp_path / "shared-executions" / "amr-0000",
        record_output_roots=record_roots,
        runner=runner,
    )

    assert execution.execution_status == "FAILED"
    assert all(not (root / "amrfinder.tsv").exists() for root in record_roots.values())
