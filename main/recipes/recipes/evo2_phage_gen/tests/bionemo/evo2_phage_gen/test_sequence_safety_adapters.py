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

import csv
import subprocess
from pathlib import Path

from bionemo.evo2_phage_gen import sequence_safety_adapters as adapters
from bionemo.evo2_phage_gen.design_scope import HostDomain
from bionemo.evo2_phage_gen.sequence_safety import SafetyState
from bionemo.evo2_phage_gen.sequence_safety_adapters import (
    GenomeInput,
    PredictedGene,
    ToolRuntime,
    observe_tool_version,
    prepare_orf_artifacts,
    run_amrfinder_batch,
    run_phrogs_batch,
    run_toxin_batch,
)


class _NoGenes:
    def predict(self, sequence: str, *, circular: bool):
        return ()


class _OneGene:
    def predict(self, sequence: str, *, circular: bool):
        return (PredictedGene(1, 30, "+", sequence[:30], "M" * 10),)


def _tool(tmp_path: Path, name: str) -> ToolRuntime:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n")
    return ToolRuntime(path, ("version",))


def _completed(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout, "")


def test_genome_input_normalizes_sequence_and_rejects_invalid_data() -> None:
    assert GenomeInput("phix|rollout", "atg nn").sequence == "ATGNN"


def test_observe_tool_version_logs_selected_runtime(tmp_path: Path, monkeypatch) -> None:
    runtime = _tool(tmp_path, "diamond")
    calls: list[list[str]] = []
    environments: list[dict[str, str]] = []
    monkeypatch.setenv("BASH_ENV", "/etc/bash.bashrc")
    monkeypatch.setenv("ENV", "/etc/shinit_v2")
    monkeypatch.setenv("SAFETY_TEST_VALUE", "preserved")

    def runner(argv, **kwargs):
        calls.append(argv)
        environments.append(kwargs["env"])
        return _completed("diamond version 2.1.24\n")

    assert observe_tool_version(runtime, runner=runner) == "diamond version 2.1.24"
    assert calls == [[str(runtime.path), "version"]]
    assert environments[0]["SAFETY_TEST_VALUE"] == "preserved"
    assert "BASH_ENV" not in environments[0]
    assert "ENV" not in environments[0]


def test_orf_workers_preserve_record_order(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, int] = {}

    class Executor:
        def __init__(self, *, max_workers: int, **_kwargs):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, values):
            return [function(value) for value in values]

    monkeypatch.setattr(adapters, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(adapters, "_new_predictor", _OneGene)
    genomes = tuple(GenomeInput(f"g{index}", "ATG" * 20) for index in range(3))

    artifacts = prepare_orf_artifacts(genomes, tmp_path / "orfs", workers=8)

    assert observed == {"max_workers": 3}
    assert [record.sequence_id for record in artifacts.query_records if record.evidence_path == "pyrodigal-gv"] == [
        "g0",
        "g1",
        "g2",
    ]


def test_tiny_genome_without_orfs_is_review_required_not_a_crash(tmp_path: Path) -> None:
    genome = GenomeInput("tiny", "ATG")
    artifacts = prepare_orf_artifacts(
        (genome,),
        tmp_path / "orfs",
        predictor=_NoGenes(),
        minimum_fallback_amino_acids=8,
    )
    assert artifacts.query_records == ()

    toxin = run_toxin_batch(
        (genome,),
        artifacts,
        runtime=_tool(tmp_path, "diamond"),
        database=tmp_path / "toxins.dmnd",
        database_version="current",
        work_dir=tmp_path / "toxin",
        runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    lysogeny = run_phrogs_batch(
        (genome,),
        artifacts,
        runtime=_tool(tmp_path, "mmseqs"),
        database=tmp_path / "phrogs",
        lookup=tmp_path / "lookup.tsv",
        database_version="current",
        host_domain=HostDomain.BACTERIA,
        strict_lysis=True,
        work_dir=tmp_path / "phrogs-work",
        runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert toxin["tiny"].class_result.state is SafetyState.INDETERMINATE
    assert lysogeny["tiny"].class_result.state is SafetyState.INDETERMINATE
    assert toxin["tiny"].execution_status == "NOT_RUN"
    assert lysogeny["tiny"].execution_status == "NOT_RUN"


def test_toxin_no_hit_is_a_measured_pass(tmp_path: Path) -> None:
    genome = GenomeInput("g1", "ATG" * 20)
    artifacts = prepare_orf_artifacts((genome,), tmp_path / "orfs", predictor=_OneGene())
    runtime = _tool(tmp_path, "diamond")

    def runner(argv, **kwargs):
        if argv[1] == "version":
            return _completed("diamond 2")
        Path(argv[argv.index("--out") + 1]).write_text("")
        return _completed()

    result = run_toxin_batch(
        (genome,),
        artifacts,
        runtime=runtime,
        database=tmp_path / "toxins.dmnd",
        database_version="UniProt current",
        work_dir=tmp_path / "toxin",
        runner=runner,
    )["g1"]
    assert result.class_result.state is SafetyState.PASS
    assert result.execution_status == "COMPLETED_AND_PARSED"


def test_toxin_tool_failure_is_indeterminate(tmp_path: Path) -> None:
    genome = GenomeInput("g1", "ATG" * 20)
    artifacts = prepare_orf_artifacts((genome,), tmp_path / "orfs", predictor=_OneGene())

    def runner(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "diamond")

    result = run_toxin_batch(
        (genome,),
        artifacts,
        runtime=_tool(tmp_path, "diamond"),
        database=tmp_path / "toxins.dmnd",
        database_version="current",
        work_dir=tmp_path / "toxin",
        runner=runner,
    )["g1"]
    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.execution_status == "FAILED"


def test_amrfinder_supports_nucleotide_only_hits(tmp_path: Path) -> None:
    genome = GenomeInput("g1", "ATG" * 20)
    artifacts = prepare_orf_artifacts((genome,), tmp_path / "orfs", predictor=_NoGenes())
    runtime = _tool(tmp_path, "amrfinder")

    def runner(argv, **kwargs):
        if len(argv) == 2 and argv[1] in {"version", "--version"}:
            return _completed("amrfinder 4.2.7")
        output = Path(argv[argv.index("-o") + 1])
        columns = adapters._AMRFINDER_COLUMNS[:-1]
        row = dict.fromkeys(columns, "NA")
        row.update(
            {
                "Protein id": "NA",
                "Contig id": "g1",
                "Start": "1",
                "Stop": "30",
                "Strand": "+",
                "Element symbol": "blaX",
                "Element name": "example",
                "Scope": "core",
                "Type": "AMR",
                "Method": "ALLELE",
                "Alignment length": "10",
                "Closest reference accession": "WP_1",
            }
        )
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerow(row)
        return _completed()

    result = run_amrfinder_batch(
        (genome,),
        artifacts,
        runtime=runtime,
        database=tmp_path / "amr-db",
        database_version="current",
        work_dir=tmp_path / "amr",
        runner=runner,
    )["g1"]
    assert result.class_result.state is SafetyState.FAIL
    assert result.class_result.findings[0].evidence_path == "amrfinder-nucleotide"


def test_amrfinder_failure_keeps_diagnostics(tmp_path: Path) -> None:
    genome = GenomeInput("g1", "ATG" * 20)
    artifacts = prepare_orf_artifacts((genome,), tmp_path / "orfs", predictor=_OneGene())
    runtime = _tool(tmp_path, "amrfinder")

    def runner(argv, **kwargs):
        if len(argv) == 2 and argv[1] in {"version", "--version"}:
            return _completed("amrfinder 4.2.7")
        raise subprocess.CalledProcessError(
            7,
            argv,
            output="partial tool output\n",
            stderr="database unavailable\n",
        )

    work_dir = tmp_path / "amr"
    result = run_amrfinder_batch(
        (genome,),
        artifacts,
        runtime=runtime,
        database=tmp_path / "amr-db",
        database_version="current",
        work_dir=work_dir,
        runner=runner,
    )["g1"]

    assert result.class_result.state is SafetyState.INDETERMINATE
    assert result.execution_status == "FAILED"
    assert (work_dir / "amrfinder.log").read_text() == (
        "stdout:\npartial tool output\n\nstderr:\ndatabase unavailable\n"
    )


def test_phrogs_high_confidence_hit_fails_bacterial_profile(tmp_path: Path) -> None:
    genome = GenomeInput("g1", "ATG" * 20)
    artifacts = prepare_orf_artifacts((genome,), tmp_path / "orfs", predictor=_OneGene())
    runtime = _tool(tmp_path, "mmseqs")
    lookup = tmp_path / "lookup.tsv"
    lookup.write_text(
        "phrog\tannot\tcategory\tconfidence\tmatched_term\n"
        "phrog_1\tintegrase\tintegration and excision\thigh_confidence\tintegrase\n"
    )
    primary = next(record for record in artifacts.query_records if record.evidence_path == "pyrodigal-gv")

    def runner(argv, **kwargs):
        if argv[1] == "version":
            return _completed("mmseqs 18")
        if argv[1] == "convertalis":
            Path(argv[5]).write_text(f"phrog_1\t{primary.query_id}\t80\t10\t10\t10\t1\t1\t1e-30\t100\n")
        return _completed()

    result = run_phrogs_batch(
        (genome,),
        artifacts,
        runtime=runtime,
        database=tmp_path / "profiles",
        lookup=lookup,
        database_version="PHROGs current",
        host_domain=HostDomain.BACTERIA,
        strict_lysis=True,
        work_dir=tmp_path / "phrogs-work",
        runner=runner,
    )["g1"]
    assert result.class_result.state is SafetyState.FAIL
    assert result.class_result.findings[0].accession == "phrog_1"
