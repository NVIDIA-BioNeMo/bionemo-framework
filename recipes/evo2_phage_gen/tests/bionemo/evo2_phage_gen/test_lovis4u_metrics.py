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

import tomllib
from collections import UserList
from pathlib import Path
from types import SimpleNamespace

import bionemo.evo2_phage_gen.lovis4u_metrics as lovis4u_metrics
from bionemo.evo2_phage_gen.lovis4u_metrics import _cluster_command_with_threads, _ThreadedSubprocess


def test_cluster_command_applies_tunable_threads_only_to_cluster():
    cluster = ["mmseqs", "cluster", "query", "result", "tmp"]
    createdb = ["mmseqs", "createdb", "input", "query"]

    assert _cluster_command_with_threads(cluster, 8) == [*cluster, "--threads", "8"]
    assert _cluster_command_with_threads(createdb, 8) == createdb
    assert _cluster_command_with_threads(cluster, None) == cluster


def test_cluster_command_preserves_explicit_threads():
    command = ["mmseqs", "cluster", "query", "result", "tmp", "--threads", "4"]

    assert _cluster_command_with_threads(command, 8) == command


def test_cluster_command_preserves_string_commands():
    command = "mmseqs cluster query result tmp"

    assert _cluster_command_with_threads(command, 8) is command


def test_cluster_command_preserves_other_command_containers():
    command = UserList(["mmseqs", "cluster", "query", "result", "tmp"])

    assert _cluster_command_with_threads(command, 8) is command


def test_threaded_subprocess_proxy_delegates_without_mutating_wrapped_module():
    calls = []

    class Wrapped:
        PIPE = object()

        @staticmethod
        def run(command, **kwargs):
            calls.append((command, kwargs))
            return "completed"

    proxy = _ThreadedSubprocess(Wrapped, 6)

    assert proxy.run(["mmseqs", "cluster", "query", "result", "tmp"], check=True) == "completed"
    assert calls == [(["mmseqs", "cluster", "query", "result", "tmp", "--threads", "6"], {"check": True})]
    assert proxy.PIPE is Wrapped.PIPE
    assert Wrapped.run(["mmseqs", "createdb"], check=False) == "completed"
    assert calls[-1][0] == ["mmseqs", "createdb"]


def test_lovis4u_dependency_version():
    recipe_root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((recipe_root / "pyproject.toml").read_text())

    assert "lovis4u==0.2.0" in project["project"]["dependencies"]


def test_main_runs_mmseqs_clustering_with_expected_configuration(monkeypatch):
    calls = []

    class Parameters:
        def __init__(self):
            self.cmd_arguments = {"config_file": "config.ini"}
            self.args = {"verbose": True, "gff": ["input.gff"], "gb": None, "mmseqs": True}

        def parse_cmd_arguments(self):
            calls.append("parse")

        def load_config(self, path):
            calls.append(("load_config", path))

    class Loci:
        def __init__(self, *, parameters):
            calls.append(("loci", parameters.args.copy()))

        def load_loci_from_extended_gff(self, paths):
            calls.append(("gff", paths))

        def mmseqs_cluster(self):
            calls.append("cluster")

    original_subprocess = object()
    fake_lovis4u = SimpleNamespace(
        Manager=SimpleNamespace(Parameters=Parameters),
        DataProcessing=SimpleNamespace(Loci=Loci, subprocess=original_subprocess),
    )
    monkeypatch.setattr(lovis4u_metrics, "lovis4u", fake_lovis4u)
    monkeypatch.setenv("LOVIS4U_MMSEQS_BINARY", "/tools/mmseqs")
    monkeypatch.setenv("LOVIS4U_MMSEQS_THREADS", "6")

    lovis4u_metrics.main()

    loci_args = next(item[1] for item in calls if isinstance(item, tuple) and item[0] == "loci")
    assert loci_args["verbose"] is False
    assert loci_args["mmseqs_binary"] == "/tools/mmseqs"
    assert calls[-2:] == [("gff", ["input.gff"]), "cluster"]
    assert fake_lovis4u.DataProcessing.subprocess is original_subprocess
