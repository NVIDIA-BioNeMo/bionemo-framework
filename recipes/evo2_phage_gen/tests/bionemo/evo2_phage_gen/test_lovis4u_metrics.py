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
