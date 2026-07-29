# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from bionemo.evo2_phage_gen.lovis4u_metrics import _cluster_command_with_threads


def test_cluster_command_applies_tunable_threads_only_to_cluster():
    cluster = ["mmseqs", "cluster", "query", "result", "tmp"]
    createdb = ["mmseqs", "createdb", "input", "query"]

    assert _cluster_command_with_threads(cluster, 8) == [*cluster, "--threads", "8"]
    assert _cluster_command_with_threads(createdb, 8) == createdb
    assert _cluster_command_with_threads(cluster, None) == cluster


def test_cluster_command_preserves_explicit_threads():
    command = ["mmseqs", "cluster", "query", "result", "tmp", "--threads", "4"]

    assert _cluster_command_with_threads(command, 8) == command
