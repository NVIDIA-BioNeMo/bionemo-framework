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

"""Run the LoVis4u clustering artifacts consumed by phage synteny scoring."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import lovis4u


def _cluster_command_with_threads(command: Sequence[str] | str, threads: int | None) -> Sequence[str] | str:
    """Apply the configured thread count only to the LoVis4u MMseqs cluster call."""
    if not isinstance(command, (list, tuple)):
        return command
    updated = list(command)
    if threads is not None and len(updated) > 1 and updated[1] == "cluster" and "--threads" not in updated:
        updated.extend(["--threads", str(threads)])
    return updated


class _ThreadedSubprocess:
    """Delegate subprocess attributes while adapting only MMseqs cluster calls."""

    def __init__(self, wrapped: Any, threads: int | None):
        self._wrapped = wrapped
        self._threads = threads

    def run(self, command: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        """Run one command with the configured cluster thread override."""
        return self._wrapped.run(_cluster_command_with_threads(command, self._threads), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate constants and helpers to the wrapped subprocess module."""
        return getattr(self._wrapped, name)


def main() -> None:
    """Parse normal LoVis4u arguments, cluster proteins, and skip unconsumed rendering."""
    parameters = lovis4u.Manager.Parameters()
    parameters.parse_cmd_arguments()
    parameters.load_config(parameters.cmd_arguments["config_file"])
    parameters.args["verbose"] = False

    mmseqs_binary = os.environ.get("LOVIS4U_MMSEQS_BINARY")
    if mmseqs_binary:
        parameters.args["mmseqs_binary"] = mmseqs_binary

    raw_threads = os.environ.get("LOVIS4U_MMSEQS_THREADS")
    threads = int(raw_threads) if raw_threads else None
    if threads is not None and threads < 1:
        raise ValueError("LOVIS4U_MMSEQS_THREADS must be positive")

    original_subprocess = lovis4u.DataProcessing.subprocess
    lovis4u.DataProcessing.subprocess = _ThreadedSubprocess(original_subprocess, threads)
    try:
        loci = lovis4u.DataProcessing.Loci(parameters=parameters)
        if parameters.args["gff"]:
            loci.load_loci_from_extended_gff(parameters.args["gff"])
        elif parameters.args["gb"]:
            loci.load_loci_from_gb(parameters.args["gb"])
        else:
            raise ValueError("LoVis4u metrics-only mode requires -gff or -gb")
        if not parameters.args["mmseqs"]:
            raise ValueError("Phage synteny scoring requires LoVis4u MMseqs clustering")
        loci.mmseqs_cluster()
    finally:
        lovis4u.DataProcessing.subprocess = original_subprocess


if __name__ == "__main__":
    main()
