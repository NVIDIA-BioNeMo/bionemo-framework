# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Run the LoVis4u clustering artifacts consumed by phage synteny scoring."""

from __future__ import annotations

import os
from collections.abc import Sequence

import lovis4u


def _cluster_command_with_threads(command: Sequence[str], threads: int | None) -> list[str]:
    """Apply the configured thread count only to the LoVis4u MMseqs cluster call."""
    updated = list(command)
    if threads is not None and len(updated) > 1 and updated[1] == "cluster" and "--threads" not in updated:
        updated.extend(["--threads", str(threads)])
    return updated


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

    original_run = lovis4u.DataProcessing.subprocess.run

    def run_with_threads(command, *args, **kwargs):
        return original_run(_cluster_command_with_threads(command, threads), *args, **kwargs)

    lovis4u.DataProcessing.subprocess.run = run_with_threads
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
        lovis4u.DataProcessing.subprocess.run = original_run


if __name__ == "__main__":
    main()
