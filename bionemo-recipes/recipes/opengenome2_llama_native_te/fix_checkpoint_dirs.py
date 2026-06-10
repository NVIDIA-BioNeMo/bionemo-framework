#!/usr/bin/env python3

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

"""Quick Lepton job to rename old checkpoint directories so current runs can save cleanly.

Usage:
    python fix_checkpoint_dirs.py
"""

from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import LeptonContainer
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
from leptonai.api.v2.client import APIClient


NODE_GROUP = "yo-bom-lepton-001"
RESOURCE_SHAPE = "gpu.8xh100-sxm"
JOB_NAME = "fix-checkpoint-dirs"

SCRIPT = r"""#!/bin/bash
set -e

echo "=========================================="
echo "Fixing checkpoint directories on NFS"
echo "=========================================="

# --- fl2: move old run checkpoints, create fresh dir ---
FL2="/data/savithas/checkpoints/og2-7b-fp8-refactor-bf16-fl2-fp32mw"  # pragma: allowlist secret
FL2_OLD="${FL2}-old-run"

echo ""
echo "=== fl2 directory ==="
echo "Current contents:"
ls -la ${FL2}/train_fsdp2/ 2>/dev/null || echo "(directory empty or missing)"

if [ -d "${FL2}/train_fsdp2/step_15000" ]; then
    echo "Old run checkpoints detected (step_15000+). Moving..."
    mv "${FL2}" "${FL2_OLD}"
    echo "Moved to: ${FL2_OLD}"
    mkdir -p "${FL2}/train_fsdp2"
    echo "Created fresh: ${FL2}/train_fsdp2/"
else
    echo "No old run checkpoints found (no step_15000). Checking what's there..."
    ls -la ${FL2}/train_fsdp2/ 2>/dev/null || echo "(empty)"
fi

echo ""
echo "fl2 after fix:"
ls -la ${FL2}/train_fsdp2/ 2>/dev/null || echo "(empty - ready for new checkpoints)"

# --- fl4: check if 5k checkpoint exists ---
FL4="/data/savithas/checkpoints/og2-7b-fp8-refactor-bf16-fl4-fp32mw"  # pragma: allowlist secret

echo ""
echo "=== fl4 directory ==="
echo "Current contents:"
ls -la ${FL4}/train_fsdp2/ 2>/dev/null || echo "(directory empty or missing)"

if [ -d "${FL4}/train_fsdp2/step_5000" ]; then
    echo "step_5000 checkpoint EXISTS in fl4!"
else
    echo "step_5000 NOT found in fl4."
    # Check if old run checkpoints are clobbering this one too
    if [ -d "${FL4}/train_fsdp2/step_15000" ] || [ -d "${FL4}/train_fsdp2/step_20000" ]; then
        FL4_OLD="${FL4}-old-run"
        echo "Old run checkpoints detected. Moving..."
        mv "${FL4}" "${FL4_OLD}"
        echo "Moved to: ${FL4_OLD}"
        mkdir -p "${FL4}/train_fsdp2"
        echo "Created fresh: ${FL4}/train_fsdp2/"
    fi
fi

echo ""
echo "=========================================="
echo "Done! Summary:"
echo "=========================================="
echo "fl2: $(ls ${FL2}/train_fsdp2/ 2>/dev/null | tr '\n' ' ' || echo 'empty')"
echo "fl4: $(ls ${FL4}/train_fsdp2/ 2>/dev/null | tr '\n' ' ' || echo 'empty')"
if [ -d "${FL2_OLD}" ]; then
    echo "fl2 old run: $(ls ${FL2_OLD}/train_fsdp2/ 2>/dev/null | tr '\n' ' ')"
fi
if [ -d "${FL4_OLD}" ]; then
    echo "fl4 old run: $(ls ${FL4_OLD}/train_fsdp2/ 2>/dev/null | tr '\n' ' ')"
fi
echo "=========================================="
"""


def main():
    """Submit a quick Lepton job to fix checkpoint directories."""
    client = APIClient()

    node_groups = client.nodegroup.list_all()
    node_group_map = {ng.metadata.name: ng for ng in node_groups}

    if NODE_GROUP not in node_group_map:
        available = ", ".join(sorted(node_group_map.keys()))
        raise SystemExit(f"Node group '{NODE_GROUP}' not found.\nAvailable: {available}")

    chosen_group = node_group_map[NODE_GROUP]
    valid_node_ids = {n.metadata.id_ for n in client.nodegroup.list_nodes(chosen_group.metadata.id_)}

    job_spec = LeptonJobUserSpec(
        resource_shape=RESOURCE_SHAPE,
        affinity=LeptonResourceAffinity(
            allowed_dedicated_node_groups=[chosen_group.metadata.id_],
            allowed_nodes_in_node_group=valid_node_ids,
        ),
        container=LeptonContainer(
            image="nvcr.io/nvidia/pytorch:26.02-py3",
            command=["bash", "-c", SCRIPT],
        ),
        completions=1,
        parallelism=1,
        envs=[],
        image_pull_secrets=["lepton-nvidia-cvai-bnmo-trng"],
        mounts=[
            {
                "path": "/BioNeMo",
                "mount_path": "/data",
                "from": "node-nfs:fs1",
            },
        ],
    )

    job = LeptonJob(spec=job_spec, metadata=Metadata(id=JOB_NAME))

    try:
        launched_job = client.job.create(job)
        if launched_job.status:
            print(f"Job launched: {JOB_NAME}")
            print(
                f"View at: https://dashboard.dgxc-lepton.nvidia.com/workspace/"
                f"vfco61g2/compute/jobs/detail/{launched_job.metadata.id_}/replicas/list"
            )
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)


if __name__ == "__main__":
    main()
