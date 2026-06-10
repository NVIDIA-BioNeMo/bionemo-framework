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

"""Simple Lepton job submission for non-agent training runs.

Submits a single- or multi-node torchrun job without the Claude agent.
Used for running BF16 baselines or other standard training jobs.

Usage:
    python submit_training_lepton.py --config-name=og2_bf16_baseline_1node
"""

import hydra
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import EnvValue, EnvVar, LeptonContainer
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
from leptonai.api.v2.client import APIClient
from omegaconf import DictConfig, OmegaConf


def _resolve_scheduling_target(client, cfg: DictConfig):
    """Resolve node group and resource shape."""
    desired_node_group = str(cfg.node_group).strip()
    resource_shape = str(cfg.resource_shape).strip()

    node_groups = client.nodegroup.list_all()
    node_group_map = {ng.metadata.name: ng for ng in node_groups}

    if desired_node_group not in node_group_map:
        available = ", ".join(sorted(node_group_map.keys()))
        raise SystemExit(f"Node group '{desired_node_group}' not found.\nAvailable: {available}")

    chosen_group = node_group_map[desired_node_group]
    valid_node_ids = {n.metadata.id_ for n in client.nodegroup.list_nodes(chosen_group.metadata.id_)}

    return chosen_group, valid_node_ids, resource_shape


def launch_training_job(client, cfg: DictConfig):
    """Launch a standard training job (no Claude agent)."""
    chosen_group, valid_node_ids, resource_shape = _resolve_scheduling_target(client, cfg)

    num_nodes = cfg.get("num_nodes", 1)
    git_branch = cfg.get("git_branch", "")
    repo_root = cfg.get("repo_root", "/data/savithas/bionemo-framework")

    git_sync_script = ""
    if git_branch:
        git_sync_script = f"""
# Git sync to specified branch
cd {repo_root}
find .git -name "*.lock" -delete 2>/dev/null || true
git fetch origin
git checkout {git_branch} 2>/dev/null || git checkout -b {git_branch} origin/{git_branch}
git reset --hard origin/{git_branch}
echo "Git sync complete! Commit: $(git rev-parse HEAD)"
"""

    # Build checkpoint symlink setup if resuming from external checkpoint
    resume_setup_script = ""
    resume_from = cfg.get("resume_from", None)
    if resume_from and resume_from.get("external_checkpoint"):
        ckpt_dir = cfg.hydra_overrides.get("checkpoint.ckpt_dir", "")
        ext_ckpt = resume_from.external_checkpoint
        step = resume_from.step
        resume_setup_script = f"""
# Symlink external checkpoint for resume
mkdir -p {ckpt_dir}/train_fsdp2
if [ ! -e {ckpt_dir}/train_fsdp2/step_{step} ]; then
    ln -s {ext_ckpt}/train_fsdp2/step_{step} {ckpt_dir}/train_fsdp2/step_{step}
    echo "Symlinked checkpoint: {ext_ckpt}/train_fsdp2/step_{step}"
else
    echo "Checkpoint already exists at {ckpt_dir}/train_fsdp2/step_{step}"
fi
ls -la {ckpt_dir}/train_fsdp2/
"""

    # Build the torchrun command from Hydra overrides
    hydra_overrides = OmegaConf.to_container(cfg.get("hydra_overrides", {}), resolve=True) or {}
    override_str = " \\\n  ".join(f"{k}={v}" for k, v in hydra_overrides.items())

    container_script = f"""#!/bin/bash
set -e

echo "=========================================="
echo "OpenGenome2 Training Job"
echo "Node rank: $NODE_RANK / $NNODES"
echo "GPUs: {cfg.gpus_per_node}x H100"
echo "=========================================="

# Initialize Lepton environment
wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh
chmod +x init.sh
source init.sh

export MASTER_PORT=29400
export NCCL_TIMEOUT_MS=1800000
export NCCL_DEBUG=WARN
export HF_HOME=/data/savithas/cache
{git_sync_script}

# Install requirements
cd {cfg.code_path}
pip install -r requirements.txt

# Login to wandb
wandb login ${{WANDB_API_KEY}}

echo "=========================================="
echo "Launching torchrun training..."
echo "Config: {cfg.hydra_config}"
echo "=========================================="
{resume_setup_script}
cd {cfg.code_path}
torchrun \\
  --nproc_per_node={cfg.gpus_per_node} \\
  --nnodes=$NNODES \\
  --node_rank=$NODE_RANK \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  {cfg.train_script} \\
  --config-name {cfg.hydra_config} \\
  {override_str}

echo "=========================================="
echo "Training complete!"
echo "=========================================="
"""

    command = ["bash", "-c", container_script]

    env_vars = [
        EnvVar(name="WANDB_API_KEY", value_from=EnvValue(secret_name_ref=cfg.wandb_secret)),
    ]

    nfs_source_path = cfg.get("nfs", {}).get("source_path", "/BioNeMo")
    nfs_mount_path = cfg.get("nfs", {}).get("mount_path", "/data")
    nfs_source = cfg.get("nfs", {}).get("nfs_source", "node-nfs:fs1")

    mounts = [
        {
            "path": nfs_source_path,
            "mount_path": nfs_mount_path,
            "from": nfs_source,
        },
    ]

    job_spec = LeptonJobUserSpec(
        resource_shape=resource_shape,
        affinity=LeptonResourceAffinity(
            allowed_dedicated_node_groups=[chosen_group.metadata.id_],
            allowed_nodes_in_node_group=valid_node_ids,
        ),
        container=LeptonContainer(
            image=cfg.container.image,
            command=command,
        ),
        completions=num_nodes,
        parallelism=num_nodes,
        envs=env_vars,
        image_pull_secrets=[cfg.container.registry_auth],
        mounts=mounts,
    )

    job = LeptonJob(spec=job_spec, metadata=Metadata(id=cfg.job_name))

    try:
        launched_job = client.job.create(job)
        if launched_job.status:
            print(f"  Job launched: {cfg.job_name}")
            workspace_id = cfg.get("workspace_id", "vfco61g2")
            print(
                f"  View at: https://dashboard.dgxc-lepton.nvidia.com/workspace/"
                f"{workspace_id}/compute/jobs/detail/{launched_job.metadata.id_}/replicas/list"
            )
            return True
    except Exception as e:
        print(f"  ERROR submitting job {cfg.job_name}: {e}")
        return False


@hydra.main(version_base=None, config_path="lepton_configs", config_name="og2_bf16_baseline_1node")
def main(cfg: DictConfig):
    """Submit a standard training job to Lepton."""
    print("=" * 60)
    print(f"Training Job: {cfg.job_name}")
    print("=" * 60)
    print(f"  Nodes: {cfg.num_nodes} x {cfg.gpus_per_node} GPUs")
    print(f"  Config: {cfg.hydra_config}")
    if cfg.get("git_branch"):
        print(f"  Git branch: {cfg.git_branch}")
    print()

    client = APIClient()
    OmegaConf.resolve(cfg)

    success = launch_training_job(client, cfg)
    if not success:
        print("\nJob submission failed!")
        exit(1)

    print("\nTraining job submitted successfully!")


if __name__ == "__main__":
    main()
