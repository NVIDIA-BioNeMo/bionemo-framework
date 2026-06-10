#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Lepton Job submission script for OpenGenome2 training (v2 with git checkout support)."""

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

    # Filter out excluded nodes if specified
    if getattr(cfg, "exclude_nodes", None):
        exclude_set = set(cfg.exclude_nodes)
        original_count = len(valid_node_ids)
        valid_node_ids = valid_node_ids - exclude_set
        if original_count != len(valid_node_ids):
            print(f"  ⚠️  Excluding {original_count - len(valid_node_ids)} nodes: {sorted(exclude_set)}")
            print(f"  ✓ Remaining valid nodes: {len(valid_node_ids)}")
        if len(valid_node_ids) < cfg.num_nodes:
            raise SystemExit(
                f"ERROR: Need {cfg.num_nodes} nodes but only {len(valid_node_ids)} available after exclusions!"
            )

    return chosen_group, valid_node_ids, resource_shape


def launch_single_job(client, cfg: DictConfig):
    """Launch a single multinode job."""
    chosen_group, valid_node_ids, resource_shape = _resolve_scheduling_target(client, cfg)

    # Build Hydra overrides
    hydra_overrides = [
        f"dataset.micro_batch_size={cfg.micro_batch_size}",
        f"grad_acc_steps={cfg.grad_acc_steps}",
        f"num_train_steps={cfg.num_train_steps}",
        f'dataset.load_dataset_kwargs.path="{cfg.dataset_path}"',
    ]

    # Add data_dir if present
    if cfg.get("data_dir") and cfg.data_dir:
        hydra_overrides.append(f'dataset.load_dataset_kwargs.data_dir="{cfg.data_dir}"')

    # Add checkpoint config
    resume_from_checkpoint = cfg.get("resume_from_checkpoint", False)
    async_save = cfg.get("async_save", False)
    hydra_overrides.extend(
        [
            f"checkpoint.ckpt_dir={cfg.checkpoint_dir}",
            f"checkpoint.save_every_n_steps={cfg.save_every_n_steps}",
            f"checkpoint.resume_from_checkpoint={str(resume_from_checkpoint).lower()}",
            f"checkpoint.async_save={str(async_save).lower()}",
        ]
    )

    # Add FP8 config if present
    if "fp8_enabled" in cfg:
        hydra_overrides.append(f"fp8_config.enabled={str(cfg.fp8_enabled).lower()}")
    if "fp8_recipe" in cfg:
        hydra_overrides.append(f"fp8_config.fp8_recipe={cfg.fp8_recipe}")
    if "fp8_format" in cfg:
        hydra_overrides.append(f"fp8_config.fp8_format={cfg.fp8_format}")

    # Add init settings if specified in config
    if "spike_no_more_embedding_init" in cfg:
        hydra_overrides.append(f"spike_no_more_embedding_init={str(cfg.spike_no_more_embedding_init).lower()}")
    if "skip_embedding_weight_decay" in cfg:
        hydra_overrides.append(f"skip_embedding_weight_decay={str(cfg.skip_embedding_weight_decay).lower()}")
    if "use_megatron_scaled_init" in cfg:
        hydra_overrides.append(f"use_megatron_scaled_init={str(cfg.use_megatron_scaled_init).lower()}")
    if "use_weight_decay_grouping" in cfg:
        hydra_overrides.append(f"use_weight_decay_grouping={str(cfg.use_weight_decay_grouping).lower()}")
    if "use_megatron_loss_reduction" in cfg:
        hydra_overrides.append(f"++use_megatron_loss_reduction={str(cfg.use_megatron_loss_reduction).lower()}")

    # Add meta device init setting
    if "use_meta_device" in cfg:
        hydra_overrides.append(f"++use_meta_device={str(cfg.use_meta_device).lower()}")

    # Add FP32 master weights setting (use ++ to add or override)
    if "use_fp32_master_weights" in cfg:
        hydra_overrides.append(f"++use_fp32_master_weights={str(cfg.use_fp32_master_weights).lower()}")

    # Add num_workers if specified
    if "num_workers" in cfg:
        hydra_overrides.append(f"dataset.num_workers={cfg.num_workers}")

    # Add buffer_size if specified (for shuffle buffer override)
    if "buffer_size" in cfg:
        hydra_overrides.append(f"dataset.buffer_size={cfg.buffer_size}")

    # Add shuffle_sequences if specified (for dual shuffling)
    if "shuffle_sequences" in cfg:
        hydra_overrides.append(f"dataset.shuffle_sequences={str(cfg.shuffle_sequences).lower()}")

    # Add logger frequency (support both naming conventions)
    if "logger_frequency" in cfg:
        hydra_overrides.append(f"logger.frequency={cfg.logger_frequency}")
    elif "log_frequency" in cfg:
        hydra_overrides.append(f"logger.frequency={cfg.log_frequency}")

    # Add metrics JSON file config (for baseline comparison / Claude agent optimization)
    if "metrics_json_path" in cfg:
        hydra_overrides.append(f'++logger.metrics_json_path="{cfg.metrics_json_path}"')
    if "metrics_json_frequency" in cfg:
        hydra_overrides.append(f"++logger.metrics_json_frequency={cfg.metrics_json_frequency}")

    # Add CP size if specified (use ++ to add key if not in base config)
    if "cp_size" in cfg:
        hydra_overrides.append(f"++cp_size={cfg.cp_size}")

    # Add validation settings if specified
    if "validation_enabled" in cfg:
        hydra_overrides.append(f"++validation.enabled={str(cfg.validation_enabled).lower()}")
    if "validation_interval" in cfg:
        hydra_overrides.append(f"++validation.eval_interval={cfg.validation_interval}")

    # Add use_sequence_packing if specified
    if "use_sequence_packing" in cfg:
        hydra_overrides.append(f"++use_sequence_packing={str(cfg.use_sequence_packing).lower()}")

    # Agent control plane overrides
    agent_enabled = cfg.get("agent_enabled", False)
    if agent_enabled:
        agent_dir = cfg.get("agent_dir", "/data/agent")
        hydra_overrides.extend(
            [
                "++agent.enabled=true",
                f'++agent.control_file="{agent_dir}/control.yaml"',
                f'++agent.metrics_file="{agent_dir}/metrics.jsonl"',
                f'++agent.journal_file="{agent_dir}/journal.jsonl"',
            ]
        )

    # Add extra config_kwargs overrides (for attn_input_format, self_attn_mask_type, num_key_value_heads, etc.)
    hydra_overrides.extend(
        f"++config_kwargs.{key}={cfg[key]}"
        for key in ["attn_input_format", "self_attn_mask_type", "num_key_value_heads"]
        if key in cfg
    )

    # Add arbitrary extra hydra overrides (list of raw override strings)
    if "extra_hydra_overrides" in cfg:
        hydra_overrides.extend(str(override) for override in cfg.extra_hydra_overrides)

    # Add wandb config
    hydra_overrides.extend(
        [
            f'wandb.project="{cfg.wandb_project}"',
            f'wandb.name="{cfg.wandb_name}"',
        ]
    )

    # Format as bash command arguments (each arg on new line with backslash continuation)
    hydra_args_formatted = " \\\n  ".join(hydra_overrides)

    # Git branch checkout logic
    git_branch = cfg.get("git_branch", "")
    repo_root = cfg.get("repo_root", "/data/savithas/bionemo-framework")

    git_sync_script = ""
    if git_branch:
        git_sync_script = f"""
# Git sync to specified branch (only on master node to avoid race conditions)
if [ "$NODE_RANK" = "0" ]; then
  echo "=========================================="
  echo "[Rank 0] Syncing to branch: {git_branch}"
  echo "=========================================="
  cd {repo_root}
  # Remove any stale lock files from previous failed git operations
  find .git -name "*.lock" -delete 2>/dev/null || true
  git fetch origin
  git checkout {git_branch}
  git pull origin {git_branch}
  echo "Git sync complete!"
  echo "Current commit: $(git rev-parse HEAD)"
  # Create a marker file to signal other nodes
  echo "$(git rev-parse HEAD)" > /tmp/git_sync_complete_marker
  echo "=========================================="
else
  echo "[Rank $NODE_RANK] Waiting for rank 0 to complete git sync..."
  # Wait for rank 0 to finish (check for marker or just wait a bit)
  sleep 30
  cd {repo_root}
  echo "[Rank $NODE_RANK] Current commit: $(git rev-parse HEAD)"
fi
"""

    training_script = f"""#!/bin/bash
set -e

echo "=========================================="
echo "OpenGenome2 7B Training"
echo "Nodes: {cfg.num_nodes}"
echo "GPUs per node: {cfg.gpus_per_node}"
echo "Total GPUs: {cfg.num_nodes * cfg.gpus_per_node}"
echo "Micro batch size: {cfg.micro_batch_size}"
echo "Grad acc steps: {cfg.grad_acc_steps}"
echo "Effective batch size: {cfg.micro_batch_size * cfg.num_nodes * cfg.gpus_per_node * cfg.grad_acc_steps}"
echo "FP8: {cfg.get("fp8_enabled", False)}"
echo "FP32 Master Weights: {cfg.get("use_fp32_master_weights", False)}"
echo "Steps: {cfg.num_train_steps:,}"
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
cd {cfg.code_path}
pip install -r requirements.txt

# Ensure checkpoint directory exists
mkdir -p {cfg.checkpoint_dir}
"""
    # Add pre-training commands if specified (e.g., symlink checkpoint steps)
    if "pre_training_commands" in cfg:
        for cmd in cfg.pre_training_commands:
            training_script += f"\n{cmd}\n"
    # Ensure metrics JSON output directory exists if configured
    metrics_json_path = cfg.get("metrics_json_path", "")
    if metrics_json_path:
        training_script += f"""
# Ensure metrics JSON output directory exists
mkdir -p $(dirname {metrics_json_path})
"""
    training_script += f"""

python -c "from huggingface_hub import login; login(token='${{HF_TOKEN}}')"
wandb login ${{WANDB_API_KEY}}

echo "=========================================="
echo "Starting multinode training..."
echo "=========================================="

torchrun \\
  --nnodes=$NNODES \\
  --nproc_per_node={cfg.gpus_per_node} \\
  --node_rank=$NODE_RANK \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  {cfg.train_script} \\
  --config-name={cfg.hydra_config} \\
  {hydra_args_formatted}

echo "=========================================="
echo "Training completed!"
echo "=========================================="
"""

    command = ["bash", "-c", training_script]

    env_vars = [
        EnvVar(name="WANDB_API_KEY", value_from=EnvValue(secret_name_ref=cfg.wandb_secret)),
        EnvVar(name="HF_TOKEN", value_from=EnvValue(secret_name_ref=cfg.hf_secret)),
    ]

    # Use NFS settings from config, with defaults for H100
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
        completions=cfg.num_nodes,
        parallelism=cfg.num_nodes,
        envs=env_vars,
        image_pull_secrets=[cfg.container.registry_auth],
        mounts=mounts,
    )

    job = LeptonJob(spec=job_spec, metadata=Metadata(id=cfg.job_name))

    try:
        launched_job = client.job.create(job)
        if launched_job.status:
            print(f"  ✓ Job launched: {cfg.job_name}")
            workspace_id = cfg.get("workspace_id", "vfco61g2")
            print(
                f"    View at: https://dashboard.dgxc-lepton.nvidia.com/workspace/{workspace_id}/compute/jobs/detail/{launched_job.metadata.id_}/replicas/list"
            )
            return True
    except Exception as e:
        print(f"  ERROR submitting job {cfg.job_name}: {e}")
        return False


def launch_agent_sidecar(client, cfg: DictConfig):
    """Launch the agent daemon as a CPU-only sidecar job with NFS access.

    The agent monitors training metrics on shared NFS and can write control
    files for hot-reload interventions. It does not require GPUs.
    """
    chosen_group, valid_node_ids, _ = _resolve_scheduling_target(client, cfg)

    agent_dir = cfg.get("agent_dir", "/data/agent")
    code_path = cfg.get(
        "code_path", "/data/savithas/bionemo-framework/bionemo-recipes/recipes/opengenome2_llama_native_te"
    )
    repo_root = cfg.get("repo_root", "/data/savithas/bionemo-framework")
    git_branch = cfg.get("git_branch", "")
    observe_only_flag = "" if cfg.get("agent_intervene", False) else "--observe-only"

    git_sync_agent = ""
    if git_branch:
        git_sync_agent = f"""
cd {repo_root}
find .git -name "*.lock" -delete 2>/dev/null || true
git fetch origin
git checkout {git_branch}
git pull origin {git_branch}
echo "Agent: git sync to {git_branch} complete ($(git rev-parse HEAD))"
"""

    agent_script = f"""#!/bin/bash
set -e

echo "=========================================="
echo "Agent Daemon Sidecar"
echo "Monitoring: {agent_dir}/metrics.jsonl"
echo "Control:    {agent_dir}/control.yaml"
echo "=========================================="

{git_sync_agent}

mkdir -p {agent_dir}
cd {code_path}
pip install pyyaml

echo "Starting agent daemon..."
python agent_daemon.py \\
  --metrics-file {agent_dir}/metrics.jsonl \\
  --control-file {agent_dir}/control.yaml \\
  --journal-file {agent_dir}/journal.jsonl \\
  --fp8-stats-dir {agent_dir}/fp8_stats \\
  --monitor-interval 30 \\
  {observe_only_flag}
"""

    command = ["bash", "-c", agent_script]

    nfs_source_path = cfg.get("nfs", {}).get("source_path", "/BioNeMo")
    nfs_mount_path = cfg.get("nfs", {}).get("mount_path", "/data")
    nfs_source = cfg.get("nfs", {}).get("nfs_source", "node-nfs:fs1")

    mounts = [{"path": nfs_source_path, "mount_path": nfs_mount_path, "from": nfs_source}]

    agent_resource_shape = cfg.get("agent_resource_shape", "cpu.small")

    job_spec = LeptonJobUserSpec(
        resource_shape=agent_resource_shape,
        affinity=LeptonResourceAffinity(
            allowed_dedicated_node_groups=[chosen_group.metadata.id_],
        ),
        container=LeptonContainer(
            image=cfg.container.image,
            command=command,
        ),
        completions=1,
        parallelism=1,
        envs=[],
        image_pull_secrets=[cfg.container.registry_auth],
        mounts=mounts,
    )

    agent_job_name = f"{cfg.job_name}-agent"
    job = LeptonJob(spec=job_spec, metadata=Metadata(id=agent_job_name))

    try:
        launched_job = client.job.create(job)
        if launched_job.status:
            print(f"  Agent sidecar launched: {agent_job_name}")
            return True
    except Exception as e:
        print(f"  ERROR submitting agent sidecar {agent_job_name}: {e}")
        return False


@hydra.main(version_base=None, config_path="lepton_configs", config_name="og2_7b_thd_fp8_mixed")
def main(cfg: DictConfig):
    """Main function to launch OG2 job."""
    print("=" * 60)
    print(f"Launching OpenGenome2 job: {cfg.job_name}")
    print("=" * 60)

    # Show git branch if specified
    if cfg.get("git_branch"):
        print(f"  Will checkout branch: {cfg.git_branch}")

    client = APIClient()
    OmegaConf.resolve(cfg)

    success = launch_single_job(client, cfg)
    if not success:
        print("\n Job submission failed!")
        exit(1)

    print("\n Training job submitted successfully!")

    # Launch agent sidecar if enabled
    if cfg.get("agent_enabled", False):
        print("\nLaunching agent sidecar...")
        if launch_agent_sidecar(client, cfg):
            print("  Agent sidecar submitted successfully!")
        else:
            print("  WARNING: Agent sidecar submission failed (training continues without agent)")


if __name__ == "__main__":
    main()
