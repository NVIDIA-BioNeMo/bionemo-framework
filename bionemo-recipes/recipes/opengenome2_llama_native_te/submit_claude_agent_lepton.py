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

"""Lepton job submission script that runs Claude Code as an autonomous FP8 Precision Agent.

On rank 0, Claude Code is installed and given the OG2 FP8 Agent Guide. Claude autonomously
manages the training loop: launching torchrun, monitoring metrics, adjusting layer precision,
and producing reports. On other ranks, nodes wait for torchrun commands from rank 0.

Usage:
    # MVP demo (single node, tiny model)
    python submit_claude_agent_lepton.py --config-name=claude_agent_demo

    # Full FP8 agent (multi-node, OG2-7B)
    python submit_claude_agent_lepton.py --config-name=og2_fp8_agent

    # Override strategy
    python submit_claude_agent_lepton.py --config-name=og2_fp8_agent promotion_strategy=tail_in
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


def _build_warm_start_section(cfg: DictConfig) -> str:
    """Build the warm-start section of the agent prompt, or a fresh-start message."""
    warm_start = cfg.get("warm_start", None)
    if warm_start is None or not getattr(warm_start, "enabled", False):
        return "## Start Mode\n\nFresh start — begin from scratch with all layers in $INITIAL_PRECISION."

    fp8_layers = list(OmegaConf.to_container(warm_start.fp8_layers, resolve=True))
    strategy = cfg.get("promotion_strategy", "ends_in")

    # Compute which layers are BF16 (not in fp8_layers)
    num_layers = 32  # OG2-7B has 32 transformer layers
    bf16_layers = sorted(set(range(1, num_layers + 1)) - set(fp8_layers))

    lines = [
        "## Warm Start from Existing Checkpoint",
        "",
        "This run resumes from an existing checkpoint with a pre-configured precision schedule.",
        f"Layers {bf16_layers} are already in BF16; layers {fp8_layers} are in FP8.",
        "Do NOT start from scratch. Follow the warm-start procedure below.",
        "",
        "```",
        f"EXTERNAL_CHECKPOINT    = {warm_start.external_checkpoint}",
        f"LKG_STEP               = {warm_start.lkg_step}",
        f"INITIAL_FP8_LAYERS     = {fp8_layers}",
        f"DEMOTION_ROUND         = {warm_start.demotion_round}",
    ]

    if strategy == "ends_in":
        lines += [
            f"BOTTOM_PTR             = {warm_start.bottom_ptr}",
            f"TOP_PTR                = {warm_start.top_ptr}",
        ]
    elif strategy == "tail_in":
        lines += [
            f"TAIL_PTR               = {warm_start.tail_ptr}",
        ]
    elif strategy == "gradual":
        lines += [
            f"CENTER_LOW             = {warm_start.get('center_low', 16)}",
            f"CENTER_HIGH            = {warm_start.get('center_high', 17)}",
        ]

    checkpoint_root = cfg.get("checkpoint_root", "/data/savithas/checkpoints")

    lines += [
        "```",
        "",
        "### Warm-Start Procedure",
        "",
        "Before your first training launch:",
        "",
        f"**IMPORTANT**: Use `CHECKPOINT_ROOT={checkpoint_root}` for checkpoint paths,",
        "NOT `WORKSPACE_ROOT`. The `checkpoint.ckpt_dir` CLI argument must be",
        f"`{checkpoint_root}/<run_name>` (e.g., `{checkpoint_root}/ends_in_20260318_143000`).",
        "",
        "1. Create your checkpoint directory:",
        "   ```",
        f"   mkdir -p {checkpoint_root}/<run_name>/train_fsdp2",
        "   ```",
        "2. Symlink the external checkpoint into your checkpoint directory:",
        "   ```",
        f"   ln -s {warm_start.external_checkpoint}/train_fsdp2/step_{warm_start.lkg_step}"
        f" {checkpoint_root}/<run_name>/train_fsdp2/step_{warm_start.lkg_step}",
        "   ```",
        "3. Verify the symlink resolves correctly:",
        "   ```",
        f"   ls -la {checkpoint_root}/<run_name>/train_fsdp2/step_{warm_start.lkg_step}/",
        f"   ls {checkpoint_root}/<run_name>/train_fsdp2/step_{warm_start.lkg_step}/.metadata",
        "   ```",
        f"4. Set `fp8_layers` to `{fp8_layers}` for the first launch.",
        f"5. Initialize `state.json` with `demotion_round={warm_start.demotion_round}`, "
        f"`lkg_step={warm_start.lkg_step}`, and the pointer values above.",
        "",
        f"The agent picks up at round {warm_start.demotion_round + 1} of `{strategy}`.",
        f"On the first check-in (step {warm_start.lkg_step + cfg.get('checkin_interval', 100)}), "
        "compare against the BF16 baseline at that step.",
        "If the check-in passes, training continues. If it fails, demote the next layers per the strategy.",
        "",
        "### Important",
        "",
        "- The external checkpoint was trained with a DIFFERENT precision schedule. "
        "The optimizer state is matched to that schedule. Only further demotions (FP8 -> BF16) are safe. "
        "Do NOT promote layers back to FP8 that were already in BF16.",
        f"- `checkpoint.ckpt_dir` stays FIXED at `{checkpoint_root}/<run_name>` for the entire session "
        "(same rule as fresh start). NEVER use WORKSPACE_ROOT for checkpoint.ckpt_dir.",
    ]

    return "\n".join(lines)


def _build_agent_prompt(cfg: DictConfig) -> str:
    """Read claude_agent_prompt.txt and fill in config values."""
    import pathlib

    prompt_path = pathlib.Path(__file__).parent / "claude_agent_prompt.txt"
    template = prompt_path.read_text()

    warm_start_section = _build_warm_start_section(cfg)

    workspace_root = cfg.get("workspace_root", "/data/savithas/agent_runs")
    launch_dir = f"{workspace_root}/.launches/{cfg.job_name}"

    # Choose guide based on strategy
    strategy = cfg.get("promotion_strategy", "ends_in")
    guide_filename = cfg.get("guide_filename", None)
    if guide_filename is None:
        guide_filename = "OG2_FP8_1NODE_DEMO_GUIDE.md" if strategy == "gradual" else "OG2_FP8_AGENT_GUIDE.md"

    return template.format(
        code_path=cfg.code_path,
        num_nodes=cfg.num_nodes,
        gpus_per_node=cfg.gpus_per_node,
        num_train_steps=cfg.num_train_steps,
        checkin_interval=cfg.get("checkin_interval", 100),
        tolerance_pct=cfg.get("tolerance_pct", 5.0),
        promotion_strategy=strategy,
        workspace_root=workspace_root,
        checkpoint_root=cfg.get("checkpoint_root", "/data/savithas/checkpoints"),
        wandb_project=cfg.get("wandb_project", "opengenome2-7b"),
        launch_dir=launch_dir,
        warm_start_section=warm_start_section,
        guide_filename=guide_filename,
        baseline_logfile=cfg.get("baseline_logfile", "baseline_bf16.json"),
    )


def launch_claude_agent_job(client, cfg: DictConfig):
    """Launch a multi-node job where rank 0 runs Claude Code as the FP8 Precision Agent."""
    chosen_group, valid_node_ids, resource_shape = _resolve_scheduling_target(client, cfg)

    agent_prompt = _build_agent_prompt(cfg)
    num_nodes = cfg.get("num_nodes", 1)
    workspace_root = cfg.get("workspace_root", "/data/savithas/agent_runs")
    launch_dir = f"{workspace_root}/.launches/{cfg.job_name}"

    # Git branch checkout logic (rank 0 only for NFS safety)
    git_branch = cfg.get("git_branch", "")
    repo_root = cfg.get("repo_root", "/data/savithas/bionemo-framework")

    git_sync_script = ""
    if git_branch:
        git_sync_script = f"""
# Git sync to specified branch (only on rank 0 to avoid NFS race conditions)
if [ "$NODE_RANK" = "0" ]; then
  echo "=========================================="
  echo "[Rank 0] Syncing to branch: {git_branch}"
  echo "=========================================="
  cd {repo_root}
  find .git -name "*.lock" -delete 2>/dev/null || true
  git fetch origin
  git checkout {git_branch} 2>/dev/null || git checkout -b {git_branch} origin/{git_branch}
  git reset --hard origin/{git_branch}
  echo "Git sync complete! Commit: $(git rev-parse HEAD)"
  echo "=========================================="
else
  echo "[Rank $NODE_RANK] Waiting for rank 0 to complete git sync..."
  sleep 30
  cd {repo_root}
  echo "[Rank $NODE_RANK] Current commit: $(git rev-parse HEAD)"
fi
"""

    container_script = f"""#!/bin/bash
set -e

echo "=========================================="
echo "Claude Code FP8 Precision Agent - Lepton"
echo "Node rank: $NODE_RANK / $NNODES"
echo "GPUs: {cfg.gpus_per_node}x H100"
echo "=========================================="

# 1. Initialize Lepton environment (sets MASTER_ADDR, MASTER_PORT, NODE_RANK, NNODES)
wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh
chmod +x init.sh
source init.sh

export MASTER_PORT=29400
export NCCL_TIMEOUT_MS=1800000
export NCCL_DEBUG=WARN
export HF_HOME=/data/savithas/cache

# Write env vars to a file so they survive the `su` to claude-agent on rank 0.
# Without this, Claude Code's torchrun would get empty MASTER_ADDR/NODE_RANK/NNODES.
cat > /tmp/training_env.sh << ENV_EOF
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$NODE_RANK
export NNODES=$NNODES
export NCCL_TIMEOUT_MS=$NCCL_TIMEOUT_MS
export NCCL_DEBUG=$NCCL_DEBUG
export HF_HOME=$HF_HOME
export WANDB_API_KEY=$WANDB_API_KEY
export PATH=$PATH
export LD_LIBRARY_PATH=${{LD_LIBRARY_PATH:-}}
export CUDA_HOME=${{CUDA_HOME:-}}

# Pre-built torchrun prefix — Claude MUST use this instead of constructing its own.
# This ensures correct --nnodes, --node_rank, --master_addr for multi-node training.
export TORCHRUN_PREFIX="torchrun --nproc_per_node={cfg.gpus_per_node} --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
ENV_EOF
chmod 644 /tmp/training_env.sh
echo "Env vars written to /tmp/training_env.sh"
echo "  MASTER_ADDR=$MASTER_ADDR NODE_RANK=$NODE_RANK NNODES=$NNODES"
echo "  TORCHRUN_PREFIX will be: torchrun --nproc_per_node={cfg.gpus_per_node} --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
{git_sync_script}

# 2. Install Python training requirements (all nodes)
cd {cfg.code_path}
pip install -r requirements.txt

# 3. Login to wandb (all nodes, needed for distributed logging)
wandb login ${{WANDB_API_KEY}}

# 4. Create workspace and launch coordination directories
mkdir -p {workspace_root}
mkdir -p {launch_dir}

# Clean stale round files from previous job runs (same job_name reuses the launch dir on NFS).
# Without this, workers immediately pick up old round_1_ready and start torchrun before rank 0 is ready.
if [ "$NODE_RANK" = "0" ]; then
  echo "Cleaning stale launch files from {launch_dir}..."
  rm -f {launch_dir}/round_* {launch_dir}/done 2>/dev/null || true
  echo "Launch dir cleaned."
fi
# Workers wait for rank 0 to finish cleanup before polling
sleep 5

# ============================================================
# RANK 0: Run Claude Code as the FP8 Precision Agent
# OTHER RANKS: Poll for barrier-based round files and run torchrun
# ============================================================

if [ "$NODE_RANK" = "0" ]; then
  echo "=========================================="
  echo "[Rank 0] Setting up Claude Code agent..."
  echo "=========================================="

  # Install Node.js 22 LTS
  echo "Installing Node.js 22 LTS..."
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
  echo "Node.js version: $(node --version)"
  echo "npm version: $(npm --version)"

  # Install Claude Code CLI
  echo "Installing Claude Code..."
  npm install -g @anthropic-ai/claude-code
  echo "Claude Code installed: $(claude --version)"

  # Create non-root user (Claude Code refuses --dangerously-skip-permissions as root)
  echo "Creating non-root user for Claude Code..."
  useradd -m -s /bin/bash claude-agent
  chown -R claude-agent:claude-agent {workspace_root}

  # Ensure claude-agent can access CUDA devices, NCCL, and shared memory
  # Without this, torchrun launched by Claude fails at init_process_group
  echo "Setting CUDA/NCCL device permissions for claude-agent..."
  chmod a+rw /dev/nvidia* 2>/dev/null || true
  chmod a+rw /dev/infiniband/* 2>/dev/null || true
  # Add claude-agent to video group (for GPU access)
  usermod -aG video claude-agent 2>/dev/null || true
  # Ensure checkpoint directory is writable
  mkdir -p {cfg.get("checkpoint_root", "/data/savithas/checkpoints")}
  chown -R claude-agent:claude-agent {cfg.get("checkpoint_root", "/data/savithas/checkpoints")}
  # Ensure NFS code path is readable/writable
  chmod -R a+rw {cfg.code_path} 2>/dev/null || true

  # Write the agent prompt to a file
  cat > /tmp/agent_prompt.txt << 'AGENT_PROMPT_EOF'
{agent_prompt}
AGENT_PROMPT_EOF
  chmod 644 /tmp/agent_prompt.txt

  # Create a torchrun wrapper that FORCES correct multi-node flags.
  # This is a failsafe: even if Claude ignores $TORCHRUN_PREFIX and writes
  # "torchrun --nnodes=1 ...", the wrapper strips wrong flags and injects correct ones.
  # We replace the actual torchrun binary so it works regardless of PATH or .bashrc.
  REAL_TORCHRUN=$(which torchrun)
  cp "$REAL_TORCHRUN" "${{REAL_TORCHRUN}}.real"
  cat > "$REAL_TORCHRUN" << TORCHRUN_WRAPPER_EOF
#!/bin/bash
# Torchrun wrapper: strips any --nnodes/--node_rank/--master_addr/--master_port/--nproc_per_node
# flags and replaces them with the correct values from the environment.
# This prevents Claude from accidentally launching single-node training.
# Real torchrun is at: ${{REAL_TORCHRUN}}.real

ARGS=()
SKIP_NEXT=false
for arg in "\\$@"; do
  if \\$SKIP_NEXT; then
    SKIP_NEXT=false
    continue
  fi
  case "\\$arg" in
    --nnodes=*|--node_rank=*|--master_addr=*|--master_port=*|--nproc_per_node=*)
      echo "[torchrun-wrapper] Stripped flag: \\$arg" >&2
      ;;
    --nnodes|--node_rank|--master_addr|--master_port|--nproc_per_node)
      echo "[torchrun-wrapper] Stripped flag: \\$arg (and next arg)" >&2
      SKIP_NEXT=true
      ;;
    *)
      ARGS+=("\\$arg")
      ;;
  esac
done

echo "[torchrun-wrapper] Injecting: --nproc_per_node={cfg.gpus_per_node} --nnodes=\\$NNODES --node_rank=\\$NODE_RANK --master_addr=\\$MASTER_ADDR --master_port=\\$MASTER_PORT" >&2
exec ${{REAL_TORCHRUN}}.real \\
  --nproc_per_node={cfg.gpus_per_node} \\
  --nnodes=\\$NNODES \\
  --node_rank=\\$NODE_RANK \\
  --master_addr=\\$MASTER_ADDR \\
  --master_port=\\$MASTER_PORT \\
  "\\${{ARGS[@]}}"
TORCHRUN_WRAPPER_EOF
  chmod 755 "$REAL_TORCHRUN"
  echo "Torchrun wrapper installed (replaced $REAL_TORCHRUN, original at ${{REAL_TORCHRUN}}.real)"

  # Write a wrapper script for the non-root user
  cat > /tmp/run_claude.sh << 'WRAPPER_EOF'
#!/bin/bash
set -e

# Source env vars from root shell (MASTER_ADDR, NODE_RANK, NNODES, TORCHRUN_PREFIX, etc.)
source /tmp/training_env.sh
echo "Env check: MASTER_ADDR=$MASTER_ADDR NODE_RANK=$NODE_RANK NNODES=$NNODES"
echo "TORCHRUN_PREFIX=$TORCHRUN_PREFIX"

cd {cfg.code_path}

# Verify CUDA access as claude-agent user
echo "CUDA sanity check (as claude-agent)..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())" || echo "WARNING: CUDA not accessible!"
echo ""

echo "Testing Claude Code authentication..."
claude --dangerously-skip-permissions \\
  --model {cfg.claude_model} \\
  -p "Say OK if you can read this." 2>&1 | head -5
echo "Auth check complete."

echo "=========================================="
echo "Starting FP8 Precision Agent..."
echo "Strategy: {cfg.get("promotion_strategy", "ends_in")}"
echo "=========================================="

AGENT_LOG="{workspace_root}/claude_agent_output.log"
echo "Agent output will be logged to: $AGENT_LOG"
echo "Timestamp before launch: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

PROMPT=$(cat /tmp/agent_prompt.txt)
echo "Prompt size: $(echo "$PROMPT" | wc -c) bytes"

# Redirect directly to file (avoids pipe buffering from tee which blocks output for 10-20 min).
# Then tail -f streams the file to container logs in real time.
claude --dangerously-skip-permissions \\
  --model {cfg.claude_model} \\
  -p "$PROMPT" > "$AGENT_LOG" 2>&1 &
CLAUDE_PID=$!
echo "Claude Code launched (PID: $CLAUDE_PID)"

# Give Claude a moment to start writing, then tail the log
sleep 3
tail -f "$AGENT_LOG" &
TAIL_PID=$!

# Wait for Claude to finish
wait $CLAUDE_PID
AGENT_EXIT=$?
kill $TAIL_PID 2>/dev/null || true

echo "=========================================="
echo "FP8 Precision Agent finished (exit code: $AGENT_EXIT)."
echo "Timestamp after exit: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Log file size: $(wc -c < "$AGENT_LOG") bytes"
echo "=========================================="
WRAPPER_EOF
  chmod 755 /tmp/run_claude.sh

  # Run as non-root user, preserving full environment (no dash = keep env)
  su claude-agent -c "bash /tmp/run_claude.sh"

else
  echo "=========================================="
  echo "[Rank $NODE_RANK] Worker node — barrier-based round polling"
  echo "Launch dir: {launch_dir}"
  echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
  echo "=========================================="

  # Worker nodes poll NFS for barrier files written by the Claude agent on rank 0.
  # All workers block on the SAME round_N_ready file, ensuring they start torchrun together.
  # This prevents the desync bug where independent counters caused workers to be on different rounds.
  ROUND=1
  while true; do
    echo "[Rank $NODE_RANK] Waiting for round $ROUND (polling for round_${{ROUND}}_ready)..."
    while [ ! -f "{launch_dir}/round_${{ROUND}}_ready" ] && \
          [ ! -f "{launch_dir}/done" ]; do
      sleep 5
    done

    # Check for completion signal
    if [ -f "{launch_dir}/done" ]; then
      echo "[Rank $NODE_RANK] Done signal received. Exiting."
      break
    fi

    # Source training args written by Claude (contains TRAIN_CMD variable)
    source "{launch_dir}/round_${{ROUND}}_args.env"
    echo "=========================================="
    echo "[Rank $NODE_RANK] Starting round $ROUND"
    echo "TRAIN_CMD=$TRAIN_CMD"
    echo "=========================================="

    # Run torchrun (exits when rank 0 dies/kills or training completes)
    cd {cfg.code_path}
    torchrun \\
      --nproc_per_node={cfg.gpus_per_node} \\
      --nnodes=$NNODES \\
      --node_rank=$NODE_RANK \\
      --master_addr=$MASTER_ADDR \\
      --master_port=$MASTER_PORT \\
      $TRAIN_CMD || true

    echo "[Rank $NODE_RANK] Round $ROUND finished"
    ROUND=$((ROUND + 1))
  done

  echo "[Rank $NODE_RANK] All rounds complete. Exiting."
fi
"""

    command = ["bash", "-c", container_script]

    env_vars = [
        EnvVar(name="ANTHROPIC_AUTH_TOKEN", value_from=EnvValue(secret_name_ref=cfg.anthropic_secret)),
        EnvVar(name="ANTHROPIC_BASE_URL", value=cfg.anthropic_base_url),
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


@hydra.main(version_base=None, config_path="lepton_configs", config_name="og2_fp8_agent")
def main(cfg: DictConfig):
    """Submit a Lepton job that runs Claude Code as the FP8 Precision Agent."""
    print("=" * 60)
    print(f"FP8 Precision Agent - Job: {cfg.job_name}")
    print("=" * 60)
    print(f"  Claude model: {cfg.claude_model}")
    print(f"  Nodes: {cfg.num_nodes} x {cfg.gpus_per_node} GPUs")
    print(f"  Training config: {cfg.get('hydra_config', 'N/A')}")
    print(f"  Steps: {cfg.num_train_steps:,}")
    print(f"  Strategy: {cfg.get('promotion_strategy', 'ends_in')}")
    print(f"  Workspace: {cfg.get('workspace_root', 'N/A')}")

    if cfg.get("git_branch"):
        print(f"  Git branch: {cfg.git_branch}")

    print()

    client = APIClient()
    OmegaConf.resolve(cfg)

    success = launch_claude_agent_job(client, cfg)
    if not success:
        print("\nJob submission failed!")
        exit(1)

    print("\nFP8 Precision Agent job submitted successfully!")
    print(f"Check {cfg.get('workspace_root', '')}/*/report.md for results after completion.")


if __name__ == "__main__":
    main()
