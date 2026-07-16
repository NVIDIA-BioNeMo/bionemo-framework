---
name: bionemo-phage-design-adapt-execution
description: Use when a phage-design workflow must discover or adapt to local GPU, SSH, Slurm, Lepton, manual, or unfamiliar execution infrastructure and produce durable launch, monitoring, resume, or handoff commands.
---

# Adapt Phage Execution

Turn the actual environment into explicit human-runnable stage scripts. Generated scripts—not chat—are the command source of truth.

## Discover before choosing

1. Search installed and site-local skills for an environment adapter; use it when appropriate.
2. Inspect without exposing secrets: harness/persistence; hosts; CPU/RAM; GPU model/count/topology, free memory, utilization, and process occupancy; network; credential mechanisms; modules/containers/venv; scheduler account/partition/QOS/time; telemetry; and existing jobs. For every intended input, cache, temporary, checkpoint, log, and result path, record mount/visibility/durability, free bytes and inodes when relevant, and a reversible writability probe.
3. Classify local-gpu, ssh-gpu, slurm, lepton, manual, or documented hybrid. The concrete default is an agentic CLI on a GPU node; reuse the same scripts over SSH.
4. Map paths as shared/node-local and durable/ephemeral, including readers, writer/upload owner, capacity, backup, and visibility across control/compute/monitor nodes.
5. Read [execution-contract.md](references/execution-contract.md) and the single volatile [resource and OOM policy](references/resource-and-oom-policy.md). Write planning/execution/ENVIRONMENT.yaml, EXECUTION_PLAN.md, ACTIONS.yaml, and scripts/ under the result root. Record unknowns rather than guessing. Default resource tuning maximizes measured stable GPU memory and compute utilization with a safety margin; never trade away full-genome context, approved token batch, correctness, or recoverability to improve utilization.

## Select an operating pattern

- **Local GPU:** run stage scripts directly; use a recurring facility or durable local session for due-gated monitoring.
- **SSH GPU:** verify mounts/revision remotely, then launch and monitor by stable run identity.
- **Slurm:** submit from login; run compute-heavy work only in an allocation. Capture job ID, resolved submission script, scheduler/application logs, and site polling policy.
- **Lepton:** inspect `../../ci/lepton/README.md`, `../../ci/lepton/requirements.txt`, and current launcher help from the recipe root. Generate pinned config plus submit/status/log/resume instructions only after resolving runtime fields. Do not claim untested image, endpoint, mount, auth, or secret variants.
- **Manual:** generate complete ordered scripts, acceptance checks, and expected outputs; wait for the user to report completion.

For this recipe, run `.ci_build.sh` before sourcing `.ci_test_env.sh` from the recipe root. Re-discover filenames in another checkout.

## Persist and recover safely

Use /goal, /loop, recurring monitors, or equivalent harness features when available; otherwise use bounded polling, user handoff, cron/systemd, tmux/screen, or nohup according to policy. Timerless loops must read next_check_at and return without scheduler/disk/telemetry queries when not due. Never make a proprietary harness command a prerequisite.

Generate separate launch, one-tick monitor, stop, and resume/relaunch scripts. Prove prior work is absent or terminal before relaunch. Resume only from a verified unchanged-lineage checkpoint; otherwise create a new attempt. Default to no object-store sync until destination, credential mechanism, write owner, cost, retention, and restore behavior are confirmed.

If unfamiliar infrastructure needs durable special logic, propose a local environment skill and list discoveries rather than improvising.
