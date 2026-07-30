# Execution contract

## ENVIRONMENT.yaml

Record observed values, discovery commands, timestamps, and unresolved fields. Use null plus a reason for unknowns.

```yaml
schema_version: 1
profile: local-gpu        # local-gpu|ssh-gpu|slurm|lepton|manual|hybrid
harness:
  name: unknown
  recurring_modes: []
hosts:
  control: {name: null, role: agent, os: null}
  launch: {name: null, role: null}
  compute: {name: null, role: null}
access: {ssh: null, scheduler: null, network_egress: null}
compute:
  gpus: []                 # model, UUID, memory, utilization, processes, topology
  gpu_observed_at: null
  cpu_count: null
  ram_bytes: null
scheduler:
  type: null
  account: null
  partition_or_node_group: null
  qos: null
  walltime: null
  job_id: null
runtime:
  container: null
  modules: []
  recipe_root: null
  build_script: null
  source_env_script: null
  command_versions: {}
telemetry:
  local: {tensorboard: true, path: null}
  wandb: {entity: null, project_family: null, sft: {enabled: false, project: null, run_id: null}, rl: {enabled: false, project: null, run_id: null}}
credentials:
  mechanisms: []
storage:
  paths: []
  sync:
    enabled: false
    destination: null
    credential_mechanism: null
    write_owner_host: null
    source_path: null
    cadence: null
    include: []
    exclude: []
    atomicity: null
    checksum_policy: null
    retention: null
    estimated_cost: null
    restore_test: null
unresolved: []
```

For each required input, cache, temporary, checkpoint, log, and result path record host/mount, visibility, shared or node-local, durable or ephemeral, readers, writer, free/total bytes, free inodes when relevant, contents, and the timestamp/result of a reversible create-write-fsync-rename-delete probe when it must be writable. Do not infer writability from permission bits alone. Identify one owner for every write/upload and resolve checkpoint durability before launch.

## EXECUTION_PLAN.md

Record chosen profile/evidence; control/launch/compute/monitor/upload owners; runtime build/revision; absolute paths/mount visibility; resources/hardware fit; launch/monitor/stop/resume scripts; telemetry and polling policy; sync/retention/restore decision; unresolved fields and authority needed.

## Generated scripts and namespace

Put reusable project actions in planning/execution/scripts/. Put stage actions in STAGE/runs/ATTEMPT/scripts/. Record both in planning/execution/ACTIONS.yaml and copy the exact launch command into attempt command.sh.

Use composite ID STAGE/ATTEMPT/NNN, or project/NNN for setup. NNN is monotonic inside its namespace. Never overwrite an action path. Typical stage scripts:

- 010-preflight.sh: verify revision, tools, runtime, hardware, mounts, space, inputs, and hashes.
- 020-launch-STAGE.sh: create owned directories, activate environment, run resolved command, capture process/job identity and exit.
- 030-monitor-STAGE.sh: perform one due-gated observation and append state.
- 040-stop-STAGE.sh: request graceful checkpoint/termination.
- 050-resume-STAGE.sh: verify lineage and terminal/absent prior work before resume or new attempt.
- 060-sync-results.sh: only when approved; upload from declared owner/path and verify.
- 070-restore-test.sh: restore a representative artifact to scratch and verify.

Each ACTIONS item records id, intent, prerequisites, script path, script/command/config hashes, executor, host, scheduler/cloud ID and URL, start/end/exit, logs, outputs/hashes, and idempotence/resume guard. Trace shell, Slurm, Lepton, and handed-off actions. Nothing runnable exists only in chat. Add a guarded run-all only after individual steps stabilize.

Use set -Eeuo pipefail in Bash, quote expansions, absolute paths, and no embedded secrets. Prefer atomic status writes. Never silently overwrite an attempt or duplicate a submission.

## Rate-limited monitoring

A one-tick monitor reads monitor/state.json before any scheduler, filesystem, process, or telemetry query. Persist last_checked_at, next_check_at, backoff_attempt, phase, and independent due times for scheduler, logs, disk/checkpoints, and telemetry. In timerless /goal loops, return successfully without querying when not due.

Use phase-aware cadence:

- Observe launch, first few steps, first validation, and first verified checkpoint more often, while honoring site policy. Default scheduler floor is at least 60 seconds; cheap local log checks may be 30–60 seconds.
- After the first healthy checkpoint, back off to minutes or validation cadence, independently per source, with 10–20% jitter.
- Reset backoff on state change, new validation/checkpoint, or health alert, but never tight-loop.

For Slurm, prefer targeted squeue -j JOB_ID while active and sacct -j JOB_ID for terminal/accounting state. Do not repeatedly scan cluster-wide sinfo or allocate srun solely to monitor. Avoid recursive directory walks; use known files, mtimes, and incremental offsets. Rate-limit W&B/cloud APIs independently.

Treat a failed gate, absent process, stale heartbeat, or unchanged progress marker beyond its declared bound as actionable—not as healthy waiting. Capture evidence, perform only bounded in-scope repair/retry, and then advance or write an explicit terminal state. Never leave a supervisor logically active with no worker and no scheduled action.

## Profile gates

### Local GPU and SSH

Confirm intended GPU host with nvidia-smi and hostname, paths, and session lifetime. SSH uses configured aliases and strict failures; never distribute keys/tokens. Treat disconnect recovery separately from training resume.

### Slurm

Discover command/site polling policy, account, partition, QOS, GPU GRES, topology, time, modules/container, mounts, and preemption. Submit on login; train/heavy-QC on compute. Record job ID immediately. Resume only complete checkpoints. Assign object-store upload to an approved compute, transfer, or service owner.

### Lepton

Treat `<repository_root>/ci/lepton/` as evidence of a Hydra job path and pinned client, not a universal launcher. Discover authentication, workspace/endpoint, API, node group/resources, image/registry, mounts, secret references, egress, naming, status/log/cancel APIs, and durable-output ownership. Never copy site-specific defaults blindly.

From the selected recipe working directory, run `.ci_build.sh`, then source `.ci_test_env.sh` before resolving entry points. Emit project-owned config and submit/status/log/cancel/resume wrappers. Pin revision/container digest. Leave unverified endpoint/image/auth variants unresolved.

### Object storage

Leave sync disabled until destination, credential mechanism, owner, cost, retention, and restore are confirmed. Define direction, cadence, include/exclude, temporary naming, completion markers, checksums, policy, deletion behavior, and restore test. Upload only finalized snapshots and publish completion last. Never delete local artifacts merely because upload returned success.
