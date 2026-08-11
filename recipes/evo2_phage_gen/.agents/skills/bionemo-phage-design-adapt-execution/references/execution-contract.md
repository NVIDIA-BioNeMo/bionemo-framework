# Execution contract

## ENVIRONMENT.yaml

Record observed values, discovery commands, timestamps, and unresolved fields. Use null plus a reason for unknowns.

```yaml
schema_version: 3
profile: local-gpu        # local-gpu|ssh-gpu|slurm|lepton|manual|hybrid
harness:
  name: unknown
  recurring_modes: []
hosts:
  control: {name: null, role: agent, os: null}  # control workstation or approved service host
  login: {name: null, role: submission, policy_source: null}
  build: {name: null, role: null}
  launch: {name: null, role: null}
  compute: {name: null, role: null}
  transfer: {name: null, role: null}
access: {ssh: null, scheduler: null, network_egress: null}
compute:
  gpus: []                 # model, UUID, memory, utilization, processes, topology
  gpu_observed_at: null
  gpu_visibility_status: unresolved  # unresolved|host-confirmed|host-unavailable
  gpu_probe_contexts: []   # command, time, host identity, sandbox/container/host boundary, result
  cpu_count: null
  ram_bytes: null
scheduler:
  type: null
  policy_sources: []
  head_login_node_policy: null
  account: null
  partition_or_node_group: null
  allowed_partitions: []
  partition_limits: []      # name, account/QOS, node types, GRES, maximum_walltime, preemption
  qos: null
  walltime: null
  job_id: null
runtime:
  mode: null                # host-uv|docker|site-container|other
  already_containerized: null
  host_uv: {lock_identity: null, environment_path: null}
  container: {engine: null, base_identity: null, image_digest: null, mount_map: [], gpu_runtime: null}
  modules: []
  recipe_root: null
  build_script: null
  source_env_script: null
  command_versions: {}
telemetry:
  local: {tensorboard: true, path: null}
  wandb:
    policy: auto-enable-unless-opted-out
    installed: null
    auth_status: unchecked       # unchecked|authenticated|unavailable|opted-out
    auth_mechanism: null         # mechanism name only; never a secret value
    attempts: []
    entity: null
    project_family: null
    sft: {enabled: null, project: null, run_id: null, url: null}
    calibration: {enabled: null, project: null, run_id: null, url: null}
    rl: {enabled: null, project: null, run_id: null, url: null}
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

Local telemetry remains authoritative. Unless the user explicitly opts out, discover the installed
W&B interface and make a bounded authentication attempt using supported mechanisms already available
to the session. Presence-check `WANDB_API_KEY` without emitting it; check for an `api.wandb.ai` netrc
entry or an existing authenticated session/settings file without copying credential contents; and
inspect current `wandb` help/status before using its login flow. Do not require the user to reconfirm
an entity or project when the integration can resolve them safely. Enable project-owned SFT,
calibration, and RL runs when authentication succeeds. A checked-in `wandb_enabled: false` default is
not an opt-out. If the user has no account, does not know W&B, or the package, network, or
authentication is unavailable after bounded attempts, set W&B disabled in the attempt's resolved
config so launch cannot fail on telemetry, continue with local telemetry, and record the mechanism
names tried, sanitized error category, and remediation. Never persist a key or login output that
contains one.

When reading a schema-1 or schema-2 environment record, treat absent W&B, site-policy, host-role, and
runtime-boundary fields as unresolved rather than as an opt-out or inherited capability. Preserve the
old record and emit schema 3 in the new attempt or approved project update.

For each required input, cache, temporary, checkpoint, log, and result path record host/mount, visibility, shared or node-local, durable or ephemeral, readers, writer, free/total bytes, free inodes when relevant, contents, and the timestamp/result of a reversible create-write-fsync-rename-delete probe when it must be writable. Do not infer writability from permission bits alone. Identify one owner for every write/upload and resolve checkpoint durability before launch.

Choose control, login/submission, build, compute, monitor, and transfer placement from current site
policy and observed mount/network visibility. A control workstation may drive the cluster without
hosting the worker. Record where environment construction, downloads, preprocessing, training, QC,
monitoring, and synchronization run; an unspecified role is unresolved, not permission to use a
head or login node.

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

Each ACTIONS item records id, intent, prerequisites, script path, script/command/config hashes, executor, host, execution-facility or scheduler/cloud ID and URL, start/end/exit, logs, outputs/hashes, and idempotence/resume guard. Trace shell, Slurm, Lepton, and handed-off actions. Nothing runnable exists only in chat. Add a guarded run-all only after individual steps stabilize.

Use set -Eeuo pipefail in Bash, quote expansions, absolute paths, and no embedded secrets. Prefer atomic status writes. Never silently overwrite an attempt or duplicate a submission.

## Resource-aware admission control

Consume `planning/DEPENDENCY_GRAPH.yaml` as the launch graph. On every admission cycle:

1. Reconcile durable node and attempt state with facility-native status.
2. Select nodes whose hard dependencies and gates succeeded (`dependency-ready`).
3. Refresh capacity, occupancy, reservations, CPU, RAM, GPU, storage/I/O headroom, and write-scope conflicts or exclusive locks.
4. Admit a highest-priority safe set whose aggregate requests fit (`resource-admissible`); no optimal solver is required.
5. Persist reservations and stable handles before launch success.
6. Mark non-fitting ready nodes `queued` with the resource reason. Release reservations only after verified facility-native terminal-state reconciliation; reconsider queued nodes after launch, completion, failure, capacity change, or material plan change.

Keep independent DAG nodes independent: a queued node blocks only its descendants, not safe unrelated work. A due-gated monitor is not a blocking phase.

## Agent-session continuity

For every long-running stage, choose any environment- or harness-native execution facility whose worker lifetime and stable status/log queries outlive the current agent shell, chat, PTY, and tool call. Scheduler/cloud jobs, harness-managed durable jobs, services, and detached sessions are examples, not requirements. Record the actual lifetime scope. Use `/goal`, `/loop`, recurring monitors, or equivalent harness features when available to invoke due-gated one-tick monitoring and re-evaluate and admit the safe ready set after a verified terminal handoff; this coordinates the workflow and is not worker ownership unless documented as such. A PID or lock, shell backgrounding, `nohup`, or `setsid` alone is not proof.

Before launch succeeds, persist the reservation and, in `RUN.yaml` and `ACTIONS.yaml`, the facility and stable handle, host, lifetime scope, status/log/stop queries, command/config/source/working-directory identity, logs, heartbeat/progress marker, checkpoint/resume state, and restart policy. Re-query the handle and verify it identifies the launched work using facility-appropriate evidence.

At every new agent session, read durable project, action, attempt, reservation, and monitor state before mutation, then query the facility by handle. Adopt matching live work without duplication; advance completed work; resume or relaunch failed/terminal work only after checkpoint verification; leave unresolved identity untouched. Reconcile terminal state before releasing a reservation, then re-evaluate and admit the safe ready set. Automatic restart is optional and, when used, must be bounded and invoke an idempotent verified-checkpoint resume path.

## Rate-limited monitoring

Each active node retains independent due times. A one-tick monitor reads `monitor/state.json` before any scheduler, filesystem, process, or telemetry query. Persist `last_checked_at`, `next_check_at`, `backoff_attempt`, phase, and independent due times for scheduler, logs, disk/checkpoints, and telemetry. Set due times from measured or expected step/event timing, risk, query cost, and the next decision boundary rather than a universal heartbeat. In timerless `/goal` loops, return successfully without querying when not due; a not-due monitor does not prevent admission of other ready work.

Use phase-aware cadence:

- Observe launch, first steps, and early validation/checkpoint evidence at intervals justified by expected event timing and risk, while honoring site policy; do not default to a 30-second heartbeat.
- After one or two healthy validation/checkpoint cycles—or equivalent progress evidence when those events are not applicable—derive each source's wall-clock cadence from measured step/event timing and a useful fraction of the next validation, checkpoint, or early-stop decision boundary—for example, the duration of about 10 steps. A stable long-running phase may use 5–30 minutes.
- Reset backoff on state change, new validation/checkpoint, or health alert, but never tight-loop.

For Slurm, prefer targeted squeue -j JOB_ID while active and sacct -j JOB_ID for terminal/accounting state. Do not repeatedly scan cluster-wide sinfo or allocate srun solely to monitor. Avoid recursive directory walks; use known files, mtimes, and incremental offsets. Rate-limit W&B/cloud APIs independently.

Treat a failed gate, absent process, stale heartbeat, or unchanged progress marker beyond its declared bound as actionable—not as healthy waiting. Capture evidence, perform only bounded in-scope repair/retry, and then advance or write an explicit terminal state. Never leave a supervisor logically active with no worker and no scheduled action.

## Profile gates

### Local GPU and SSH

Confirm intended GPU host with `nvidia-smi` and `hostname`, paths, and session lifetime. Record whether each probe ran in an agent sandbox, container, SSH session, scheduler allocation, or directly on the host. A sandbox-visible GPU failure does not establish a host driver failure: sandboxes may omit GPU devices, NVML access, driver libraries, or accelerator permissions even when the host is healthy. Repeat `hostname`, `nvidia-smi`, `/dev/nvidia*`, and runtime CUDA visibility through a permitted host-visible execution context before diagnosing missing drivers or hardware. If host-visible execution requires elevation, request approval; do not bypass the boundary. Until that comparison exists, keep GPU capacity unresolved and do not install/reload drivers, switch to CPU, reduce the approved GPU topology, or abandon a local-GPU plan based only on the sandbox result. Record both contexts and any mismatch. SSH uses configured aliases and strict failures; never distribute keys/tokens. Treat disconnect recovery separately from training resume.

For local execution, first detect whether the agent is already containerized and whether a usable host
`uv` environment, Docker engine/daemon, GPU runtime, and durable job facility are actually available.
Honor an explicit user choice; otherwise choose host-uv or Docker from compatibility, reproducibility,
isolation, mount, GPU, and recovery evidence and record the rationale. In either boundary, run the
recipe's `.ci_build.sh` before sourcing `.ci_test_env.sh` from the recipe working directory.

For host-uv, bind the environment to the lock/source identity and keep it project-owned. For Docker,
record the Dockerfile/build context, pinned base and final image identity, GPU/runtime smoke, UID/GID,
network policy, and explicit code/data/cache/result mounts; keep credentials out of the image and
command line. Persist container name/ID plus status/log/attach/stop/recovery commands as the facility
handle. When already containerized, do not launch nested Docker merely because its CLI exists; require
verified daemon reachability, mount-path translation, GPU propagation, and user intent or a clear
compatibility benefit.

### Slurm

Resolve the current head/login-node policy, user associations, allowed_partitions, account/QOS combinations,
per-partition `maximum_walltime`, node types/GRES/topology, preemption/signals, egress, modules/container
mechanism, and shared versus node-local mounts from site documentation, a bounded current query, and
submission evidence as permitted. A user-provided known-good site script is useful evidence; compare
its account, directives, modules, mounts, image, and entry point with current policy and help rather
than treating old syntax as authority. Do not infer permission from a partition merely appearing in
a global listing.

Prefer the agent/control loop on a control workstation or approved service host. On a strict site,
the login node performs only bounded policy-permitted submission and targeted status/log queries: no
`tmux`, environment installation, container build, data/model download, preprocessing, training, QC,
or broad filesystem work. Place those tasks in approved build, compute, transfer, or service jobs.
Use a matching pinned base image and the site's current container mechanism. If a project environment
must live on shared storage, key it by lock/source/base identity, build it once under a lock in a
staging path, validate it, and use atomic promotion to an immutable consumer path; never let concurrent
jobs mutate one shared environment. From the recipe working directory, run `.ci_build.sh`, then source
`.ci_test_env.sh` before resolving entry points.

When a healthy stage cannot fit the discovered walltime, select a site-supported bounded restart
method. A [Slurm singleton dependency](https://slurm.schedmd.com/sbatch.html#OPT_dependency) serializes
previously launched jobs with the same job name and user; if used, give each project/stage/attempt a
unique stable job name, prequeue only the approved number of segments, and record every job ID and
dependency. Each segment must reconcile predecessor state, accept only a complete unchanged-lineage
checkpoint, and exit safely rather than duplicate work. Prove the first automatic continuation reads
the prior segment's checkpoint and produces forward progress plus the next durable checkpoint/artifact
before relying on the chain. Bound resubmission and checkpoint cadence before walltime; do not
use one singleton name across distinct DAG stages or combine it blindly with scheduler requeue.

Run resource synchronization as a short shared-disk transfer job on an approved transfer/compute node
when the login node or training allocation is unsuitable. Record job IDs immediately, resume only
complete checkpoints, and assign object-store upload to an approved compute, transfer, or service
owner. Keep `squeue`/`sacct` queries targeted and due-gated; never poll `sinfo` or `squeue` in a tight
loop.

### Lepton

Treat `<repository_root>/ci/lepton/` as evidence of a Hydra job path and pinned client, not a universal launcher. Discover authentication, workspace/endpoint, API, node group/resources, image/registry, mounts, secret references, egress, naming, status/log/cancel APIs, and durable-output ownership. Never copy site-specific defaults blindly.

From the selected recipe working directory, run `.ci_build.sh`, then source `.ci_test_env.sh` before resolving entry points. Emit project-owned config and submit/status/log/cancel/resume wrappers. Pin revision/container digest. Leave unverified endpoint/image/auth variants unresolved.

### Object storage

Leave sync disabled until destination, credential mechanism, owner, cost, retention, and restore are confirmed. Define direction, cadence, include/exclude, temporary naming, completion markers, checksums, policy, deletion behavior, and restore test. Upload only finalized snapshots and publish completion last. Never delete local artifacts merely because upload returned success.
