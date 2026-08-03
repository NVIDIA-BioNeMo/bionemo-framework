---
name: bionemo-phage-design-adapt-execution
description: Use when a phage-design workflow must discover or adapt to local GPU, SSH, Slurm, Lepton, manual, or unfamiliar execution infrastructure and produce durable launch, monitoring, resume, or handoff commands.
---

# Adapt Phage Execution

Use the controller's recorded colocated roots. If absent, apply the local [workspace contract](../bionemo-phage-design/references/workspace-contract.md) or stop; never invoke portable bootstrap.

Turn the actual environment into explicit human-runnable stage scripts. Generated scripts—not chat—are the command source of truth. Long-running work must survive and remain queryable across agent-session restarts.

## Discover before choosing

1. Search installed and site-local skills for an environment adapter; use it when appropriate.
2. Inspect without exposing secrets: harness/persistence; hosts; CPU/RAM; GPU model/count/topology, free memory, utilization, and process occupancy; network; credential mechanisms; modules/containers/venv; scheduler account/partition/QOS/time; telemetry; and existing jobs. Treat an environment with uncertain provenance or residue from another run as untrusted: prefer a fresh project-owned virtual environment or container, pin its construction, and verify it before reuse. Never delete or overwrite an old environment merely because it may be polluted; resolve its exact path and obtain the authority required by the project cleanup contract. For every intended input, cache, temporary, checkpoint, log, and result path, record mount/visibility/durability, free bytes and inodes when relevant, and a reversible writability probe.
3. Classify local-gpu, ssh-gpu, slurm, lepton, manual, or documented hybrid. The concrete default is an agentic CLI on a GPU node; reuse the same scripts over SSH.
4. Map paths as shared/node-local and durable/ephemeral, including readers, writer/upload owner, capacity, backup, and visibility across control/compute/monitor nodes.
5. Discover the installed W&B integration and supported authentication flow without printing secret values. Unless the user opted out, attempt bounded authentication using already available mechanisms such as `WANDB_API_KEY` presence, an `api.wandb.ai` netrc entry, an existing authenticated session/settings file, or the current CLI login flow discovered from help. Do not pass keys on command lines or copy them into artifacts. Enable W&B when the probe succeeds; otherwise explicitly disable it in this attempt's resolved config, record attempts and the actionable reason for local-only fallback, and continue without blocking the scientific run. A user does not need a W&B account or prior knowledge of W&B to run the workflow.
6. Read [execution-contract.md](references/execution-contract.md) and the single volatile [resource and OOM policy](references/resource-and-oom-policy.md). Write planning/execution/ENVIRONMENT.yaml, EXECUTION_PLAN.md, ACTIONS.yaml, and scripts/ under the result root. Record unknowns rather than guessing. Honor the planned available-resource topology unless target-length fit or throughput evidence justifies a recorded change; never trade away full-genome context, approved token batch, correctness, or recoverability to improve utilization.

In a read-only or planning-only session, do not claim those files were written. Instead provide a compact planned-artifacts block with each exact path, minimum required contents, ordered launch/monitor/stop/resume script names, acceptance checks, and the command that would materialize or validate each item. Read-only execution does not turn chat into the command source of truth.

## Select an operating pattern

- **Local GPU:** run bounded preflight or smoke work directly. Launch longer work through any proven facility whose worker lifetime and later status/log queries are independent of the current agent shell, chat, PTY, and tool call.
- **SSH GPU:** verify mounts/revision remotely, then launch and monitor by stable run identity.
- **Slurm:** submit from login; run compute-heavy work only in an allocation. Capture job ID, resolved submission script, scheduler/application logs, and site polling policy.
- **Lepton:** inspect `<repository_root>/ci/lepton/{README.md,requirements.txt}` and current launcher help, using the recorded checkout rather than the skill path. Generate pinned config plus submit/status/log/resume instructions only after resolving runtime fields, including egress. Do not claim untested image, endpoint, mount, auth, or secret variants.
- **Manual:** generate complete ordered scripts, acceptance checks, and expected outputs; wait for the user to report completion.

With the selected recipe as the working directory, run `<recipe_root>/.ci_build.sh` before sourcing `<recipe_root>/.ci_test_env.sh`. Re-discover filenames in another checkout.

## Persist and recover safely

Choose by demonstrated lifetime and query semantics, not mechanism name; a proven harness-native long-running job or infrastructure facility may qualify. Use `/goal`, `/loop`, recurring monitors, or equivalent harness features when available to invoke the due-gated one-tick monitor and advance the next approved stage after verified completion. This coordinates the workflow; it owns the worker only when documented to survive agent-session restart. A PID or lock, shell backgrounding, `nohup`, or `setsid` alone is not proof. Follow [execution-contract.md](references/execution-contract.md) for agent-session continuity.

On every fresh or restarted agent session, reconcile durable project/action/attempt/monitor records with the facility native status before mutation. Adopt matching live work without duplication; advance completed work; resume or relaunch terminal work only after checkpoint verification; leave unresolved identity untouched. After approval, activate the facility and verify its stable handle and status/log queries. Add bounded no-progress detection and in-scope repair/retry. Timerless loops must honor `next_check_at`; never make a proprietary harness command a prerequisite. Observe startup and the first one or two validation/checkpoint cycles more closely. Then, or after equivalent healthy progress evidence when those events are not applicable, derive each source's wall-clock cadence from measured step/event timing and a useful fraction of the next validation, checkpoint, or early-stop decision boundary—for example, the duration of about 10 steps. A stable long-running phase may use 5–30 minutes.

Generate separate launch, one-tick monitor, stop, and resume/relaunch scripts. Prove prior work is absent or terminal before relaunch. Resume only from a verified unchanged-lineage checkpoint; otherwise create a new attempt. Default to no object-store sync until destination, credential mechanism, write owner, cost, retention, and restore behavior are confirmed.

If unfamiliar infrastructure needs durable special logic, propose a local environment skill and list discoveries rather than improvising.
