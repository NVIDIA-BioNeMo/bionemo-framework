# Ordered action and summary traceability

Material RL actions must exist as files, not only chat or shell history. Name executable steps in order and by intent, such as `010-preflight.sh`, `020-launch-rl.sh`, and `030-monitor-once.sh`, or the execution adapter's scheduler equivalent. An optional guarded run-all script is acceptable only after individual steps are stable and idempotence/resume behavior is understood.

For every action record exact script and resolved-config hashes, executor/host, scheduler or cloud job ID and URL when present, start/end time, exit status, stdout/stderr paths, and output hashes. Link each action to append-only `RUNLOG.md` and `monitor/events.jsonl` entries. Never leave the only runnable command in chat.

The concise RL `SUMMARY.md` and `OUTPUTS.yaml` must expose:

- SFT project/run/checkpoint lineage and hashes;
- RL policy and KL-reference identities;
- W&B URL when enabled and local TensorBoard path;
- scheduler/cloud job ID and URL when present;
- selected RL checkpoint and evidence report;
- resolved config, QC/validation report, logs, and source-state paths;
- best step, stopping-evidence step, final status, and blocker/next action.

Keep command-by-command detail in the append-only run log and monitor events rather than bloating the summary.
