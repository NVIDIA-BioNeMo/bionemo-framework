# Ordered rollout action traceability

Persist every material rollout/QC action as an intent-named ordered script, for example `010-preflight.sh`, `020-generate-pilot.sh`, `030-canonicalize.sh`, and `040-run-hard-qc.sh`, or the execution adapter's scheduler equivalent. Never leave the only executable command in chat. Add a guarded run-all script only after component steps are stable, repeatable, and safe to resume.

For every action record exact script/config/command hashes, executor and host, scheduler/cloud job ID and URL, start/end time, exit status, stdout/stderr paths, and output hashes. Append operational detail to `RUNLOG.md` and `monitor/events.jsonl`.

Expose in concise `SUMMARY.md` and `OUTPUTS.yaml`: SFT and RL checkpoint lineage; W&B URL when enabled; TensorBoard path; scheduler/cloud job ID/URL; resolved config and source state; generation manifest; final QC report; clustering/ranking report; selected designs; logs; and final status/next action. Keep the full command ledger out of the concise summary.
