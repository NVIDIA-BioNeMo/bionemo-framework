# Final rollout reporting contract

Keep the rollout attempt self-contained and traceable. At minimum write:

- `RUN.yaml`: status, execution environment, sanitized command/version/hash, checkpoint and schema-4 SFT lineage, design-spec path/hash and approved scope, prompt/generation/filter/tool/source identities, safety asset schema, scanner CLI/output-manifest versions, and resolved single-record or batch topology;
- `command.sh` and adapter-produced scheduler/SSH/human scripts;
- `logs/`, `metrics/`, and append-only `monitor/events.jsonl`;
- raw design manifest with stable ID, batch, prompt ID, seed, sequence hash, validity, and source artifact;
- canonicalization/exact-dedup map and representative FASTA;
- hard-QC table with every component, missing/tool error, nested result, OR branches, overall pass, all six control expectations/outcomes, and validated/replayed safety manifest identity;
- safety shared-execution inventory with exact record bindings, command/input/output hashes, `NOT_STARTED`/`FAILED`/`COMPLETED_AND_PARSED` lifecycle, and reason evidence explaining why unresolved attempts block PASS;
- 99%-cluster membership/representative tables plus MMseqs2 version and exact identity/coverage parameters;
- ranking table with predeclared component values, uncertainty, ordering, and selected/not-selected rationale;
- final-order manifest containing only hard-QC-passing representatives and their complete provenance;
- `OUTPUTS.yaml`, concise `SUMMARY.md`, and append-only `RUNLOG.md`.

## Summary contents

Lead with requested mode/count and achieved result. State:

- attempted, valid, exact-unique, raw hard-QC pass, and unique passing-cluster counts/rates;
- target orders/reserve and whether it was achieved;
- checkpoint hash and exact SFT/RL lineage;
- complete-genome output and mutable-scope confirmation, with any approved reduction and its decision record;
- filter-profile and code/config hashes, safety asset schema 3, scanner CLI version 2/output manifest schema 2, and runtime batch-size/workers/tool-thread topology;
- six-control aggregate/class outcomes plus every shared execution lifecycle and any failed/unstarted reason;
- uniqueness identity/coverage/topology semantics;
- counts remaining after each filter and OR branch/dominance results;
- accumulation curve, saturation evidence, and conservative yield estimate;
- ranking method and final picks;
- incomplete tools, approximations, uncertainties, failures, and next decision.

Label these as computational screening results. Computational QC does not establish biological viability,
productive infection, therapeutic suitability, clinical safety or efficacy, or regulatory acceptability.
State the remaining phenotypic and wet-lab endpoints plus the expert, biosafety, clinical, and regulatory
review appropriate to the intended use.

Never report “unique” without naming exact versus clustered, denominator, identity, coverage, and whether clustering occurred before or after QC. Never report a pass rate without its filter-profile ID and source artifact.

## Reproducibility checks

Before completion, verify counts reconcile across manifests; every selected sequence maps to one raw generation and one passing canonical representative; every final pick satisfies the approved complete-genome/lifecycle integrity contract; all required tools and shared executions are `COMPLETED_AND_PARSED`; scanner manifest validation/replay and all six controls match; hashes match; rerunning canonicalization/filter aggregation produces identical results; and final picks contain no duplicate cluster representatives.
