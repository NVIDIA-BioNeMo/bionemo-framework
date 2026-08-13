# Resolve current generation and QC commands

Command syntax is deliberately absent from `SKILL.md` because recipe interfaces drift.

At runtime, inspect the owning recipe's `pyproject.toml` script table, each selected entry point's `--help`, launcher/source implementation, nearest current config, and environment setup. Match those interfaces to the installed package and source hash. Pass the resolved operations to `bionemo-phage-design-adapt-execution` for local GPU, SSH, scheduler, or human-executed rendering.

Resolve external models and databases through the shared [asset-failover contract](../../bionemo-phage-design/references/command-discovery.md#external-asset-failover). Persist sanitized invocations in `command.sh` and exact non-secret executable/package/config/source versions and hashes in `RUN.yaml` and `source_state.json`. Use environment-variable or credential-helper references at runtime; sanitize command/config copies and reject persistence or publication while a token, signed URL, cookie, password, or credential-bearing query parameter remains. Smoke a tiny deterministic generation and one QC record before scaling.

For sequence safety, discover `evo2_phage_sequence_safety scan` and
`evo2_phage_validate_safety_controls` from the current entry points. The scanner currently identifies
CLI version 2 and output manifest schema 2; validate and replay that manifest before candidate PASS.
Record `shared_executions` and lifecycle status `NOT_STARTED`, `FAILED`, or
`COMPLETED_AND_PARSED`. A nonstarted or failed attempt may omit raw output, but its affected records
remain `INDETERMINATE` and cannot be promoted. Resolve batching through the central external-tool
policy rather than copying a fixed topology here. Resolve topology from the current parser: circular -> no topology argument because circular is the default;
linear -> `--linear`; reject any other topology before launch. Never invent or pass
`--circular`.

For Arc prerequisites, resolve the portable recipe-relative default
`data/external/arc_evo2/phage_gen/data/NC_001422_1.fna` from `recipe_root`, or record an intentional
`--genetic-architecture-import-fasta` override and its sequence hash. Do not resurrect the legacy
cluster/user path.

For DUST QC, the current `NucleotideQCConfig` default maximum end-mask fraction is `0.9` and the
configurable `dustmasker_timeout_s` default is 300 seconds. Preserve non-integer level values. A missing
binary, non-zero process exit, or timeout is explicit failure evidence rather than an unbounded wait or
an inferred pass.

When interfaces change, update concrete examples here only after checking `pyproject.toml`, `--help`, source, and a smoke run. Keep core skill prose stable. Do not invent generation, QC, monitor, or scheduler commands that the checked-out recipe does not expose.
