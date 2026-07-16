# Resolve current generation and QC commands

Command syntax is deliberately absent from `SKILL.md` because recipe interfaces drift.

At runtime, inspect the owning recipe's `pyproject.toml` script table, each selected entry point's `--help`, launcher/source implementation, nearest current config, and environment setup. Match those interfaces to the installed package and source hash. Pass the resolved operations to `adapt-phage-execution` for local GPU, SSH, scheduler, or human-executed rendering.

Save exact invocations in `command.sh` and exact executable/package/config/source versions and hashes in `RUN.yaml` and `source_state.json`. Smoke a tiny deterministic generation and one QC record before scaling.

When interfaces change, update concrete examples here only after checking `pyproject.toml`, `--help`, source, and a smoke run. Keep core skill prose stable. Do not invent generation, QC, monitor, or scheduler commands that the checked-out recipe does not expose.
