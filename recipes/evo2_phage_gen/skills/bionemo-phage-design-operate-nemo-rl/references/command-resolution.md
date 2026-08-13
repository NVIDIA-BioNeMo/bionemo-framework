# Resolve current RL commands

Do not preserve a launch command here: entry points and flags are versioned implementation details.

For every attempt:

1. locate the owning recipe and read its current `pyproject.toml` script table;
2. inspect the selected entry point's `--help`, launcher source, nearest tested config, and environment setup;
3. reconcile those interfaces with the installed package and pinned NeMo-RL source/version;
4. ask `bionemo-phage-design-adapt-execution` to render the resolved invocation for local GPU, SSH, scheduler, or human-executed scripts;
5. render runtime credentials only as environment-variable references, approved secret-store
   references, or credential helpers; never inline tokens, signed URLs, cookies, passwords, or
   credential-bearing query parameters;
6. sanitize the resolved invocation and config copy before saving `command.sh`, `RUN.yaml`, and
   `source_state.json`; scan their materialized bytes and reject persistence or publication while any
   secret-bearing value remains, retaining only the mechanism name and reproducible command structure;

If the interface changes, update this resolution procedure only when discovery locations or required evidence change. Put concrete, version-specific examples in this file—not `SKILL.md`—after verifying them against `--help` and source. Never document or invoke a monitor command unless the current recipe actually exposes it; otherwise use adapter-generated process/scheduler checks and repeatable metric-inspection scripts.
