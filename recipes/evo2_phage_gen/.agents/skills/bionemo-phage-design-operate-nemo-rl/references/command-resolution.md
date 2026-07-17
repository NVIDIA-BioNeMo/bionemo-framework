# Resolve current RL commands

Do not preserve a launch command here: entry points and flags are versioned implementation details.

For every attempt:

1. locate the owning recipe and read its current `pyproject.toml` script table;
2. inspect the selected entry point's `--help`, launcher source, nearest tested config, and environment setup;
3. reconcile those interfaces with the installed package and pinned NeMo-RL source/version;
4. resolve the selected generation backend from config and source; for vLLM record the official
   package/tag or wheel provenance, BioNeMo plugin/model revision, NeMo-RL revision, and every
   recipe-owned patch hash, then run deterministic patch apply/reverse/runtime checks;
5. inspect the nearest backend-specific config and tests for prompt groups, local generation batch,
   DP seed coordinates, chosen logprobs, refit, and TP/DP placement;
6. ask `bionemo-phage-design-adapt-execution` to render the resolved invocation for local GPU, SSH, scheduler, or human-executed scripts;
7. save the exact executable invocation in `command.sh` and its interpreter/package/source/config versions and hashes in `RUN.yaml` and `source_state.json`;
8. perform a non-destructive check and bounded one-step smoke before the full launch.

If the interface changes, update this resolution procedure only when discovery locations or required evidence change. Put concrete, version-specific examples in this file—not `SKILL.md`—after verifying them against `--help` and source. Never document or invoke a monitor command unless the current recipe actually exposes it; otherwise use adapter-generated process/scheduler checks and repeatable metric-inspection scripts. Never substitute an upstream vLLM-core patch or runtime monkeypatch for a missing recipe/plugin or NeMo-RL integration without stop-and-review.
