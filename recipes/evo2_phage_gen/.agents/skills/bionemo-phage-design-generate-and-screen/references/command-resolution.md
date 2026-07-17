# Resolve current generation and QC commands

Command syntax is deliberately absent from `SKILL.md` because recipe interfaces drift.

At runtime, inspect the owning recipe's `pyproject.toml` script table, each selected entry point's `--help`, launcher/source implementation, nearest current config, and environment setup. Match those interfaces to the installed package and source hash. Pass the resolved operations to `bionemo-phage-design-adapt-execution` for local GPU, SSH, scheduler, or human-executed rendering.

Perform that check on the host that will run generation. Build the environment
with `./.ci_build.sh`, source `.ci_test_env.sh` in the same shell, and verify the
resolved Python, Evo2 plugin, vLLM version, GPU visibility, and entry-point help
before allocating a large rollout. From the selected RL MBridge checkpoint,
resolve and run `python -m bionemo.evo2.vllm.export` into a new directory, then
hash the export config, index, manifest, and shards. The subsequent generation
command must name that export, not an older base-policy export.

Treat TP2 MP+async O2/balanced, compilation mode 3, and exact-batch
`FULL_AND_PIECEWISE` graphs as the measured two-H100 reference. For a different
allocation, resolve a supported TP/DP topology from the assigned devices,
capacity-test the real local wave sizes, and retain exact output/logprob/QC
gates. Higher TP is valid when supported; additional engine groups require
disjoint CUDA visibility plus disjoint request and seed partitions.

Save exact export and generation invocations in `command.sh` and exact executable/package/config/source versions and hashes in `RUN.yaml` and `source_state.json`. Smoke a tiny deterministic generation and one QC record before scaling.

When interfaces change, update concrete examples here only after checking `pyproject.toml`, `--help`, source, and a smoke run. Keep core skill prose stable. Do not invent generation, QC, monitor, or scheduler commands that the checked-out recipe does not expose.
