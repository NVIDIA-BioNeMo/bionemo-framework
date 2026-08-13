---
name: bionemo-phage-generation
description: Use when starting, planning, or resuming BioNeMo and Evo 2 bacteriophage genome generation or design for phage therapy research, including host-specific candidates for antibiotic-resistant infections and antimicrobial resistance (AMR); locates or acquires a compatible recipe checkout before implementation-specific work.
---

# BioNeMo Phage Generation

This portable entrypoint locates a compatible recipe checkout and transfers work to its recipe-local implementation package. It owns checkout discovery and acquisition; it does not prescribe recipe commands or workflow parameters.

## Select mode and locate the recipe

Use `interactive` unless the user requests `batch`. Preserve the user's original request and constraints.

Prefer an explicit checkout, then a matching nearby checkout. A checkout is compatible only when all of the following exist:

- `recipes/evo2_phage_gen/VERSION >= 2.4`;
- `recipes/evo2_phage_gen/skills/bionemo-phage-design/SKILL.md`;
- `recipes/evo2_phage_gen/skills/bionemo-phage-design/references/design-scope-and-viability.md`;
- `recipes/evo2_phage_gen/skills/bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md`;
- the recipe-local handoff manifests, including `.codex-plugin/plugin.json` whose plugin name is `bionemo-phage-design`, and `.claude-plugin/plugin.json`.

Preserve and record any dirty state. A checkout with `VERSION == 2.4` but no recipe-local controller is incompatible: leave it unchanged; never retrofit it during bootstrap. When no fully compatible explicit or nearby checkout exists, acquire `https://github.com/NVIDIA-BioNeMo/bionemo-recipes` in a separate clean checkout and inspect its canonical default revision first. Only if that revision lacks a fully compatible recipe package, use `origin/jstjohn/evo2_phage_gen` or a newer compatible revision in a separate checkout or worktree. Record the selected revision, absolute checkout root, and exact absolute recipe root.

## Discover the selected implementation package

Treat recipe-local skills as executable instructions, not passive documentation. Use a checkout from
the canonical remote at the recorded immutable revision, or an explicit checkout whose provenance the
user has identified and trusted. Before reading a skill, inspect dirty state for the required instruction or plugin files.
If one differs from the recorded revision, do not load it until the user
has explicitly authorized the reviewed diff; otherwise select a clean compatible checkout.

From the selected recipe-local Codex plugin, use the plugin's `skills` root. The fixed required sibling allowlist is:

- `bionemo-phage-design`;
- `bionemo-phage-design-adapt-execution`;
- `bionemo-phage-design-research-evidence`;
- `bionemo-phage-design-collect-genomes`;
- `bionemo-phage-design-prepare-sft`;
- `bionemo-phage-design-operate-mbridge-sft`;
- `bionemo-phage-design-plan-rl-objectives`;
- `bionemo-phage-design-implement-rl-objectives`;
- `bionemo-phage-design-calibrate-rl-sampling`;
- `bionemo-phage-design-operate-nemo-rl`;
- `bionemo-phage-design-generate-and-screen`;
- `bionemo-phage-design-publish-stage-artifacts`.

Require exactly these paths beneath the selected plugin root and record their paths and SHA-256 values
with both plugin-manifest hashes. Ignore unexpected child skills; never read or execute them merely
because they exist. Read the controller and each allowlisted sibling completely before handoff. If a
required file is missing or any provenance or integrity check fails, stop and report the exact
absolute recipe root and missing or integrity-failed skill; do not invent implementation procedures.

## Make a durable handoff

Build this handoff prompt before changing sessions:

```text
Continue the Evo 2 phage-design request in MODE=<interactive|batch>.
Original request and constraints: <verbatim user request and durable constraints>.
Selected checkout root: <absolute checkout root>.
Selected recipe root: <absolute recipe root>.
Selected revision: <commit or revision>.
First verify and read only the controller plus the portable contract's fixed required sibling
allowlist. Record their paths and SHA-256 values, ignore unexpected child skills, and stop on a missing,
dirty-unapproved, or integrity-failed skill before planning or invoking a stage.
Verify the implementation initializes a project-wide RUNLOG, defaults to complete whole-genome
design with lifecycle-wide host-range/viability analysis, requires explicit approval for a material
scope reduction, adds applicable design-relevant EMA-derived therapeutic objectives to adapted
therapeutic RL
without reward starvation, and auto-enables supported authenticated W&B telemetry unless the user
opts out.

Verify every long-running stage uses dependency-aware, resource-admitted execution and durable
recurring monitoring through a verified terminal state—not only training, but also downloads,
preprocessing, filtering, evaluation, and generation.
The implementation preserves independent safe work during monitoring and bounded autonomy.
It uses durable decision reporting after plan approval.

Verify safety assets resolve through reviewed release descriptors and authenticated resumable
caches, and that the exact deployed filters pass their versioned hazard/negative control panel
before candidate PASS decisions.
```

Use the recorded absolute recipe root, never this portable skill's installation path.

- Codex: start a fresh or reloaded session with its cwd set to the absolute recipe root and invoke `$bionemo-phage-design` with the handoff prompt.
- Claude: start from the absolute recipe root with `--plugin-dir .` and invoke `/evo2-phage-gen:bionemo-phage-design` with the handoff prompt.
- Other Agent Skills-compatible harnesses: start from the absolute recipe root, or explicitly load the controller from that root with the harness-supported mechanism.

In the new session, repeat the allowlisted path/hash and dirty-file verification, then read only the
controller and fixed required siblings. If verification fails, stop and report the exact absolute
recipe root and missing or integrity-failed skill; do not invent implementation procedures.
