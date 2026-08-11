---
name: bionemo-phage-generation
description: Use when starting, planning, or resuming BioNeMo phage generation and a compatible Evo 2 phage-generation recipe checkout must be located or acquired before implementation-specific work.
---

# BioNeMo Phage Generation

This portable entrypoint locates a compatible recipe checkout and transfers work to its recipe-local implementation package. It owns checkout discovery and acquisition; it does not prescribe recipe commands or workflow parameters.

## Select mode and locate the recipe

Use `interactive` unless the user requests `batch`. Preserve the user's original request and constraints.

Prefer an explicit checkout, then a matching nearby checkout. A checkout is compatible only when all of the following exist:

- `recipes/evo2_phage_gen/VERSION >= 2.4`;
- `recipes/evo2_phage_gen/.agents/skills/bionemo-phage-design/SKILL.md`;
- `recipes/evo2_phage_gen/.agents/skills/bionemo-phage-design/references/design-scope-and-viability.md`;
- `recipes/evo2_phage_gen/.agents/skills/bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md`;
- the recipe-local handoff manifests, including `.agents/.codex-plugin/plugin.json` whose plugin name is `bionemo-phage-design`, and `.agents/.claude-plugin/plugin.json`.

Preserve and record any dirty state. A checkout with `VERSION == 2.4` but no recipe-local controller is incompatible: leave it unchanged; never retrofit it during bootstrap. When no fully compatible explicit or nearby checkout exists, acquire `https://github.com/NVIDIA-BioNeMo/bionemo-recipes` in a separate clean checkout and inspect its canonical default revision first. Only if that revision lacks a fully compatible recipe package, use `origin/jstjohn/evo2_phage_gen` or a newer compatible revision in a separate checkout or worktree. Record the selected revision, absolute checkout root, and exact absolute recipe root.

## Discover the selected implementation package

From the selected recipe-local Codex plugin, use the plugin's `skills` root. Enumerate every immediate child containing `SKILL.md`; require `bionemo-phage-design` among the discoverable skills. Record the discovered names, then read the controller and every discovered sibling completely before handoff. If enumeration or loading fails, stop and report the exact absolute recipe root and missing skill; do not invent implementation procedures.

## Make a durable handoff

Build this handoff prompt before changing sessions:

```
Continue the Evo 2 phage-design request in MODE=<interactive|batch>.
Original request and constraints: <verbatim user request and durable constraints>.
Selected checkout root: <absolute checkout root>.
Selected recipe root: <absolute recipe root>.
Selected revision: <commit or revision>.
First enumerate the selected recipe-local plugin skills root and verify the controller plus every
discovered sibling are loadable; read every SKILL.md completely before planning or invoking a stage.
Verify the implementation initializes a project-wide RUNLOG, defaults to complete whole-genome
design with lifecycle-wide host-range/viability analysis, requires explicit approval for a material
scope reduction, adds applicable design-relevant EMA-derived therapeutic objectives to adapted
therapeutic RL
without reward starvation, and auto-enables supported authenticated W&B telemetry unless the user
opts out.

Verify the implementation defaults to dependency-aware, resource-admitted execution, continues
independent safe work during monitoring, and uses bounded autonomy after plan approval while
using durable decision reporting for material in-envelope decisions.
```

Use the recorded absolute recipe root, never this portable skill's installation path.

- Codex: start a fresh or reloaded session with its cwd set to the absolute recipe root and invoke `$bionemo-phage-design` with the handoff prompt.
- Claude: start from the absolute recipe root with `--plugin-dir .agents` and invoke `/evo2-phage-gen:bionemo-phage-design` with the handoff prompt.
- Other Agent Skills-compatible harnesses: start from the absolute recipe root, or explicitly load the controller from that root with the harness-supported mechanism.

In the new session, repeat the selected plugin-skills-root enumeration, verify `bionemo-phage-design` and every discovered sibling are loadable, record the names, and read them completely. If discovery fails, stop and report the exact absolute recipe root and missing skill; do not invent implementation procedures.
