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
- the recipe-local handoff manifests, including `.agents/.codex-plugin/plugin.json` whose plugin name is `bionemo-phage-design`, and `.agents/.claude-plugin/plugin.json`.

Preserve and record any dirty state. A checkout with `VERSION == 2.4` but no recipe-local controller is incompatible: leave it unchanged and continue the existing ordered canonical-default then fallback acquisition policy in a separate clean checkout. Never retrofit that checkout during bootstrap. Record the selected revision, absolute checkout root, and exact absolute recipe root.

## Make a durable handoff

Build this handoff prompt before changing sessions:

```
Continue the Evo 2 phage-design request in MODE=<interactive|batch>.
Original request and constraints: <verbatim user request and durable constraints>.
Selected checkout root: <absolute checkout root>.
Selected recipe root: <absolute recipe root>.
Selected revision: <commit or revision>.
First verify the recipe-local controller and all expected sibling skills are discoverable,
then read their SKILL.md files completely before planning or invoking a stage.
```

Use the recorded absolute recipe root, never this portable skill's installation path.

- Codex: start a fresh or reloaded session with its cwd set to the absolute recipe root and invoke `$bionemo-phage-design` with the handoff prompt.
- Claude: start from the absolute recipe root with `--plugin-dir .agents` and invoke `/evo2-phage-gen:bionemo-phage-design` with the handoff prompt.
- Other Agent Skills-compatible harnesses: start from the absolute recipe root, or explicitly load the controller from that root with the harness-supported mechanism.

In the new session, verify that `bionemo-phage-design` and the expected siblings are discoverable from the recipe-local package and read them completely. If discovery fails, stop and report the exact absolute recipe root and missing skill; do not invent implementation procedures.
