---
name: bionemo-phage-generation
description: Use when starting, planning, or resuming BioNeMo and Evo 2 bacteriophage genome generation or design for phage therapy research, including host-specific candidates for antibiotic-resistant infections and antimicrobial resistance (AMR); locates or acquires a compatible recipe checkout before implementation-specific work.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# BioNeMo Phage Generation

This portable entrypoint locates a compatible recipe checkout and transfers work to its recipe-local implementation package.

## Select mode and locate the recipe

Use `interactive` unless the user requests `batch`. Preserve the user's original request and constraints.

Prefer a checkout supplied by the user, then a compatible nearby checkout. Compatibility requires:

- `recipes/evo2_phage_gen/VERSION >= 2.5`;
- `recipes/evo2_phage_gen/skills/bionemo-phage-design/SKILL.md`.

If none is available, acquire `https://github.com/NVIDIA-BioNeMo/bionemo-recipes` or another compatible revision without overwriting an existing checkout. Record the selected checkout and recipe roots and, when available, its revision.

## Make a durable handoff

Build this handoff prompt before changing sessions:

```text
Continue the Evo 2 phage-design request in MODE=<interactive|batch>.
Original request and constraints: <verbatim user request and durable constraints>.
Selected checkout root: <absolute checkout root>.
Selected recipe root: <absolute recipe root>.
Selected revision: <commit or revision>.
```

Start the next session in the recorded recipe root, not this portable skill's installation path, and invoke the local `bionemo-phage-design` controller using the harness-supported mechanism.
