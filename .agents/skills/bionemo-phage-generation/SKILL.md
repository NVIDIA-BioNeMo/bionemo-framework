---
name: bionemo-phage-generation
description: Use when starting, planning, or resuming BioNeMo phage generation and a compatible Evo 2 phage-generation recipe checkout must be located or acquired before implementation-specific work.
---

# BioNeMo Phage Generation

## Capabilities

Plan or resume evidence review, genome collection, SFT, GDPO objective planning and training, sampling calibration, generation, QC, lineage, monitoring, and optional publication. Leave recipe-specific commands and implementation choices to the checked-out recipe.

## Choose a mode

Default to interactive planning. Use batch only when requested. In batch mode, infer reversible choices only from durable records; stop for missing authority, irreversible risk, or unresolved material biology.

## Locate or acquire the recipe

Prefer an explicit checkout, then a matching nearby checkout. Reuse either when `recipes/evo2_phage_gen/VERSION >= 2.4`; preserve and record any dirty state, and isolate later mutations as needed. Dirtiness alone never triggers the special branch. If no compatible checkout exists, clone or acquire the canonical default revision from https://github.com/NVIDIA-BioNeMo/bionemo-recipes in a separate clean checkout and inspect it. Only after that canonical default revision is shown not to contain `recipes/evo2_phage_gen/VERSION >= 2.4`, obtain `origin/jstjohn/evo2_phage_gen` or a newer compatible revision in a separate checkout or worktree. Record the repository revision and absolute checkout and recipe roots.

## Hand off to the checked-out implementation

Reload or start the agent from `recipes/evo2_phage_gen` so its `.agents/skills` are discovered. Read `bionemo-phage-design` and relevant sibling skills completely, then hand control to `bionemo-phage-design`. Treat the current checkout's docs, pyproject entry points, configs, and tests as implementation authority.
