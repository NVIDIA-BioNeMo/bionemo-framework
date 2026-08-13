# Evo 2 Phage Generation Recipe Instructions

## Agent skill maintenance

The skills under `skills/` are part of this recipe's maintained interface and may depend
on its current code, commands, configurations, output layouts, and documentation.
The `.agents/skills` path is only a relative compatibility symlink to `../skills`; keep
`skills/` as the single source of truth. The recipe's `.claude-plugin/` and
`.codex-plugin/` manifests live beside `skills/` at the recipe root.

After changing a public Python API, CLI entry point or option, configuration schema, workflow
contract, output path or schema, README procedure, or user-facing documentation:

1. Search `skills/` for the old interface or behavior.
2. Update every affected `SKILL.md`, reference, evaluation, and plugin description in the same
   change.
3. Run the skill evaluation validator and focused skill-runner tests.

Keep recipe-specific implementation guidance here rather than in the repository-level portable
`bionemo-phage-generation` skill.
