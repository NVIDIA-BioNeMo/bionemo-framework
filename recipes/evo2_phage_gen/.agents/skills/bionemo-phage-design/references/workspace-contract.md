# Colocated workspace and provenance contract

This controller runs only from the recipe-local package that exposed it. Derive `recipe_root` from that package: it must be the `recipes/evo2_phage_gen` directory containing this controller at `.agents/skills/bionemo-phage-design/SKILL.md`. Do not use an unrelated or external skill root.

Validate only the colocated markers needed to guard a corrupt invocation: the recipe's `pyproject.toml`, `.agents/skills/bionemo-phage-design/SKILL.md`, and the local Codex and Claude plugin manifests. If any marker is missing, stop, state the exact observed path and missing marker, and direct the user back to `bionemo-phage-generation`.

Derive `repository_root` from `recipe_root` with read-only Git discovery. Record the repository revision and dirty state for provenance. Do not clone, fetch, switch, select a revision, prove a version floor, or compare this package with another bundle.

```bash
git -C "$recipe_root" rev-parse --show-toplevel
git -C "$recipe_root" rev-parse HEAD
git -C "$recipe_root" status --short
```

Use `recipe_root` as the working directory for recipe commands. Keep generated run state under `<recipe_root>/results/<slug>...`; repository-wide read-only inspection may use `repository_root`.

For an actual user-authorized source mutation, record the chosen branch/worktree/copy, source revision, dirty state, and authoritative source path before mutation. Respect copied-file mappings, tracked symlinks, and recipe boundaries; never add cross-recipe imports. This isolation guidance does not locate or validate the implementation checkout.

Resolve sibling references and checked assets from the colocated recipe-local package. If controller-recorded roots are absent, derive them with this local contract or stop; never invoke portable bootstrap from a leaf skill.
