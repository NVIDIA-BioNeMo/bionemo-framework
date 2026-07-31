# Workspace contract

The target BioNeMo Recipes checkout is independent of the skill installation. Never infer a checkout or recipe path from this skill's directory.

## Locate or acquire the checkout

Prefer an explicit user path, then a matching current/ancestor checkout. Before reusing one, record its branch, worktree state, and remotes with read-only Git inspection. Otherwise:

```bash
git clone https://github.com/NVIDIA-BioNeMo/bionemo-recipes
cd bionemo-recipes
```

Validate `recipes/evo2_phage_gen/pyproject.toml`. Only when that recipe is absent, leave any existing checkout untouched and acquire a separate clean checkout from one of these equivalent canonical sources. Record the exact commit.

```bash
# https://github.com/NVIDIA-BioNeMo/bionemo-recipes/pull/1699
git clone https://github.com/NVIDIA-BioNeMo/bionemo-recipes bionemo-recipes-evo2-phage-gen
cd bionemo-recipes-evo2-phage-gen
git fetch https://github.com/NVIDIA-BioNeMo/bionemo-recipes pull/1699/head
git switch --detach FETCH_HEAD

# Upstream branch jstjohn/evo2_phage_gen
git clone --branch jstjohn/evo2_phage_gen --single-branch \
  https://github.com/NVIDIA-BioNeMo/bionemo-recipes bionemo-recipes-evo2-phage-gen
```

An agent may start at the repository root or elsewhere. Resolve `repository_root` first. After selecting mode, recipe workspace, target, and slug, record absolute `recipe_root` and `result_root` before recipe commands.

If this skill is globally installed or imported and the checkout also contains the recipe-local implementation bundle under `recipes/evo2_phage_gen/.agents/skills`, compare their revisions or file hashes. Report meaningful differences and record which bundle and hash govern the run; never silently mix installed and checkout bundle versions.

The repository-root `.agents` package contains the portable `bionemo-phage-generation` entrypoint. It is not a substitute for the recipe-local implementation bundle.

## Select the recipe workspace

- For case-study replication, default to `<repository_root>/recipes/evo2_phage_gen` without asking a nonmaterial question.
- For adapted source changes, ask whether to edit the owning recipe or isolate changes. This recipe has tracked relative symlinks to sibling recipe assets. Put a recipe-only copy under the same checkout's `recipes/` directory and verify every tracked relative symlink, or use a full-checkout copy or worktree. If the user requests copy+symlink, link only to a recipe inside that complete layout and treat its checkout as authoritative; never link a standalone external recipe copy.
- Respect copied-file mappings and recipe boundaries; never add cross-recipe imports.
- Record the choice, source revision, dirty state, and authoritative source path before mutation.

Use the selected recipe as the working directory for every actual recipe command, including environment setup. Keep generated run state under `<recipe_root>/results/<slug>...`; repository-wide commands may run from `repository_root`.

Resolve sibling references and checked assets from the installed skill bundle. Resolve checkout files only from the recorded repository and recipe roots.
