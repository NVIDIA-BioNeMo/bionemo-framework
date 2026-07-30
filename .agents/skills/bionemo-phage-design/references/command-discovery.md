# Command discovery and drift contract

Resolve commands from the checkout and active environment at execution time. Skill examples and old run logs are snapshots, never authority. Use the absolute repository and selected recipe roots recorded by [workspace-contract.md](workspace-contract.md); never derive them from the skill path.

## Source-isolation gate

Before changing recipe code, confirm the user owns the checkout or has authorized the branch/worktree/copy. Record branch, revision, status, and a hash or saved copy of the pre-existing dirty diff in source_state.json. Preserve unrelated changes. Do not hand-edit copied destinations or symlink targets without following repository policy. If safe isolation cannot be established, stop before source mutation and offer a branch, worktree, or user-owned copy.

## Resolution order

1. In the selected recipe, read current `pyproject.toml`, especially `[project.scripts]`.
2. With that recipe as the working directory, discover and run `.ci_build.sh` before sourcing `.ci_test_env.sh`; verify names in the checkout.
3. Run the selected entry point's --help and, when supported, --version. Resolve it with command -v.
4. Inspect the mapped source function, nearest maintained config, focused tests, and current README under the recorded roots. Follow repository symlinks deliberately.
5. Compare required inputs/flags with the intended stage. Fail on unknown/removed options instead of guessing.
6. Put the exact resolved command in attempt command.sh; record executable path, versions, repository revision, source/config hashes, and help-text hash.

Keep shared defaults unchanged. Materialize a project-owned resolved config in the attempt. Quote paths and redact secret values while preserving the credential mechanism.

## Snapshot names

Names such as evo2_phage_download_sft_data, preprocess_evo2, train_evo2, evo2_phage_run_gdpo, evo2_phage_generation, evo2_phage_score_fasta, and download_bionemo_data are discovery hints. Confirm names/interfaces in pyproject.toml and --help.

## Drift maintenance

When an entry point, CLI option, config schema, environment script, or symlink boundary changes:

1. update owning code/tests/docs under repository policy;
2. search the discovered installed skill bundle for the old interface;
3. update this reference and affected stage references;
4. rerun skill validation and command-resolution smoke.

Do not scatter full launch syntax through SKILL.md. The runtime command.sh is authoritative.
