# Command discovery and drift contract

Resolve commands from the colocated recipe and active environment at execution time. Skill examples and old run logs are snapshots, never authority. Use the absolute repository and recipe roots recorded by [workspace-contract.md](workspace-contract.md); never derive them from an unrelated skill path.

## Source-isolation gate

Before changing recipe code, confirm the user owns the checkout or has authorized the branch/worktree/copy. Record branch, revision, status, and a hash or saved copy of the pre-existing dirty diff in source_state.json. Preserve unrelated changes. Do not hand-edit copied destinations or symlink targets without following repository policy. If safe isolation cannot be established, stop before source mutation and offer a branch, worktree, or user-owned copy.

## Resolution order

1. In the selected recipe, read current `pyproject.toml`, especially `[project.scripts]`.
2. With that recipe as the working directory, discover and run `.ci_build.sh` before sourcing `.ci_test_env.sh`; verify names in the repository.
3. Run the selected entry point's --help and, when supported, --version. Resolve it with command -v.
4. Inspect the mapped source function, nearest maintained config, focused tests, and current README under the recorded colocated roots. Follow repository symlinks deliberately.
5. Compare required inputs/flags with the intended stage. Fail on unknown/removed options instead of guessing.
6. Put the exact resolved command in attempt command.sh; record executable path, versions, repository revision, source/config hashes, and help-text hash.

Keep shared defaults unchanged. Materialize a project-owned resolved config in the attempt. Quote paths and redact secret values while preserving the credential mechanism.

## Snapshot names

Names such as evo2_phage_download_sft_data, preprocess_evo2, train_evo2, evo2_phage_run_gdpo, evo2_phage_generation, evo2_phage_score_fasta, and download_bionemo_data are discovery hints. Confirm names/interfaces in pyproject.toml and --help.

## External asset failover

Resolve each external model, dataset, or database by pinned identity/version/license and required downstream interface, not one URL. In the attempt asset manifest record required contents, ordered source candidates, independently published signatures/digests and size when available, optional transform/validation hooks, and every failed source.

After bounded retries, prefer TLS-valid official mirrors or archival releases. Never disable TLS silently. Insecure transport is allowed when the bytes match an independently trusted signature or strong cryptographic digest obtained through a secure channel, or after explicit informed user authorization that records the transport risk and reduced provenance assurance. A checksum obtained only through the same broken endpoint is not independent authentication. Stage the payload, compute local SHA-256, validate size/magic/archive contents, then promote atomically.

Classify every fallback as an exact mirror or a derived substitute. A derived asset is allowed only when the approved downstream interface does not require the original raw contents; record immutable input/output hashes, exact transformation and tool versions, the deviation, expected identities/counts, and positive plus independent negative/no-signal behavior controls. Never silently replace raw sequences with profiles or consensus sequences. Block when no verifiable exact or validated interface-compatible asset exists unless the user explicitly changes the asset contract.

For PHROGs v4, the Pharokka v1.8.0 Zenodo record 17110353 is a TLS-valid candidate source for a derived MMseqs search database when QC needs that interface. Reverify its record, checksums, transformation, family inventory, and controls. It is not by itself an exact `FAA_phrog.tar.gz` mirror; a raw-sequence consumer still needs authenticated exact contents or an explicit contract change.

## Drift maintenance

When an entry point, CLI option, config schema, environment script, or symlink boundary changes:

1. update owning code/tests/docs under repository policy;
2. search the discovered recipe-local skill package for the old interface;
3. update this reference and affected stage references;
4. rerun skill validation and command-resolution smoke.

Do not scatter full launch syntax through SKILL.md. The runtime command.sh is authoritative.
