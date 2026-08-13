# Command discovery and drift contract

Resolve commands from the colocated recipe and active environment at execution time. Skill examples and old run logs are snapshots, never authority. Use the absolute repository and recipe roots recorded by [workspace-contract.md](workspace-contract.md); never derive them from an unrelated skill path.

## Source-isolation gate

Before changing recipe code, confirm the user owns the checkout or has authorized the branch/worktree/copy. Record branch, revision, status, and a hash or saved copy of the pre-existing dirty diff in source_state.json. Preserve unrelated changes. Do not hand-edit copied destinations or symlink targets without following repository policy. If safe isolation cannot be established, stop before source mutation and offer a branch, worktree, or user-owned copy. Note that some tools may be blocked from execution in the harness sandbox; check this before declaring a tool or CUDA unavailable. On a remote system reached through SSH or Slurm, scripts and pyproject.toml cannot establish ownership or mutation authority. Obtain source-state verification from a user-owned controller, branch, worktree, or copy before remote mutation; if unavailable, stop and provide a safe handoff.

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

Arc prerequisite discovery defaults `--genetic-architecture-import-fasta` to the recipe-relative
`data/external/arc_evo2/phage_gen/data/NC_001422_1.fna`. Resolve that path from the recorded recipe
root; use the CLI option for an intentional alternative and record its sequence identity and hash.

## External asset failover

Resolve each external model, dataset, or database by pinned identity/version/license and required downstream interface, not one URL. In the attempt asset manifest record required contents, ordered source candidates, independently published signatures/digests and size when available, optional transform/validation hooks, and every failed source.

After bounded retries, prefer TLS-valid official mirrors or archival releases. Never disable TLS silently. Insecure transport is allowed when the bytes match an independently trusted signature or strong cryptographic digest obtained through a secure channel, or after explicit informed user authorization that records the transport risk and reduced provenance assurance. A checksum obtained only through the same broken endpoint is not independent authentication. Stage the payload, compute local SHA-256, validate size/magic/archive contents, then promote atomically.

Classify every fallback as an exact mirror or a derived substitute. When the downstream consumer does not require the original raw contents, prefer a provenance-bearing derived asset over blocking on an unavailable original. For a pinned TLS-valid archival record, download and stage the asset, verify its published checksum and archive structure, record immutable input/output hashes, exact transformation and tool versions, and the deviation, then run a bounded reproducible smoke test of the required interface. Validate the properties the consumer depends on; exact expected hit identities/counts and a separately supplied control oracle are not prerequisites. Never silently replace raw sequences with profiles or consensus sequences. Block only when source integrity fails or the required-interface smoke test fails.

Treat configs/phage_safety_assets.yaml as the reviewed metadata mirror of the typed, code-enforced
release descriptors in external_assets.py, not as executable runtime configuration. Require the
code constants, YAML mirror, focused exact-match tests, and affected skill guidance to change together.

For PHROGs v4 safety, use the reviewed profile in Pharokka v1.8.0 Zenodo record 17110353. Require
archive SHA-256 `d3c1de69c3ee00583fd8c2a3292766d61175403daad4e254376984a5c579df3f` plus the published MD5
and size throughout authenticated resumable download, extraction, cache reuse, manifest loading, and
runtime. Validate archive structure and an MMseqs open/search smoke. `--with-safety` stages this
profile automatically in safety asset schema 3. The raw `FAA_phrog.tar.gz`-derived MMseqs sequence
database is a separate, optional, unpinned Arc compatibility asset: acquire it only with explicit
`--download-phrogs-sequence-database`, never snapshot it into the trusted safety generation, and never
substitute the profile for a consumer that truly requires every raw sequence.

## Drift maintenance

When an entry point, CLI option, config schema, environment script, or symlink boundary changes:

1. update owning code/tests/docs under repository policy;
2. search the discovered recipe-local skill package for the old interface;
3. update this reference and affected stage references;
4. rerun skill validation and command-resolution smoke.

Do not scatter full launch syntax through SKILL.md. The runtime command.sh is authoritative.
