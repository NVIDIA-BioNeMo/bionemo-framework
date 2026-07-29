---
name: bionemo-phage-design-publish-stage-artifacts
description: Use when a phage-design user requests publication, backup, or an intermediate snapshot of selected checkpoints, validation generations, logs, results, lineage, or final deliverables to object, cloud, mounted, or network storage.
---

# Publish Phage Stage Artifacts

Publish durable, decision-relevant artifacts without coupling scientific progress to transfer availability.

## Plan the publication contract

During intake, record whether publishing is enabled and, if so:

- destination URI/path and backend, namespace, visibility, retention, and credential source;
- stable project slug plus an immutable dated/versioned run prefix, such as `<project>/YYYYMMDD-vN/`;
- stage-end, per-validation, user-requested, and final-reconciliation triggers;
- expected contents, exclusions, approximate sizes, upload client, and verification method.

Keep the same project/run hierarchy at the top level locally and remotely when practical: stable project prefix, immutable `YYYYMMDD-vN` run, then stage subdirectories. Select a configured destination-native client. For S3-compatible storage, discover `aws`, `s5cmd`, and `s3cmd`; for other cloud stores consider their native CLI or configured `rclone`; for mounted/NFS storage prefer `rsync`. Prefer an existing compatible installation over changing the environment. Collision-check and freeze the run prefix before transfer; reuse it across stages and never overwrite a prior run implicitly. Resolve whether access is private, shared, or public; never infer permission to broaden access. If publishing is not requested, state that no artifact sync is planned and stop. If it is requested but no usable client, mount, or credentials exist, make a bounded in-scope enablement attempt, record the gap, and continue the scientific workflow.

## Select artifacts by role

At a stage end, publish:

- the checkpoint selected for the next stage, with lineage, resolved configuration, tokenizer/model metadata, and selection evidence;
- `SUMMARY.md`, `OUTPUTS.yaml`, run/status/config records, manifests, hashes, key metrics/tables/plots, and useful terminal or experiment-tracker logs;
- key generations and scored intermediates needed to reproduce or audit the decision.

For an ongoing run, publish settled outputs from each validation event and its metrics/log pointers, not every checkpoint. At a user-requested intermediate point, snapshot the requested scope plus enough lineage and state to interpret or resume it. Exclude caches, downloaded databases/models already identified by immutable provenance, temporary scorer workdirs, secrets, and redundant unselected checkpoints unless the user asks.

Infer roles from stage contracts and outputs rather than hard-coded paths. Large projects may contain terabytes of checkpoints; inventory sizes before transfer and avoid broad result-root syncs.

## Upload safely

1. Wait for an artifact's atomic completion marker or verify that files are closed and stable. Build an explicit allowlist; never sync a result root by exclusion alone.
2. Write or update `<result-root>/PUBLICATION.yaml` with backend, stage, trigger, local path, role, size, SHA-256 or provider checksum, destination URI/path, source lineage, status, attempts, and timestamps. Never record credentials.
3. Audit staged content for credentials, signed URLs, environment dumps, private endpoints, personal data, and restricted assets. Exclude auth/config files and sanitize copies of necessary logs without altering scientific originals; record exclusions/redactions and both source and published hashes. Avoid debug modes or commands that expose secrets.
4. For shared/public destinations, verify the user's access intent and licensing before transfer, then check effective destination access afterward without changing ACLs unless authorized.
5. Transfer to a staging prefix, then publish the final prefix or completion marker. Use resumable/idempotent client behavior, bounded retries, and concurrency that does not starve training or scoring.
6. Record failures without blocking an otherwise healthy scientific stage. Retry at the next trigger or final reconciliation.
7. Verify destination listings, sizes, checksums, and effective access when supported; spot-read copies when provider checksums are ambiguous. Record verified destination references without secret-bearing query strings in stage outputs and the project summary.

## Reconcile

After every stage and at project end, compare the upload manifest with the selected handoff artifacts and current `PROJECT.yaml`, `SUMMARY.md`, and stage `OUTPUTS.yaml`. Upload key omissions, verify remote accessibility, and report uploaded, superseded, intentionally excluded, and unresolved items. Never claim publication from a successful command alone.
