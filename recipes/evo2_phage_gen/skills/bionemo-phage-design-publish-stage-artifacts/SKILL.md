---
name: bionemo-phage-design-publish-stage-artifacts
description: Use when a phage-design user requests publication, backup, or an intermediate snapshot of selected checkpoints, validation generations, logs, results, or final deliverables to object, cloud, mounted, or network storage.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Publish Phage Stage Artifacts

Publish only when the user requests it. Work from the selected project and stage outputs rather than syncing the entire result tree.

Include the selected checkpoint and settings needed to use it, concise summaries and runlogs, key metrics/plots, final generations and scores, and other artifacts needed to understand or reproduce the scientific decision. Exclude caches, redundant checkpoints, downloaded public databases/models, temporary work directories, credentials, signed URLs, private endpoints, and restricted data unless the user explicitly requests and is authorized to share them.

Choose the checkpoint form by its intended use. For the selected SFT model that initializes RL or is distributed for model use, publish `RESULT_ROOT/rl/sft-checkpoint/` and its `preparation-manifest.json`; this is the runtime-sanitized, optimizer-free payload actually consumed by RL. Require manifest schema 2 with `model_object_state_preserved: true`; rerun preparation instead of publishing a schema-1 payload. Do not sync the raw full-state SFT checkpoint unless the user explicitly requests an exact SFT-training resume backup. A selected NeMo-RL checkpoint needed to resume RL remains a separate full-state artifact.

With AWS, point `aws s3 sync` at the prepared `rl/sft-checkpoint` directory rather than the source SFT checkpoint tree, then verify the manifest and its recorded direct `iter_*` payload at the destination. Preserve the same distinction in other object-storage tools and in the publication inventory.

Use a destination and transfer tool available in the current environment. Confirm the intended access level and applicable licenses before broadening access. Avoid overwriting a prior run, use resumable transfer for large files, and do not let publication failure invalidate an otherwise completed scientific stage.

Verify that the expected remote objects are present and accessible to the intended audience. Record the destination, selected contents, exclusions, transfer result, and any unresolved omission in `RUNLOG.md`.
