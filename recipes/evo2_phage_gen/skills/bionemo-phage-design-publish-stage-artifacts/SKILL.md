---
name: bionemo-phage-design-publish-stage-artifacts
description: Use when a phage-design user requests publication, backup, or an intermediate snapshot of selected checkpoints, validation generations, logs, results, or final deliverables to object, cloud, mounted, or network storage.
---

# Publish Phage Stage Artifacts

Publish only when the user requests it. Work from the selected project and stage outputs rather than syncing the entire result tree.

Include the selected checkpoint and settings needed to use it, concise summaries and runlogs, key metrics/plots, final generations and scores, and other artifacts needed to understand or reproduce the scientific decision. Exclude caches, redundant checkpoints, downloaded public databases/models, temporary work directories, credentials, signed URLs, private endpoints, and restricted data unless the user explicitly requests and is authorized to share them.

Use a destination and transfer tool available in the current environment. Confirm the intended access level and applicable licenses before broadening access. Avoid overwriting a prior run, use resumable transfer for large files, and do not let publication failure invalidate an otherwise completed scientific stage.

Verify that the expected remote objects are present and accessible to the intended audience. Record the destination, selected contents, exclusions, transfer result, and any unresolved omission in `RUNLOG.md`.
