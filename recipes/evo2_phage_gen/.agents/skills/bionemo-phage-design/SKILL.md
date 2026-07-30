---
name: bionemo-phage-design
description: Use when planning or running an Evo 2 phage-design project that may include evidence review, genome collection, SFT, GDPO reinforcement learning, checkpoint operations, or final design screening.
---

# Phage Design Controller

Coordinate the project; delegate each stage to its owning skill. Keep the workflow reproducible, evidence-backed, and portable across agent harnesses.

## Start with discovery

Before proposing a plan, locate and read every sibling SKILL.md below. Do not rely on descriptions alone.

- bionemo-phage-design
- bionemo-phage-design-adapt-execution
- bionemo-phage-design-research-evidence
- bionemo-phage-design-collect-genomes
- bionemo-phage-design-prepare-sft
- bionemo-phage-design-operate-mbridge-sft
- bionemo-phage-design-plan-rl-objectives
- bionemo-phage-design-implement-rl-objectives
- bionemo-phage-design-calibrate-rl-sampling
- bionemo-phage-design-operate-nemo-rl
- bionemo-phage-design-generate-and-screen
- bionemo-phage-design-publish-stage-artifacts

Search available skill roots if a sibling is absent. Record a capability gap; do not invent procedures. Resolve the checkout with [workspace-contract.md](references/workspace-contract.md), read [project-contract.md](references/project-contract.md) before creating files, and follow [command-discovery.md](references/command-discovery.md) before commands. Enumerate `assets/literature/**/MANIFEST.json` and route relevant assets.

## Intake and plan

01. Select `interactive` unless the user requests `batch`. Interactive mode inspects and iterates before material launches. Batch mode follows the supplied brief and durable records, stopping for material ambiguity, missing authority, or unsafe irreversible action. Follow [project-contract.md](references/project-contract.md); harness memory is not authoritative.
02. Resolve only the absolute repository root with the workspace contract; record branch/status and any installed-versus-checkout skill difference. On re-entry to existing results, reconcile durable state with the recorded execution facility before new mutation: adopt live work, advance completed work, and never duplicate unresolved work.
03. Choose case-study-replication or adapted-design, a concrete target, and the intended outcome. Select the recipe workspace: replication defaults to checked-in `recipes/evo2_phage_gen`; adapted work uses the user's verified owning recipe, in-checkout copy, or full-checkout worktree.
04. Invoke bionemo-phage-design-adapt-execution. Inspect the repository, results, hardware, execution plane, available skills/models, storage paths, capacity, and writability. State job locations and a per-stage GPU topology matrix; inventory local GPU occupancy before sizing.
05. Build a compact per-skill matrix of inputs, outputs, knowns, gaps, and need; resolve dependencies in stage order.
06. Create slug `<target>-<objective>-<mode>` and `<recipe_root>/results/<slug>[-YYYYMMDD]`, adding a date only on request or collision. Record absolute recipe and result roots before emitting recipe commands.
07. With one clear target, default SFT curation to target-similarity bucket/control-prefix conditioning while allowing opt-out. After collection, agree on context: propose p99.9 or the affordable maximum plus worst-case control/prompt/EOD overhead and required alignment. Change the RL length basis only for an explicit expansion/contraction goal.
08. Unless fresh-only, detect compatible SFT runs locally and in configured result roots. Distinguish status inspection from reuse; present materially different candidates and ask whether to reuse or retrain.
09. After SFT selection and objective/QC approval, invoke bionemo-phage-design-calibrate-rl-sampling. Freeze its prompt compatibility, training mixture, independent validation, paths, and hashes.
10. Write plan/assumptions/decisions in the result root, then invoke required stages. After approval, activate the declared agent-independent execution facility and any available recurring due-gated monitor/advancer, persist and re-query the facility's stable handle, and report success. Leaf skills own attempts; the controller owns root indexes. When publication is requested, record destination, cadence, contents, exclusions, client, and verification and invoke the publication skill at requested/stage/validation/final points; otherwise record no sync.

## Storage gate

Make capacity and cleanup a launch gate using [storage-planning.md](references/storage-planning.md). Forecast sequence artifacts from total bases: for this pipeline, 60 million bases are about 68 MB training-ready, 140 MB compact, or 384 MB retained preparation. Budget about 91 GB per SFT checkpoint and 78 GB per RL checkpoint, plus role-retained state, one checkpoint write, and transient QC/clustering space. Retain latest resumable, best/nondominated, user-pinned, and selected handoff state; prune obsolete state only after evidence is durable. Never prune active, incomplete, selected, or uploading state. If capacity is short, stop before launch and ask the user to free/add space or approve cleanup.

## Design logic

Specify scientific endpoints, invariants, acceptance evidence, and authority boundaries. Let stage operators choose and adapt reversible methods from measured evidence; prescribe an exact mechanism only when reproducibility or correctness depends on it.

Treat replication as a provenance-pinned case study, not a reason to preserve flawed data membership or underuse approved hardware. Preserve target split sizes when feasible, but require cluster-disjoint train/validation/test membership over paper-exact membership. For a new phage or goal, expect several RL rewards and final filters to change—not only gene essentiality. Research target-specific viability preservation, bootability enrichment, essential genes, synteny when relevant, positive/negative thresholds, desired directional change, and diversity. Translate the user's intent into aligned online rewards and final hard filters; do not reuse target-specific thresholds without evidence.

Require GDPO and 1/cluster_size diversity at 99% by default unless a justified alternative is selected. Keep every reward in [0,1], with documented baseline/chance zero, target one, monotonic partial credit, and fail-closed missing data.

## Lineage gate for RL

Do not launch or summarize RL without exact SFT lineage: project, stage name/type, run, checkpoint iteration/path or artifact, checkpoint hash, base-model identity, dataset/split/config hashes, selection metric/evidence/rationale, and proof the checkpoint belongs to that stage. Also require calibrated prompt/sampling lineage. Cross-project SFT is valid when every portable identity, resolved path, and checksum verifies. Preserve both lineage blocks in RL request, run, outputs, and concise summary.

## Handoff discipline

- Keep SUMMARY.md concise and current; append operational detail to RUNLOG.md.
- Read planning/execution/ACTIONS.yaml before each handoff. Expose concise provenance and jump links in PROJECT.yaml and SUMMARY.md; keep action detail append-only.
- Record decisions before mutation, commands before launch, and checksums after artifacts settle.
- Require stage SUMMARY.md and OUTPUTS.yaml before handing outputs downstream.
- Pause when evidence cannot resolve a biologically material choice or execution needs new authority.
