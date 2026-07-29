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

Search available skill roots if a sibling is not beside this skill. Record a missing skill as a capability gap; do not invent its procedures. Read [project-contract.md](references/project-contract.md) before creating project files, and follow [command-discovery.md](references/command-discovery.md) before emitting commands. Enumerate assets/literature/**/MANIFEST.json and tell sub-skills which checked-in paper assets are relevant.

## Intake and plan

1. Select `interactive` unless the user explicitly requests `batch`. Interactive mode inspects first, presents a compact initial plan, and iterates with the user before material launches. Batch mode treats the supplied brief as controlling intent, makes a traceable best effort from durable project records, and stops only for a material biological ambiguity, missing authority, or unsafe irreversible action. Follow the mode and portable-memory contract in [project-contract.md](references/project-contract.md); proprietary harness memory is never the source of truth.
2. Invoke bionemo-phage-design-adapt-execution during intake. Inspect repository, existing results, hardware, execution plane, installed site skills, public model access, all required storage paths, free capacity, and writability before asking questions. State exactly where jobs will run and the per-stage GPU topology matrix; when compute is local, inventory visible GPUs and occupancy before sizing work.
3. Build a compact matrix with one row per skill: required inputs, expected outputs, known values, missing values, and whether needed. Resolve dependencies in stage order.
4. Choose case-study-replication or adapted-design. Select a concrete reference phage or virus consistent with the recipe and available model evidence; record reference genome/hash, host, related genome collection, and intended outcome.
5. Create slug `<target>-<objective>-<mode>` and result root `results/<slug>[-YYYYMMDD]`. Add date only on request or collision. Ask only for choices still material after inspection.
6. Detect compatible SFT runs in this and other result roots unless the user explicitly requires fresh acquisition/training or forbids prior-run reuse. Distinguish status inspection from artifact reuse; never search for reusable local assets after a fresh-only instruction. Otherwise present evidence and ask whether to reuse one or train anew.
7. After selecting SFT and approving/implementing the objective/QC contract, invoke bionemo-phage-design-calibrate-rl-sampling. Freeze its SFT prompt-compatibility, training-mixture, and independent validation contracts with paths and hashes.
8. Write plan, assumptions, and decisions in the result root, then invoke only required stage skills. After approval, immediately activate the declared durable supervisor/recurrence mechanism and report whether activation succeeded. Leaf skills write within their assigned attempt; update root indexes only as controller.
9. If the user requests artifact publication, invoke bionemo-phage-design-publish-stage-artifacts at the requested point, each completed stage, each ongoing validation event, and final reconciliation. Record destination/backend, cadence, contents, exclusions, client, and verification in the initial plan. Otherwise state that no artifact sync is planned; missing optional publication capability never blocks science.

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
