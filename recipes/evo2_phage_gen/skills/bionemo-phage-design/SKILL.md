---
name: bionemo-phage-design
description: Use when planning or running an Evo 2 bacteriophage genome-design project for phage therapy research, including host-specific candidates for antibiotic-resistant infections and antimicrobial resistance (AMR); coordinates evidence review, genome collection, SFT, GDPO reinforcement learning, checkpoint operations, safety QC, generation, and final screening.
---

# Phage Design Controller

Coordinate the project; delegate each stage to its owning skill. Keep the workflow reproducible, evidence-backed, and portable.

## Mandatory offline-continuation response

When the user will be offline and an approved plan continues, begin the response with this `DAG and records` checklist, filling in each item:

```text
## DAG and records
- Graph: planning/DEPENDENCY_GRAPH.yaml is updated to show ...
- Active independent work: ...
- Blocked descendants and gates: ...
- Resource queue: ...
- Records: in-envelope decisions/deviations go to planning/DECISIONS.md and root RUNLOG.md; no routine question is needed.
```

Escalate only when the autonomy envelope is exceeded. Do not omit the checklist because work is awaiting a gate or uses no GPUs.

## Start with discovery

Before proposing a plan, locate and read every sibling SKILL.md below (in addition to this). Do not rely on descriptions alone.

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

The local sibling package is required. A missing sibling is a package integrity error: stop, report the missing skill and recipe-local package path, and do not search unrelated roots or acquire another checkout. Derive colocated roots with [workspace-contract.md](references/workspace-contract.md), read [project-contract.md](references/project-contract.md) and [design-scope-and-viability.md](references/design-scope-and-viability.md) before creating files, and follow [command-discovery.md](references/command-discovery.md) before commands. For a therapeutic project, also read the cleaned local [EMA draft phage-therapy quality guideline](references/ema-2025-draft-phage-therapy-quality-guideline.md) and verify its status against the linked official EMA record before treating it as current guidance. Enumerate `assets/literature/**/MANIFEST.json` and route relevant assets. The `king-2025-generative-phage-design` bundle contains the [CC BY bioRxiv v1 release](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full), not Science-hosted files; cite the [final Science publication](https://www.science.org/doi/10.1126/science.aec2657) as the publication of record and retain the bioRxiv v1 provenance whenever bundled content is used. Bundled papers are not a complete evidence review for a new target.

## Intake and plan

1. Select `interactive` unless the user requests `batch`. Interactive mode iterates the initial plan; batch derives it from durable authority. After authorization, both modes act autonomously within the recorded envelope while reporting decisions and escalating only outside it. Follow [project-contract.md](references/project-contract.md); harness memory is not authoritative.
2. Resolve the absolute recipe and repository roots with the workspace contract; record revision and dirty state for provenance. On re-entry to existing results, reconcile durable state with the recorded execution facility before new mutation: adopt live work, advance completed work, and never duplicate unresolved work.
3. Choose case-study-replication or adapted-design, a concrete target, intended use, and outcome. Unless the user clearly states another use, provisionally treat adapted work as therapeutic and make that assumption visible for revision. Reject an endpoint that increases replication within eukaryotic cells before planning or implementation. This is not a blanket ban on non-replicative eukaryotic entry or host-range work, which requires case-specific evidence and safety review. Default adapted work to complete whole-genome candidates and a whole-genome mutable scope. Record `planning/DESIGN_SPEC.yaml` with the intended-use rationale, lifecycle-wide endpoint, protected traits, viable-reference set, and any proposed scope reduction. Never infer a locus-only or tail-fiber-only design from similarity, synteny, host-range emphasis, or metric exclusions; obtain explicit approval for that material reduction. Select the recipe workspace by always using the selected checkout's colocated `recipes/evo2_phage_gen` package as `recipe_root`. When source mutation is needed, use a user-authorized branch, full-checkout worktree, or copy that retains this colocated package; never substitute a different owning recipe.
4. Invoke the bionemo-phage-design-adapt-execution skill. Inspect the repository, results, hardware, execution plane, available skills/models, storage paths, capacity, and writability. State job locations and a per-stage GPU topology matrix; inventory local GPU occupancy before sizing.
5. Build a compact per-skill matrix of inputs, outputs, knowns, gaps, and need.

Build and maintain `planning/DEPENDENCY_GRAPH.yaml`. Treat the plan as a DAG: launch every safe, non-conflicting node that is dependency-ready and resource-admissible; numeric stage/action order is not execution order. Monitoring an active job is not an exclusive phase. A blocked node blocks only its descendants, so continue unrelated safe work.

Treat every operation that can outlive the current tool call or agent turn as long-running, not only SFT or RL. This includes downloads, extraction and indexing, genome preprocessing, safety and QC filtering, evaluation, generation, training, and optimization. Route it through `bionemo-phage-design-adapt-execution`, activate a recurring due-gated monitor before returning, and keep the workflow alive until the facility reports a verified terminal state. A background launch is not completion.

06. Create slug `<target>-<objective>-<mode>` and `<recipe_root>/results/<slug>[-YYYYMMDD]`, adding a date only on request or collision. Before any stage attempt, initialize the root `PROJECT.yaml`, `SUMMARY.md`, append-only `RUNLOG.md`, `planning/DEPENDENCY_GRAPH.yaml`, `planning/DESIGN_SPEC.yaml`, and `planning/DECISIONS.md`; record absolute recipe and result roots before emitting recipe commands.
07. With one clear target, default SFT curation to target-similarity bucket/control-prefix conditioning while allowing opt-out. Treat conditioning as a steerable signal, never as an edit mask. After collection, agree on context: propose p99.9 or the affordable maximum plus worst-case control/prompt/EOD overhead and required alignment. After the final leakage-controlled split, require a post-collection training-budget feedback decision from usable corpus/token mass and the effective batch; do not inherit a publication step count unchanged. Change the RL length basis only for an explicit expansion/contraction goal.
08. Unless fresh-only, detect compatible SFT runs locally and in configured result roots. Distinguish status inspection from reuse; present materially different candidates and ask whether to reuse or retrain.
09. After SFT selection and objective/QC approval, invoke the bionemo-phage-design-calibrate-rl-sampling skill. Freeze its prompt compatibility, training mixture, independent validation, paths, and hashes.
10. Append planning, assumptions, decisions, and every material handoff to the root `RUNLOG.md`, then invoke required stages. Unless the user opts out, auto-enable W&B for SFT, sampling calibration, and RL whenever the current integration is installed and authentication succeeds through a supported mechanism; never expose credentials, and record bounded attempts plus the fallback reason when unavailable. A checked-in `wandb_enabled: false` is not project policy. Keep local telemetry authoritative. After approval, activate the declared agent-independent execution facility and any available recurring due-gated monitor/advancer, persist and re-query the facility's stable handle, and report success. Leaf skills own attempts; the controller owns root indexes. When publication is requested, record destination, cadence, contents, exclusions, client, and verification and invoke the publication skill at requested/stage/validation/final points; otherwise record no sync.

Treat every safety database/model as a reviewed release descriptor, not a permanent URL or implicit `latest`.
Safety asset manifest schema 3 authenticates the complete trusted generation before atomic
publication. Its PHROGs profile comes from Pharokka v1.8.0 Zenodo record 17110353 and requires archive
SHA-256 `d3c1de69c3ee00583fd8c2a3292766d61175403daad4e254376984a5c579df3f` in addition to the published
MD5 and size. `--with-safety` stages that profile automatically; the optional unpinned raw-FAA Arc
compatibility database requires explicit `--download-phrogs-sequence-database` and stays outside the
trusted bundle. Prefer a verified persistent content-addressed cache with authenticated resumable transfers, then authenticate
complete bytes before extraction, cache reuse, or trust. BLAST+ 2.17 resolves reviewed x86_64 and
aarch64 archives, but the complete safety bundle still has x86_64-only tools; `--with-safety` must
refuse other architectures before mutation until every required binary is natively resolved.
The Python-only AMRFinder source-bin override remains operator-attested bytes, not authenticated
source-build provenance or evidence of full ARM support.
A future version or location becomes a descriptor only after comparable identity, interface, reconciliation,
and control review. Missing required evidence remains indeterminate rather than being concealed.

## Storage gate

Make capacity and cleanup a launch gate using [storage-planning.md](references/storage-planning.md). Forecast sequence artifacts from total bases: for this pipeline, 60 million bases are about 68 MB training-ready, 140 MB compact, or 384 MB retained preparation. Budget about 91 GB per SFT checkpoint and 78 GB per RL checkpoint, plus role-retained state, one checkpoint write, and transient QC/clustering space. Retain latest resumable, best/nondominated, user-pinned, and selected handoff state; prune obsolete state only after evidence is durable. Never prune active, incomplete, selected, or uploading state. If capacity is short, stop before launch and ask the user to free/add space or approve cleanup.

## Design logic

Specify scientific endpoints, whole-genome design scope, invariants, acceptance evidence, and authority boundaries. Let stage operators choose and adapt reversible methods from measured evidence; prescribe an exact mechanism only when reproducibility or correctness depends on it.

Treat replication as a provenance-pinned case study, not a reason to preserve flawed data membership or underuse approved hardware. Preserve target split sizes when feasible, but require cluster-disjoint train/validation/test membership over paper-exact membership. For a new phage or goal, expect several RL rewards and final filters to change—not only gene essentiality. Follow the whole-genome and lifecycle contract: research viability, adsorption and genome entry, intracellular defense/counter-defense, takeover and replication, morphogenesis and packaging, productive lysis, [therapeutic suitability and safety-related exclusion criteria](references/ema-2025-draft-phage-therapy-quality-guideline.md) when applicable, essential/key genes, regulatory architecture, synteny, composition, calibrated similarity to viable relatives, desired host-range direction, and diversity. A strong host-range model may integrate several axes, but remains one calibrated signal. Translate the user's intent into aligned online rewards, final hard filters, and experimental validation; do not reuse target-specific thresholds without evidence.

For adapted-design work with therapeutic intended use, default every applicable, design-relevant [EMA-derived guardrail](references/design-scope-and-viability.md#apply-intended-use-therapeutic-guardrails) with a defensible measurable proxy into its own online RL component and retain the corresponding hard-QC or experimental endpoint. The recipe's default PhiX174 case-study replication is also customized: keep filters 1–6, 8, and 9 enabled, keep filter 7 disabled, and add applicable safety objectives by default. Preserve historical and added component sets separately, and never directly compare aggregate rewards computed from different sets. For explicitly non-therapeutic adapted work, record the intended-use applicability decision. Prevent sparse components from starving the portfolio through independent measurement, calibrated partial credit, and runtime/support diagnosis, never by silently dropping the guardrail.

Use `GDPO` and `1/cluster_size` diversity at 99% by default unless a justified alternative is selected. Keep every reward in [0,1], with documented baseline/chance zero, target one, monotonic partial credit, and zero credit plus a recorded reason for missing or invalid data.

## Lineage gate for RL

Do not implement, calibrate, launch, or summarize RL without the required schema-4 `sft_lineage` block or a verified immutable manifest containing every field: source project/root; stage name/type and run/attempt; checkpoint iteration, resolved path or artifact ID/version, content/manifest hash, and stage-binding evidence; base provider/public ID/version/format/hash; dataset and leakage-controlled split-manifest hashes; resolved-config and source-state hashes; and selection metric/evidence, best/stop step, rationale, summary, and outputs paths. General evidence or a checkpoint path cannot substitute. Also require separate calibrated prompt/sampling lineage. Cross-project SFT is valid when every portable identity, resolved path, and checksum verifies. Preserve both lineage blocks in RL request, run, outputs, and concise summary.

## Handoff discipline

- Keep SUMMARY.md concise and current; append operational detail to RUNLOG.md.
- Read planning/execution/ACTIONS.yaml before each handoff. Expose concise provenance and jump links in PROJECT.yaml and SUMMARY.md; keep action detail append-only.
- Record decisions before mutation, commands before launch, and checksums after artifacts settle.
- Require stage SUMMARY.md and OUTPUTS.yaml before handing outputs downstream.
- Pause when evidence cannot resolve a biologically material choice or execution needs new authority.
