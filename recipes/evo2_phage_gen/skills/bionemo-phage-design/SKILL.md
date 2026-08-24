---
name: bionemo-phage-design
description: Use when planning or running an Evo 2 bacteriophage genome-design project for phage therapy research, including host-specific candidates for antibiotic-resistant infections and antimicrobial resistance (AMR); coordinates evidence review, genome collection, SFT, GDPO reinforcement learning, checkpoint operations, safety QC, generation, and final screening.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Phage Design Controller

Coordinate the stages as a computational phage and AI scientist keeping an electronic lab notebook. Record enough to repeat and interpret the experiment, and keep the structure proportionate to the work.

The research skill points to the phage-generation paper and supplement, related design evidence, and a local transcription of the draft EMA phage-therapy quality guideline. Use them as needed for specialized methods, results, or guideline questions; the EMA text is a historical draft rather than current regulatory advice.

## Intake and plan

1. Select `interactive` unless the user requests `batch`. Interactive mode iterates the initial plan; batch derives it from the user's request. After approval, both modes can act autonomously within that scope.
2. Resolve the recipe, repository, and result roots using the [workspace guidance](references/workspace-guidance.md). On re-entry, inspect existing results and active jobs before starting new work.
3. Choose case-study-replication or adapted-design, a concrete target, intended use, and outcome. Unless the user states another use, provisionally treat adapted work as therapeutic and make that assumption visible for revision. Reject an endpoint that increases replication within eukaryotic cells; assess non-replicative entry or host-range work on its own evidence. Default to complete whole-genome candidates and obtain explicit approval before narrowing to a locus, module, RBP, or fixed backbone. Record the intended use, lifecycle endpoint, protected traits, viable references, and scope in the project summary and runlog.
4. Invoke the bionemo-phage-design-adapt-execution skill to inspect the checkout, existing results, available hardware, storage, and current commands before sizing jobs.
5. Create slug `<target>-<objective>-<mode>` and `<recipe_root>/results/<slug>[-YYYYMMDD]`, adding a date only on request or collision. Initialize the compact [project lab notebook](references/lab-notebook-guidance.md).
6. With one clear target, default SFT curation to target-similarity bucket/control-prefix conditioning while allowing opt-out. Treat conditioning as a steerable signal, never as an edit mask. Choose context from the tokenized genome-length distribution plus control/prompt/EOD overhead. Set the training budget from the usable corpus and effective batch rather than inheriting a publication step count.
7. Unless fresh-only, detect compatible SFT runs locally and in configured result roots, then ask whether to reuse or retrain when the choice is material.
8. After SFT selection and objective/QC approval, invoke the bionemo-phage-design-calibrate-rl-sampling skill to establish prompt compatibility, training mixture, and independent validation.
9. Record the commands, settings, inputs, checkpoints, results, and important decisions in the project runlog as the work proceeds. Keep local logs authoritative. Invoke the publication skill only when publication is requested.

For a new phage-design project using the 7B family, default to the trained-further long-context NGC checkpoint `evo2/7b-1m:1.0` and model size `evo2_7b`, even when the selected sequence length is shorter than 1M. Do not replace an existing run's `evo2_7b_base` checkpoint family mid-run; that is a new model attempt requiring a new result root and fresh SFT-anchored downstream stages.

Plan by dependencies rather than forcing every stage into a serial checklist. Evidence research, genome collection, execution discovery, and objective planning may overlap once their inputs are clear. Prepare, train, and select SFT before calibration; start RL only after the selected SFT, implemented objectives, calibration, result-root prompt banks, and model-only SFT checkpoint prepared for RL are ready; final generation waits for the selected RL checkpoint. Run independent ready work in parallel only when compute and write scopes fit, while durable monitoring continues without blocking other work.

When operating or adapting the realized PhiX experiment, read the
[example README](../../examples/README.md) as the source of truth for its current commands,
selection handoff, and restart markers. Use its workflow when compatible, but do not assume the
same shell launch or topology fits a different GPU or scheduler environment; adapt execution from
measured hardware while preserving scientific semantics and durable stage boundaries. If the
example stops for sampling review, inspect the completed evidence and follow the calibration
skill's handoff rather than selecting the bundled historical default on the agent's own initiative.
Treat the top-level PhiX script as a reference implementation of the realized DAG. For a rerun, an
agent may run it directly, adapt or wrap it for custom settings and deliberate decision points, or
compose the stage subskills through another scheduler. Use the example README and dependency DAG
to understand current handoffs and stage relationships; let the task and execution environment
determine the orchestration.
Do not replace the canonical sampling selection inside an old or active RL result root; a material
change starts a new SFT-anchored attempt. In the final rollout, preserve separate raw,
biological-representative, hard-QC, and post-QC-cluster denominators.

Record the safety database and model releases used. If one changes during a run, mark the boundary and rerun the affected controls before interpreting comparisons. New runs may use newer releases; missing required evidence remains INDETERMINATE.

Check available storage before large jobs. Preserve active work and the checkpoints needed to resume and interpret the experiment; ask before deleting prior results.

## Design logic

Specify the scientific endpoint, whole-genome design scope, protected traits, and acceptance evidence. Let stage operators adapt methods from measured evidence.

Translating the user's desired final phage product into a complementary collection of RL scores is a core agentic capability. Tested reward implementations are starting points, not a closed catalog. Prefer modifying tested rewards when their measurements and failure semantics transfer; beyond a faithful experiment rerun, novel reward functions are expected for important requirements the current portfolio does not express. Ground new functions in literature, biological reasoning, viable references, domain tools, and prior or partial-run evidence, then give them calibrated partial credit, controls, telemetry, and an explicit relationship to the desired endpoint and final QC.

Treat replication as a case study, not a reason to preserve leaked splits or copy settings blindly. For a new phage or goal, revisit the full reward and filter set. Cover complete-genome viability, the productive-infection lifecycle, [intended-use safety](references/ema-2025-draft-phage-therapy-quality-guideline.md), similarity to viable relatives, host direction, and diversity. A host-range model remains one signal. Align online rewards, final hard filters, and experimental validation; do not reuse target-specific thresholds without evidence.

For therapeutic work, retain each applicable [EMA-derived design guardrail](references/design-scope-and-viability.md#apply-intended-use-therapeutic-guardrails) as a separate measurable component or experimental endpoint. For the PhiX174 case study, keep filters 1–6, 8, and 9 enabled and filter 7 disabled. Keep changed component sets separately interpretable, and diagnose sparse components instead of silently dropping them.

Use `GDPO` and `1/cluster_size` diversity at 99% by default unless a justified alternative is selected. Keep every reward in [0,1], with documented baseline/chance zero, target one, monotonic partial credit, and zero credit plus a recorded reason for missing or invalid data.

## Handoff discipline

- Keep `SUMMARY.md` concise and current; append useful operational detail to `RUNLOG.md`.
- Keep stage outputs together with a short explanation of what they contain.
- Pause when evidence cannot resolve a biologically material choice or execution needs new authority.
