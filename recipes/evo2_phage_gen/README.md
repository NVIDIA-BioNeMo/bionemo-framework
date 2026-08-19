# Evo 2 Phage Design

This recipe fine-tunes Evo 2 for phage genomes, runs GDPO, and screens generated designs. The result below is the current computational PhiX174 case study; it is not evidence of wet-lab bootability or viability.

## Current agent-driven result: PhiX174 SFT+RL case study

SFT was performed on the same set of sequences as the original publication, with an added stage to verify that no sequences held-out for validation were 99% or
more similar to any sequences in the training sets. 14266 genomes were in training, 100 in validation and 100 in test. 12,000 maximum steps were performed, and the checkpoint at
step 5,600 was chosen by the agent as having the lowest validation loss at 0.750670. It had a similar test set loss of 0.798180. This is significantly higher than
the loss reported by the evo2 microviridae model, which may have been overfit by the 12,000th step. A loss in this range however is more in line with validation/test
set losses reported by the model on other validation sets when training the original 7B model, so this may be a better starting point for RL than the published microviridae checkpoint.

The current AMR, toxin, and lysogeny screen was run retrospectively on the 14,466 SFT input
records. It found 14,465 PASS, 0 FAIL, and 1 INDETERMINATE: `sft_000540` (source accession
`MH617328.1`) had a lysogeny review-profile signal. This is a possible hit for review, not a
confirmed hazard. The completed SFT run used the full corpus before this audit.

Pre-RL calibration was performed to choose settings for RL. A temperature of 1.0 performed similarly to other settings, so was chosen. We also chose a mix of 50% length 16 prompts, and 50% length 24 prompts.

GDPO ran for 500 steps. Step 430 was selected because it had the highest
`val:phage_qc/binary_safety_qualified_full_qc_cluster_deduplicated_rate`. The latest target-profile rollout passed at 61%; the publication-era screen is shown only for context because it used a different pipeline and is not a controlled enrichment baseline.

| Evaluation                                              | Screen                                          |               Result |
| ------------------------------------------------------- | ----------------------------------------------- | -------------------: |
| Prior GDPO run: published Microviridae SFT + step 190   | Arc filters 1–6, 8, and 9; filter 7 disabled    |   358/1,000 (35.80%) |
| Latest SFT+GDPO run: step 430                           | Same target profile                             |   610/1,000 (61.00%) |
| Latest step-430 diagnostic                              | All Arc filters, including filter 7             |     22/1,000 (2.20%) |
| Publication baseline: published Microviridae SFT, no RL | Publication-era screen; not directly comparable | 15/110,000 (~0.014%) |

The target profile intentionally disables architecture-removal filter 7 and retains total-gene logic.

Target-profile offline filter counts:

```text
1,000 generated
  → 998 valid nucleotide
  → 994 length/GC
  → 992 nucleotide/ORF
  → 957 protein/CheckV/GA gates
  → 815 tropism
  → 815 representatives at 99% identity
  → 815 AAI
  → 698 required genes
  → 610 synteny/total-gene final passes
```

A retrospective run of the current AMR, toxin, and lysogeny screen accounted for all 1,000 local
rollout records: 997 PASS, 0 biological FAIL, 1 INDETERMINATE 135-nt record with no predicted genes,
and 2 records rejected by nucleotide QC because each contained one ambiguous `R`. The safety result
is separate from the 610-record target-profile result above.

The concise [case-study notes](skills/bionemo-phage-design/references/case-study-results.md) keep the current end-to-end run distinct from the earlier released-SFT shortcut.

## Run the workflow with an agent

From the repository root, use `$bionemo-phage-generation` to locate or acquire a compatible checkout and begin high-level planning:

```bash
codex \
  'Use $bionemo-phage-generation to locate or acquire a compatible Evo 2 phage-generation recipe checkout, then begin interactive planning for the PhiX174 GDPO case-study replication.'
```

The portable bootstrap skill hands off to the recipe-local implementation skills after the checkout is resolved. From `recipes/evo2_phage_gen`, use `$bionemo-phage-design` to execute the workflow:

```bash
codex \
  'Use $bionemo-phage-design in interactive case-study-replication mode. Reproduce the PhiX174 GDPO case study. Inspect existing results and propose the plan before launching jobs.'
```

The recipe-local skill bundle includes a validated [Codex plugin manifest](.codex-plugin/plugin.json). Other Agent Skills-compatible harnesses can start from [the implementation controller skill](skills/bionemo-phage-design/SKILL.md).

With Claude Code, run the following from `recipes/evo2_phage_gen` to load the recipe-local plugin:

```bash
claude --plugin-dir . \
  '/evo2-phage-gen:bionemo-phage-design Use interactive case-study-replication mode. Reproduce the PhiX174 GDPO case study. Inspect existing results and propose the plan before launching jobs.'
```

Review the proposed plan, approve or revise it, then tell the same agent to execute the approved
plan through the terminal report. A proposal alone does not launch the experiment.

## Reproduce the current end-to-end PhiX174 result manually

The [8×H100 PhiX174 example](examples/README.md) is the realized, agent-free workflow. Its one
shell script uses maintained package entry points plus the calibration helpers in
[`scripts/`](scripts/README.md); it does not execute attempt-local code from an archived result.

From `recipes/evo2_phage_gen` on a node with eight H100 80 GB GPUs:

```bash
./.ci_build.sh

# Optional but recommended unless a durable scheduler owns the command:
tmux new -s phix174-e2e

./examples/phix174_8xh100.sh \
  --result-root "$PWD/results/phix174-8xh100"
```

The top-level command downloads the public corpus, current external tools and databases, reruns the
safety controls, excludes non-PASS SFT inputs, builds leakage-controlled SFT splits, trains and
selects SFT, calibrates sampling, verifies every enabled RL measurement on the PhiX174 reference,
runs a full-shape pilot and DP8 GDPO, selects RL from comparable validation, generates exactly 1,000
whole genomes, scores all of them with the selected pre-RL SFT,
and runs the current sequence-safety, target Arc, and filter-7 diagnostic screens. The GDPO config includes AMR, toxin, and lysogeny objectives and
uses the selected SFT checkpoint as its KL anchor. On the reference 160-CPU allocation, large safety
scans use 32-record batches and no phase exceeds 64 tool threads; see the
[realized example](examples/README.md) for the adjustable settings.

The realized SFT stage validates and saves every 400 optimizer steps, retaining the three lowest
validation-loss checkpoints plus the latest resume checkpoint. Its checkpoint directory contains
`validation_metrics.json` with every scalar validation metric and `checkpoint_metrics.json` with
the save-time validation assignment plus a `best_checkpoint` relative directory pointer. This
example stops if the raw `lm loss` metric is absent or cannot be matched; TensorBoard exposes the same value as `lm loss validation`.

The example records commands and periodic liveness observations in `RUNLOG.md` without copying the
environment. The realized example defaults to PHROGs v4 from the Pharokka v1.11.0 Zenodo
bundle through public `PHAROKKA_DATABASE_*` environment variables. Override the three values together when using another compatible release. It derives the Arc-compatible
consensus database locally and writes download progress to
`inputs/external-assets.log`. Transfers use bounded retries and resume partial files. Database
updates are recorded and checked with the control panel rather than rejected or silently replaced
with historical versions. Its final report keeps PASS, FAIL, and INDETERMINATE
counts separate and writes the intersection of safety-PASS and target-profile candidates. It also
writes total and mean per-nucleotide SFT log probability for all 1,000 designs, checks residual
score-length correlation, and adds the scores and accepted ordering to `rollout/final-designs.json`.
Likelihood ordering is omitted when that length diagnostic remains strongly confounded. This is a
within-protocol enrichment signal motivated by [Black et al.](https://doi.org/10.64898/2026.06.12.731871),
not a bootability threshold.

Useful operator modes are:

```bash
# No downloads or GPUs; inspect the complete command plan.
./examples/phix174_8xh100.sh --dry-run --result-root /tmp/phix174-plan

# Prepare public data/tools/controls first.
./examples/phix174_8xh100.sh --prepare-only \
  --result-root "$PWD/results/phix174-8xh100"

# Resume at a stage boundary after correcting an interruption.
./examples/phix174_8xh100.sh --resume-from 30 \
  --result-root "$PWD/results/phix174-8xh100"
```

After an interrupted unfinished stage, rerun the original top-level command with the same result
root. Completed stage markers are skipped and cached or partial downloads are reused; deleting the
result directory is not required. Only one invocation may use a result root at a time; a concurrent
invocation exits instead of sharing stage work directories.

Read the [example README](examples/README.md) for scientific review stops, monitoring behavior,
safety details, and the result layout. A scratch clone or worktree per
campaign is preferable to copying source code into a run directory.

## Download and preprocess the publication-era SFT data

The released inputs and historical preprocessing path are available:

```bash
evo2_phage_download_sft_data --include-raw
preprocess_evo2 --config configs/sft_microviridae_preprocess.yaml
```

The public [CC BY bioRxiv version](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full) links the publication-era supplement and data.

## Summarize a sequence-safety scan

For a large FASTA on comparable hardware, use bounded batches and parallel ORF prediction rather
than preparing every record serially:

```bash
evo2_phage_sequence_safety scan \
  --input-fasta genomes.fasta --output-dir results/PROJECT/safety \
  --policy configs/phage_safety_policy.yaml \
  --asset-manifest data/external/safety/asset_manifest.yaml \
  --host-domain BACTERIA --host-evidence-json "$HOST_EVIDENCE_JSON" --strict-lysis \
  --batch-size 128 --orf-workers 32 --threads 32 --phrogs-threads 64
```

Batch size controls how often the external tools and databases are started; when memory permits,
use a batch large enough to cover one RL generation batch and benchmark larger FASTA scans before
scaling. Choose smaller thread and worker values when CPU affinity, memory, I/O, or concurrent work
requires it. The scan log names the active detector and reports per-phase timing. Asset preparation
derives the small lysogeny search database from the PHROGs families selected in the current Pharokka
release, while recording the release used. After the scan completes, write a compact PASS, FAIL, and
INDETERMINATE summary:

```bash
evo2_phage_summarize_safety_manifest \
  --manifest results/PROJECT/sft/scan/manifest.json \
  --output results/PROJECT/sft/scan/safety_tally.json
```

The command checks that the per-record results and totals agree. If only representatives were
scanned, keep representative and source-record denominators distinct.

## Troubleshooting

- If an entrypoint is missing, rerun `.ci_build.sh`, source `.ci_test_env.sh`, and check `pyproject.toml` plus `<command> --help`.
- If GDPO runs out of memory, preserve whole-genome context and the effective batch while following the [compute guidance](skills/bionemo-phage-design-adapt-execution/references/compute-guidance.md).
- If external QC fails, verify the large databases completed successfully and inspect the per-stage logs under the rollout root.
- Use the checkpoint selected by validation evidence rather than treating a historical step as a fixed target.

## Acknowledgements

- [Samuel King (@samhkin)](https://github.com/samhkin) for discussions about the recipe and promising RL filters, including the proposal to test Arc filters 1–6, 8, and 9 as a challenging subset.
- Jessica Sacher and [Jan Zheng (@janzheng)](https://github.com/janzheng) for introductions to the phage community, discussions that shaped the skill and recipe, and hands-on testing and feedback.
- [Avery Noonan (@Noonanav)](https://github.com/Noonanav) for discussions about GenoPHI and phage–host interaction models more broadly.
- Michael Poon and colleagues at [Tabula Bio](https://www.tabulabio.com/) for encouraging support for appropriately scoped design work beyond whole-phage-genome generation.
- Eric Bastien and Nick Conley of [Locus Biosciences](https://locus-bio.com) for valuable feedback, including the addition of Shiga toxin-converting phage 933W and WOPip1 as positive controls for the safety filters.

## References and background reading

### Publications

- [Final Science publication: *Generative design of novel bacteriophages with genome language models*](https://www.science.org/doi/10.1126/science.aec2657)
- [Evo 2 publication of record: *Genome modelling and design across all domains of life with Evo 2*](https://doi.org/10.1038/s41586-026-10176-5)
- [CC BY bioRxiv v1 paper with linked supplement and data](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full)

### Books

- Merry Youle, [*Thinking Like a Phage*](https://thinkinglikeaphage.wordpress.com/), for an accessible tour of phage biology.
- Tom Ireland, [*The Good Virus*](https://www.hachette.co.uk/titles/tom-ireland/the-good-virus/9781529365283/), for the history and future promise of bacteriophage therapy.
- Steffanie Strathdee and Thomas Patterson, [*The Perfect Predator*](https://theperfectpredator.com/), for the clinical story that helped renew interest in phage therapy.

### Recipe resources

- [PhiX174 case-study results](skills/bionemo-phage-design/references/case-study-results.md)
- [Public Microviridae SFT checkpoint](https://huggingface.co/evo-design/evo-2-7b-8k-microviridae)
- [Recipe commands and dependencies](pyproject.toml)
- [Evo 2 model and checkpoint notes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/main/recipes/evo2_megatron/README.md)
