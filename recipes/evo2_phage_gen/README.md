# Evo 2 Phage Design

This recipe fine-tunes Evo 2 for phage genomes, runs GDPO, and screens generated designs. The result below is the current computational PhiX174 case study; it is not evidence of wet-lab bootability or viability.

## Current agent-driven result: PhiX174 SFT+RL case study

SFT was performed on the same set of sequences as the original publication, with an added stage to verify that no sequences held-out for validation were 99% or
more similar to any sequences in the training sets. 14266 genomes were in training, 100 in validation and 100 in test. 12,000 maximum steps were performed, and the checkpoint at
step 5,600 was chosen by the agent as having the lowest validation loss at 0.750670. It had a similar test set loss of 0.798180. This is significantly higher than
the loss reported by the evo2 microviridae model, which may have been overfit by the 12,000th step. A loss in this range however is more in line with validation/test
set losses reported by the model on other validation sets when training the original 7B model, so this may be a better starting point for RL than the published microviridae checkpoint.

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

## Reproduce the current end-to-end PhiX174 result

The successful end-to-end run used this sequence:

1. Build the recipe environment with `./.ci_build.sh`, then source `.ci_test_env.sh`.
2. Ask the controller for a fresh-base PhiX174 replication plan, review it, and run through the final report.
3. Collect and validate the Microviridae genomes, then make a cluster-held-out split. The recorded split contained 14,266 training, 100 validation, and 100 test genomes with no near-duplicate leakage at the chosen boundary.
4. Fine-tune fresh Evo 2 7B to a 12,000-step ceiling. Select checkpoints by validation loss; the run selected step 5,600 and confirmed it on the held-out test set.
5. Define and validate the objective/QC profile while SFT runs. Calibrate the selected SFT checkpoint; the run chose temperature 1.0 and a 50:50 mixture of 16- and 24-nucleotide prompts.
6. Run data-parallel GDPO with 96 designs per step, validation every 10 steps, and a 500-step ceiling. Select from validation evidence; the run selected step 430.
7. Generate exactly 1,000 designs from the selected checkpoint. Apply the target filter profile with filter 7 disabled, then run a separate filter-7-enabled diagnostic and write the final report.

The scripts under the recorded result preserve the implementation used for that experiment, but some contain attempt-local paths. Let the controller generate current commands for a new result root instead of copying those archived scripts verbatim.
Run long stages through a facility that survives the chat process, keep the job identifier and logs in `RUNLOG.md`, and reattach until success or failure is known.

## Historical RL-only shortcut from the released SFT checkpoint

This shorter path reuses the public 12,000-iteration Microviridae SFT checkpoint and reruns GDPO, generation, and filtering. It does not reproduce the current fresh-base SFT plus step-430 result. Run every command below from `recipes/evo2_phage_gen`.

The historical RL shape used one node with 2× H100 80 GB GPUs. Other hardware may work after reducing batch sizes or changing parallelism. External QC preparation also requires network access and substantial disk space.

### 1. Build the environment

```bash
./.ci_build.sh
source .ci_test_env.sh
```

### 2. Prepare external QC and Arc

```bash
evo2_phage_prepare_external_assets \
  --external-dir data/external \
  --bin-dir .venv/bin \
  --download-large-databases

evo2_phage_prepare_arc_pipeline \
  --output-dir data/arc_pipeline_patched

TROPISM_DB_DIR="data/external/mmseqs/NC_001422_1_Gprotein"
mkdir -p "$TROPISM_DB_DIR"
mmseqs createdb \
  data/external/arc_evo2/phage_gen/data/NC_001422.1_Gprotein.fasta \
  "$TROPISM_DB_DIR/mmseqs_db_NC_001422_1_Gprotein"
```

This prepares MMseqs2, BLAST/DUST, DIAMOND, HMMER, PHROGs, CheckV, and the Arc workflow used by the final filters.
Record the tool and database releases used. If one changes during a run, note when it changed and rerun the affected controls before comparing or screening candidates. New runs may use newer releases.

### 3. Convert the released Microviridae SFT checkpoint

```bash
: "${TARGET_CONTEXT_LENGTH:?Set from the agreed upper bound of the tokenized training-genome length distribution}"

evo2_convert_vortex_to_mbridge \
  --hf-repo-id evo-design/evo-2-7b-8k-microviridae \
  --hf-filename evo2_7b_microviridae.pt \
  --revision a363aa61d628e5342d5ee148bc0dbac27a1533b7 \
  --mbridge-ckpt-dir data/checkpoints/evo2_7b_microviridae_mbridge \
  --model-size evo2_7b_microviridae \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 \
  --seq-length "$TARGET_CONTEXT_LENGTH" \
  --mixed-precision-recipe bf16_mixed
```

The explicit filename is required because the hosted file does not match the converter's inferred default name.

### 4. Create the training and validation prompts

```bash
evo2_phage_generation prepare-rl-prompts --data-dir data

evo2_phage_generation write-prompts \
  --output-dir data \
  --prompt-lengths 10 \
  --num-prompts 96 \
  --id-prefix phage_prompts_paper_useful_rl_validation
```

The second command creates the 96-row validation file referenced by the historical GDPO configuration.

### 5. Run GDPO

```bash
RESULT_ROOT="$PWD/results/phix174-gdpo-replication"
RL_ROOT="$RESULT_ROOT/rl"
PHAGE_WANDB_ENABLED="${PHAGE_WANDB_ENABLED:-false}"
mkdir -p "$RL_ROOT"

evo2_phage_run_gdpo --config configs/gdpo_phage_megatron.yaml \
  checkpointing.pretrained_checkpoint.path=data/checkpoints/evo2_7b_microviridae_mbridge \
  checkpointing.checkpoint_dir="$RL_ROOT/checkpoints" \
  env.phage_qc.external_qc.work_dir="$RL_ROOT/external_qc" \
  env.phage_qc.mmseqs_cluster_diversity.work_dir="$RL_ROOT/mmseqs_cluster_diversity" \
  logger.log_dir="$RL_ROOT/logs" \
  logger.wandb_enabled="$PHAGE_WANDB_ENABLED" \
  logger.wandb.project=evo2_phage_design_rl_gdpo \
  logger.wandb.name=phix174-gdpo-replication \
  logger.tensorboard_enabled=true
```

The overrides set the checkpoint and result paths for the historical config. W&B is optional; set `PHAGE_WANDB_ENABLED=true` only when it is already available, and keep the local TensorBoard logs. Validation runs every 10 steps with a 500-step ceiling. Select the best checkpoint from sustained full-QC and diversity evidence; do not stop at step 190 merely because it was best in the recorded run.

### 6. Generate the 1,000-design rollout

Set `SELECTED_CHECKPOINT` to the checkpoint selected from your run. The path below is the expected path if step 190 is selected.

```bash
RESULT_ROOT="$PWD/results/phix174-gdpo-replication"
RL_ROOT="$RESULT_ROOT/rl"
SELECTED_CHECKPOINT="$RL_ROOT/checkpoints/step_190/policy/weights/iter_0000000"
ROLLOUT_ROOT="$RESULT_ROOT/rollout-step190-n1000"

RUN_ROOT="$ROLLOUT_ROOT" \
CKPT_DIR="$SELECTED_CHECKPOINT" \
PROMPT_LENGTHS="10" \
TEMPERATURES="1.0" \
NUM_PROMPTS=1000 \
TARGET_LENGTH=6000 \
TOP_K=4 TOP_P=1.0 SEED=7 \
NPROC_PER_NODE=2 TENSOR_PARALLEL_SIZE=2 PROMPT_BATCH_SIZE=64 \
  scripts/run_paper_hpo_generation.sh
```

Numeric results can vary with the code, runtime, and accelerator environment; record those along with the sampling settings.

### 7. Apply the target filter profile

```bash
RESULT_ROOT="$PWD/results/phix174-gdpo-replication"
ROLLOUT_ROOT="$RESULT_ROOT/rollout-step190-n1000"

RUN_ROOT="$ROLLOUT_ROOT" \
TARGET_RECORDS=1000 \
CELL_GLOB='phix174_prompt10_temp1.0.manifest1000.fasta' \
GENETIC_ARCHITECTURE_REMOVE_FILTER=0 \
  scripts/run_paper_hpo_full_arc_scoring.sh
```

The main outputs are:

- `scores/hpo_full_arc_summary.md` and `.csv` for the aggregate result;
- per-stage counts under `arc_filtering/`;
- `qc6_synteny_filter_seqs.csv` and `.fasta` for final passing designs;
- generation manifests, resolved Arc configs, logs, and intermediate FASTA files under the rollout root.

Run the filter-7-enabled diagnostic in a separate rollout directory so its fixed summary filenames do not overwrite the target-profile outputs.

## Download and preprocess the publication-era SFT data

The released inputs and historical preprocessing path are available:

```bash
evo2_phage_download_sft_data --include-raw
preprocess_evo2 --config configs/sft_microviridae_preprocess.yaml
```

The bundled paper supplement describes the publication-era configuration.

## Summarize a validated sequence-safety scan

After `evo2_phage_sequence_safety scan` publishes a terminal schema-2 manifest, emit exact
PASS, FAIL, INDETERMINATE, class-state, reason-code, and mutually exclusive class-combination counts:

```bash
evo2_phage_summarize_safety_manifest \
  --manifest results/PROJECT/sft/runs/ATTEMPT/artifacts/scan/manifest.json \
  --output results/PROJECT/sft/runs/ATTEMPT/artifacts/safety_tally.json
```

The command revalidates the full scan result and detector evidence before writing the tally. If only
representative sequences were scanned, convert to source-record counts only from a complete representative mapping.

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
- [Bundled CC BY bioRxiv v1 paper, supplement, figures, and data](skills/bionemo-phage-design/assets/literature/king-2025-generative-phage-design/) ([bioRxiv source](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full))

### Books

- Merry Youle, [*Thinking Like a Phage*](https://thinkinglikeaphage.wordpress.com/), for an accessible tour of phage biology.
- Tom Ireland, [*The Good Virus*](https://www.hachette.co.uk/titles/tom-ireland/the-good-virus/9781529365283/), for the history and future promise of bacteriophage therapy.
- Steffanie Strathdee and Thomas Patterson, [*The Perfect Predator*](https://theperfectpredator.com/), for the clinical story that helped renew interest in phage therapy.

### Recipe resources

- [PhiX174 case-study results](skills/bionemo-phage-design/references/case-study-results.md)
- [Public Microviridae SFT checkpoint](https://huggingface.co/evo-design/evo-2-7b-8k-microviridae)
- [Recipe commands and dependency pins](pyproject.toml)
- [Evo 2 model and checkpoint notes](../evo2_megatron/README.md)
