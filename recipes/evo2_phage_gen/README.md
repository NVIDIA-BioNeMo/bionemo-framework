# Evo 2 Phage Design

This recipe fine-tunes Evo 2 for phage genomes, runs GDPO, and screens generated designs. The result below is the current computational PhiX174 case study; it is not evidence of wet-lab bootability or viability.

## Current result: PhiX174 RL case study

The selected checkpoint was GDPO step 190. Training was monitored through at least step 250 under a 500-step ceiling; step 190 was selected after later checkpoints showed worse quality or diversity, not because 190 is a prescribed stopping step.

| Evaluation                                                                     | Filter profile                           |             Result |
| ------------------------------------------------------------------------------ | ---------------------------------------- | -----------------: |
| Fixed 96-design validation, before clustering                                  | Full target QC                           |     50/96 (52.08%) |
| Fixed 96-design validation, after the run's configured 99%-identity clustering | Full target QC                           |     48/96 (50.00%) |
| Independent 1,000-design Arc rollout from step 190                             | Filters 1–6, 8, and 9; filter 7 disabled | 358/1,000 (35.80%) |
| Diagnostic branch from the same offline rollout                                | Filter 7 also enabled                    |    5/1,000 (0.50%) |

The target profile intentionally disables architecture-removal filter 7 and retains total-gene logic. The 96-design online validation and 1,000-design offline rollout use different pipelines and clustering contracts; their rates must not be combined.

Target-profile offline waterfall:

```text
1,000 generated
  → 996 valid nucleotide
  → 974 length/GC
  → 940 nucleotide/ORF
  → 897 protein/CheckV/GA gates
  → 861 tropism
  → 645 representatives at 99% identity
  → 425 AAI
  → 425 required genes
  → 358 synteny/total-gene final passes
```

The checked evidence and source hashes are in [historical-evidence.md](../../.agents/skills/bionemo-phage-design/references/historical-evidence.md). The recorded source revision is `99673b047a196352afcbb35e7aa4200127af2616`.

## Agent-run end-to-end result

Not recorded yet. This section will be replaced with the first complete run performed through the repository-level agent workflow, including its target, SFT lineage, selected RL checkpoint, rollout size, and final QC result.

## Run the workflow with an agent

From the repository root:

```bash
codex \
  'Use $bionemo-phage-design in interactive case-study-replication mode. Reproduce the PhiX174 GDPO case study. Inspect existing results and propose the plan before launching jobs.'
```

The skill bundle lives at the repository root and includes a validated [Codex plugin manifest](../../.agents/.codex-plugin/plugin.json). The controller resolves the selected recipe independently of its install location, runs recipe commands from that directory, and keeps results there. Other Agent Skills-compatible harnesses can start from [the controller skill](../../.agents/skills/bionemo-phage-design/SKILL.md).

With Claude Code, load the same top-level plugin from the repository root:

```bash
claude --plugin-dir .agents \
  '/evo2-phage-gen:bionemo-phage-design Use interactive case-study-replication mode. Reproduce the PhiX174 GDPO case study. Inspect existing results and propose the plan before launching jobs.'
```

## Reproduce the RL result

The shortest path reuses the public 12,000-iteration Microviridae SFT checkpoint and reruns GDPO, generation, and filtering. Run every command below from `recipes/evo2_phage_gen`.

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

This prepares MMseqs2, BLAST/DUST, DIAMOND, HMMER, PHROGs, CheckV, and the pinned Arc workflow used by the final filters.

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
mkdir -p "$RL_ROOT"

evo2_phage_run_gdpo --config configs/gdpo_phage_megatron.yaml \
  checkpointing.pretrained_checkpoint.path=data/checkpoints/evo2_7b_microviridae_mbridge \
  checkpointing.checkpoint_dir="$RL_ROOT/checkpoints" \
  env.phage_qc.external_qc.work_dir="$RL_ROOT/external_qc" \
  env.phage_qc.mmseqs_cluster_diversity.work_dir="$RL_ROOT/mmseqs_cluster_diversity" \
  logger.log_dir="$RL_ROOT/logs" \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=true
```

The overrides replace project-specific checkpoint, output, and W&B settings in the checked-in historical config. Validation runs every 10 steps with a 500-step ceiling. Select the best checkpoint from sustained full-QC and diversity evidence; do not stop at step 190 merely because it was best in the recorded run.

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

Exact numeric reproduction is source-revision-sensitive; the current inference code includes the persistent-RNG correction used by this command.

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

## Re-run SFT from the published data

The released inputs and historical preprocessing path are available:

```bash
evo2_phage_download_sft_data --include-raw
preprocess_evo2 --config configs/sft_microviridae_preprocess.yaml
```

There is not yet a checked-in canonical launcher for the complete paper SFT run, so the main reproduction path above uses the released SFT checkpoint. For a new run, download the training genomes, inspect their tokenized length distribution, and agree on a high-coverage context rule (propose p99.9 or the affordable maximum), including worst-case control/prompt/EOD overhead and required alignment, before selecting the model, effective token batch, and SFT/RL settings. The bundled paper supplement preserves the exact historical configuration.

## Troubleshooting

- If an entrypoint is missing, rerun `.ci_build.sh`, source `.ci_test_env.sh`, and check `pyproject.toml` plus `<command> --help`.
- If GDPO runs out of memory, lower the microbatch first while preserving the effective global batch; see the [resource and OOM guide](../../.agents/skills/bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md).
- If external QC fails, verify the large databases completed successfully and inspect the per-stage logs under the rollout root.
- If a fresh run selects a checkpoint other than step 190, use that checkpoint for rollout and report its validation evidence. Step 190 is historical context, not a fixed target.

## References

- [Generative design of novel bacteriophages with genome language models](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full)
- [Checked paper, supplement, and figure assets](../../.agents/skills/bionemo-phage-design/assets/literature/king-2025-generative-phage-design/)
- [Skill validation record](../../.agents/skills/bionemo-phage-design/assets/VALIDATION.md)
- [Historical result evidence](../../.agents/skills/bionemo-phage-design/references/historical-evidence.md)
- [Public Microviridae SFT checkpoint](https://huggingface.co/evo-design/evo-2-7b-8k-microviridae)
- [Recipe commands and dependency pins](pyproject.toml)
- [Evo 2 model and checkpoint notes](../evo2_megatron/README.md)
