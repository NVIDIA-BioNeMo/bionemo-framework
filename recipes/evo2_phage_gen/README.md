# Evo 2 Phage Design

This recipe fine-tunes Evo 2 for phage genomes, runs GDPO, and screens generated designs. The result below is the current computational PhiX174 case study; it is not evidence of wet-lab bootability or viability.

## Current result: PhiX174 RL case study

The selected checkpoint was GDPO step 190. Training was monitored through at least step 250 under a 500-step ceiling; step 190 was selected after later checkpoints showed worse quality or diversity, not because 190 is a prescribed stopping step.

| Evaluation | Filter profile | Result |
| --- | --- | ---: |
| Fixed 96-design validation, before clustering | Full target QC | 50/96 (52.08%) |
| Fixed 96-design validation, after the run's configured 99%-identity clustering | Full target QC | 48/96 (50.00%) |
| Independent 1,000-design Arc rollout from step 190 | Filters 1–6, 8, and 9; filter 7 disabled | 358/1,000 (35.80%) |
| Diagnostic branch from the same offline rollout | Filter 7 also enabled | 5/1,000 (0.50%) |

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

The checked evidence and source hashes are in [historical-evidence.md](.agents/skills/bionemo-phage-design/references/historical-evidence.md). The recorded source revision is `99673b047a196352afcbb35e7aa4200127af2616`.

## Agent-run end-to-end result

Not recorded yet. This section will be replaced with the first complete run performed through the recipe-local agent workflow, including its target, SFT lineage, selected RL checkpoint, rollout size, and final QC result.

## Run the workflow with an agent

From the repository root:

```bash
codex -C recipes/evo2_phage_gen \
  'Use $bionemo-phage-design in interactive case-study-replication mode. Reproduce the PhiX174 GDPO case study. Inspect existing results and propose the plan before launching jobs.'
```

Codex discovers the skills in this recipe's `.agents/skills/` directory because `-C` makes the recipe its working directory. The same bundle includes a validated [Codex plugin manifest](.agents/.codex-plugin/plugin.json) for packaging, while other Agent Skills-compatible harnesses can start from [the controller skill](.agents/skills/bionemo-phage-design/SKILL.md).

With Claude Code, load the same skills as a recipe-local plugin:

```bash
cd recipes/evo2_phage_gen
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

The build creates two deliberate environments. The activated main environment
uses the container's BioNeMo/Megatron Python and Torch stack for training. It
also retains the exact recursive NeMo-RL source and creates a locked isolated
vLLM actor environment under `$NEMO_RL_VENV_DIR`. Resolve its interpreter with:

```bash
VLLM_PYTHON="$(python -c 'from bionemo.evo2_phage_gen.nemo_rl_patches import vllm_actor_python_executable; print(vllm_actor_python_executable())')"
test -x "$VLLM_PYTHON"
```

No `PYTHONPATH` override is required after this clean build. On older container
Torch builds that lack `torch._opaque_base`, use the isolated actor interpreter
for vLLM; do not patch upstream vLLM core.

### vLLM inference with the qualified performance profile

Export the selected RL MBridge checkpoint before generation. Do not point vLLM
at a stale pre-RL export:

```bash
RL_CHECKPOINT=/path/to/selected-step/policy/weights/iter_0000000
VLLM_EXPORT="$PWD/results/selected-step-vllm"

python -m bionemo.evo2.vllm.export \
  "$RL_CHECKPOINT" "$VLLM_EXPORT" \
  --max-shard-size 2GiB

sha256sum \
  "$VLLM_EXPORT/config.json" \
  "$VLLM_EXPORT/model.safetensors.index.json" \
  "$VLLM_EXPORT/manifest.json"
```

The export manifest records the source checkpoint inventory and generated
config/index hashes. Use a fresh output directory for every selected policy.
Supply `--config /path/to/verified/config-or-export` only when the checkpoint
does not contain sufficient model config, and record that extra authority.

For ordinary standalone generation, use the public inference module through
the locked actor interpreter. The paired RL arguments make the CLI verify the
checkpoint config/tensor inventory and tokenizer semantics against the export
before vLLM engine construction:

```bash
RL_TOKENIZER_JSON="$PWD/../evo2_megatron/tokenizers/nucleotide_fast_tokenizer_512/tokenizer.json"

"$VLLM_PYTHON" -m bionemo.evo2.vllm.infer \
  --model "$VLLM_EXPORT" \
  --rl-checkpoint "$RL_CHECKPOINT" \
  --rl-tokenizer-json "$RL_TOKENIZER_JSON" \
  --prompt "ATCGATCGATCGATCG" \
  --max-new-tokens 5988 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 4 \
  --tensor-parallel-size auto \
  --batch-size 96 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.91 \
  --optimization-level 2 \
  --performance-mode balanced \
  --async-scheduling \
  --output-file results/evo2-vllm-generations.jsonl
```

Use `--prompt-file` for a caller-owned JSONL prompt bank. `auto` uses all
visible assigned GPUs and selects the multiprocess executor when TP is greater
than one. The CLI derives the required context length unless
`--max-model-len` is supplied explicitly. Prompt files may use either the
flat `{\"id\": ..., \"prompt\": ...}` inference schema or the recipe's
`messages`/`prompt_id`/`length_stratum`/`rollout_ordinal` schema. The latter
preserves those grouping coordinates in every output row and uses each
caller-owned `validation_seed`; the checked-in P8/K12 validation manifest can
therefore be passed directly without a conversion script.

Use one persistent engine when comparing first-use and steady mixed inference.
The following additions run the checked-in P8/K12 bank twice without replaying
request identities or seeds:

```bash
  --prompt-file data/phage_prompts_paper_useful_rl_validation_mixed_8x12.jsonl \
  --repetitions 2 \
  --generation-seed-stride 1000003 \
  --output-file results/evo2-vllm-p8k12-two-wave.jsonl
```

The generated manifest's `physical_waves` entries report public engine-generate
and post-output validation time separately for each B96 wave. The second entry
is the steady persistent-engine comparison. Every repeated row retains its
prompt group while advancing rollout ordinal, order index, request ID, and seed.

The optimized Evo2 vLLM path is selected through the benchmark profile rather
than independent private vLLM patches. This example reproduces the measured
two-H100 reference profile: O2/balanced runtime selection, Inductor compilation mode 3,
`FULL_AND_PIECEWISE` CUDA graphs including the exact B96 capture size,
the multiprocess executor, and async scheduling:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_PLUGINS=evo2

"$VLLM_PYTHON" -m bionemo.evo2.vllm.benchmark \
  --backend vllm \
  --checkpoint /path/to/evo2-vllm-export \
  --manifest /path/to/workload-manifest.json \
  --topology tp2 \
  --max-model-len 6016 \
  --max-new-tokens 5988 \
  --request-count 96 \
  --global-wave-size 96 \
  --max-num-seqs 96 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.91 \
  --optimization-level 2 \
  --performance-mode balanced \
  --distributed-executor-backend mp \
  --async-scheduling \
  --output results/evo2-vllm-b96.json
```

For this reference command, the resolved artifact must report TP2, executor
`mp`, async scheduling enabled, compilation mode 3, `FULL_AND_PIECEWISE`, and
capture size 96. Omit
`--async-scheduling` only for an explicit sync comparison. Use
`"$VLLM_PYTHON" -m ...`; running a module file by path can let the local
`profile.py` shadow Python's standard-library module.

On a system other than the qualified two-H100 node, first discover assigned GPU
count/topology/free memory and run a small correctness/capacity probe. Use every
assigned GPU through a supported topology. TP2 is the measured reference, not a
maximum: TP4 or TP8 may be used when model/context memory or the tested hardware
favors a wider tensor-parallel engine. Use remaining GPUs as disjoint DP engine
groups when that improves throughput, with disjoint prompt/seed partitions and
no CUDA-device overlap. Recompute local wave size and include every actual local
batch shape in the CUDA-graph capture list. Preserve the same output/logprob/QC
gates when comparing topologies; choose the fastest end-to-end configuration
with memory headroom rather than assuming that higher TP is always faster.

### Mixed RL batching contract

The production mixed workload uses eight prompt-length strata and twelve
stochastic generations per prompt: `P=8`, `K=12`, and
`train_global_batch_size=P*K=96`. The local vLLM generation request contains
`GBS/DP` rows: 96 for DP1, or 48 for each DP2 engine. The DP2 validation
manifest assigns six rollouts from every length stratum to each rank.

`policy.train_micro_batch_size` is only the MCore forward/backward chunk size;
it is not a prompt count and does not define advantage groups. Capacity-test the
full local training batch first, then use the largest stable divisor of the local
batch and accumulate the remaining chunks. Capacity-test
`policy.logprob_batch_size` independently. Assemble all K rewards for a prompt
before within-prompt advantage normalization, including when its rollouts span
DP ranks.

The primary frozen validation bank is
`data/phage_prompts_paper_useful_rl_validation_mixed_8x12.jsonl`
(SHA256 `fa9bc74d3784333a5daf29f2c1149dbd7baa302907723ca449aec4bd5e1b8a6b`).
Report every length stratum plus the fixed equal-weight aggregate. The
`validation_prompt10_96` bank remains a historical single-length control.
The one-step `gdpo_phage_vllm_tp2_one_step_smoke.yaml` uses P8xK2/GBS16 and is
explicitly capacity-bounded; it is not the production batch contract.

Run that bounded end-to-end smoke through the production launcher before
scaling to P8xK12/GBS96:

```bash
evo2_phage_run_gdpo --config configs/gdpo_phage_vllm_tp2_one_step_smoke.yaml \
  checkpointing.pretrained_checkpoint.path=/path/to/mbridge-checkpoint \
  policy.model_name=/path/to/fresh-vllm-export \
  policy.tokenizer.name="$PWD/../evo2_megatron/tokenizers/nucleotide_fast_tokenizer_512" \
  checkpointing.checkpoint_dir="$PWD/results/vllm-gdpo-smoke/checkpoints" \
  logger.log_dir="$PWD/results/vllm-gdpo-smoke/logs"
```

The launcher keeps policy/reference workers in the main environment and routes
only the vLLM generation actors to the pinned actor interpreter. For production,
set P8xK12/GBS96, use local generation batch `GBS/DP`, and capacity-test MBS and
logprob batch size independently without changing prompt groups.

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
evo2_convert_vortex_to_mbridge \
  --hf-repo-id evo-design/evo-2-7b-8k-microviridae \
  --hf-filename evo2_7b_microviridae.pt \
  --revision a363aa61d628e5342d5ee148bc0dbac27a1533b7 \
  --mbridge-ckpt-dir data/checkpoints/evo2_7b_microviridae_mbridge \
  --model-size evo2_7b_microviridae \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 \
  --seq-length 10240 \
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

There is not yet a checked-in canonical launcher for the complete paper SFT run, so the main reproduction path above uses the released SFT checkpoint. The published full fine-tune used 12,000 iterations, 32 H100 GPUs, a sample batch of 32, context length 10,240, 327,680 tokens per optimizer step, initial learning rate `1e-5`, 5% warmup, and cosine decay to `1e-6`.

## Troubleshooting

- If an entrypoint is missing, rerun `.ci_build.sh`, source `.ci_test_env.sh`, and check `pyproject.toml` plus `<command> --help`.
- If GDPO runs out of memory, lower the microbatch first while preserving the effective global batch; see the [resource and OOM guide](.agents/skills/bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md).
- If external QC fails, verify the large databases completed successfully and inspect the per-stage logs under the rollout root.
- If a fresh run selects a checkpoint other than step 190, use that checkpoint for rollout and report its validation evidence. Step 190 is historical context, not a fixed target.

## References

- [Generative design of novel bacteriophages with genome language models](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full)
- [Checked paper, supplement, and figure assets](.agents/skills/bionemo-phage-design/assets/literature/king-2025-generative-phage-design/)
- [Skill validation record](.agents/skills/bionemo-phage-design/assets/VALIDATION.md)
- [Historical result evidence](.agents/skills/bionemo-phage-design/references/historical-evidence.md)
- [Public Microviridae SFT checkpoint](https://huggingface.co/evo-design/evo-2-7b-8k-microviridae)
- [Recipe commands and dependency pins](pyproject.toml)
- [Evo 2 model and checkpoint notes](../evo2_megatron/README.md)
