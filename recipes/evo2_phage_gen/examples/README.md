# PhiX174 whole-genome example on 8×H100

[`phix174_8xh100.sh`](phix174_8xh100.sh) is the agent-free realization of the current recipe. It
prepares public inputs and tools, trains SFT and GDPO, generates 1,000 complete genomes, scores
every design with the selected SFT, and applies the current sequence-safety and PhiX174 design
screens. These are computational candidates, not
evidence of bootability, therapeutic fitness, or wet-lab safety.

The reference command was tested from `recipes/evo2_phage_gen` on eight H100 80 GB GPUs:

```bash
./.ci_build.sh

# Recommended when a durable batch scheduler is not keeping the process alive.
tmux new -s phix174-e2e

./examples/phix174_8xh100.sh \
  --model-variant 7b-base \
  --result-root "$PWD/results/phix174-8xh100"
```

The build creates the virtual env used by `.ci_test_env.sh`, which the example sources internally.
The primary command uses `evo2/7b-8k:1.0`, model size `evo2_7b_base`, because the published
Microviridae work and the realized PhiX case-study lineage used the 7B-base family. For a fresh
PhiX experiment, the trained-further 1M checkpoint is probably the better starting point and can
still run at this example's shorter 10,240-token sequence length:

```bash
./examples/phix174_8xh100.sh \
  --model-variant 7b-1m \
  --result-root "$PWD/results/phix174-7b-1m"
```

That selects `evo2/7b-1m:1.0`, model size `evo2_7b`, and the matching NeMo-RL provider name.
Checkpoint lineage is authoritative; a mislabeled `policy.model_name` in old run metadata does not
change which pretrained weights were used. Nevertheless, keep the provider name consistent in new
configs. A result root that already contains a 7B-base SFT run must remain on that family when
resuming. Moving to 1M is a new result root and SFT-anchored attempt, not a mid-run override.
`NUM_GPUS` defaults to 8 and controls SFT processes, RL topology, calibration GPUs, rollout
workers, and likelihood scoring. `NUM_CPUS` defaults to `nproc` and sets the Ray logical-CPU budget.
The script checks the configured GPU count. Other accelerator models are allowed with a warning
because memory, batch, and parallelism settings may need tuning; the reference settings are not a
performance claim for another topology. `NUM_GPUS` must currently be even for the SFT
tensor-parallel layout. Restricted agent sandboxes often hide `nvidia-smi`, so launch on the
allocated compute node. Use a scratch clone or worktree when a run needs code or config changes.

Useful controls:

```bash
# Inspect all commands without downloads or GPUs.
./examples/phix174_8xh100.sh --dry-run --result-root /tmp/phix174-plan

# Download public data/tools and run safety controls before allocating the full training window.
./examples/phix174_8xh100.sh --prepare-only \
  --result-root "$PWD/results/phix174-8xh100"

# Resume at a stage boundary; completed stages are skipped automatically.
./examples/phix174_8xh100.sh --resume-from 30 \
  --model-variant 7b-base \
  --result-root "$PWD/results/phix174-8xh100"

# Continue an interrupted 7B-base run whose stage-40 pilot did not complete.
./examples/phix174_8xh100.sh --resume-from 40 \
  --model-variant 7b-base \
  --result-root "$PWD/results/phix174-8xh100"

# Explicitly use the historical case-study settings instead of requiring fresh calibration support.
./examples/phix174_8xh100.sh \
  --result-root "$PWD/results/phix174-8xh100" \
  --sampling-selection examples/default-sampling-selection.yaml
```

If an unfinished stage is interrupted, rerun the original command with the same result root. The
script reuses completed stages and cached or partial downloads, so the run directory need not be
deleted. Only one invocation may use a result root at a time. If AMRFinder fails, its captured
output is retained as `amrfinder/amrfinder.log` below that scan directory.

### Choose or override sampling settings

Stage 30 uses `${RESULT_ROOT}/calibration/sampling-selection.yaml` as the canonical sampling
selection. When `--sampling-selection PATH` is supplied, the script validates the file and copies
it to that location before running any stage. The supplied file therefore works both on the first
invocation and when resuming after calibration. If the canonical file already exists, stage 30
uses it and skips the hard check against the bundled historical choice. Without either file, the
script writes the bundled default only when fresh calibration supports it; otherwise it stops for
review without repeating completed calibration generation or scoring.

After a selection is available, stage 30 creates `${RESULT_ROOT}/rl/train.jsonl` and
`${RESULT_ROOT}/rl/validation.jsonl`. Stage 40 automatically checks the run-specific training bank
and passes both files to training; do not create the template `data/phix174_rl_*` files manually.
A rerun with the same result root recreates either prompt bank if needed while reusing completed
calibration work.

For the specific post-calibration stop caused by an unsupported historical choice, either accept
the historical settings explicitly:

```bash
./examples/phix174_8xh100.sh \
  --result-root "$PWD/results/phix174-8xh100" \
  --sampling-selection examples/default-sampling-selection.yaml
```

This logs a warning because the selection is an operator override, not a conclusion from the new
calibration. Subsequent reruns with the same result root may omit `--sampling-selection`; the copied
canonical file remains in the run record.

To make an evidence-based selection instead, inspect
`calibration/scoring/selection-evidence.csv`. Exclude rows where `eligible` or
`metric_environment_ok` is false, confirm the external measurements are available, compare the
aggregate reward and target-signal confidence intervals, and reject settings whose apparent yield
comes from high target/SFT copy rates or weak within-setting diversity. Prefer a stable
quality-diversity plateau over a noisy maximum, and prefer temperature 1.0 only when it is
practically comparable. Prompt lengths in one file share one temperature and are deployed as an
equal mixture.

Copy `examples/default-sampling-selection.yaml` as a schema example and edit every value:

```yaml
temperature: 0.9
top_k: 4
top_p: 1.0
max_new_tokens: 5976
prompt_lengths: [12, 24]
rl_seed: 42
rollout_seed: 7
seed_stride: 1000003
```

The file must contain exactly these keys. Temperatures and token counts must be positive, `top_p`
must be in `(0, 1]`, prompt lengths must be unique PhiX174 prefixes between 0 and 65 nt, and the
longest prompt plus `max_new_tokens` must fit the 10,240-token context. This example uses equal
prompt strata, so their count must divide the 12-record training bank, 96-record validation bank,
1,000-record final rollout, and configured GPU count. One, two, or four strata work with the
reference eight-GPU shape.

You may write the reviewed file directly to
`results/phix174-8xh100/calibration/sampling-selection.yaml` and rerun the original command, or
keep it elsewhere and pass `--sampling-selection PATH`. On an old or active RL result root, pass
that option again only for the identical reviewed selection. Replacing it in place can introduce
prompt, validation, and sampling drift partway through RL. A material change belongs in a new
result root and RL attempt from the selected SFT checkpoint, with the earlier run retained as
evidence.

Long-running work also writes narrower completion markers: `20-sft.done`,
`30-calibration-generation.done`, `30-calibration-scoring.done`, `40-rl.done`, and
stage-50 markers for `rollout`, `deduplication`, `sft-likelihood`, `sequence-safety`,
`target-profile`, `filter7-diagnostic`, `final-clustering`, and `report`. These let a resumed run
skip an accepted result while still performing its downstream selection, evaluation, screening,
clustering, and reporting. Each completed stage-50 marker is checked against its required output
before it is reused. For example, after inspecting
checkpoints from an interrupted SFT run:

```bash
touch results/phix174-8xh100/stages/20-sft.done
./examples/phix174_8xh100.sh --resume-from 20 \
  --result-root "$PWD/results/phix174-8xh100"
```

Create a narrow marker manually only when that substage's outputs are complete enough to accept;
ordinary successful runs create it automatically.

The six dependency-ordered stages are input/control preparation, SFT safety and splitting, SFT
training and selection, sampling calibration, model-only SFT-to-RL checkpoint preparation plus
GDPO pilot/training, and final generation, biological deduplication, SFT likelihood scoring,
hard-QC screening, post-QC clustering, and reporting.
Calibration uses `NUM_GPUS` in parallel; SFT, GDPO, rollout generation, and likelihood scoring use
the same configured count. On the reference node, where `nproc` reports 160 logical CPUs, the large safety
scans use 128-record batches, 32 parallel ORF predictions, 32 threads for AMRFinder/DIAMOND, and 64
for the PHROGs MMseqs search. The GDPO reward phases are sequential and use at most 64 tool threads;
`NUM_CPUS` sets the Ray CPU budget. `CALIBRATION_WORKERS` and the existing `SAFETY_*`
environment variables can lower the CPU sub-budgets on a smaller or shared node. Recheck global
batch divisibility and memory use when changing GPU topology.
Long commands remain supervised by the top-level process, with a meaningful liveness update every
ten minutes. Run that process in tmux or a scheduler so it survives a disconnected chat or shell.

The realized example defaults to the Pharokka v1.11.0 Zenodo bundle through public
`PHAROKKA_DATABASE_*` environment variables. Override the three values together when using another compatible release. It uses the bundle’s PHROGs v4 profiles and
annotations, derives the consensus database used by Arc, and logs progress to
`inputs/external-assets.log`. The transfer is bounded and resumable. Databases are refreshed rather
than pinned to the historical run: the script records the installed state and reruns positive,
review, and negative controls; changed behavior stops for review instead of silently selecting an
older database. Required detector failures remain INDETERMINATE.
The previous SFT-corpus audit observed 14,465 PASS and one lysogeny-review INDETERMINATE among
14,466 inputs; every new run records and uses its own result.

SFT selection uses optimizer-step validation. A larger global batch can complete an epoch before
enough optimizer updates exist to show overfitting, so a best value at the run boundary stops for
more follow-up even if that requires several epochs.
The SFT command keeps the best three validation-loss checkpoints plus the latest resume point.
`validation_metrics.json` stores all scalar validation measurements, and `checkpoint_metrics.json`
stores each save-time metric assignment and a `best_checkpoint` relative directory pointer. The
example requires the raw `lm loss` key; the corresponding TensorBoard tag is `lm loss validation`.
Before the GDPO pilot, stage 40 runs:

```bash
evo2_phage_prepare_sft_checkpoint_for_rl \
  --source-checkpoint results/phix174-8xh100/sft/train/evo2/checkpoints/iter_NNNNNNN \
  --output-dir results/phix174-8xh100/rl/sft-checkpoint
```

The command rewrites the distributed checkpoint without optimizer, scheduler, or RNG state,
removes `train_state.pt`, and nulls process-local model callbacks and timers in the copied
`run_config.yaml`. It leaves the selected SFT checkpoint unchanged and writes
`rl/sft-checkpoint/preparation-manifest.json`; the script then uses the manifest's direct
`iter_*` path for RL readiness, policy initialization, and the fixed SFT KL anchor. Rerunning the
same top-level command validates and reuses a matching prepared checkpoint. An existing but
mismatched or incomplete prepared checkpoint stops for review rather than being overwritten. A
matching schema-1 preparation is rebuilt atomically because that older reducer could omit
serialized Transformer Engine model state required by strict checkpoint loading.

PhiX174 reference runs through the exact configured RL environment and every enabled external,
diversity, and safety measurement must report support. The pilot then checks training and checkpoint
behavior; rewards stay in `[0, 1]`, baseline/chance means `0`, missing or failed measurements cannot
look favorable, and the selected SFT checkpoint remains the KL anchor.

Stage 50 scores all 1,000 raw designs with the selected pre-RL SFT checkpoint and its `+~`
conditioning prefix, while exact, circular, and reverse-complement biological equivalents are
collapsed before expensive safety and Arc screening. Arc's internal clustering is disabled. Only
safety-PASS representatives that pass the target hard-QC branch enter the final MMseqs clustering
at 99% identity, 80% coverage, coverage mode 0, and cluster mode 0; filter 7 remains a separate
diagnostic branch. `rollout/sft-likelihood/ranked-designs.csv` therefore retains raw-design total
and mean per-nucleotide log probability, while `rollout/final-designs.json` reconciles the raw,
biological-representative, hard-QC, and post-QC-cluster denominators. `accepted_candidates.fasta`
contains one representative per final cluster and follows the SFT ranking when it is usable. The
report also computes Spearman correlation between length and the normalized score.
At absolute rho 0.5 or greater, it preserves the likelihood results but keeps the accepted FASTA in
generation order because residual length bias makes the likelihood ordering unreliable.

This use is supported—but not validated as a selection rule for a new campaign—by
[Black et al.](https://doi.org/10.64898/2026.06.12.731871), who found that Evo 2 likelihood
separated experimentally bootable from non-bootable PhiX174 designs. Both that paper and the local
scorer comparison support within-protocol enrichment, not a universal threshold or proof of
bootability; raw total log probability is retained for inspection but is not the ranking score
because it scales with sequence length.

The result root is the electronic lab notebook. `RUNLOG.md` records commands, liveness, failures, and
stage completion; `settings.json` contains only the small scientific setting allowlist. Checkpoint
selection, calibration, objective-health, likelihood scores, safety summaries, target and diagnostic
Arc waterfalls, deduplication mapping, hard-QC set, final cluster memberships, accepted candidates,
and the final `SUMMARY.md` remain beside their stage outputs. The final report carries the selected
checkpoints and sampling settings plus concise safety tool/database and clustering provenance. A
boundary-best checkpoint, changed safety control, unsupported calibration choice, or unhealthy RL
objective is an intentional review stop; routine setup and execution need no agent intervention.

## Current PhiX174 GDPO score definitions

This is the human-readable contract for the 15 objectives in
`configs/gdpo_phage_megatron.yaml`. It is also the worked example for the run-specific
`artifacts/RL_SCORE_DEFINITIONS.md` that an agent writes when designing or changing objectives;
the E2E shell script does not generate that artifact. These thresholds reproduce the current
PhiX174 computational profile, not universal phage-design optima or evidence of bootability.

GDPO receives each row below as a separate `[0, 1]` objective. The scalar `weight_*` settings are
diagnostic and do not reweight objectives after GDPO normalization. The first 12 objectives are
forced to zero unless the sequence has an exact sequence-safety `PASS`; the three safety
objectives remain unmasked so failures still provide learning signal. Unless a row says otherwise,
missing, invalid, non-finite, or failed measurements receive zero. Final checkpoint selection uses
the stricter safety-qualified, full-QC, cluster-deduplicated pass rate rather than mean reward.

### Sequence feasibility

| Objective (reward column)                    | Zero credit                                                                                          | Full credit                                                                                                                                | Partial credit and rationale                                                                                                                                                                                                                                                           |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `valid_nt_chars` (`reward_valid_nt_chars`)   | Any emitted character outside A/C/G/T.                                                               | No non-ACGT character is present.                                                                                                          | Binary. The raw helper regards an empty string as having no invalid character, but empty output fails length and aggregate nucleotide gates and cannot receive the safety-qualified GDPO objective. This prevents malformed sequence text from satisfying downstream biological tools. |
| `genome_length` (`reward_genome_length`)     | Length at or below 2,000 nt, or at or above 8,000 nt.                                                | 4,000–6,000 nt inclusive.                                                                                                                  | Let *L* be length. Below 4,000: `max(0, 1 - (4000-L)/2000)`; above 6,000: `max(0, 1 - (L-6000)/2000)`. The target band brackets the 5,386-nt [PhiX174 reference genome](https://www.ncbi.nlm.nih.gov/nuccore/NC_001422.1).                                                             |
| `gc_content` (`reward_gc_content`)           | Mathematically at or below −5% or at or above 100%; only the 100% endpoint is physically attainable. | 30–65% inclusive.                                                                                                                          | Below 30%: `max(0, 1 - (30-GC)/35)`; above 65%: `max(0, 1 - (GC-65)/35)`. Thus 0% GC still scores 1/7, while 100% scores 0. The broad band rejects extreme composition while leaving room around the PhiX reference.                                                                   |
| `nt_homopolymer` (`reward_nt_homopolymer`)   | No finite positive run length reaches exactly zero; invalid or unavailable measurements score zero.  | Maximum nucleotide run *H* ≤ 10 bases.                                                                                                     | For *H* > 10, score `10/H`, decreasing asymptotically toward zero. Long homopolymers are discouraged because they are low-complexity and can complicate synthesis and sequencing.                                                                                                      |
| `dustmask_end` (`reward_dustmask_end`)       | No valid masked fraction reaches zero; failed or invalid DUST measurement scores zero.               | Maximum DUST-masked fraction *F* over either terminal 200-nt window ≤ 0.9.                                                                 | For 0.9 < *F* ≤ 1, score `0.9/F` (0.9–1.0). The separate nucleotide-pass objective supplies the hard cutoff. DUST detects low-complexity sequence using the approach described by [Morgulis et al.](https://doi.org/10.1089/cmb.2006.13.1028).                                         |
| `nucleotide_pass` (`reward_nucleotide_pass`) | Any component gate fails.                                                                            | All characters are A/C/G/T, length is 4,000–6,000 nt, GC is 30–65%, maximum homopolymer is ≤10, and both terminal DUST fractions are ≤0.9. | Binary conjunction. It supplies an explicit feasibility milestone in addition to the dense component objectives.                                                                                                                                                                       |

### Protein evidence, architecture, and diversity

| Objective (reward column)                                               | Zero credit                                                                                                                                             | Full credit                                                                                                                                    | Partial credit and rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `protein_hit_count` (`reward_external_protein_hit_count`)               | No measured PHROGs hit, missing output, or tool failure.                                                                                                | At least 7 protein-database hits.                                                                                                              | For *C*=1…6 hits, score `C/7`. [PHROGs](https://doi.org/10.1093/nargab/lqab067) supplies curated prokaryotic-virus protein families, so this is graded evidence of recognizable phage coding content, not proof of function.                                                                                                                                                                                                                                                                                        |
| `tropism` (`reward_external_tropism`)                                   | No measured hit or 0% identity to the PhiX G spike protein.                                                                                             | Best measured protein identity *I* ≥ 60%.                                                                                                      | For 0 < *I* < 60%, score `I/60`; identity above the threshold is not penalized. This preserves the current Arc target-host-recognition proxy used by the [PhiX design workflow](https://www.science.org/doi/10.1126/science.aec2657).                                                                                                                                                                                                                                                                               |
| `required_genes` (`reward_external_required_genes`)                     | Missing metrics, an empty required-label definition, or zero labels matched.                                                                            | Every configured required label is matched and the definition contains at least 9 labels.                                                      | Score `(matched/total) × min(total/9, 1)`, where `total` is the number of configured required labels. The nine current labels are terminase, endolysin, major spike protein, DNA replication initiation, DNA condensation, major head protein, head morphogenesis, pilot protein for DNA ejection, and Arc's literal `nan` compatibility label for the reference ORF with unknown function. This supplies partial credit for recovered reference functions without treating annotations as experimental validation. |
| `synteny` (`reward_external_synteny`)                                   | Impossible `syntenic > total`, missing output, or tool failure. No valid finite gene-count pair otherwise reaches exactly zero.                         | `(syntenic,total)` is one of `(10,10)`, `(10,11)`, `(10,12)`, `(11,12)`, or `(12,12)`; `(11,11)` is intentionally excluded by the Arc profile. | Let `d_total` be distance of total genes from [10,12] and `d_pair` Manhattan distance to the nearest full-credit pair. Score `1/(1+d_total) × 1/(1+d_pair)`. It rewards gradual recovery of PhiX-like gene count and ordering while retaining the exact final gate.                                                                                                                                                                                                                                                 |
| `average_protein_identity` (`reward_external_average_protein_identity`) | Missing identity, no measured proteins, or tool failure.                                                                                                | AAI ≤95% with at least 10 measured protein entries.                                                                                            | Score `novelty × min(gene_count/10,1)`, where novelty is 1 through 95% AAI and `max(0.25, (100-AAI)/5)` above 95%. Novelty falls to 0.25 by 98.75% and then plateaus; evidence grows linearly through 10 proteins. This separates novelty from annotation support.                                                                                                                                                                                                                                                  |
| `mmseqs_cluster_diversity` (`reward_mmseqs_cluster_diversity`)          | The genome fails basic nucleotide feasibility, is missing from MMseqs output, or the tool fails. No finite cluster size otherwise reaches exactly zero. | A singleton within its prompt group.                                                                                                           | With 99% minimum sequence identity, a member of a cluster of size *N* scores `1/N`. This directly rewards within-batch sequence diversity; [MMseqs2](https://doi.org/10.1038/nbt.3988) provides the clustering implementation.                                                                                                                                                                                                                                                                                      |

### Mandatory whole-genome safety objectives

All three classes are required for the PhiX bacterial-host profile. Each class uses the same
categorical mapping: `PASS → 1`, a review-eligible measured `INDETERMINATE` with findings `→ 0.25`,
and `FAIL`, unmeasured, invalid, or tool-failed `→ 0`. The hard safety gate still requires all
required classes to be `PASS`; partial credit never qualifies a candidate for acceptance. This
separation follows the conservative screening posture in `configs/phage_safety_policy.yaml` and
regulatory emphasis on excluding detrimental genes and temperate behavior in [EMA's phage-therapy
quality guideline](https://www.ema.europa.eu/en/quality-aspects-phage-therapy-medicinal-products).

| Objective (reward column)                    | Full-credit state | What the class screens                                                                   |
| -------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| `safety_amr` (`reward_safety_amr`)           | `PASS`            | Acquired antimicrobial-resistance determinants.                                          |
| `safety_toxin` (`reward_safety_toxin`)       | `PASS`            | Toxin and virulence-associated protein evidence.                                         |
| `safety_lysogeny` (`reward_safety_lysogeny`) | `PASS`            | Integrases, repressors, excision machinery, and other lysogeny/temperate-phage evidence. |
