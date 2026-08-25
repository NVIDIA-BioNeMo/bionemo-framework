# PhiX174 whole-genome example on 8×H100

[`phix174_8xh100.sh`](phix174_8xh100.sh) prepares public inputs, performs supervised fine-tuning
(SFT), and optimizes the model with [GDPO](https://arxiv.org/abs/2601.05242), a multi-reward
reinforcement learning (RL) method. It then generates a rollout of 1,000 complete genomes and
applies the current safety and PhiX174 design screens. This reference launcher has been tested on an
8×H100 server. Other topologies may require adjusted settings. The `bionemo-phage-design` skill can
adapt the settings and help run and monitor the job.

This example uses filters 1–6, 8, and 9 from
[Figure 2H](https://www.science.org/doi/10.1126/science.aec2657#F2), following Samuel King's
recommendation (personal correspondence). For RL, these filters become separate scores with partial
credit as candidates approach their passing thresholds. See the
[PhiX174 GDPO score definitions](#current-phix174-gdpo-score-definitions) section below for
more information on these RL scores and how partial credit is assigned.

The end-to-end run takes approximately 4 days on a server with 8 H100 GPUs, and uses 1.5TB of storage.

## Latest completed PhiX174 result

The 8×H100 rerun completed on 2026-08-24:

| Final-rollout denominator                                  |       Count |
| ---------------------------------------------------------- | ----------: |
| Raw generated and SFT-likelihood scored                    |       1,000 |
| Biological representatives after circular/RC deduplication |       1,000 |
| Submitted to safety / excluded by pre-safety QC            |     991 / 9 |
| Safety PASS / FAIL / INDETERMINATE                         | 989 / 0 / 2 |
| Safety-PASS target hard-QC representatives                 |         513 |
| Post-QC 99%-identity clusters and accepted representatives |         511 |

The denominators proceed from raw generation through biological deduplication, safety screening,
and target hard QC, then post-QC clustering. Likelihood is a within-protocol ranking signal; these
computational candidates are not evidence of bootability or wet-lab safety. See the
[case-study notes](../skills/bionemo-phage-design/references/case-study-results.md) for historical context.

## Quick start

Run from `recipes/evo2_phage_gen`. The publication-style reproduction uses `7b-base`, and for this run
we supply our default RL sampling selection settings, skipping the part where we stall for user
input if these settings are not the best by supplying a `--sampling-selection` file:

```bash
./.ci_build.sh
tmux new -s phix174-e2e
./examples/phix174_8xh100.sh \
  --model-variant 7b-base \
  --sampling-selection "examples/default-sampling-selection.yaml" \
  --result-root "$PWD/results/phix174-8xh100"
```

For a fresh PhiX experiment, use the trained-further 7B-1M model and a new result root:

```bash
./examples/phix174_8xh100.sh \
  --model-variant 7b-1m \
  --sampling-selection "examples/default-sampling-selection.yaml" \
  --result-root "$PWD/results/phix174-7b-1m"
```

`7b-base` is the default and matches the published Microviridae lineage. `7b-1m` selects
`evo2/7b-1m:1.0` and provider `evo2_7b`. Checkpoint lineage is authoritative, and the script
refuses to change model families within an existing result root.
The two-character SFT conditioning prefix remains input context but is excluded from next-token
loss; the biological sequence immediately after it remains supervised.

## Common operations

```bash
# Inspect commands without downloads or GPUs.
./examples/phix174_8xh100.sh --dry-run --result-root /tmp/phix174-plan

# Prepare public inputs, tools, databases, and controls only.
./examples/phix174_8xh100.sh --prepare-only --result-root "$PWD/results/phix174-8xh100"

# Resume the RL stage of the 7b-base run.
./examples/phix174_8xh100.sh --resume-from 40 --model-variant 7b-base --result-root "$PWD/results/phix174-8xh100"

# Use an explicitly reviewed sampling selection, without blocking if the automatically identified
#  top setting differs. In practice automatic selection can be noisy, and these settings should work
#  well for phix174.
./examples/phix174_8xh100.sh --sampling-selection examples/default-sampling-selection.yaml --result-root "$PWD/results/phix174-8xh100"
```

Completed stages and substages are skipped. Reuse the same result root and sampling selection when
resuming; a material sampling or model change should start a new run.

### Optional W&B logging

Authenticate with `wandb login` (or provide `WANDB_API_KEY` through a secret manager), then opt in:

```bash
./examples/phix174_8xh100.sh --wandb --wandb-entity YOUR_ENTITY \
  --wandb-sft-project evo2-phage-design-sft \
  --wandb-rl-project evo2-phage-design-gdpo \
  --result-root "$PWD/results/phix174-8xh100"
```

The project flags are optional and show their defaults. Run names are derived from the result-root
name and model variant. W&B covers the full SFT and GDPO runs; smoke tests, held-out evaluation,
and the one-step GDPO pilot remain local. Never pass an API key as a launcher argument. Completed
stages are not uploaded retroactively.

### Choose sampling settings

[`default-sampling-selection.yaml`](default-sampling-selection.yaml) is both the historical PhiX
choice and the schema for a custom selection. To stop after the sweep and scoring, before prompt
banks or RL are created, run:

```bash
./examples/phix174_8xh100.sh --calibrate-only --result-root "$PWD/results/phix174-8xh100"
```

Inspect `results/phix174-8xh100/calibration/scoring/selection-evidence.csv` and its neighboring
score/novelty artifacts. Prefer eligible settings with working metrics, useful hard-QC signal,
low copying, diverse outputs, and a stable quality-diversity plateau. An agent may perform this
review and write the custom choice when the user delegates it. Copy the example YAML, edit all
eight fields, then continue without repeating the completed sweep:

```bash
cp examples/default-sampling-selection.yaml /tmp/phix174-sampling.yaml
# Edit /tmp/phix174-sampling.yaml, then:
./examples/phix174_8xh100.sh --resume-from 30 --sampling-selection /tmp/phix174-sampling.yaml --result-root "$PWD/results/phix174-8xh100"
```

The script validates and records the file as `calibration/sampling-selection.yaml`. Do not replace
that canonical selection after RL has begun; a material change should use a new result root.

## Script workflow and outputs

| Stage | Work                                                                                        |
| ----- | ------------------------------------------------------------------------------------------- |
| 00    | Download inputs, tools, databases, and run controls                                         |
| 10    | Safety-screen inputs and build leakage-controlled SFT splits                                |
| 20    | Train, select, and evaluate SFT                                                             |
| 30    | Calibrate generation and materialize RL prompt banks                                        |
| 40    | Prepare the model-only SFT checkpoint, run the pilot/check, train GDPO, and select RL       |
| 50    | Generate, deduplicate, SFT-score, safety-screen, hard-QC, cluster, and report 1,000 genomes |

The result root is the computational notebook. `RUNLOG.md` records commands and liveness;
`settings.json` records key settings; `SUMMARY.md` and `rollout/final-designs.json` reconcile
selected checkpoints, safety outcomes, QC denominators, clustering, and accepted candidates.
The terminal and `RUNLOG.md` end with `RUN COMPLETE`, `RUN PAUSED`, or `RUN FAILED` plus stage
progress; only `RUN COMPLETE` denotes successful completion of the requested invocation.
The original SFT checkpoint remains available for exact resume and likelihood scoring; RL uses
the smaller prepared checkpoint under `rl/sft-checkpoint/`. The `rollout/accepted_candidates.fasta`
contains the final filter-passing, deduplicated candidates for further analysis. When its scores are
informative and not strongly length-associated, this FASTA is ordered by mean per-nucleotide
likelihood under the selected SFT model; otherwise, generation order is retained.
[Black et al., “Quantifying evolutionary novelty and design efficiency in generative genome
design”](https://www.biorxiv.org/content/10.64898/2026.06.12.731871v1.full) found that Evo 2
likelihood predicted experimental viability within a previously filtered PhiX174 dataset, but the
score used here remains a within-protocol ranking signal rather than a bootability probability or
transferable threshold.

The reference topology is eight H100 80 GB GPUs. `NUM_GPUS` defaults to 8 and `NUM_CPUS` to
`nproc`. SFT tensor parallelism defaults to 1 on a single GPU and 2 otherwise; override
`SFT_TENSOR_PARALLEL_SIZE` only after a full-shape smoke test. When there are fewer GPUs than
prompt strata, the final rollout preserves the full mixture by running deterministic GPU waves.
Use tmux or a scheduler for the long stages.

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
