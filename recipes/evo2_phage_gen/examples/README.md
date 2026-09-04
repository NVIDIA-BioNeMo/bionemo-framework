# PhiX174 whole-genome example on 8×H100

[`phix174_8xh100.sh`](phix174_8xh100.sh) prepares public inputs, performs supervised fine-tuning
(SFT), and optimizes the model with [GDPO](https://arxiv.org/abs/2601.05242), a multi-reward
reinforcement learning (RL) method. It then generates a rollout of 1,000 complete genomes and
applies the current safety and PhiX174 design screens. This reference launcher has been tested on an
8×H100 server. Other topologies may require adjusted settings. The `bionemo-phage-design` skill can
adapt the settings and help run and monitor the job.

This example uses filters 1–6, 8, and 9 from
[Figure 2H](https://www.science.org/doi/10.1126/science.aec2657#F2), following Samuel King's
recommendation (personal correspondence). RL represents their measurable constraints as separate
graded or categorical objectives; reward full-credit regions and final hard-pass thresholds are not
interchangeable, and batch diversity is not a per-genome hard gate. See the
[PhiX174 GDPO score definitions](#current-phix174-gdpo-score-definitions) section below for
the exact reward, checkpoint-selection, and final-filter relationships.

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

That completed run used the earlier single-origin prompt semantics. The current launcher starts a
new `results/phix174-8xh100-mixed-anchors` result root and does not resume that run in place.

## Quick start

Run from `recipes/evo2_phage_gen`. The PhiX follow-up uses the published-lineage `7b-base` checkpoint
with the current mixed-anchor defaults. Supplying `--sampling-selection` explicitly skips the
fresh-calibration review stop:

```bash
./.ci_build.sh
tmux new -s phix174-e2e
./examples/phix174_8xh100.sh \
  --model-variant 7b-base \
  --sampling-selection "examples/default-sampling-selection.yaml" \
  --result-root "$PWD/results/phix174-8xh100-mixed-anchors"
```

The build reuses the native Torch/CUDA/Transformer Engine stack already in the
NVIDIA PyTorch container. The launcher sources the resulting recipe-local
`.ci_test_env.sh`; for standalone commands, source that file yourself.

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
./examples/phix174_8xh100.sh --prepare-only --result-root "$PWD/results/phix174-8xh100-mixed-anchors"

# Resume the RL stage of the 7b-base run.
./examples/phix174_8xh100.sh --resume-from 40 --model-variant 7b-base --result-root "$PWD/results/phix174-8xh100-mixed-anchors"

# Use an explicitly reviewed sampling selection, without blocking if the automatically identified
#  top setting differs. In practice automatic selection can be noisy, and these settings should work
#  well for phix174.
./examples/phix174_8xh100.sh --sampling-selection examples/default-sampling-selection.yaml --result-root "$PWD/results/phix174-8xh100-mixed-anchors"
```

Completed stages and substages are skipped. Reuse the same result root and sampling selection when
resuming; `--resume-from` selects where checking resumes but does not create missing state or
completion markers. A material prompt, reward, sampling, or model change should start a new result
root. Stage 20 records its real-data restart smoke separately as `stages/20-sft-smoke.done`; it
validates and reuses a complete converted base checkpoint instead of redownloading or reconverting
it. Stage 40 likewise keeps `stages/40-pilot.done`, `stages/40-pilot-check.done`, and
`stages/40-rl.done` distinct, and reuses only a validated schema-2 optimizer-free prepared SFT
checkpoint. Do not create a marker unless its operation is known to have succeeded.

### Optional W&B logging

Authenticate with `wandb login` (or provide `WANDB_API_KEY` through a secret manager), then opt in:

```bash
./examples/phix174_8xh100.sh --wandb --wandb-entity YOUR_ENTITY \
  --wandb-sft-project evo2-phage-design-sft \
  --wandb-rl-project evo2-phage-design-gdpo \
  --result-root "$PWD/results/phix174-8xh100-mixed-anchors"
```

The project flags are optional and show their defaults. Run names are derived from the result-root
name and model variant. SFT has its own optional W&B run. The GDPO W&B run is not created until the
full stage-40 training starts: calibration and the pilot stay offline, while the full run honors
the entity, project, and name overrides. Never pass an API key as a launcher argument. Completed
stages are not uploaded retroactively.

### Choose sampling settings

[`default-sampling-selection.yaml`](default-sampling-selection.yaml) contains the current PhiX
defaults and is the schema for a custom selection. Passing `--sampling-selection` is an explicit
override, not a choice derived from the fresh calibration evidence. The defaults use temperature
1, top-k 5, and top-p 0.999; sampled EOD is retained so the length reward can teach termination.
The decoder ceiling reaches approximately
5,466–5,474 nt across the prompt mixture, within the length reward's upper declining slope and below
its 5,494-nt zero edge. The 5,359–5,391-nt full-credit band sits inside the broader
5,306–5,493-nt hard acceptance interval. For the circular PhiX reference, the deployed
16- and 24-nt prompts start at 1-based positions 2,285 (`after_f`) and 3,918 (`after_h`). These
intervals avoid annotated CDS/features and have lower overlapping-CDS occupancy than the old
coordinate origin, but an unannotated regulatory element may still exist. The calibration also
includes origin position 1 as a comparison control. This rotation strategy does not apply to a
linear genome, whose biological endpoints must be preserved.

Each 96-rollout GDPO update uses 16 prompt records × 6 generations. With DP8, each 12-request decode
batch contains both anchors at one prompt length, while the global update contains all four
anchor×length strata. Validation independently contains the same mixture. The 1,000-design final
rollout uses 250 prompts per stratum: two 500-record, same-length files alternate anchors and combine
to exactly 1,000 records.

The launcher sets `max_model_len` to the smallest 256-token boundary covering the longest selected
prompt, its two-token `+~` prefix, and `max_new_tokens` (5,632 with the defaults), rather than
allocating the unused remainder of the 10,240-token SFT context. Native RL trajectories retain the
first sampled EOD and its log-probability, mask only synthetic padding, and exclude EOD and any
post-EOD physical samples from biological QC. Filtered policy replay also keeps each sampled action
in a normalized target-preserving support; generation-versus-replay error telemetry remains enabled.

`RL_PROMPT_BATCH_SIZE` controls the packed mixed-length decode group size (default 12); set a larger
value only on a qualified device profile that has enough memory for the corresponding cache.

To stop after the sweep and scoring, before prompt banks or RL are created, run:

```bash
./examples/phix174_8xh100.sh --calibrate-only --result-root "$PWD/results/phix174-8xh100-mixed-anchors"
```

Inspect `results/phix174-8xh100-mixed-anchors/calibration/scoring/selection-evidence.csv` and its neighboring
score/novelty artifacts. Prefer eligible settings with working metrics, useful hard-QC signal,
low copying, diverse outputs, and a stable quality-diversity plateau. An agent may perform this
review and write the custom choice when the user delegates it. Copy the example YAML, including its
named `prompt_anchors`, then continue without repeating the completed sweep:

```bash
cp examples/default-sampling-selection.yaml /tmp/phix174-sampling.yaml
# Edit /tmp/phix174-sampling.yaml, then:
./examples/phix174_8xh100.sh --resume-from 30 --sampling-selection /tmp/phix174-sampling.yaml --result-root "$PWD/results/phix174-8xh100-mixed-anchors"
```

The script validates and records the file as `calibration/sampling-selection.yaml`. Do not replace
that canonical selection after RL has begun; a material change should use a new result root.

Calibration scoring uses the same sequence-safety asset manifest, policy, bacterial host domain,
and confirmed versioned PhiX host evidence as online RL. Sampling cells fail closed when required
external or safety evidence is unexplained or unavailable; exact documented biological
inapplicability remains candidate-level `INDETERMINATE`, not a safety `PASS`.

## Script workflow and outputs

| Stage | Work                                                                                        |
| ----- | ------------------------------------------------------------------------------------------- |
| 00    | Download inputs, tools, databases, and run controls                                         |
| 10    | Safety-screen inputs and build leakage-controlled SFT splits                                |
| 20    | Train, select, and evaluate SFT                                                             |
| 30    | Calibrate generation and materialize RL prompt banks                                        |
| 40    | Prepare the model-only SFT checkpoint, run the pilot/check, train GDPO, and select RL       |
| 50    | Generate, SFT-score, deduplicate, safety-screen, hard-QC, cluster, and report 1,000 genomes |

The result root is the computational notebook. `RUNLOG.md` records commands and liveness;
`settings.json` records key settings; `SUMMARY.md` and `rollout/final-designs.json` reconcile
selected checkpoints, safety outcomes, QC denominators, clustering, and accepted candidates.
The terminal and `RUNLOG.md` end with `RUN COMPLETE`, `RUN PAUSED`, or `RUN FAILED` plus stage
progress; only `RUN COMPLETE` denotes successful completion of the requested invocation.
Final likelihood scoring runs before deduplication and uses the validated direct model path in
`rl/sft-checkpoint/preparation-manifest.json`, as does RL. The original full-state SFT checkpoint
remains available for exact SFT resume. The `rollout/accepted_candidates.fasta`
contains the final filter-passing, deduplicated candidates for further analysis. When its scores are
informative, not strongly length-associated, and drawn from one circular origin, this FASTA may be
ordered by mean per-nucleotide likelihood under the selected SFT model. Mixed-origin runs retain
generation order because whole-sequence language-model likelihood depends on the linearized origin;
the scores remain recorded as diagnostics.
The final rollout and the inner GDPO rollout both use packed dynamic prefill and batched recurrent
decode for medium/long generation. GDPO assigns 12 requests to each of eight data-parallel replicas
for its 96-sequence step and requires exact-length, EOS-suppressed completions. Final selected-SFT
likelihood uses packed prediction for ragged batches while preserving FASTA record mappings. NeMo-RL's
separate `policy.sequence_packing` option remains disabled until its gradient-bearing loss/backward
path is independently qualified; it does not control rollout packing.
Each shard keeps its decode rows logically active and captures only the physical full/remainder
batch shapes it actually runs. Graph reuse requires stable registered model storage; an unexpected
parameter or buffer address change on any TP/PP/CP rank forces replica-wide recapture without
crossing DP replicas. Qualify at least two rollout/offload-refit cycles. Report cold end-to-end,
setup-free generation, and steady-decode throughput separately; set
`EVO2_EXACT_PHASE_EVIDENCE=1` only for synchronized phase and allocator diagnostics.
[Black et al., “Quantifying evolutionary novelty and design efficiency in generative genome
design”](https://www.biorxiv.org/content/10.64898/2026.06.12.731871v1.full) found that Evo 2
likelihood predicted experimental viability within a previously filtered PhiX174 dataset, but the
score used here remains a within-protocol ranking signal rather than a bootability probability or
transferable threshold.

Before GDPO, the exact configured environment scores the coordinate origin and both deployed
reference rotations together and requires identical reward, filter, and measurement-support
outcomes. Arc hard QC removes ORFipy calls beginning wholly inside its appended pseudocircular tail,
while retaining cross-origin ORFs, so tail length cannot create rotation-dependent duplicate genes.
Raw endpoint-local DUST fractions can differ by linear origin, but the tested PhiX rotations have
the same reward and pass outcome; the exact control enforces the deployed outcome contract.

The reference topology is eight H100 80 GB GPUs. `NUM_GPUS` defaults to 8 and `NUM_CPUS` to
`nproc`. SFT tensor parallelism defaults to 1 on a single GPU and 2 otherwise; override
`SFT_TENSOR_PARALLEL_SIZE` only after a full-shape smoke test. Packed dynamic prefill interleaves
the selected prompt lengths inside each deterministic GPU shard, so length-stratum count does not
need to divide a microbatch or the GPU count. Use tmux or a scheduler for the long stages.

BF16 remains the portable default. After H100/H200 qualification, pass
`--hopper-fp8-inference` to use regular FP8 across compatible 7B linears for calibration, rollout,
and likelihood scoring without changing GDPO training precision. Vortex-style FP8 is separate.
Decode batches divisible by eight avoid regular FP8's alignment fallback.

## Current PhiX174 GDPO score definitions

This is the human-readable contract for the 16 objectives in
`configs/gdpo_phage_megatron.yaml`. It is also the worked example for the run-specific
`artifacts/RL_SCORE_DEFINITIONS.md` that an agent writes when designing or changing objectives;
the E2E shell script does not generate that artifact. These thresholds reproduce the current
PhiX174 computational profile, not universal phage-design optima or evidence of bootability.

GDPO receives each row below as a separate `[0, 1]` objective. The scalar `weight_*` settings are
diagnostic and do not reweight objectives after GDPO normalization. The first 13 objectives are
forced to zero unless the sequence has an exact sequence-safety `PASS`; the three safety
objectives remain unmasked so failures still provide learning signal. Missing, invalid, non-finite,
or unavailable measurements map to zero when scoring returns a row. Configured Arc, DUST, or
diversity-command failures can instead fail the scoring batch rather than fabricate a biological
score. Final checkpoint selection uses the stricter safety-qualified, full-QC,
cluster-deduplicated pass rate rather than mean reward.

### Objective implementation map

The ordered objective inventory and reward-column wiring live in the
[GDPO configuration](../configs/gdpo_phage_megatron.yaml). The environment validates and
assembles that matrix in
[`gdpo_objective_scores_from_scored`](../src/bionemo/evo2_phage_gen/nemo_rl_env.py), including
the exact-safety mask. The implementations for the individual terms are:

| Objective                  | Primary implementation                                                                                                                                                                                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `valid_nt_chars`           | [`has_valid_nt_chars`](../src/bionemo/evo2_phage_gen/qc.py) and [`score_nucleotide_metrics`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                             |
| `genome_length`            | [`_genome_length_score`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                                                                                 |
| `gc_content`               | [`calculate_gc_content`](../src/bionemo/evo2_phage_gen/qc.py) and [`_interval_score`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                    |
| `nt_homopolymer`           | [`calculate_nt_homopolymer_len`](../src/bionemo/evo2_phage_gen/qc.py) and [`_upper_bound_ratio_score`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                   |
| `dustmask_end`             | [`calculate_dustmasker_metrics`](../src/bionemo/evo2_phage_gen/qc.py) and [`score_nucleotide_metrics`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                   |
| `nucleotide_pass`          | [`score_nucleotide_metrics`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                                                                             |
| `protein_hit_count`        | [`_add_mmseqs_hit_rewards`](../src/bionemo/evo2_phage_gen/reward.py) and [`protein_alignment_integrity`](../src/bionemo/evo2_phage_gen/protein_evidence.py)                                                                                                       |
| `tropism`                  | [`smooth_protein_match_integrity`](../src/bionemo/evo2_phage_gen/protein_evidence.py), [`summarize_smooth_reference_evidence`](../src/bionemo/evo2_phage_gen/protein_evidence.py), and [`_add_smooth_reference_rewards`](../src/bionemo/evo2_phage_gen/reward.py) |
| `required_genes`           | [`summarize_required_gene_evidence`](../src/bionemo/evo2_phage_gen/protein_evidence.py) and [`_add_required_gene_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                               |
| `synteny`                  | [`smooth_protein_match_integrity`](../src/bionemo/evo2_phage_gen/protein_evidence.py), [`score_smooth_reference_architecture`](../src/bionemo/evo2_phage_gen/protein_evidence.py), and [`_add_smooth_reference_rewards`](../src/bionemo/evo2_phage_gen/reward.py) |
| `gene_a_origin`            | [`score_gene_a_origin`](../src/bionemo/evo2_phage_gen/protein_evidence.py) and [`_add_smooth_reference_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                         |
| `average_protein_identity` | [`summarize_full_length_aai`](../src/bionemo/evo2_phage_gen/protein_evidence.py) and [`_add_average_protein_identity_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                           |
| `mmseqs_cluster_diversity` | [`add_mmseqs_cluster_diversity_rewards`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                                                                                                 |
| `safety_amr`               | [`run_amrfinder_batch`](../src/bionemo/evo2_phage_gen/sequence_safety_adapters.py) and [`sequence_safety_reward_fields`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                 |
| `safety_toxin`             | [`run_toxin_batch`](../src/bionemo/evo2_phage_gen/sequence_safety_adapters.py) and [`sequence_safety_reward_fields`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                     |
| `safety_lysogeny`          | [`run_phrogs_batch`](../src/bionemo/evo2_phage_gen/sequence_safety_adapters.py) and [`sequence_safety_reward_fields`](../src/bionemo/evo2_phage_gen/reward.py)                                                                                                    |

### How rewards, gates, and selection differ

- **RL objectives** are the rows below. Their `[0, 1]` values shape training; full credit is a target,
  not an automatic final acceptance boundary.
- **Checkpoint selection** uses
  `binary_safety_qualified_full_qc_cluster_deduplicated_rate`. It requires exact safety `PASS`, full
  credit on the binary-core rewards—including the 5,359–5,391-nt length band—then the independent
  external hard-pass flags, and counts one representative per online 99%-identity cluster. Smooth
  synteny, tropism, A-origin, required-gene, and AAI targets are shaping terms rather than implicit
  hard gates.
- **Final per-genome QC** uses exact safety `PASS` plus the Arc target-profile waterfall: A/C/G/T
  only; length 5,306–5,493 nt; GC 30–65%; homopolymer ≤10; at least seven distinct PHROG families
  with ≥0.75 query and target coverage; a PhiX G hit at 60–100% identity with ≥0.95 query and target
  coverage; mean PHROG identity 0–95% over coverage-qualified hits; all 10 callable required slots
  intact at the 0.75 coverage threshold; and complete one-to-one circular synteny without excess
  homolog copies. DUST ≤0.9 is an online/checkpoint condition, not a repeated Arc final gate.
- **Set-level selection** clusters final passers at 99% identity and retains representatives. The
  rollout-relative diversity reward is therefore not interpreted as an intrinsic genome PASS/FAIL
  property.

This PhiX replication has no biological property deliberately targeted outside the viable lineage;
viable PhiX/Sinsheimervirus controls are therefore the default naturalness prior. A future directional
goal may leave its natural distribution, but its gate should be validated against known positives and
negatives while independent viability and safety gates remain. A proposed final gate can first be
replayed on saved per-candidate measurements; a new RL attempt is needed only if the change is adopted
in the online reward path.

### Sequence feasibility

| Objective (reward column)                    | Zero credit                                                                                                            | Full credit                                                                                                                                | Partial credit and rationale                                                                                                                                                                                                                                                                                                            |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `valid_nt_chars` (`reward_valid_nt_chars`)   | Any emitted character outside A/C/G/T.                                                                                 | No non-ACGT character is present.                                                                                                          | Binary. The raw helper regards an empty string as having no invalid character, but empty output fails length and aggregate nucleotide gates and cannot receive the safety-qualified GDPO objective. This prevents malformed sequence text from satisfying downstream biological tools.                                                  |
| `genome_length` (`reward_genome_length`)     | Length at or below 5,305 nt, or at or above 5,494 nt.                                                                  | 5,359–5,391 nt inclusive.                                                                                                                  | Ramp linearly from 0 to 1 over 5,305–5,359 nt and from 1 to 0 over 5,391–5,494 nt. Exact FASTA lengths for 155 same-capsid Sinsheimervirus genomes span 5,339–5,388 nt (p5 5,359; median/reference 5,386); INPHARED's rounded kilobase field is not used. PhiX has no terminal repeat, so deposited and packaged lengths are identical. |
| `gc_content` (`reward_gc_content`)           | Mathematically at or below −5% or at or above 100%; only the 100% endpoint is physically attainable.                   | 30–65% inclusive.                                                                                                                          | Below 30%: `max(0, 1 - (30-GC)/35)`; above 65%: `max(0, 1 - (GC-65)/35)`. Thus 0% GC still scores 1/7, while 100% scores 0. The broad band rejects extreme composition while leaving room around the PhiX reference.                                                                                                                    |
| `nt_homopolymer` (`reward_nt_homopolymer`)   | No finite positive run length reaches exactly zero; an ineligible row is subsequently zeroed by the exact-safety mask. | Maximum nucleotide run *H* ≤ 10 bases.                                                                                                     | For *H* > 10, score `10/H`, decreasing asymptotically toward zero. Long homopolymers are discouraged because they are low-complexity and can complicate synthesis and sequencing.                                                                                                                                                       |
| `dustmask_end` (`reward_dustmask_end`)       | No valid masked fraction reaches zero; an ineligible row is subsequently zeroed by the exact-safety mask.              | Maximum DUST-masked fraction *F* over either terminal 200-nt window ≤ 0.9.                                                                 | For 0.9 < *F* ≤ 1, score `0.9/F` (0.9–1.0). A failed external DUST command fails the scoring batch. The online nucleotide-pass objective supplies the binary cutoff; the final Arc target profile does not repeat this gate. DUST uses the approach described by [Morgulis et al.](https://doi.org/10.1089/cmb.2006.13.1028).           |
| `nucleotide_pass` (`reward_nucleotide_pass`) | Any component gate fails.                                                                                              | All characters are A/C/G/T, length is 5,306–5,493 nt, GC is 30–65%, maximum homopolymer is ≤10, and both terminal DUST fractions are ≤0.9. | Binary online conjunction. Its non-DUST thresholds are also applied by final Arc QC; the DUST distinction is described above.                                                                                                                                                                                                           |

### Protein evidence, architecture, and diversity

| Objective (reward column)                                               | Zero credit                                                                                                                                                               | Full credit                                                                                                                                                | Partial credit and rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `protein_hit_count` (`reward_external_protein_hit_count`)               | No measured PHROGs hit, missing alignment lengths, or missing output from an otherwise completed Arc run.                                                                 | Effective unique-family coverage ≥7; the hard gate separately requires at least 7 unique families meeting the calibrated coverage threshold.               | For each PHROGs target family, retain the best `min(query_coverage,target_coverage)` and sum across unique families; reward `min(sum/7,1)`. Presence credit is identity-independent. The PHROG-consensus threshold is 0.75, calibrated because consensus lengths differ systematically from member proteins; the exact PhiX spike gate remains 0.95. Duplicate ORFs cannot inflate family evidence, and fragments cannot pass the hard gate.                                                                                                                                              |
| `tropism` (`reward_external_tropism`)                                   | No significant called-ORF match to PhiX G, or raw integrity ≤0.001.                                                                                                       | E-value ≤1e-5, identity ≥95%, and both reference and candidate-ORF coverage ≥0.99.                                                                         | For E-values from 1 to 1e-5, significance ramps linearly in `-log10(E)`; multiply it by `(min(identity/0.95,1) × min(reference_coverage/0.99,1) × min(candidate_coverage/0.99,1))^1.5`, then rescale values above the 0.001 shuffled-null floor to start at 0.01. The independent hard gate remains identity ≥60% and both coverages ≥0.95, so low or partial evidence can guide RL without passing final QC. The G proxy follows the [PhiX design workflow](https://www.science.org/doi/10.1126/science.aec2657).                                                                        |
| `required_genes` (`reward_external_required_genes`)                     | Missing metrics, missing alignment lengths, an empty required-label definition, or zero evidence.                                                                         | All 10 ORFipy-callable gene-copy slots have full reciprocal-coverage evidence.                                                                             | Score `(coverage_sum/total) × min(total/10,1)`. The hard gate is less strict: all 10 slots need ≥0.75 query and target coverage. Nine functional labels are used, with two one-to-one slots for head morphogenesis (B/D); within that repeated label, one ORF or PHROG target cannot satisfy both copies. A\* remains outside this ORF-caller-based term, while the architecture gate distinguishes callable reference loci and penalizes extra homolog copies.                                                                                                                           |
| `synteny` (`reward_external_synteny`)                                   | No significant called-ORF match exceeds the 0.001 shuffled-null integrity floor.                                                                                          | All callable reference loci have one-to-one ORF matches at ≥90% identity and ≥0.95 coverage on both sides, in circular order, with no excess homolog mass. | Each edge uses the same significance ramp and convex integrity formula as tropism, with 90% identity and 0.95 reciprocal-coverage full-credit targets. Maximum-weight one-to-one assignment gives content *C*; circular order-preserving assignment gives *O*; excess homolog mass gives *D*. Score `clip(0.25C + 0.75O - 0.75D, 0, 1)` with a fixed reference-locus denominator. Properly ordered weak homologs outscore the same scrambled homologs, while deletion, fusion, and duplication cannot improve the score. Hard LoVis clustering remains a separate final-pass measurement. |
| `gene_a_origin` (`reward_gene_a_origin`)                                | No assigned A ORF with a credible functional origin near its expected in-frame offset, or no A match evidence.                                                            | A has full match integrity and one exact functional 28-nt site at offset 345 in the assigned A ORF.                                                        | Within ±30 nt in the same frame, score the 10-nt recognition region, positions 4–7 around the nick quadratically, and the following 18-nt binding region; multiply by a linear offset score, A-match integrity, and `1/strong_site_count`. Positions 29–30 of the 30-nt reference motif do not affect this score because [cleavage experiments](https://www.sciencedirect.com/science/article/pii/S0021925820820791) found them dispensable. This is an online shaping objective and diagnostic pass, not a new final hard gate.                                                          |
| `average_protein_identity` (`reward_external_average_protein_identity`) | Missing identity or alignment lengths, no coverage-qualified family hits, or missing output from an otherwise completed Arc run.                                          | PHROG-family mean identity ≤95% with at least 10 unique coverage-qualified families.                                                                       | Compute mean identity from one best hit per target family after the 0.75 PHROG-consensus coverage gate, then score `novelty × min(family_count/10,1)`. Novelty is 1 through 95% and `max(0.25, (100-AAI)/5)` above 95%. Final Arc QC requires mean identity ≤95% after the upstream seven-family gate; it does not require the reward's 10-family full-credit state. This is family-level PHROG identity, not exact-reference AAI.                                                                                                                                                        |
| `mmseqs_cluster_diversity` (`reward_mmseqs_cluster_diversity`)          | The genome fails the valid-character, hard-length, GC, or homopolymer prefilter, or is missing from MMseqs output. No finite cluster size otherwise reaches exactly zero. | A singleton within its prompt group.                                                                                                                       | With 99% minimum sequence identity, a member of a cluster of size *N* scores `1/N`. DUST is not part of this prefilter, and a failed MMseqs command fails the scoring batch. This rewards within-batch diversity but is not a per-genome hard gate; final passers are clustered separately for set-level selection. [MMseqs2](https://doi.org/10.1038/nbt.3988) provides the clustering implementation.                                                                                                                                                                                   |

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
