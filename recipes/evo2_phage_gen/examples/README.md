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
  --result-root "$PWD/results/phix174-8xh100"
```

The build creates the virtual env used by `.ci_test_env.sh`, which the example sources internally.
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
  --result-root "$PWD/results/phix174-8xh100"
```

If an unfinished stage is interrupted, rerun the original command with the same result root. The
script reuses completed stages and cached or partial downloads, so the run directory need not be
deleted. Only one invocation may use a result root at a time. If AMRFinder fails, its captured
output is retained as `amrfinder/amrfinder.log` below that scan directory.

SFT training, RL training, and final generation also write narrower completion markers:
`20-sft.done`, `40-rl.done`, and `50-rollout.done`. These let a resumed run skip an accepted
long-running result while still performing its downstream selection, evaluation, screening, and
reporting. For example, after inspecting checkpoints from an interrupted SFT run:

```bash
touch results/phix174-8xh100/stages/20-sft.done
./examples/phix174_8xh100.sh --resume-from 20 \
  --result-root "$PWD/results/phix174-8xh100"
```

Create a narrow marker manually only when that substage's outputs are complete enough to accept;
ordinary successful runs create it automatically.

The six dependency-ordered stages are input/control preparation, SFT safety and splitting, SFT
training and selection, sampling calibration, GDPO pilot/training, and final generation, SFT
likelihood scoring, and screening.
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
more follow-up even if that requires several epochs. Before the generated GDPO pilot, the known
The SFT command keeps the best three validation-loss checkpoints plus the latest resume point.
`validation_metrics.json` stores all scalar validation measurements, and `checkpoint_metrics.json`
stores each save-time metric assignment and a `best_checkpoint` relative directory pointer. The
example requires the raw `lm loss` key; the corresponding TensorBoard tag is `lm loss validation`.
PhiX174 reference runs through the exact configured RL environment and every enabled external,
diversity, and safety measurement must report support. The pilot then checks training and checkpoint
behavior; rewards stay in `[0, 1]`, baseline/chance means `0`, missing or failed measurements cannot
look favorable, and the selected SFT checkpoint remains the KL anchor.

The final scoring pass uses the selected pre-RL SFT checkpoint with its `+~` conditioning prefix.
`rollout/sft-likelihood/ranked-designs.csv` contains total and mean per-nucleotide log probability
for all 1,000 designs. `rollout/final-designs.json` joins those values to each design's safety state,
target-profile result, and accepted rank; `accepted_candidates.fasta` follows the same ranking when
it is usable. The report also computes Spearman correlation between length and the normalized score.
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
Arc results, accepted candidates, and the final `SUMMARY.md` remain beside their stage outputs. A
boundary-best checkpoint, changed safety control, unsupported calibration choice, or unhealthy RL
objective is an intentional review stop; routine setup and execution need no agent intervention.
