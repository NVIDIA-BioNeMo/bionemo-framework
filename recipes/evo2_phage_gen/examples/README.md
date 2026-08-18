# PhiX174 whole-genome example on 8×H100

[`phix174_8xh100.sh`](phix174_8xh100.sh) is the agent-free realization of the current recipe. It
prepares public inputs and tools, trains SFT and GDPO, generates 1,000 complete genomes, scores
every design with the selected SFT, and applies the current sequence-safety and PhiX174 design
screens. These are computational candidates, not
evidence of bootability, therapeutic fitness, or wet-lab safety.

From `recipes/evo2_phage_gen` on a node with eight H100 80 GB GPUs:

```bash
./.ci_build.sh

# Recommended when a durable batch scheduler is not keeping the process alive.
tmux new -s phix174-e2e

./examples/phix174_8xh100.sh \
  --result-root "$PWD/results/phix174-8xh100"
```

The build creates the virtual env used by `.ci_test_env.sh`, which the example sources internally.
The script verifies that it is running on eight H100 GPUs; restricted agent sandboxes often hide
`nvidia-smi`, so launch it on the allocated compute node. Use a scratch clone or worktree when
a run needs code or config changes.

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

The six dependency-ordered stages are input/control preparation, SFT safety and splitting, SFT
training and selection, sampling calibration, GDPO pilot/training, and final generation, SFT
likelihood scoring, and screening.
Calibration uses the eight GPUs in parallel; SFT and GDPO use their configured distributed layouts.
Long commands remain supervised by the top-level process, with a meaningful liveness update every
ten minutes. Run that process in tmux or a scheduler so it survives a disconnected chat or shell.

Databases are refreshed rather than pinned to the historical run. The script records the current
installed state and reruns positive, review, and negative controls; changed behavior stops for review
instead of silently selecting an older database. Required detector failures remain INDETERMINATE.
The previous SFT-corpus audit observed 14,465 PASS and one lysogeny-review INDETERMINATE among
14,466 inputs; every new run records and uses its own result.

SFT selection uses optimizer-step validation. A larger global batch can complete an epoch before
enough optimizer updates exist to show overfitting, so a best value at the run boundary stops for
more follow-up even if that requires several epochs. The GDPO pilot exercises the complete reward
path before the full run; rewards stay in `[0, 1]`, baseline/chance means `0`, missing or failed
measurements cannot look favorable, and the selected SFT checkpoint remains the KL anchor.

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
