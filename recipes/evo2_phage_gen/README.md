# Evo 2 Phage Design

This recipe fine-tunes Evo 2 on phage genomes, runs
[GDPO](https://arxiv.org/abs/2601.05242)—an NVIDIA-developed method for stable multi-reward policy
optimization—generates complete candidate genomes, and applies sequence-safety and design screens.
A key extension beyond the
[original published workflow](https://www.science.org/doi/10.1126/science.aec2657) is the
reinforcement learning (RL) stage, which optimizes generation toward user-defined computational
design criteria rather than relying only on post-generation filtering.

The included PhiX174 example follows Samuel King's recommended set of filters (1–6, 8, and 9) from
[Figure 2H](https://www.science.org/doi/10.1126/science.aec2657#F2). These become RL objectives with
partial credit for candidates that approach each threshold. See the
[PhiX174 GDPO score definitions](examples/README.md#current-phix174-gdpo-score-definitions) for the
objective definitions and partial-credit rules. The publication reported 15 filter passers among
110,000 generated sequences. An earlier completed end-to-end GDPO run produced 610 target-profile
passes among 1,000 designs, while the latest run retained 511 post-QC accepted representatives among
1,000 designs. These results are descriptive rather than a controlled enrichment comparison because
the runs used different checkpoints, screening pipelines, and selection procedures.

| Run                                                  | Reported computational outcome                     |
| :--------------------------------------------------- | :------------------------------------------------- |
| GDPO replication using Arc's Microviridae checkpoint | 358/1,000 target-profile passes (35.8%)            |
| End-to-end GDPO replication 1                        | 610/1,000 target-profile passes (61.0%)            |
| End-to-end GDPO replication 2                        | 511/1,000 post-QC accepted representatives (51.1%) |
| Published workflow without RL                        | 15/110,000 filter passers (approximately 0.014%)   |

The recipe includes agent skills led by the top-level `bionemo-phage-design` skill. They can adapt
the example launcher to the available GPU environment, help plan and implement related design
tasks, and monitor long-running jobs.

## Quick start

See the [example README](examples/README.md) for additional launch options and operational details.

The end-to-end run takes approximately 4 days on a server with 8 H100 GPUs, and uses 1.5TB of storage.

Run the following from `recipes/evo2_phage_gen` to reproduce the `7b-base` end-to-end configuration
summarized above:

```bash
./.ci_build.sh
./examples/phix174_8xh100.sh \
  --model-variant 7b-base  \
  --sampling-selection "examples/default-sampling-selection.yaml" \
  --result-root "$PWD/results/phix174-8xh100"
```

Like the Evo 2 recipe, `.ci_build.sh` creates an editable Python 3.12 virtual
environment with `--system-site-packages` and reuses the NVIDIA PyTorch
container's Torch, TorchVision, Triton, Transformer Engine, CUDA, and cuDNN
stack. The two recipes share the same Evo 2 runtime pins and native-extension
fallback; this recipe then adds NeMo-RL and phage workflow dependencies. The
example launcher sources `.ci_test_env.sh` itself; source it explicitly before
running individual recipe commands by hand.

Use tmux or a scheduler for long runs. `NUM_GPUS` defaults to 8, `NUM_CPUS` to `nproc`, and
`SFT_TENSOR_PARALLEL_SIZE` may adapt a measured smaller topology.
The [8×H100 example](examples/README.md) documents dry runs, preparation-only mode, stage markers,
sampling overrides, outputs, and all RL objectives.

## Run with an agent

From the repository root, ask your preferred coding agent to adapt and run the workflow:

```text
Use $bionemo-phage-generation to adapt the PhiX174 example for this GB300 node, run and monitor it, and use the 7B-1M checkpoint instead of 7B-8K.
```

## References

- [PhiX174 phage-design publication](https://www.science.org/doi/10.1126/science.aec2657)
- [Evo 2 publication](https://doi.org/10.1038/s41586-026-10176-5)
- [Public Microviridae SFT checkpoint](https://huggingface.co/evo-design/evo-2-7b-8k-microviridae)
- [Evo 2 model and checkpoint notes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/main/recipes/evo2_megatron/README.md)

## Acknowledgements

Thanks to Samuel King, Jessica Sacher, Jan Zheng, Avery Noonan, Michael Poon, and colleagues at
Tabula Bio, and to Eric Bastien and Nick Conley at Locus Biosciences, for discussions, guidance, and
feedback that shaped the recipe and its safety controls.
