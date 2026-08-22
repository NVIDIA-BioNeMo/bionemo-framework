# Evo 2 Phage Design

This recipe fine-tunes Evo 2 on phage genomes, runs GDPO, generates complete candidate genomes,
and applies sequence-safety and design screens. Results are computational candidates, not evidence
of bootability, therapeutic fitness, or wet-lab safety.

## Latest completed PhiX174 result

| Stage                                   |                                                   Result |
| --------------------------------------- | -------------------------------------------------------: |
| SFT split                               |                 14,266 train / 100 validation / 100 test |
| Selected SFT checkpoint                 | step 5,600; validation loss 0.750670; test loss 0.798180 |
| Selected GDPO checkpoint                |                                          step 430 of 500 |
| Target-profile rollout                  |                                        610/1,000 (61.0%) |
| Full Arc diagnostic, including filter 7 |                                          22/1,000 (2.2%) |

The previous retrospective safety scan found 997 PASS, no biological FAIL, one INDETERMINATE
short sequence, and two nucleotide-QC rejects. These numbers describe the latest completed run and
should be refreshed after the current end-to-end run finishes. See the
[example README](examples/README.md) for the workflow and
[case-study notes](skills/bionemo-phage-design/references/case-study-results.md) for provenance.

## Quick start

Run from `recipes/evo2_phage_gen`. The publication-style reproduction uses 7B-base:

```bash
./.ci_build.sh
./examples/phix174_8xh100.sh --model-variant 7b-base --result-root "$PWD/results/phix174-8xh100"
```

For a fresh experiment, the trained-further 7B-1M model is generally preferred:

```bash
./examples/phix174_8xh100.sh --model-variant 7b-1m --result-root "$PWD/results/phix174-7b-1m"
```

Do not change model families within an existing result root. Resume an interrupted base run with:

```bash
./examples/phix174_8xh100.sh --resume-from 40 --model-variant 7b-base --result-root "$PWD/results/phix174-8xh100"
```

Use tmux or a scheduler for long runs. `NUM_GPUS` defaults to 8, `NUM_CPUS` to `nproc`, and
`SFT_TENSOR_PARALLEL_SIZE` may adapt a measured smaller topology.
The [8×H100 example](examples/README.md) documents dry runs, preparation-only mode, stage markers,
sampling overrides, outputs, and all RL objectives.

## Run with an agent

From the repository root, the portable skill locates a compatible checkout:

```bash
codex 'Use $bionemo-phage-generation to begin interactive planning for the PhiX174 GDPO case study.'
```

From this recipe directory, the local controller can inspect or execute a run:

```bash
codex 'Use $bionemo-phage-design in interactive case-study-replication mode. Inspect existing results and propose the plan before launching jobs.'
```

The recipe-local skill bundle is also exposed through [.codex-plugin/plugin.json](.codex-plugin/plugin.json).

## Useful commands

```bash
# Inspect the complete command plan without downloads or GPUs.
./examples/phix174_8xh100.sh --dry-run --result-root /tmp/phix174-plan

# Download and preprocess the publication-era SFT inputs directly.
evo2_phage_download_sft_data --include-raw
preprocess_evo2 --config configs/sft_microviridae_preprocess.yaml
```

If an entrypoint is missing, rerun `./.ci_build.sh` and source `.ci_test_env.sh`. For GPU memory or
topology changes, preserve whole-genome context and effective batch size; see the
[compute guidance](skills/bionemo-phage-design-adapt-execution/references/compute-guidance.md).

## References

- [PhiX174 phage-design publication](https://www.science.org/doi/10.1126/science.aec2657)
- [Evo 2 publication](https://doi.org/10.1038/s41586-026-10176-5)
- [Public Microviridae SFT checkpoint](https://huggingface.co/evo-design/evo-2-7b-8k-microviridae)
- [Evo 2 model and checkpoint notes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/main/recipes/evo2_megatron/README.md)

## Acknowledgements

Thanks to Samuel King, Jessica Sacher, Jan Zheng, Avery Noonan, Michael Poon and colleagues at
Tabula Bio, and Eric Bastien and Nick Conley at Locus Biosciences for discussions and feedback that
shaped the recipe and its safety controls.
