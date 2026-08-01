---
name: bionemo-recipes-acceleration
description: >-
  Accelerate existing PyTorch/HuggingFace model training code with NVIDIA Transformer Engine,
  following the patterns proven in BioNeMo Recipes: FP8/MXFP8/NVFP4 quantization recipes, fused
  TransformerLayer, THD sequence packing, and quantized_model_init. Measures precision choice with
  TE's GEMM benchmark and validates the port with the BioNeMo BaseModelTest harness. Hard-stops
  with a report when no BioNeMo reference architecture matches. Do NOT use for genomics pipeline
  acceleration — use genomics-workflow-acceleration.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "torch>=2.4; transformer_engine[pytorch]>=2.0; CUDA GPU (Hopper or newer for FP8)"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, AskUserQuestion
metadata:
  version: "1.0.0"
  author: Zoey Zhang <zozhang@nvidia.com>
  domain: model-training
  tags:
    - transformer-engine
    - fp8
    - mxfp8
    - nvfp4
    - sequence-packing
    - bionemo
    - pytorch
    - acceleration
---

# BioNeMo Recipes acceleration

## Purpose

Port an external PyTorch or HuggingFace model codebase onto the Transformer Engine acceleration
patterns that are already proven and tested inside the BioNeMo Recipes repository. The skill
probes the user's GPU, benchmarks candidate precisions with TE's own GEMM tool, rewrites (or
configures) the model at the depth the code warrants, and proves the result correct with the
shared `BaseModelTest` CI harness. When no BioNeMo reference architecture is a close enough match,
the skill stops and explains why rather than producing an unvalidated port.

## When to Use This Skill

**Trigger on any of:**

- "add FP8/MXFP8/NVFP4 to my training script"
- "speed up my (protein/DNA/causal) LM with Transformer Engine"
- "add sequence packing / THD attention"
- "port this HuggingFace model to TE"
- "quantized_model_init", "FusedAdam", "te.TransformerLayer"
- "what precision should I use for training on H100/H200/B200"
- "accelerate my ESM2 / Llama / Mixtral fine-tuning"

**Do NOT trigger on:**

- Genomics pipeline (Nextflow, Snakemake, WDL) → use `genomics-workflow-acceleration`
- Diffusion, score-based, equivariant (SE3/E3), GNN, SSM/Mamba models → hard stop with report
- Megatron-LM training → direct user to `$BIONEMO_RECIPES/recipes/evo2_megatron/`
- Inference serving / vLLM → `$BIONEMO_RECIPES/recipes/vllm_inference/`

## Examples

**Transformer model — full port:**
> "Add FP8 training to my ESM2 fine-tuning script in `/workspace/my_esm2/`"

Output: branch `bionemo-accel/encoder-mlm` with FP8 configs, parity-checked, and `ACCELERATION_REPORT.md` at the repo root.

**Causal LM — THD packing added:**
> "Speed up my Llama fine-tuning loop with Transformer Engine and THD sequence packing"

Output: full TE port with THD attention, `DataCollatorWithFlattening`, Tier 1 + Tier 2 validation, `ACCELERATION_REPORT.md`.

**Out-of-scope — hard stop:**
> "Add FP8 to my SE(3)-equivariant GNN for protein structure prediction"

Output: `ACCELERATION_REPORT.md` identifying the architecture as equivariant/GNN, zero source files modified, reason for rejection named.

## Prerequisites

### Run the skill in your training environment

**This skill must be invoked in the same Python environment your model training script uses** —
the same virtualenv, conda env, or container. `scripts/probe_hardware.py` installs torch and
Transformer Engine into whichever `python` it is run with; if that is the wrong interpreter the
packages land in the wrong place and the port cannot import them.

To confirm you are in the right environment before proceeding:

```bash
which python          # or: conda info --envs
pip show torch        # should match the version your training script uses
```

### torch and transformer_engine

`scripts/probe_hardware.py` (Phase 2) checks for both packages and, if either is missing, prompts
to install them:

- **torch** — installed as `torch==2.9.0` if absent.
- **transformer_engine** — installed as `transformer-engine[pytorch]==2.9.0 --no-build-isolation`
  if absent. The `--no-build-isolation` flag is required because TE links against the installed
  torch and CUDA headers at build time.

Pass `--no-install` to skip the prompt and exit immediately instead.

The recommended container (which ships both at tested versions) is `nvcr.io/nvidia/pytorch:26.04-py3`.

### BioNeMo Recipes reference library

This skill reads `models/` and `recipes/` from a bionemo-recipes checkout as its reference
library. When running outside one, clone it first:

```bash
git clone https://github.com/NVIDIA-BioNeMo/bionemo-recipes.git
export BIONEMO_RECIPES=$PWD/bionemo-recipes
```

When running inside a bionemo-recipes checkout, set `$BIONEMO_RECIPES` to the repo root. Phase 0
resolves and verifies this variable before any other step.

## Guardrails

1. **Match before you modify.** Phase 1 gates everything. No reference architecture match → write
   report and stop. Do not partial-port.
2. **Never modify the BioNeMo Recipes repository** (`$BIONEMO_RECIPES`). It is a read-only
   reference. All edits land in the target codebase, on a new branch.
3. **Low precision ships disabled.** Generate FP8/FP4 configs with `enabled: false` — the user
   opts in with one line.
4. **Preserve the original.** The parity check needs the unported model as its baseline.
5. **Report honestly.** Every acceleration skipped, every test that was skipped rather than
   passed, and every caveat goes in `ACCELERATION_REPORT.md`.

## Composed pieces (read on demand — do not inline)

| Read when                                                           | File                                  |
| ------------------------------------------------------------------- | ------------------------------------- |
| Phase 1 — architecture matching and hard-stop decision              | `references/architecture-matching.md` |
| Phases 2–3 — hardware probe output, GEMM benchmark, recipe decision | `references/precision-selection.md`   |
| Phase 4 — rewrite depth, converter, autocast, quantized_model_init  | `references/te-conversion.md`         |
| Phase 4 — THD packing, cu_seqlens, DataCollatorWithFlattening       | `references/sequence-packing.md`      |
| Phase 5 — tiered validation, BaseModelTest hooks, CI reproduction   | `references/validation.md`            |
| Phase 2 — hardware and TE version probe                             | `scripts/probe_hardware.py`           |
| Phase 3 — GEMM benchmark wrapper                                    | `scripts/run_gemm_benchmark.py`       |
| Phase 5 — parity check scaffold (write to target repo)              | `assets/parity_check.py.tmpl`         |
| Phase 5 — BaseModelTest subclass scaffold (write to target repo)    | `assets/test_modeling_ported.py.tmpl` |
| Phase 5 — pytest conftest scaffold (write to target repo)           | `assets/conftest.py.tmpl`             |
| Phase 6 — acceleration report scaffold (write to target repo)       | `assets/ACCELERATION_REPORT.md.tmpl`  |

Load a reference file once when entering the relevant phase. Do not load all of them upfront.
The `assets/` files are code scaffolds written into the target repo during Phases 5–6, not loaded
for reading — open them only when generating the corresponding output file.

## Instructions

**Phase 0 — Inventory.** Resolve `$BIONEMO_RECIPES`. Create `.bionemo-accel/` in the target repo
and record: entry points, model definition files, optimizer, dataloader; framework (raw loop / HF
Trainer / Accelerate / Lightning / Megatron); whether TE is already imported; model dimensions
(`hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`,
`num_hidden_layers`, `vocab_size`); `torch.__version__`, `transformer_engine.__version__`. Write
`.bionemo-accel/inventory.json`. Add `.bionemo-accel/` to the target's `.gitignore`.

**Phase 1 — Architecture match, or hard stop.** Read `references/architecture-matching.md`. Score
the target against the four supported families (encoder/MLM, causal LM dense, MoE, genomics LM).
Require a high-confidence match on attention pattern, normalization, and MLP form. Write
`.bionemo-accel/match.json`; or write the hard-stop `ACCELERATION_REPORT.md` and exit.

**Phase 2 — Hardware and TE probe.** Run:

```bash
python $SKILL_DIR/scripts/probe_hardware.py -o .bionemo-accel/hardware.json
```

where `$SKILL_DIR` is the directory containing this `SKILL.md`. Read
`references/precision-selection.md` §Phase 2 for sm_120/sm_80 caveats and TE version guidance.

**Phase 3 — Precision selection.** Run:

```bash
python $SKILL_DIR/scripts/run_gemm_benchmark.py \
    --inventory .bionemo-accel/inventory.json \
    --hardware  .bionemo-accel/hardware.json \
    -o .bionemo-accel/gemm
```

Read `references/precision-selection.md` §Phase 3 for interpretation — GEMM speedup is an upper
bound on end-to-end speedup; autocast vs pre-quantize gap is quantization overhead; a ≈1.0× result
means suspected kernel fallback, not no benefit.

**Phase 4 — Rewrite.** Read `references/te-conversion.md` and `references/sequence-packing.md`.
Create branch `bionemo-accel/<family>` in the target repo. Apply Depth A (already TE), B (full
port + converter), or C (kernel swaps only) at the depth warranted by the code.

**Phase 5 — Validate.** Read `references/validation.md`. Generate `parity_check.py` from
`assets/parity_check.py.tmpl` (Tier 1, always); generate `tests/test_modeling_ported.py` and
`tests/conftest.py` from the corresponding templates (Tier 2, when Depth B produced a converter);
reproduce CI (Tier 3, when containerised). Fix the port if any tier fails.

**Phase 6 — Report.** Generate `ACCELERATION_REPORT.md` from `assets/ACCELERATION_REPORT.md.tmpl`
and write it to the target repo root. Lead with the measured numbers and the one-line config
change to enable FP8.

## Handoff contracts

Each phase reads its predecessor's JSON artifact. Do not skip phases or produce a downstream
artifact without the upstream ones in place.

| Source  | Artifact                              | Consumed by                 |
| ------- | ------------------------------------- | --------------------------- |
| Phase 0 | `.bionemo-accel/inventory.json`       | Phases 1, 3                 |
| Phase 1 | `.bionemo-accel/match.json` (or exit) | Phases 2, 4                 |
| Phase 2 | `.bionemo-accel/hardware.json`        | Phase 3                     |
| Phase 3 | `.bionemo-accel/gemm/summary.json`    | Phase 6                     |
| Phase 3 | `.bionemo-accel/precision.json`       | Phase 4 (config generation) |
| Phase 5 | tier results                          | Phase 6                     |

On an interrupted run, Phases 0–2 may reuse existing artifacts if the environment has not changed.
Phases 3–6 always re-run.

## Limitations

- **Single-L4 CI blind spot.** PR CI in this repo runs on one L4. Tests gated on
  `requires_multi_gpu` or `requires_datacenter_hardware` (H100/H200/B100/B200/B300) are skipped,
  not passed. MXFP8 and NVFP4 paths need Blackwell. A green CI run is not evidence those work.
- **sm_120 (RTX 50xx).** BioNeMo bounds MXFP8/NVFP4 at compute capability < 12.0; fused THD
  attention is xfailed. Sequence packing still works via flash_attn.
- **sm_80 (A100).** Fused THD attention is xfailed; packing works via flash_attn.
- **mFSDP + quantized_model_init.** Xfailed (BIONEMO-3012). Do not enable with megatron-fsdp.
- **MoE.** Only straightforward top-k routing matches the Mixtral reference. Exotic routing is a
  hard stop.

## Validation

Full protocol in `references/validation.md`. Summary:

- **Tier 1 — always** — forward logits + loss parity, backward, FP8 recipe sweep, BSHD-vs-THD.
- **Tier 2 — when converter generated** — `BaseModelTest` harness (~30 inherited tests: golden
  values, conversion round trips, init, FP8 recipe sweep).
- **Tier 3 — when containerised** — CI reproduction in `nvcr.io/nvidia/pytorch:26.04-py3`. State
  what was skipped and why.

## Responsible use

- Never modify `$BIONEMO_RECIPES` — it is a read-only reference.
- Low precision ships disabled; the user enables it deliberately.
- Report every skipped test, loosened tolerance, and limitation in `ACCELERATION_REPORT.md`.

## Available Scripts

| Script | Purpose | Key Arguments |
| --- | --- | --- |
| `scripts/probe_hardware.py` | Probe GPU and TE install; detect per-recipe FP8/MXFP8/NVFP4 support; optionally install missing packages | `-o <path>` output JSON path; `--no-install` skip install prompt |
| `scripts/run_gemm_benchmark.py` | Run TE GEMM benchmark in autocast and pre-quantize modes; write speedup plots and summary JSON | `--inventory <json>`; `--hardware <json>`; `-o <dir>`; `--shapes <MxKxN>`; `--allow-clone`; `--verbose-kernels` |

Invoke scripts from the pipeline phases using bash or `run_script()`:

```python
# Phase 2 — hardware probe
run_script("scripts/probe_hardware.py", args=["-o", ".bionemo-accel/hardware.json"])

# Phase 3 — GEMM benchmark
run_script("scripts/run_gemm_benchmark.py", args=[
    "--inventory", ".bionemo-accel/inventory.json",
    "--hardware",  ".bionemo-accel/hardware.json",
    "-o",          ".bionemo-accel/gemm",
])
```

## Templates

The `assets/` directory holds code scaffolds that are filled in and written into the **target
repo** during Phases 5–6. They are not loaded for reading — open each only when generating its
corresponding output file.

- `assets/parity_check.py.tmpl` → `<target>/parity_check.py` (Tier 1 validation, Phase 5)
- `assets/test_modeling_ported.py.tmpl` → `<target>/tests/test_modeling_ported.py` (Tier 2, Phase 5)
- `assets/conftest.py.tmpl` → `<target>/tests/conftest.py` (Tier 2, Phase 5)
- `assets/ACCELERATION_REPORT.md.tmpl` → `<target>/ACCELERATION_REPORT.md` (Phase 6)

## Inputs

**Required**

| Input | Source | Description |
| --- | --- | --- |
| Target codebase path | User prompt or current working directory | Repo or folder containing the model to accelerate |
| `$BIONEMO_RECIPES` | Environment variable | Path to a bionemo-recipes checkout (read-only reference) |

**Optional (passed to scripts)**

| Input | Flag | Default | Description |
| --- | --- | --- | --- |
| Skip install prompt | `--no-install` (probe_hardware) | off | Exit 1 immediately if torch or TE missing |
| Manual GEMM shapes | `--shapes MxKxN` (run_gemm_benchmark) | derived from inventory | Override model-config-based shapes |
| TE source tree | `--te-source <path>` or `$TE_SOURCE_DIR` (run_gemm_benchmark) | auto-detected | Explicit path to a Transformer Engine checkout |
| Auto-clone TE | `--allow-clone` (run_gemm_benchmark) | off | Shallow-clone TE source if benchmark script not found locally |
| Kernel dispatch log | `--verbose-kernels` (run_gemm_benchmark) | off | Set `NVTE_LOG_LEVEL=1` to confirm kernel dispatch |

## Output Format

All artifacts are written into the **target repo**, never into `$BIONEMO_RECIPES`.

| Artifact | Phase | Written | Description |
| --- | --- | --- | --- |
| `.bionemo-accel/inventory.json` | 0 | Always | Model dimensions, entry points, framework, TE import status |
| `.bionemo-accel/match.json` | 1 | On match | Architecture family, confidence score, matched reference |
| `.bionemo-accel/hardware.json` | 2 | Always | GPU, compute capability, per-recipe TE support flags |
| `.bionemo-accel/gemm/summary.json` | 3 | Always | GEMM speedup for autocast and pre-quantize modes, interpretation reminders |
| `.bionemo-accel/precision.json` | 3 | Always | Selected recipe and rationale |
| `parity_check.py` | 5 | Always | Tier 1 forward-pass parity check against the original model |
| `tests/test_modeling_ported.py` | 5 | Depth B only | BaseModelTest harness subclass for the ported model |
| `tests/conftest.py` | 5 | Depth B only | pytest plugin registration (`pytest_plugins = ["tests.common.fixtures"]`) |
| `ACCELERATION_REPORT.md` | 6 | Always | Final report: measured speedups, one-line FP8 enable config, limitations |
| `ACCELERATION_REPORT.md` (hard stop) | 1 | On no match | Architecture rejection reason; no other files written or modified |

**Branch:** `bionemo-accel/<family>` is created in the target repo on a successful port (e.g. `bionemo-accel/encoder-mlm`, `bionemo-accel/causal-lm-dense`).

## Troubleshooting

| Error / Symptom | Cause | Solution |
| --- | --- | --- |
| Hard stop at Phase 1 | Architecture is not one of the four supported families | Read `references/architecture-matching.md`; no partial port is attempted — report names the disqualifying axis |
| `Could not find benchmarks/gemm/benchmark_gemm.py` | TE pip wheel does not include source; no NGC container or local checkout found | Pass `--te-source <path>`, set `TE_SOURCE_DIR`, or add `--allow-clone` |
| GEMM speedup near 1.0× | Silent kernel fallback to a lower-precision kernel | Re-run with `--verbose-kernels`; confirm expected dispatch in `NVTE_LOG_LEVEL=1` output before concluding there is no benefit |
| `torch not importable` after install | Script ran in the wrong Python environment | Verify `which python` inside your training venv/conda env; re-run `probe_hardware.py` in that environment |
| `transformer_engine.pytorch.autocast` missing | TE version predates the API BioNeMo recipes use | Upgrade to `transformer-engine[pytorch]>=2.0` or use `nvcr.io/nvidia/pytorch:26.04-py3` |
| Parity check fails after port | Weight conversion bug or `te.autocast` scope too narrow | Compare QKV packing against `$BIONEMO_RECIPES/models/esm2/convert.py`; verify `te.autocast` wraps the full forward pass |
| `quantized_model_init` + megatron-fsdp fails | BIONEMO-3012 (upstream xfail) | Do not combine `quantized_model_init` with `megatron-fsdp` until resolved; note in report |
