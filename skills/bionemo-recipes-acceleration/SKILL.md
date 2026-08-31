---
name: bionemo-recipes-acceleration
description: >-
  Accelerate existing PyTorch/HuggingFace model training code with NVIDIA Transformer Engine,
  following the patterns proven in BioNeMo Recipes: FP8/MXFP8/NVFP4 quantization recipes, fused
  TransformerLayer, THD sequence packing, and quantized_model_init. Measures precision choice with
  TE's GEMM benchmark and validates the port with the BioNeMo BaseModelTest harness. Hard-stops
  with a report only for architectures with no TE analogue — diffusion, GNN/equivariant, and
  state-space models. Do NOT use for genomics pipeline acceleration — use
  genomics-workflow-acceleration.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "torch>=2.4; transformer_engine[pytorch]>=2.0; CUDA GPU (Hopper or newer for FP8)"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, AskUserQuestion
metadata:
  version: "1.1.0"
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

## Failure classes

Two failure classes appear in `ACCELERATION_REPORT.md` and in `.bionemo-accel/` artifacts. Use
the correct one; conflating them produces misleading reports and blocks retries.

| Class   | Meaning                                                                                                                                                                                                                                   | Retryable | Examples                                                                                                                                              |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ENV_`  | Declared dependencies exist but could not be installed or imported in this environment. The architecture has not been judged. The run cannot proceed until the environment is fixed, but a clean environment may succeed.                 | Yes       | `torch` or `transformer_engine` not importable; `probe_hardware.py` exits 1 due to a missing package; `pip install` failed; wrong Python interpreter. |
| `ARCH_` | The target architecture has no TE analogue, or the model definition cannot be located or executed for a reason intrinsic to the target (no weights, no tokenizer, no sample input). No amount of environment fixing will unblock the run. | No        | Diffusion/score-based, GNN/equivariant, state-space model; causal vs bidirectional mismatch; model class defined dynamically at runtime.              |

**Critical distinction:** "no forward pass can be run" is `ARCH_` only when the blocker is
intrinsic to the target (missing weights, no tokenizer, purely-generated code). If the blocker is
an uninstallable dependency, emit `ENV_` and stop — do not treat it as an architectural judgment.

## Guardrails

1. **Match before you modify.** Phase 1 decides the reference and the depth before any edit. An
   architecture with no TE analogue → write the report and stop. A weak-but-workable match is not a
   stop: port at the depth the block supports (down to Depth C kernel swaps) and state the
   limitation. What is never acceptable is a port presented as validated when it is not.
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

Load a reference file once when entering the relevant phase. Do not load all of them upfront.

## Instructions

**Phase 0 — Inventory.** Resolve `$BIONEMO_RECIPES`. Create `.bionemo-accel/` in the target repo
and record: entry points, model definition files, optimizer, dataloader; framework (raw loop / HF
Trainer / Accelerate / Lightning / Megatron); whether TE is already imported; model dimensions
(`hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`,
`num_hidden_layers`, `vocab_size`); `torch.__version__`, `transformer_engine.__version__`. Write
`.bionemo-accel/inventory.json`. Add `.bionemo-accel/` to the target's `.gitignore`.

Before proceeding to Phase 1, install the target's declared dependencies so that a forward pass
can run and `import` checks succeed:

```bash
# prefer uv when available
if [ -f pyproject.toml ]; then
    uv pip install -e . 2>/dev/null || pip install -e .
elif [ -f requirements.txt ]; then
    uv pip install -r requirements.txt 2>/dev/null || pip install -r requirements.txt
fi
```

Install in the same environment as `probe_hardware.py` (same interpreter). If install fails,
record the error in `.bionemo-accel/inventory.json` under `"dep_install_error"` and emit `ENV_`
— do not proceed to architecture matching until the target's own deps are resolvable.

**Phase 1 — Architecture match, or hard stop.** Read `references/architecture-matching.md`. Score
the target on the six rubric axes against the reference menu (encoder/MLM pre-norm and post-norm,
causal LM dense, MoE, genomics LM, encoder–decoder). **Attention pattern is the only required
match** — the other five axes select the reference and the port depth, and each mismatch is
recorded as a caveat. Write `.bionemo-accel/match.json`, noting which reference each piece of the
port comes from. Hard-stop only for diffusion, GNN/equivariant, or state-space architectures, an
attention-pattern mismatch, an unlocatable model definition, or no runnable forward pass; then
write the hard-stop `ACCELERATION_REPORT.md` and exit.

**Phase 2 — Hardware and TE probe.** Run:

```bash
python $SKILL_DIR/scripts/probe_hardware.py -o .bionemo-accel/hardware.json
```

where `$SKILL_DIR` is the directory containing this `SKILL.md`. If the script exits non-zero
because `torch` or `transformer_engine` could not be imported or installed, write an
`ACCELERATION_REPORT.md` with failure class `ENV_` and stop — this is a dependency provisioning
failure, not an architectural judgment. Read `references/precision-selection.md` §Phase 2 for
sm_120/sm_80 caveats and TE version guidance.

**Phase 3 — Precision selection.** Run:

```bash
python $SKILL_DIR/scripts/run_gemm_benchmark.py \
    --inventory .bionemo-accel/inventory.json \
    --hardware  .bionemo-accel/hardware.json \
    -o .bionemo-accel/gemm
```

The benchmark iterates only over the recipes `probe_hardware.py` reported as supported (via
`--no-fp8` / `--no-fp4` skip flags derived from `hardware.json`). A non-zero exit is **not** a
reason to abandon the port:

- If `summary.json` was written (at least one mode succeeded), proceed to Phase 4 using the
  available measurements.
- If `summary.json` is absent (all modes failed), proceed to Phase 4 using the `supported_recipes`
  list from `hardware.json` as the basis for the precision recommendation; note "benchmark
  unavailable" in `precision.json` and in the final report.

Read `references/precision-selection.md` §Phase 3 for interpretation — GEMM speedup is an upper
bound on end-to-end speedup; autocast vs pre-quantize gap is quantization overhead; a ≈1.0× result
means suspected kernel fallback, not no benefit.

**Phase 4 — Rewrite.** Read `references/te-conversion.md` and `references/sequence-packing.md`.
Create branch `bionemo-accel/<family>` in the target repo. Apply Depth A (already TE), B (full
port + converter), B-postnorm (hand-built block), B-encdec (two TE stacks), or C (kernel swaps
only) at the depth warranted by the code.

**Phase 5 — Validate.** Read `references/validation.md`. Generate `parity_check.py` from
`assets/parity_check.py.tmpl` (Tier 1, always); generate `tests/test_modeling_ported.py` and
`tests/conftest.py` from the corresponding templates (Tier 2, when Depth B produced a converter
**or when THD packing was applied at any depth** — `test_golden_values_thd` is the packing
correctness proof and must run even for Depth A ports; use the no-HF-counterpart path from
`references/validation.md` §"When the target has no HF counterpart" if no converter exists);
reproduce CI (Tier 3, when containerised). Fix the port if any tier fails.

**Phase 6 — Report.** Generate `ACCELERATION_REPORT.md` from `assets/ACCELERATION_REPORT.md.tmpl`
and write it to the target repo root. Lead with the measured numbers and the one-line config
change to enable FP8.

## Handoff contracts

Each phase reads its predecessor's JSON artifact. Do not skip phases or produce a downstream
artifact without the upstream ones in place.

| Source  | Artifact                              | Consumed by                 | Required?                                                                             |
| ------- | ------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------- |
| Phase 0 | `.bionemo-accel/inventory.json`       | Phases 1, 3                 | Yes                                                                                   |
| Phase 1 | `.bionemo-accel/match.json` (or exit) | Phases 2, 4                 | Yes                                                                                   |
| Phase 2 | `.bionemo-accel/hardware.json`        | Phase 3                     | Yes                                                                                   |
| Phase 3 | `.bionemo-accel/gemm/summary.json`    | Phase 6                     | No — absent when all benchmark modes fail; note "benchmark unavailable" in the report |
| Phase 3 | `.bionemo-accel/precision.json`       | Phase 4 (config generation) | Yes — fall back to `supported_recipes` from `hardware.json` if summary absent         |
| Phase 5 | tier results                          | Phase 6                     | Yes                                                                                   |

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
- **MoE.** Only straightforward top-k routing matches the Mixtral reference. Exotic routing (expert
  choice, soft MoE) is not a hard stop — port the attention and norm layers, leave the router
  alone, and say so in the report.

## Validation

Full protocol in `references/validation.md`. Summary:

- **Tier 1 — always** — forward logits + loss parity, backward, FP8 recipe sweep, BSHD-vs-THD.
- **Tier 2 — when converter generated or THD packing applied** — `BaseModelTest` harness (~30
  inherited tests: golden values, `test_golden_values_thd` packing proof, conversion round trips,
  init, FP8 recipe sweep). Depth A + packing ports use the no-HF-counterpart path (identity
  converters, skip conversion tests, checked-in baseline).
- **Tier 3 — when containerised** — CI reproduction in `nvcr.io/nvidia/pytorch:26.04-py3`. State
  what was skipped and why.

## Responsible use

- Never modify `$BIONEMO_RECIPES` — it is a read-only reference.
- Low precision ships disabled; the user enables it deliberately.
- Report every skipped test, loosened tolerance, and limitation in `ACCELERATION_REPORT.md`.

## Available Scripts

| Script                          | Purpose                                                                                                  | Key Arguments                                                                                                   |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `scripts/probe_hardware.py`     | Probe GPU and TE install; detect per-recipe FP8/MXFP8/NVFP4 support; optionally install missing packages | `-o <path>` output JSON path; `--no-install` skip install prompt                                                |
| `scripts/run_gemm_benchmark.py` | Run TE GEMM benchmark in autocast and pre-quantize modes; write speedup plots and summary JSON           | `--inventory <json>`; `--hardware <json>`; `-o <dir>`; `--shapes <MxKxN>`; `--allow-clone`; `--verbose-kernels` |

Invoke scripts from the pipeline phases using bash or `run_script()`:

```python
# Phase 2 — hardware probe
run_script("scripts/probe_hardware.py", args=["-o", ".bionemo-accel/hardware.json"])

# Phase 3 — GEMM benchmark
run_script(
    "scripts/run_gemm_benchmark.py",
    args=[
        "--inventory",
        ".bionemo-accel/inventory.json",
        "--hardware",
        ".bionemo-accel/hardware.json",
        "-o",
        ".bionemo-accel/gemm",
    ],
)
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

| Input                | Source                                   | Description                                              |
| -------------------- | ---------------------------------------- | -------------------------------------------------------- |
| Target codebase path | User prompt or current working directory | Repo or folder containing the model to accelerate        |
| `$BIONEMO_RECIPES`   | Environment variable                     | Path to a bionemo-recipes checkout (read-only reference) |

**Optional (passed to scripts)**

| Input               | Flag                                                          | Default                | Description                                                   |
| ------------------- | ------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------- |
| Skip install prompt | `--no-install` (probe_hardware)                               | off                    | Exit 1 immediately if torch or TE missing                     |
| Manual GEMM shapes  | `--shapes MxKxN` (run_gemm_benchmark)                         | derived from inventory | Override model-config-based shapes                            |
| TE source tree      | `--te-source <path>` or `$TE_SOURCE_DIR` (run_gemm_benchmark) | auto-detected          | Explicit path to a Transformer Engine checkout                |
| Auto-clone TE       | `--allow-clone` (run_gemm_benchmark)                          | off                    | Shallow-clone TE source if benchmark script not found locally |
| Kernel dispatch log | `--verbose-kernels` (run_gemm_benchmark)                      | off                    | Set `NVTE_LOG_LEVEL=1` to confirm kernel dispatch             |

## Output Format

All artifacts are written into the **target repo**, never into `$BIONEMO_RECIPES`.

| Artifact                             | Phase | Written                        | Description                                                                                                                    |
| ------------------------------------ | ----- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `.bionemo-accel/inventory.json`      | 0     | Always                         | Model dimensions, entry points, framework, TE import status                                                                    |
| `.bionemo-accel/match.json`          | 1     | On match                       | Architecture family, confidence score, matched reference                                                                       |
| `.bionemo-accel/hardware.json`       | 2     | Always                         | GPU, compute capability, per-recipe TE support flags                                                                           |
| `.bionemo-accel/gemm/summary.json`   | 3     | When ≥1 benchmark run succeeds | GEMM speedup for successful modes, skip flags applied, interpretation reminders                                                |
| `.bionemo-accel/precision.json`      | 3     | Always                         | Selected recipe and rationale                                                                                                  |
| `parity_check.py`                    | 5     | Always                         | Tier 1 forward-pass parity check against the original model                                                                    |
| `tests/test_modeling_ported.py`      | 5     | Depth B only                   | BaseModelTest harness subclass for the ported model                                                                            |
| `tests/conftest.py`                  | 5     | Depth B only                   | pytest plugin registration (`pytest_plugins = ["tests.common.fixtures"]`)                                                      |
| `ACCELERATION_REPORT.md`             | 6     | Always                         | Final report: measured speedups, one-line FP8 enable config, limitations                                                       |
| `ACCELERATION_REPORT.md` (hard stop) | 1     | On out-of-scope architecture   | Rejection reason (no TE analogue, attention-pattern mismatch, or no runnable forward pass); no other files written or modified |

**Branch:** `bionemo-accel/<family>` is created in the target repo on a successful port (e.g. `bionemo-accel/encoder-mlm`, `bionemo-accel/causal-lm-dense`).

## Troubleshooting

| Error / Symptom                                    | Cause                                                                                                                                  | Solution                                                                                                                                 |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Hard stop at Phase 1                               | Architecture has no TE analogue (diffusion, GNN/equivariant, state-space), attention pattern mismatches, or no forward pass can be run | Read `references/architecture-matching.md`. An advisory-axis mismatch alone is never a stop — it selects the reference and depth instead |
| `Could not find benchmarks/gemm/benchmark_gemm.py` | TE pip wheel does not include source; no NGC container or local checkout found                                                         | Pass `--te-source <path>`, set `TE_SOURCE_DIR`, or add `--allow-clone`                                                                   |
| GEMM speedup near 1.0×                             | Silent kernel fallback to a lower-precision kernel                                                                                     | Re-run with `--verbose-kernels`; confirm expected dispatch in `NVTE_LOG_LEVEL=1` output before concluding there is no benefit            |
| `torch not importable` after install               | Script ran in the wrong Python environment                                                                                             | Verify `which python` inside your training venv/conda env; re-run `probe_hardware.py` in that environment                                |
| `transformer_engine.pytorch.autocast` missing      | TE version predates the API BioNeMo recipes use                                                                                        | Upgrade to `transformer-engine[pytorch]>=2.0` or use `nvcr.io/nvidia/pytorch:26.04-py3`                                                  |
| Parity check fails after port                      | Weight conversion bug or `te.autocast` scope too narrow                                                                                | Compare QKV packing against `$BIONEMO_RECIPES/models/esm2/convert.py`; verify `te.autocast` wraps the full forward pass                  |
| `quantized_model_init` + megatron-fsdp fails       | BIONEMO-3012 (upstream xfail)                                                                                                          | Do not combine `quantized_model_init` with `megatron-fsdp` until resolved; note in report                                                |
