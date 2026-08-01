# Architecture matching and the hard stop

Phase 1 decides whether the port happens at all. Get this wrong and everything downstream is a
plausible-looking model that silently computes something else.

## The four supported families

Each family exists because this repo contains a working, tested TE implementation of it. There is no
fifth family — if the target does not fit one of these, there is nothing to copy from.

### 1. Encoder / masked LM

**References:** `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py` (canonical), `$BIONEMO_RECIPES/models/codonfm/modeling_codonfm_te.py`,
`$BIONEMO_RECIPES/models/amplify/`
**Converter reference:** `$BIONEMO_RECIPES/models/esm2/convert.py`
**Recipe reference:** `$BIONEMO_RECIPES/recipes/esm2_native_te/`

Fingerprint:

- Bidirectional attention — no causal mask anywhere in the attention path.
- MLM head (`lm_head`, tied or untied to the embedding) or a token/sequence classification head.
- LayerNorm (not RMSNorm), GELU MLP.
- Learned absolute or rotary position embeddings.

### 2. Causal LM, dense

**References:** `$BIONEMO_RECIPES/models/llama3/modeling_llama_te.py` (canonical), `$BIONEMO_RECIPES/models/qwen/modeling_qwen2_te.py`,
`$BIONEMO_RECIPES/models/qwen/modeling_qwen3_te.py`
**Converter reference:** `$BIONEMO_RECIPES/models/llama3/convert.py`, `$BIONEMO_RECIPES/models/qwen/convert_qwen2.py`,
`$BIONEMO_RECIPES/models/qwen/convert_qwen3.py`
**Recipe reference:** `$BIONEMO_RECIPES/recipes/llama3_native_te/`

Fingerprint:

- Causal attention mask.
- RMSNorm.
- SwiGLU / gated MLP (`gate_proj` + `up_proj` + `down_proj`).
- Grouped-query attention — `num_key_value_heads < num_attention_heads`.
- Rotary position embeddings.

### 3. Mixture of experts

**References:** `$BIONEMO_RECIPES/models/mixtral/modeling_mixtral_te.py`
**Converter reference:** `$BIONEMO_RECIPES/models/mixtral/convert.py`
**Supporting:** `$BIONEMO_RECIPES/models/mixtral/fused_token_router.py`, `fused_a2a.py`,
`fused_indices_converter.py`

Fingerprint: everything in family 2, plus a router / `num_local_experts` / `num_experts_per_tok`
top-k gating replacing the dense MLP.

MoE is the hardest port — expert-parallel all-to-all and the fused router are load-bearing. Only
attempt it when the target's routing is a straightforward top-k over a linear gate. Anything
exotic (expert choice routing, soft MoE, shared experts with unusual normalization) is a hard stop.

### 4. Genomics causal LM

**References:** `$BIONEMO_RECIPES/recipes/opengenome2_llama_native_te/` (a Llama-family fork; its
`modeling_llama_te.py` is a managed copy of `$BIONEMO_RECIPES/models/llama3/modeling_llama_te.py`)

Fingerprint: family 2's block structure over a nucleotide/codon vocabulary, small vocab, long
context, THD-packed GQA configs. Treat as family 2 for the rewrite; the distinction matters for
config defaults (vocab padding, sequence length, packing) and for which recipe to copy.

## The rubric

Score the target on six axes. Record each as `match` / `mismatch` / `unknown` in
`.bionemo-accel/match.json`, with the evidence (file and symbol) that decided it.

| Axis                | What to look for                                              |
| ------------------- | ------------------------------------------------------------- |
| Attention pattern   | causal vs bidirectional; sliding window; any custom bias term |
| Normalization       | LayerNorm vs RMSNorm; pre-norm vs post-norm                   |
| MLP form            | plain up/down + activation vs gated SwiGLU vs MoE             |
| Positional encoding | learned absolute, RoPE, ALiBi, relative, none                 |
| Attention grouping  | MHA vs GQA vs MQA                                             |
| Head structure      | MLM, causal LM, classification, regression, multi-task        |

**Proceed only when attention pattern, normalization, and MLP form all `match`.** Those three
determine whether `te.TransformerLayer` can express the block at all. The other three axes are
config parameters — a mismatch there means "set a different flag", not "cannot port".

Record the confidence and the *specific* deciding evidence. "Looks like Llama" is not evidence;
`model.py::DecoderBlock` uses `RMSNorm` + `SwiGLU` + `num_key_value_heads=8` is.

## Hard stop: architectures that are out of scope

Stop, write the report, change nothing, when the target is any of:

- **Diffusion / score-based models** — the denoiser conditioning path (timestep embeddings, AdaLN
  modulation) is not expressible as a `te.TransformerLayer`.
- **GNNs and equivariant networks** (SE(3), E(3), tensor-field networks) — message passing and
  irrep-typed tensors have no TE analogue.
- **State-space models** (Mamba, S4, Hyena) — the block is a scan, not attention. Note: Evo2 lives
  in `$BIONEMO_RECIPES/recipes/evo2_megatron/` on the Megatron stack, which this skill does not target.
- **Encoder–decoder** (T5, BART) — cross-attention needs a second TE layer type wired differently
  than any reference here.
- **Vision-only backbones** — `$BIONEMO_RECIPES/recipes/vit/` exists but has no `BaseModelTest` coverage, so there is
  no validation path.
- **Megatron-LM based code** — `$BIONEMO_RECIPES/recipes/eden_megatron/` and `$BIONEMO_RECIPES/recipes/evo2_megatron/` already handle
  precision through `--mixed-precision-recipe`. Point the user there; do not port.
- **Any hybrid** whose transformer block cannot round-trip through a `te.TransformerLayer`.

Also hard stop when:

- The three required rubric axes do not all `match`.
- The model definition cannot be located or is generated dynamically at runtime.
- No forward pass can be run for the parity check (no weights, no tokenizer, no sample input) —
  because then Phase 5 cannot prove anything.

## The hard-stop report

Fill `assets/ACCELERATION_REPORT.md.tmpl` with the hard-stop variant. It must state:

1. What architecture was detected, with the file and class names that identified it.
2. Which reference family scored highest, and its score on each of the six axes.
3. The **specific rubric axis that failed** and why it is disqualifying.
4. Which accelerations, if any, would still be safe to apply by hand — for example, TE `FusedAdam`
   and `torch.compile` are architecture-agnostic — clearly marked as *not applied and not validated
   by this skill*.
5. A pointer to the nearest recipe if the user wants to port manually.

Then exit. Do not offer to "try anyway".
