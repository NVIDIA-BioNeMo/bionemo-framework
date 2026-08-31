# Architecture matching and the hard stop

Phase 1 decides how the port happens, and in a few cases whether it happens at all. Get this wrong
and everything downstream is a plausible-looking model that silently computes something else.

Most targets are portable. The job of this phase is to pick the right reference to copy and the
right depth to work at — not to look for reasons to refuse.

## Reference implementations you can borrow from

This is what the repo has working, tested TE code for. It is a menu, not a taxonomy: a target that
resembles none of these closely can still be ported, just at less depth and with more of the block
written by hand.

### Encoder / masked LM — pre-norm

**References:** `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py` (canonical), `$BIONEMO_RECIPES/models/codonfm/modeling_codonfm_te.py`,
`$BIONEMO_RECIPES/models/amplify/`
**Converter reference:** `$BIONEMO_RECIPES/models/esm2/convert.py`
**Recipe reference:** `$BIONEMO_RECIPES/recipes/esm2_native_te/`

Fingerprint:

- Bidirectional attention — no causal mask anywhere in the attention path.
- MLM head (`lm_head`, tied or untied to the embedding) or a token/sequence classification head.
- LayerNorm (not RMSNorm), applied *before* the sublayer.
- Plain, non-gated MLP. The activation is a config value, not part of the fingerprint — see the
  MLP form row of the rubric.
- Learned absolute, rotary, or no position embeddings.

Built on `te.TransformerLayer`.

### Encoder / masked LM — post-norm

**Reference:** `$BIONEMO_RECIPES/models/geneformer/src/geneformer/modeling_bert_te.py::TEBertLayer`
**Converter reference:** `$BIONEMO_RECIPES/models/geneformer/src/geneformer/convert.py`
(`convert_geneformer_hf_to_te` / `convert_geneformer_te_to_hf`)
**Recipe reference:** `$BIONEMO_RECIPES/recipes/geneformer_native_te_mfsdp_fp8/`

Fingerprint: everything above, except LayerNorm is applied *after* the residual add.

This variant deliberately does **not** use `te.TransformerLayer`, because that layer is pre-norm.
From the `TEBertLayer` docstring:

> Geneformer/HF BERT (POST-norm): Input -> Attention -> Dropout -> Residual Add -> LayerNorm -> MLP
> -> Dropout -> Residual Add -> LayerNorm -> Output. Typical TransformerLayer (PRE-norm): Input ->
> [LayerNorm Attn inside MultiheadAttention] -> Dropout -> Residual Add -> \[LayerNorm MLP inside
> LayerNormMLP\] -> Dropout -> Residual Add -> Output.

So the block is assembled from `te.MultiheadAttention(input_layernorm=False)`, `te.LayerNorm`, and
two `te.Linear`. Geneformer is also an all-ReLU encoder — it *requires* `hidden_act="relu"` — which
makes it the reference to reach for whenever a target's activation or norm placement doesn't look
like ESM-2's.

Note: geneformer has no `BaseModelTest` coverage, so Tier 2 follows the codonfm no-HF-upstream
template described in `references/validation.md`.

### Causal LM, dense

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

### Mixture of experts

**References:** `$BIONEMO_RECIPES/models/mixtral/modeling_mixtral_te.py`
**Converter reference:** `$BIONEMO_RECIPES/models/mixtral/convert.py`
**Supporting:** `$BIONEMO_RECIPES/models/mixtral/fused_token_router.py`, `fused_a2a.py`,
`fused_indices_converter.py`

Fingerprint: everything in the dense causal LM, plus a router / `num_local_experts` /
`num_experts_per_tok` top-k gating replacing the dense MLP.

MoE is the hardest port — expert-parallel all-to-all and the fused router are load-bearing. The
Mixtral reference matches straightforward top-k routing over a linear gate. When the routing is
exotic (expert choice, soft MoE, shared experts with unusual normalization), do not force it through
the fused router: port the attention and norm layers, leave the routing code untouched, and say
exactly that in the report.

### Genomics causal LM

**References:** `$BIONEMO_RECIPES/recipes/opengenome2_llama_native_te/` (a Llama-family fork; its
`modeling_llama_te.py` is a managed copy of `$BIONEMO_RECIPES/models/llama3/modeling_llama_te.py`)

Fingerprint: the dense causal LM block structure over a nucleotide/codon vocabulary, small vocab,
long context, THD-packed GQA configs. Treat it as a dense causal LM for the rewrite; the distinction
matters for config defaults (vocab padding, sequence length, packing) and for which recipe to copy.

### Encoder–decoder

**Reference:** none in this repo — this is a TE capability entry, not a worked model.

`te.TransformerLayer` supports cross-attention natively via `layer_type`. From the vendored copy at
`$BIONEMO_RECIPES/recipes/codonfm_ptl_te/src/models/components/encodon_te_layer.py`:

> `layer_type: {'encoder', 'decoder'}, default = 'encoder'` — if set to `decoder`, an additional
> cross-attn block is added after self-attn. This can be used for structures like `T5` Transformer
> in conjunction with the `encoder` option.

The same file builds `self.inter_attention` with `attention_type="cross"` when
`layer_type == "decoder"`, and threads `encoder_output` and `enc_dec_attn_mask` through the forward.
Every BioNeMo model happens to pass `layer_type="encoder"`; that is a gap in coverage, not a
limitation of TE.

Build the encoder stack the way the encoder references do, then the decoder stack with
`layer_type="decoder"`, `self_attn_mask_type` causal, and `enc_dec_attn_mask_type` for the cross
path. Because nothing in-repo exercises this, three caveats are mandatory in the report:

- **No converter template.** `$BIONEMO_RECIPES/models/esm2/convert.py::_pack_qkv_weight` packs
  self-attention QKV only. `inter_attention` takes Q from the decoder and KV from the encoder
  output, so that transform is hand-written.
- **No `BaseModelTest` coverage.** The harness assumes a single stack. Tier 2 is the codonfm
  no-HF-upstream path at best; Tier 1 parity is the real proof.
- **THD packing across a cross-attention boundary is untested here.** Default to BSHD.

### Mixing pieces

Nothing requires a target to take everything from one entry. A post-norm encoder with continuous
inputs can take its block from geneformer and its `quantized_model_init` handling, FP8 config
plumbing, and padded-vocab treatment from `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py`. Say in
`.bionemo-accel/match.json` which reference each piece came from.

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

**Proceed when attention pattern matches.** Causal vs bidirectional is the one axis that changes
what the model computes and cannot be recovered by configuration. The other five are advisory: they
decide *which reference to copy and how much of the block you write by hand*, and each mismatch goes
in the report as a caveat, not a rejection.

What a mismatch on each advisory axis actually costs:

- **Normalization.** LayerNorm vs RMSNorm is a `normalization=` kwarg on the TE layer. Pre- vs
  post-norm selects the reference: pre-norm → `te.TransformerLayer` (ESM-2); post-norm →
  hand-assembled `TEBertLayer` (geneformer). Neither is a stop.
- **MLP form.** This axis is about *structure* — plain up/down vs gated vs MoE. **Activation choice
  is not a mismatch.** `te.TransformerLayer` accepts `gelu, geglu, qgelu, qgeglu, relu, reglu, srelu, sreglu, silu, swiglu`; `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py` passes
  `config.encoder_activation` through unmodified, `$BIONEMO_RECIPES/models/codonfm/modeling_codonfm_te.py`
  whitelists `gelu`/`relu`/`silu`, and `$BIONEMO_RECIPES/models/geneformer/` is ReLU throughout. A
  ReLU encoder is an encoder.
- **Positional encoding.** A config choice, including `none`. Note that
  `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py::NVEsmEmbeddings` raises on anything but rotary,
  so a target with no position encoding drops the `RotaryPositionEmbedding` construction and the
  `rotary_pos_emb=` kwarg rather than configuring them.
- **Attention grouping.** `num_gqa_groups`. The encoder references set it equal to
  `num_attention_heads` for plain MHA.
- **Head structure.** Multi-task, regression, and continuous-value heads are portable — the block is
  what gets accelerated. What they change is the *validation* path: with no HF counterpart, Tier 2
  follows the codonfm template in `references/validation.md`.

Record the confidence and the *specific* deciding evidence. "Looks like Llama" is not evidence;
`model.py::DecoderBlock` uses `RMSNorm` + `SwiGLU` + `num_key_value_heads=8` is.

## Hard stop: architectures that are out of scope

Three architecture families have no TE analogue at all. Stop, write the report, change nothing:

- **Diffusion / score-based models** — the denoiser conditioning path (timestep embeddings, AdaLN
  modulation) is not expressible as a `te.TransformerLayer`.
- **GNNs and equivariant networks** (SE(3), E(3), tensor-field networks) — message passing and
  irrep-typed tensors have no TE analogue.
- **State-space models** (Mamba, S4, Hyena) — the block is a scan, not attention. Note: Evo2 lives
  in `$BIONEMO_RECIPES/recipes/evo2_megatron/` on the Megatron stack, which this skill does not target.

Also stop when:

- The attention pattern does not match any reference — causal vs bidirectional is not configurable.
- The model definition cannot be located or is generated dynamically at runtime.
- No forward pass can be run for the parity check because of a reason intrinsic to the target —
  no weights, no tokenizer, no sample input — because then Phase 5 cannot prove anything. Use
  failure class `ARCH_` here.

Do **not** use `ARCH_` when a forward pass cannot be run because a dependency failed to install.
That is failure class `ENV_`: the architecture has not been judged, and the run may succeed in a
clean environment. See the "Failure classes" section in `SKILL.md`.

Not a hard stop, but route elsewhere: **Megatron-LM based code** —
`$BIONEMO_RECIPES/recipes/eden_megatron/` and `$BIONEMO_RECIPES/recipes/evo2_megatron/` already
handle precision through `--mixed-precision-recipe`. Point the user there rather than porting.

Not a hard stop, but Tier 1 only: **vision-only backbones** — `$BIONEMO_RECIPES/recipes/vit/` exists
but has no `BaseModelTest` coverage, so parity rests on the Tier 1 check alone. Say so in the report.

When an advisory axis is weak enough that `te.TransformerLayer` cannot express the block at all, the
answer is Depth C in `references/te-conversion.md` — architecture-agnostic wins with the limitation
stated — not a refusal.

## The hard-stop report

All architectural hard-stops use failure class `ARCH_`. Fill `assets/ACCELERATION_REPORT.md.tmpl`
with the hard-stop variant. It must state:

1. What architecture was detected, with the file and class names that identified it.
2. Which reference scored highest, and its score on each of the six axes.
3. The **specific reason it is out of scope** — one of the three families above, an attention-pattern
   mismatch, or the absence of a runnable forward pass. An advisory-axis mismatch is never the
   reason; if that is all you have, port at the depth the block supports instead.
4. Which accelerations, if any, would still be safe to apply by hand — for example, TE `FusedAdam`
   and `torch.compile` are architecture-agnostic — clearly marked as *not applied and not validated
   by this skill*.
5. A pointer to the nearest recipe if the user wants to port manually.

For the three out-of-scope families, then exit; do not offer to "try anyway".
