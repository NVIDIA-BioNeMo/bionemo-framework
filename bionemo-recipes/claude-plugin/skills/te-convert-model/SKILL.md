---
name: te-convert-model
description: >
  Convert a HuggingFace PreTrainedModel to use NVIDIA TransformerEngine layers.
  Triggers when user asks to TE-ify, add TransformerEngine, convert for FP8,
  or optimize transformer layers with TE.
allowed-tools: Read, Grep, Glob, Write, Edit, Bash, Agent
argument-hint: '[model-path or HF model ID]'
---

# Convert a HuggingFace Model to TransformerEngine

You are converting a HuggingFace `PreTrainedModel` to use NVIDIA TransformerEngine (TE) layers. This enables FP8/FP4 quantization, fused attention kernels, and optimized distributed training.

## Reference Files

Before starting, read the reference files in this skill's `reference/` directory:

- `esm2_convert.py` — Encoder (BERT-like) conversion pattern
- `llama3_convert.py` — Decoder (causal LM) conversion pattern
- `state.py` — State dict transformation framework
- `esm2_modeling_te.py` — Encoder TE model implementation
- `llama3_modeling_te.py` — Decoder TE model implementation

## Step-by-Step Workflow

### Step 1: Analyze the Source Model

Read the model files to identify:

- **Architecture type**: encoder (BERT, ESM, RoBERTa) vs decoder (GPT, Llama, Mistral) vs encoder-decoder
- **Attention pattern**: MHA (all heads same), GQA (grouped query), MQA (single KV head)
- **Layer structure**: Find `nn.TransformerEncoderLayer`, `nn.MultiheadAttention`, or custom attention
- **FFN pattern**: Standard (dense→activation→dense) vs SwiGLU (gate_proj, up_proj, down_proj)
- **Normalization**: LayerNorm vs RMSNorm
- **Position embeddings**: Absolute, rotary (RoPE), ALiBi, etc.

### Step 2: Create the NV Config Class

Extend the source model's config class with TE-specific fields:

```python
from transformers import SomeConfig  # The original config class


class NVSomeConfig(SomeConfig):
    model_type: str = "nv_some"  # New model type for HF registry

    def __init__(
        self,
        # TE attention format: "bshd" (padded) or "thd" (packed sequences)
        attn_input_format: str = "bshd",
        # Fuse Q/K/V into single parameter for optimized kernels
        fuse_qkv_params: bool = True,
        # Padded vocab size for FP8 (must be divisible by 16)
        padded_vocab_size: int | None = None,
        # Per-layer quantization: ["fp8", "fp4", None] per layer
        layer_precision: list[str | None] | None = None,
        # Initialize directly in quantized format
        use_quantized_model_init: bool = False,
        # For decoder models: causal attention mask
        # self_attn_mask_type: str = "padding_causal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_input_format = attn_input_format
        self.fuse_qkv_params = fuse_qkv_params
        self.padded_vocab_size = padded_vocab_size or getattr(self, "vocab_size", None)
        self.layer_precision = layer_precision
        self.use_quantized_model_init = use_quantized_model_init

        # Validate layer_precision
        if layer_precision is not None:
            if len(layer_precision) != self.num_hidden_layers:
                raise ValueError(
                    f"layer_precision must have length {self.num_hidden_layers}"
                )
```

### Step 3: Build the TE Model Class

Replace standard attention with `transformer_engine.pytorch.TransformerLayer`:

```python
import transformer_engine.pytorch as te
import transformer_engine.common.recipe
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding


class NVSomeModel(PreTrainedModel):
    config_class = NVSomeConfig

    def __init__(self, config, fp8_recipe=None, fp4_recipe=None):
        super().__init__(config)
        self._fp8_recipe = fp8_recipe
        self._fp4_recipe = fp4_recipe

        # Embeddings (standard PyTorch, NOT TE)
        self.embed_tokens = nn.Embedding(config.padded_vocab_size, config.hidden_size)

        # Build TE transformer layers
        layers = []
        for i in range(config.num_hidden_layers):
            with self.get_autocast_context(i, init=True):
                layers.append(
                    te.TransformerLayer(
                        hidden_size=config.hidden_size,
                        ffn_hidden_size=config.intermediate_size,
                        num_attention_heads=config.num_attention_heads,
                        # For GQA models:
                        num_gqa_groups=getattr(
                            config, "num_key_value_heads", config.num_attention_heads
                        ),
                        # Encoder: "LayerNorm", Decoder: "RMSNorm"
                        normalization="LayerNorm",  # or "RMSNorm"
                        # Encoder: "gelu", Decoder: "swiglu"
                        activation="gelu",  # or "swiglu"
                        # Encoder: "encoder", Decoder: omit or use default
                        layer_type="encoder",  # omit for decoder
                        attn_input_format=config.attn_input_format,
                        self_attn_mask_type=getattr(
                            config, "attn_mask_type", "padding"
                        ),
                        fuse_qkv_params=config.fuse_qkv_params,
                        qkv_weight_interleaved=True,
                        layer_number=i + 1,  # 1-indexed!
                        bias=True,  # False for Llama-style
                        params_dtype=config.dtype,
                        device=(
                            "meta"
                            if torch.get_default_device() == torch.device("meta")
                            else "cuda"
                        ),
                    )
                )
        self.layers = nn.ModuleList(layers)

        # Final layer norm
        self.norm = te.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Or for RMSNorm: te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads
        )
```

**Key patterns:**

- `layer_number` is 1-indexed (not 0-indexed)
- Encoder models use `layer_type="encoder"` and `normalization="LayerNorm"`
- Decoder models use `normalization="RMSNorm"` and `activation="swiglu"`
- GQA models set `num_gqa_groups=config.num_key_value_heads`
- For encoder models, `num_gqa_groups=config.num_attention_heads` (MHA = all heads)

**Override `state_dict()` to filter TE internal state:**

```python
def state_dict(self, *args, **kwargs):
    sd = super().state_dict(*args, **kwargs)
    return {k: v for k, v in sd.items() if not k.endswith("_extra_state")}
```

**Implement `init_empty_weights()` for meta device support:**

```python
def init_empty_weights(self):
    for module in self.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    self.embed_tokens.to_empty(device="cuda")
    self.embed_tokens.apply(self._init_weights)
    self.tie_weights()
```

**Implement `get_autocast_context()` for FP8/FP4:**

```python
from contextlib import nullcontext


def get_autocast_context(self, layer_number, init=False, outer=False):
    if self.config.layer_precision is None:
        return nullcontext()
    if outer:
        if "fp8" not in self.config.layer_precision:
            return nullcontext()
        return te.autocast(enabled=True, recipe=self._fp8_recipe)
    precision = self.config.layer_precision[layer_number]
    recipe = {"fp8": self._fp8_recipe, "fp4": self._fp4_recipe}.get(precision)
    if init and self.config.use_quantized_model_init:
        if precision in ("fp8", "fp4"):
            return te.quantized_model_init(recipe=recipe)
        return nullcontext()
    if precision in ("fp8", "fp4"):
        return te.autocast(enabled=True, recipe=recipe)
    return te.autocast(enabled=False)
```

**LM Head — keep in higher precision:**

```python
class NVSomeForMaskedLM(PreTrainedModel):
    def forward(self, ...):
        hidden = self.model(...)
        # Disable FP8 for the LM head to maintain precision
        with te.autocast(enabled=False):
            logits = self.lm_head(hidden)
        # Truncate padded vocab logits
        if self.config.padded_vocab_size != self.config.vocab_size:
            logits = logits[..., :self.config.vocab_size]
```

### Step 4: Write the State Dict Mapping

Create a mapping dict that renames HF state dict keys to TE keys:

**For encoder models (BERT-like):**

```python
mapping = {
    # Attention output projection
    "encoder.layer.*.attention.output.dense.weight": "encoder.layers.*.self_attention.proj.weight",
    "encoder.layer.*.attention.output.dense.bias": "encoder.layers.*.self_attention.proj.bias",
    # Attention LayerNorm → TE's layernorm_qkv (fused LN + QKV)
    "encoder.layer.*.attention.LayerNorm.weight": "encoder.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "encoder.layer.*.attention.LayerNorm.bias": "encoder.layers.*.self_attention.layernorm_qkv.layer_norm_bias",
    # FFN layers → TE's layernorm_mlp (fused LN + MLP)
    "encoder.layer.*.intermediate.dense.weight": "encoder.layers.*.layernorm_mlp.fc1_weight",
    "encoder.layer.*.intermediate.dense.bias": "encoder.layers.*.layernorm_mlp.fc1_bias",
    "encoder.layer.*.output.dense.weight": "encoder.layers.*.layernorm_mlp.fc2_weight",
    "encoder.layer.*.output.dense.bias": "encoder.layers.*.layernorm_mlp.fc2_bias",
    # FFN LayerNorm
    "encoder.layer.*.LayerNorm.weight": "encoder.layers.*.layernorm_mlp.layer_norm_weight",
    "encoder.layer.*.LayerNorm.bias": "encoder.layers.*.layernorm_mlp.layer_norm_bias",
}
```

**For decoder models (Llama-like):**

```python
mapping = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.layers.*.input_layernorm.weight": "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "model.layers.*.self_attn.o_proj.weight": "model.layers.*.self_attention.proj.weight",
    "model.layers.*.post_attention_layernorm.weight": "model.layers.*.layernorm_mlp.layer_norm_weight",
    "model.layers.*.mlp.down_proj.weight": "model.layers.*.layernorm_mlp.fc2_weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}
```

**TE naming conventions:**

- `self_attention.layernorm_qkv` — Fused LayerNorm + QKV projection
  - `.weight` = QKV weights, `.bias` = QKV bias
  - `.layer_norm_weight` = LayerNorm weight, `.layer_norm_bias` = LayerNorm bias
- `self_attention.proj` — Output projection
- `layernorm_mlp` — Fused LayerNorm + MLP
  - `.fc1_weight` / `.fc1_bias` = First FFN layer (or gate+up for SwiGLU)
  - `.fc2_weight` / `.fc2_bias` = Second FFN layer
  - `.layer_norm_weight` / `.layer_norm_bias` = LayerNorm

### Step 5: Write Bidirectional Conversion

**HF → TE conversion:**

```python
import state  # The transform system from reference/state.py


def convert_hf_to_te(model_hf, **config_kwargs):
    te_config = NVSomeConfig(**model_hf.config.to_dict(), **config_kwargs)
    with torch.device("meta"):
        model_te = NVSomeForMaskedLM(te_config)

    return state.apply_transforms(
        model_hf,
        model_te,
        mapping,
        transforms=[
            # For encoder: pack Q/K/V into fused QKV
            _pack_qkv_weight,
            _pack_qkv_bias,
            # For decoder: use TransformFns
            # state.state_transform(
            #     source_key=("*.q_proj.weight", "*.k_proj.weight", "*.v_proj.weight"),
            #     target_key="*.layernorm_qkv.weight",
            #     fn=state.TransformFns.merge_qkv,
            # ),
            # state.state_transform(
            #     source_key=("*.gate_proj.weight", "*.up_proj.weight"),
            #     target_key="*.layernorm_mlp.fc1_weight",
            #     fn=state.TransformFns.merge_fc1,
            # ),
        ],
    )
```

**TE → HF conversion:**

```python
def convert_te_to_hf(model_te, **config_kwargs):
    import inspect

    te_config_dict = model_te.config.to_dict()
    valid_keys = set(inspect.signature(OriginalConfig.__init__).parameters)
    filtered = {k: v for k, v in te_config_dict.items() if k in valid_keys}
    hf_config = OriginalConfig(**filtered, **config_kwargs)

    with torch.device("meta"):
        model_hf = OriginalModel(hf_config)

    reverse_mapping = {v: k for k, v in mapping.items()}
    return state.apply_transforms(
        model_te,
        model_hf,
        reverse_mapping,
        transforms=[_unpack_qkv_weight, _unpack_qkv_bias],
        state_dict_ignored_entries=[...],  # tied weights, etc.
    )
```

**QKV packing for MHA (encoder) — interleaved format:**

```python
@state.state_transform(
    source_key=("*.query.weight", "*.key.weight", "*.value.weight"),
    target_key="*.layernorm_qkv.weight",
)
def _pack_qkv_weight(ctx, query, key, value):
    concat = torch.cat((query, key, value), dim=0)
    num_heads = ctx.target.config.num_attention_heads
    concat = concat.view(3, num_heads, -1, query.size(-1))
    concat = concat.transpose(0, 1).contiguous()
    return concat.view(-1, query.size(-1))
```

**QKV merging for GQA (decoder) — use TransformFns:**

```python
state.state_transform(
    source_key=("*.q_proj.weight", "*.k_proj.weight", "*.v_proj.weight"),
    target_key="*.layernorm_qkv.weight",
    fn=state.TransformFns.merge_qkv,
)
```

### Step 6: Write Golden Value Test

```python
def test_golden_values():
    model_hf = OriginalModel.from_pretrained("model-id", dtype=torch.bfloat16).cuda()
    model_te = convert_hf_to_te(model_hf)
    model_te.to("cuda")

    input_data = prepare_test_input()

    with torch.no_grad():
        hf_out = model_hf(**input_data)
        te_out = model_te(**input_data)

    torch.testing.assert_close(te_out.loss, hf_out.loss, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(te_out.logits, hf_out.logits, atol=2.0, rtol=1e-4)
```

### Step 7: Add AUTO_MAP for HuggingFace Integration

Define in the model file for `AutoModel.from_pretrained()` with `trust_remote_code=True`:

```python
AUTO_MAP = {
    "AutoConfig": "model_file_name.NVSomeConfig",
    "AutoModel": "model_file_name.NVSomeModel",
    "AutoModelForMaskedLM": "model_file_name.NVSomeForMaskedLM",  # or ForCausalLM
}
```

## Important Notes

- **Copy `state.py`** from the reference directory into the user's project. It is a standalone utility.
- **Embedding layer stays in PyTorch** — only transformer layers use TE.
- **FP32 rotary embeddings** — always compute RoPE outside `torch.autocast` for stability.
- **Tied weights** — call `self.tie_weights()` after conversion and after `init_empty_weights()`.
- **`_extra_state`** — TE adds internal state that must be filtered from `state_dict()`.
- **Vocab padding** — for FP8, pad vocab to multiple of 16; fill padding with zeros (embeddings) or min float (bias).
