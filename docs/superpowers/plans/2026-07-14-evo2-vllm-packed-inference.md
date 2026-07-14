# Evo2 vLLM Packed Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an out-of-tree vLLM 0.20.0 Evo2/Vortex backend that is accurate, refittable from NeMo-RL, distributed across the available two H100s, and at least as fast as the current MCore path for a 96-genome GDPO rollout.

**Architecture:** Register a lazy `Evo2ForCausalLM` plugin and use vLLM's native attention, tensor-parallel linear, sampler, scheduler, cache allocator, CUDA-graph, and NeMo-RL lifecycle paths. Evo2 Hyena layers expose two uniform fp32 Mamba-style cache tensors while custom segmented Triton FIR and modal-IIR operations enforce request boundaries for packed mixed-length prefill and decode. Preserve MBridge parameter names at the loader boundary so stock NeMo-RL refit streams weights directly, with derived filter buffers refreshed in place outside generation.

**Tech Stack:** Python 3.13, PyTorch 2.11, Triton, vLLM 0.20.0, Transformers, safetensors, NeMo-RL, Ray, Megatron Bridge, pytest, Ruff, Nsight Systems, two H100 80GB GPUs.

## Global Constraints

- Work only in `/data/jstjohn/evo2-vllm-lab`; treat `/data/jstjohn/evo2-mcore-pr5274-lab` as read-only until the final advisory review.
- Pin vLLM exactly to `0.20.0`; expose it as an optional extra and do not add it to ordinary Evo2 training installs.
- Keep all vLLM source and package metadata under `recipes/evo2_megatron`; the phage recipe's existing `src/bionemo/evo2` symlink resolves to that source, so never create a destination copy or copy-map entry.
- Use the base Evo2 7B Microviridae checkpoint, never a prior RL checkpoint, for final standalone and GDPO startup tests.
- Use bf16 parameters, fp32 projection/operator recurrent state, prefix caching disabled, and speculative decoding disabled.
- Every packed recurrent operation consumes authoritative request boundaries and cache-slot indices; no state may cross requests.
- Block `0` is vLLM's null recurrent-cache block and must never be mutated by graph-padding requests.
- No Python request loop may remain in production short-prompt prefill or decode.
- The required hardware proofs are TP2/DP1 and TP1/DP2; TP2/DP2 is a documented design unless four GPUs become available.
- Final performance uses 96 requests, homogeneous controls and mixed prompt lengths 4-12, the production generation length and stop behavior, three warmups, and at least ten interleaved repetitions.
- Do not patch the pinned vLLM checkout for the final implementation.

## File Map

Production source:

```text
recipes/evo2_megatron/src/bionemo/evo2/vllm/
  __init__.py       stable public exports without eager vLLM imports
  plugin.py         idempotent Transformers and ModelRegistry registration
  config.py         Evo2Config, pattern validation, state-shape calculations
  packed_fir.py     scalar/dense reference, Triton prefill/decode FIR dispatch
  packed_iir.py     scalar/dense reference, Triton modal recurrence, long prefill
  hyena.py          HCS/HCM/HCL vLLM recurrent mixer and custom-op boundary
  layers.py         attention, RMSNorm/residual, and SwiGLU decoder layers
  model.py          Evo2Model/Evo2ForCausalLM and hybrid-cache interface
  weights.py        MBridge/Vortex name mapping, TP loaders, derived buffers
  export.py         streaming MBridge-to-vLLM config/safetensors export CLI
  accuracy.py       logits, logprob, identity, replay, and refit comparison CLI
  benchmark.py      backend-neutral workload manifest and JSON benchmark CLI
```

Focused tests:

```text
recipes/evo2_megatron/tests/bionemo/evo2/vllm/
  conftest.py
  test_plugin.py
  test_config.py
  test_packed_fir.py
  test_packed_iir.py
  test_weights.py
  test_export.py
  test_layers.py
  test_hyena.py
  test_model.py
  test_accuracy.py
  test_benchmark.py
```

Integration and evidence:

```text
recipes/evo2_phage_gen/configs/gdpo_phage_vllm.yaml
recipes/evo2_phage_gen/tests/bionemo/evo2_phage_gen/test_configs.py
recipes/evo2_phage_gen/tests/bionemo/evo2_phage_gen/test_rl_readiness.py
docs/evo2-vllm/tp-dp-contract.md
docs/evo2-vllm/mcore-pr5274-api-review.md
docs/evo2-vllm/requirement-audit.md
/data/jstjohn/evo2-vllm-lab/baseline/
/data/jstjohn/evo2-vllm-lab/artifacts/
```

---

### Task 1: Reproducible Runtime And Lazy Plugin

**Files:**
- Modify: `recipes/evo2_megatron/pyproject.toml`
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/__init__.py`
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/plugin.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_plugin.py`

**Interfaces:**
- Consumes: vLLM 0.20.0 `ModelRegistry.register_model(str, str)`.
- Produces: `register() -> None`, entry point `evo2 = bionemo.evo2.vllm.plugin:register`, optional extra `vllm = ["vllm==0.20.0"]`.

- [ ] **Step 1: Create the vLLM environment and immutable manifest**

Run:

```bash
cd /data/jstjohn/evo2-vllm-lab/nemo-rl
uv sync --locked --extra vllm
uv pip install --python .venv/bin/python --no-deps -e /data/jstjohn/evo2-vllm-lab/bionemo-recipes/recipes/evo2_megatron
.venv/bin/python -c 'import json, platform, torch, vllm; print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "vllm": vllm.__version__, "cuda": torch.version.cuda}, sort_keys=True))' > /data/jstjohn/evo2-vllm-lab/artifacts/environment.json
```

Expected: the JSON records vLLM `0.20.0`, PyTorch `2.11.x`, CUDA, and Python `3.13.x`.

- [ ] **Step 2: Write the failing lazy-registration test**

```python
def test_register_is_lazy_and_idempotent(monkeypatch):
    calls = []
    supported = set()
    monkeypatch.setattr("vllm.ModelRegistry.get_supported_archs", lambda: supported)

    def record(name, target):
        calls.append((name, target))
        supported.add(name)

    monkeypatch.setattr("vllm.ModelRegistry.register_model", record)
    from bionemo.evo2.vllm.plugin import register

    register()
    register()
    assert calls == [("Evo2ForCausalLM", "bionemo.evo2.vllm.model:Evo2ForCausalLM")]
```

- [ ] **Step 3: Verify the test is red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_plugin.py -q`

Expected: collection fails because `bionemo.evo2.vllm.plugin` does not exist.

- [ ] **Step 4: Implement lazy idempotent registration and package metadata**

```python
def register() -> None:
    from vllm import ModelRegistry

    if "Evo2ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Evo2ForCausalLM",
            "bionemo.evo2.vllm.model:Evo2ForCausalLM",
        )
```

Extend the existing optional-dependency table and add the entry point in `recipes/evo2_megatron/pyproject.toml` only:

```toml
[project.optional-dependencies]
vllm = ["vllm==0.20.0"]

[project.entry-points."vllm.general_plugins"]
evo2 = "bionemo.evo2.vllm.plugin:register"
```

Do not add a phage destination directory: `recipes/evo2_phage_gen/src/bionemo/evo2` is already a symlink to the Evo2 source tree.

- [ ] **Step 5: Verify registration, copied files, and import isolation**

Run:

```bash
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_plugin.py -q
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/python -c 'from importlib.metadata import entry_points; assert any(ep.name == "evo2" for ep in entry_points(group="vllm.general_plugins"))'
```

Expected: all commands pass and importing `bionemo.evo2.vllm` does not initialize CUDA.

- [ ] **Step 6: Commit**

```bash
git add recipes/evo2_megatron
git commit -m "feat: register optional Evo2 vLLM plugin"
```

### Task 2: Evo2 Configuration And Uniform State Shapes

**Files:**
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/config.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_config.py`
- Modify: `recipes/evo2_megatron/src/bionemo/evo2/vllm/plugin.py`
- Modify: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_plugin.py`

**Interfaces:**
- Consumes: Transformers `AutoConfig.register`.
- Produces: `Evo2Config(PretrainedConfig)`, `operator_types: tuple[str, ...]`, `layers_block_type: list[str]`, `local_state_shapes(tp_size: int) -> tuple[tuple[int, int], tuple[int, int]]`.
- State shapes: projection `(3 * hidden_size // tp_size, 2)` and padded operator `(hidden_size // tp_size, 127)`.

- [ ] **Step 1: Write red tests for 1B, 7B, validation, and TP shapes**

```python
@pytest.mark.parametrize(
    ("pattern", "layers"),
    [("SDH*SDHSDH*SDHSDH*SDHSDH*", 25), ("SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*", 32)],
)
def test_layer_pattern(pattern, layers):
    config = Evo2Config(hidden_size=4096, num_hidden_layers=layers, hybrid_override_pattern=pattern)
    assert len(config.operator_types) == layers
    assert config.layers_block_type == ["attention" if symbol == "*" else "mamba" for symbol in pattern]

def test_tp2_state_shapes_are_uniform():
    config = Evo2Config(hidden_size=4096, num_hidden_layers=4, hybrid_override_pattern="SDH*")
    assert config.local_state_shapes(2) == ((6144, 2), (2048, 127))
```

Also assert rejection of an incorrect pattern length, unknown symbols, and hidden sizes not divisible by TP. Model-level tests in Task 10 reject prefix caching and speculative decoding because those are vLLM runtime settings, not checkpoint config fields.

Extend `test_plugin.py` with a registration test that calls `register()` and verifies `AutoConfig.for_model("evo2")` constructs `Evo2Config` without importing the vLLM model implementation.

- [ ] **Step 2: Verify red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_config.py -q`

Expected: import fails for missing `Evo2Config`.

- [ ] **Step 3: Implement the serializable config and register it lazily**

```python
class Evo2Config(PretrainedConfig):
    model_type = "evo2"

    def __init__(self, *, hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
                 intermediate_size=11008, vocab_size=512, max_position_embeddings=8192,
                 hybrid_override_pattern="SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*",
                 short_conv_length=3, hcs_filter_length=7, hcm_filter_length=128,
                 hcl_state_size=16, num_groups_hcs=256, num_groups_hcm=256,
                 rms_norm_eps=1e-6, rotary_base=10000.0, **kwargs):
        super().__init__(architectures=["Evo2ForCausalLM"], tie_word_embeddings=True, **kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hybrid_override_pattern = hybrid_override_pattern
        self.short_conv_length = short_conv_length
        self.hcs_filter_length = hcs_filter_length
        self.hcm_filter_length = hcm_filter_length
        self.hcl_state_size = hcl_state_size
        self.num_groups_hcs = num_groups_hcs
        self.num_groups_hcm = num_groups_hcm
        self.rms_norm_eps = rms_norm_eps
        self.rotary_base = rotary_base
        self._validate()
        self.operator_types = tuple(hybrid_override_pattern)
        self.layers_block_type = ["attention" if kind == "*" else "mamba" for kind in self.operator_types]

    def _validate(self):
        if len(self.hybrid_override_pattern) != self.num_hidden_layers:
            raise ValueError("hybrid_override_pattern length must equal num_hidden_layers")
        invalid = set(self.hybrid_override_pattern) - {"S", "D", "H", "*"}
        if invalid:
            raise ValueError(f"unsupported Evo2 layer symbols: {sorted(invalid)}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    def local_state_shapes(self, tp_size: int):
        if self.hidden_size % tp_size:
            raise ValueError("hidden_size must be divisible by tensor parallel size")
        local_hidden = self.hidden_size // tp_size
        return (3 * local_hidden, self.short_conv_length - 1), (local_hidden, self.hcm_filter_length - 1)
```

Update `plugin.register()` to import `Evo2Config` inside the function and register it with `AutoConfig` idempotently before registering the lazy vLLM model target. This keeps Task 1 independently importable while making installed checkpoint configs discoverable after Task 2.

- [ ] **Step 4: Verify config serialization and tests**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_config.py -q`

Expected: all tests pass and `Evo2Config.from_pretrained(config.save_pretrained(tmp_path))` round-trips every field.

- [ ] **Step 5: Commit**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm/config.py recipes/evo2_megatron/src/bionemo/evo2/vllm/plugin.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_config.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_plugin.py
git commit -m "feat: define Evo2 vLLM configuration"
```

### Task 3: Boundary-Safe Packed FIR Reference

**Files:**
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/packed_fir.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_fir.py`

**Interfaces:**
- Produces: `packed_fir_reference(x, weight, bias, state_cache, query_start_loc, state_indices, has_initial_state, *, group_size, gated_bias, flip_filter) -> torch.Tensor`.
- Layout: `x/output [total_tokens, channels]`, `weight [filters, taps]`, `state_cache [blocks, channels, taps - 1]`.
- Side effect: update only valid nonzero state blocks in place; preserve null block and all unrelated blocks.

- [ ] **Step 1: Write the scalar-oracle tests**

Construct lengths `list(range(4, 13)) * 10 + [4, 5, 6, 7, 8, 9]`, reverse valid state slots, insert repeated null slot `0`, and compare packed output/state with 96 independent calls to the existing `step_fir` token loop. Parameterize taps `3`, `7`, and `128`, empty versus populated initial state, filter sharing (`group_size=1` and `16`), gated bias, and flipped HCM filters. Fill cache padding and unrelated blocks with finite nonzero sentinels and assert exact preservation.

- [ ] **Step 2: Verify red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_fir.py -q`

Expected: import fails for missing `packed_fir_reference`.

- [ ] **Step 3: Implement the reference with explicit segment ownership**

```python
def packed_fir_reference(x, weight, bias, state_cache, query_start_loc, state_indices,
                         has_initial_state, *, group_size=1, gated_bias=False, flip_filter=False):
    output = torch.empty_like(x)
    taps = weight.shape[-1]
    for request_index in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[request_index])
        end = int(query_start_loc[request_index + 1])
        slot = int(state_indices[request_index])
        initial = state_cache[slot].clone() if slot and bool(has_initial_state[request_index]) else x.new_zeros((x.shape[1], taps - 1), dtype=torch.float32)
        history = initial
        for token_index in range(start, end):
            current = x[token_index].float()
            window = torch.cat((history, current[:, None]), dim=-1)
            filters = weight.repeat_interleave(group_size, dim=0).float()
            if flip_filter:
                filters = filters.flip(-1)
            value = (window * filters).sum(-1)
            if bias is not None:
                value = value + (bias.float() * current if gated_bias else bias.float())
            output[token_index] = value.to(x.dtype)
            history = window[:, 1:]
        if slot:
            state_cache[slot].copy_(history)
    return output
```

Keep this function test-only/diagnostic by documenting that production paths call the CUDA dispatch.

- [ ] **Step 4: Verify reference equivalence and source independence**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_fir.py -q`

Expected: all CPU and CUDA reference cases pass; the tests import the Megatron engine only as the independent oracle.

- [ ] **Step 5: Commit**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm/packed_fir.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_fir.py
git commit -m "test: specify packed Evo2 FIR semantics"
```

### Task 4: Triton Packed FIR Prefill And Decode

**Files:**
- Modify: `recipes/evo2_megatron/src/bionemo/evo2/vllm/packed_fir.py`
- Modify: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_fir.py`

**Interfaces:**
- Produces: `packed_causal_fir(..., max_query_len: int) -> torch.Tensor` with the Task 3 arguments and identical mutation semantics.
- Dispatch regimes: direct segmented Triton for decode and query lengths up to 32; direct segmented or length-bucketed convolution for longer HCM prefill, selected by measured crossover.

- [ ] **Step 1: Write failing CUDA equivalence and graph tests**

Use bf16 activation widths `1920`, `4096`, and `12288`, taps `3/7/128`, mixed lengths `4-12`, request reorder, null padding to graph batch `128`, and two consecutive calls with reused slots. Compare output at `rtol=2e-2, atol=2e-2` and fp32 state at `rtol=2e-5, atol=2e-5`. Capture a decode call in `torch.cuda.CUDAGraph`, replay with changed input values, and prove output and state change without pointer changes.

- [ ] **Step 2: Verify red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_fir.py -q`

Expected: `packed_causal_fir` is missing.

- [ ] **Step 3: Add one segmented kernel launch per FIR operation**

Implement a grid `(num_requests, cdiv(channels, BLOCK_C))`. Each program loads its request's start/end and nonzero cache slot, loops `MAX_QUERY_LEN` under a token mask, accumulates taps in fp32, writes real token outputs, and writes the terminal ring only when `slot != 0`. Select `KERNEL_SIZE` and `MAX_QUERY_LEN` as compile-time constants; pass `group_size` so channel `c` reads filter `c // group_size`. Allocate output before entering the custom op so graph replay performs no Python allocation.

- [ ] **Step 4: Add measured HCM crossover without semantic padding**

Benchmark direct K128 against requests bucketed by power-of-two length and a depthwise convolution. Retain the bucket path only when its median is at least 5% faster for three repeated measurements. Gather output at real positions and update state from each request's true endpoint; a padded endpoint is never authoritative.

- [ ] **Step 5: Verify tests and microbenchmark**

Run:

```bash
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_fir.py -q
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/python -m bionemo.evo2.vllm.packed_fir --benchmark --batch-size 96 --prompt-lengths 4:12 --channels 4096 --taps 3,7,128 --output /data/jstjohn/evo2-vllm-lab/artifacts/packed-fir.json
```

Expected: tests pass, no per-request kernel launch appears, and JSON records direct/bucket medians plus the selected path.

- [ ] **Step 6: Commit**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm/packed_fir.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_fir.py
git commit -m "feat: add segmented Triton FIR kernels"
```

### Task 5: Boundary-Safe Modal HCL Reference

**Files:**
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/packed_iir.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_iir.py`

**Interfaces:**
- Produces: `packed_iir_reference(recurrent_input, gate, decay, residues, diagonal, state_cache, query_start_loc, state_indices, has_initial_state, *, state_size=16) -> torch.Tensor`.
- Layout: token tensors `[total_tokens, channels]`, coefficients `[channels, 16]`, diagonal `[channels]`, padded cache `[blocks, channels, 127]`.
- Recurrence: `state = decay * state + recurrent_input`, `output = gate * (sum(residues * state) + diagonal * recurrent_input)`.

- [ ] **Step 1: Write failing scalar-oracle tests**

Use 96 mixed segments of lengths 4-12, random populated states, reversed slots, repeated block `0`, and nonzero sentinels in cache columns 16-126. Compare each token and final state to the existing `engine.step_iir` scalar loop after applying Evo2's actual `x2 * v` recurrent input and `x1` output gate. Assert one packed call equals two chunked calls at every split position from 1 through 11.

- [ ] **Step 2: Verify red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_iir.py -q`

Expected: import fails for missing `packed_iir_reference`.

- [ ] **Step 3: Implement the independent reference**

```python
def packed_iir_reference(recurrent_input, gate, decay, residues, diagonal, state_cache,
                         query_start_loc, state_indices, has_initial_state, *, state_size=16):
    output = torch.empty_like(recurrent_input)
    for request_index in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[request_index])
        end = int(query_start_loc[request_index + 1])
        slot = int(state_indices[request_index])
        state = state_cache[slot, :, :state_size].clone() if slot and bool(has_initial_state[request_index]) else torch.zeros_like(state_cache[0, :, :state_size])
        for token_index in range(start, end):
            drive = recurrent_input[token_index].float()
            state.mul_(decay).add_(drive[:, None])
            mixed = (residues * state).sum(-1) + diagonal.float() * drive
            output[token_index] = (gate[token_index].float() * mixed).to(output.dtype)
        if slot:
            state_cache[slot, :, :state_size].copy_(state)
    return output
```

- [ ] **Step 4: Verify reference and padding preservation**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_iir.py -q`

Expected: all output/state/chunking cases pass and columns 16-126 are bitwise unchanged.

- [ ] **Step 5: Commit**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm/packed_iir.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_iir.py
git commit -m "test: specify packed Evo2 modal recurrence"
```

### Task 6: Triton HCL Prefill And Decode

**Files:**
- Modify: `recipes/evo2_megatron/src/bionemo/evo2/vllm/packed_iir.py`
- Modify: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_iir.py`

**Interfaces:**
- Produces: `packed_modal_iir(..., max_query_len: int) -> torch.Tensor` with Task 5 semantics.
- Production short path: one segmented Triton launch for query lengths up to 32 and decode.
- Production long path: exact/power-of-two request buckets using the existing modal FFT with final state gathered at the real endpoint.

- [ ] **Step 1: Write failing CUDA, compile, and graph tests**

Parameterize channels `1920/4096`, fp32 order `16`, bf16 token tensors, query lengths `1` and mixed `4-12`, cache reorder/null padding, and two generations with slot reuse. Compare output at `rtol=2e-2, atol=2e-2`, state at `rtol=2e-5, atol=2e-5`, eager versus `torch.compile`, and eager versus CUDA-graph replay.

- [ ] **Step 2: Verify red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_iir.py -q`

Expected: `packed_modal_iir` is missing.

- [ ] **Step 3: Implement the fixed-coefficient segmented kernel**

Use grid `(num_requests, cdiv(channels, BLOCK_C))`; load the 16 fp32 state values per channel, iterate at most `MAX_QUERY_LEN` real tokens under masks, fuse recurrence, residue reduction, diagonal term, gate, output store, and terminal state store. Treat `slot == 0` as zero initial state and suppress its terminal write. Do not expand residues over tokens and do not allocate per-request B/C tensors.

- [ ] **Step 4: Add long-prefill bucketing with true-endpoint state**

Group request indices by exact or power-of-two length on the CPU metadata path, gather only each bucket's token tensor, invoke modal FFT, scatter real outputs, and calculate the terminal modal state at `length - 1`. Prove with a test that adding 64 padded zeros to a request neither changes its real outputs nor decays the stored terminal state.

- [ ] **Step 5: Verify and profile HCL**

Run:

```bash
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_iir.py -q
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/python -m bionemo.evo2.vllm.packed_iir --benchmark --batch-size 96 --prompt-lengths 4:12 --channels 4096 --state-size 16 --output /data/jstjohn/evo2-vllm-lab/artifacts/packed-iir.json
```

Expected: tests pass, the short workload uses one kernel launch per HCL layer, and no token-sized residues temporary is present.

- [ ] **Step 6: Commit**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm/packed_iir.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_packed_iir.py
git commit -m "feat: add segmented Triton HCL recurrence"
```

### Task 7: MBridge-Compatible Weight Loading And Export

**Files:**
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/weights.py`
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/export.py`
- Modify: `recipes/evo2_megatron/pyproject.toml`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_weights.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_export.py`

**Interfaces:**
- Produces: `load_evo2_weights(model: nn.Module, weights: Iterable[tuple[str, Tensor]]) -> set[str]`.
- Produces: `refresh_derived_filters(module_names: set[str]) -> None` and CLI `evo2_export_mbridge_to_vllm`.
- Accepts startup names in native vLLM safetensors, MBridge names such as `decoder.layers.0.mixer...`, and existing Vortex names such as `blocks.0.filter...`.

- [ ] **Step 1: Write failing mapping and refit-order tests**

Create a four-layer `SDH*` synthetic state dict containing embedding, fused QKV, fused FC1, HCS, HCM `h/decay`, HCL `p/gamma/R`, norms, biases, and output projection. Load it in normal and reversed chunk order. Assert TP1 exact tensors, TP2 axis-0 shards, tied embedding/output ownership, complete consumed-key sets, and identical derived buffers:

```python
expected_hcm = h[:, :128].float() * decay[:, :128].float()
expected_modal_decay = torch.exp(-torch.exp(p.float() + gamma.float()))
```

Assert an unknown non-passthrough key and any missing mandatory key raises with its full name.

- [ ] **Step 2: Verify red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_weights.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_export.py -q`

Expected: imports fail for the missing weight loader and exporter.

- [ ] **Step 3: Implement stable target names and TP loaders**

Keep model parameters named after MBridge where vLLM does not impose packing. Use vLLM parameter `weight_loader` attributes for embeddings, `MergedColumnParallelLinear`, `QKVParallelLinear`, and row-parallel outputs. Map TE-fused layernorm names to each layer's explicit RMSNorm. Preserve HCM `h/decay` and HCL `p/gamma/R` as parameters; register nonpersistent fp32 `effective_filter` and `modal_decay` buffers.

- [ ] **Step 4: Refresh derived buffers after either source changes**

```python
def refresh_hcm(module):
    module.effective_filter.copy_(
        module.filter.h[:, : module.kernel_size].float()
        * module.filter.decay[:, : module.kernel_size].float()
    )

def refresh_hcl(module):
    module.modal_decay.copy_(
        torch.exp(-torch.exp(module.filter.p.float() + module.filter.gamma.float()))
    )
```

Call the matching refresh after loading either member of a derived pair. This makes every chunk order converge to the same final buffer before NeMo-RL resumes generation and keeps conversion launches out of decode.

- [ ] **Step 5: Implement streaming export**

Reuse `load_mbridge_state_dict` metadata iteration, write target tensors into bounded safetensors shards, emit `config.json` with architecture `Evo2ForCausalLM`, and emit `model.safetensors.index.json`. Add script entry point:

```toml
evo2_export_mbridge_to_vllm = "bionemo.evo2.vllm.export:main"
```

The exporter records source checkpoint path, source iteration, SHA256 of config/index, model provider, dtype, and converter git commit in `manifest.json`.

- [ ] **Step 6: Verify loader, export, and bounded memory**

Run:

```bash
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_weights.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_export.py -q
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/python -m bionemo.evo2.vllm.export --help
```

Expected: tests pass; synthetic export round-trips; peak host memory stays below 1.5 times the largest input shard plus output shard.

- [ ] **Step 7: Commit**

```bash
git add recipes/evo2_megatron
git commit -m "feat: load and export Evo2 vLLM weights"
```

### Task 8: vLLM Attention, MLP, Norm, And Residual Layers

**Files:**
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/layers.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_layers.py`

**Interfaces:**
- Produces: `Evo2Attention`, `Evo2MLP`, `Evo2AttentionDecoderLayer`, and shared `apply_pre_norm_residual` behavior.
- Uses vLLM `QKVParallelLinear`, `RowParallelLinear`, `MergedColumnParallelLinear`, `RMSNorm`, `Attention`, and rotary embedding.

- [ ] **Step 1: Write failing deterministic layer tests**

Build tiny dimensions `hidden=64`, `heads=4`, `intermediate=128`, zero dropout, no QKV bias, and output projection bias enabled only when present in source. Copy random weights into an independent dense PyTorch implementation and compare prefill hidden states, residuals, QKV rotary application, SwiGLU output, and final gradients-disabled dtype behavior at `rtol=2e-2, atol=2e-2`.

- [ ] **Step 2: Verify red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_layers.py -q`

Expected: import fails for missing layer classes.

- [ ] **Step 3: Implement Evo2 attention using vLLM primitives**

Use equal Q/K/V head counts, head dimension `hidden_size // num_attention_heads`, rotary base from config, causal `Attention`, and row-parallel output projection. Return `(hidden_states, residual)` with the same pre-norm and two-residual ordering as `HyenaLayer`/Megatron transformer layers.

- [ ] **Step 4: Implement fused FC1 SwiGLU and FC2**

Use `MergedColumnParallelLinear(hidden_size, [intermediate_size, intermediate_size])`; split gate/up locally, compute `silu(gate) * up`, and use `RowParallelLinear(intermediate_size, hidden_size, input_is_parallel=True)`.

- [ ] **Step 5: Verify TP1 layer parity and compile support**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_layers.py -q`

Expected: dense-reference and `torch.compile` tests pass.

- [ ] **Step 6: Commit**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm/layers.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_layers.py
git commit -m "feat: add Evo2 vLLM decoder layers"
```

### Task 9: Hyena Mixer And vLLM Recurrent Cache Adapter

**Files:**
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/hyena.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_hyena.py`

**Interfaces:**
- Produces: `Evo2HyenaMixer(MambaBase, PluggableLayer)`, `Evo2HyenaDecoderLayer`, custom op `torch.ops.bionemo_evo2.hyena_mixer`.
- Cache states: projection `[blocks, 3 * local_hidden, 2]`, operator `[blocks, local_hidden, 127]`; logical operator widths HCS `6`, HCM `127`, HCL `16`.
- Metadata: vLLM `Mamba1AttentionMetadata` with decode tokens first and packed prefill tokens second.

- [ ] **Step 1: Write failing mixer parity tests**

For each symbol `S/D/H`, compare a tiny mixer against the existing Megatron/Vortex equations for: independent prefill; prefill split into chunks; 96 mixed prompt lengths; decode after prefill; a mixed scheduler call containing decode and prefill; reversed cache slots; null graph padding; and operator-cache padding sentinels. Assert that internal token order places decode before prefill and that engine output restores request order.

- [ ] **Step 2: Verify red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_hyena.py -q`

Expected: import fails for missing `Evo2HyenaMixer`.

- [ ] **Step 3: Construct TP-aware projections and recurrent parameters**

Use `MergedColumnParallelLinear(hidden_size, [hidden_size, hidden_size, hidden_size])`, projection FIR over all three local streams, split into `x1/x2/v`, dispatch HCS over `x2*v` then gate by `x1`, dispatch HCM with flipped effective filter and gated diagonal then gate by `x1`, dispatch HCL with recurrent input `x2*v` and gate `x1`, and finish with `RowParallelLinear(local_hidden -> hidden_size, input_is_parallel=True)`.

- [ ] **Step 4: Register an opaque graph-safe custom op**

Follow vLLM's `mamba_mixer` pattern: place the layer in `compilation_config.static_forward_context`, initialize `kv_cache` to two empty tensors for later cache binding, allocate caller-owned output, resolve the layer by encoded name inside the custom op, and provide a fake implementation that mutates only output. `forward_impl` reads `Mamba1AttentionMetadata` from `get_forward_context()` and launches packed decode and prefill operations without request loops.

- [ ] **Step 5: Implement uniform state methods**

```python
def get_state_shape(self):
    return self.config.local_state_shapes(get_tensor_model_parallel_world_size())

def get_state_dtype(self):
    return (torch.float32, torch.float32)

@property
def mamba_type(self):
    return "mamba1"
```

Slice only the logical HCS/HCL operator state; never read or write columns beyond that logical width.

- [ ] **Step 6: Verify mixer, graph, and no-loop tests**

Run:

```bash
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_hyena.py -q
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/python -m ruff check recipes/evo2_megatron/src/bionemo/evo2/vllm recipes/evo2_megatron/tests/bionemo/evo2/vllm
```

Expected: parity/graph tests pass and profiler assertions see one packed recurrent call per operation, not 96 calls.

- [ ] **Step 7: Commit**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm/hyena.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_hyena.py
git commit -m "feat: integrate Evo2 Hyena with vLLM cache"
```

### Task 10: Full Hybrid Causal LM And Engine Smoke

**Files:**
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/model.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_model.py`

**Interfaces:**
- Produces: `Evo2Model` and `Evo2ForCausalLM(nn.Module, HasInnerState, IsHybrid)`.
- Produces: `get_mamba_state_shape_from_config`, `get_mamba_state_dtype_from_config`, `get_mamba_state_copy_func`, `compute_logits`, and `load_weights`.
- Uses vLLM `VocabParallelEmbedding`, `ParallelLMHead`, `LogitsProcessor`, `make_layers`, and `support_torch_compile`.

- [ ] **Step 1: Write failing model-structure tests**

Instantiate an `SDH*` tiny config under vLLM's TP1 test context and assert exact layer classes/order, tied vocabulary behavior, final RMSNorm, uniform recurrent shapes, fp32 state dtype, profile-forward shape, and rejection when prefix caching or speculative decoding is requested.

- [ ] **Step 2: Verify model tests are red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_model.py -q`

Expected: import fails for missing `Evo2ForCausalLM`.

- [ ] **Step 3: Implement the hybrid model interface**

```python
@support_torch_compile
class Evo2Model(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        self.embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_prefix: build_evo2_layer(vllm_config, layer_prefix),
            prefix=f"{prefix}.decoder.layers",
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class Evo2ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    has_inner_state = True
    is_hybrid = True
```

Implement forward exactly as vLLM's Jamba model: embeddings on first PP rank, `(hidden_states, residual)` through local layers, final norm on last rank, logits through tied parallel head, and `load_weights` through Task 7.

- [ ] **Step 4: Implement model-level recurrent shape methods**

Return `config.local_state_shapes(tensor_parallel_size)`, `(torch.float32, torch.float32)`, and vLLM's Mamba1 state-copy functions. Do not inherit `SupportsMambaPrefixCaching`; explicitly require `mamba_cache_mode="none"` and `enable_prefix_caching=False` for this release.

- [ ] **Step 5: Write and run a tiny vLLM engine smoke**

Export deterministic tiny weights to a temporary checkpoint, start `vllm.LLM(skip_tokenizer_init=True, enforce_eager=True, enable_prefix_caching=False)`, and generate four requests with lengths `4, 7, 9, 12`. Assert each request appears once in input order, greedy replay is exact, logprobs are finite, and two requests sharing the same suffix do not share recurrent state.

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_model.py -q`

Expected: model and in-process vLLM V1 engine tests pass.

- [ ] **Step 6: Repeat engine smoke with compile and CUDA graphs**

Run the same test with `enforce_eager=False`, warm twice, generate twice, and inspect logs/metrics to prove graph capture and replay occurred. Assert no graph recapture on the second identical 96-request decode shape.

- [ ] **Step 7: Commit**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm/model.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_model.py
git commit -m "feat: serve Evo2 through vLLM"
```

### Task 11: Real 1B And Base 7B Accuracy Gates

**Files:**
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/accuracy.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_accuracy.py`
- Modify: `recipes/evo2_megatron/tests/bionemo/evo2/test_evo2.py`
- Modify: `recipes/evo2_megatron/tests/bionemo/evo2/run/test_infer.py`

**Interfaces:**
- Produces JSON records containing prompt tokens, generated tokens, first-step logits summary, processed logprobs, sequence identity, topology, dtype, checkpoint manifest hash, and seed.
- Environment manifest defines `EVO2_1B_MBRIDGE_CHECKPOINT` and `EVO2_7B_MICROVIRIDAE_BASE_CHECKPOINT`; tests skip only when explicitly marked non-model CI, while release commands require both.

- [ ] **Step 1: Record checkpoint identity before export**

Run the exporter for the real 1B fixture and base 7B Microviridae checkpoint and save each `manifest.json` under `/data/jstjohn/evo2-vllm-lab/artifacts/checkpoints/`. Verify neither source path contains `step_`, `rl`, `grpo`, or `gdpo` for the 7B base model; compare its metadata to the GDPO config's declared pretrained path.

- [ ] **Step 2: Write failing first-token and identity tests**

For fixed biological prompts, compare vLLM against the existing Vortex/MCore path for close first-token logits and processed logprobs. Port the current 1B second-half golden fixture so both backends generate the same repeated greedy token IDs and compute second-half identity with the existing helper, not a new metric implementation.

- [ ] **Step 3: Verify the tests expose numerical differences**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_accuracy.py -q -m model`

Expected before final tuning: tests either fail with a localized layer/logit mismatch or pass; any failure record includes the first divergent layer/token and maximum absolute/relative error.

- [ ] **Step 4: Localize and fix mismatches without weakening gates**

Use forward hooks at embedding, every decoder output, final norm, and logits. Correct weight orientation, rotary convention, gate order, bias mode, or recurrent state update at the first divergent boundary. Keep first-step bf16 logit tolerance at `rtol=2e-2, atol=2e-2`, fp32 state tolerance at `2e-5`, and existing second-half identity threshold unchanged.

- [ ] **Step 5: Add stochastic sampling and replay checks**

Run temperature/top-k/top-p combinations used by GDPO with a fixed seed. Assert exact replay for the same seed/topology, changed tokens for a different seed, finite processed logprobs for every generated token, and stop-token inclusion matching NeMo-RL.

- [ ] **Step 6: Run and persist real accuracy evidence**

Run:

```bash
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/python -m bionemo.evo2.vllm.accuracy --checkpoint "$EVO2_1B_MBRIDGE_CHECKPOINT" --topology tp1 --output /data/jstjohn/evo2-vllm-lab/artifacts/accuracy-1b.json
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/python -m bionemo.evo2.vllm.accuracy --checkpoint "$EVO2_7B_MICROVIRIDAE_BASE_CHECKPOINT" --topology tp2 --output /data/jstjohn/evo2-vllm-lab/artifacts/accuracy-7b-tp2.json
```

Expected: no gate is below the existing identity threshold; vLLM biological identity is within five percentage points of serial MCore; first-step logits/logprobs pass the fixed tolerances.

- [ ] **Step 7: Commit**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm/accuracy.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_accuracy.py recipes/evo2_megatron/tests/bionemo/evo2/test_evo2.py recipes/evo2_megatron/tests/bionemo/evo2/run/test_infer.py
git commit -m "test: validate real Evo2 vLLM accuracy"
```

### Task 12: Stock NeMo-RL Refit And GDPO Smoke

**Files:**
- Create: `recipes/evo2_phage_gen/configs/gdpo_phage_vllm.yaml`
- Modify: `recipes/evo2_phage_gen/tests/bionemo/evo2_phage_gen/test_configs.py`
- Modify: `recipes/evo2_phage_gen/tests/bionemo/evo2_phage_gen/test_rl_readiness.py`
- Modify: `recipes/evo2_phage_gen/src/bionemo/evo2_phage_gen/rl_readiness.py`
- Modify: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_model.py`

**Interfaces:**
- Uses stock `VllmInternalWorkerExtension._load_weights`, `prepare_refit_info`, sleep/wake, `logprobs_mode="processed_logprobs"`, and pretokenized `prompt_token_ids`.
- Config starts training from the base Microviridae checkpoint and uses GDPO with batch-wise normalization, the current objective components, and KL `0.001`.

- [ ] **Step 1: Write a failing configuration contract test**

Assert the new YAML has `policy.generation.backend: vllm`, `generation_batch_size: 96`, `skip_tokenizer_init: true`, `enable_prefix_caching: false`, `enforce_eager: false`, `load_format: dummy` for refit smoke, no speculative config, base Microviridae pretrained path, and no prior RL checkpoint. Assert validation and training batch values are explicit in YAML rather than command-only overrides.

- [ ] **Step 2: Verify red**

Run: `recipes/evo2_phage_gen/.venv/bin/pytest recipes/evo2_phage_gen/tests/bionemo/evo2_phage_gen/test_configs.py recipes/evo2_phage_gen/tests/bionemo/evo2_phage_gen/test_rl_readiness.py -q`

Expected: YAML is missing.

- [ ] **Step 3: Add the vLLM GDPO config and readiness checks**

Derive all reward/GDPO/training fields from the current stock-GDPO YAML and change only generation backend fields, vLLM worker environment, and checkpoint export path. Add readiness checks for vLLM `0.20.0`, plugin entry point visibility, disabled prefix/speculation, checkpoint manifest, and two free GPUs.

- [ ] **Step 4: Test stock streamed refit directly**

Start the tiny engine with dummy weights, obtain training-style `state_dict_info`, call the stock NeMo-RL worker extension with two differently ordered weight chunks, generate once, change embedding plus one HCS/HCM/HCL parameter each, refit, sleep/wake, and generate again. Assert all changed parameters and derived buffers have new values, generated logits change, graph pointers remain valid, and old recurrent state is absent.

- [ ] **Step 5: Run one validation and one training-step smoke**

Launch in a new named tmux session from the phage recipe environment with `grpo.max_num_steps=1`, `grpo.val_at_start=true`, and the vLLM YAML. Keep the base checkpoint, full GDPO objective, real QC/scoring, generation batch 96, and production sequence length. Persist stdout, W&B-disabled local metrics, GPU telemetry, and checkpoint output under `/data/jstjohn/evo2-vllm-lab/artifacts/gdpo-smoke/`.

Expected: validation, generation, reward, policy/reference logprobs, advantage calculation, and one optimizer step complete; the second rollout after refit uses changed weights.

- [ ] **Step 6: Verify no NeMo-RL fork was required**

Run: `git -C /data/jstjohn/evo2-vllm-lab/nemo-rl status --short`

Expected: empty output. If a genuine generic NeMo-RL defect blocks the stock path, record a minimal upstream patch separately and keep the Evo2 implementation functional against an unmodified checkout.

- [ ] **Step 7: Commit**

```bash
git add recipes/evo2_phage_gen/configs/gdpo_phage_vllm.yaml recipes/evo2_phage_gen/tests/bionemo/evo2_phage_gen/test_configs.py recipes/evo2_phage_gen/tests/bionemo/evo2_phage_gen/test_rl_readiness.py recipes/evo2_phage_gen/src/bionemo/evo2_phage_gen/rl_readiness.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_model.py
git commit -m "feat: run Evo2 GDPO with vLLM"
```

### Task 13: TP2, DP2, And Multi-Axis Contract

**Files:**
- Modify: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_model.py`
- Create: `docs/evo2-vllm/tp-dp-contract.md`

**Interfaces:**
- TP2/DP1: one vLLM engine with two TP workers and channel-sharded parameters/states.
- TP1/DP2: two NeMo-RL vLLM engines, 48 requests per replica, independent cache slots/seeds, one global ordered result.
- TP2/DP2 design: four ranks in TP groups `{0,1}` and `{2,3}`, DP replicas across those groups, distinct refit rank prefixes and deterministic global-request seeds.

- [ ] **Step 1: Write distributed result validators**

For a manifest with stable `request_id`, assert every ID appears exactly once, global order matches input order after gather, each generated length/logprob aligns with its request, and repeated runs with the same global seed are exact. Assert TP2 first-token logits match TP1 and DP2 outputs match a serial reference within fixed tolerances.

- [ ] **Step 2: Run TP2/DP1 on both H100s**

Run the 7B accuracy command with tensor parallel size 2 and save worker logs, NCCL topology, per-rank peak memory, and output JSON under `artifacts/distributed/tp2/`.

Expected: both GPUs sustain compute, each owns half the channel-sharded Hyena state/weights, and output passes Task 11 gates.

- [ ] **Step 3: Run TP1/DP2 on both H100s**

Start two NeMo-RL engines with one GPU each, shard the 96-request manifest into deterministic 48-request sets, generate concurrently, and gather once by `request_id`. Save replica seeds, worker/device mapping, timings, and outputs under `artifacts/distributed/dp2/`.

Expected: both GPUs sustain compute concurrently; no request duplication/loss; aggregate output passes Task 11 gates.

- [ ] **Step 4: Test refit in both executable topologies**

Run two rollout/refit cycles for TP2 and DP2. Assert every TP shard and DP replica receives the new weight version, emits changed logits, and has empty recurrent state for new request IDs.

- [ ] **Step 5: Document TP2/DP2 without overstating hardware proof**

Specify rank groups, model/state channel shards, per-replica scheduler/cache, source-to-destination refit rank mapping, seed formula `seed + global_request_id`, and ordered result gather. Label the four-GPU topology `designed, not executed on the two-GPU host` unless four GPUs are actually available.

- [ ] **Step 6: Commit**

```bash
git add recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_model.py docs/evo2-vllm/tp-dp-contract.md
git commit -m "test: validate Evo2 vLLM TP and DP"
```

### Task 14: Reproducible 96-Genome MCore Versus vLLM Benchmark

**Files:**
- Create: `recipes/evo2_megatron/src/bionemo/evo2/vllm/benchmark.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_benchmark.py`
- Create: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/data/gdpo_mixed_96.json`

**Interfaces:**
- Produces: `WorkloadManifest`, `BenchmarkSample`, and aggregate JSON with generation seconds, generated tokens/s, requests/s, TTFT, median/p95 inter-token latency, peak allocated/reserved bytes, graph captures, kernel launches, host/device copies, and synchronizations.
- Backends run in separate pinned environments but consume the same tokenized manifest and emit the same schema.

- [ ] **Step 1: Write failing manifest and aggregation tests**

Assert exactly 96 request IDs, prompt lengths all within 4-12, every length represented, stable token IDs/hash, production max-new-token and stop settings, seed, dtype, topology, and base checkpoint hash. Feed synthetic samples with outliers and assert median, p95, MAD/dispersion, and pass/fail calculations.

- [ ] **Step 2: Verify red**

Run: `/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_benchmark.py -q`

Expected: benchmark module and manifest are missing.

- [ ] **Step 3: Implement one benchmark schema for both backends**

Create a subprocess CLI with `--backend mcore|vllm`, `--checkpoint`, `--manifest`, `--topology tp2|dp2`, `--warmups`, `--repetitions`, and `--output`. Time from requests submitted through all responses materialized; separately record prefill/TTFT and steady decode. Reset peak memory before each measured repetition and synchronize only outside the timed interval.

- [ ] **Step 4: Capture immutable MCore baseline**

Use the current production MCore implementation and its working environment, CUDA graphs enabled, the base 7B Microviridae checkpoint, and both homogeneous and mixed 4-12 manifests. Run two warmups and five samples first, then three warmups and ten interleaved final samples. Save raw JSON and logs under `/data/jstjohn/evo2-vllm-lab/baseline/mcore/`.

- [ ] **Step 5: Capture vLLM TP2 and DP2 candidates**

Run the same manifests/settings with compile and CUDA graphs enabled. Interleave candidate samples with MCore final samples to reduce clock/temperature drift. Save raw JSON, clocks, `nvidia-smi dmon`, and logs under `artifacts/benchmark/vllm-tp2/` and `artifacts/benchmark/vllm-dp2/`.

- [ ] **Step 6: Enforce the performance gate**

Choose the faster valid vLLM topology. Pass only when median generated tokens/s and requests/s meet or exceed MCore within measured variance, median and p95 inter-token latency do not regress, peak memory is no more than 5% above MCore, accuracy still passes, and profiles show no per-token host/device sync/copy or serial request fallback.

- [ ] **Step 7: Commit harness and immutable workload**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm/benchmark.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_benchmark.py recipes/evo2_megatron/tests/bionemo/evo2/vllm/data/gdpo_mixed_96.json
git commit -m "bench: compare Evo2 MCore and vLLM rollout"
```

### Task 15: Profile-Guided Throughput Closure

**Files:**
- Modify when selected by evidence: `recipes/evo2_megatron/src/bionemo/evo2/vllm/packed_fir.py`
- Modify when selected by evidence: `recipes/evo2_megatron/src/bionemo/evo2/vllm/packed_iir.py`
- Modify when selected by evidence: `recipes/evo2_megatron/src/bionemo/evo2/vllm/hyena.py`
- Modify when selected by evidence: `recipes/evo2_megatron/src/bionemo/evo2/vllm/model.py`
- Modify: `recipes/evo2_megatron/tests/bionemo/evo2/vllm/test_benchmark.py`

**Interfaces:**
- Consumes: Task 14 raw samples and Nsight Systems traces.
- Produces: a passing 96-request performance gate or a quantified blocker with kernel-level evidence; no lower workload substitutes.

- [ ] **Step 1: Capture one steady decode window and one mixed prefill window**

Run Nsight with CUDA profiler API capture around measured windows only. Record GPU kernel durations, launch gaps, NCCL collectives, graph nodes, allocator activity, CPU scheduling, copies, and synchronizations. Save `.nsys-rep`, SQLite export, command, environment, and sample ID under `artifacts/profiles/`.

- [ ] **Step 2: Apply the first matching decision rule**

Use these rules in order and rerun the focused correctness tests after each accepted change:

| Evidence | Required change | Acceptance check |
|---|---|---|
| HCS/HCM projection plus operator FIR launches consume over 10% of decode GPU time | Add one fused projection-FIR/operator-FIR Triton path in `hyena.py` for S/D | Same outputs/states; decode median improves at least 3% |
| K128 direct FIR consumes over 10% of mixed-prefill time | Select the proven bucketed convolution/FFT crossover in `packed_fir.py` | Same true-endpoint state; mixed-prefill median improves at least 5% |
| HCL recurrence consumes over 10% of decode time | Tune `BLOCK_C`, warps, and state vectorization in `packed_iir.py` | Same fp32 state tolerance; decode median improves at least 3% |
| Host gaps or copies occur once per token | Move metadata/output buffers to persistent graph inputs and remove `.item()`, `.cpu()`, or implicit allocation from decode | No per-token host sync/copy remains |
| CUDA graph recaptures after warmup | Stabilize graph batch sizes at the selected capture sizes including 96/128 and keep cache pointers stable | Zero recaptures in ten measured repetitions |
| TP2 NCCL dominates while DP2 is valid | Select TP1/DP2 as production topology | DP2 passes accuracy/refit and aggregate throughput gate |
| Scheduler leaves either GPU idle in DP2 | Submit both 48-request replica shards concurrently before waiting and gather once | Both GPU-utilization traces overlap for the generation interval |

- [ ] **Step 3: Reject optimizations that trade away behavior**

Discard any change that weakens tolerances, writes cache padding, treats packed requests as one recurrence, adds a per-request decode loop, changes sampling/logprobs, skips refit, requires a vLLM patch, or improves only a tiny/random model.

- [ ] **Step 4: Rerun the complete accuracy and distributed suite**

Run all vLLM tests, 1B identity, 7B base accuracy, TP2, DP2, two-refit cycles, copied-file check, and Ruff. Expected: all prior gates still pass.

- [ ] **Step 5: Rerun the final interleaved benchmark**

Use three warmups and at least ten MCore/candidate repetitions on both homogeneous and mixed prompt manifests. Expected: the selected candidate meets every Task 14 gate. Keep all repetitions, including outliers, in raw evidence.

- [ ] **Step 6: Commit only measured wins**

```bash
git add recipes/evo2_megatron/src/bionemo/evo2/vllm recipes/evo2_megatron/tests/bionemo/evo2/vllm
git commit -m "perf: close Evo2 vLLM rollout throughput"
```

### Task 16: Final Read-Only PR5274 MCore API Advisory

**Files:**
- Create: `docs/evo2-vllm/mcore-pr5274-api-review.md`
- No files under `/data/jstjohn/evo2-mcore-pr5274-lab` may be modified.

**Interfaces:**
- Consumes: the then-current MCore lab commit/API plus the completed vLLM contracts and benchmark evidence.
- Produces: backend-neutral API recommendations categorized as required, useful, or unnecessary for PR5274.

- [ ] **Step 1: Snapshot the MCore lab without interrupting it**

Run read-only `git status --short`, `git log --oneline`, `git diff`, `rg`, and `sed` commands in `/data/jstjohn/evo2-mcore-pr5274-lab`. Record commit IDs and whether uncommitted work exists. Do not send tmux input, stop processes, edit files, or change branches.

- [ ] **Step 2: Map concrete contracts side by side**

Compare MCore and vLLM for:

```text
packed metadata: query_start_loc, request order, active/padded entries
state metadata: projection/operator shape, dtype, logical width, null slot
kernel inputs: x, filters/decays/residues, bias/diagonal, state, boundaries, slots
lifecycle: allocate, prefill, decode, reorder, free, sleep, wake, refit, graph replay
sampling: processed logprob, seed derivation, stop behavior, output ordering
distributed: TP channel shard, DP request shard, refit rank mapping
```

Quote exact local symbols and file/line references for both implementations.

- [ ] **Step 3: Recommend a backend-neutral kernel/state API**

Evaluate and document whether these signatures should live below both backends in `bionemo-evo2`:

```python
packed_fir(x, weight, bias, state, cu_seqlens, state_slots, has_initial_state,
           *, group_size, gated_bias, flip_filter, output)
packed_modal_iir(recurrent_input, gate, decay, residues, diagonal, state,
                 cu_seqlens, state_slots, has_initial_state, *, output)
```

Require explicit mutation, dtype, null-slot, graph-safety, and true-endpoint semantics. Recommend MCore adapters and vLLM adapters own only their scheduler-specific metadata translation.

- [ ] **Step 4: Separate PR5274 recommendations by value**

The report must answer:

1. Which PR5274 state-shape, packed-metadata, projection, kernel, seed, sampling-logprob, and lifecycle contracts should be incorporated or changed for vLLM interoperability?
2. Which APIs remain valuable for Megatron-native inference even if GDPO production moves entirely to vLLM?
3. Which Vortex-specific or backend-specific changes should remain outside MCore?
4. Which duplicated kernels should move beneath both backends, based on correctness and measured performance rather than code aesthetics?
5. Whether the vLLM path simplifies the phage recipe enough to retire MCore-specific rollout patches, with an exact deletion list only for behavior proven redundant.

- [ ] **Step 5: Verify advisory accuracy with the current lab owner state**

Re-read the latest MCore lab status/commit before finalizing in case its API changed during vLLM implementation. Update file/line references; leave the lab untouched.

- [ ] **Step 6: Commit the advisory**

```bash
git add docs/evo2-vllm/mcore-pr5274-api-review.md
git commit -m "docs: advise PR5274 from vLLM results"
```

### Task 17: Final Verification And Requirement Audit

**Files:**
- Create: `docs/evo2-vllm/requirement-audit.md`
- Modify: `docs/superpowers/specs/2026-07-14-evo2-vllm-packed-inference-design.md` only if implementation evidence requires a factual correction; never weaken a gate.

**Interfaces:**
- Produces: one evidence-indexed pass/fail row for every design requirement and user-requested completion gate.

- [ ] **Step 1: Run the focused and copied-file suite from a clean process**

Run:

```bash
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/pytest recipes/evo2_megatron/tests/bionemo/evo2/vllm -q
/data/jstjohn/evo2-vllm-lab/nemo-rl/.venv/bin/python -m ruff check recipes/evo2_megatron/src/bionemo/evo2/vllm recipes/evo2_megatron/tests/bionemo/evo2/vllm
python ci/scripts/check_copied_files.py
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Re-run real-model release gates**

Run 1B second-half identity, 7B base accuracy, stochastic replay, TP2, DP2, two-refit cycles, one-validation/one-training GDPO smoke, and final 96-request benchmarks. Verify every command uses the checkpoint hashes and workload hash recorded in manifests.

- [ ] **Step 3: Audit evidence completeness**

Create a table with columns `Requirement`, `Command`, `Artifact`, `Result`, and `Notes`. Include packed HCS/HCM/HCL boundaries, mixed 4-12 prompts, logprobs, CUDA graphs/compile, null/padding behavior, refit, sleep/wake, TP2, DP2, TP2/DP2 design, full 96 generation timing, accuracy, memory, no vLLM source patch, clean NeMo-RL checkout, and final MCore API review.

- [ ] **Step 4: Scan for shortcuts and unfinished markers**

Run source searches for request loops in decode, `.item()`/`.cpu()` on the graph path, vLLM monkey patches, disabled tests, loosened tolerances, non-base final checkpoints, and untracked evidence. Inspect every hit and record disposition in the audit.

- [ ] **Step 5: Verify repository and external checkout state**

Run `git status --short` in the implementation worktree, NeMo-RL checkout, and vLLM checkout. Expected: implementation changes are committed; NeMo-RL and vLLM are clean; the MCore lab state is unchanged from its read-only snapshots except for work performed independently by its owner.

- [ ] **Step 6: Commit the audit**

```bash
git add docs/evo2-vllm/requirement-audit.md
git commit -m "docs: audit Evo2 vLLM completion evidence"
```

- [ ] **Step 7: Mark the goal complete only after every required row passes**

If any required row lacks direct evidence or fails its fixed gate, continue from the responsible task. A quantified performance or correctness miss is not completion.
