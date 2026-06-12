# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evo2LoRA tests: adapter wiring, freeze patterns, weight-tying validation, and integration.

The 1B model used in TestEvo2LoRAFreeze has a mixed hybrid architecture:
"SDH*SDHSDH*SDHSDH*SDHSDH*" (25 layers).
    - Hyena layers (S/D/H) at indices 0-2, 4-9, 11-16, 18-23
    - Attention layers (*) at indices 3, 10, 17, 24

This means patterns like "linear_qkv" match ONLY in attention layers, while
"dense_projection" matches ONLY in Hyena mixer layers — important edge cases.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from bionemo.evo2.models.evo2_lora import Evo2LoRA
from bionemo.evo2.models.evo2_provider import Hyena1bModelProvider

from ..utils import distributed_model_parallel_state


_PRE_MLP_NORM_NAME = "pre_mlp_layernorm"
_DEFAULT_LORA_TARGETS = ["dense_projection", "linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]


# ── Model construction ────────────────────────────────────────────────────────


def _build_1b_model():
    """Instantiate a real Hyena 1B model (vocab_size=512, use_te=True)."""
    config = Hyena1bModelProvider(
        vocab_size=512,  # matches nucleotide_fast_tokenizer_512
        use_te=True,
    )
    config.finalize()
    return config.provide(pre_process=True, post_process=True).cuda()


# ── Helpers ───────────────────────────────────────────────────────────────────


def reset_trainable(model):
    """Set requires_grad=True on every parameter."""
    for p in model.parameters():
        p.requires_grad = True


def grad_state(model):
    """Return {param_name: requires_grad} for every named parameter."""
    return {n: p.requires_grad for n, p in model.named_parameters()}


def apply_freeze(model, patterns):
    """Reset grad, apply freeze, return grad state."""
    reset_trainable(model)
    Evo2LoRA(skip_freeze_modules=patterns, target_modules=[]).freeze_model(model, training=True)
    return grad_state(model)


def direct_params(model, path):
    """Parameter names that are DIRECT members of the module at path (no recursion)."""
    mod = model
    for part in path.split("."):
        mod = getattr(mod, part)
    return {f"{path}.{n}" for n, _ in mod.named_parameters(recurse=False)}


def subtree_params(model, path):
    """All parameter names anywhere under the module at path."""
    mod = model
    for part in path.split("."):
        mod = getattr(mod, part)
    return {f"{path}.{n}" for n, _ in mod.named_parameters()}


def by_short_name(model, short_name):
    """Direct parameters of every module whose last path segment == short_name."""
    result = set()
    for path, mod in model.named_modules():
        if path and path.split(".")[-1] == short_name:
            result |= {f"{path}.{n}" for n, _ in mod.named_parameters(recurse=False)}
    return result


def assert_grad_state(state, expected_trainable):
    """Raise AssertionError if any parameter's requires_grad differs from expected."""
    errors = []
    for name, rg in state.items():
        if name in expected_trainable and not rg:
            errors.append(f"  SHOULD be trainable but frozen : {name}")
        elif name not in expected_trainable and rg:
            errors.append(f"  SHOULD be frozen  but trainable: {name}")
    if errors:
        raise AssertionError("Grad-state mismatch:\n" + "\n".join(errors))


def _pre_mixer_norm_name(model) -> str:
    """Return the pre-mixer norm module name for Hyena layers (layer 0)."""
    layer0_child_names = {n for n, _ in model.decoder.layers[0].named_children()}
    return "input_layernorm" if "input_layernorm" in layer0_child_names else "norm"


def _word_emb_trainable_expected(model) -> set:
    """Parameters that share identity with embedding.word_embeddings.weight."""
    named_params = dict(model.named_parameters())
    word_emb_param = named_params.get("embedding.word_embeddings.weight")
    return {n for n, p in model.named_parameters() if p is word_emb_param}


# ── Small toy model (used in TestEvo2LoRAAdapterWiring) ──────────────────────


class _MLP(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden, ffn)
        self.linear_fc2 = nn.Linear(ffn, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(torch.relu(self.linear_fc1(x)))


class _SmallModel(nn.Module):
    """Tiny model with nested structure so wildcard patterns like ``*.linear_fc2`` work."""

    def __init__(self, vocab_size: int = 64, hidden: int = 32, ffn: int = 64):
        super().__init__()
        self.embedding = nn.ModuleDict({"word_embeddings": nn.Embedding(vocab_size, hidden)})
        self.mlp = _MLP(hidden, ffn)
        self.output_proj = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embedding.word_embeddings(input_ids)
        return self.output_proj(self.mlp(h))


# ── Tiny Hyena provider (used in weight-tying and integration tests) ──────────


@dataclass
class _TinyHyenaProvider(Hyena1bModelProvider):
    """Smallest viable Hyena config: 2 layers, no activation checkpointing."""

    num_layers: int = 2
    hybrid_override_pattern: str = "SD"
    recompute_granularity: str | None = None
    recompute_method: str | None = None
    recompute_num_layers: int = 0


@dataclass
class _WTTiedProvider(_TinyHyenaProvider):
    """Minimum viable Hyena model with share_embeddings_and_output_weights=True.

    vocab_size is set explicitly so ``provide()`` can be called without going
    through the full recipe/tokenizer setup (which is what normally sets it).
    """

    vocab_size: int = 256  # nucleotide_fast_tokenizer_256 vocab size


@dataclass
class _WTUntiedProvider(_TinyHyenaProvider):
    """Same as _WTTiedProvider but with independent embedding and output weights."""

    vocab_size: int = 256
    share_embeddings_and_output_weights: bool = False


# ── Case specs (used in TestEvo2LoRAFreeze) ───────────────────────────────────
# Each entry: (id, patterns_or_fn, expected_fn)
#   patterns_or_fn: list[str]  OR  callable(model) -> list[str]
#   expected_fn:    callable(model) -> set[str]

_CASE_SPECS = [
    # ── empty / no-op ──────────────────────────────────────────────────
    (
        "empty_skip_all_frozen",
        [],
        lambda m: set(),
    ),
    # ── wildcard "*": skip freezing everything ─────────────────────────
    (
        "wildcard_star_nothing_frozen",
        ["*"],
        lambda m: {n for n, _ in m.named_parameters()},
    ),
    # ── exact short-name: word_embeddings ─────────────────────────────
    (
        "exact_word_embeddings",
        ["word_embeddings"],
        _word_emb_trainable_expected,
    ),
    # ── exact short-name: pre-mixer norm (Hyena layers only) ──────────
    (
        "exact_hyena_pre_mixer_norm_hyena_layers",
        lambda m: [_pre_mixer_norm_name(m)],
        lambda m: by_short_name(m, _pre_mixer_norm_name(m)),
    ),
    # ── exact short-name: input_layernorm (Attention layers only) ──────
    (
        "exact_input_layernorm_attention_layers_only",
        ["input_layernorm"],
        lambda m: by_short_name(m, "input_layernorm"),
        # input_layernorm is the pre-mixer norm in Attention layers (3, 10, 17, 24).
        # Hyena layers have no input_layernorm — tests layer-type selectivity.
    ),
    # ── exact short-name: pre-MLP layernorm ───────────────────────────
    (
        "exact_pre_mlp_layernorm_all_layers",
        [_PRE_MLP_NORM_NAME],
        lambda m: by_short_name(m, _PRE_MLP_NORM_NAME),
    ),
    # ── exact short-name: linear_fc1 (Hyena + Attention MLP) ──────────
    (
        "exact_linear_fc1_all_layers",
        ["linear_fc1"],
        lambda m: by_short_name(m, "linear_fc1"),
        # MLP exists in BOTH Hyena layers and Attention layers.
    ),
    # ── exact short-name: linear_fc2 (Hyena + Attention MLP) ──────────
    (
        "exact_linear_fc2_all_layers",
        ["linear_fc2"],
        lambda m: by_short_name(m, "linear_fc2"),
    ),
    # ── Attention-specific: linear_qkv ────────────────────────────────
    (
        "exact_linear_qkv_attention_layers_only",
        ["linear_qkv"],
        lambda m: by_short_name(m, "linear_qkv"),
        # linear_qkv exists ONLY in the 4 attention layers (3, 10, 17, 24).
    ),
    # ── Hyena-specific: dense_projection ──────────────────────────────
    (
        "exact_dense_projection_hyena_layers_only",
        ["dense_projection"],
        lambda m: by_short_name(m, "dense_projection"),
        # dense_projection exists ONLY in the 21 Hyena mixer layers.
    ),
    # ── Hyena-specific: dense (output projection in HyenaMixer) ───────
    (
        "exact_dense_hyena_layers_only",
        ["dense"],
        lambda m: by_short_name(m, "dense"),
        # dense is the output projection linear inside HyenaMixer (RowParallelLinear).
    ),
    # ── composite module: "mixer" matches at multiple depths ──────────
    (
        "exact_mixer_matches_inner_hyena_mixer_direct_params",
        ["mixer"],
        lambda m: by_short_name(m, "mixer"),
        # "mixer" matches every module whose short name is "mixer", at any depth.
        # There are TWO such modules per Hyena layer:
        #   - decoder.layers.X.mixer       (layer-level wrapper, no direct params)
        #   - decoder.layers.X.mixer.mixer (inner HyenaMixer, has conv_bias directly)
        # by_short_name() collects direct params from both.
    ),
    (
        "exact_mlp_composite_no_direct_params",
        ["mlp"],
        lambda m: set(),
        # "mlp" matches the mlp module but it has no direct params (only children do).
    ),
    # ── wildcard: whole-layer subtrees ────────────────────────────────
    (
        "wildcard_layer0_full_subtree",
        ["*.layers.0.*"],
        lambda m: subtree_params(m, "decoder.layers.0"),
    ),
    (
        "wildcard_layer3_attention_full_subtree",
        ["*.layers.3.*"],
        lambda m: subtree_params(m, "decoder.layers.3"),
        # Layer 3 is an Attention (*) layer.
    ),
    (
        "wildcard_last_attention_layer",
        ["*.layers.24.*"],
        lambda m: subtree_params(m, "decoder.layers.24"),
        # The last layer is also an Attention layer.
    ),
    # ── wildcard: single module across depths ─────────────────────────
    (
        "wildcard_layer0_pre_mixer_norm_only",
        lambda m: [f"*.layers.0.{_pre_mixer_norm_name(m)}"],
        lambda m: direct_params(m, f"decoder.layers.0.{_pre_mixer_norm_name(m)}"),
        # Matches only layer 0's norm, not layer 1's.
    ),
    (
        "wildcard_layer0_mixer_children_not_mixer_itself",
        ["*.layers.0.mixer.*"],
        lambda m: subtree_params(m, "decoder.layers.0.mixer"),
        # "*.layers.0.mixer.*" requires an extra segment after "mixer".
    ),
    # ── wildcard: MLP children across all layers ──────────────────────
    (
        "wildcard_mlp_children_all_layers",
        ["*.mlp.*"],
        lambda m: by_short_name(m, "linear_fc1") | by_short_name(m, "linear_fc2"),
        # "*.mlp.*" matches linear_fc1 and linear_fc2 in all 25 layers.
    ),
    # ── wildcard: decoder final norm ──────────────────────────────────
    (
        "wildcard_decoder_final_norm",
        ["decoder.final_norm"],
        lambda m: subtree_params(m, "decoder.final_norm"),
        # Exact full-path match (no wildcards).
    ),
    # ── no match ──────────────────────────────────────────────────────
    (
        "nonexistent_pattern_all_frozen",
        ["does_not_exist"],
        lambda m: set(),
    ),
    # ── multiple patterns (union) ─────────────────────────────────────
    (
        "multiple_word_embeddings_and_linear_qkv",
        ["word_embeddings", "linear_qkv"],
        lambda m: _word_emb_trainable_expected(m) | by_short_name(m, "linear_qkv"),
    ),
    (
        "multiple_layer0_wildcard_and_final_norm",
        ["*.layers.0.*", "final_norm"],
        lambda m: subtree_params(m, "decoder.layers.0") | subtree_params(m, "decoder.final_norm"),
    ),
    (
        "multiple_hyena_and_attention_specific",
        ["dense_projection", "linear_qkv"],
        lambda m: by_short_name(m, "dense_projection") | by_short_name(m, "linear_qkv"),
        # Selects Hyena-specific and Attention-specific modules simultaneously.
    ),
    (
        "multi_both_dense_modules",
        ["dense_projection", "dense"],
        lambda m: by_short_name(m, "dense_projection") | by_short_name(m, "dense"),
    ),
    (
        "multiple_all_norms",
        lambda m: [_pre_mixer_norm_name(m), _PRE_MLP_NORM_NAME, "final_norm"],
        lambda m: (
            by_short_name(m, _pre_mixer_norm_name(m))
            | by_short_name(m, _PRE_MLP_NORM_NAME)
            | subtree_params(m, "decoder.final_norm")
        ),
    ),
]


# ── Integration test helpers ──────────────────────────────────────────────────


def _build_pretrain_config(
    result_dir: Path,
    *,
    train_iters: int = 1,
    lora: bool = False,
    skip_freeze: list[str] | None = None,
    lora_target_modules: list[str] | None = None,
    pretrained_ckpt_dir: str | None = None,
):
    """Build a minimal ConfigContainer for a tiny Hyena model.

    Uses ``_evo2_common`` from the recipe library (same function that
    ``train_evo2`` calls) with the smallest viable model so the test
    runs quickly on a single GPU.

    Args:
        result_dir: Where to write checkpoints and logs.
        train_iters: Total number of training iterations.
        lora: Whether to enable LoRA fine-tuning.
        skip_freeze: Modules to keep trainable when LoRA freezes the base model.
        lora_target_modules: LoRA target modules (defaults to ``_DEFAULT_LORA_TARGETS``).
        pretrained_ckpt_dir: Path to a pretrained checkpoint (required when ``lora=True``).
    """
    from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH
    from bionemo.evo2.recipes.evo2 import _evo2_common

    cfg = _evo2_common(
        model_provider=_TinyHyenaProvider,
        hf_tokenizer_model_or_path=DEFAULT_HF_TOKENIZER_MODEL_PATH,
        dir=str(result_dir),
        name="evo2",
        mock=True,
        dataset_seed=33,
        seed=42,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        train_iters=train_iters,
        global_batch_size=1,
        micro_batch_size=1,
        seq_length=64,
        lr=1e-2,
        min_lr=1e-3,
        lr_warmup_iters=0,
        precision_config="bf16_mixed",
        lora_finetune=lora,
        lora_dim=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=lora_target_modules or _DEFAULT_LORA_TARGETS,
        lora_skip_freeze_modules=skip_freeze or [],
    )

    cfg.checkpoint.save_interval = train_iters
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.exit_on_missing_checkpoint = False
    cfg.checkpoint.use_checkpoint_args = False

    cfg.train.eval_interval = train_iters
    cfg.train.eval_iters = 0

    cfg.logger.tensorboard_dir = None

    if pretrained_ckpt_dir:
        cfg.checkpoint.finetune = True
        cfg.checkpoint.pretrained_checkpoint = pretrained_ckpt_dir
        cfg.checkpoint.dist_ckpt_strictness = "ignore_all"

    return cfg


def _pretrain_base_model(base_dir: Path, *, train_iters: int = 1) -> Path:
    """Train a base model for 1 step and return the checkpoint directory."""
    from megatron.bridge.training.pretrain import pretrain

    from bionemo.evo2.models.evo2_provider import hyena_forward_step

    cfg = _build_pretrain_config(base_dir, train_iters=train_iters)
    pretrain(cfg, hyena_forward_step)

    ckpt_parent = base_dir / "evo2" / "checkpoints"
    assert ckpt_parent.exists(), f"Base model checkpoint dir not found at {ckpt_parent}"
    return ckpt_parent


def _load_dist_checkpoint_keys(ckpt_dir: Path) -> set[str]:
    """Read metadata from a torch_dist checkpoint and return the set of tensor keys."""
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    reader = FileSystemReader(str(ckpt_dir))
    meta = reader.read_metadata()
    return {k for k in meta.state_dict_metadata if hasattr(meta.state_dict_metadata[k], "size")}


def _load_dist_checkpoint_tensors(ckpt_dir: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    """Load specific tensors from a torch_dist checkpoint (single-rank, no dist required)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    reader = FileSystemReader(str(ckpt_dir))
    meta = reader.read_metadata()
    state_dict = {}
    for k in keys:
        m = meta.state_dict_metadata[k]
        state_dict[k] = torch.empty(m.size, dtype=m.properties.dtype)
    dcp.load(state_dict, storage_reader=reader, no_dist=True)
    return state_dict


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _suppress_dynamo_errors():
    """Suppress torch.compile errors for GPU tests (broken Triton env).

    Restores the original value when the module's tests are done so other
    test modules in the same process are unaffected.
    """
    old = torch._dynamo.config.suppress_errors
    torch._dynamo.config.suppress_errors = True
    yield
    torch._dynamo.config.suppress_errors = old


@pytest.fixture(scope="module")
def hyena_1b_model(_suppress_dynamo_errors):
    """Build and yield a real Hyena 1B model within a distributed context."""
    with distributed_model_parallel_state(seed=42):
        yield _build_1b_model()


@pytest.fixture
def freeze_case(request, hyena_1b_model):
    """Resolve (patterns, expected) for a parametrized freeze test case.

    Uses indirect parametrize: request.param is a (id, patterns_or_fn, expected_fn) tuple
    from _CASE_SPECS.
    """
    _id, patterns_or_fn, expected_fn = request.param
    patterns = patterns_or_fn(hyena_1b_model) if callable(patterns_or_fn) else patterns_or_fn
    expected = expected_fn(hyena_1b_model)
    return patterns, expected


@pytest.fixture(scope="function")
def wt_models(_suppress_dynamo_errors):
    """Create a tied and an untied Hyena model within a single distributed context.

    Fresh models are created for each test so that successful LoRA applications
    (which freeze parameters and attach adapter wrappers in-place) do not pollute
    subsequent tests in the same class.
    """
    with distributed_model_parallel_state():
        tied_provider = _WTTiedProvider()
        tied_provider.finalize()
        tied = tied_provider.provide(pre_process=True, post_process=True)
        untied_provider = _WTUntiedProvider()
        untied_provider.finalize()
        untied = untied_provider.provide(pre_process=True, post_process=True)
        yield tied, untied


@pytest.fixture(scope="class")
def base_ckpt(tmp_path_factory, _suppress_dynamo_errors) -> Path:
    """Pretrain a base model once for the entire integration test class."""
    base_dir = tmp_path_factory.mktemp("base")
    return _pretrain_base_model(base_dir)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestEvo2LoRAAdapterWiring:
    """Fast check that Evo2LoRA correctly marks adapter and skip-freeze params."""

    def test_adapter_params_always_trainable(self):
        """LoRA adapter params should be trainable regardless of skip_freeze setting."""
        for skip in [[], ["word_embeddings"]]:
            lora = Evo2LoRA(
                target_modules=["linear_fc1", "linear_fc2"],
                skip_freeze_modules=skip,
                dim=4,
                alpha=8,
                dropout=0.0,
            )
            model = lora(_SmallModel(), training=True)

            adapter_params = [
                (n, p)
                for n, p in model.named_parameters()
                if ".adapter." in n or "linear_in" in n or "linear_out" in n
            ]
            assert len(adapter_params) > 0, "Should have adapter parameters"
            for name, param in adapter_params:
                assert param.requires_grad, f"Adapter param {name} should be trainable"

    @pytest.mark.parametrize(
        "target_modules, skip_freeze",
        [
            (["linear_fc1", "linear_fc2"], ["linear_fc2"]),
            (["linear_fc1"], ["*"]),
            (["*.linear_fc2"], ["linear_fc2"]),
            (["linear_fc2"], ["*.linear_fc2"]),
            (["mlp.*"], ["linear_fc2"]),
            (["mlp.*"], ["*.linear_*"]),
        ],
        ids=["exact", "star_skip", "dotstar_target", "dotstar_skip", "parent_glob_target", "both_wildcards"],
    )
    def test_errors_on_target_skip_freeze_overlap(self, target_modules, skip_freeze):
        """Evo2LoRA must raise ValueError when target and skip-freeze patterns overlap."""
        lora = Evo2LoRA(
            target_modules=target_modules,
            skip_freeze_modules=skip_freeze,
            dim=4,
            alpha=8,
            dropout=0.0,
        )
        with pytest.raises(ValueError, match="skip_freeze_modules and target_modules must not overlap"):
            lora(_SmallModel(), training=True)

    @pytest.mark.parametrize(
        "target_modules, skip_freeze",
        [
            (["linear_fc1", "linear_fc2"], ["word_embeddings"]),
            (["*.linear_*"], ["do_not_exist"]),
            (["do_not_exist"], ["*"]),
        ],
        ids=["disjoint", "glob_target_no_skip_match", "no_target_match_star_skip"],
    )
    def test_no_error_when_skip_freeze_disjoint_from_targets(self, target_modules, skip_freeze):
        """No error when skip_freeze_modules and target_modules don't overlap on any module."""
        lora = Evo2LoRA(
            target_modules=target_modules,
            skip_freeze_modules=skip_freeze,
            dim=4,
            alpha=8,
            dropout=0.0,
        )
        lora(_SmallModel(), training=True)


@pytest.mark.timeout(300)
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.usefixtures("_suppress_dynamo_errors")
class TestEvo2LoRAFreeze:
    """Verify Evo2LoRA.freeze_model correctly freezes/unfreezes parameters.

    Each parametrized case applies a list of skip_freeze_modules patterns to
    the model and checks that exactly the expected parameters remain trainable.
    The model is built once per module (module-scoped fixture) and each test
    resets requires_grad via apply_freeze before checking the result.
    """

    def test_model_structure(self, hyena_1b_model):
        """Sanity-check that the 1B model has the expected hybrid architecture."""
        model = hyena_1b_model
        all_linear_qkv = by_short_name(model, "linear_qkv")
        assert all_linear_qkv, "Expected linear_qkv modules in attention layers"
        assert any(".layers.3." in p for p in all_linear_qkv), "Expected linear_qkv in layer 3"

        all_dense_proj = by_short_name(model, "dense_projection")
        assert all_dense_proj, "Expected dense_projection modules in Hyena layers"
        assert not any(".layers.3." in p for p in all_dense_proj), (
            "dense_projection should not appear in attention layer 3"
        )

        all_dense = by_short_name(model, "dense")
        assert all_dense, "Expected dense modules in Hyena layers"
        assert not any(".layers.3." in p for p in all_dense), "dense should not appear in attention layer 3"

    @pytest.mark.parametrize("freeze_case", _CASE_SPECS, ids=[s[0] for s in _CASE_SPECS], indirect=True)
    def test_freeze_pattern(self, freeze_case, hyena_1b_model):
        """Freeze model with patterns and verify the expected grad state."""
        patterns, expected = freeze_case
        state = apply_freeze(hyena_1b_model, patterns)
        assert_grad_state(state, expected)
        reset_trainable(hyena_1b_model)  # restore for the next test


@pytest.mark.timeout(120)
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.usefixtures("_suppress_dynamo_errors")
class TestEvo2LoRAWeightTyingValidation:
    """Evo2LoRA must enforce the weight-tying contract on real Hyena models.

    When share_embeddings_and_output_weights=True, ``embedding.word_embeddings``
    owns the shared tensor and ``output_layer.weight`` is None.  Any
    configuration that would treat the two sides asymmetrically must be
    rejected with an explicit error.  Setting
    share_embeddings_and_output_weights=False opts out; all combinations are
    then valid.

    Test matrix
    -----------
    Config axis:  target_modules / skip_freeze_modules
    Value axis:   word_embeddings only / output_layer only / both
    Model axis:   tied (_WTTiedProvider) / untied (_WTUntiedProvider)

    Constraints
    -----------
    - ``word_embeddings`` is a ``VocabParallelEmbedding`` and does not support
      LoRA adapters in Megatron Bridge.  Including it in ``target_modules``
      always raises ``ValueError``, regardless of weight tying.
    - ``output_layer`` is a ``ColumnParallelLinear`` and *does* support LoRA,
      but only when ``share_embeddings_and_output_weights=False``.  When weight
      tying is enabled ``output_layer.weight is None``, so LoRA cannot be applied.
    """

    # ------------------------------------------------------------------
    # target_modules with word_embeddings — always rejected
    # VocabParallelEmbedding does not support LoRA adapters
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "target_modules",
        [["word_embeddings"], ["output_layer"], ["embedding.*"], ["output_*"]],
        ids=["word_embeddings_only", "output_layer_only", "embedding_wildcard", "output_wildcard"],
    )
    def test_target_modules_one_side_raises_asymmetry_with_tying(self, wt_models, target_modules: list[str]) -> None:
        """With weight tying, listing only one side of the tied pair raises our symmetry ValueError.

        Evo2LoRA catches the asymmetry before handing off to Megatron Bridge.
        The wildcard cases verify the model-walk approach: ``embedding.*`` matches
        the real path ``embedding.word_embeddings`` and ``output_*`` matches
        ``output_layer``; neither would be detected by a synthetic-name check.
        """
        tied, _ = wt_models
        lora = Evo2LoRA(target_modules=target_modules, dim=4, alpha=8, dropout=0.0)
        with pytest.raises(ValueError, match="share_embeddings_and_output_weights"):
            lora(tied, training=True)

    @pytest.mark.parametrize(
        "target_modules, use_tied",
        [
            (["word_embeddings", "output_layer"], True),
            (["word_embeddings"], False),
            (["word_embeddings", "output_layer"], False),
        ],
        ids=["tied_both", "untied_word_only", "untied_both"],
    )
    def test_target_modules_word_embeddings_rejected_by_mbridge(
        self, wt_models, target_modules: list[str], use_tied: bool
    ) -> None:
        """VocabParallelEmbedding does not support LoRA; Megatron Bridge rejects word_embeddings as a target.

        When weight tying is enabled and both sides are listed our symmetry check
        passes, but Megatron Bridge then raises because VocabParallelEmbedding has
        no LoRA adapter support.  For untied models there is no symmetry constraint,
        so Megatron Bridge is always the one that rejects the request.
        """
        tied, untied = wt_models
        model = tied if use_tied else untied
        lora = Evo2LoRA(target_modules=target_modules, dim=4, alpha=8, dropout=0.0)
        with pytest.raises(Exception):
            lora(model, training=True)

    # ------------------------------------------------------------------
    # target_modules with output_layer x share_embeddings_and_output_weights
    # ------------------------------------------------------------------

    def test_target_modules_output_layer_accepted_without_tying(self, wt_models) -> None:
        """output_layer is a valid LoRA target when weight tying is disabled."""
        _, untied = wt_models
        lora = Evo2LoRA(target_modules=["output_layer"], dim=4, alpha=8, dropout=0.0)
        lora(untied, training=True)  # must not raise

    # ------------------------------------------------------------------
    # skip_freeze_modules x share_embeddings_and_output_weights=True
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "skip_freeze",
        [["word_embeddings"], ["output_layer"], ["embedding.*"], ["output_*"]],
        ids=["word_embeddings_only", "output_layer_only", "embedding_wildcard", "output_wildcard"],
    )
    def test_skip_freeze_one_side_raises_with_tying(self, wt_models, skip_freeze: list[str]) -> None:
        """Listing only one side of a tied pair in skip_freeze_modules must raise ValueError.

        The wildcard cases verify the model-walk approach: ``embedding.*`` matches
        the real path ``embedding.word_embeddings`` and ``output_*`` matches
        ``output_layer``; neither would be detected by a synthetic-name check.
        """
        tied, _ = wt_models
        lora = Evo2LoRA(
            target_modules=_DEFAULT_LORA_TARGETS,
            skip_freeze_modules=skip_freeze,
            dim=4,
            alpha=8,
            dropout=0.0,
        )
        with pytest.raises(ValueError, match="share_embeddings_and_output_weights"):
            lora(tied, training=True)

    def test_skip_freeze_both_accepted_with_tying(self, wt_models) -> None:
        """Listing both tied layers in skip_freeze_modules is valid."""
        tied, _ = wt_models
        lora = Evo2LoRA(
            target_modules=_DEFAULT_LORA_TARGETS,
            skip_freeze_modules=["word_embeddings", "output_layer"],
            dim=4,
            alpha=8,
            dropout=0.0,
        )
        lora(tied, training=True)  # must not raise

    # ------------------------------------------------------------------
    # skip_freeze_modules x share_embeddings_and_output_weights=False
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "skip_freeze",
        [["word_embeddings"], ["output_layer"], ["word_embeddings", "output_layer"]],
        ids=["word_embeddings_only", "output_layer_only", "both"],
    )
    def test_skip_freeze_accepted_without_tying(self, wt_models, skip_freeze: list[str]) -> None:
        """Without weight tying all skip_freeze_modules combinations are valid."""
        _, untied = wt_models
        lora = Evo2LoRA(
            target_modules=_DEFAULT_LORA_TARGETS,
            skip_freeze_modules=skip_freeze,
            dim=4,
            alpha=8,
            dropout=0.0,
        )
        lora(untied, training=True)  # must not raise


@pytest.mark.timeout(300)
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.usefixtures("_suppress_dynamo_errors")
class TestEvo2LoRAPretrainIntegration:
    """End-to-end: pretrain() with LoRA + skip_freeze → checkpoint → verify → resume.

    A base model is pretrained once (via the ``base_ckpt`` fixture) and reused
    by every test in this class.
    """

    def test_lora_checkpoint_excludes_frozen_embeddings(self, tmp_path: Path, base_ckpt: Path):
        """LoRA WITHOUT skip_freeze → checkpoint does NOT contain embedding keys."""
        from megatron.bridge.training.pretrain import pretrain

        from bionemo.evo2.models.evo2_provider import hyena_forward_step

        lora_dir = tmp_path / "lora_frozen"
        cfg = _build_pretrain_config(
            lora_dir,
            lora=True,
            skip_freeze=[],
            pretrained_ckpt_dir=str(base_ckpt),
        )
        pretrain(cfg, hyena_forward_step)

        ckpt_dir = lora_dir / "evo2" / "checkpoints" / "iter_0000001"
        assert ckpt_dir.exists(), f"Checkpoint not found at {ckpt_dir}"

        keys = _load_dist_checkpoint_keys(ckpt_dir)
        emb_keys = [k for k in keys if "word_embeddings" in k]
        adapter_keys = [k for k in keys if ".adapter." in k]

        assert len(emb_keys) == 0, f"Checkpoint should NOT contain word_embeddings when frozen. Found: {emb_keys}"
        assert len(adapter_keys) > 0, "Checkpoint should still contain LoRA adapter keys."

    @pytest.mark.parametrize(
        "skip_freeze, expected_key_substr, lora_targets",
        [
            # word_embeddings and output_layer share a weight tensor when tying is enabled;
            # both must appear in skip_freeze to satisfy the symmetry contract.
            (["word_embeddings", "output_layer"], "word_embeddings", None),
            (["final_norm"], "final_norm", None),
            (["dense"], "mixer.dense.", None),
            (["linear_fc2"], "mlp.linear_fc2.", ["dense_projection", "linear_qkv", "linear_proj", "linear_fc1"]),
        ],
        ids=["word_embeddings", "final_norm", "dense", "linear_fc2"],
    )
    def test_lora_skip_freeze_saves_and_trains_module(
        self,
        tmp_path: Path,
        base_ckpt: Path,
        skip_freeze: list[str],
        expected_key_substr: str,
        lora_targets: list[str] | None,
    ):
        """LoRA + skip_freeze → checkpoint contains the unfrozen module and its weights changed."""
        from megatron.bridge.training.pretrain import pretrain

        from bionemo.evo2.models.evo2_provider import hyena_forward_step

        lora_dir = tmp_path / f"lora_{skip_freeze[0]}"
        cfg = _build_pretrain_config(
            lora_dir,
            train_iters=1,
            lora=True,
            skip_freeze=skip_freeze,
            lora_target_modules=lora_targets,
            pretrained_ckpt_dir=str(base_ckpt),
        )
        pretrain(cfg, hyena_forward_step)

        lora_iter1 = lora_dir / "evo2" / "checkpoints" / "iter_0000001"
        assert lora_iter1.exists(), f"LoRA checkpoint not found at {lora_iter1}"

        keys = _load_dist_checkpoint_keys(lora_iter1)
        unfrozen_keys = [k for k in keys if expected_key_substr in k and ".adapter." not in k]
        adapter_keys = [k for k in keys if ".adapter." in k]

        assert len(unfrozen_keys) > 0, (
            f"Checkpoint should contain '{expected_key_substr}' (unfrozen via skip_freeze). "
            f"Keys sample: {sorted(keys)[:20]}"
        )
        assert len(adapter_keys) > 0, f"Checkpoint should contain LoRA adapter keys. Keys sample: {sorted(keys)[:20]}"

        base_iter1 = base_ckpt / "iter_0000001"
        base_tensors = _load_dist_checkpoint_tensors(base_iter1, unfrozen_keys)
        lora_tensors = _load_dist_checkpoint_tensors(lora_iter1, unfrozen_keys)

        for k in unfrozen_keys:
            assert not torch.equal(base_tensors[k], lora_tensors[k]), (
                f"{k} should differ between base and LoRA checkpoints, "
                "proving the unfrozen module was actually trained."
            )
