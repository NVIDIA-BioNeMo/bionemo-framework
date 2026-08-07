# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Vortex -> MBridge checkpoint converter.

Converts ARC/Vortex Evo2 ``.pt`` checkpoints, including
``evo2_7b_microviridae.pt``, into the Megatron Bridge DCP format used for
BioNeMo Evo2 training and inference.
"""

import argparse
import codecs
import logging
from collections import OrderedDict
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import huggingface_hub.errors
import torch
from huggingface_hub import hf_hub_download

from bionemo.evo2.models.evo2_provider import HYENA_MODEL_OPTIONS, HyenaModelProvider
from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import package_mbridge_checkpoint


logger = logging.getLogger(__name__)
_FP32_SUFFIXES = frozenset({"h", "decay", "gamma", "R", "p"})


MICROVIRIDAE_MODEL_SIZE = "evo2_7b_microviridae"
MICROVIRIDAE_SEQ_LENGTH = 10_240
VORTEX_MODEL_OPTIONS = dict(HYENA_MODEL_OPTIONS)
VORTEX_MODEL_OPTIONS[MICROVIRIDAE_MODEL_SIZE] = HYENA_MODEL_OPTIONS["evo2_7b_base"]


def download_vortex_checkpoint(
    repo_id: str,
    filename: str | None = None,
    cache_dir: Path | None = None,
    revision: str | None = None,
) -> Path:
    """Download a Vortex checkpoint from Hugging Face.

    Args:
        repo_id: Hugging Face repository ID.
        filename: Optional checkpoint filename. Defaults to ``<repo-name>.pt``.
        cache_dir: Optional local cache directory.
        revision: Optional repository revision.

    Returns:
        Path to the downloaded checkpoint.
    """
    if filename is None:
        filename = f"{repo_id.split('/')[-1]}.pt"

    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(cache_dir) if cache_dir else None,
                revision=revision,
            )
        )
    except huggingface_hub.errors.EntryNotFoundError as exc:
        raise FileNotFoundError(f"Could not find {filename!r} in Hugging Face repo {repo_id!r}") from exc


def load_vortex_state_dict(path: Path) -> OrderedDict[str, Any]:
    """Load a Vortex ``.pt`` checkpoint without executing checkpoint code."""
    with torch.serialization.safe_globals([BytesIO, codecs.encode]):
        raw = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)

    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("Vortex checkpoint is not a non-empty state dictionary")

    state_dict = raw["module"] if "module" in raw else raw
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Vortex checkpoint module is not a non-empty state dictionary")

    normalized_state: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"Vortex checkpoint key is not a string: {key!r}")
        normalized_key = key.removeprefix("module.")
        if normalized_key in normalized_state:
            raise ValueError(f"duplicate normalized Vortex key: {normalized_key}")
        normalized_state[normalized_key] = value
    return normalized_state


def _merge_fc1(l1_weight: torch.Tensor, l2_weight: torch.Tensor) -> torch.Tensor:
    """Merge Vortex MLP gate/up-projection weights into MBridge ``linear_fc1``."""
    return torch.cat([l1_weight, l2_weight], dim=0)


def _nextafter_n(tensor: torch.Tensor, direction: float, steps: int) -> torch.Tensor:
    """Move ``tensor`` by ``steps`` representable fp32 values in ``direction``."""
    result = tensor
    target = torch.full_like(tensor, torch.inf if direction > 0 else -torch.inf)
    for _ in range(steps):
        result = torch.nextafter(result, target)
    return result


def _invert_log_poles(
    log_poles: torch.Tensor,
    num_groups: int,
    gamma_min: float = 0.01,
    gamma_max: float = 0.1,
    search_radius: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose deterministic MBridge ``p`` and ``gamma`` values for Vortex log poles.

    MBridge exports ``log_poles = -exp(p) * exp(gamma)``.  Vortex stores only
    that product, so the inverse has infinitely many valid parameterizations.

    Original 1B/7B checkpoints show trained ``p`` and ``gamma`` moving together
    near a balanced split of the product, rather than staying close to the
    initial ``gamma = log(U(gamma_min, gamma_max))`` range. Prefer exact fp32
    round-trip solutions with the smallest ``abs(p - gamma)``. Direct and
    init-range seeds are kept as fallbacks because exact Vortex reproduction is
    more important than the prior when the two conflict.
    """
    log_poles = log_poles.to(torch.float32)
    if log_poles.dim() == 3 and log_poles.shape[-1] == 1:
        log_poles = log_poles.squeeze(-1)
    state_size = log_poles.numel() // num_groups
    log_poles = log_poles.reshape(num_groups, state_size)
    if not torch.all(log_poles < 0):
        raise ValueError("Vortex log_poles must be strictly negative to invert to MBridge p/gamma")

    target = -log_poles
    log_gamma_min = torch.tensor(gamma_min, dtype=torch.float32).log()
    log_gamma_max = torch.tensor(gamma_max, dtype=torch.float32).log()
    log_gamma_mid = (log_gamma_min + log_gamma_max) / 2
    balanced_gamma = 0.5 * torch.log(target)
    direct_gamma = torch.zeros_like(target)
    gamma_seeds = [
        balanced_gamma,
        direct_gamma,
        torch.full_like(target, log_gamma_mid),
        torch.full_like(target, log_gamma_min),
        torch.full_like(target, log_gamma_max),
    ]

    best_p = torch.log(target)
    best_gamma = direct_gamma
    best_primary = torch.full_like(target, torch.inf)
    best_secondary = torch.full_like(target, torch.inf)
    solved = torch.zeros_like(target, dtype=torch.bool)

    for gamma_seed in gamma_seeds:
        for gamma_offset in range(-search_radius, search_radius + 1):
            if gamma_offset == 0:
                gamma = gamma_seed
            else:
                gamma = _nextafter_n(gamma_seed, 1.0 if gamma_offset > 0 else -1.0, abs(gamma_offset))

            base_p = torch.log(target / torch.exp(gamma))
            for p_offset in range(-search_radius, search_radius + 1):
                if p_offset == 0:
                    p = base_p
                else:
                    p = _nextafter_n(base_p, 1.0 if p_offset > 0 else -1.0, abs(p_offset))

                exact = (-(torch.exp(p) * torch.exp(gamma))) == log_poles
                primary = (p - gamma).abs()
                below = torch.clamp(log_gamma_min - gamma, min=0)
                above = torch.clamp(gamma - log_gamma_max, min=0)
                secondary = below + above
                update = exact & (
                    (primary < best_primary) | ((primary == best_primary) & (secondary < best_secondary))
                )
                best_p = torch.where(update, p, best_p)
                best_gamma = torch.where(update, gamma, best_gamma)
                best_primary = torch.where(update, primary, best_primary)
                best_secondary = torch.where(update, secondary, best_secondary)
                solved |= exact

    if not torch.all(solved):
        unsolved = int((~solved).sum().item())
        raise ValueError(
            f"Could not find exact fp32 p/gamma inverse for {unsolved} log_poles values "
            f"within search_radius={search_radius}"
        )

    p = best_p
    gamma = best_gamma
    return p, gamma


def _compute_initial_medium_decay(
    d_model: int,
    filter_len: int,
    decay_preset: str = "weak",
) -> torch.Tensor:
    """Compute the deterministic medium-filter decay initialization."""
    if decay_preset == "strong":
        log_r_min = 0
        log_r_max = 2
    elif decay_preset == "normal":
        log_r_min = -1
        log_r_max = 2
    elif decay_preset == "weak":
        log_r_min = -2
        log_r_max = 2
    else:
        raise ValueError(f"Unknown decay preset: {decay_preset}")

    t = torch.linspace(0, 1, filter_len, dtype=torch.float32)[None]
    decay_domain = torch.logspace(log_r_min, log_r_max, d_model, dtype=torch.float32)[:, None]
    return torch.exp(-decay_domain * t)


def _invert_medium_filter(filter_h: torch.Tensor, decay_preset: str = "weak") -> tuple[torch.Tensor, torch.Tensor]:
    """Choose deterministic MBridge ``h`` and ``decay`` values for a medium filter.

    MBridge exports Vortex medium filters as ``h * decay``. The inverse is
    ambiguous. Use ``decay = 1`` so the current exporter reproduces the Vortex
    filter tensor bitwise exactly.
    """
    h = filter_h
    if h.dim() == 3 and h.shape[1] == 1:
        h = h.squeeze(1)
    h = h.to(torch.float32)
    decay = torch.ones_like(h, dtype=torch.float32)
    return h, decay


def _required_mbridge_keys(model_provider: HyenaModelProvider, te_enabled: bool) -> set[str]:
    """Return structural model keys that every compatible Vortex checkpoint must produce."""
    required = {
        "embedding.word_embeddings.weight",
        "decoder.final_norm.weight",
    }
    for layer_idx, symbol in enumerate(model_provider.hybrid_override_pattern):
        prefix = f"decoder.layers.{layer_idx}"
        if symbol == "*":
            norm_key = (
                f"{prefix}.self_attention.linear_qkv.layer_norm_weight"
                if te_enabled
                else f"{prefix}.input_layernorm.weight"
            )
            required.update(
                {
                    norm_key,
                    f"{prefix}.self_attention.linear_qkv.weight",
                    f"{prefix}.self_attention.linear_proj.weight",
                    f"{prefix}.self_attention.linear_proj.bias",
                }
            )
        else:
            norm_key = f"{prefix}.mixer.dense_projection.layer_norm_weight" if te_enabled else f"{prefix}.norm.weight"
            required.update(
                {
                    norm_key,
                    f"{prefix}.mixer.dense_projection.weight",
                    f"{prefix}.mixer.hyena_proj_conv.short_conv_weight",
                    f"{prefix}.mixer.dense.weight",
                    f"{prefix}.mixer.dense.bias",
                }
            )
            if symbol == "S":
                required.add(f"{prefix}.mixer.mixer.short_conv.short_conv_weight")
            elif symbol == "D":
                required.update(
                    {
                        f"{prefix}.mixer.mixer.conv_bias",
                        f"{prefix}.mixer.mixer.filter.h",
                        f"{prefix}.mixer.mixer.filter.decay",
                    }
                )
            elif symbol == "H":
                required.update(
                    {
                        f"{prefix}.mixer.mixer.conv_bias",
                        f"{prefix}.mixer.mixer.filter.p",
                        f"{prefix}.mixer.mixer.filter.gamma",
                        f"{prefix}.mixer.mixer.filter.R",
                    }
                )

        post_norm_key = (
            f"{prefix}.mlp.linear_fc1.layer_norm_weight" if te_enabled else f"{prefix}.pre_mlp_layernorm.weight"
        )
        required.update({post_norm_key, f"{prefix}.mlp.linear_fc1.weight", f"{prefix}.mlp.linear_fc2.weight"})
    return required


def _validate_rotary_frequencies(rotary: torch.Tensor, model_provider: HyenaModelProvider) -> None:
    """Validate a Vortex rotary buffer before omitting this runtime-derived state."""
    if not rotary.dtype.is_floating_point:
        raise ValueError(f"Vortex rotary frequencies must be floating point, got {rotary.dtype}")

    rotary_dim = model_provider.hidden_size // model_provider.num_attention_heads
    expected = 1.0 / (
        float(model_provider.rotary_base) ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    actual = rotary.detach().reshape(-1).float()
    if actual.shape != expected.shape:
        raise ValueError(
            f"Vortex rotary frequencies have shape {tuple(rotary.shape)}; "
            f"target provider expects {tuple(expected.shape)}"
        )

    precision_dtype = rotary.dtype
    if rotary.dtype == torch.float32 and torch.equal(actual, actual.to(torch.bfloat16).float()):
        # Some released checkpoints store BF16-quantized rotary values in an
        # FP32 tensor. Validate those at their actual source precision.
        precision_dtype = torch.bfloat16
    tolerance = torch.finfo(precision_dtype).eps
    if not torch.isfinite(actual).all() or not torch.allclose(actual, expected, rtol=tolerance, atol=tolerance):
        raise ValueError("Vortex rotary frequencies disagree with the target model provider")


def vortex_to_mbridge_state_dict(
    vortex_state_dict: dict[str, Any],
    model_provider: HyenaModelProvider,
    te_enabled: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Convert a Vortex state dict to MBridge key format.

    Args:
        vortex_state_dict: Flat Vortex state dict.
        model_provider: Evo2 model provider with target hyperparameters.
        te_enabled: Whether TransformerEngine fused layernorm keys should be produced.

    Returns:
        Ordered MBridge-style state dict.
    """
    pattern = model_provider.hybrid_override_pattern
    mbridge_sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    src = OrderedDict(vortex_state_dict)

    embed_key = "embedding_layer.weight"
    unembed_key = "unembed.weight"
    if embed_key in src:
        if unembed_key not in src:
            raise ValueError("Vortex checkpoint is missing tied output projection unembed.weight")
        if not torch.equal(src[embed_key], src[unembed_key]):
            raise ValueError("Vortex embedding_layer.weight and unembed.weight differ; MBridge expects tied weights")
        mbridge_sd["embedding.word_embeddings.weight"] = src.pop(embed_key)
        src.pop(unembed_key)

    for layer_idx, symbol in enumerate(pattern):
        block_prefix = f"blocks.{layer_idx}"
        prefix = f"decoder.layers.{layer_idx}"

        if symbol == "*":
            _convert_attention_layer(src, mbridge_sd, block_prefix, prefix, te_enabled, model_provider)
        else:
            _convert_hyena_layer(
                src,
                mbridge_sd,
                block_prefix,
                prefix,
                symbol,
                te_enabled,
                model_provider,
            )
        _convert_mlp(
            src,
            mbridge_sd,
            block_prefix,
            prefix,
            te_enabled,
            model_provider.hidden_size,
            model_provider.ffn_hidden_size,
        )

    final_norm_key = "norm.scale"
    if final_norm_key in src:
        mbridge_sd["decoder.final_norm.weight"] = src.pop(final_norm_key)

    missing = sorted(_required_mbridge_keys(model_provider, te_enabled) - mbridge_sd.keys())
    if missing:
        raise ValueError(
            f"Vortex checkpoint is incompatible with target pattern {pattern!r}: "
            f"missing required MBridge keys ({len(missing)}): {missing[:20]}"
        )

    # These are runtime-generated caches/state rather than model parameters.
    # Each framework rebuilds its own copies when loading the converted model.
    ignored_suffixes = ("._extra_state", ".filter.t")
    unmapped = [k for k in src if not k.endswith(ignored_suffixes)]
    if unmapped:
        raise ValueError(f"unmapped Vortex entries are fatal: {sorted(unmapped)[:20]} ({len(unmapped)} total)")

    params_dtype = model_provider.params_dtype
    for key, source_tensor in mbridge_sd.items():
        suffix = key.rsplit(".", 1)[-1]
        target_dtype = torch.float32 if suffix in _FP32_SUFFIXES else params_dtype
        converted_tensor = source_tensor.to(target_dtype) if source_tensor.dtype != target_dtype else source_tensor
        mbridge_sd[key] = converted_tensor.contiguous()

    return mbridge_sd


def _convert_hyena_layer(
    src: OrderedDict[str, torch.Tensor],
    dst: OrderedDict[str, torch.Tensor],
    block_prefix: str,
    prefix: str,
    symbol: str,
    te_enabled: bool,
    model_provider: HyenaModelProvider,
) -> None:
    if te_enabled:
        ln_key = f"{prefix}.mixer.dense_projection.layer_norm_weight"
    else:
        ln_key = f"{prefix}.norm.weight"
    if (key := f"{block_prefix}.pre_norm.scale") in src:
        dst[ln_key] = src.pop(key)

    if (key := f"{block_prefix}.projections.weight") in src:
        dst[f"{prefix}.mixer.dense_projection.weight"] = src.pop(key)

    if (key := f"{block_prefix}.filter.short_filter_weight") in src:
        w = src.pop(key)
        expected_shape = (3 * model_provider.hidden_size, 1, 3)
        if tuple(w.shape) != expected_shape:
            raise ValueError(f"{block_prefix} has invalid Vortex projection FIR: {tuple(w.shape)}")
        dst[f"{prefix}.mixer.hyena_proj_conv.short_conv_weight"] = w[:, 0, :]

    if (key := f"{block_prefix}.out_filter_dense.weight") in src:
        dst[f"{prefix}.mixer.dense.weight"] = src.pop(key)
    if (key := f"{block_prefix}.out_filter_dense.bias") in src:
        dst[f"{prefix}.mixer.dense.bias"] = src.pop(key)

    if symbol == "S":
        if (key := f"{block_prefix}.filter.h") in src:
            dst[f"{prefix}.mixer.mixer.short_conv.short_conv_weight"] = src.pop(key)
    elif symbol == "D":
        if (key := f"{block_prefix}.filter.D") in src:
            dst[f"{prefix}.mixer.mixer.conv_bias"] = src.pop(key)
        if (key := f"{block_prefix}.filter.h") in src:
            filter_h = src[key]
            medium_groups = model_provider.num_groups_hyena_medium or model_provider.num_groups_hyena
            medium_conv_len = getattr(model_provider, "hyena_medium_conv_len", 128) or 128
            expected_shape = (medium_groups, 1, medium_conv_len)
            if not isinstance(filter_h, torch.Tensor) or tuple(filter_h.shape) != expected_shape:
                shape = tuple(filter_h.shape) if isinstance(filter_h, torch.Tensor) else type(filter_h).__name__
                raise ValueError(f"{block_prefix} has invalid Vortex medium filter: {shape}")
            h, decay = _invert_medium_filter(src.pop(key))
            dst[f"{prefix}.mixer.mixer.filter.h"] = h
            dst[f"{prefix}.mixer.mixer.filter.decay"] = decay
    elif symbol == "H":
        if (key := f"{block_prefix}.filter.D") in src:
            dst[f"{prefix}.mixer.mixer.conv_bias"] = src.pop(key)
        log_poles_key = f"{block_prefix}.filter.log_poles"
        residues_key = f"{block_prefix}.filter.residues"
        if log_poles_key in src and residues_key in src:
            log_poles = src[log_poles_key]
            residues = src[residues_key]
            num_groups = model_provider.num_groups_hyena
            if (
                not isinstance(log_poles, torch.Tensor)
                or not isinstance(residues, torch.Tensor)
                or log_poles.ndim != 3
                or log_poles.shape[0] != num_groups
                or log_poles.shape[-1] != 1
                or tuple(residues.shape) != tuple(log_poles.shape[:-1])
                or not torch.isfinite(log_poles).all()
                or not torch.all(log_poles < 0)
            ):
                raise ValueError(f"{block_prefix} has invalid Vortex poles/residues")
        if log_poles_key in src:
            p, gamma = _invert_log_poles(src.pop(log_poles_key), model_provider.num_groups_hyena)
            dst[f"{prefix}.mixer.mixer.filter.p"] = p
            dst[f"{prefix}.mixer.mixer.filter.gamma"] = gamma
        if residues_key in src:
            dst[f"{prefix}.mixer.mixer.filter.R"] = src.pop(residues_key).to(torch.float32)


def _convert_attention_layer(
    src: OrderedDict[str, torch.Tensor],
    dst: OrderedDict[str, torch.Tensor],
    block_prefix: str,
    prefix: str,
    te_enabled: bool,
    model_provider: HyenaModelProvider,
) -> None:
    if te_enabled:
        ln_key = f"{prefix}.self_attention.linear_qkv.layer_norm_weight"
    else:
        ln_key = f"{prefix}.input_layernorm.weight"
    if (key := f"{block_prefix}.pre_norm.scale") in src:
        dst[ln_key] = src.pop(key)

    if (key := f"{block_prefix}.inner_mha_cls.Wqkv.weight") in src:
        dst[f"{prefix}.self_attention.linear_qkv.weight"] = src.pop(key)
    if (key := f"{block_prefix}.inner_mha_cls.out_proj.weight") in src:
        dst[f"{prefix}.self_attention.linear_proj.weight"] = src.pop(key)
    if (key := f"{block_prefix}.inner_mha_cls.out_proj.bias") in src:
        dst[f"{prefix}.self_attention.linear_proj.bias"] = src.pop(key)
    if (key := f"{block_prefix}.inner_mha_cls.rotary_emb.inv_freq") in src:
        _validate_rotary_frequencies(src.pop(key), model_provider)


def _convert_mlp(
    src: OrderedDict[str, torch.Tensor],
    dst: OrderedDict[str, torch.Tensor],
    block_prefix: str,
    prefix: str,
    te_enabled: bool,
    hidden_size: int,
    ffn_hidden_size: int,
) -> None:
    if te_enabled:
        post_norm_key = f"{prefix}.mlp.linear_fc1.layer_norm_weight"
    else:
        post_norm_key = f"{prefix}.pre_mlp_layernorm.weight"
    if (key := f"{block_prefix}.post_norm.scale") in src:
        dst[post_norm_key] = src.pop(key)

    l1_key = f"{block_prefix}.mlp.l1.weight"
    l2_key = f"{block_prefix}.mlp.l2.weight"
    if l1_key in src and l2_key in src:
        l1_weight = src[l1_key]
        l2_weight = src[l2_key]
        expected_shape = (ffn_hidden_size, hidden_size)
        if tuple(l1_weight.shape) != expected_shape or tuple(l2_weight.shape) != expected_shape:
            raise ValueError(f"{block_prefix} has invalid Vortex gated-MLP geometry")
        dst[f"{prefix}.mlp.linear_fc1.weight"] = _merge_fc1(src.pop(l1_key), src.pop(l2_key))

    if (key := f"{block_prefix}.mlp.l3.weight") in src:
        dst[f"{prefix}.mlp.linear_fc2.weight"] = src.pop(key)


def _default_te_extra_state() -> torch.Tensor:
    """Default TE extra-state value for modules without serialized FP8 state."""
    return torch.empty(0, dtype=torch.uint8)


def _add_mbridge_te_extra_states(
    state_dict: OrderedDict[str, Any],
    model_provider: HyenaModelProvider,
) -> None:
    """Populate MBridge TE ``_extra_state`` keys required by checkpoint load."""

    def add_extra_state(key: str, value: Any) -> None:
        state_dict[key] = value

    for i, symbol in enumerate(model_provider.hybrid_override_pattern):
        prefix = f"decoder.layers.{i}"

        if symbol == "*":
            add_extra_state(f"{prefix}.self_attention.core_attention._extra_state", _default_te_extra_state())
            add_extra_state(f"{prefix}.self_attention.linear_qkv._extra_state", _default_te_extra_state())
            add_extra_state(f"{prefix}.self_attention.linear_proj._extra_state", _default_te_extra_state())
        else:
            add_extra_state(f"{prefix}.mixer.dense_projection._extra_state", _default_te_extra_state())
            add_extra_state(f"{prefix}.mixer.dense._extra_state", _default_te_extra_state())

        add_extra_state(f"{prefix}.mlp.linear_fc1._extra_state", _default_te_extra_state())
        add_extra_state(f"{prefix}.mlp.linear_fc2._extra_state", _default_te_extra_state())

    add_extra_state("decoder.final_norm._extra_state", _default_te_extra_state())
    add_extra_state("output_layer._extra_state", None)


def vortex_to_mbridge(
    vortex_ckpt_path: Path | str,
    mbridge_ckpt_dir: Path,
    model_size: str,
    tokenizer_path: Path,
    seq_length: int | None = None,
    te_enabled: bool = True,
    mixed_precision_recipe: str = "bf16_mixed",
) -> Path:
    """Convert a Vortex checkpoint file into a packaged MBridge checkpoint."""
    vortex_path = Path(vortex_ckpt_path)
    logger.info("Loading Vortex checkpoint from %s", vortex_path)
    vortex_sd = load_vortex_state_dict(vortex_path)

    provider_cls = VORTEX_MODEL_OPTIONS[model_size]
    kwargs = {}
    if model_size == MICROVIRIDAE_MODEL_SIZE and seq_length is None:
        kwargs["seq_length"] = MICROVIRIDAE_SEQ_LENGTH
    if seq_length is not None:
        kwargs["seq_length"] = seq_length
    model_provider = provider_cls(**kwargs)

    logger.info("Converting Vortex to MBridge with pattern=%s", model_provider.hybrid_override_pattern)
    mbridge_sd = vortex_to_mbridge_state_dict(vortex_sd, model_provider, te_enabled=te_enabled)
    del vortex_sd
    if te_enabled:
        _add_mbridge_te_extra_states(mbridge_sd, model_provider)
    logger.info("Converted to %d MBridge keys", len(mbridge_sd))

    return package_mbridge_checkpoint(
        mbridge_sd,
        mbridge_ckpt_dir=mbridge_ckpt_dir,
        model_provider=model_provider,
        tokenizer_path=tokenizer_path,
        mixed_precision_recipe=mixed_precision_recipe,
    )


def main():
    """CLI entry point for Vortex-to-MBridge conversion."""
    parser = argparse.ArgumentParser(description="Convert Vortex Evo2 checkpoint to MBridge format")
    parser.add_argument("--vortex-ckpt-path", type=Path, default=None, help="Local Vortex .pt checkpoint path")
    parser.add_argument("--hf-repo-id", type=str, default=None, help="Hugging Face repo ID for the Vortex checkpoint")
    parser.add_argument("--hf-filename", type=str, default=None, help="Checkpoint filename in the Hugging Face repo")
    parser.add_argument("--revision", type=str, default=None, help="Optional Hugging Face revision")
    parser.add_argument("--mbridge-ckpt-dir", type=Path, required=True, help="Output MBridge checkpoint directory")
    parser.add_argument("--model-size", type=str, choices=sorted(VORTEX_MODEL_OPTIONS.keys()), required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--mixed-precision-recipe", type=str, default="bf16_mixed")
    parser.add_argument("--no-te", action="store_true", help="Disable TE fused layernorm key mapping")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.vortex_ckpt_path is None:
        if args.hf_repo_id is None:
            raise ValueError("Provide either --vortex-ckpt-path or --hf-repo-id")
        args.vortex_ckpt_path = download_vortex_checkpoint(
            args.hf_repo_id,
            filename=args.hf_filename,
            revision=args.revision,
        )

    vortex_to_mbridge(
        vortex_ckpt_path=args.vortex_ckpt_path,
        mbridge_ckpt_dir=args.mbridge_ckpt_dir,
        model_size=args.model_size,
        tokenizer_path=args.tokenizer_path,
        seq_length=args.seq_length,
        te_enabled=not args.no_te,
        mixed_precision_recipe=args.mixed_precision_recipe,
    )


if __name__ == "__main__":
    main()
