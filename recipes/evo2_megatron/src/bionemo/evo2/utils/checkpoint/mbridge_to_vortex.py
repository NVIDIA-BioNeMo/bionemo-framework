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

"""MBridge -> Vortex checkpoint exporter.

Converts a Megatron Bridge DCP checkpoint to ARC's Vortex inference format
(a single .pt file compatible with the stripedhyena inference engine).
"""

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

from bionemo.evo2.models.evo2_provider import HYENA_MODEL_OPTIONS, HyenaModelProvider


logger = logging.getLogger(__name__)


def load_mbridge_state_dict(mbridge_ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Load state dict from an mbridge DCP checkpoint directory.

    Args:
        mbridge_ckpt_dir: Path to the mbridge checkpoint root (containing iter_XXXXXXX/),
            or directly to an iter_XXXXXXX directory.

    Returns:
        Flat state dict with all tensor parameters.
    """
    import re

    if re.match(r"^iter_\d+$", mbridge_ckpt_dir.name):
        iter_dir = mbridge_ckpt_dir
    elif (latest_file := mbridge_ckpt_dir / "latest_checkpointed_iteration.txt").exists():
        iteration = latest_file.read_text().strip()
        iter_dir = mbridge_ckpt_dir / f"iter_{int(iteration):07d}"
    else:
        iter_dirs = sorted(mbridge_ckpt_dir.glob("iter_*"))
        if not iter_dirs:
            raise FileNotFoundError(f"No iter_* directories in {mbridge_ckpt_dir}")
        iter_dir = iter_dirs[-1]

    reader = FileSystemReader(str(iter_dir))
    metadata = reader.read_metadata()

    state_dict = {}
    for key, item_meta in metadata.state_dict_metadata.items():
        if isinstance(item_meta, BytesStorageMetadata):
            continue
        state_dict[key] = torch.empty(item_meta.size, dtype=item_meta.properties.dtype, device="cpu")

    dcp.load(state_dict=state_dict, storage_reader=reader, no_dist=True)
    return state_dict


def _split_fc1(fc1_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split merged linear_fc1.weight back into gate (l1) and up-projection (l2).

    MBridge stores gate and up-projection concatenated: [w1; w2].
    Vortex expects them as separate l1.weight and l2.weight tensors.
    """
    half = fc1_weight.shape[0] // 2
    return fc1_weight[:half], fc1_weight[half:]


def _compute_log_poles_and_residues(
    p: torch.Tensor,
    gamma: torch.Tensor,
    residue_param: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Vortex log_poles and residues from Savanna/MBridge filter parameters.

    Args:
        p: Pole parameter tensor of shape (num_groups, state_size).
        gamma: Gamma (decay scaling) parameter tensor of shape (num_groups, state_size).
        residue_param: Residue parameter tensor (the 'R' parameter) of shape (num_groups, state_size).
        num_groups: Number of hyena groups (typically hidden_size).

    Returns:
        Tuple of (log_poles, residues) tensors.
    """
    state_size = p.numel() // num_groups
    p = p.to(torch.float32).reshape(num_groups, state_size)
    gamma = gamma.to(torch.float32)
    residues = residue_param.to(torch.float32).reshape(num_groups, state_size)

    exp_gamma = torch.exp(gamma)
    logp = -torch.exp(p) * exp_gamma
    logp = logp[..., None]

    return logp, residues


def mbridge_to_vortex_state_dict(
    mbridge_state_dict: dict[str, torch.Tensor],
    model_provider: HyenaModelProvider,
    te_enabled: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Convert an MBridge state dict to Vortex format.

    Args:
        mbridge_state_dict: State dict loaded from mbridge DCP checkpoint.
        model_provider: The HyenaModelProvider instance with model hyperparameters.
        te_enabled: Whether TransformerEngine fused layernorm keys are used.

    Returns:
        OrderedDict in Vortex format (embedding_layer, blocks, unembed, norm).
    """
    pattern = model_provider.hybrid_override_pattern
    num_groups = model_provider.num_groups_hyena
    medium_conv_len = getattr(model_provider, "hyena_medium_conv_len", 128)
    rotary_dim = model_provider.hidden_size // model_provider.num_attention_heads
    rotary_base = float(getattr(model_provider, "rotary_base", 10000))

    vortex_sd: OrderedDict[str, torch.Tensor] = OrderedDict()

    embed_key = "embedding.word_embeddings.weight"
    if embed_key in mbridge_state_dict:
        embed_w = mbridge_state_dict.pop(embed_key)
        vortex_sd["embedding_layer.weight"] = embed_w
        vortex_sd["unembed.weight"] = embed_w

    for layer_idx, symbol in enumerate(pattern):
        prefix = f"decoder.layers.{layer_idx}"
        block_prefix = f"blocks.{layer_idx}"

        if symbol != "*":
            _convert_hyena_layer(
                mbridge_state_dict,
                vortex_sd,
                prefix,
                block_prefix,
                symbol,
                te_enabled,
                num_groups,
                medium_conv_len,
            )
        else:
            _convert_attention_layer(
                mbridge_state_dict,
                vortex_sd,
                prefix,
                block_prefix,
                te_enabled,
                rotary_dim,
                rotary_base,
            )

        _convert_mlp(mbridge_state_dict, vortex_sd, prefix, block_prefix, te_enabled)

    final_norm_key = "decoder.final_norm.weight"
    if final_norm_key in mbridge_state_dict:
        vortex_sd["norm.scale"] = mbridge_state_dict.pop(final_norm_key)

    return vortex_sd


def _validate_vortex_keys(vortex_sd: dict[str, torch.Tensor], pattern: str) -> None:
    """Validate that all mandatory keys are present in the converted vortex state dict.

    Raises:
        ValueError: If any mandatory keys are missing.
    """
    mandatory = {"embedding_layer.weight", "unembed.weight", "norm.scale"}

    for layer_idx, symbol in enumerate(pattern):
        bp = f"blocks.{layer_idx}"
        mandatory.add(f"{bp}.pre_norm.scale")
        mandatory.add(f"{bp}.post_norm.scale")
        mandatory.add(f"{bp}.mlp.l1.weight")
        mandatory.add(f"{bp}.mlp.l2.weight")
        mandatory.add(f"{bp}.mlp.l3.weight")

        if symbol == "*":
            mandatory.add(f"{bp}.inner_mha_cls.Wqkv.weight")
            mandatory.add(f"{bp}.inner_mha_cls.out_proj.weight")
        else:
            mandatory.add(f"{bp}.projections.weight")
            mandatory.add(f"{bp}.out_filter_dense.weight")

    missing = sorted(mandatory - set(vortex_sd.keys()))
    if missing:
        raise ValueError(f"Vortex conversion produced {len(missing)} missing mandatory keys: {missing[:20]}")


def _convert_hyena_layer(
    src: dict[str, torch.Tensor],
    dst: OrderedDict[str, torch.Tensor],
    prefix: str,
    block_prefix: str,
    symbol: str,
    te_enabled: bool,
    num_groups: int,
    medium_conv_len: int,
) -> None:
    """Convert a single hyena layer (S, D, or H) from mbridge to vortex.

    Pops consumed keys from ``src`` to free memory incrementally.
    """
    if te_enabled:
        ln_key = f"{prefix}.mixer.dense_projection.layer_norm_weight"
    else:
        ln_key = f"{prefix}.norm.weight"
    if ln_key in src:
        dst[f"{block_prefix}.pre_norm.scale"] = src.pop(ln_key)

    dense_proj_key = f"{prefix}.mixer.dense_projection.weight"
    if dense_proj_key in src:
        dst[f"{block_prefix}.projections.weight"] = src.pop(dense_proj_key)

    short_filter_key = f"{prefix}.mixer.hyena_proj_conv.short_conv_weight"
    if short_filter_key in src:
        w = src.pop(short_filter_key)
        if w.dim() == 2:
            w = w[:, None]
        dst[f"{block_prefix}.filter.short_filter_weight"] = w

    dense_weight_key = f"{prefix}.mixer.dense.weight"
    if dense_weight_key in src:
        dst[f"{block_prefix}.out_filter_dense.weight"] = src.pop(dense_weight_key)

    dense_bias_key = f"{prefix}.mixer.dense.bias"
    if dense_bias_key in src:
        dst[f"{block_prefix}.out_filter_dense.bias"] = src.pop(dense_bias_key)

    if symbol == "S":
        sc_key = f"{prefix}.mixer.mixer.short_conv.short_conv_weight"
        if sc_key in src:
            dst[f"{block_prefix}.filter.h"] = src.pop(sc_key)

    elif symbol == "D":
        conv_bias_key = f"{prefix}.mixer.mixer.conv_bias"
        if conv_bias_key in src:
            dst[f"{block_prefix}.filter.D"] = src.pop(conv_bias_key)

        h_key = f"{prefix}.mixer.mixer.filter.h"
        decay_key = f"{prefix}.mixer.mixer.filter.decay"
        if h_key in src and decay_key in src:
            h = src.pop(h_key)
            decay = src.pop(decay_key)
            trunc_len = min(medium_conv_len, h.shape[1]) if h.dim() > 1 else medium_conv_len
            h_trunc = h[:, :trunc_len] * decay[:, :trunc_len]
            del h, decay
            dst[f"{block_prefix}.filter.h"] = h_trunc.unsqueeze(1)

    elif symbol == "H":
        conv_bias_key = f"{prefix}.mixer.mixer.conv_bias"
        if conv_bias_key in src:
            dst[f"{block_prefix}.filter.D"] = src.pop(conv_bias_key)

        p_key = f"{prefix}.mixer.mixer.filter.p"
        gamma_key = f"{prefix}.mixer.mixer.filter.gamma"
        r_key = f"{prefix}.mixer.mixer.filter.R"
        if p_key in src and gamma_key in src and r_key in src:
            log_poles, residues = _compute_log_poles_and_residues(
                src.pop(p_key), src.pop(gamma_key), src.pop(r_key), num_groups
            )
            dst[f"{block_prefix}.filter.log_poles"] = log_poles
            dst[f"{block_prefix}.filter.residues"] = residues


def _compute_inv_freq(rotary_dim: int, rotary_base: float) -> torch.Tensor:
    """Compute rotary embedding inverse frequencies.

    Args:
        rotary_dim: Dimension of the rotary embeddings (hidden_size // num_attention_heads).
        rotary_base: Base for the rotary frequency computation.

    Returns:
        Tensor of shape (rotary_dim // 2,) with inverse frequencies.
    """
    return 1.0 / (rotary_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))


def _convert_attention_layer(
    src: dict[str, torch.Tensor],
    dst: OrderedDict[str, torch.Tensor],
    prefix: str,
    block_prefix: str,
    te_enabled: bool,
    rotary_dim: int,
    rotary_base: float,
) -> None:
    """Convert a single attention layer (*) from mbridge to vortex.

    Pops consumed keys from ``src`` to free memory incrementally.
    """
    if te_enabled:
        ln_key = f"{prefix}.self_attention.linear_qkv.layer_norm_weight"
    else:
        ln_key = f"{prefix}.input_layernorm.weight"
    if ln_key in src:
        dst[f"{block_prefix}.pre_norm.scale"] = src.pop(ln_key)

    qkv_key = f"{prefix}.self_attention.linear_qkv.weight"
    if qkv_key in src:
        dst[f"{block_prefix}.inner_mha_cls.Wqkv.weight"] = src.pop(qkv_key)

    proj_weight_key = f"{prefix}.self_attention.linear_proj.weight"
    if proj_weight_key in src:
        dst[f"{block_prefix}.inner_mha_cls.out_proj.weight"] = src.pop(proj_weight_key)

    proj_bias_key = f"{prefix}.self_attention.linear_proj.bias"
    if proj_bias_key in src:
        dst[f"{block_prefix}.inner_mha_cls.out_proj.bias"] = src.pop(proj_bias_key)

    dst[f"{block_prefix}.inner_mha_cls.rotary_emb.inv_freq"] = _compute_inv_freq(rotary_dim, rotary_base)


def _convert_mlp(
    src: dict[str, torch.Tensor],
    dst: OrderedDict[str, torch.Tensor],
    prefix: str,
    block_prefix: str,
    te_enabled: bool,
) -> None:
    """Convert MLP weights for a layer, splitting merged fc1 back into gate + up-proj.

    Pops consumed keys from ``src`` to free memory incrementally.
    """
    if te_enabled:
        post_norm_key = f"{prefix}.mlp.linear_fc1.layer_norm_weight"
    else:
        post_norm_key = f"{prefix}.pre_mlp_layernorm.weight"
    if post_norm_key in src:
        dst[f"{block_prefix}.post_norm.scale"] = src.pop(post_norm_key)

    fc1_key = f"{prefix}.mlp.linear_fc1.weight"
    if fc1_key in src:
        l1_weight, l2_weight = _split_fc1(src.pop(fc1_key))
        dst[f"{block_prefix}.mlp.l1.weight"] = l1_weight
        dst[f"{block_prefix}.mlp.l2.weight"] = l2_weight

    fc2_key = f"{prefix}.mlp.linear_fc2.weight"
    if fc2_key in src:
        dst[f"{block_prefix}.mlp.l3.weight"] = src.pop(fc2_key)


def mbridge_to_vortex(
    mbridge_ckpt_dir: Path,
    output_path: Path,
    model_size: str,
    te_enabled: bool = True,
) -> Path:
    """Convert an MBridge checkpoint to Vortex format.

    Args:
        mbridge_ckpt_dir: Path to the mbridge checkpoint directory.
        output_path: Path for the output .pt file.
        model_size: Model size key (e.g. 'evo2_1b_base').
        te_enabled: Whether TE fused layernorm keys are used.

    Returns:
        Path to the saved vortex .pt file.
    """
    provider_cls = HYENA_MODEL_OPTIONS[model_size]
    model_provider = provider_cls()

    logger.info(f"Loading mbridge checkpoint from {mbridge_ckpt_dir}...")
    mbridge_sd = load_mbridge_state_dict(mbridge_ckpt_dir)
    logger.info(f"Loaded {len(mbridge_sd)} keys")

    logger.info(f"Converting to vortex format (pattern={model_provider.hybrid_override_pattern})...")
    vortex_sd = mbridge_to_vortex_state_dict(mbridge_sd, model_provider, te_enabled=te_enabled)
    del mbridge_sd
    logger.info(f"Converted to {len(vortex_sd)} vortex keys")

    _validate_vortex_keys(vortex_sd, model_provider.hybrid_override_pattern)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vortex_sd, str(output_path))
    logger.info(f"Saved vortex checkpoint to {output_path}")

    config = _build_vortex_config(model_provider)
    config_path = output_path.parent / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved vortex config to {config_path}")

    return output_path


def _build_vortex_config(provider: HyenaModelProvider) -> dict:
    """Build a minimal Vortex config.json from the model provider."""
    return {
        "num_layers": provider.num_layers,
        "hidden_size": provider.hidden_size,
        "num_attention_heads": provider.num_attention_heads,
        "ffn_hidden_size": provider.ffn_hidden_size,
        "seq_length": provider.seq_length,
        "hybrid_override_pattern": provider.hybrid_override_pattern,
        "num_groups_hyena": provider.num_groups_hyena,
        "vocab_size": getattr(provider, "vocab_size", 512),
    }


def main():
    """CLI entry point for mbridge-to-vortex export."""
    parser = argparse.ArgumentParser(description="Export MBridge checkpoint to Vortex format")
    parser.add_argument("--mbridge-ckpt-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True, help="Output .pt file path")
    parser.add_argument("--model-size", type=str, choices=sorted(HYENA_MODEL_OPTIONS.keys()), required=True)
    parser.add_argument("--no-te", action="store_true", help="Disable TE fused layernorm key mapping")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    mbridge_to_vortex(
        mbridge_ckpt_dir=args.mbridge_ckpt_dir,
        output_path=args.output_path,
        model_size=args.model_size,
        te_enabled=not args.no_te,
    )


if __name__ == "__main__":
    main()
