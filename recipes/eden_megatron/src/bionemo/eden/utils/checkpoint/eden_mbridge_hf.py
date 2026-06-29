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

"""Eden (Llama) MBridge <-> HuggingFace checkpoint converters.

Pure state-dict manipulation — no distributed init, no model instantiation,
no GPU required.

CLI entry points
~~~~~~~~~~~~~~~~
* ``eden_export_mbridge_to_hf``  — MBridge DCP  ->  HuggingFace Llama
* ``eden_convert_hf_to_mbridge`` — HuggingFace Llama  ->  MBridge DCP
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter

from bionemo.eden.models.eden_provider import EDEN_MODEL_OPTIONS
from bionemo.eden.utils.checkpoint.mbridge_checkpoint_utils import load_mbridge_state_dict


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QKV interleaving helpers  (mirrors NeMo TransformFns.merge_qkv / split_qkv)
# ---------------------------------------------------------------------------


def _merge_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Interleave separate Q, K, V projections into Megatron's packed ``linear_qkv``.

    Layout: ``[q_group0..., k0, v0, q_group1..., k1, v1, ...]``
    where each group has ``heads_per_group`` Q slices followed by one K and one V.
    """
    heads_per_group = num_heads // num_kv_heads
    head_dim = q.shape[0] // num_heads
    q = q.view(num_heads, head_dim, -1)
    k = k.view(num_kv_heads, head_dim, -1)
    v = v.view(num_kv_heads, head_dim, -1)

    chunks = []
    for i in range(num_kv_heads):
        chunks.append(q[i * heads_per_group : (i + 1) * heads_per_group])
        chunks.append(k[i : i + 1])
        chunks.append(v[i : i + 1])

    qkv = torch.cat(chunks, dim=0)
    return qkv.reshape(-1, qkv.shape[-1])


def _split_qkv(
    qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved ``linear_qkv`` back into separate Q, K, V."""
    heads_per_group = num_heads // num_kv_heads
    head_dim = qkv.shape[0] // (num_heads + 2 * num_kv_heads)
    qkv_total_dim = num_heads + 2 * num_kv_heads
    qkv_3d = qkv.reshape(qkv_total_dim, head_dim, -1)
    hidden = qkv_3d.shape[-1]

    q_slices = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_kv_heads)
        ]
    )
    k_slices = torch.arange(heads_per_group, qkv_total_dim, heads_per_group + 2)
    v_slices = torch.arange(heads_per_group + 1, qkv_total_dim, heads_per_group + 2)

    return (
        qkv_3d[q_slices].reshape(-1, hidden),
        qkv_3d[k_slices].reshape(-1, hidden),
        qkv_3d[v_slices].reshape(-1, hidden),
    )


# ---------------------------------------------------------------------------
# Export: MBridge -> HuggingFace
# ---------------------------------------------------------------------------


def _unstack_layers(mbridge_sd: dict[str, torch.Tensor], num_layers: int) -> None:
    """Convert stacked layer tensors to per-layer indexed keys in place.

    MBridge DCP checkpoints may store layer weights as stacked tensors with
    shape ``[num_layers, ...]`` under keys like ``decoder.layers.mlp.linear_fc1.weight``.
    This function splits them into indexed keys like
    ``decoder.layers.0.mlp.linear_fc1.weight``, ``decoder.layers.1.mlp.linear_fc1.weight``, etc.
    """
    prefix = "decoder.layers."
    stacked_keys = [k for k in list(mbridge_sd.keys()) if k.startswith(prefix) and not k[len(prefix) :][0].isdigit()]
    for key in stacked_keys:
        tensor = mbridge_sd.pop(key)
        suffix = key[len(prefix) :]
        if tensor.shape[0] != num_layers:
            logger.warning(
                f"Stacked key {key} has leading dim {tensor.shape[0]} but expected {num_layers} layers, skipping"
            )
            mbridge_sd[key] = tensor
            continue
        for i in range(num_layers):
            mbridge_sd[f"decoder.layers.{i}.{suffix}"] = tensor[i]


def _stack_layers(mbridge_sd: dict[str, torch.Tensor], num_layers: int) -> None:
    """Convert per-layer indexed keys to stacked tensors in place.

    Reverses :func:`_unstack_layers`: gathers ``decoder.layers.{i}.X`` for
    ``i`` in ``0..num_layers-1`` and stacks them into a single
    ``decoder.layers.X`` tensor with shape ``[num_layers, ...]``.

    This produces the format that Megatron expects for homogeneous-layer models.
    """
    import re

    prefix = "decoder.layers."
    indexed_keys = [k for k in mbridge_sd if k.startswith(prefix) and re.match(r"\d+\.", k[len(prefix) :])]

    suffixes: set[str] = set()
    for key in indexed_keys:
        rest = key[len(prefix) :]
        suffix = re.sub(r"^\d+\.", "", rest)
        suffixes.add(suffix)

    for suffix in sorted(suffixes):
        layer_tensors = []
        for i in range(num_layers):
            indexed_key = f"decoder.layers.{i}.{suffix}"
            if indexed_key not in mbridge_sd:
                logger.warning(f"Missing indexed key {indexed_key} during stacking, skipping suffix {suffix}")
                break
            layer_tensors.append(mbridge_sd[indexed_key])
        else:
            for i in range(num_layers):
                del mbridge_sd[f"decoder.layers.{i}.{suffix}"]
            mbridge_sd[f"decoder.layers.{suffix}"] = torch.stack(layer_tensors)


def mbridge_to_hf_state_dict(
    mbridge_sd: dict[str, torch.Tensor],
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    te_enabled: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert an Eden MBridge state dict to HuggingFace LlamaForCausalLM keys.

    Args:
        mbridge_sd: State dict loaded from an MBridge DCP checkpoint.
            Handles both indexed keys (``decoder.layers.0.…``) and stacked
            keys (``decoder.layers.mlp.…`` with a leading ``num_layers`` dim).
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads (Q heads).
        num_kv_heads: Number of key/value heads (GQA groups).
        te_enabled: Whether TE fused layernorm keys are used.

    Returns:
        State dict with HuggingFace ``model.layers.*`` keys.
    """
    _unstack_layers(mbridge_sd, num_layers)

    hf_sd: dict[str, torch.Tensor] = {}

    hf_sd["model.embed_tokens.weight"] = mbridge_sd.pop("embedding.word_embeddings.weight")

    if "output_layer.weight" in mbridge_sd:
        hf_sd["lm_head.weight"] = mbridge_sd.pop("output_layer.weight")

    final_norm_key = "decoder.final_layernorm.weight"
    if final_norm_key in mbridge_sd:
        hf_sd["model.norm.weight"] = mbridge_sd.pop(final_norm_key)

    for i in range(num_layers):
        pfx = f"decoder.layers.{i}"
        hpfx = f"model.layers.{i}"

        # Input layernorm
        if te_enabled:
            ln_key = f"{pfx}.self_attention.linear_qkv.layer_norm_weight"
        else:
            ln_key = f"{pfx}.input_layernorm.weight"
        if ln_key in mbridge_sd:
            hf_sd[f"{hpfx}.input_layernorm.weight"] = mbridge_sd.pop(ln_key)

        # Post-attention layernorm (pre-MLP)
        if te_enabled:
            post_ln_key = f"{pfx}.mlp.linear_fc1.layer_norm_weight"
        else:
            post_ln_key = f"{pfx}.pre_mlp_layernorm.weight"
        if post_ln_key in mbridge_sd:
            hf_sd[f"{hpfx}.post_attention_layernorm.weight"] = mbridge_sd.pop(post_ln_key)

        # QKV: split interleaved -> separate q/k/v
        qkv_key = f"{pfx}.self_attention.linear_qkv.weight"
        if qkv_key in mbridge_sd:
            q, k, v = _split_qkv(mbridge_sd.pop(qkv_key), num_heads, num_kv_heads)
            hf_sd[f"{hpfx}.self_attn.q_proj.weight"] = q
            hf_sd[f"{hpfx}.self_attn.k_proj.weight"] = k
            hf_sd[f"{hpfx}.self_attn.v_proj.weight"] = v

        # Output projection
        proj_key = f"{pfx}.self_attention.linear_proj.weight"
        if proj_key in mbridge_sd:
            hf_sd[f"{hpfx}.self_attn.o_proj.weight"] = mbridge_sd.pop(proj_key)

        # MLP: split fused fc1 -> gate + up
        fc1_key = f"{pfx}.mlp.linear_fc1.weight"
        if fc1_key in mbridge_sd:
            gate, up = torch.chunk(mbridge_sd.pop(fc1_key), 2, dim=0)
            hf_sd[f"{hpfx}.mlp.gate_proj.weight"] = gate
            hf_sd[f"{hpfx}.mlp.up_proj.weight"] = up

        fc2_key = f"{pfx}.mlp.linear_fc2.weight"
        if fc2_key in mbridge_sd:
            hf_sd[f"{hpfx}.mlp.down_proj.weight"] = mbridge_sd.pop(fc2_key)

    unmapped = {k for k in mbridge_sd if "_extra_state" not in k}
    if unmapped:
        logger.warning(f"Unmapped MBridge keys ({len(unmapped)}): {sorted(unmapped)[:20]}")

    return hf_sd


def export_mbridge_to_hf(ckpt_dir: Path, hf_output_dir: Path, model_size: str, te_enabled: bool = True) -> Path:
    """Export an Eden MBridge checkpoint to HuggingFace format.

    Args:
        ckpt_dir: Path to the MBridge checkpoint (``iter_XXXXXXX`` or parent).
        hf_output_dir: Destination directory for the HF checkpoint.
        model_size: Eden model size key (e.g. ``eden_7b``).
        te_enabled: Whether TE fused layernorm keys are used.

    Returns:
        *hf_output_dir* after a successful export.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    provider_cls = EDEN_MODEL_OPTIONS[model_size]
    provider = provider_cls()

    logger.info(f"Loading MBridge checkpoint from {ckpt_dir}...")
    mbridge_sd = load_mbridge_state_dict(ckpt_dir)
    logger.info(f"Loaded {len(mbridge_sd)} keys")

    hf_sd = mbridge_to_hf_state_dict(
        mbridge_sd,
        num_layers=provider.num_layers,
        num_heads=provider.num_attention_heads,
        num_kv_heads=provider.num_query_groups,
        te_enabled=te_enabled,
    )
    del mbridge_sd
    logger.info(f"Converted to {len(hf_sd)} HF keys")

    hf_config = LlamaConfig(
        hidden_size=provider.hidden_size,
        intermediate_size=provider.ffn_hidden_size,
        num_hidden_layers=provider.num_layers,
        num_attention_heads=provider.num_attention_heads,
        num_key_value_heads=provider.num_query_groups,
        vocab_size=getattr(provider, "vocab_size", 512),
        max_position_embeddings=provider.seq_length,
        rms_norm_eps=provider.layernorm_epsilon,
        rope_theta=provider.rotary_base,
        torch_dtype=torch.bfloat16,
        tie_word_embeddings=provider.share_embeddings_and_output_weights,
    )
    hf_config.architectures = ["LlamaForCausalLM"]

    hf_output_dir = Path(hf_output_dir)
    hf_output_dir.mkdir(parents=True, exist_ok=True)

    hf_model = LlamaForCausalLM(hf_config)
    hf_model.load_state_dict(hf_sd, strict=False)
    hf_model.save_pretrained(str(hf_output_dir))
    del hf_model

    logger.info(f"Exported MBridge -> HF at {hf_output_dir}")
    return hf_output_dir


# ---------------------------------------------------------------------------
# Import: HuggingFace -> MBridge
# ---------------------------------------------------------------------------


def hf_to_mbridge_state_dict(
    hf_sd: dict[str, torch.Tensor],
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    te_enabled: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert a HuggingFace LlamaForCausalLM state dict to MBridge keys.

    Produces the stacked layer format that Megatron uses for homogeneous
    models: ``decoder.layers.mlp.…`` with shape ``[num_layers, ...]``.

    Args:
        hf_sd: State dict from a HuggingFace Llama model.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads (Q heads).
        num_kv_heads: Number of key/value heads (GQA groups).
        te_enabled: Whether TE fused layernorm keys are used.

    Returns:
        State dict with MBridge-style stacked layer keys.
    """
    mbridge_sd: dict[str, torch.Tensor] = {}

    mbridge_sd["embedding.word_embeddings.weight"] = hf_sd.pop("model.embed_tokens.weight")

    if "lm_head.weight" in hf_sd:
        mbridge_sd["output_layer.weight"] = hf_sd.pop("lm_head.weight")

    if "model.norm.weight" in hf_sd:
        mbridge_sd["decoder.final_layernorm.weight"] = hf_sd.pop("model.norm.weight")

    for i in range(num_layers):
        hpfx = f"model.layers.{i}"
        pfx = f"decoder.layers.{i}"

        # Input layernorm
        ln_src = f"{hpfx}.input_layernorm.weight"
        if ln_src in hf_sd:
            if te_enabled:
                mbridge_sd[f"{pfx}.self_attention.linear_qkv.layer_norm_weight"] = hf_sd.pop(ln_src)
            else:
                mbridge_sd[f"{pfx}.input_layernorm.weight"] = hf_sd.pop(ln_src)

        # Post-attention layernorm
        post_ln_src = f"{hpfx}.post_attention_layernorm.weight"
        if post_ln_src in hf_sd:
            if te_enabled:
                mbridge_sd[f"{pfx}.mlp.linear_fc1.layer_norm_weight"] = hf_sd.pop(post_ln_src)
            else:
                mbridge_sd[f"{pfx}.pre_mlp_layernorm.weight"] = hf_sd.pop(post_ln_src)

        # QKV: merge separate q/k/v -> interleaved
        q_key = f"{hpfx}.self_attn.q_proj.weight"
        k_key = f"{hpfx}.self_attn.k_proj.weight"
        v_key = f"{hpfx}.self_attn.v_proj.weight"
        if q_key in hf_sd and k_key in hf_sd and v_key in hf_sd:
            qkv = _merge_qkv(hf_sd.pop(q_key), hf_sd.pop(k_key), hf_sd.pop(v_key), num_heads, num_kv_heads)
            mbridge_sd[f"{pfx}.self_attention.linear_qkv.weight"] = qkv

        # Output projection
        proj_src = f"{hpfx}.self_attn.o_proj.weight"
        if proj_src in hf_sd:
            mbridge_sd[f"{pfx}.self_attention.linear_proj.weight"] = hf_sd.pop(proj_src)

        # MLP: merge gate + up -> fused fc1
        gate_key = f"{hpfx}.mlp.gate_proj.weight"
        up_key = f"{hpfx}.mlp.up_proj.weight"
        if gate_key in hf_sd and up_key in hf_sd:
            mbridge_sd[f"{pfx}.mlp.linear_fc1.weight"] = torch.cat([hf_sd.pop(gate_key), hf_sd.pop(up_key)], dim=0)

        down_key = f"{hpfx}.mlp.down_proj.weight"
        if down_key in hf_sd:
            mbridge_sd[f"{pfx}.mlp.linear_fc2.weight"] = hf_sd.pop(down_key)

    unmapped = set(hf_sd)
    if unmapped:
        logger.warning(f"Unmapped HF keys ({len(unmapped)}): {sorted(unmapped)[:20]}")

    _stack_layers(mbridge_sd, num_layers)

    return mbridge_sd


def convert_hf_to_mbridge(
    hf_model_dir: Path,
    mbridge_ckpt_dir: Path,
    model_size: str,
    te_enabled: bool = True,
) -> Path:
    """Convert a HuggingFace Llama checkpoint to MBridge DCP format.

    Args:
        hf_model_dir: Directory containing a HuggingFace Llama checkpoint.
        mbridge_ckpt_dir: Destination directory for the MBridge DCP checkpoint.
        model_size: Eden model size key (e.g. ``eden_7b``).
        te_enabled: Whether TE fused layernorm keys are used.

    Returns:
        *mbridge_ckpt_dir* after a successful conversion.
    """
    from safetensors.torch import load_file as load_safetensors

    provider_cls = EDEN_MODEL_OPTIONS[model_size]
    provider = provider_cls()

    # Load HF state dict (try safetensors first, fall back to bin)
    hf_model_dir = Path(hf_model_dir)
    safetensor_files = sorted(hf_model_dir.glob("*.safetensors"))
    bin_files = sorted(hf_model_dir.glob("*.bin"))

    if safetensor_files:
        logger.info(f"Loading {len(safetensor_files)} safetensors file(s)...")
        hf_sd: dict[str, torch.Tensor] = {}
        for sf in safetensor_files:
            hf_sd.update(load_safetensors(str(sf), device="cpu"))
    elif bin_files:
        logger.info(f"Loading {len(bin_files)} bin file(s)...")
        hf_sd = {}
        for bf in bin_files:
            hf_sd.update(torch.load(str(bf), map_location="cpu", weights_only=True))
    else:
        raise FileNotFoundError(f"No .safetensors or .bin weight files in {hf_model_dir}")

    logger.info(f"Loaded {len(hf_sd)} HF keys")

    mbridge_sd = hf_to_mbridge_state_dict(
        hf_sd,
        num_layers=provider.num_layers,
        num_heads=provider.num_attention_heads,
        num_kv_heads=provider.num_query_groups,
        te_enabled=te_enabled,
    )
    del hf_sd
    logger.info(f"Converted to {len(mbridge_sd)} MBridge keys")

    mbridge_ckpt_dir = Path(mbridge_ckpt_dir)
    mbridge_ckpt_dir.mkdir(parents=True, exist_ok=True)
    iter_dir = mbridge_ckpt_dir / "iter_0000001"
    iter_dir.mkdir(parents=True, exist_ok=True)

    writer = FileSystemWriter(str(iter_dir), single_file_per_rank=False, thread_count=os.cpu_count())
    dcp.save(state_dict=mbridge_sd, storage_writer=writer, no_dist=True)
    del mbridge_sd

    with open(mbridge_ckpt_dir / "latest_checkpointed_iteration.txt", "w") as f:
        f.write("1\n")

    logger.info(f"Converted HF -> MBridge at {mbridge_ckpt_dir}")
    return mbridge_ckpt_dir


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def main_export():
    """CLI entry point for ``eden_export_mbridge_to_hf``."""
    parser = argparse.ArgumentParser(description="Export an Eden MBridge checkpoint to HuggingFace Llama format")
    parser.add_argument(
        "--mbridge-ckpt-dir",
        type=Path,
        required=True,
        help="Path to the MBridge checkpoint directory (iter_XXXXXXX or parent)",
    )
    parser.add_argument(
        "--hf-output-dir",
        type=Path,
        required=True,
        help="Destination directory for the HuggingFace checkpoint",
    )
    parser.add_argument("--model-size", type=str, choices=sorted(EDEN_MODEL_OPTIONS.keys()), required=True)
    parser.add_argument("--no-te", action="store_true", help="Disable TE fused layernorm key mapping")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    export_mbridge_to_hf(args.mbridge_ckpt_dir, args.hf_output_dir, args.model_size, te_enabled=not args.no_te)


def main_import():
    """CLI entry point for ``eden_convert_hf_to_mbridge``."""
    parser = argparse.ArgumentParser(description="Convert a HuggingFace Llama checkpoint to Eden MBridge format")
    parser.add_argument(
        "--hf-model-dir",
        type=Path,
        required=True,
        help="Path to HuggingFace checkpoint directory (config.json + weights)",
    )
    parser.add_argument(
        "--mbridge-ckpt-dir",
        type=Path,
        required=True,
        help="Destination directory for the MBridge DCP checkpoint",
    )
    parser.add_argument("--model-size", type=str, choices=sorted(EDEN_MODEL_OPTIONS.keys()), required=True)
    parser.add_argument("--no-te", action="store_true", help="Disable TE fused layernorm key mapping")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    convert_hf_to_mbridge(args.hf_model_dir, args.mbridge_ckpt_dir, args.model_size, te_enabled=not args.no_te)


if __name__ == "__main__":
    main_export()
