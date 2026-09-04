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

"""Conversion utilities between HuggingFace Mixtral and TransformerEngine formats."""

import inspect
from typing import List

import torch
from transformers import MixtralConfig, MixtralForCausalLM

import state
from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM


GLU_INTERLEAVE_SIZE = 32

_BASE_MAPPING = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.layers.*.input_layernorm.weight": "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "model.layers.*.self_attn.o_proj.weight": "model.layers.*.self_attention.proj.weight",
    "model.layers.*.post_attention_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
    "model.layers.*.mlp.gate.weight": "model.layers.*.mlp.gate.weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

_GROUPED_LINEAR_EXPERT_MAPPING = {
    "model.layers.*.mlp.experts.gate_up_proj": "model.layers.*.mlp.experts_gate_up_weight",
    "model.layers.*.mlp.experts.down_proj": "model.layers.*.mlp.experts_down_weight",
}

_QKV_TRANSFORM = state.state_transform(
    source_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="model.layers.*.self_attention.layernorm_qkv.weight",
    fn=state.TransformFns.merge_qkv,
)

_QKV_REVERSE_TRANSFORM = state.state_transform(
    source_key="model.layers.*.self_attention.layernorm_qkv.weight",
    target_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
    fn=state.TransformFns.split_qkv,
)


def interleave_glu_gate_up(gate_up: torch.Tensor, block_size: int = GLU_INTERLEAVE_SIZE) -> torch.Tensor:
    """Convert HF gate|up concat layout to ScaledSwiGLU GLU-interleaved layout.

    HF stores gate (w1) and up (w3) concatenated along the output dimension as
    ``[all w1 | all w3]``. ``ScaledSwiGLU(glu_interleave_size=block_size)`` expects
    ``[w1 block, w3 block, w1 block, w3 block, ...]``.
    """
    inter_size = gate_up.shape[0] // 2
    if inter_size % block_size != 0:
        raise ValueError(f"intermediate_size ({inter_size}) must be divisible by glu_interleave_size ({block_size})")

    out = gate_up.new_empty(gate_up.shape)
    num_blocks = inter_size // block_size
    for block_idx in range(num_blocks):
        gate_start = block_idx * block_size
        gate_end = gate_start + block_size
        up_start = inter_size + gate_start
        out_start = 2 * block_idx * block_size
        out[out_start : out_start + block_size] = gate_up[gate_start:gate_end]
        out[out_start + block_size : out_start + 2 * block_size] = gate_up[up_start : up_start + block_size]
    return out


def deinterleave_glu_gate_up(gate_up: torch.Tensor, block_size: int = GLU_INTERLEAVE_SIZE) -> torch.Tensor:
    """Inverse of :func:`interleave_glu_gate_up`."""
    inter_size = gate_up.shape[0] // 2
    if inter_size % block_size != 0:
        raise ValueError(f"intermediate_size ({inter_size}) must be divisible by glu_interleave_size ({block_size})")

    out = gate_up.new_empty(gate_up.shape)
    num_blocks = inter_size // block_size
    for block_idx in range(num_blocks):
        gate_start = block_idx * block_size
        gate_end = gate_start + block_size
        up_start = inter_size + gate_start
        in_start = 2 * block_idx * block_size
        out[gate_start:gate_end] = gate_up[in_start : in_start + block_size]
        out[up_start : up_start + block_size] = gate_up[in_start + block_size : in_start + 2 * block_size]
    return out


def split_fused_experts_gate_up(gate_up_proj: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Split stacked HF gate_up_proj into per-expert GLU-interleaved TE weights."""
    return tuple(interleave_glu_gate_up(expert_weight) for expert_weight in gate_up_proj)


def merge_fused_experts_gate_up(*expert_weights: torch.Tensor) -> torch.Tensor:
    """Merge per-expert GLU-interleaved TE weights into stacked HF gate_up_proj."""
    return torch.stack([deinterleave_glu_gate_up(weight) for weight in expert_weights])


def split_fused_experts_down(down_proj: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Split stacked HF down_proj into per-expert TE weights."""
    return tuple(expert_weight for expert_weight in down_proj)


def merge_fused_experts_down(*expert_weights: torch.Tensor) -> torch.Tensor:
    """Merge per-expert TE down weights into stacked HF down_proj."""
    return torch.stack(list(expert_weights))


def _get_expert_ffn_mode(config_kwargs: dict, model=None) -> str:
    """Resolve the expert representation to convert to/from.

    When converting TE->HF we read it from the live TE model's config; when converting HF->TE we take
    it from the caller's config overrides (defaulting to ``grouped_linear``). This picks which expert
    key mapping + transforms ``_build_mapping`` / ``_build_transforms`` use.
    """
    if model is not None and hasattr(model, "config"):
        return getattr(model.config, "expert_ffn_mode", "grouped_linear")
    return config_kwargs.get("expert_ffn_mode", "grouped_linear")


def _build_mapping(expert_ffn_mode: str) -> dict:
    mapping = dict(_BASE_MAPPING)
    if expert_ffn_mode == "grouped_linear":
        mapping.update(_GROUPED_LINEAR_EXPERT_MAPPING)
    return mapping


def _build_transforms(expert_ffn_mode: str, *, hf_to_te: bool) -> list:
    """Build the multi-key state-dict transforms for the chosen expert representation and direction.

    Always includes the QKV merge/split (HF stores separate q/k/v; TE fuses them). For
    ``fused_grouped_mlp`` it additionally maps HF's stacked ``gate_up_proj`` / ``down_proj`` to TE's
    discrete per-expert ``weight{i}`` params, and interleaves the gate/up blocks to match
    ``ScaledSwiGLU(glu_interleave_size=32)`` (``grouped_linear`` keeps the stacked layout, handled by
    the plain key mapping instead). ``hf_to_te`` selects the forward or inverse transform.
    """
    transforms = [_QKV_TRANSFORM if hf_to_te else _QKV_REVERSE_TRANSFORM]
    if expert_ffn_mode == "fused_grouped_mlp":
        if hf_to_te:
            transforms.extend(
                [
                    state.state_transform(
                        source_key="model.layers.*.mlp.experts.gate_up_proj",
                        target_key="model.layers.*.mlp.experts_gate_up.weight*",
                        fn=split_fused_experts_gate_up,
                    ),
                    state.state_transform(
                        source_key="model.layers.*.mlp.experts.down_proj",
                        target_key="model.layers.*.mlp.experts_down.weight*",
                        fn=split_fused_experts_down,
                    ),
                ]
            )
        else:
            transforms.extend(
                [
                    state.state_transform(
                        source_key="model.layers.*.mlp.experts_gate_up.weight*",
                        target_key="model.layers.*.mlp.experts.gate_up_proj",
                        fn=merge_fused_experts_gate_up,
                    ),
                    state.state_transform(
                        source_key="model.layers.*.mlp.experts_down.weight*",
                        target_key="model.layers.*.mlp.experts.down_proj",
                        fn=merge_fused_experts_down,
                    ),
                ]
            )
    return transforms


def _fused_expert_state_dict_ignored_entries(model_te: NVMixtralForCausalLM) -> List[str]:
    """Duplicate Sequential op weights alias discrete expert modules; ignore in conversion."""
    return [key for key in model_te.state_dict() if "._experts_ffn_op." in key and ".weight" in key]


def convert_mixtral_hf_to_te(model_hf: MixtralForCausalLM, **config_kwargs) -> NVMixtralForCausalLM:
    """Convert a Hugging Face Mixtral model to a Transformer Engine model.

    Args:
        model_hf: The Hugging Face Mixtral model.
        **config_kwargs: Additional configuration kwargs to be passed to NVMixtralConfig.

    Returns:
        The Transformer Engine Mixtral model.
    """
    expert_ffn_mode = _get_expert_ffn_mode(config_kwargs)
    te_config = NVMixtralConfig(**model_hf.config.to_dict(), **config_kwargs)
    with torch.device("meta"):
        model_te = NVMixtralForCausalLM(te_config)

    ignored_entries = (
        _fused_expert_state_dict_ignored_entries(model_te) if expert_ffn_mode == "fused_grouped_mlp" else None
    )

    output_model = state.apply_transforms(
        model_hf,
        model_te,
        _build_mapping(expert_ffn_mode),
        _build_transforms(expert_ffn_mode, hf_to_te=True),
        state_dict_ignored_entries=ignored_entries,
    )

    output_model.model.rotary_emb.inv_freq = model_hf.model.rotary_emb.inv_freq.clone()

    return output_model


def convert_mixtral_te_to_hf(model_te: NVMixtralForCausalLM, **config_kwargs) -> MixtralForCausalLM:
    """Convert a Transformer Engine Mixtral model to a Hugging Face model.

    Args:
        model_te: The Transformer Engine Mixtral model.
        **config_kwargs: Additional configuration kwargs to be passed to MixtralConfig.

    Returns:
        The Hugging Face Mixtral model.
    """
    expert_ffn_mode = _get_expert_ffn_mode(config_kwargs, model_te)
    te_config_dict = model_te.config.to_dict()
    valid_keys = set(inspect.signature(MixtralConfig.__init__).parameters)
    filtered_config = {k: v for k, v in te_config_dict.items() if k in valid_keys}
    cast_dtype = None
    if expert_ffn_mode == "fused_grouped_mlp":
        source_dtype = getattr(model_te.config, "dtype", None) or getattr(model_te.config, "torch_dtype", None)
        if source_dtype is not None and filtered_config.get("dtype") is None:
            filtered_config["dtype"] = source_dtype
        cast_dtype = source_dtype
    hf_config = MixtralConfig(**filtered_config, **config_kwargs)

    with torch.device("meta"):
        model_hf = MixtralForCausalLM(hf_config)

    output_model = state.apply_transforms(
        model_te,
        model_hf,
        {v: k for k, v in _build_mapping(expert_ffn_mode).items()},
        _build_transforms(expert_ffn_mode, hf_to_te=False),
        cast_dtype=cast_dtype,
    )

    output_model.model.rotary_emb.inv_freq = model_te.model.rotary_emb.inv_freq.clone()
    output_model.tie_weights()

    return output_model
