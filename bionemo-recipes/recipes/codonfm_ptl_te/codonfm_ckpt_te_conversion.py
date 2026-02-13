#!/usr/bin/env python3

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


"""Bidirectional checkpoint converter between PyTorch checkpoint and Transformer Engine (TE) checkpoint.

For example https://huggingface.co/nvidia/NV-CodonFM-Encodon-TE-Cdwt-1B-v1 is a converted checkpoint from https://huggingface.co/nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1PyTorch checkpoint

Usage:
    python codon_fm_ckpt_convert.py --src /path/to/NV-CodonFM-Encodon-Cdwt-1B-v1.ckpt --dst /path/to/NV-CodonFM-Encodon-TE-Cdwt-1B-v1.ckpt --direction te2pytorch
"""

import argparse
import logging
import os

import torch
from safetensors.torch import save_file as safetensors_save_file

from src.utils.load_checkpoint import load_checkpoint


logger = logging.getLogger(__name__)

ALLOWED_HYPERPARAMETER_KEYS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "hidden_act",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "initializer_range",
    "layer_norm_eps",
    "pad_token_id",
    "position_embedding_type",
    "classifier_dropout",
    "rotary_theta",
    "ignore_index",
    "loss_type",
    "lora",
    "lora_alpha",
    "lora_r",
    "lora_dropout",
)

# PYTorch -> TE keymap
PYTORCH_TO_TE_KEYMAP = {
    "model.layers.*.pre_attn_layer_norm.weight": "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "model.layers.*.pre_attn_layer_norm.bias": "model.layers.*.self_attention.layernorm_qkv.layer_norm_bias",
    "model.layers.*.attention.qkv.weight": "model.layers.*.self_attention.layernorm_qkv.weight",
    "model.layers.*.attention.qkv.bias": "model.layers.*.self_attention.layernorm_qkv.bias",
    "model.layers.*.attention.rotary_emb.inv_freq": None,  # Delete in favor of TE RoPE embedding values.
    "model.layers.*.post_attn_layer_norm.weight": "model.layers.*.self_attention.proj.layer_norm_weight",
    "model.layers.*.post_attn_layer_norm.bias": "model.layers.*.self_attention.proj.layer_norm_bias",
    "model.layers.*.post_attn_dense.weight": "model.layers.*.self_attention.proj.weight",
    "model.layers.*.post_attn_dense.bias": "model.layers.*.self_attention.proj.bias",
    "model.layers.*.pre_ffn_layer_norm.weight": "model.layers.*.mlp_fc1.layer_norm_weight",
    "model.layers.*.pre_ffn_layer_norm.bias": "model.layers.*.mlp_fc1.layer_norm_bias",
    "model.layers.*.intermediate_dense.weight": "model.layers.*.mlp_fc1.weight",
    "model.layers.*.intermediate_dense.bias": "model.layers.*.mlp_fc1.bias",
    "model.layers.*.post_ffn_layer_norm.weight": "model.layers.*.mlp_fc2.layer_norm_weight",
    "model.layers.*.post_ffn_layer_norm.bias": "model.layers.*.mlp_fc2.layer_norm_bias",
    "model.layers.*.output_dense.weight": "model.layers.*.mlp_fc2.weight",
    "model.layers.*.output_dense.bias": "model.layers.*.mlp_fc2.bias",
    "model.cls.0.weight": "model.cls.0.weight",  # Linear
    "model.cls.0.bias": "model.cls.0.bias",
    "model.cls.2.weight": "model.cls.2.layer_norm_weight",  # LayerNorm -> LayerNormLinear
    "model.cls.2.bias": "model.cls.2.layer_norm_bias",
    "model.cls.3.weight": "model.cls.2.weight",  # Linear -> LayerNormLinear
    "model.cls.3.bias": "model.cls.2.bias",
}

# TE -> PyTorch keymap (inverse)
TE_TO_PYTORCH_KEYMAP = {
    "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight": "model.layers.*.pre_attn_layer_norm.weight",
    "model.layers.*.self_attention.layernorm_qkv.layer_norm_bias": "model.layers.*.pre_attn_layer_norm.bias",
    "model.layers.*.self_attention.layernorm_qkv.weight": "model.layers.*.attention.qkv.weight",
    "model.layers.*.self_attention.layernorm_qkv.bias": "model.layers.*.attention.qkv.bias",
    "model.layers.*.self_attention.proj.layer_norm_weight": "model.layers.*.post_attn_layer_norm.weight",
    "model.layers.*.self_attention.proj.layer_norm_bias": "model.layers.*.post_attn_layer_norm.bias",
    "model.layers.*.self_attention.proj.weight": "model.layers.*.post_attn_dense.weight",
    "model.layers.*.self_attention.proj.bias": "model.layers.*.post_attn_dense.bias",
    "model.layers.*.mlp_fc1.layer_norm_weight": "model.layers.*.pre_ffn_layer_norm.weight",
    "model.layers.*.mlp_fc1.layer_norm_bias": "model.layers.*.pre_ffn_layer_norm.bias",
    "model.layers.*.mlp_fc1.weight": "model.layers.*.intermediate_dense.weight",
    "model.layers.*.mlp_fc1.bias": "model.layers.*.intermediate_dense.bias",
    "model.layers.*.mlp_fc2.layer_norm_weight": "model.layers.*.post_ffn_layer_norm.weight",
    "model.layers.*.mlp_fc2.layer_norm_bias": "model.layers.*.post_ffn_layer_norm.bias",
    "model.layers.*.mlp_fc2.weight": "model.layers.*.output_dense.weight",
    "model.layers.*.mlp_fc2.bias": "model.layers.*.output_dense.bias",
    "model.cls.0.weight": "model.cls.0.weight",  # Linear
    "model.cls.0.bias": "model.cls.0.bias",
    "model.cls.2.layer_norm_weight": "model.cls.2.weight",  # LayerNormLinear -> LayerNorm
    "model.cls.2.layer_norm_bias": "model.cls.2.bias",
    "model.cls.2.weight": "model.cls.3.weight",  # LayerNormLinear -> Linear
    "model.cls.2.bias": "model.cls.3.bias",
}


def concatenate_qkv(src: dict, hyper_parameters: dict):
    """Concatenate Q, K, V weights and biases in the source state dict.

    Creates new 'qkv' keys and removes individual query/key/value keys.
    Used when converting PyTorch -> TE.
    """
    preprocessed = {}

    # Find all layer indices
    layer_indices = set()
    for key in src.keys():
        if "layers." in key and ".attention.query.weight" in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_indices.add(parts[i + 1])
                    break

    # Concatenate Q, K, V for each layer
    keys_to_skip = set()
    for layer_idx in layer_indices:
        # Concatenate weights
        q_key = f"model.layers.{layer_idx}.attention.query.weight"
        k_key = f"model.layers.{layer_idx}.attention.key.weight"
        v_key = f"model.layers.{layer_idx}.attention.value.weight"

        if all(k in src for k in [q_key, k_key, v_key]):
            q_weight = src[q_key]
            k_weight = src[k_key]
            v_weight = src[v_key]
            # Interleave Q, K, V by head
            q_weight = q_weight.reshape(
                hyper_parameters["num_attention_heads"],
                int(hyper_parameters["hidden_size"] / hyper_parameters["num_attention_heads"]),
                -1,
            )
            k_weight = k_weight.reshape(
                hyper_parameters["num_attention_heads"],
                int(hyper_parameters["hidden_size"] / hyper_parameters["num_attention_heads"]),
                -1,
            )
            v_weight = v_weight.reshape(
                hyper_parameters["num_attention_heads"],
                int(hyper_parameters["hidden_size"] / hyper_parameters["num_attention_heads"]),
                -1,
            )
            stacked = torch.stack([q_weight, k_weight, v_weight], dim=1)
            concatenated_weight = stacked.reshape(3 * hyper_parameters["hidden_size"], -1)
            preprocessed[f"model.layers.{layer_idx}.attention.qkv.weight"] = concatenated_weight
            keys_to_skip.update([q_key, k_key, v_key])
            logger.info(f"Concatenated Q, K, V weights for layer {layer_idx}")

        # Concatenate biases
        q_bias_key = f"model.layers.{layer_idx}.attention.query.bias"
        k_bias_key = f"model.layers.{layer_idx}.attention.key.bias"
        v_bias_key = f"model.layers.{layer_idx}.attention.value.bias"

        if all(k in src for k in [q_bias_key, k_bias_key, v_bias_key]):
            q_bias = src[q_bias_key]
            k_bias = src[k_bias_key]
            v_bias = src[v_bias_key]

            q_bias = q_bias.reshape(
                hyper_parameters["num_attention_heads"],
                int(hyper_parameters["hidden_size"] / hyper_parameters["num_attention_heads"]),
            )
            k_bias = k_bias.reshape(
                hyper_parameters["num_attention_heads"],
                int(hyper_parameters["hidden_size"] / hyper_parameters["num_attention_heads"]),
            )
            v_bias = v_bias.reshape(
                hyper_parameters["num_attention_heads"],
                int(hyper_parameters["hidden_size"] / hyper_parameters["num_attention_heads"]),
            )

            # Interleave Q, K, V biases by head
            stacked_bias = torch.stack([q_bias, k_bias, v_bias], dim=1)
            concatenated_bias = stacked_bias.reshape(3 * hyper_parameters["hidden_size"])
            preprocessed[f"model.layers.{layer_idx}.attention.qkv.bias"] = concatenated_bias
            keys_to_skip.update([q_bias_key, k_bias_key, v_bias_key])
            logger.info(f"Concatenated Q, K, V biases for layer {layer_idx}")

    # Copy over all other keys that weren't concatenated
    preprocessed.update({key: value for key, value in src.items() if key not in keys_to_skip})

    return preprocessed


def split_qkv(src: dict, hyper_parameters: dict):
    """Split concatenated QKV weights and biases back into separate Q, K, V tensors.

    Used when converting TE -> PyTorch.
    """
    split_dict = {}

    # Find all layer indices
    layer_indices = set()
    for key in src.keys():
        if "layers." in key and ".attention.qkv.weight" in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_indices.add(parts[i + 1])
                    break

    # Split QKV for each layer
    keys_to_skip = set()
    for layer_idx in layer_indices:
        # Split weights
        qkv_key = f"model.layers.{layer_idx}.attention.qkv.weight"

        if qkv_key in src:
            qkv_weight = src[qkv_key]  # [3*num_heads*head_dim, model_dim]
            # Reshape to [num_heads, 3, head_dim, model_dim]
            qkv_weight = qkv_weight.reshape(
                hyper_parameters["num_attention_heads"],
                3,
                int(hyper_parameters["hidden_size"] / hyper_parameters["num_attention_heads"]),
                -1,
            )
            # Extract Q, K, V
            q_weight = qkv_weight[:, 0, :, :].reshape(hyper_parameters["hidden_size"], -1)
            k_weight = qkv_weight[:, 1, :, :].reshape(hyper_parameters["hidden_size"], -1)
            v_weight = qkv_weight[:, 2, :, :].reshape(hyper_parameters["hidden_size"], -1)

            split_dict[f"model.layers.{layer_idx}.attention.query.weight"] = q_weight
            split_dict[f"model.layers.{layer_idx}.attention.key.weight"] = k_weight
            split_dict[f"model.layers.{layer_idx}.attention.value.weight"] = v_weight
            keys_to_skip.add(qkv_key)
            logger.info(f"Split QKV weights for layer {layer_idx}")

        # Split biases
        qkv_bias_key = f"model.layers.{layer_idx}.attention.qkv.bias"

        if qkv_bias_key in src:
            qkv_bias = src[qkv_bias_key]  # [3*num_heads*head_dim]
            # Reshape to [num_heads, 3, head_dim]
            qkv_bias = qkv_bias.reshape(
                hyper_parameters["num_attention_heads"],
                3,
                int(hyper_parameters["hidden_size"] / hyper_parameters["num_attention_heads"]),
            )
            # Extract Q, K, V
            q_bias = qkv_bias[:, 0, :].reshape(hyper_parameters["hidden_size"])
            k_bias = qkv_bias[:, 1, :].reshape(hyper_parameters["hidden_size"])
            v_bias = qkv_bias[:, 2, :].reshape(hyper_parameters["hidden_size"])

            split_dict[f"model.layers.{layer_idx}.attention.query.bias"] = q_bias
            split_dict[f"model.layers.{layer_idx}.attention.key.bias"] = k_bias
            split_dict[f"model.layers.{layer_idx}.attention.value.bias"] = v_bias
            keys_to_skip.add(qkv_bias_key)
            logger.info(f"Split QKV biases for layer {layer_idx}")

    # Copy over all other keys that weren't split
    split_dict.update({key: value for key, value in src.items() if key not in keys_to_skip})

    return split_dict


def convert_state_dict(src: dict, keymap: dict):
    """Convert the src state dictionary according to the keymap.

    Asterisks in the key-map will be replaced by the layer index
    during mapping and conversion.
    """
    dst_state_dict = {}
    for name, weight in src.items():
        # Locate the mapping for the parameter name.
        keymap_matched = False
        for source_key, target_key in keymap.items():
            if "*" in source_key:
                # Parse '*' matches.
                assert target_key is None or "*" in target_key, "Source and target asterisk usage must be consistent."
                prefix_postfix = source_key.split("*")
                assert len(prefix_postfix) == 2, f"Only one '*' is permitted in the key-map: {source_key}"
                if name.startswith(prefix_postfix[0]) and name.endswith(prefix_postfix[1]):
                    if target_key is None:
                        # Delete the parameter in the source,
                        # do not set in the target.
                        keymap_matched = True
                        break
                    # Matched pattern. Update asterisk value in target key.
                    source_ast_location = len(prefix_postfix[0].split(".")) - 1
                    source_ast_value = name.split(".")[source_ast_location]
                    target_key_ast_split = target_key.split("*")
                    target_key = target_key_ast_split[0] + source_ast_value + target_key_ast_split[1]  # noqa: PLW2901
                    dst_state_dict[target_key] = weight
                    keymap_matched = True
                    logger.info(f"Mapped {name} -> {target_key}.")
                    break
            elif name == source_key:
                if target_key is None:
                    # Delete the parameter in the source,
                    # do not set in the target.
                    keymap_matched = True
                    break
                # Parse non-asterisk matches.
                assert target_key is None or "*" not in target_key, (
                    "Source and target asterisk usage must be consistent."
                )
                dst_state_dict[target_key] = weight
                keymap_matched = True
                logger.info(f"Mapped {name} -> {target_key}.")
                break
        if not keymap_matched:
            # No-op. Copy the original name and weight into the target state dict.
            dst_state_dict[name] = weight
            logger.info(f"Converted {name} as-is without any mapping.")

    return dst_state_dict


def filter_hyper_parameters(hyper_parameters: dict) -> dict:
    """Keep only conversion-compatible hyperparameter keys."""
    return {key: value for key, value in hyper_parameters.items() if key in ALLOWED_HYPERPARAMETER_KEYS}


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)

    # Parse arguments.
    parser = argparse.ArgumentParser(description="Bidirectional checkpoint converter between PyTorch and TE")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source checkpoint path (can be .ckpt file or directory with safetensors)",
    )
    parser.add_argument("--dst", type=str, required=True, help="Destination checkpoint path for .ckpt file")
    parser.add_argument(
        "--direction",
        type=str,
        required=True,
        choices=["pytorch2te", "te2pytorch"],
        help="Conversion direction: 'pytorch2te' or 'te2pytorch'",
    )
    args = parser.parse_args()

    # Load source checkpoint (automatically detects format)
    logger.info(f"Loading checkpoint from {args.src}")
    src_checkpoint = load_checkpoint(args.src, map_location="cpu")
    src_checkpoint["hyper_parameters"] = filter_hyper_parameters(src_checkpoint["hyper_parameters"])

    # Perform conversion based on direction
    if args.direction == "pytorch2te":
        logger.info("Converting PyTorch -> TE")
        # Step 1: Concatenate Q, K, V
        preprocessed_state_dict = concatenate_qkv(src_checkpoint["state_dict"], src_checkpoint["hyper_parameters"])
        # Step 2: Apply keymap
        dst_state_dict = convert_state_dict(preprocessed_state_dict, PYTORCH_TO_TE_KEYMAP)
    else:  # te2pytorch
        logger.info("Converting TE -> PyTorch")
        # Step 1: Apply keymap
        converted_state_dict = convert_state_dict(src_checkpoint["state_dict"], TE_TO_PYTORCH_KEYMAP)
        # Step 2: Split QKV
        dst_state_dict = split_qkv(converted_state_dict, src_checkpoint["hyper_parameters"])

    # Prepare final checkpoint
    dst_checkpoint = {
        "state_dict": dst_state_dict,
        "hyper_parameters": src_checkpoint["hyper_parameters"],
    }

    # Save the converted checkpoint in pickled format
    torch.save(dst_checkpoint, args.dst)
    logger.info(f"Successfully converted checkpoint saved to {args.dst}")

    # Save the state_dict in safetensors format alongside the .ckpt file
    safetensors_path = os.path.splitext(args.dst)[0] + ".safetensors"
    safetensors_save_file(dst_state_dict, safetensors_path)
    logger.info(f"Successfully saved safetensors checkpoint to {safetensors_path}")


if __name__ == "__main__":
    main()
