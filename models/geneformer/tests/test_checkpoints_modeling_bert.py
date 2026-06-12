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

"""Tests for Geneformer checkpoint integration with Transformer Engine models."""

import pytest
import torch
from transformers import AutoModelForMaskedLM, set_seed


def load_geneformer_model(model_name):
    """Helper function to load the correct Geneformer model variant."""
    if model_name == "Geneformer-V2-316M":
        # Default model (no subfolder needed)
        return AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer", revision="f45a6c7d")
    else:
        # Use subfolder for specific model variants
        return AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer", subfolder=model_name, revision="f45a6c7d")


# Model variants with detailed information
MODEL_VARIANTS = [
    (
        "Geneformer-V1-10M",
        {
            "description": "10M parameters, June 2021",
            "input_size": 2048,
            "vocabulary": "~25K protein-coding or non-coding RNA genes",
            "training_data": "~30M human single cell transcriptomes",
        },
    ),
    (
        "Geneformer-V2-104M",
        {
            "description": "104M parameters, Dec 2024",
            "input_size": 4096,
            "vocabulary": "~20K protein-coding genes",
            "training_data": "~104M human single cell transcriptomes",
        },
    ),
    (
        "Geneformer-V2-104M_CLcancer",
        {
            "description": "104M parameters, Dec 2024, cancer cell line specific",
            "input_size": 4096,
            "vocabulary": "~20K protein-coding genes",
            "training_data": "~104M human single cell transcriptomes (cancer cell lines)",
        },
    ),
    (
        "Geneformer-V2-316M",
        {
            "description": "316M parameters, Dec 2024 (default)",
            "input_size": 4096,
            "vocabulary": "~20K protein-coding genes",
            "training_data": "~104M human single cell transcriptomes",
        },
    ),
]

DEFAULT_MODEL_VARIANT = [MODEL_VARIANTS[0]]


@pytest.mark.parametrize("model_variant", DEFAULT_MODEL_VARIANT, ids=[variant[0] for variant in DEFAULT_MODEL_VARIANT])
def test_geneformer_checkpoint_loss(model_variant, input_data):
    """Test that the TE model can process input data and produce valid loss outputs."""

    set_seed(42)

    model_name, _ = model_variant

    # Load the specific Geneformer checkpoint from Hugging Face
    model_hf = load_geneformer_model(model_name)

    # Convert the pretrained HF model to TE format to get the same weights
    from geneformer.convert import convert_geneformer_hf_to_te

    model_te = convert_geneformer_hf_to_te(model_hf)

    device = torch.device("cuda")
    model_hf = model_hf.to(device)
    model_te = model_te.to(device)

    # Ensure both models are in eval mode for consistent behavior
    model_hf.eval()
    model_te.eval()

    # Use realistic geneformer input data instead of random inputs
    test_input = {k: v.to(device) for k, v in input_data.items()}

    # Test both models can process the input
    with torch.no_grad():
        te_outputs = model_te(**test_input)
        hf_outputs = model_hf(**test_input)

    # Verify both models produce valid outputs
    assert te_outputs.loss is not None, "TE model should produce loss"
    assert hf_outputs.loss is not None, "HF model should produce loss"
    assert te_outputs.logits.shape == hf_outputs.logits.shape, "Both models should have same output shape"

    # Verify losses are close to each other (indicating similar model behavior)
    torch.testing.assert_close(
        te_outputs.loss,
        hf_outputs.loss,
        atol=1e-2,
        rtol=1e-3,
        msg=lambda x: f"TE loss ({te_outputs.loss:.4f}) and HF loss ({hf_outputs.loss:.4f}) should be close: {x}",
    )

    # Clean up
    del model_hf, model_te
    torch.cuda.empty_cache()


@pytest.mark.parametrize("model_variant", DEFAULT_MODEL_VARIANT, ids=[variant[0] for variant in DEFAULT_MODEL_VARIANT])
def test_geneformer_checkpoint_weight_compatibility(model_variant):
    """Test that our TE model can potentially load weights from the actual Geneformer checkpoints."""
    from geneformer.modeling_bert_te import BertForMaskedLM as TEBertForMaskedLM
    from geneformer.modeling_bert_te import TEBertConfig

    model_name, model_info = model_variant

    model_hf = load_geneformer_model(model_name)

    hf_state_dict = model_hf.state_dict()

    # Create our TE model with the same architecture
    te_config_dict = {
        "hidden_size": model_hf.config.hidden_size,
        "num_hidden_layers": model_hf.config.num_hidden_layers,
        "num_attention_heads": model_hf.config.num_attention_heads,
        "intermediate_size": model_hf.config.intermediate_size,
        "max_position_embeddings": model_hf.config.max_position_embeddings,
        "vocab_size": model_hf.config.vocab_size,
        "attention_probs_dropout_prob": getattr(model_hf.config, "attention_probs_dropout_prob", 0.1),
        "hidden_dropout_prob": getattr(model_hf.config, "hidden_dropout_prob", 0.1),
        "hidden_act": getattr(model_hf.config, "hidden_act", "relu"),
        "initializer_range": getattr(model_hf.config, "initializer_range", 0.02),
        "layer_norm_eps": getattr(model_hf.config, "layer_norm_eps", 1e-12),
        "pad_token_id": getattr(model_hf.config, "pad_token_id", 0),
        "model_type": getattr(model_hf.config, "model_type", "bert"),
        "torch_dtype": torch.float32,
        "use_te_layers": True,
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }

    te_config = TEBertConfig(**te_config_dict)
    model_te = TEBertForMaskedLM(te_config)

    te_state_dict = model_te.state_dict()

    _run_compatibility_analysis(hf_state_dict, te_state_dict, te_config)

    del model_hf, model_te
    torch.cuda.empty_cache()


# Helper functions for parameter compatibility analysis
def _expand_pattern(pattern, state_dict):
    """Expand wildcard patterns like 'bert.encoder.layer.*.attention.output.dense.weight'"""
    expanded = []
    for key in state_dict.keys():
        # Check if this key matches the pattern
        if "bert.encoder.layer." in pattern and "bert.encoder.layer." in key:
            # Extract layer number from the key
            key_parts = key.split(".")
            pattern_parts = pattern.split(".")

            # Find the layer number in the key
            layer_num = None
            for i, part in enumerate(key_parts):
                if part == "layer" and i + 1 < len(key_parts) and key_parts[i + 1].isdigit():
                    layer_num = key_parts[i + 1]
                    break

            if layer_num is not None:
                # Check if the key structure matches the pattern structure
                if len(key_parts) == len(pattern_parts) and all(
                    p1 == p2 or p2 == "*" for p1, p2 in zip(key_parts, pattern_parts)
                ):
                    # Replace wildcard with actual layer number
                    expanded_pattern = pattern.replace("*", layer_num)
                    expanded.append((expanded_pattern, key))

    return expanded


def _get_parameter_mapping():
    """Get the mapping from HF BERT format to TE format.

    This mapping extends the base conversion mapping with unpacked QKV parameters.
    Since _unpack_fused_qkv_in_te_state_dict unpacks the fused QKV into individual
    Q, K, V parameters in HF format, we need identity mappings for them.
    """
    from geneformer.convert import mapping as base_mapping

    # Start with the base mapping from convert.py
    extended_mapping = base_mapping.copy()

    # Add mappings for unpacked Q, K, V parameters (identity mappings after unpacking)
    qkv_mappings = {
        "bert.encoder.layer.*.attention.self.query.weight": "bert.encoder.layer.*.attention.self.query.weight",
        "bert.encoder.layer.*.attention.self.query.bias": "bert.encoder.layer.*.attention.self.query.bias",
        "bert.encoder.layer.*.attention.self.key.weight": "bert.encoder.layer.*.attention.self.key.weight",
        "bert.encoder.layer.*.attention.self.key.bias": "bert.encoder.layer.*.attention.self.key.bias",
        "bert.encoder.layer.*.attention.self.value.weight": "bert.encoder.layer.*.attention.self.value.weight",
        "bert.encoder.layer.*.attention.self.value.bias": "bert.encoder.layer.*.attention.self.value.bias",
    }

    extended_mapping.update(qkv_mappings)

    return extended_mapping


def _check_wildcard_mapping(hf_pattern, te_pattern, hf_state_dict, te_state_dict):
    """Check compatibility for wildcard patterns."""
    compatible_params = 0
    incompatible_params = 0
    missing_params = 0

    expanded = _expand_pattern(hf_pattern, hf_state_dict)
    for expanded_hf, original_hf in expanded:
        # Extract layer number from the expanded HF pattern
        parts = expanded_hf.split(".")
        layer_num = None
        for i, part in enumerate(parts):
            if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_num = parts[i + 1]
                break

        if layer_num is not None:
            # Get the corresponding TE parameter name
            expanded_te = te_pattern.replace("*", layer_num)
            if expanded_te in te_state_dict:
                hf_param = hf_state_dict[original_hf]
                te_param = te_state_dict[expanded_te]
                if hf_param.shape == te_param.shape:
                    compatible_params += 1
                    print(f"Mapped: {original_hf} -> {expanded_te}")
                else:
                    incompatible_params += 1
                    print(f"Shape mismatch: {original_hf} -> {expanded_te}: HF={hf_param.shape}, TE={te_param.shape}")
            else:
                missing_params += 1
                print(f"Missing mapped parameter: {original_hf} -> {expanded_te}")

    return compatible_params, incompatible_params, missing_params


def _check_direct_mapping(hf_pattern, hf_state_dict, te_state_dict):
    """Check compatibility for direct (non-wildcard) patterns."""
    if hf_pattern in hf_state_dict and hf_pattern in te_state_dict:
        hf_param = hf_state_dict[hf_pattern]
        te_param = te_state_dict[hf_pattern]
        if hf_param.shape == te_param.shape:
            print(f"Direct: {hf_pattern}")
            return 1, 0, 0
        else:
            print(f"Shape mismatch: {hf_pattern}: HF={hf_param.shape}, TE={te_param.shape}")
            return 0, 1, 0
    elif hf_pattern in hf_state_dict:
        print(f"Missing: {hf_pattern}")
        return 0, 0, 1
    return 0, 0, 0


def _analyze_parameter_compatibility(hf_state_dict, te_state_dict):
    """Analyze parameter compatibility between HF and TE models."""
    compatible_params = 0
    incompatible_params = 0
    missing_params = 0

    mapping = _get_parameter_mapping()

    # Check mapped parameters
    for hf_pattern, te_pattern in mapping.items():
        if "*" in hf_pattern:
            # Handle wildcard mapping
            comp, incomp, miss = _check_wildcard_mapping(hf_pattern, te_pattern, hf_state_dict, te_state_dict)
            compatible_params += comp
            incompatible_params += incomp
            missing_params += miss
        else:
            # Direct mapping (no wildcards)
            comp, incomp, miss = _check_direct_mapping(hf_pattern, hf_state_dict, te_state_dict)
            compatible_params += comp
            incompatible_params += incomp
            missing_params += miss

    return compatible_params, incompatible_params, missing_params


def _print_compatibility_results(compatible_params, incompatible_params, missing_params, total_params):
    """Print the compatibility analysis results."""
    print("\nParameter compatibility analysis:")
    print(f"  - Compatible parameters: {compatible_params}")
    print(f"  - Incompatible parameters: {incompatible_params}")
    print(f"  - Missing parameters: {missing_params}")
    print(f"  - Total checkpoint parameters: {total_params}")

    compatibility_ratio = compatible_params / total_params
    print(f"  - Compatibility ratio: {compatibility_ratio:.2%}")

    return compatibility_ratio


def _unpack_fused_qkv_in_te_state_dict(te_state_dict, num_layers, num_heads):
    """Unpack fused QKV parameters in TE state dict to match HF format for comparison."""
    from geneformer.convert import _unpack_qkv_bias, _unpack_qkv_weight

    unpacked_te_state_dict = te_state_dict.copy()

    # Create a mock context object to use the original unpack functions
    class MockContext:
        def __init__(self, num_heads):
            self.source = type("Config", (), {"config": type("Config", (), {"num_attention_heads": num_heads})()})()

    mock_ctx = MockContext(num_heads)

    # Access the underlying functions bypassing the decorator
    unpack_weight_func = _unpack_qkv_weight.transform
    unpack_bias_func = _unpack_qkv_bias.transform

    for layer_idx in range(num_layers):
        # Unpack fused QKV weight
        fused_weight_key = f"bert.encoder.layer.{layer_idx}.self_attention.qkv.weight"
        if fused_weight_key in unpacked_te_state_dict:
            fused_weight = unpacked_te_state_dict[fused_weight_key]
            query_weight, key_weight, value_weight = unpack_weight_func(mock_ctx, fused_weight)

            # Add individual Q, K, V weights to the state dict
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.query.weight"] = query_weight
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.key.weight"] = key_weight
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.value.weight"] = value_weight

            # Remove the fused weight
            del unpacked_te_state_dict[fused_weight_key]

        # Unpack fused QKV bias
        fused_bias_key = f"bert.encoder.layer.{layer_idx}.self_attention.qkv.bias"
        if fused_bias_key in unpacked_te_state_dict:
            fused_bias = unpacked_te_state_dict[fused_bias_key]
            query_bias, key_bias, value_bias = unpack_bias_func(mock_ctx, fused_bias)

            # Add individual Q, K, V biases to the state dict
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.query.bias"] = query_bias
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.key.bias"] = key_bias
            unpacked_te_state_dict[f"bert.encoder.layer.{layer_idx}.attention.self.value.bias"] = value_bias

            # Remove the fused bias
            del unpacked_te_state_dict[fused_bias_key]

    return unpacked_te_state_dict


def _run_compatibility_analysis(hf_state_dict, te_state_dict, te_config):
    """Run the parameter compatibility analysis."""
    # Unpack fused QKV parameters in TE state dict for accurate comparison
    unpacked_te_state_dict = _unpack_fused_qkv_in_te_state_dict(
        te_state_dict, te_config.num_hidden_layers, te_config.num_attention_heads
    )

    print(f"Unpacked TE state dict: {len(te_state_dict)} -> {len(unpacked_te_state_dict)} parameters")

    # Run the analysis with unpacked TE state dict
    compatible_params, incompatible_params, missing_params = _analyze_parameter_compatibility(
        hf_state_dict, unpacked_te_state_dict
    )

    compatibility_ratio = _print_compatibility_results(
        compatible_params, incompatible_params, missing_params, len(hf_state_dict)
    )

    assert compatibility_ratio == 1.0, (
        f"Expected 100% compatibility after unpacking fused QKV parameters, but got {compatibility_ratio:.2%}. "
        f"All parameters should be mappable between HF and TE models."
    )
