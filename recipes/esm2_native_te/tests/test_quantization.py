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

import re
import sys
from pathlib import Path

import pytest
import yaml


sys.path.append(Path(__file__).parent.parent.as_posix())

from quantization import generate_layer_regex, resolve_layer_precision, update_quant_stats_config


# -- resolve_layer_precision --


def test_fp8_enabled_no_layers_defaults_all():
    """When fp8 is enabled with no explicit layers, all layers should default to FP8."""
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=True, fp4_enabled=False, fp8_layers=None, fp4_layers=None
    )
    assert result == ["fp8", "fp8", "fp8", "fp8", "fp8", "fp8"]


def test_fp4_enabled_no_layers_defaults_all():
    """When fp4 is enabled with no explicit layers, all layers should default to FP4."""
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=False, fp4_enabled=True, fp8_layers=None, fp4_layers=None
    )
    assert result == ["fp4", "fp4", "fp4", "fp4", "fp4", "fp4"]


def test_fp8_explicit_layers():
    """Explicit 1-indexed fp8_layers should produce fp8 at those positions."""
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=True, fp4_enabled=False, fp8_layers=[1, 3, 5], fp4_layers=None
    )
    assert result == ["fp8", None, "fp8", None, "fp8", None]


def test_fp4_explicit_layers():
    """Explicit 1-indexed fp4_layers should produce fp4 at those positions."""
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=False, fp4_enabled=True, fp8_layers=None, fp4_layers=[2, 4, 6]
    )
    assert result == [None, "fp4", None, "fp4", None, "fp4"]


def test_mixed_fp8_fp4_explicit():
    """Both enabled with explicit non-overlapping layers should work correctly."""
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=True, fp4_enabled=True, fp8_layers=[1, 3, 4], fp4_layers=[2, 5]
    )
    assert result == ["fp8", "fp4", "fp8", "fp8", "fp4", None]


def test_both_enabled_no_layers_raises():
    """Both enabled with no layer lists should raise ValueError."""
    with pytest.raises(ValueError, match="Both fp8_config and fp4_config are enabled"):
        resolve_layer_precision(num_layers=6, fp8_enabled=True, fp4_enabled=True, fp8_layers=None, fp4_layers=None)


def test_overlapping_layers_raises():
    """Overlapping layer assignments should raise ValueError."""
    with pytest.raises(ValueError, match="fp8_layers and fp4_layers cannot have overlapping"):
        resolve_layer_precision(
            num_layers=6, fp8_enabled=True, fp4_enabled=True, fp8_layers=[1, 2, 3], fp4_layers=[3, 4, 5]
        )


def test_disabled_ignores_layers():
    """When a format is disabled, its layers should be ignored."""
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=False, fp4_enabled=False, fp8_layers=[1, 2, 3], fp4_layers=[4, 5, 6]
    )
    assert result == [None, None, None, None, None, None]


def test_both_disabled():
    """Both disabled with no layers should return all None."""
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=False, fp4_enabled=False, fp8_layers=None, fp4_layers=None
    )
    assert result == [None, None, None, None, None, None]


def test_large_model_defaults_all():
    """Auto-population should work correctly for larger models (e.g. 36 layers)."""
    result = resolve_layer_precision(
        num_layers=36, fp8_enabled=True, fp4_enabled=False, fp8_layers=None, fp4_layers=None
    )
    assert result == ["fp8"] * 36


def test_fp8_enabled_empty_list():
    """An explicit empty list should remain empty (not default to all)."""
    result = resolve_layer_precision(num_layers=6, fp8_enabled=True, fp4_enabled=False, fp8_layers=[], fp4_layers=None)
    assert result == [None, None, None, None, None, None]


def test_both_enabled_fp8_specified_fp4_defaults_to_remaining():
    """When both enabled, FP8 has explicit layers, FP4 should default to the remaining layers."""
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=True, fp4_enabled=True, fp8_layers=[1, 2, 3], fp4_layers=None
    )
    assert result == ["fp8", "fp8", "fp8", "fp4", "fp4", "fp4"]


def test_both_enabled_fp4_specified_fp8_defaults_to_remaining():
    """When both enabled, FP4 has explicit layers, FP8 should default to the remaining layers."""
    result = resolve_layer_precision(
        num_layers=6, fp8_enabled=True, fp4_enabled=True, fp8_layers=None, fp4_layers=[4, 5, 6]
    )
    assert result == ["fp8", "fp8", "fp8", "fp4", "fp4", "fp4"]


def test_returns_correct_length():
    """Result list length should always equal num_layers."""
    for n in [1, 6, 48]:
        result = resolve_layer_precision(
            num_layers=n, fp8_enabled=False, fp4_enabled=False, fp8_layers=None, fp4_layers=None
        )
        assert len(result) == n


# -- generate_layer_regex --


def test_single_layer():
    """Single layer should produce a simple regex."""
    regex = generate_layer_regex([3])
    assert re.search(regex, "model.model.encoder.layers.3.self_attention.layernorm_qkv")
    assert not re.search(regex, "model.model.encoder.layers.2.self_attention.layernorm_qkv")


def test_multiple_layers():
    """Multiple layers should match any of them."""
    regex = generate_layer_regex([1, 2, 3])
    assert re.search(regex, "model.model.encoder.layers.1.self_attention.layernorm_qkv")
    assert re.search(regex, "model.model.encoder.layers.2.layernorm_mlp.fc1")
    assert re.search(regex, "model.model.encoder.layers.3.layernorm_mlp.fc2")
    assert not re.search(regex, "model.model.encoder.layers.4.self_attention.proj")


def test_matches_correct_sublayers():
    """Regex should only match layernorm_qkv, proj, fc1, fc2."""
    regex = generate_layer_regex([1])
    assert re.search(regex, "model.model.encoder.layers.1.self_attention.layernorm_qkv_something")
    assert re.search(regex, "model.model.encoder.layers.1.self_attention.proj_something")
    assert re.search(regex, "model.model.encoder.layers.1.layernorm_mlp.fc1_something")
    assert re.search(regex, "model.model.encoder.layers.1.layernorm_mlp.fc2_something")
    # Should not match unrelated sublayer names
    assert not re.search(regex, "model.model.encoder.layers.1.self_attention.some_other_thing")


def test_none_returns_disabled_pattern():
    """None should return a pattern that matches nothing."""
    regex = generate_layer_regex(None)
    assert "DISABLED" in regex
    assert not re.search(regex, "model.model.encoder.layers.1.self_attention.layernorm_qkv")


def test_empty_list_returns_disabled_pattern():
    """Empty list should return a pattern that matches nothing."""
    regex = generate_layer_regex([])
    assert "DISABLED" in regex


def test_1indexed_layer_names():
    """Regex should use 1-indexed layer numbers (matching debug API naming)."""
    regex = generate_layer_regex([1])
    # Should match layers.1 (1-indexed first layer)
    assert re.search(regex, "model.model.encoder.layers.1.self_attention.layernorm_qkv")
    # Should NOT match layers.0 (0-indexed first layer)
    assert not re.search(regex, "model.model.encoder.layers.0.self_attention.layernorm_qkv")


# -- update_quant_stats_config --


@pytest.fixture
def fp8_only_config(tmp_path):
    """Create an FP8-only stats config file."""
    config = {
        "example_fp8_tensor_stat_collection": {
            "enabled": True,
            "layers": {
                "layer_name_regex_pattern": "PLACEHOLDER",
            },
            "transformer_engine": {
                "LogFp8TensorStats": {
                    "enabled": True,
                    "tensors_struct": [{"tensor": "activation", "stats": ["underflows%"], "freq": 10}],
                }
            },
        }
    }
    config_path = tmp_path / "fp8_stats.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


@pytest.fixture
def fp4_fp8_config(tmp_path):
    """Create a combined FP4+FP8 stats config file."""
    config = {
        "example_fp4_tensor_stat_collection": {
            "enabled": True,
            "layers": {
                "layer_name_regex_pattern": "PLACEHOLDER",
            },
            "transformer_engine": {
                "LogNvfp4TensorStats": {"enabled": True},
            },
        },
        "example_fp8_tensor_stat_collection": {
            "enabled": True,
            "layers": {
                "layer_name_regex_pattern": "PLACEHOLDER",
            },
            "transformer_engine": {
                "LogFp8TensorStats": {"enabled": True},
            },
        },
    }
    config_path = tmp_path / "fp4_fp8_stats.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


def test_fp8_layers_updates_regex(fp8_only_config):
    """FP8 layer list should update the regex in the output config."""
    output_path = update_quant_stats_config(config_file=fp8_only_config, fp4_layers=None, fp8_layers=[1, 2, 3])
    with open(output_path) as f:
        result = yaml.safe_load(f)
    regex = result["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"]
    assert re.search(regex, "model.model.encoder.layers.1.self_attention.layernorm_qkv")
    assert re.search(regex, "model.model.encoder.layers.3.layernorm_mlp.fc2")
    assert not re.search(regex, "model.model.encoder.layers.4.self_attention.proj")


def test_none_layers_disables_matching(fp8_only_config):
    """None layers should set regex to match nothing."""
    output_path = update_quant_stats_config(config_file=fp8_only_config, fp4_layers=None, fp8_layers=None)
    with open(output_path) as f:
        result = yaml.safe_load(f)
    regex = result["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"]
    assert "DISABLED" in regex


def test_fp4_section_disabled_fp8_still_updated(fp4_fp8_config):
    """FP4 stats section should be disabled (not yet supported), FP8 should still be updated."""
    output_path = update_quant_stats_config(config_file=fp4_fp8_config, fp4_layers=[1, 2, 3], fp8_layers=[4, 5, 6])
    with open(output_path) as f:
        result = yaml.safe_load(f)

    # FP4 section should be disabled
    assert result["example_fp4_tensor_stat_collection"]["enabled"] is False

    # FP8 regex should still match layers 4-6
    fp8_regex = result["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"]
    assert re.search(fp8_regex, "model.model.encoder.layers.5.self_attention.proj")
    assert not re.search(fp8_regex, "model.model.encoder.layers.2.self_attention.proj")


def test_original_file_not_modified(fp8_only_config):
    """update_quant_stats_config should write to a temp file, not modify the original."""
    with open(fp8_only_config) as f:
        original_content = f.read()

    output_path = update_quant_stats_config(config_file=fp8_only_config, fp4_layers=None, fp8_layers=[1, 2])

    assert output_path != fp8_only_config
    with open(fp8_only_config) as f:
        assert f.read() == original_content


def test_preserves_other_config_fields(fp8_only_config):
    """Non-layer fields in the config should be preserved."""
    output_path = update_quant_stats_config(config_file=fp8_only_config, fp4_layers=None, fp8_layers=[1])
    with open(output_path) as f:
        result = yaml.safe_load(f)
    # The transformer_engine section should still be there
    assert result["example_fp8_tensor_stat_collection"]["transformer_engine"]["LogFp8TensorStats"]["enabled"] is True


def test_missing_section_is_skipped(fp8_only_config):
    """If fp4 section doesn't exist in config, it should be silently skipped."""
    # fp8_only_config has no fp4 section — passing fp4_layers should not error
    output_path = update_quant_stats_config(config_file=fp8_only_config, fp4_layers=[1, 2], fp8_layers=[3, 4])
    with open(output_path) as f:
        result = yaml.safe_load(f)
    # Only FP8 section should exist and be updated
    assert "example_fp4_tensor_stat_collection" not in result
    regex = result["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"]
    assert re.search(regex, "model.model.encoder.layers.3.self_attention.layernorm_qkv")


def test_with_real_fp4_config():
    """Test with the actual fp4_debugging_stats.yaml file."""
    config_path = Path(__file__).parent.parent / "fp4_debugging_stats.yaml"
    if not config_path.exists():
        pytest.skip("fp4_debugging_stats.yaml not found")

    output_path = update_quant_stats_config(config_file=str(config_path), fp4_layers=[1, 2, 3], fp8_layers=[4, 5, 6])
    with open(output_path) as f:
        result = yaml.safe_load(f)

    # FP4 section should be disabled (not yet supported in current TE release)
    assert result["example_fp4_tensor_stat_collection"]["enabled"] is False

    # FP8 section should still be updated and working
    fp8_regex = result["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"]
    assert re.search(fp8_regex, "model.model.encoder.layers.5.self_attention.proj")
    assert not re.search(fp8_regex, "model.model.encoder.layers.2.self_attention.proj")
