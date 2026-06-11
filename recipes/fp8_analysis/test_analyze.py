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

"""Unit tests for analyze_and_create_heatmap.py."""

from pathlib import Path

import pandas as pd
import pytest
from analyze_and_create_heatmap import (
    auto_detect_components,
    create_heatmap,
    parse_layer_metadata,
    parse_log_file,
)


# Get the directory containing the test dummy logs
TEST_DIR = Path(__file__).parent
DUMMY_LOGS_ESM2 = TEST_DIR / "dummy_logs_esm2"
DUMMY_LOGS_LLAMA3 = TEST_DIR / "dummy_logs_llama3"


@pytest.fixture
def esm2_log_dir():
    """Return the path to ESM2 dummy logs directory."""
    return DUMMY_LOGS_ESM2


@pytest.fixture
def llama3_log_dir():
    """Return the path to Llama3 dummy logs directory."""
    return DUMMY_LOGS_LLAMA3


class TestParseLayerMetadata:
    """Test parse_layer_metadata function."""

    def test_parse_esm2_metadata(self, esm2_log_dir):
        """Test parsing ESM2 model metadata."""
        structure = parse_layer_metadata(esm2_log_dir)

        assert structure is not None
        assert "encoder_layers" in structure
        assert "head_layers" in structure
        assert "embedding_layers" in structure
        assert "other_layers" in structure

        # ESM2 has 6 encoder layers (1-6)
        assert len(structure["encoder_layers"]) == 6
        encoder_layer_nums = [layer["num"] for layer in structure["encoder_layers"]]
        assert set(encoder_layer_nums) == {1, 2, 3, 4, 5, 6}

        # ESM2 has head layers (lm_head.dense, lm_head.decoder)
        assert len(structure["head_layers"]) >= 1
        head_layer_names = [layer for layer in structure["head_layers"]]
        assert any("lm_head" in name for name in head_layer_names)

    def test_parse_llama3_metadata(self, llama3_log_dir):
        """Test parsing Llama3 model metadata.

        Note: The metadata parser looks for `.encoder.layers.` pattern which ESM2 uses.
        Llama3 uses `.model.layers.` pattern, so encoder layers won't be detected
        in the metadata parsing step. However, the auto_detect_components function
        handles both patterns correctly.
        """
        structure = parse_layer_metadata(llama3_log_dir)

        assert structure is not None
        assert "encoder_layers" in structure
        assert "head_layers" in structure
        assert "other_layers" in structure

        # Llama3 uses model.model.layers.X pattern which is not matched by
        # the .encoder.layers. pattern in parse_layer_metadata.
        # The layers are classified as "other_layers" instead.
        # This is expected behavior - the component detection handles this correctly.
        assert len(structure["encoder_layers"]) == 0

        # Check that layers exist in other_layers instead
        other_layer_names = structure["other_layers"]
        layers_with_model_pattern = [layer for layer in other_layer_names if "model.layers" in layer]
        assert len(layers_with_model_pattern) > 0, "Llama3 layers should be in other_layers"

        # Llama3 has lm_head
        assert len(structure["head_layers"]) >= 1

    def test_missing_metadata_file(self, tmp_path):
        """Test handling of missing metadata file."""
        structure = parse_layer_metadata(tmp_path)
        assert structure is None


class TestParseLogFile:
    """Test parse_log_file function."""

    def test_parse_esm2_statistics_log(self, esm2_log_dir):
        """Test parsing ESM2 statistics log file."""
        log_file = esm2_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "metric_name" in df.columns
        assert "iteration" in df.columns
        assert "value" in df.columns

        # Check that we have underflow metrics
        underflow_metrics = df[df["metric_name"].str.contains("underflows%")]
        assert len(underflow_metrics) > 0

        # Check that we have metrics for multiple encoder layers
        layer_metrics = df[df["metric_name"].str.contains("encoder.layers")]
        assert len(layer_metrics) > 0

        # Check iteration range
        assert df["iteration"].min() == 0
        assert df["iteration"].max() > 0

    def test_parse_llama3_statistics_log(self, llama3_log_dir):
        """Test parsing Llama3 statistics log file."""
        log_file = llama3_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "metric_name" in df.columns
        assert "iteration" in df.columns
        assert "value" in df.columns

        # Check that we have underflow metrics
        underflow_metrics = df[df["metric_name"].str.contains("underflows%")]
        assert len(underflow_metrics) > 0

        # Check for llama3-style layer names
        layer_metrics = df[df["metric_name"].str.contains("model.layers")]
        assert len(layer_metrics) > 0

    def test_parse_log_values_are_numeric(self, esm2_log_dir):
        """Test that parsed values are numeric."""
        log_file = esm2_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)

        assert df["value"].dtype in ["float64", "float32"]
        assert df["iteration"].dtype in ["int64", "int32"]


class TestAutoDetectComponents:
    """Test auto_detect_components function."""

    def test_detect_esm2_components(self, esm2_log_dir):
        """Test component detection for ESM2."""
        log_file = esm2_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)
        structure = parse_layer_metadata(esm2_log_dir)

        components = auto_detect_components(df, structure)

        assert len(components) > 0

        # Check that encoder layers are detected
        encoder_components = [c for c in components if c["type"] == "encoder"]
        assert len(encoder_components) == 6  # ESM2 has 6 layers

        # Check component structure
        for comp in components:
            assert "position" in comp
            assert "type" in comp
            assert "identifier" in comp
            assert "label" in comp
            assert "metric" in comp
            assert "group" in comp

        # Check that encoder layers are sorted
        encoder_identifiers = [c["identifier"] for c in encoder_components]
        assert encoder_identifiers == sorted(encoder_identifiers)

    def test_detect_llama3_components(self, llama3_log_dir):
        """Test component detection for Llama3."""
        log_file = llama3_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)
        structure = parse_layer_metadata(llama3_log_dir)

        components = auto_detect_components(df, structure)

        assert len(components) > 0

        # Check that encoder layers are detected
        encoder_components = [c for c in components if c["type"] == "encoder"]
        assert len(encoder_components) == 2  # Llama3 dummy has 2 layers

        # Check component labels
        encoder_labels = [c["label"] for c in encoder_components]
        assert "L1" in encoder_labels
        assert "L2" in encoder_labels

    def test_component_metrics_are_gradient_underflows(self, esm2_log_dir):
        """Test that detected components use gradient underflow metrics."""
        log_file = esm2_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)
        structure = parse_layer_metadata(esm2_log_dir)

        components = auto_detect_components(df, structure)

        for comp in components:
            assert "gradient_underflows%" in comp["metric"]


class TestCreateHeatmap:
    """Tests for heatmap generation."""

    def test_create_esm2_heatmap(self, esm2_log_dir, tmp_path):
        """Test heatmap generation for ESM2 logs."""
        log_file = esm2_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)
        structure = parse_layer_metadata(esm2_log_dir)
        components = auto_detect_components(df, structure)

        output_dir = tmp_path / "heatmaps"
        output_file = create_heatmap(df, components, output_dir, suffix="_esm2_test")

        assert output_file is not None
        assert output_file.exists()
        assert output_file.name == "heatmap_highres_esm2_test.png"
        assert output_file.stat().st_size > 0  # File has content

    def test_create_llama3_heatmap(self, llama3_log_dir, tmp_path):
        """Test heatmap generation for Llama3 logs."""
        log_file = llama3_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)
        structure = parse_layer_metadata(llama3_log_dir)
        components = auto_detect_components(df, structure)

        output_dir = tmp_path / "heatmaps"
        output_file = create_heatmap(df, components, output_dir, suffix="_llama3_test")

        assert output_file is not None
        assert output_file.exists()
        assert output_file.name == "heatmap_highres_llama3_test.png"
        assert output_file.stat().st_size > 0  # File has content

    def test_heatmap_creates_output_directory(self, esm2_log_dir, tmp_path):
        """Test that heatmap creation creates the output directory if it doesn't exist."""
        log_file = esm2_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)
        structure = parse_layer_metadata(esm2_log_dir)
        components = auto_detect_components(df, structure)

        # Use a nested directory that doesn't exist
        output_dir = tmp_path / "nested" / "output" / "heatmaps"
        assert not output_dir.exists()

        output_file = create_heatmap(df, components, output_dir)

        assert output_dir.exists()
        assert output_file.exists()

    def test_heatmap_with_empty_suffix(self, esm2_log_dir, tmp_path):
        """Test heatmap generation with empty suffix."""
        log_file = esm2_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)
        structure = parse_layer_metadata(esm2_log_dir)
        components = auto_detect_components(df, structure)

        output_dir = tmp_path / "heatmaps"
        output_file = create_heatmap(df, components, output_dir, suffix="")

        assert output_file is not None
        assert output_file.name == "heatmap_highres.png"

    def test_heatmap_returns_none_for_empty_data(self, tmp_path):
        """Test that heatmap returns None when there's no data."""
        # Create empty DataFrame
        df = pd.DataFrame(columns=["metric_name", "iteration", "value"])
        components = [
            {
                "position": 1,
                "type": "encoder",
                "identifier": 1,
                "label": "L1",
                "display_label": "1",
                "metric": "nonexistent_metric",
                "group": "Encoder",
            }
        ]

        output_dir = tmp_path / "heatmaps"
        output_file = create_heatmap(df, components, output_dir)

        assert output_file is None

    def test_heatmap_file_is_valid_png(self, esm2_log_dir, tmp_path):
        """Test that generated heatmap is a valid PNG file."""
        log_file = esm2_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
        df = parse_log_file(log_file)
        structure = parse_layer_metadata(esm2_log_dir)
        components = auto_detect_components(df, structure)

        output_dir = tmp_path / "heatmaps"
        output_file = create_heatmap(df, components, output_dir)

        # Check PNG magic bytes
        with open(output_file, "rb") as f:
            header = f.read(8)
        # PNG files start with these magic bytes
        png_signature = b"\x89PNG\r\n\x1a\n"
        assert header == png_signature, "Output file should be a valid PNG"
