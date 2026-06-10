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

"""Integration tests: Claude adds FP8 support to a TE model."""

from validators import ast_checks, pattern_checks


class TestFp8Addition:
    """Test that Claude can add FP8 support to an existing TE model."""

    def test_claude_adds_fp8_to_bert(self, claude_runner, pre_te_ified_bert_dir):
        """Claude adds FP8 quantization support to a TE-ified BERT model."""
        claude_runner.run(
            "Add FP8 quantization support to this TransformerEngine model. "
            "Add layer_precision config, get_autocast_context() method, "
            "and vocabulary padding for FP8 compatibility.",
            cwd=str(pre_te_ified_bert_dir),
        )

        te_model_candidates = list(pre_te_ified_bert_dir.glob("*te*.py")) + list(
            pre_te_ified_bert_dir.glob("*_nv*.py")
        )
        assert len(te_model_candidates) > 0, "No TE model file found"

        te_model_file = te_model_candidates[0]
        ast_checks.validate_python_file(te_model_file)
        pattern_checks.has_layer_precision_config(te_model_file)
        pattern_checks.has_fp8_autocast(te_model_file)
