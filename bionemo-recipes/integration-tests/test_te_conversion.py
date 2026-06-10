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

"""Integration tests: Claude converts vanilla models to TransformerEngine."""

from validators import ast_checks, pattern_checks


class TestBertTeConversion:
    """Test that Claude can TE-ify a vanilla BERT model."""

    def test_claude_creates_te_model(self, claude_runner, bert_fixture_dir):
        """Claude converts a vanilla BERT to use TransformerEngine."""
        claude_runner.run(
            "Convert this HuggingFace BERT model to use TransformerEngine. "
            "Create: 1) A TE model file with NV config and model classes, "
            "2) A convert.py with bidirectional conversion and state dict mapping, "
            "3) A basic golden value test. "
            "Follow the patterns from the bionemo-recipes reference files.",
            cwd=str(bert_fixture_dir),
        )

        te_model_candidates = list(bert_fixture_dir.glob("*te*.py")) + list(bert_fixture_dir.glob("*_nv*.py"))
        assert len(te_model_candidates) > 0, "No TE model file created"

        convert_file = bert_fixture_dir / "convert.py"
        assert convert_file.exists(), "convert.py not created"

        for py_file in bert_fixture_dir.glob("*.py"):
            ast_checks.validate_python_file(py_file)

        te_model_file = te_model_candidates[0]
        pattern_checks.has_te_imports(te_model_file)
        pattern_checks.has_state_dict_mapping(convert_file)
        pattern_checks.has_bidirectional_conversion(convert_file)


class TestLlamaTeConversion:
    """Test that Claude can TE-ify a vanilla Llama model (decoder with GQA)."""

    def test_claude_creates_te_model(self, claude_runner, llama_fixture_dir):
        """Claude converts a vanilla Llama to use TransformerEngine."""
        claude_runner.run(
            "Convert this HuggingFace Llama-style causal LM model to use TransformerEngine. "
            "This model uses Group Query Attention (GQA) with separate Q/K/V projections "
            "and a SwiGLU FFN with gate/up projections. "
            "Create: 1) A TE model file, 2) A convert.py with bidirectional conversion, "
            "3) A basic golden value test. "
            "Follow the patterns from the bionemo-recipes reference files.",
            cwd=str(llama_fixture_dir),
        )

        te_model_candidates = list(llama_fixture_dir.glob("*te*.py")) + list(llama_fixture_dir.glob("*_nv*.py"))
        assert len(te_model_candidates) > 0, "No TE model file created"

        convert_file = llama_fixture_dir / "convert.py"
        assert convert_file.exists(), "convert.py not created"

        for py_file in llama_fixture_dir.glob("*.py"):
            ast_checks.validate_python_file(py_file)

        te_model_file = te_model_candidates[0]
        pattern_checks.has_te_imports(te_model_file)
        pattern_checks.has_state_dict_mapping(convert_file)
        pattern_checks.has_bidirectional_conversion(convert_file)
