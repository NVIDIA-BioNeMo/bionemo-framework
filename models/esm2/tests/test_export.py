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

import pytest
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer

from export import export_hf_checkpoint


@pytest.fixture
def exported_8m_checkpoint(tmp_path):
    export_hf_checkpoint("esm2_t6_8M_UR50D", tmp_path)
    return tmp_path / "esm2_t6_8M_UR50D"


def test_auto_model_loading(exported_8m_checkpoint):
    model, loading_info = AutoModel.from_pretrained(
        exported_8m_checkpoint, trust_remote_code=True, output_loading_info=True
    )
    assert type(model).__name__.endswith("NVEsmModel")
    assert not loading_info["mismatched_keys"]
    assert not loading_info["error_msgs"]


def test_auto_model_for_masked_lm_loading(exported_8m_checkpoint):
    model_for_masked_lm, loading_info = AutoModelForMaskedLM.from_pretrained(
        exported_8m_checkpoint, trust_remote_code=True, output_loading_info=True
    )
    assert type(model_for_masked_lm).__name__.endswith("NVEsmForMaskedLM")
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert not loading_info["mismatched_keys"]
    assert not loading_info["error_msgs"]


def test_auto_model_for_token_classification_loading(exported_8m_checkpoint):
    model_for_token_classification, loading_info = AutoModelForTokenClassification.from_pretrained(
        exported_8m_checkpoint,
        trust_remote_code=True,
        output_loading_info=True,
        num_labels=8,
    )
    assert type(model_for_token_classification).__name__.endswith("NVEsmForTokenClassification")
    assert model_for_token_classification.num_labels == 8
    assert model_for_token_classification.classifier.weight.shape[0] == 8
    assert not loading_info["mismatched_keys"]
    assert not loading_info["error_msgs"]


def test_auto_tokenizer_loading(exported_8m_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(exported_8m_checkpoint)
    assert tokenizer is not None


def test_exported_checkpoint_files(exported_8m_checkpoint):
    """Test that required files (LICENSE, README.md) are present in the exported directory."""

    assert (exported_8m_checkpoint / "LICENSE").is_file(), "LICENSE file is missing in the export directory"
    readme_path = exported_8m_checkpoint / "README.md"
    assert readme_path.is_file(), "README.md file is missing in the export directory"
    with open(readme_path, "r") as f:
        readme_contents = f.read()
    assert "**Number of model parameters:** 7.5 x 10^6" in readme_contents, (
        f"README.md does not contain the expected parameter count line: {readme_contents}"
    )
    assert (
        "Hugging Face 07/29/2025 via [https://huggingface.co/nvidia/esm2_t6_8M_UR50D]"
        "(https://huggingface.co/nvidia/esm2_t6_8M_UR50D)"
    ) in readme_contents, f"README.md does not contain the expected Hugging Face link line: {readme_contents}"
    assert "**Benchmark Score:** 0.48" in readme_contents, (
        f"README.md does not contain the expected CAMEO score line: {readme_contents}"
    )
    assert "**Benchmark Score:** 0.37" in readme_contents, (
        f"README.md does not contain the expected CASP14 score line: {readme_contents}"
    )
