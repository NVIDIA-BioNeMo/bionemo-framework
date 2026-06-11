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


import os
import subprocess

import pytest
import torch
from transformers import AutoModelForCausalLM

from bionemo.core.data.load import load


@pytest.fixture(scope="module")
def eden_llama_og2_step_182313_on_evo2_rrna_highly_conserved_PMC4140814():
    """Test data for Evo2 llama inference.

    Returns:
        tree
            .
            ├── per_layer_activations
            │   └── activations_rank000_dl00_batch000000.pt
            ├── predictions__rank_0__dp_rank_0.pt
            ├── ribosomal_rrna_highly_conserved_PMC4140814.fasta
            └── seq_idx_map.json

    1 directory, 4 files
    """
    return load("evo2_llama/eden_llama_og2_step_182313_on_evo2_rrna_highly_conserved_PMC4140814:1.0")


@pytest.fixture(scope="module")
def llama_7b_8k_og2():
    return load("evo2_llama/7B-8k-og2:1.0")


@pytest.mark.skipif(os.environ.get("BIONEMO_DATA_SOURCE") != "pbss", reason="Test data is not available on NGC")
def test_golden_values_llama(
    tmp_path, eden_llama_og2_step_182313_on_evo2_rrna_highly_conserved_PMC4140814, llama_7b_8k_og2
):
    fasta_path = (
        eden_llama_og2_step_182313_on_evo2_rrna_highly_conserved_PMC4140814
        / "ribosomal_rrna_highly_conserved_PMC4140814.fasta"
    )
    gold_values_path = (
        eden_llama_og2_step_182313_on_evo2_rrna_highly_conserved_PMC4140814 / "predictions__rank_0__dp_rank_0.pt"
    )
    output_dir = tmp_path / "predictions_llama"
    prediction_cmd = (
        f"predict_evo2 --fasta {fasta_path} --ckpt-dir {llama_7b_8k_og2} --output-dir {output_dir} --model-size 7B"
    )
    subprocess.run(prediction_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    predictions = torch.load(output_dir / "predictions__rank_0__dp_rank_0.pt", weights_only=True)
    gold_values = torch.load(gold_values_path, weights_only=True)
    assert predictions["token_logits"].shape == gold_values["token_logits"].shape
    torch.testing.assert_close(predictions["token_logits"], gold_values["token_logits"], atol=0.5, rtol=0)


@pytest.mark.skipif(os.environ.get("BIONEMO_DATA_SOURCE") != "pbss", reason="Test data is not available on NGC")
def test_checkpoint_conversion(
    tmp_path, eden_llama_og2_step_182313_on_evo2_rrna_highly_conserved_PMC4140814, llama_7b_8k_og2
):
    target_dir = tmp_path / "llama_7b_8k_og2"
    convert_cmd = f"evo2_nemo2_to_hf --model-type llama  --model-path {llama_7b_8k_og2} --output-dir {target_dir}"
    subprocess.run(convert_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert target_dir.exists()
    assert target_dir.is_dir()
    hf_model = AutoModelForCausalLM.from_pretrained(
        target_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,  # Force loading from local path, not HF Hub
        use_cache=False,  # Disable use_cache to get the correct forward pass outside of generate.
    ).eval()
    # # Add hooks to capture inputs/outputs for forward pass
    # activations = {}
    # def capture_hook(name):
    #     def hook(module, input, output):
    #         # if not isinstance(input, torch.Tensor):
    #         #     input = None
    #         # if not isinstance(output, torch.Tensor):
    #         #     output = None
    #         activations[name] = {
    #             'input': input,
    #             'output': output
    #         }
    #     return hook
    # # Register hooks on key layers
    # for name, module in hf_model.named_modules():
    #     module.register_forward_hook(capture_hook(name))
    fasta_path = (
        eden_llama_og2_step_182313_on_evo2_rrna_highly_conserved_PMC4140814
        / "ribosomal_rrna_highly_conserved_PMC4140814.fasta"
    )
    with open(fasta_path, "r") as f:
        fasta_seq = f.readlines()[1].strip()
    input_ids = torch.tensor([ord(c) for c in fasta_seq]).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = hf_model(input_ids)
    gold_values_path = (
        eden_llama_og2_step_182313_on_evo2_rrna_highly_conserved_PMC4140814 / "predictions__rank_0__dp_rank_0.pt"
    )
    gold_values = torch.load(gold_values_path, weights_only=True)
    assert outputs.logits.shape == gold_values["token_logits"].shape
    torch.testing.assert_close(outputs.logits, gold_values["token_logits"].to(dtype=torch.bfloat16), atol=0.5, rtol=0)
