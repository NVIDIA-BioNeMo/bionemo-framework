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

import json
import os
from pathlib import Path

import pytest
import torch
from nemo.collections.llm.gpt.model.llama import HFLlamaExporter

from bionemo.core.data.load import load
from bionemo.evo2.models.llama import HFEdenLlamaImporter
from bionemo.llm.lightning import batch_collator
from bionemo.testing.subprocess_utils import run_command_in_subprocess


REPO_PATH = Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent


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


def predict_metagenome(
    model_checkpoint_path: Path, metagenome_fasta_path: Path, output_path: Path
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    cmd = f"""predict_evo2 \
            --eden-tokenizer \
            --model-size 7B \
            --fasta {metagenome_fasta_path} \
            --ckpt-dir {model_checkpoint_path} \
            --output-log-prob-seqs \
            --log-prob-collapse-option per_token \
            --output-dir {output_path}"""
    run_command_in_subprocess(cmd, os.getcwd())
    with open(output_path / "seq_idx_map.json", "r") as jsonf:
        fasta_to_index = json.load(jsonf)
    preds_list = [torch.load(f) for f in output_path.glob("*.pt")]
    all_pt_data = batch_collator([item for item in preds_list if item is not None])
    return all_pt_data, fasta_to_index  # type: ignore


@pytest.mark.skipif(os.environ.get("BIONEMO_DATA_SOURCE") != "pbss", reason="Test data is not available on NGC")
@pytest.mark.slow
def test_eden_llama_roundtrip(
    tmp_path, llama_7b_8k_og2: Path, eden_llama_og2_step_182313_on_evo2_rrna_highly_conserved_PMC4140814: Path
):
    """Test that converting NeMo -> HF -> NeMo produces the same model."""
    fasta_path = (
        eden_llama_og2_step_182313_on_evo2_rrna_highly_conserved_PMC4140814
        / "ribosomal_rrna_highly_conserved_PMC4140814.fasta"
    )
    assert llama_7b_8k_og2.exists() and fasta_path.exists()

    exporter = HFLlamaExporter(llama_7b_8k_og2)
    hf_path = tmp_path / "hf_checkpoint"
    exporter.apply(hf_path)
    importer = HFEdenLlamaImporter(hf_path)
    importer.apply(tmp_path / "nemo_checkpoint")
    original_predictions, original_fasta_to_index = predict_metagenome(
        llama_7b_8k_og2, fasta_path, tmp_path / "original_predictions"
    )
    new_predictions, new_fasta_to_index = predict_metagenome(
        tmp_path / "nemo_checkpoint", fasta_path, tmp_path / "new_predictions"
    )
    assert original_fasta_to_index == new_fasta_to_index, "Fasta to index mapping is not the same, need better logic."
    for key in ["seq_idx", "log_probs_seqs", "loss_mask"]:
        torch.testing.assert_close(
            original_predictions[key],
            new_predictions[key],
            msg=lambda diff: f"Results for {key} are not the same:\n{diff}",
        )
