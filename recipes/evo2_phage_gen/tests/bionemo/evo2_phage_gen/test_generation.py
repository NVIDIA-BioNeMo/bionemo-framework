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

"""Tests for ``bionemo.evo2_phage_gen.generation``."""

import json

from bionemo.evo2_phage_gen.generation import infer_jsonl_to_fasta, phix174_prompts, write_prompt_sweep_jsonl


def test_phix174_prompts_use_reference_prefixes():
    """Prompt strings should be paper-style soft-token prefixes of the PhiX174 consensus start."""
    assert phix174_prompts(prompt_lengths=[4, 5, 9]) == {
        4: "+~GAGT",
        5: "+~GAGTT",
        9: "+~GAGTTTTAT",
    }


def test_write_prompt_sweep_jsonl_repeats_prompts(tmp_path):
    """Prompt JSONL files should be ready for ``infer_evo2 --prompt-file``."""
    paths = write_prompt_sweep_jsonl(tmp_path, prompt_lengths=[4, 6], num_prompts=2, id_prefix="test")

    assert [path.name for path in paths] == ["test_prompt4_2.jsonl", "test_prompt6_2.jsonl"]
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert records == [
        {"id": "test_prompt4_0000", "prompt": "+~GAGT"},
        {"id": "test_prompt4_0001", "prompt": "+~GAGT"},
    ]


def test_infer_jsonl_to_fasta_prepends_prompt_and_trims_eos(tmp_path):
    """FASTA reconstruction should combine prompt and completion before QC."""
    input_jsonl = tmp_path / "prompt4_temp0.7.jsonl"
    input_jsonl.write_text(
        json.dumps({"id": "seq1", "prompt": "+~GAGT", "completion": "ACGT STOP"})
        + "\n"
        + json.dumps({"id": "seq2", "prompt": "+~GAGT", "completion": "tgca"})
        + "\n"
    )
    output_fasta = tmp_path / "generated.fasta"

    infer_jsonl_to_fasta([input_jsonl], output_fasta)

    assert output_fasta.read_text() == (">seq1|prompt4_temp0.7\nGAGTACGT\n>seq2|prompt4_temp0.7\nGAGTTGCA\n")
