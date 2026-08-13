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

import importlib.util
from pathlib import Path

import pandas as pd

from bionemo.evo2_phage_gen.qc import NucleotideQCConfig


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "score_paper_hpo_generation.py"
SPEC = importlib.util.spec_from_file_location("score_paper_hpo_generation", SCRIPT)
assert SPEC and SPEC.loader
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)


def test_summary_uses_the_shared_nucleotide_qc_config():
    scored = pd.DataFrame(
        {
            "valid_nt_chars": [True],
            "genome_length": [10],
            "gc_content": [50.0],
            "max_nt_homopolymer_length": [2],
            "reward_nucleotide_pass": [True],
        }
    )
    config = NucleotideQCConfig(
        genome_length_min=8,
        genome_length_max=12,
        gc_content_min=40.0,
        gc_content_max=60.0,
        homopolymer_max=3,
    )

    summary = scoring._summarize_cell("cell", 1, 0.7, scored, config)

    assert summary["length_pass"] == 1
    assert summary["gc_pass"] == 1
    assert summary["homopolymer_pass"] == 1
