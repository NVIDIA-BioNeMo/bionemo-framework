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

"""Tests for recipe configuration files."""

from pathlib import Path

import yaml


RECIPE_ROOT = Path(__file__).parents[3]


def test_arc_genome_design_filtering_local_config_is_safe_by_default():
    """The Arc pipeline config should parse and avoid external tools by default."""
    config_path = RECIPE_ROOT / "configs" / "arc_genome_design_filtering_local.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["nucleotide_filtering"] is True
    assert config["orf_filtering"] is False
    assert config["homology_filtering"] is False
    assert config["diversification_filtering"] is False
    assert config["genetic_architecture_visualization_and_synteny_filtering"] is False
    assert config["reference_genome_fasta"].endswith("data/external/arc_evo2/phage_gen/data/NC_001422_1.fna")
    assert config["reference_tropism_protein"].endswith(
        "data/external/arc_evo2/phage_gen/data/NC_001422.1_Gprotein.fasta"
    )


def test_arc_curated_smoke_config_targets_bundled_candidates():
    """The curated smoke config should run only Arc's dependency-light nucleotide stage."""
    config_path = RECIPE_ROOT / "configs" / "arc_genome_design_filtering_curated_smoke.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["evo_gen_seqs_fasta_file_save_location"].endswith(
        "data/external/arc_evo2/phage_gen/data/all_generated_phages.fasta"
    )
    assert config["nucleotide_filtering"] is True
    assert config["orf_filtering"] is False
    assert config["homology_filtering"] is False
    assert config["diversification_filtering"] is False
    assert config["genetic_architecture_visualization_and_synteny_filtering"] is False


def test_docs_and_configs_do_not_use_stale_workspace_paths():
    """Recipe docs and configs should be portable across checkout locations."""
    checked_paths = [
        RECIPE_ROOT / "README.md",
        RECIPE_ROOT / "examples" / "replication_walkthrough.ipynb",
        *sorted((RECIPE_ROOT / "configs").rglob("*.yaml")),
    ]

    stale_prefix = "/workspaces/bionemo-framework"
    offenders = [path for path in checked_paths if stale_prefix in path.read_text()]

    assert offenders == []


def test_grpo_config_enables_opt_in_evo2_batched_decode():
    """GRPO should keep Megatron generation batch capacity aligned with Evo2 batched decode."""
    config_path = RECIPE_ROOT / "configs" / "grpo_phage_megatron.yaml"
    config = yaml.safe_load(config_path.read_text())

    generation_batch_size = config["policy"]["generation_batch_size"]
    mcore_generation_config = config["policy"]["generation"]["mcore_generation_config"]

    assert generation_batch_size == 8
    assert mcore_generation_config["max_requests"] == generation_batch_size
    assert mcore_generation_config["evo2_batched_decode_size"] == generation_batch_size
