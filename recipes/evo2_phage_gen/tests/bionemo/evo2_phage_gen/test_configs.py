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

import tomllib
from pathlib import Path

import yaml


RECIPE_ROOT = Path(__file__).parents[3]


def test_config_directory_contains_only_supported_runtime_configs():
    """The public config directory should expose only supported end-to-end runtime inputs."""
    config_dir = RECIPE_ROOT / "configs"
    expected = {
        "arc_genome_design_filtering_local.yaml",
        "gdpo_phage_megatron.yaml",
        "grpo_phage_megatron.yaml",
        "nemo_rl_defaults/grpo_math_1B.yaml",
        "nemo_rl_defaults/grpo_math_1B_megatron.yaml",
        "phage_safety_assets.yaml",
        "phage_safety_policy.yaml",
        "phage_safety_reference_controls.yaml",
        "sft_microviridae_dataset.yaml",
        "sft_microviridae_preprocess.yaml",
    }

    actual = {path.relative_to(config_dir).as_posix() for path in config_dir.rglob("*.yaml")}

    assert actual == expected


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


def test_grpo_config_uses_prompt_batch_size_for_evo2_generation():
    """GRPO should default to the known-good serial Evo2 Megatron generation path."""
    config_path = RECIPE_ROOT / "configs" / "grpo_phage_megatron.yaml"
    config = yaml.safe_load(config_path.read_text())

    generation_batch_size = config["policy"]["generation_batch_size"]
    generation_config = config["policy"]["generation"]
    mcore_generation_config = config["policy"]["generation"]["mcore_generation_config"]
    tensor_model_parallel_size = config["policy"]["megatron_cfg"]["tensor_model_parallel_size"]

    assert generation_config["max_new_tokens"] == config["env"]["phage_qc"]["genome_length_max"] - 4
    assert config["env"]["phage_qc"]["weight_nucleotide_pass"] == 0.0
    assert config["env"]["phage_qc"]["dustmask_filter"] is True
    assert config["env"]["phage_qc"]["dustmasker_bin"] == "dustmasker"
    assert config["env"]["phage_qc"]["dustmask_use_external"] is True
    assert config["env"]["phage_qc"]["weight_dustmask_end"] == 1.0
    external_qc = config["env"]["phage_qc"]["external_qc"]
    sequence_safety = config["env"]["phage_qc"]["sequence_safety"]
    assert sequence_safety["enabled"] is True
    assert sequence_safety["host_domain"] == "BACTERIA"
    assert sequence_safety["host_evidence"]["confirmed"] is True
    assert sequence_safety["host_evidence"]["replication_host_domains"] == ["BACTERIA"]
    assert sequence_safety["policy_path"] == "configs/phage_safety_policy.yaml"
    assert sequence_safety["asset_manifest_path"] == "data/external/safety/asset_manifest.yaml"
    assert external_qc["lovis4u_mmseqs_threads"] == 8
    assert external_qc["lovis4u_metrics_only"] is True
    assert generation_config["temperature"] > 0.0
    assert generation_config["top_k"] is None
    assert generation_config["top_p"] == 1.0
    assert generation_batch_size == 1
    assert mcore_generation_config["max_requests"] % tensor_model_parallel_size == 0
    assert mcore_generation_config["max_requests"] >= generation_batch_size
    assert mcore_generation_config["prompt_batch_size"] == generation_batch_size
    assert "evo2_batched_decode_size" not in mcore_generation_config


def test_gdpo_config_uses_positional_objectives_and_mmseqs_diversity():
    """GDPO should return macro-objective rewards and pin MMseqs clustering semantics."""
    config_path = RECIPE_ROOT / "configs" / "gdpo_phage_megatron.yaml"
    config = yaml.safe_load(config_path.read_text())
    env_config = config["env"]["phage_qc"]
    mmseqs_config = env_config["mmseqs_cluster_diversity"]
    objectives = env_config["gdpo_objectives"]

    assert config["defaults"] == "grpo_phage_megatron.yaml"
    assert env_config["reward_output_mode"] == "gdpo"
    assert config["loss_fn"]["reference_policy_kl_penalty"] == 0.001
    assert config["loss_fn"]["token_level_loss"] is False
    assert config["grpo"]["seq_logprob_error_threshold"] == 1.5
    assert config["policy"]["generation"]["mcore_generation_config"]["generation_adapter_config"]["seed"] == 42
    assert config["policy"]["megatron_cfg"]["optimizer"]["lr"] == 1.0e-6
    assert config["policy"]["megatron_cfg"]["optimizer"]["min_lr"] == 1.0e-7
    assert config["policy"]["megatron_cfg"]["scheduler"]["lr_warmup_init"] == 1.0e-7
    assert (
        config["checkpointing"]["metric_name"]
        == "val:phage_qc/binary_safety_qualified_full_qc_cluster_deduplicated_rate"
    )
    assert [objective["name"] for objective in objectives] == [
        "valid_nt_chars",
        "genome_length",
        "gc_content",
        "nt_homopolymer",
        "dustmask_end",
        "nucleotide_pass",
        "protein_hit_count",
        "tropism",
        "required_genes",
        "synteny",
        "average_protein_identity",
        "mmseqs_cluster_diversity",
        "safety_amr",
        "safety_toxin",
        "safety_lysogeny",
    ]
    assert all(len(objective["columns"]) == 1 for objective in objectives)
    assert all("reward" not in objective["columns"] for objective in objectives)
    objective_by_name = {objective["name"]: objective for objective in objectives}
    assert "reward_mmseqs_cluster_diversity" in objective_by_name["mmseqs_cluster_diversity"]["columns"]
    assert "reward_dustmask_end" in objectives[4]["columns"]
    for name in ("safety_amr", "safety_toxin", "safety_lysogeny"):
        assert objective_by_name[name]["requires_safety_eligibility"] is False
    for objective in objectives[:-3]:
        assert objective["requires_safety_eligibility"] is True
    assert env_config["weight_mmseqs_cluster_diversity"] == 1.0
    assert env_config["dustmask_filter"] is True
    assert env_config["weight_dustmask_end"] == 1.0
    assert env_config["external_qc"]["fail_on_error"] is True
    assert env_config["external_qc"]["tool_bin_dir"] == "data/external/bin"
    assert env_config["external_qc"]["timeout_seconds"] == 1800
    assert env_config["external_qc"]["lovis4u_parallel_jobs"] == 12
    assert env_config["external_qc"]["lovis4u_collect_pdfs"] is False
    assert mmseqs_config == {
        "enabled": True,
        "mmseqs_bin": "data/external/bin/mmseqs",
        "work_dir": "data/checkpoints/phage_gdpo_base_microviridae_batched96_stockgdpo_fullfalse_decodefix_clusterfix_gdpo12_mmseqs_cluster_diversity",
        "keep_artifacts": False,
        "min_seq_id": 0.99,
        "coverage": 0.0,
        "cov_mode": 0,
        "seq_id_mode": 0,
        "cluster_mode": 0,
        "threads": 16,
        "verbosity": 0,
    }
    assert config["grpo"]["num_generations_per_prompt"] == 96
    assert config["grpo"]["val_at_start"] is False
    assert config["grpo"]["val_at_end"] is True
    assert config["policy"]["train_global_batch_size"] == 96
    assert config["policy"]["train_micro_batch_size"] == 1
    assert config["policy"]["generation_batch_size"] == 96
    assert config["policy"]["logprob_batch_size"] == 1
    mcore_generation_config = config["policy"]["generation"]["mcore_generation_config"]
    assert mcore_generation_config["prompt_batch_size"] == 96
    assert mcore_generation_config["max_requests"] == 96
    assert "lr1e-6-kl0.001" in config["logger"]["wandb"]["name"]
    assert "batched96" in config["logger"]["wandb"]["name"]
    assert config["logger"]["wandb"]["name"].startswith("gdpo-phage")


def test_every_inherited_grpo_and_gdpo_config_keeps_mandatory_safety_enabled():
    """Supported GRPO and GDPO configs must keep the mandatory safety gate."""
    from bionemo.evo2_phage_gen.rl_readiness import _load_config_with_defaults

    config_dir = RECIPE_ROOT / "configs"
    config_paths = sorted({*config_dir.glob("grpo_phage*.yaml"), *config_dir.glob("gdpo_phage*.yaml")})
    assert config_paths

    for config_path in config_paths:
        resolved = _load_config_with_defaults(config_path)
        safety = resolved["env"]["phage_qc"]["sequence_safety"]
        assert type(safety["enabled"]) is bool and safety["enabled"] is True, config_path.name
        assert safety["host_domain"] in {"BACTERIA", "ARCHAEA", "BACTERIA_AND_ARCHAEA"}, config_path.name
        evidence = safety["host_evidence"]
        assert type(evidence["confirmed"]) is bool and evidence["confirmed"] is True, config_path.name
        assert set(evidence["replication_host_domains"]) <= {
            "BACTERIA",
            "ARCHAEA",
            "BACTERIA_AND_ARCHAEA",
        }, config_path.name
        for path_key in (
            "policy_path",
            "asset_manifest_path",
            "diamond_tool_pin_path",
            "mmseqs_tool_pin_path",
            "work_dir",
        ):
            assert isinstance(safety[path_key], str) and safety[path_key], (config_path.name, path_key)

        if config_path.name.startswith("gdpo_"):
            assert (
                resolved["checkpointing"]["metric_name"]
                == "val:phage_qc/binary_safety_qualified_full_qc_cluster_deduplicated_rate"
            ), config_path.name
            objectives = resolved["env"]["phage_qc"]["gdpo_objectives"]
            objective_by_name = {objective["name"]: objective for objective in objectives}
            assert {
                "safety_amr",
                "safety_toxin",
                "safety_lysogeny",
            } <= objective_by_name.keys(), config_path.name
            for name, objective in objective_by_name.items():
                assert type(objective.get("requires_safety_eligibility")) is bool, (config_path.name, name)
                assert objective["requires_safety_eligibility"] is (not name.startswith("safety_")), (
                    config_path.name,
                    name,
                )


def test_report_runtime_declares_tabulate_dependency():
    """Installed report commands must include pandas' Markdown-table backend."""
    config = tomllib.loads((RECIPE_ROOT / "pyproject.toml").read_text())

    assert "tabulate" in config["project"]["dependencies"]


def test_recipe_docker_context_excludes_generated_assets_and_runs_nonroot():
    """The recipe build context must exclude generated assets and drop root after build."""
    ignore_path = RECIPE_ROOT / ".dockerignore"
    assert ignore_path.is_file()
    patterns = set(ignore_path.read_text().splitlines())
    assert {
        "data/checkpoints",
        "data/external",
        "data/arc_pipeline_patched",
        "dist",
    } <= patterns

    dockerfile = (RECIPE_ROOT / "Dockerfile").read_text()
    assert "useradd" in dockerfile
    assert "\nUSER bionemo\n" in dockerfile
