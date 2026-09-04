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


def test_config_directory_contains_only_supported_runtime_configs():
    """The public config directory should expose only supported end-to-end runtime inputs."""
    config_dir = RECIPE_ROOT / "configs"
    expected = {
        "arc_genome_design_filtering_local.yaml",
        "gdpo_phage_megatron.yaml",
        "grpo_phage_megatron.yaml",
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
    dtensor_config = config["policy"]["dtensor_cfg"]
    tensor_model_parallel_size = config["policy"]["megatron_cfg"]["tensor_model_parallel_size"]
    train_data = config["data"]["train"]

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
    assert sequence_safety["batch_size"] == 128
    assert sequence_safety["orf_workers"] == 16
    assert sequence_safety["phrogs_threads"] == 16
    assert external_qc["lovis4u_mmseqs_threads"] == 8
    assert external_qc["lovis4u_metrics_only"] is True
    assert generation_config["temperature"] > 0.0
    assert generation_config["top_k"] is None
    assert generation_config["top_p"] == 1.0
    assert generation_batch_size == 1
    assert config["policy"]["model_name"] == "bionemo/evo2_7b"
    assert config["policy"]["offload_optimizer_for_logprob"] is False
    assert config["policy"]["megatron_cfg"]["enabled"] is True
    assert dtensor_config["enabled"] is False
    assert "_v2" not in dtensor_config
    assert config["checkpointing"]["model_save_format"] is None
    assert config["checkpointing"]["pretrained_checkpoint"]["format"] == "megatron_bridge"
    assert mcore_generation_config["max_requests"] % tensor_model_parallel_size == 0
    assert mcore_generation_config["max_requests"] >= generation_batch_size
    assert mcore_generation_config["prompt_batch_size"] == generation_batch_size
    assert "evo2_batched_decode_size" not in mcore_generation_config
    assert config["logger"]["tensorboard_enabled"] is True
    assert config["logger"]["tensorboard"] == {}
    assert train_data["dataset_name"] == "bionemo.evo2_phage_gen.nemo_rl_processors.PhageOpenAIFormatDataset"


def test_gdpo_config_uses_positional_objectives_and_mmseqs_diversity():
    """GDPO should return macro-objective rewards and use 99% MMseqs diversity."""
    config_path = RECIPE_ROOT / "configs" / "gdpo_phage_megatron.yaml"
    config = yaml.safe_load(config_path.read_text())
    env_config = config["env"]["phage_qc"]
    mmseqs_config = env_config["mmseqs_cluster_diversity"]
    objectives = env_config["gdpo_objectives"]
    validation_data = config["data"]["validation"]

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
    assert env_config["external_qc"]["lovis4u_parallel_jobs"] == 32
    assert env_config["external_qc"]["lovis4u_mmseqs_threads"] == 8
    assert env_config["external_qc"]["lovis4u_collect_pdfs"] is False
    assert env_config["sequence_safety"]["batch_size"] >= config["policy"]["generation_batch_size"]
    assert env_config["sequence_safety"] == {
        "batch_size": 128,
        "orf_workers": 32,
        "phrogs_threads": 64,
        "threads": 32,
    }
    assert config["run_id"] == "phix174_gdpo"
    assert mmseqs_config["work_dir"] == "data/checkpoints/${run_id}_mmseqs_cluster_diversity"
    assert {key: value for key, value in mmseqs_config.items() if key != "work_dir"} == {
        "enabled": True,
        "mmseqs_bin": "data/external/bin/mmseqs",
        "keep_artifacts": False,
        "min_seq_id": 0.99,
        "coverage": 0.0,
        "cov_mode": 0,
        "seq_id_mode": 0,
        "cluster_mode": 0,
        "threads": 64,
        "verbosity": 0,
    }
    assert config["grpo"]["num_prompts_per_step"] == 2
    assert config["grpo"]["num_generations_per_prompt"] == 48
    assert config["grpo"]["num_prompts_per_step"] * config["grpo"]["num_generations_per_prompt"] == 96
    assert config["grpo"]["val_at_start"] is False
    assert config["grpo"]["val_at_end"] is True
    assert config["policy"]["train_global_batch_size"] == 96
    assert config["policy"]["train_micro_batch_size"] == 1
    assert config["policy"]["generation_batch_size"] == 96
    assert config["policy"]["logprob_batch_size"] == 1
    mcore_generation_config = config["policy"]["generation"]["mcore_generation_config"]
    assert mcore_generation_config["prompt_batch_size"] == 12
    assert mcore_generation_config["max_requests"] == 12
    assert mcore_generation_config["generation_adapter_config"]["ignore_eos"] is True
    assert mcore_generation_config["generation_adapter_config"]["strict_generation"] is True
    assert config["policy"]["sequence_packing"]["enabled"] is False
    assert config["logger"]["wandb_enabled"] is False
    assert config["logger"]["wandb"]["name"] == "phix174-gdpo"
    assert config["cluster"] == {"gpus_per_node": 8, "num_nodes": 1}
    assert validation_data["dataset_name"] == "bionemo.evo2_phage_gen.nemo_rl_processors.PhageOpenAIFormatDataset"


def test_phix_example_documents_every_gdpo_objective():
    """The fixed PhiX example should explain every score enabled by its GDPO config."""
    config = yaml.safe_load((RECIPE_ROOT / "configs" / "gdpo_phage_megatron.yaml").read_text())
    readme = (RECIPE_ROOT / "examples" / "README.md").read_text()
    heading = "## Current PhiX174 GDPO score definitions"

    assert heading in readme
    score_section = readme.split(heading, maxsplit=1)[1]
    for objective in config["env"]["phage_qc"]["gdpo_objectives"]:
        assert f"`{objective['name']}`" in score_section


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
            "diamond_bin",
            "mmseqs_bin",
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
