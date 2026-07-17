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

import json
from pathlib import Path

import yaml

from bionemo.evo2_phage_gen.generation import ensure_paper_useful_rl_prompt_files


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
    assert generation_config["temperature"] > 0.0
    assert generation_config["top_k"] is None
    assert generation_config["top_p"] == 1.0
    assert generation_batch_size == 1
    assert mcore_generation_config["max_requests"] % tensor_model_parallel_size == 0
    assert mcore_generation_config["max_requests"] >= generation_batch_size
    assert mcore_generation_config["prompt_batch_size"] == generation_batch_size
    assert "evo2_batched_decode_size" not in mcore_generation_config


def test_grpo_rl100_config_targets_best_paper_region():
    """The 100-step RL config should use the best downstream HPO prompt region."""
    config_path = RECIPE_ROOT / "configs" / "grpo_phage_megatron_rl100.yaml"
    config = yaml.safe_load(config_path.read_text())
    prompt_path = RECIPE_ROOT / config["data"]["train"]["data_path"]
    validation_path = RECIPE_ROOT / config["data"]["validation"]["data_path"]
    generation_config = config["policy"]["generation"]

    ensure_paper_useful_rl_prompt_files(RECIPE_ROOT / "data")
    assert prompt_path.exists()
    assert validation_path.exists()
    prompts = [
        json.loads(line)["messages"][0]["content"].removeprefix("+~")
        for line in prompt_path.read_text().splitlines()
        if line.strip()
    ]
    validation_prompts = [
        json.loads(line)["messages"][0]["content"].removeprefix("+~")
        for line in validation_path.read_text().splitlines()
        if line.strip()
    ]
    assert [len(prompt) for prompt in prompts] == [4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 11]
    assert len(validation_prompts) == 64
    assert {len(prompt) for prompt in validation_prompts} == {10}
    assert config["data"]["validation"]["env_name"] == "phage_qc"
    assert config["grpo"]["max_num_epochs"] >= config["grpo"]["max_num_steps"]
    assert config["grpo"]["max_num_steps"] == 100
    assert config["grpo"]["num_generations_per_prompt"] == 96
    assert config["grpo"]["val_period"] == 10
    assert config["grpo"]["val_at_start"] is True
    assert config["grpo"]["val_at_end"] is True
    assert config["grpo"]["val_batch_size"] == 96
    assert config["grpo"]["max_val_samples"] == 96
    assert config["policy"]["generation_batch_size"] == 96
    assert generation_config["max_new_tokens"] == 5989
    assert generation_config["temperature"] == 1.0
    assert generation_config["top_k"] == 4
    assert config["env"]["phage_qc"]["weight_nucleotide_pass"] == 1.0
    assert config["env"]["phage_qc"]["dustmask_filter"] is True
    assert config["env"]["phage_qc"]["dustmask_end_window"] == 200
    assert config["env"]["phage_qc"]["weight_dustmask_end"] == 1.0
    for removed_weight in [
        "weight_orf",
        "weight_coding_density",
        "weight_genetic_architecture",
        "weight_checkv",
        "weight_training_data_identity",
        "weight_reference_genome_identity",
        "weight_mmseqs_clustering",
        "weight_diversity",
    ]:
        assert removed_weight not in config["env"]["phage_qc"]
    assert config["env"]["phage_qc"]["external_qc"]["enabled"] is True
    assert config["env"]["phage_qc"]["external_qc"]["enable_tropism"] is True
    assert config["env"]["phage_qc"]["external_qc"]["enable_synteny"] is True
    assert config["env"]["phage_qc"]["external_qc"]["synteny_mode"] == "full"
    for removed_flag in [
        "checkv_db_path",
        "enable_orf",
        "enable_coding_density",
        "enable_genetic_architecture",
        "enable_checkv",
        "enable_training_data_identity",
        "enable_reference_genome_identity",
        "enable_mmseqs_clustering",
        "enable_diversity",
    ]:
        assert removed_flag not in config["env"]["phage_qc"]["external_qc"]
    assert config["env"]["phage_qc"]["external_qc"]["enable_average_protein_identity"] is True
    assert config["env"]["phage_qc"]["external_qc"]["enable_required_genes"] is True
    assert config["env"]["phage_qc"]["weight_synteny"] == 0.25
    assert config["env"]["phage_qc"]["weight_average_protein_identity"] == 0.25
    assert config["env"]["phage_qc"]["weight_required_genes"] == 0.1
    assert config["env"]["phage_qc"]["external_qc"]["required_genes_evidence_target"] == 9.0
    assert config["env"]["phage_qc"]["external_qc"]["lovis4u_parallel_jobs"] == 12
    assert config["env"]["phage_qc"]["external_qc"]["lovis4u_collect_pdfs"] is False
    assert config["logger"]["wandb_enabled"] is True
    assert config["logger"]["wandb"]["project"] == "evo2_phage_design_rl_focused_qc"
    assert config["logger"]["wandb"]["name"] == "grpo-phage-rl100-full-qc-batched96"
    mcore_generation_config = generation_config["mcore_generation_config"]
    assert mcore_generation_config["prompt_batch_size"] == 96
    assert mcore_generation_config["max_requests"] == 96
    assert mcore_generation_config["enable_chunked_prefill"] is False
    assert (
        mcore_generation_config["generation_adapter"]
        == "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"
    )

    arc_config = yaml.safe_load((RECIPE_ROOT / "configs" / "arc_genome_design_filtering_local.yaml").read_text())
    assert arc_config["training_data_sequence_identity_filter"] is False
    assert arc_config["training_data_sequence_identity_range"] == [0, 98.9]
    assert arc_config["genetic_architecture_filter"] is False
    assert arc_config["mmseqs_reference_genome_sequence_identity_remove_filter"] is False
    assert arc_config["genetic_architecture_remove_filter"] is False
    assert arc_config["mmseqs_reference_genome_sequence_identity_keep_range"] == [0, 98.9]
    assert arc_config["syntenic_gene_count_range"] == [10, 12]
    assert arc_config["total_gene_count_range"] == [10, 12]
    assert arc_config["syntenic_total_gene_count_remove"] == [[11, 11]]
    assert arc_config["required_genes_evidence_target"] == 9.0
    assert arc_config["lovis4u_parallel_jobs"] == 12
    assert arc_config["lovis4u_chunk_size"] == 12
    assert arc_config["lovis4u_collect_pdfs"] is False
    assert arc_config["allow_gff_product_order_synteny_fallback"] is False


def test_grpo_rl500_config_extends_current_full_qc_rollout():
    """The 500-step config should directly carry the current full-QC rollout settings."""
    config_path = RECIPE_ROOT / "configs" / "grpo_phage_megatron_rl500.yaml"
    config = yaml.safe_load(config_path.read_text())
    generation_config = config["policy"]["generation"]
    mcore_generation_config = generation_config["mcore_generation_config"]

    assert config["defaults"] == "grpo_phage_megatron.yaml"
    assert config["grpo"]["max_num_epochs"] == 500
    assert config["grpo"]["num_generations_per_prompt"] == 96
    assert config["grpo"]["max_num_steps"] == 500
    assert config["grpo"]["val_batch_size"] == 96
    assert config["grpo"]["max_val_samples"] == 96
    assert config["grpo"]["val_at_start"] is True
    assert config["grpo"]["seq_logprob_error_threshold"] == 1.5
    assert config["policy"]["train_global_batch_size"] == 96
    assert config["policy"]["generation_batch_size"] == 96
    assert generation_config["max_new_tokens"] == 5989
    assert generation_config["temperature"] == 1.0
    assert generation_config["top_k"] == 4
    assert mcore_generation_config["prompt_batch_size"] == 96
    assert mcore_generation_config["max_requests"] == 96
    assert (
        mcore_generation_config["generation_adapter"]
        == "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"
    )
    assert mcore_generation_config["generation_adapter_config"]["seed"] == 42
    assert config["data"]["train"]["data_path"] == "data/phage_prompts_paper_useful_rl.jsonl"
    assert config["data"]["validation"]["data_path"] == "data/phage_prompts_paper_useful_rl_validation_prompt10.jsonl"
    assert config["env"]["phage_qc"]["weight_synteny"] == 0.25
    assert config["env"]["phage_qc"]["weight_average_protein_identity"] == 0.25
    assert config["env"]["phage_qc"]["weight_required_genes"] == 0.1
    assert config["env"]["phage_qc"]["external_qc"]["lovis4u_parallel_jobs"] == 12
    assert config["env"]["phage_qc"]["external_qc"]["lovis4u_collect_pdfs"] is False
    assert config["env"]["phage_qc"]["dustmask_filter"] is True
    assert config["env"]["phage_qc"]["weight_dustmask_end"] == 1.0
    assert config["logger"]["log_dir"] == "data/checkpoints/phage_grpo_logs_rl500"
    assert config["logger"]["wandb_enabled"] is True
    assert config["logger"]["wandb"]["project"] == "evo2_phage_design_rl_focused_qc"
    assert config["logger"]["wandb"]["name"] == "grpo-phage-rl500-full-qc-batched96"
    assert config["checkpointing"]["checkpoint_dir"] == "data/checkpoints/phage_grpo_rl500_round2"


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
    assert config["grpo"]["seq_logprob_error_threshold"] == 1.5
    assert config["policy"]["generation"]["mcore_generation_config"]["generation_adapter_config"]["seed"] == 42
    assert config["policy"]["megatron_cfg"]["optimizer"]["lr"] == 1.0e-6
    assert config["policy"]["megatron_cfg"]["optimizer"]["min_lr"] == 1.0e-7
    assert config["policy"]["megatron_cfg"]["scheduler"]["lr_warmup_init"] == 1.0e-7
    assert config["checkpointing"]["metric_name"] == "val:phage_qc/binary_core_pass_cluster_deduplicated_rate"
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
    ]
    assert all(len(objective["columns"]) == 1 for objective in objectives)
    assert all("reward" not in objective["columns"] for objective in objectives)
    assert "reward_mmseqs_cluster_diversity" in objectives[-1]["columns"]
    assert "reward_dustmask_end" in objectives[4]["columns"]
    assert env_config["weight_mmseqs_cluster_diversity"] == 1.0
    assert env_config["dustmask_filter"] is True
    assert env_config["weight_dustmask_end"] == 1.0
    assert env_config["external_qc"]["fail_on_error"] is True
    assert env_config["external_qc"]["timeout_seconds"] == 1800
    assert env_config["external_qc"]["lovis4u_parallel_jobs"] == 12
    assert env_config["external_qc"]["lovis4u_collect_pdfs"] is False
    assert mmseqs_config == {
        "enabled": True,
        "mmseqs_bin": "mmseqs",
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


def test_gdpo_tp1dp2_smoke_uses_local_decode_48_but_training_microbatch_1():
    """The DP smoke should split decode 48/48 without repeating the MBS48 OOM."""
    config_path = RECIPE_ROOT / "configs" / "gdpo_phage_megatron_gdpo12_tp1dp2_mbs1_smoke.yaml"
    config = yaml.safe_load(config_path.read_text())
    policy = config["policy"]
    mcore_generation_config = policy["generation"]["mcore_generation_config"]

    assert config["defaults"] == "gdpo_phage_megatron_gdpo12_batched96_smoke.yaml"
    assert policy["train_global_batch_size"] == 96
    assert policy["generation_batch_size"] == 96
    assert policy["train_micro_batch_size"] == 1
    assert policy["logprob_batch_size"] == 1
    assert policy["megatron_cfg"]["tensor_model_parallel_size"] == 1
    assert mcore_generation_config["max_requests"] == 48
    assert mcore_generation_config["prompt_batch_size"] == 48
    output_paths = [
        config["checkpointing"]["checkpoint_dir"],
        config["env"]["phage_qc"]["external_qc"]["work_dir"],
        config["env"]["phage_qc"]["mmseqs_cluster_diversity"]["work_dir"],
        config["logger"]["log_dir"],
    ]
    assert all("tp1dp2_finalpatch_mbs1_gdpo12_smoke" in path for path in output_paths)
    assert config["checkpointing"]["pretrained_checkpoint"]["path"] == (
        "data/checkpoints/evo2_7b_microviridae_mbridge"
    )


def test_gdpo_tp2dp1_smoke_uses_full_96_request_collective_and_isolated_output_paths():
    config_path = RECIPE_ROOT / "configs" / "gdpo_phage_megatron_gdpo12_tp2dp1_dpgenfix_smoke.yaml"
    config = yaml.safe_load(config_path.read_text())
    policy = config["policy"]
    mcore_generation_config = policy["generation"]["mcore_generation_config"]

    assert config["defaults"] == "gdpo_phage_megatron_gdpo12_batched96_smoke.yaml"
    assert policy["train_global_batch_size"] == 96
    assert policy["generation_batch_size"] == 96
    assert policy["train_micro_batch_size"] == 1
    assert policy["logprob_batch_size"] == 1
    assert policy["megatron_cfg"]["tensor_model_parallel_size"] == 2
    assert mcore_generation_config["max_requests"] == 96
    assert mcore_generation_config["prompt_batch_size"] == 96
    output_paths = [
        config["checkpointing"]["checkpoint_dir"],
        config["env"]["phage_qc"]["external_qc"]["work_dir"],
        config["env"]["phage_qc"]["mmseqs_cluster_diversity"]["work_dir"],
        config["logger"]["log_dir"],
    ]
    assert all("tp2dp1_dpgenfix_gdpo12_smoke" in path for path in output_paths)
    assert config["checkpointing"]["pretrained_checkpoint"]["path"] == (
        "data/checkpoints/evo2_7b_microviridae_mbridge"
    )


def test_grpo_diagnostic_config_keeps_full_length_scoring_but_smaller_rollouts():
    """The diagnostic RL config should be faster while preserving full-length QC scoring."""
    config_path = RECIPE_ROOT / "configs" / "grpo_phage_megatron_diagnostic.yaml"
    config = yaml.safe_load(config_path.read_text())
    generation_config = config["policy"]["generation"]

    assert config["grpo"]["max_num_steps"] == 2
    assert config["grpo"]["num_generations_per_prompt"] == 8
    assert config["policy"]["train_global_batch_size"] == 8
    assert generation_config["max_new_tokens"] == 5990
    assert generation_config["temperature"] == 1.0
    assert generation_config["top_k"] == 4
    assert config["checkpointing"]["enabled"] is False


def test_grpo_batched_diagnostic_config_uses_batched_evo2_generation():
    """The batched diagnostic should exercise the non-serial Evo2 generation path."""
    config_path = RECIPE_ROOT / "configs" / "grpo_phage_megatron_batched_diagnostic.yaml"
    config = yaml.safe_load(config_path.read_text())
    mcore_generation_config = config["policy"]["generation"]["mcore_generation_config"]

    assert config["grpo"]["max_num_steps"] == 1
    assert config["grpo"]["num_generations_per_prompt"] == 8
    assert config["policy"]["generation_batch_size"] == 8
    assert config["policy"]["train_global_batch_size"] == 8
    assert mcore_generation_config["prompt_batch_size"] == 8
    assert mcore_generation_config["max_requests"] == 8
    assert mcore_generation_config["enable_chunked_prefill"] is False
    assert (
        mcore_generation_config["generation_adapter"]
        == "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"
    )
    assert config["checkpointing"]["enabled"] is False


def test_grpo_batched_no_cg_diagnostic_disables_cuda_graphs():
    """The no-CG diagnostic should isolate batched generation from CUDA graph warmup."""
    config_path = RECIPE_ROOT / "configs" / "grpo_phage_megatron_batched_no_cg_diagnostic.yaml"
    config = yaml.safe_load(config_path.read_text())
    mcore_generation_config = config["policy"]["generation"]["mcore_generation_config"]

    assert config["policy"]["generation_batch_size"] == 8
    assert mcore_generation_config["prompt_batch_size"] == 8
    assert mcore_generation_config["max_requests"] == 8
    assert mcore_generation_config["enable_chunked_prefill"] is False
    assert (
        mcore_generation_config["generation_adapter"]
        == "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"
    )
    assert mcore_generation_config["cuda_graph_impl"] == "none"
    assert mcore_generation_config["inference_cuda_graph_scope"] == "none"
    assert mcore_generation_config["use_cuda_graphs_for_non_decode_steps"] is False


def test_vllm_tp2_one_step_smoke_uses_capacity_bounded_mixed_batching():
    config_path = RECIPE_ROOT / "configs" / "gdpo_phage_vllm_tp2_one_step_smoke.yaml"
    config = yaml.safe_load(config_path.read_text())
    grpo = config["grpo"]
    policy = config["policy"]
    generation = policy["generation"]
    vllm_kwargs = generation["vllm_kwargs"]

    checks = {
        "P": grpo["num_prompts_per_step"] == 8,
        "K": grpo["num_generations_per_prompt"] == 2,
        "GBS": policy["train_global_batch_size"] == 16,
        "full local train MBS": policy["train_micro_batch_size"] == 16,
        "full local logprob batch": policy["logprob_batch_size"] == 16,
        "mixed train bank": policy["generation_batch_size"] == 16
        and config["data"]["train"]["data_path"].endswith("mixed_8.jsonl"),
        "mixed validation": grpo["val_batch_size"] == 16
        and config["data"]["validation"]["data_path"].endswith("mixed_8x12.jsonl"),
        "no inherited MCore generation": generation["mcore_generation_config"] is None,
        "MP executor": vllm_kwargs["distributed_executor_backend"] == "mp",
        "async scheduling": vllm_kwargs["async_scheduling"] is True,
        "exact capture": 16 in vllm_kwargs["compilation_config"]["cudagraph_capture_sizes"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"mixed vLLM smoke config drifted: {failed}")
