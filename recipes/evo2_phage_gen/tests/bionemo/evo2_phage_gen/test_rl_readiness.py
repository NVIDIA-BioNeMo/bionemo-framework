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

"""Tests for ``bionemo.evo2_phage_gen.rl_readiness``."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from bionemo.evo2_phage_gen import nemo_rl_env, rl_readiness
from bionemo.evo2_phage_gen.rl_readiness import check_rl_readiness


def _write_minimal_config(
    tmp_path: Path, *, include_adapter: bool = False, colocated_generation: bool | None = None
) -> Path:
    """Create a minimal GRPO config and all required referenced paths."""
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text("{}\n")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("1\n")
    iter_dir = checkpoint / "iter_0000001"
    iter_dir.mkdir()
    (iter_dir / "run_config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "_target_": "bionemo.evo2_phage_gen.rl_readiness.RLReadinessCheck",
                }
            }
        )
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    prompt_data = tmp_path / "phage_prompts.jsonl"
    prompt_data.write_text('{"messages": []}\n')

    config = {
        "defaults": str(defaults),
        "checkpointing": {"pretrained_checkpoint": {"path": str(checkpoint)}},
        "policy": {
            "model_name": "bionemo/evo2_7b_base",
            "tokenizer": {"name": str(tokenizer)},
            "megatron_cfg": {
                "enabled": True,
                "target_allowlist_prefixes": ["bionemo.evo2.", "bionemo.common."],
            },
            "generation": {"backend": "megatron"},
        },
        "data": {
            "train": {
                "dataset_name": "bionemo.evo2_phage_gen.nemo_rl_processors.PhageOpenAIFormatDataset",
                "data_path": str(prompt_data),
                "env_name": "phage_qc",
            }
        },
        "env": {"phage_qc": {}},
        "cluster": {"gpus_per_node": 2},
    }
    if colocated_generation is not None:
        config["policy"]["generation"]["colocated"] = {"enabled": colocated_generation}
    if include_adapter:
        config["policy"]["evo2_model_provider_import_path"] = "bionemo.evo2.models.evo2_provider:Hyena7bModelProvider"

    config_path = tmp_path / "grpo.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _write_control_config(tmp_path: Path) -> Path:
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    config["env"]["phage_qc"] = {
        "reward_output_mode": "gdpo",
        "weight_valid_nt_chars": 1.0,
        "weight_tropism": 1.0,
        "weight_mmseqs_cluster_diversity": 1.0,
        "gdpo_objectives": [
            {"name": "valid_nt_chars", "columns": ["reward_valid_nt_chars"]},
            {"name": "tropism", "columns": ["reward_external_tropism"]},
            {"name": "diversity", "columns": ["reward_mmseqs_cluster_diversity"]},
            {
                "name": "safety_amr",
                "columns": ["reward_safety_amr"],
                "requires_safety_eligibility": False,
            },
            {
                "name": "safety_toxin",
                "columns": ["reward_safety_toxin"],
                "requires_safety_eligibility": False,
            },
            {
                "name": "safety_lysogeny",
                "columns": ["reward_safety_lysogeny"],
                "requires_safety_eligibility": False,
            },
        ],
        "sequence_safety": {
            "enabled": True,
            "host_domain": "BACTERIA",
            "host_evidence": {
                "source": "test control",
                "source_version": "NC_001422.1",
                "replication_host_domains": ["BACTERIA"],
                "confirmed": True,
                "metadata": {},
            },
            "asset_manifest_path": str(tmp_path / "asset-manifest.yaml"),
            "diamond_bin": str(tmp_path / "diamond"),
            "mmseqs_bin": str(tmp_path / "mmseqs"),
            "policy_path": str(tmp_path / "policy.yaml"),
            "work_dir": str(tmp_path / "safety"),
            "strict_lysis": False,
            "circular": True,
            "threads": 1,
            "timeout_seconds": 30.0,
        },
        "external_qc": {
            "enabled": True,
            "enable_protein_hit_count": False,
            "enable_tropism": True,
            "enable_synteny": False,
            "enable_average_protein_identity": False,
            "enable_required_genes": False,
        },
        "mmseqs_cluster_diversity": {"enabled": True},
    }
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _control_scores(sequence: str) -> pd.DataFrame:
    row: dict[str, object] = {
        "sequence": sequence,
        "reward": 0.8,
        "reward_valid_nt_chars": 1.0,
        "reward_external_tropism": 0.75,
        "reward_mmseqs_cluster_diversity": 1.0,
        "external_qc_tool_succeeded": 1.0,
        "external_qc_measurement_available": 1.0,
        "tropism_measurement_available": 1.0,
        "mmseqs_cluster_id": "group0:0",
        "mmseqs_cluster_size": 1,
        "mmseqs_cluster_valid_for_clustering": 1.0,
        "mmseqs_cluster_missing_from_output": 0.0,
        "safety_gate_state": "PASS",
        "safety_gate_pass": 1.0,
        "safety_environment_healthy": 1.0,
        "safety_gate_measurement_available": 1.0,
    }
    for safety_class in ("amr", "toxin", "lysogeny"):
        row.update(
            {
                f"reward_safety_{safety_class}": 1.0,
                f"safety_{safety_class}_state": "PASS",
                f"safety_{safety_class}_required": 1.0,
                f"safety_{safety_class}_finding_count": 0,
                f"safety_{safety_class}_measurement_available": 1.0,
                f"safety_{safety_class}_execution_status": "COMPLETED_AND_PARSED",
            }
        )
    return pd.DataFrame([row])


def test_environment_control_runs_exact_step(tmp_path, monkeypatch):
    config_path = _write_control_config(tmp_path)
    control_fasta = tmp_path / "phix.fna"
    sequence = "ACGT" * 5
    control_fasta.write_text(f">phix\n{sequence}\n")

    def score(message_log_batch, **_kwargs):
        assert message_log_batch == [
            [
                {"role": "user", "content": "+~ACGTACGTACGTACGT"},
                {"role": "assistant", "content": "ACGT"},
            ]
        ]
        return _control_scores(sequence)

    monkeypatch.setattr(nemo_rl_env, "score_message_logs", score)

    result = rl_readiness.run_environment_control(config_path, control_fasta, tmp_path / "control")

    assert result["record_id"] == "phix"
    assert result["sequence_length"] == 20
    assert result["objectives"] == {
        "valid_nt_chars": 1.0,
        "tropism": 0.75,
        "diversity": 1.0,
        "safety_amr": 1.0,
        "safety_toxin": 1.0,
        "safety_lysogeny": 1.0,
    }
    assert result["support"] == {
        "external_qc": True,
        "tropism": True,
        "mmseqs_cluster_diversity": True,
        "safety_amr": True,
        "safety_toxin": True,
        "safety_lysogeny": True,
    }
    assert yaml.safe_load((tmp_path / "control" / "result.json").read_text()) == result


def test_environment_control_rejects_skipped_metric(tmp_path, monkeypatch):
    config_path = _write_control_config(tmp_path)
    control_fasta = tmp_path / "phix.fna"
    sequence = "ACGT" * 5
    control_fasta.write_text(f">phix\n{sequence}\n")
    scored = _control_scores(sequence)
    scored.loc[0, "tropism_measurement_available"] = 0.0
    monkeypatch.setattr(nemo_rl_env, "score_message_logs", lambda *_args, **_kwargs: scored)

    with pytest.raises(rl_readiness.RLEnvironmentControlError, match=r"tropism.*not measured"):
        rl_readiness.run_environment_control(config_path, control_fasta, tmp_path / "control")


def test_rl_readiness_reports_recipe_evo2_adapter_patch(tmp_path):
    """The readiness checker should see the recipe-local Evo2 NeMo-RL patch."""
    config_path = _write_minimal_config(tmp_path)

    checks = check_rl_readiness(config_path, expected_gpus=2)
    by_name = {check.name: check for check in checks}

    assert by_name["nemo_rl_install"].ok
    assert by_name["nemo_rl"].ok
    assert by_name["ray"].ok
    assert by_name["grpo_algorithm"].ok
    assert by_name["pretrained_checkpoint"].ok
    assert by_name["checkpoint_run_config"].ok
    assert by_name["checkpoint_bionemo_targets"].ok
    assert by_name["generation_backend"].ok
    assert by_name["megatron_generation_topology"].ok
    assert by_name["evo2_policy_adapter"].ok
    assert by_name["evo2_policy_adapter"].required
    assert "nemo-rl-evo2-mbridge-grpo.patch" in by_name["evo2_policy_adapter"].detail


def test_rl_readiness_allows_template_gap_to_be_optional(tmp_path):
    """The adapter check can be marked optional when only inspecting the scaffold."""
    config_path = _write_minimal_config(tmp_path)

    checks = check_rl_readiness(
        config_path,
        require_evo2_adapter=False,
        expected_gpus=2,
    )
    adapter_check = {check.name: check for check in checks}["evo2_policy_adapter"]

    assert not adapter_check.required


def test_rl_readiness_passes_when_adapter_path_is_configured(tmp_path):
    """A config with an explicit Evo2 provider adapter should pass local checks."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)

    checks = check_rl_readiness(config_path, expected_gpus=2)
    missing_required = [check for check in checks if check.required and not check.ok]

    assert missing_required == []


@pytest.mark.parametrize("split", ["train", "validation"])
def test_rl_readiness_rejects_path_derived_phage_task_namespace(tmp_path, split):
    """Readiness must catch the generic dataset before a long validation rollout."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    config["data"]["validation"] = dict(config["data"]["train"])
    config["data"][split]["dataset_name"] = "openai_format"
    config_path.write_text(yaml.safe_dump(config))

    checks = check_rl_readiness(config_path, expected_gpus=2)
    namespace_check = {check.name: check for check in checks}["phage_dataset_task_namespace"]

    assert not namespace_check.ok
    assert namespace_check.required
    assert "openai_format" in namespace_check.detail
    assert f"data.{split}.dataset_name" in namespace_check.detail


def test_rl_readiness_accepts_specific_megatron_bridge_iteration_directory(tmp_path):
    """A selected validation checkpoint can point directly at its iter directory."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    checkpoint_root = Path(config["checkpointing"]["pretrained_checkpoint"]["path"])
    config["checkpointing"]["pretrained_checkpoint"]["path"] = str(checkpoint_root / "iter_0000001")
    config_path.write_text(yaml.safe_dump(config))

    checks = check_rl_readiness(config_path, expected_gpus=2)
    missing_required = [check.name for check in checks if check.required and not check.ok]

    assert missing_required == []


def test_rl_readiness_checks_selected_checkpoint_override(tmp_path):
    """Readiness should inspect the checkpoint passed to the launch command."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    selected = Path(config["checkpointing"]["pretrained_checkpoint"]["path"]) / "iter_0000001"
    config["checkpointing"]["pretrained_checkpoint"]["path"] = str(tmp_path / "unused-template-checkpoint")
    config_path.write_text(yaml.safe_dump(config))

    checks = check_rl_readiness(config_path, expected_gpus=2, checkpoint_override=selected)
    missing_required = [check.name for check in checks if check.required and not check.ok]

    assert missing_required == []


def test_rl_readiness_checks_prompt_data_override(tmp_path):
    """Readiness should inspect the generated prompt bank passed to the launch command."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    config["data"]["train"]["data_path"] = str(tmp_path / "unused-template-prompts.jsonl")
    config_path.write_text(yaml.safe_dump(config))
    generated_prompts = tmp_path / "result" / "rl" / "train.jsonl"
    generated_prompts.parent.mkdir(parents=True)
    generated_prompts.write_text('{"messages": []}\n')

    checks = check_rl_readiness(
        config_path,
        expected_gpus=2,
        prompt_data_override=generated_prompts,
    )
    prompt_check = {check.name: check for check in checks}["prompt_data"]

    assert prompt_check.ok
    assert prompt_check.detail == str(generated_prompts)


def test_rl_readiness_resolves_inherited_config_values(tmp_path):
    """Readiness should validate the same inherited config that NeMo-RL launches."""
    base_config = _write_minimal_config(tmp_path, include_adapter=True)
    overlay_config = tmp_path / "overlay.yaml"
    overlay_config.write_text(
        yaml.safe_dump(
            {
                "defaults": base_config.name,
                "policy": {"generation": {"temperature": 1.0}},
            }
        )
    )

    checks = check_rl_readiness(overlay_config, expected_gpus=2)
    missing_required = [check.name for check in checks if check.required and not check.ok]

    assert missing_required == []


def test_rl_readiness_rejects_non_colocated_megatron_topology(tmp_path):
    """The two-GPU recipe scaffold targets colocated Megatron GRPO."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True, colocated_generation=False)

    checks = check_rl_readiness(config_path, expected_gpus=2)
    topology_check = {check.name: check for check in checks}["megatron_generation_topology"]

    assert not topology_check.ok
    assert topology_check.required


def test_rl_readiness_rejects_named_save_format_for_megatron_policy(tmp_path):
    """Native Megatron workers must not request an Automodel/DTensor export format."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    config["checkpointing"]["model_save_format"] = "safetensors"
    config["policy"]["dtensor_cfg"] = {"_v2": True, "enabled": False}
    config_path.write_text(yaml.safe_dump(config))

    checks = check_rl_readiness(config_path, expected_gpus=2)
    checkpoint_check = {check.name: check for check in checks}["checkpoint_save_backend_contract"]

    assert not checkpoint_check.ok
    assert checkpoint_check.required
    assert "model_save_format=None" in checkpoint_check.detail


def test_rl_readiness_accepts_native_megatron_checkpoint_contract(tmp_path):
    """Megatron training and rollout share native distributed MBridge checkpoints."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    config["checkpointing"]["model_save_format"] = None
    config["checkpointing"]["save_consolidated"] = False
    config["policy"]["dtensor_cfg"] = {"enabled": False}
    config_path.write_text(yaml.safe_dump(config))

    checks = check_rl_readiness(config_path, expected_gpus=2)
    checkpoint_check = {check.name: check for check in checks}["checkpoint_save_backend_contract"]

    assert checkpoint_check.ok
    assert "native Megatron-Bridge" in checkpoint_check.detail


def test_rl_readiness_rejects_dtensor_worker_for_megatron_checkpoint_contract(tmp_path):
    """The Evo2 Megatron path must not silently select a DTensor policy worker."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    config["checkpointing"]["model_save_format"] = None
    config["policy"]["dtensor_cfg"] = {"enabled": True}
    config_path.write_text(yaml.safe_dump(config))

    checks = check_rl_readiness(config_path, expected_gpus=2)
    checkpoint_check = {check.name: check for check in checks}["checkpoint_save_backend_contract"]

    assert not checkpoint_check.ok
    assert "dtensor_cfg.enabled=False" in checkpoint_check.detail


def test_module_check_reports_find_spec_errors(monkeypatch):
    """Import-discovery failures should become diagnostic results."""
    monkeypatch.setattr(
        rl_readiness.importlib.util,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(RuntimeError("bad spec")),
    )

    check = rl_readiness._module_check("ray", "ray", required=True)

    assert not check.ok
    assert "bad spec" in check.detail


def test_cuda_device_count_reports_import_discovery_errors(monkeypatch):
    """CUDA discovery should return unknown instead of aborting readiness."""
    monkeypatch.setattr(
        rl_readiness.importlib.util,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(RuntimeError("bad spec")),
    )

    assert rl_readiness._cuda_device_count() is None


def test_rl_readiness_falls_back_when_latest_iteration_tracker_is_invalid(tmp_path):
    """A bad tracker should not hide a valid iteration run configuration."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    checkpoint_root = Path(config["checkpointing"]["pretrained_checkpoint"]["path"])
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("not-an-integer\n")

    checks = check_rl_readiness(config_path, expected_gpus=2)
    by_name = {check.name: check for check in checks}

    assert by_name["checkpoint_run_config"].ok
    assert by_name["checkpoint_bionemo_targets"].ok


def test_rl_readiness_reports_invalid_checkpoint_yaml(tmp_path):
    """Malformed checkpoint metadata should produce a failed check."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    checkpoint_root = Path(config["checkpointing"]["pretrained_checkpoint"]["path"])
    (checkpoint_root / "iter_0000001" / "run_config.yaml").write_text(": invalid\n")

    checks = check_rl_readiness(config_path, expected_gpus=2)
    by_name = {check.name: check for check in checks}

    assert not by_name["checkpoint_bionemo_targets"].ok
    assert "could not read" in by_name["checkpoint_bionemo_targets"].detail


def test_rl_readiness_reports_invalid_gpu_count(tmp_path):
    """A non-integer GPU count should become a failed CUDA diagnostic."""
    config_path = _write_minimal_config(tmp_path, include_adapter=True)
    config = yaml.safe_load(config_path.read_text())
    config["cluster"]["gpus_per_node"] = "many"
    config_path.write_text(yaml.safe_dump(config))

    checks = check_rl_readiness(config_path, expected_gpus=2)
    cuda_check = {check.name: check for check in checks}["cuda_gpus"]

    assert not cuda_check.ok
    assert "gpus_per_node" in cuda_check.detail


def test_gpu_override(tmp_path):
    config_path = _write_minimal_config(tmp_path, include_adapter=True)

    checks = check_rl_readiness(config_path, expected_gpus=4, gpus_per_node=4)
    cuda_check = {check.name: check for check in checks}["cuda_gpus"]

    assert cuda_check.ok
    assert cuda_check.detail == "available=4, required=4"
