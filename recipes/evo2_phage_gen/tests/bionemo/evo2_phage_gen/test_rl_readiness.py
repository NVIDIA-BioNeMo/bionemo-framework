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

import yaml

from bionemo.evo2_phage_gen.rl_readiness import check_rl_readiness


def _write_minimal_nemo_rl_checkout(path: Path) -> None:
    """Create enough package structure for import-spec checks."""
    (path / "nemo_rl").mkdir(parents=True)
    (path / "nemo_rl" / "__init__.py").write_text("")
    (path / "nemo_rl" / "algorithms").mkdir()
    (path / "nemo_rl" / "algorithms" / "__init__.py").write_text("")
    (path / "nemo_rl" / "algorithms" / "grpo.py").write_text("def setup():\n    pass\n")
    (path / "ray.py").write_text("")


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
        "data": {"train": {"data_path": str(prompt_data), "env_name": "phage_qc"}},
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


def test_rl_readiness_reports_recipe_evo2_adapter_patch(tmp_path):
    """The readiness checker should see the recipe-local Evo2 NeMo-RL patch."""
    local_nemo_rl = tmp_path / "nemo-rl"
    _write_minimal_nemo_rl_checkout(local_nemo_rl)
    config_path = _write_minimal_config(tmp_path)

    checks = check_rl_readiness(config_path, local_nemo_rl=local_nemo_rl, expected_gpus=2)
    by_name = {check.name: check for check in checks}

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
    local_nemo_rl = tmp_path / "nemo-rl"
    _write_minimal_nemo_rl_checkout(local_nemo_rl)
    config_path = _write_minimal_config(tmp_path)

    checks = check_rl_readiness(
        config_path,
        local_nemo_rl=local_nemo_rl,
        require_evo2_adapter=False,
        expected_gpus=2,
    )
    adapter_check = {check.name: check for check in checks}["evo2_policy_adapter"]

    assert not adapter_check.required


def test_rl_readiness_passes_when_adapter_path_is_configured(tmp_path):
    """A config with an explicit Evo2 provider adapter should pass local checks."""
    local_nemo_rl = tmp_path / "nemo-rl"
    _write_minimal_nemo_rl_checkout(local_nemo_rl)
    config_path = _write_minimal_config(tmp_path, include_adapter=True)

    checks = check_rl_readiness(config_path, local_nemo_rl=local_nemo_rl, expected_gpus=2)
    missing_required = [check for check in checks if check.required and not check.ok]

    assert missing_required == []


def test_rl_readiness_rejects_non_colocated_megatron_topology(tmp_path):
    """The two-GPU recipe scaffold targets colocated Megatron GRPO."""
    local_nemo_rl = tmp_path / "nemo-rl"
    _write_minimal_nemo_rl_checkout(local_nemo_rl)
    config_path = _write_minimal_config(tmp_path, include_adapter=True, colocated_generation=False)

    checks = check_rl_readiness(config_path, local_nemo_rl=local_nemo_rl, expected_gpus=2)
    topology_check = {check.name: check for check in checks}["megatron_generation_topology"]

    assert not topology_check.ok
    assert topology_check.required
