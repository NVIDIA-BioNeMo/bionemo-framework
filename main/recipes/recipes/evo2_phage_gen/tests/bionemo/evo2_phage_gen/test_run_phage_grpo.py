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

from __future__ import annotations

import ast
import inspect
import logging
import sys
import textwrap
from types import SimpleNamespace

from omegaconf import OmegaConf

from bionemo.evo2_phage_gen import run_phage_grpo


def test_init_ray_passes_dashboard_and_cpu_options_to_upstream() -> None:
    calls = []

    def upstream_init_ray(**kwargs) -> None:
        calls.append(kwargs)

    run_phage_grpo._init_ray(upstream_init_ray, include_dashboard=False, num_cpus=32)

    assert calls == [{"include_dashboard": False, "num_cpus": 32}]


def test_init_ray_omits_unspecified_cpu_limit() -> None:
    calls = []

    def upstream_init_ray(**kwargs) -> None:
        calls.append(kwargs)

    run_phage_grpo._init_ray(upstream_init_ray, include_dashboard=True)

    assert calls == [{"include_dashboard": True}]


def test_ensure_prompt_data_files_logs_materialized_paths(tmp_path, monkeypatch, caplog, capsys) -> None:
    train_path = tmp_path / "phage_prompts_paper_useful_rl.jsonl"
    validation_path = tmp_path / "phage_prompts_paper_useful_rl_validation_prompt10_96.jsonl"
    generation_module = SimpleNamespace(
        ensure_paper_useful_rl_prompt_files=lambda _data_dir: {
            "train": train_path,
            "validation": validation_path,
        }
    )
    monkeypatch.setitem(sys.modules, "bionemo.evo2_phage_gen.generation", generation_module)
    config = OmegaConf.create(
        {
            "data": {
                "train": {"data_path": str(train_path)},
                "validation": {"data_path": str(validation_path)},
            }
        }
    )

    with caplog.at_level(logging.INFO, logger=run_phage_grpo.__name__):
        run_phage_grpo._ensure_prompt_data_files(config)

    assert "Materialized missing paper-useful RL prompt data:" in caplog.messages
    assert f"  {train_path}" in caplog.messages
    assert f"  {validation_path}" in caplog.messages
    assert capsys.readouterr().out == ""


def test_sync_trainer_receives_experiment_logger() -> None:
    """Keep the module logger out of the NeMo-RL trainer call."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(run_phage_grpo.main)))
    trainer_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "trainer"
    )

    assert isinstance(trainer_call.args[8], ast.Name)
    assert trainer_call.args[8].id == "experiment_logger"
