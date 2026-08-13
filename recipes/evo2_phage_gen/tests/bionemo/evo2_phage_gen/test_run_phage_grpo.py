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

import logging
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

from bionemo.evo2_phage_gen import run_phage_grpo


class _FakeRay:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def init(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


def test_init_ray_disables_optional_dashboard_and_restores_ray_init() -> None:
    ray = _FakeRay()
    original_init = ray.init

    def upstream_init_ray() -> None:
        ray.init(include_dashboard=True, resources={"test": 1})

    run_phage_grpo._init_ray(upstream_init_ray, ray, include_dashboard=False, num_cpus=32)

    assert ray.calls == [{"include_dashboard": False, "resources": {"test": 1}, "num_cpus": 32}]
    assert ray.init == original_init


def test_init_ray_preserves_dashboard_when_explicitly_enabled() -> None:
    ray = _FakeRay()

    def upstream_init_ray() -> None:
        ray.init(include_dashboard=True)

    run_phage_grpo._init_ray(upstream_init_ray, ray, include_dashboard=True)

    assert ray.calls == [{"include_dashboard": True}]


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
