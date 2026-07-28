# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from __future__ import annotations

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
