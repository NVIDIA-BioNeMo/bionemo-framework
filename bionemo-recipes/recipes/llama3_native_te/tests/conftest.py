# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
from pathlib import Path
from unittest import mock

import pytest
import torch
from transformer_engine.pytorch import fp8 as te_fp8


sys.path.append(Path(__file__).parent.parent.as_posix())
sys.path.append(Path(__file__).parent.as_posix())
from distributed_config import DistributedConfig


@pytest.fixture
def recipe_path() -> Path:
    """Return the root directory of the recipe."""
    return Path(__file__).parent.parent


@pytest.fixture
def tokenizer_path(recipe_path):
    """Get the path to the nucleotide tokenizer."""
    return str(recipe_path / "tokenizers" / "nucleotide_fast_tokenizer")


@pytest.fixture(autouse=True)
def debug_api_cleanup():
    """Ensure nvdlfw_inspect does not stay initialized across tests."""
    yield
    try:
        import nvdlfw_inspect.api as debug_api

        debug_api.end_debug()
    except Exception:  # pragma: no cover - best-effort cleanup for optional dependency
        pass


def pytest_collection_modifyitems(items):
    """Run FP8 stats logging tests first to avoid late debug initialization."""
    stats_test_names = {
        "test_sanity_ddp_fp8_stats_logging",
        "test_sanity_fsdp2_fp8_stats_logging",
        "test_sanity_ddp_fp8_partial_layers_stats_logging",
        "test_sanity_fsdp2_fp8_partial_layers_stats_logging",
    }
    stats_tests = [item for item in items if item.name in stats_test_names]
    other_tests = [item for item in items if item.name not in stats_test_names]
    items[:] = stats_tests + other_tests


# ---------------------------------------------------------------------------
# FP8 recipe parametrization
# ---------------------------------------------------------------------------

# Each entry: (recipe_class_name, hydra_overrides, check_fn)
_FP8_RECIPE_CONFIGS = [
    (
        "DelayedScaling",
        ["fp8_config.fp8_recipe=transformer_engine.common.recipe.DelayedScaling"],
        te_fp8.check_fp8_support,
    ),
    (
        "Float8CurrentScaling",
        ["fp8_config.fp8_recipe=transformer_engine.common.recipe.Float8CurrentScaling"],
        te_fp8.check_fp8_support,
    ),
    (
        "Float8BlockScaling",
        ["fp8_config.fp8_recipe=transformer_engine.common.recipe.Float8BlockScaling"],
        te_fp8.check_fp8_block_scaling_support,
    ),
    (
        "MXFP8BlockScaling",
        ["fp8_config.fp8_recipe=transformer_engine.common.recipe.MXFP8BlockScaling"],
        te_fp8.check_mxfp8_support,
    ),
]


def _parametrize_fp8_recipes():
    """Generate pytest.param objects with xfail marks for unsupported FP8 recipes."""
    params = []
    for name, overrides, check_fn in _FP8_RECIPE_CONFIGS:
        supported, reason = check_fn()
        params.append(
            pytest.param(
                overrides,
                id=name,
                marks=pytest.mark.xfail(condition=not supported, reason=reason),
            )
        )
    return params


@pytest.fixture(params=_parametrize_fp8_recipes())
def fp_recipe(request):
    """Parametrized fixture providing FP8 recipe Hydra overrides for each supported TE recipe."""
    return request.param


@pytest.fixture(scope="session", autouse=True)
def device_mesh():
    """Create a re-usable torch process group for testing.
    This is a "auto-use", session-scope fixture so that a single torch process group is created and used in all tests.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # Mock these torch.distributed functions so that we re-use the same device mesh, and don't re-create or destroy the
    # global process group.
    with (
        mock.patch("torch.distributed.init_process_group", return_value=None),
        mock.patch("torch.distributed.destroy_process_group", return_value=None),
    ):
        yield

    # At the end of all tests, destroy the process group and clear the device mesh resources.
    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
