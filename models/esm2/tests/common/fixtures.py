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

"""Shared test fixtures for BioNeMo models."""

import os
import socket

import pytest
from transformer_engine.common import recipe as recipe_module
from transformer_engine.pytorch import fp8
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends


@pytest.fixture
def unused_tcp_port() -> int:
    """Get an unused TCP port for distributed testing.

    Returns:
        An available TCP port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def use_te_debug():
    """Auto-use fixture to enable TransformerEngine debugging.

    This fixture automatically enables debug mode for TransformerEngine
    in all tests for better error messages.
    """
    import os

    os.environ["NVTE_DEBUG"] = "1"
    yield
    os.environ.pop("NVTE_DEBUG", None)


ALL_RECIPES = [
    recipe_module.DelayedScaling(),
    recipe_module.Float8CurrentScaling(),
    recipe_module.Float8BlockScaling(),
    recipe_module.MXFP8BlockScaling(),
    recipe_module.NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True),
]


def _check_recipe_support(recipe: recipe_module.Recipe):
    """Check if a recipe is supported and return (supported, reason)."""
    if isinstance(recipe, recipe_module.DelayedScaling):
        recipe_supported, reason = fp8.check_fp8_support()
    elif isinstance(recipe, recipe_module.Float8CurrentScaling):
        recipe_supported, reason = fp8.check_fp8_support()
    elif isinstance(recipe, recipe_module.Float8BlockScaling):
        recipe_supported, reason = fp8.check_fp8_block_scaling_support()
    elif isinstance(recipe, recipe_module.MXFP8BlockScaling):
        recipe_supported, reason = fp8.check_mxfp8_support()
    elif isinstance(recipe, recipe_module.NVFP4BlockScaling):
        recipe_supported, reason = fp8.check_nvfp4_support()
    else:
        recipe_supported = False
        reason = "Unsupported recipe"
    return recipe_supported, reason


def parametrize_recipes_with_support(recipes):
    """Generate pytest.param objects with skip marks for unsupported recipes."""
    parametrized_recipes = []
    for recipe in recipes:
        recipe_supported, reason = _check_recipe_support(recipe)
        parametrized_recipes.append(
            pytest.param(
                recipe,
                id=recipe.__class__.__name__,
                marks=pytest.mark.xfail(
                    condition=not recipe_supported,
                    reason=reason,
                ),
            )
        )
    return parametrized_recipes


@pytest.fixture(params=parametrize_recipes_with_support(ALL_RECIPES))
def fp8_recipe(request):
    """Fixture to parametrize the FP8 recipe."""
    return request.param


@pytest.fixture(params=["bshd", "thd"])
def input_format(request):
    """Fixture to parametrize the input format."""
    return request.param


@pytest.fixture(params=["flash_attn", "fused_attn"])
def te_attn_backend(request):
    """Fixture to parametrize the attention implementation."""
    if request.param == "flash_attn":
        os.environ["NVTE_FUSED_ATTN"] = "0"
        os.environ["NVTE_FLASH_ATTN"] = "1"
        _attention_backends["backend_selection_requires_update"] = True

    else:
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_FLASH_ATTN"] = "0"
        _attention_backends["backend_selection_requires_update"] = True

    yield request.param

    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_FLASH_ATTN", None)
    _attention_backends["backend_selection_requires_update"] = True
