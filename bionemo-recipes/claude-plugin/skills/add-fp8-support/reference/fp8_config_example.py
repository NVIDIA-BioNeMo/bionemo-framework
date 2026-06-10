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

"""Reference: FP8 recipe setup in a training script.

Shows how to create and use FP8/FP4 recipes with TransformerEngine models.
"""

from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8BlockScaling,
    Float8CurrentScaling,
    Format,
    NVFP4BlockScaling,
)


def create_fp8_recipe(recipe_name: str = "DelayedScaling", **kwargs):
    """Create an FP8 recipe by name.

    Available recipes:
    - DelayedScaling: Classic FP8, scaling factors computed with delay
    - Float8CurrentScaling: Per-tensor scaling computed each step
    - Float8BlockScaling: Block-wise scaling (MXFP8)
    - NVFP4BlockScaling: 4-bit quantization
    """
    recipes = {
        "DelayedScaling": DelayedScaling,
        "Float8CurrentScaling": Float8CurrentScaling,
        "Float8BlockScaling": Float8BlockScaling,
        "NVFP4BlockScaling": NVFP4BlockScaling,
    }
    recipe_cls = recipes[recipe_name]

    # NOTE: Format.HYBRID uses E4M3 for forward, E5M2 for backward
    if "fp8_format" not in kwargs and recipe_name != "NVFP4BlockScaling":
        kwargs["fp8_format"] = Format.HYBRID
    if "fp4_format" not in kwargs and recipe_name == "NVFP4BlockScaling":
        kwargs["fp4_format"] = Format.E2M1

    return recipe_cls(**kwargs)


# Example usage in training script:
def setup_model_with_fp8(config, layer_precision):
    """Example of setting up a TE model with FP8 quantization."""
    config.layer_precision = layer_precision

    fp8_recipe = create_fp8_recipe("DelayedScaling")

    # NOTE: Pass recipe to model constructor, not as global state
    # model = NVModelForMaskedLM(config, fp8_recipe=fp8_recipe)

    return config, fp8_recipe
