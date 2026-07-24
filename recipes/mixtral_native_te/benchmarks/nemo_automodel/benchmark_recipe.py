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

"""Compatibility entry point for AutoModel's timed LLM benchmark recipe."""

from mxfp8 import register_mxfp8_experts

from nemo_automodel.recipes.llm.benchmark import BenchmarkingRecipeForNextTokenPrediction


register_mxfp8_experts()


class MixtralBenchmarkRecipe(BenchmarkingRecipeForNextTokenPrediction):
    """Run the timed benchmark through the unified CLI's training-loop hook."""

    def run_train_validation_loop(self):
        """Delegate the hook used by ``automodel`` to the benchmark loop."""
        return self.run_benchmark()
