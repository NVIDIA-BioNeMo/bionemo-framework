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

"""Recipe-local NeMo-RL GDPO launcher for Evo2 phage optimization."""

from __future__ import annotations

from bionemo.evo2_phage_gen.run_phage_grpo import main as run_phage_rl


def main() -> None:
    """Run GDPO with the recipe GDPO config by default."""
    run_phage_rl(default_config="configs/gdpo_phage_megatron.yaml", default_algorithm="gdpo")


if __name__ == "__main__":
    main()
