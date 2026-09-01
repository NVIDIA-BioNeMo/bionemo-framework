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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Run Evo2 prediction with the slow, test-only tensor-parallel oracle.

This is deliberately a test launcher rather than a production CLI option. It replaces
TE linears with full-logical-tensor FP32 reference calculations so topology tests can
validate TP sharding independently of BF16/FP8 kernel accumulation order.
"""

import importlib.util
from pathlib import Path

import bionemo.evo2.models.evo2_provider as evo2_provider
from bionemo.evo2.run import predict as predict_module


_get_mixed_precision_config = predict_module.get_mixed_precision_config


def _get_reference_mixed_precision_config(name):
    """Keep high-precision weights gatherable by the test oracle before emulating FP8."""
    config = _get_mixed_precision_config(name)
    config.fp8_param_gather = False
    return config


def _load_tp_reference_module():
    reference_path = Path(__file__).parents[1] / "models" / "megatron" / "hyena" / "tp_reference.py"
    spec = importlib.util.spec_from_file_location("_evo2_test_tp_reference", reference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load TP reference implementation from {reference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    tp_reference = _load_tp_reference_module()
    evo2_provider.get_hyena_stack_spec = tp_reference.get_tp_reference_hyena_stack_spec
    predict_module.get_mixed_precision_config = _get_reference_mixed_precision_config
    predict_module.main()
