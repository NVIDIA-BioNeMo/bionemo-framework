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


from typing import Any, Dict

import fiddle as fdl


def config_to_dict(cfg: Any) -> Dict[str, Any]:
    """Recursively converts a fdl.Config or a dict of fdl.Configs to a dictionary."""
    if isinstance(cfg, dict):
        return {k: config_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, fdl.Config):
        return {k: config_to_dict(getattr(cfg, k)) for k in getattr(cfg, "__arguments__", {})}
    if isinstance(cfg, fdl.Partial):
        return {k: config_to_dict(getattr(cfg, k)) for k in getattr(cfg, "__arguments__", {})}
    return cfg
