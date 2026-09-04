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

import importlib
import os
import socket
import sys
from pathlib import Path

import pytest
import transformer_engine.pytorch


sys.path.append(Path(__file__).parent.parent.as_posix())
sys.path.append(Path(__file__).parent.as_posix())


pytest_plugins = ["tests.common.fixtures"]


# Fix Triton UTF-8 decoding issue by setting CUDA library path
if not os.environ.get("TRITON_LIBCUDA_PATH"):
    os.environ["TRITON_LIBCUDA_PATH"] = "/usr/local/cuda/lib64"


@pytest.fixture
def unused_tcp_port():
    """Find and return an unused TCP port for torchrun rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture(autouse=True)
def use_te_debug(monkeypatch):
    monkeypatch.setenv("NVTE_DEBUG", "1")
    monkeypatch.setenv("NVTE_DEBUG_LEVEL", "2")
    orig = sys.modules.get("transformer_engine.pytorch")
    importlib.reload(transformer_engine.pytorch)
    yield
    if orig is not None:
        sys.modules["transformer_engine.pytorch"] = orig
    else:
        importlib.reload(transformer_engine.pytorch)
