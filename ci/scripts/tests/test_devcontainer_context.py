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

"""Tests for the local development-container build context."""

import json
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
DEVCONTAINER_ROOT = REPO_ROOT / ".devcontainer"


def test_devcontainer_build_uses_minimal_local_context():
    """The local devcontainer build must not send the repository as context."""
    config = json.loads((DEVCONTAINER_ROOT / "devcontainer.json").read_text())

    assert config["build"] == {"context": ".", "dockerfile": "Dockerfile"}

    ignore_path = DEVCONTAINER_ROOT / "Dockerfile.dockerignore"
    assert ignore_path.is_file()
    assert {"**", "!Dockerfile", "!requirements.txt"} <= set(ignore_path.read_text().splitlines())

    dockerfile = (DEVCONTAINER_ROOT / "Dockerfile").read_text()
    assert "COPY . ." not in dockerfile

    start_script = (DEVCONTAINER_ROOT / "start.sh").read_text()
    assert 'docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"' in start_script
