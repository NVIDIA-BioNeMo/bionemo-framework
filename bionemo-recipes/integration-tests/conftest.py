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

"""Integration test fixtures for Claude Code + bionemo-recipes plugin."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


PLUGIN_DIR = Path(__file__).parent.parent / "claude-plugin"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class ClaudeRunner:
    """Runs Claude Code in --print mode with the bionemo-recipes plugin."""

    def run(self, prompt: str, cwd: str, max_budget: float = 5.0, timeout: int = 600) -> dict:
        """Execute a Claude Code prompt and return the JSON result.

        Args:
            prompt: The prompt to send to Claude.
            cwd: Working directory for Claude.
            max_budget: Maximum API budget in USD.
            timeout: Timeout in seconds.

        Returns:
            Parsed JSON output from Claude.
        """
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
            "--add-dir",
            str(PLUGIN_DIR),
            f"--max-budget-usd={max_budget}",
        ]
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Claude exited with code {result.returncode}.\n"
                f"stdout: {result.stdout[:2000]}\n"
                f"stderr: {result.stderr[:2000]}"
            )
        return json.loads(result.stdout)


@pytest.fixture(scope="session")
def claude_runner():
    """Provide a ClaudeRunner instance for the test session."""
    return ClaudeRunner()


@pytest.fixture
def bert_fixture_dir(tmp_path):
    """Create a temporary copy of the barebones-bert fixture."""
    src = FIXTURES_DIR / "barebones-bert"
    dst = tmp_path / "barebones-bert"
    shutil.copytree(src, dst)
    return dst


@pytest.fixture
def llama_fixture_dir(tmp_path):
    """Create a temporary copy of the barebones-llama fixture."""
    src = FIXTURES_DIR / "barebones-llama"
    dst = tmp_path / "barebones-llama"
    shutil.copytree(src, dst)
    return dst


@pytest.fixture
def pre_te_ified_bert_dir(tmp_path, claude_runner):
    """Create a TE-ified BERT fixture (for FP8 tests that need a pre-converted model)."""
    src = FIXTURES_DIR / "barebones-bert"
    dst = tmp_path / "pre-te-bert"
    shutil.copytree(src, dst)

    claude_runner.run(
        "Convert this HuggingFace BERT model to use TransformerEngine. "
        "Create the TE model class, conversion utilities, and a basic test.",
        cwd=str(dst),
        max_budget=5.0,
    )
    return dst
