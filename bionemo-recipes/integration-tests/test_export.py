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

"""Integration tests: Claude creates an export script."""

from validators import ast_checks


class TestExport:
    """Test that Claude can create a HuggingFace Hub export script."""

    def test_claude_creates_export_script(self, claude_runner, pre_te_ified_bert_dir):
        """Claude creates an export script for a TE model."""
        claude_runner.run(
            "Create an export script that converts the HuggingFace model to TE format "
            "and saves it for HuggingFace Hub distribution. Include AUTO_MAP patching "
            "in config.json for trust_remote_code support.",
            cwd=str(pre_te_ified_bert_dir),
        )

        export_file = pre_te_ified_bert_dir / "export.py"
        assert export_file.exists(), "export.py not created"
        ast_checks.validate_python_file(export_file)

        code = export_file.read_text()
        assert "AUTO_MAP" in code or "auto_map" in code, "export.py does not reference AUTO_MAP"
