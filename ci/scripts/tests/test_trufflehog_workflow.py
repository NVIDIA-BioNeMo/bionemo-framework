#!/usr/bin/env python3

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

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FALSE_POSITIVE_PATH = Path("recipes/evo2_phage_gen/tests/bionemo/evo2_phage_gen/test_sequence_safety.py")
LOB_TOKEN = re.compile(r"\b(?:live|test)_[A-Za-z0-9_]{35}\b")


class TruffleHogWorkflowTests(unittest.TestCase):
    def test_lob_shaped_test_names_are_absent_from_current_sources(self) -> None:
        paths = (
            FALSE_POSITIVE_PATH,
            Path("recipes/evo2_phage_gen/skills/bionemo-phage-design/scripts/tests/test_run_skill_evals.py"),
        )

        for path in paths:
            text = (REPO_ROOT / path).read_text(encoding="utf-8")
            self.assertIsNone(LOB_TOKEN.search(text), path)

    def test_history_exclusion_is_exact_and_current_file_remains_scanned(self) -> None:
        exclusion_path = REPO_ROOT / ".github/trufflehog-exclude-paths.txt"
        workflow = (REPO_ROOT / ".github/workflows/trufflehog.yml").read_text(encoding="utf-8")

        self.assertEqual(
            [rf"^{re.escape(FALSE_POSITIVE_PATH.as_posix())}$"],
            exclusion_path.read_text(encoding="utf-8").splitlines(),
        )
        self.assertIn("--exclude-paths=.github/trufflehog-exclude-paths.txt", workflow)
        self.assertIn(f"filesystem {FALSE_POSITIVE_PATH.as_posix()}", workflow)
        self.assertIn("id: trufflehog_current_file", workflow)
        self.assertIn("steps.trufflehog_current_file.outcome == 'failure'", workflow)

    def test_action_and_image_share_an_updateable_pinned_release(self) -> None:
        workflow = (REPO_ROOT / ".github/workflows/trufflehog.yml").read_text(encoding="utf-8")

        self.assertNotIn("trufflesecurity/trufflehog@main", workflow)
        self.assertRegex(workflow, r"uses: trufflesecurity/trufflehog@[0-9a-f]{40}\n")
        self.assertRegex(workflow, r'TRUFFLEHOG_VERSION: "\d+\.\d+\.\d+"')
        self.assertIn('"${TRUFFLEHOG_IMAGE}:${TRUFFLEHOG_VERSION}"', workflow)


if __name__ == "__main__":
    unittest.main()
