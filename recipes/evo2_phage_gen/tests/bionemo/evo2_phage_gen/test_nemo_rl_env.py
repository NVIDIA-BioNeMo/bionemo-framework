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

"""Tests for ``bionemo.evo2_phage_gen.nemo_rl_env`` helpers."""

from bionemo.evo2_phage_gen.nemo_rl_env import extract_assistant_sequence, score_message_logs


def test_extract_assistant_sequence_concatenates_assistant_messages():
    """Only assistant messages should contribute to generated DNA."""
    message_log = [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "ACGT"},
        {"role": "environment", "content": "ignored"},
        {"role": "assistant", "content": "TGCA"},
    ]

    assert extract_assistant_sequence(message_log) == "ACGTTGCA"


def test_score_message_logs_uses_phage_reward():
    """A passing assistant sequence should receive reward 1."""
    scored = score_message_logs([[{"role": "assistant", "content": "ACGT" * 1000}]])

    assert scored["reward"].tolist() == [1.0]
