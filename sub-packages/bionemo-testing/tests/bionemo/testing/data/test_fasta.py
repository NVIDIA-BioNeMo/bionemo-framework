# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import pytest

from bionemo.noodles.nvfaidx import NvFaidx
from bionemo.testing.data.fasta import ALU_SEQUENCE, create_fasta_file


@pytest.mark.parametrize("target_sequence_length, num_sequences", [(123, 3), (1234, 2), (12345, 1)])
def test_created_fasta_file_has_expected_length(
    tmp_path: Path,
    target_sequence_length: int,
    num_sequences: int,
) -> None:
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(fasta_file_path, num_sequences, target_sequence_length, repeating_dna_pattern=ALU_SEQUENCE)
    assert fasta_file_path.stat().st_size > 0
    idx = NvFaidx(fasta_file_path)
    assert len(idx) == num_sequences
    n_out = 0
    for i, (seq_name, sequence) in enumerate(sorted(idx.items())):
        assert seq_name == f"contig_{i}"
        assert len(sequence) == target_sequence_length
        if i == 0:
            assert ALU_SEQUENCE[:target_sequence_length] in sequence
        n_out += 1
    assert n_out == num_sequences
