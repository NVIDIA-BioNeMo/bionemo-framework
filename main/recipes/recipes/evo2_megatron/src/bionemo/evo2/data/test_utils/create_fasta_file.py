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


ALU_SEQUENCE: str = (
    "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACGAGGTC"
    "aggagatcgagaccatcctggctaacacggtgaaaccccgtctctactaaaaatacaaaaaattagccgggc"
    "GTGGTGGCGCGCGCCTGTAATCCCAGCTACTCGGGAGGCTGAGGCAGGAGAATGGCGTGAACCCGGGAGGCG"
    "GAGCTTGCAGTGAGCCGAGATCGCGCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA"
)


def create_fasta_file(
    fasta_file_path: Path,
    num_sequences: int,
    sequence_length: int | None = None,
    sequence_lengths: list[int] | None = None,
    repeating_dna_pattern: str = ALU_SEQUENCE,
    max_line_length: int = 80,
) -> Path:
    """Creates a fasta file with the given number of sequences, sequence length, and repeating dna pattern. Each contig uses a shifted version of the repeating pattern."""
    assert sequence_length is not None or sequence_lengths is not None
    with open(fasta_file_path, "w") as f:
        if sequence_lengths is not None:
            assert len(sequence_lengths) == num_sequences
        else:
            assert sequence_length is not None
            sequence_lengths: list[int] = [sequence_length] * num_sequences
        for i in range(num_sequences):
            # get the repeating pattern shifted by i for this contig
            repeat_pattern_for_contig = repeating_dna_pattern[i:] + repeating_dna_pattern[:i]
            # repeat the pattern enough times to reach the desired sequence length
            if sequence_lengths[i] <= len(repeat_pattern_for_contig):
                contig_output = repeat_pattern_for_contig[: sequence_lengths[i]]
            else:
                # Calculate how many complete repeats we need
                num_repeats = sequence_lengths[i] // len(repeat_pattern_for_contig)
                remainder = sequence_lengths[i] % len(repeat_pattern_for_contig)
                contig_output = repeat_pattern_for_contig * num_repeats + repeat_pattern_for_contig[:remainder]
            # verify the length of the contig is as expected
            assert len(contig_output) == sequence_lengths[i]
            # Fold the contig output into lines of max_line_length
            contig_output = "\n".join(
                contig_output[i : i + max_line_length] for i in range(0, sequence_lengths[i], max_line_length)
            )
            # write to the fasta file with the actual contig_output, not the repeating pattern
            f.write(f">contig_{i}\n{contig_output}\n")
    return fasta_file_path
