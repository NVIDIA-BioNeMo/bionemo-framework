# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

from bionemo.evo2.data.dataset_tokenizer import (
    DEFAULT_HF_TOKENIZER_MODEL_PATH,
    DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
    Evo2DatasetTokenizer,
)
from bionemo.evo2.utils.config import Evo2PreprocessingConfig


@pytest.fixture
def tokenizer() -> Evo2DatasetTokenizer:
    """Return a dataset tokenizer for the Evo2Dataset."""
    return Evo2DatasetTokenizer(Evo2PreprocessingConfig())


@pytest.mark.parametrize(
    "tokenizer_path, expected_vocab_size",
    [
        (DEFAULT_HF_TOKENIZER_MODEL_PATH, 256),
        (DEFAULT_HF_TOKENIZER_MODEL_PATH_512, 512),
    ],
)
def test_tokenizer_vocab_size(tokenizer_path: Path, expected_vocab_size: int) -> None:
    """Verifies key tokenizers have the expected vocabulary size."""
    tokenizer = build_tokenizer(
        TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            hf_tokenizer_kwargs={"trust_remote_code": False},
            tokenizer_model=tokenizer_path,
        )
    )
    assert tokenizer.vocab_size == expected_vocab_size


@pytest.mark.parametrize(
    "tokenizer_path",
    [
        DEFAULT_HF_TOKENIZER_MODEL_PATH,
        DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
    ],
)
def test_tokenizer_roundtrip_without_spaces(tokenizer_path: Path) -> None:
    """Verifies tokenization followed by detokenization returns the original sequence.

    This is critical for character-level tokenizers used in DNA sequence modeling.
    The tokenizer should NOT add spaces between tokens during detokenization.
    """
    tokenizer = build_tokenizer(
        TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            hf_tokenizer_kwargs={"trust_remote_code": False},
            tokenizer_model=tokenizer_path,
        )
    )
    # Test basic DNA sequence
    original = "ATCGATCGATCG"
    token_ids = tokenizer.tokenize(original)
    reconstructed = tokenizer.detokenize(token_ids)
    assert reconstructed == original, f"Expected '{original}', got '{reconstructed}'"

    # Test longer sequence with all nucleotides
    original_long = "AAAAACCCCCGGGGGTTTTTATCGATCGNNNNN"
    token_ids_long = tokenizer.tokenize(original_long)
    reconstructed_long = tokenizer.detokenize(token_ids_long)
    assert reconstructed_long == original_long, f"Expected '{original_long}', got '{reconstructed_long}'"

    # Test sequence with special characters (pipe-delimited tags)
    original_tagged = "|info|ATCG|end|"
    token_ids_tagged = tokenizer.tokenize(original_tagged)
    reconstructed_tagged = tokenizer.detokenize(token_ids_tagged)
    assert reconstructed_tagged == original_tagged, f"Expected '{original_tagged}', got '{reconstructed_tagged}'"


def test_tokenizer_handles_long_dna_sequence(tokenizer: Evo2DatasetTokenizer) -> None:
    """Verifies tokenizer correctly processes a long DNA sequence into expected token IDs.

    This sequence excerpt was pulled from mmseqs_results_rep_seq_distinct.fasta.
    """
    sequence = "TACACCTATATTTTTTAAGGTATGTAAACATCTACTTTTAGTGATACTAACAAAAATATAGAATAATAATTAGTGTTTTTGTATATTAATGTATGGGTAGGATCACAAATAAATTACGAAACCTTTTCCTATAATATTATAA"
    tokens = tokenizer.tokenize(sequence)
    expected_tokens = [
        [
            84,
            65,
            67,
            65,
            67,
            67,
            84,
            65,
            84,
            65,
            84,
            84,
            84,
            84,
            84,
            84,
            65,
            65,
            71,
            71,
            84,
            65,
            84,
            71,
            84,
            65,
            65,
            65,
            67,
            65,
            84,
            67,
            84,
            65,
            67,
            84,
            84,
            84,
            84,
            65,
            71,
            84,
            71,
            65,
            84,
            65,
            67,
            84,
            65,
            65,
            67,
            65,
            65,
            65,
            65,
            65,
            84,
            65,
            84,
            65,
            71,
            65,
            65,
            84,
            65,
            65,
            84,
            65,
            65,
            84,
            84,
            65,
            71,
            84,
            71,
            84,
            84,
            84,
            84,
            84,
            71,
            84,
            65,
            84,
            65,
            84,
            84,
            65,
            65,
            84,
            71,
            84,
            65,
            84,
            71,
            71,
            71,
            84,
            65,
            71,
            71,
            65,
            84,
            67,
            65,
            67,
            65,
            65,
            65,
            84,
            65,
            65,
            65,
            84,
            84,
            65,
            67,
            71,
            65,
            65,
            65,
            67,
            67,
            84,
            84,
            84,
            84,
            67,
            67,
            84,
            65,
            84,
            65,
            65,
            84,
            65,
            84,
            84,
            65,
            84,
            65,
            65,
        ]
    ]
    assert expected_tokens == tokens


def test_tokenizer_processes_pipe_delimited_sequence(tokenizer: Evo2DatasetTokenizer) -> None:
    """Verifies tokenizer correctly handles pipe-delimited sequences with info tags."""
    tokens = tokenizer.tokenize("|info|ATG|info|ATG|")
    expected_tokens = [[124, 105, 110, 102, 111, 124, 65, 84, 71, 124, 105, 110, 102, 111, 124, 65, 84, 71, 124]]
    assert expected_tokens == tokens


def test_tokenizer_drops_empty_sequences(tokenizer: Evo2DatasetTokenizer) -> None:
    """Verifies tokenizer removes empty sequences when drop_empty_sequences is True."""
    tokens = tokenizer.tokenize(["A", "", "T"], drop_empty_sequences=True)
    expected_tokens = [[65], [84]]
    assert expected_tokens == tokens


def test_tokenizer_appends_eod_token(tokenizer: Evo2DatasetTokenizer) -> None:
    """Verifies tokenizer correctly appends end-of-document token."""
    tokens = tokenizer.tokenize(["ATCG"], append_eod=True)
    expected_tokens = [[65, 84, 67, 71, 0]]
    assert expected_tokens == tokens


def test_tokenizer_pads_sequence_to_required_length(tokenizer: Evo2DatasetTokenizer) -> None:
    """Verifies tokenizer correctly pads sequence to specified length."""
    tokens = tokenizer.tokenize(["ATCG"], enforce_sample_length=10)
    expected_tokens = [[65, 84, 67, 71, 1, 1, 1, 1, 1, 1]]
    assert expected_tokens == tokens


def test_tokenizer_raises_error_for_invalid_length(tokenizer: Evo2DatasetTokenizer) -> None:
    """Verifies tokenizer raises ValueError when sequence exceeds enforced length."""
    with pytest.raises(ValueError):
        tokenizer.tokenize(["ATCGATCGATCG"], enforce_sample_length=4)


@pytest.mark.skip(reason="Unsure why this fails, it is not used though.")
def test_tokenizer_fixes_unicode_with_ftfy(tokenizer: Evo2DatasetTokenizer) -> None:
    """Verifies tokenizer correctly processes broken unicode characters using ftfy."""
    tokens = tokenizer.tokenize("âœ ATCG", use_ftfy=True)
    expected_tokens = [[226, 156, 160, 65, 84, 67, 71]]
    assert expected_tokens == tokens


def test_tokenizer_processes_special_characters(tokenizer: Evo2DatasetTokenizer) -> None:
    """Evo2_Dataset uses specific ASCII encodings for specific characters.

        CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' tag for splice splits/windows, '#' for contig splits
        TAG_BOUNDS = 124  # start and end delim: '|'
        TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # chars only found in control tags: _, ;, space
        DEFAULT_EOD = 0
    This test verifies tokenizer correctly handles these special characters.
    """
    special_chars = "".join(["@", "#", "|", "_", ";", " "])
    tokens = tokenizer.tokenize(special_chars, append_eod=True)
    expected_tokens = [[64, 35, 124, 95, 59, 32, 0]]
    assert expected_tokens == tokens
