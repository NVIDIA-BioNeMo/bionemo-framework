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

"""Script to create the HuggingFace PreTrainedTokenizerFast for nucleotide sequences.

This script creates a tokenizer that:
1. Maps each character to its ord() value (ASCII encoding)
2. Uses special tokens with NeMo convention (EOS=0, PAD=1, BOS=2, UNK=3)
3. Works with AutoTokenizer.from_pretrained()

Run this script to regenerate the tokenizer files if needed.
"""

import logging
import os

from tokenizers import Tokenizer, processors
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_nucleotide_tokenizer(
    eos_id: int = 0,
    pad_id: int = 1,
    bos_id: int = 2,
    unk_id: int = 3,
) -> PreTrainedTokenizerFast:
    """Create a PreTrainedTokenizerFast for nucleotide sequences.

    Uses special token IDs for causal language modeling:
    - BOS = 2 (beginning of sequence)
    - EOS = 0 (end of sequence)
    - PAD = 1 (padding)
    - UNK = 3 (unknown)

    Args:
        eos_id: End-of-sequence token ID (default: 0)
        pad_id: Padding token ID (default: 1)
        bos_id: Beginning-of-sequence token ID (default: 2)
        unk_id: Unknown token ID (default: 3)

    Returns:
        PreTrainedTokenizerFast ready to use and save
    """
    # Define special tokens
    special_tokens = {
        "<BOS>": bos_id,
        "<EOS>": eos_id,
        "<PAD>": pad_id,
        "<UNK>": unk_id,
    }

    # Build vocab: Map each ASCII character to its ord() value
    # IMPORTANT: Exclude reserved IDs for special tokens
    reserved_ids = set(special_tokens.values())
    vocab = {chr(i): i for i in range(256) if i not in reserved_ids}
    vocab = {**vocab, **special_tokens}

    # Create Rust tokenizer backend with WordLevel model
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<UNK>"))

    # Configure pre-tokenizer: Split into individual characters
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

    # Configure post-processor: Add BOS/EOS tokens automatically
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<BOS> $A <EOS>",
        pair="<BOS> $A <EOS> <BOS> $B <EOS>",
        special_tokens=[
            ("<BOS>", bos_id),
            ("<EOS>", eos_id),
        ],
    )

    # Wrap in HuggingFace PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<UNK>",
        pad_token="<PAD>",
        eos_token="<EOS>",
        bos_token="<BOS>",
    )

    return hf_tokenizer


def main():
    """Create and save the nucleotide tokenizer."""
    logger.info("Creating nucleotide tokenizer")

    # Create tokenizer with default settings (BOS=2, EOS=0, PAD=1, UNK=3)
    tokenizer = create_nucleotide_tokenizer()

    logger.info(f"Vocab size: {tokenizer.vocab_size}")
    logger.info(
        f"Special tokens: BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}"
    )

    # Save to default location
    save_path = os.path.join(os.path.dirname(__file__), "nucleotide_fast_tokenizer")
    tokenizer.save_pretrained(save_path)
    logger.info(f"Tokenizer saved to: {save_path}")


if __name__ == "__main__":
    main()
