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

"""Tests for CodonTHDCollator padding logic."""

import torch
from dataset import CodonTHDCollator
from tokenizer import CodonTokenizer


def _make_batch(sequences: list[str]) -> list[dict[str, str]]:
    """Create a batch of sequence dicts."""
    return [{"sequence": seq} for seq in sequences]


class TestCodonTHDCollatorPadding:
    """Test that CodonTHDCollator pads correctly for FP8/FP4 alignment."""

    def test_no_padding_when_disabled(self):
        """Without pad_to_multiple_of, total tokens can be any value."""
        tokenizer = CodonTokenizer()
        collator = CodonTHDCollator(tokenizer=tokenizer, mlm_probability=0.0, pad_to_multiple_of=None)

        # 3 short sequences will produce variable total token count
        batch = collator(_make_batch(["ATGATG", "ATGATGATG", "ATG"]))

        # cu_seq_lens should have num_sequences + 1 entries (no mock sequence)
        assert len(batch["cu_seq_lens_q"]) == 4  # 3 sequences + 1

    def test_pads_to_multiple_of_8(self):
        """Total tokens should be divisible by 8 when pad_to_multiple_of=8."""
        tokenizer = CodonTokenizer()
        collator = CodonTHDCollator(tokenizer=tokenizer, mlm_probability=0.0, pad_to_multiple_of=8)

        batch = collator(_make_batch(["ATGATG", "ATGATGATG", "ATG"]))

        assert batch["input_ids"].numel() % 8 == 0, (
            f"Expected total tokens to be divisible by 8, got {batch['input_ids'].numel()}"
        )
        assert batch["input_ids"].shape[0] == 1  # [1, total_tokens]

    def test_pads_to_multiple_of_32(self):
        """Total tokens should be divisible by 32 when pad_to_multiple_of=32."""
        tokenizer = CodonTokenizer()
        collator = CodonTHDCollator(tokenizer=tokenizer, mlm_probability=0.0, pad_to_multiple_of=32)

        batch = collator(_make_batch(["ATGATG", "ATGATGATG", "ATG"]))

        assert batch["input_ids"].numel() % 32 == 0
        assert batch["labels"].numel() % 32 == 0

    def test_padding_tokens_are_pad_id(self):
        """Padding tokens should use the tokenizer's pad_token_id."""
        tokenizer = CodonTokenizer()
        collator = CodonTHDCollator(tokenizer=tokenizer, mlm_probability=0.0, pad_to_multiple_of=64)

        batch = collator(_make_batch(["ATGATG"]))

        # With a single short sequence padded to 64, there will be many padding tokens
        total_tokens = batch["input_ids"].numel()
        assert total_tokens == 64

        # The tokenized sequence is short (CLS + 2 codons + SEP = 4 tokens)
        # Everything after that should be pad tokens
        real_token_count = batch["cu_seq_lens_q"][-2].item()  # End of last real sequence
        padding_region = batch["input_ids"][0, real_token_count:]
        assert (padding_region == tokenizer.pad_token_id).all()

    def test_padding_labels_are_minus_100(self):
        """Padding labels should be -100 (ignored in loss)."""
        tokenizer = CodonTokenizer()
        collator = CodonTHDCollator(tokenizer=tokenizer, mlm_probability=0.0, pad_to_multiple_of=64)

        batch = collator(_make_batch(["ATGATG"]))

        real_token_count = batch["cu_seq_lens_q"][-2].item()
        padding_labels = batch["labels"][0, real_token_count:]
        assert (padding_labels == -100).all()

    def test_cu_seq_lens_includes_mock_sequences(self):
        """cu_seq_lens should have extra entries for the mock padding sequence(s)."""
        tokenizer = CodonTokenizer()
        collator = CodonTHDCollator(tokenizer=tokenizer, mlm_probability=0.0, pad_to_multiple_of=64)

        num_sequences = 3
        batch = collator(_make_batch(["ATGATG", "ATGATGATG", "ATG"]))

        # Should have num_sequences + 1 (start) + at least 1 (mock) entries
        assert len(batch["cu_seq_lens_q"]) >= num_sequences + 2
        assert len(batch["cu_seq_lens_k"]) >= num_sequences + 2

        # Last entry should equal total tokens
        assert batch["cu_seq_lens_q"][-1].item() == batch["input_ids"].numel()

        # cu_seq_lens should be monotonically increasing
        for i in range(1, len(batch["cu_seq_lens_q"])):
            assert batch["cu_seq_lens_q"][i] > batch["cu_seq_lens_q"][i - 1]

    def test_cu_seq_lens_q_equals_k(self):
        """cu_seq_lens_q and cu_seq_lens_k should be identical."""
        tokenizer = CodonTokenizer()
        collator = CodonTHDCollator(tokenizer=tokenizer, mlm_probability=0.0, pad_to_multiple_of=16)

        batch = collator(_make_batch(["ATGATG", "ATGATGATG"]))

        torch.testing.assert_close(batch["cu_seq_lens_q"], batch["cu_seq_lens_k"])

    def test_no_padding_when_already_aligned(self):
        """No mock sequence should be added when total tokens is already a multiple."""
        tokenizer = CodonTokenizer()
        # Use pad_to_multiple_of=1 which is always satisfied
        collator = CodonTHDCollator(tokenizer=tokenizer, mlm_probability=0.0, pad_to_multiple_of=1)

        num_sequences = 2
        batch = collator(_make_batch(["ATGATG", "ATGATGATG"]))

        # No mock sequence needed, so cu_seq_lens has num_sequences + 1 entries
        assert len(batch["cu_seq_lens_q"]) == num_sequences + 1

    def test_exact_values_small_batch(self):
        """Test exact padding values with a known small batch."""
        tokenizer = CodonTokenizer()
        collator = CodonTHDCollator(tokenizer=tokenizer, mlm_probability=0.0, pad_to_multiple_of=8)

        # "ATG" -> CLS + ATG_token + SEP = 3 tokens
        # "ATGATG" -> CLS + ATG_token + ATG_token + SEP = 4 tokens
        # Total = 7 tokens, padded to 8
        batch = collator(_make_batch(["ATG", "ATGATG"]))

        assert batch["input_ids"].numel() == 8
        assert batch["labels"].numel() == 8

        # cu_seq_lens: [0, 3, 7, 8] (2 real sequences + 1 mock of length 1)
        expected_cu_seq_lens = torch.tensor([0, 3, 7, 8], dtype=torch.int32)
        torch.testing.assert_close(batch["cu_seq_lens_q"], expected_cu_seq_lens)

        # Last token should be pad
        assert batch["input_ids"][0, -1].item() == tokenizer.pad_token_id
        assert batch["labels"][0, -1].item() == -100

    def test_max_length_capped_at_max_seq_length(self):
        """max_length_q/k should never exceed max_seq_length, even with large padding."""
        tokenizer = CodonTokenizer()
        # Use a small max_seq_length and large pad_to_multiple_of to force remainder > max_seq_length
        collator = CodonTHDCollator(
            tokenizer=tokenizer, max_seq_length=10, mlm_probability=0.0, pad_to_multiple_of=128
        )

        # Single short sequence: CLS + ATG + SEP = 3 tokens, padded to 128 → remainder = 125
        batch = collator(_make_batch(["ATG"]))

        assert batch["input_ids"].numel() == 128
        # max_length must be capped at max_seq_length (10), not the full remainder (125)
        assert batch["max_length_q"] <= 10
        assert batch["max_length_k"] <= 10

    def test_padding_splits_long_sequences(self):
        """Padding remainder larger than max_seq_length should be split into multiple sequences."""
        tokenizer = CodonTokenizer()
        max_seq_length = 5
        pad_to_multiple_of = 100
        collator = CodonTHDCollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            mlm_probability=0.0,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        # Single short sequence: CLS + ATG + SEP = 3 tokens, padded to 100 → remainder = 97
        batch = collator(_make_batch(["ATG"]))

        assert batch["input_ids"].numel() == 100
        assert batch["input_ids"].numel() % pad_to_multiple_of == 0

        # max_length must stay within max_seq_length
        assert batch["max_length_q"] <= max_seq_length
        assert batch["max_length_k"] <= max_seq_length

        # Verify every individual sequence in cu_seq_lens is <= max_seq_length
        cu = batch["cu_seq_lens_q"]
        for i in range(1, len(cu)):
            seq_len = cu[i].item() - cu[i - 1].item()
            assert seq_len <= max_seq_length, (
                f"Sequence {i} has length {seq_len}, exceeds max_seq_length={max_seq_length}"
            )

        # Last entry must equal total tokens
        assert cu[-1].item() == batch["input_ids"].numel()


class TestCreateTHDDataloaderPadding:
    """Test that create_thd_dataloader defaults pad_to_multiple_of correctly."""

    def test_default_pad_to_multiple_of(self):
        """Default should be micro_batch_size * max_seq_length."""
        from dataset import create_thd_dataloader
        from distributed_config import DistributedConfig

        dist_config = DistributedConfig()
        micro_batch_size = 2
        max_seq_length = 512

        dataloader, _ = create_thd_dataloader(
            dist_config=dist_config,
            data_path="synthetic",
            micro_batch_size=micro_batch_size,
            max_seq_length=max_seq_length,
            mlm_probability=0.0,
        )

        # Every batch should have exactly micro_batch_size * max_seq_length tokens
        expected_total = micro_batch_size * max_seq_length
        batch = next(iter(dataloader))
        assert batch["input_ids"].numel() == expected_total, (
            f"Expected {expected_total} tokens, got {batch['input_ids'].numel()}"
        )
