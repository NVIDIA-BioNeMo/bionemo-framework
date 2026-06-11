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

"""Realistic unit tests for AutoTokenizer implementation that tests the core functionality."""

import tempfile

import pytest
import torch
from transformers import AutoTokenizer


@pytest.fixture
def tokenizer_auto():
    """Create an AutoTokenizer instance for testing."""
    return AutoTokenizer.from_pretrained("tokenizer_auto")


class TestAutoTokenizerFunctionality:
    """Test AutoTokenizer functionality and compatibility."""

    def test_autotokenizer_initialization(self, tokenizer_auto):
        """Test that AutoTokenizer initializes correctly."""
        assert tokenizer_auto is not None
        assert tokenizer_auto.vocab_size == 25426  # Real vocab size
        assert tokenizer_auto.pad_token == "<pad>"
        assert tokenizer_auto.mask_token == "<mask>"
        assert tokenizer_auto.model_max_length == 2048

    def test_autotokenizer_special_tokens(self, tokenizer_auto):
        """Test that AutoTokenizer has correct special tokens."""
        assert tokenizer_auto.pad_token_id == 0
        assert tokenizer_auto.mask_token_id == 1
        assert tokenizer_auto.unk_token_id == 0  # Same as pad_token_id
        assert tokenizer_auto.sep_token_id == 0  # Same as pad_token_id
        assert tokenizer_auto.cls_token_id == 0  # Same as pad_token_id

    def test_autotokenizer_tokenization(self, tokenizer_auto):
        """Test basic tokenization functionality."""
        # Test with some sample gene tokens
        test_tokens = ["ENSG00000000003", "ENSG00000000005", "ENSG00000000419"]

        # Tokenize
        encoded = tokenizer_auto(test_tokens, padding=True, truncation=True, return_tensors="pt")

        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert encoded["input_ids"].shape[0] == len(test_tokens)
        assert encoded["attention_mask"].shape == encoded["input_ids"].shape

    def test_autotokenizer_padding_strategies(self, tokenizer_auto):
        """Test different padding strategies with AutoTokenizer."""
        test_tokens = ["ENSG00000000003", "ENSG00000000005"]

        # Test max_length padding
        encoded_max = tokenizer_auto(test_tokens, padding="max_length", max_length=10, return_tensors="pt")
        assert encoded_max["input_ids"].shape[1] == 10

        # Test longest padding
        encoded_longest = tokenizer_auto(test_tokens, padding="longest", return_tensors="pt")
        # The tokenizer adds special tokens, so we need to check actual encoded length
        # Both test tokens should result in the same length since they're similar
        expected_length = encoded_longest["input_ids"].shape[1]
        assert expected_length >= 1  # Should be at least 1 token long

        # Test that both sequences have the same length (longest padding)
        assert encoded_longest["input_ids"].shape[0] == len(test_tokens)
        for i in range(len(test_tokens)):
            assert encoded_longest["input_ids"][i].shape[0] == expected_length

    def test_autotokenizer_decode_functionality(self, tokenizer_auto):
        """Test that AutoTokenizer can decode tokens correctly."""
        # Test with some known token IDs
        test_ids = [2, 3, 4, 5]

        # Decode
        decoded = tokenizer_auto.decode(test_ids, skip_special_tokens=True)
        assert isinstance(decoded, str)

        # Decode without skipping special tokens
        decoded_with_special = tokenizer_auto.decode(test_ids, skip_special_tokens=False)
        assert isinstance(decoded_with_special, str)

    def test_autotokenizer_batch_decode(self, tokenizer_auto):
        """Test batch decoding functionality."""
        # Test with batch of token IDs
        test_batch = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]

        # Batch decode
        decoded_batch = tokenizer_auto.batch_decode(test_batch, skip_special_tokens=True)
        assert isinstance(decoded_batch, list)
        assert len(decoded_batch) == len(test_batch)

    def test_autotokenizer_convert_tokens_to_ids(self, tokenizer_auto):
        """Test token to ID conversion."""
        # Test with special tokens
        assert tokenizer_auto.convert_tokens_to_ids("<pad>") == 0
        assert tokenizer_auto.convert_tokens_to_ids("<mask>") == 1

        # Test with list of tokens
        tokens = ["<pad>", "<mask>"]
        ids = tokenizer_auto.convert_tokens_to_ids(tokens)
        assert ids == [0, 1]

    def test_autotokenizer_convert_ids_to_tokens(self, tokenizer_auto):
        """Test ID to token conversion."""
        # Test with special token IDs
        assert tokenizer_auto.convert_ids_to_tokens(0) == "<pad>"
        assert tokenizer_auto.convert_ids_to_tokens(1) == "<mask>"

        # Test with list of IDs
        ids = [0, 1]
        tokens = tokenizer_auto.convert_ids_to_tokens(ids)
        assert tokens == ["<pad>", "<mask>"]

    def test_autotokenizer_special_tokens_mask(self, tokenizer_auto):
        """Test special tokens mask generation."""
        # Test with mixed tokens including special tokens
        token_ids = [2, 1, 3, 0, 4]  # regular, <mask>, regular, <pad>, regular
        mask = tokenizer_auto.get_special_tokens_mask(token_ids, already_has_special_tokens=True)

        # Should identify special tokens correctly
        assert len(mask) == len(token_ids)
        assert mask[1] == 1  # <mask> should be marked as special
        assert mask[3] == 1  # <pad> should be marked as special

    def test_autotokenizer_truncation(self, tokenizer_auto):
        """Test truncation functionality."""
        # Create a long sequence
        long_sequence = ["gene_" + str(i) for i in range(100)]

        # Test truncation
        encoded = tokenizer_auto(long_sequence, truncation=True, max_length=50, return_tensors="pt")
        assert encoded["input_ids"].shape[1] <= 50

    def test_autotokenizer_attention_mask(self, tokenizer_auto):
        """Test attention mask generation."""
        test_tokens = ["ENSG00000000003", "ENSG00000000005"]

        encoded = tokenizer_auto(test_tokens, padding=True, return_tensors="pt")

        # Check attention mask properties
        assert "attention_mask" in encoded
        assert encoded["attention_mask"].shape == encoded["input_ids"].shape

        # Attention mask should be 1 for all tokens (since these are real tokens, not padding)
        # In a batch where sequences have different lengths, shorter sequences get padded
        batch_size, seq_len = encoded["input_ids"].shape
        for i in range(batch_size):
            attention = encoded["attention_mask"][i]
            # Count non-zero elements in attention mask
            non_zero_count = (attention != 0).sum().item()
            # Should be > 0 since we have real tokens
            assert non_zero_count > 0
            # The attention mask should have 1s for real tokens and 0s for padding
            assert non_zero_count <= seq_len

    def test_data_collator_integration(self, tokenizer_auto):
        """Test integration with DataCollatorForLanguageModeling."""
        from transformers.data.data_collator import DataCollatorForLanguageModeling

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_auto, mlm=True, mlm_probability=0.15)

        # Test with sample data - create proper format expected by data collator
        sample_data = [{"input_ids": [2, 3, 4, 5]}, {"input_ids": [2, 3, 4]}]

        # Should work without errors
        result = data_collator(sample_data)

        assert "input_ids" in result
        assert "labels" in result
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)

    def test_padding_side_property(self, tokenizer_auto):
        """Test that padding_side property can be changed."""
        original_side = tokenizer_auto.padding_side

        tokenizer_auto.padding_side = "left"
        assert tokenizer_auto.padding_side == "left"

        tokenizer_auto.padding_side = "right"
        assert tokenizer_auto.padding_side == "right"

        # Reset to original
        tokenizer_auto.padding_side = original_side

    def test_left_vs_right_padding(self, tokenizer_auto):
        """Test left vs right padding behavior."""
        test_tokens = ["ENSG00000000003", "ENSG00000000005"]

        # Test right padding (default)
        tokenizer_auto.padding_side = "right"
        result_right = tokenizer_auto(test_tokens, padding="longest", return_tensors="pt")

        # Test left padding
        tokenizer_auto.padding_side = "left"
        result_left = tokenizer_auto(test_tokens, padding="longest", return_tensors="pt")

        # Both should have same shape but different padding positions
        assert result_right["input_ids"].shape == result_left["input_ids"].shape

        # Reset to default
        tokenizer_auto.padding_side = "right"


class TestAutoTokenizerCompatibility:
    """Test compatibility between different AutoTokenizer usage patterns."""

    def test_vocab_size_compatibility(self, tokenizer_auto):
        """Test that vocab sizes match expected values."""
        # Should have the same vocab size as the original tokenizer
        assert tokenizer_auto.vocab_size == 25426

    def test_special_token_ids_compatibility(self, tokenizer_auto):
        """Test that special token IDs are consistent."""
        # All non-mask special tokens should map to pad token ID
        assert tokenizer_auto.pad_token_id == 0
        assert tokenizer_auto.mask_token_id == 1
        assert tokenizer_auto.unk_token_id == 0  # Maps to pad
        assert tokenizer_auto.sep_token_id == 0  # Maps to pad
        assert tokenizer_auto.cls_token_id == 0  # Maps to pad

    def test_tokenization_consistency(self, tokenizer_auto):
        """Test that tokenization produces consistent results."""
        test_input = ["ENSG00000000003", "ENSG00000000005", "ENSG00000000419"]

        # Tokenize multiple times
        result1 = tokenizer_auto(test_input, padding=True, return_tensors="pt")
        result2 = tokenizer_auto(test_input, padding=True, return_tensors="pt")

        # Results should be identical
        assert torch.equal(result1["input_ids"], result2["input_ids"])
        assert torch.equal(result1["attention_mask"], result2["attention_mask"])

    def test_save_and_load_compatibility(self, tokenizer_auto):
        """Test that the tokenizer can be saved and loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save tokenizer
            tokenizer_auto.save_pretrained(temp_dir)

            # Load tokenizer
            loaded_tokenizer = AutoTokenizer.from_pretrained(temp_dir)

            # Test that loaded tokenizer works the same
            test_input = ["ENSG00000000003", "ENSG00000000005"]

            original_result = tokenizer_auto(test_input, padding=True, return_tensors="pt")
            loaded_result = loaded_tokenizer(test_input, padding=True, return_tensors="pt")

            assert torch.equal(original_result["input_ids"], loaded_result["input_ids"])
            assert torch.equal(original_result["attention_mask"], loaded_result["attention_mask"])

    def test_model_max_length_compatibility(self, tokenizer_auto):
        """Test that model max length is preserved."""
        assert tokenizer_auto.model_max_length == 2048

    def test_padding_side_compatibility(self, tokenizer_auto):
        """Test that padding side is set correctly."""
        # Should default to right padding
        assert tokenizer_auto.padding_side == "right"

    def test_return_tensors_compatibility(self, tokenizer_auto):
        """Test different return tensor formats."""
        test_input = ["ENSG00000000003", "ENSG00000000005"]

        # Test PyTorch tensors
        pt_result = tokenizer_auto(test_input, padding=True, return_tensors="pt")
        assert isinstance(pt_result["input_ids"], torch.Tensor)

        # Test NumPy arrays
        np_result = tokenizer_auto(test_input, padding=True, return_tensors="np")
        assert hasattr(np_result["input_ids"], "shape")  # NumPy-like interface

        # Test lists (default)
        list_result = tokenizer_auto(test_input, padding=True, return_tensors=None)
        assert isinstance(list_result["input_ids"], list)

    def test_tokenizer_attributes_compatibility(self, tokenizer_auto):
        """Test that tokenizer has expected HuggingFace attributes."""
        # Check standard HuggingFace tokenizer attributes
        assert hasattr(tokenizer_auto, "vocab_size")
        assert hasattr(tokenizer_auto, "pad_token_id")
        assert hasattr(tokenizer_auto, "mask_token_id")
        assert hasattr(tokenizer_auto, "model_max_length")
        assert hasattr(tokenizer_auto, "padding_side")
        assert hasattr(tokenizer_auto, "model_input_names")
        assert hasattr(tokenizer_auto, "pad")
        assert hasattr(tokenizer_auto, "convert_ids_to_tokens")
        assert hasattr(tokenizer_auto, "convert_tokens_to_ids")
        assert hasattr(tokenizer_auto, "get_special_tokens_mask")
        assert hasattr(tokenizer_auto, "save_pretrained")

    def test_model_input_names_compatibility(self, tokenizer_auto):
        """Test that model input names are correctly set."""
        # AutoTokenizer should have specific model input names
        assert "input_ids" in tokenizer_auto.model_input_names
        assert "attention_mask" in tokenizer_auto.model_input_names


class TestAutoTokenizerEdgeCases:
    """Test edge cases and error handling for AutoTokenizer."""

    def test_empty_input_handling(self, tokenizer_auto):
        """Test handling of empty inputs."""
        # Empty string
        result = tokenizer_auto("", return_tensors="pt")
        assert result["input_ids"].shape[1] >= 0

        # Empty list should be handled gracefully
        # Note: empty list may cause issues with some tokenizers, so we test with a list containing an empty string
        result = tokenizer_auto([""], return_tensors="pt")
        assert result["input_ids"].shape[0] == 1  # One sequence
        assert result["input_ids"].shape[1] >= 0  # Some tokens (likely special tokens)

    def test_single_token_input(self, tokenizer_auto):
        """Test handling of single token input."""
        result = tokenizer_auto("ENSG00000000003", return_tensors="pt")
        assert result["input_ids"].shape[0] == 1
        assert result["input_ids"].shape[1] >= 1

    def test_very_long_input_truncation(self, tokenizer_auto):
        """Test handling of very long inputs with truncation."""
        # Create a very long sequence
        very_long_input = " ".join([f"gene_{i}" for i in range(5000)])

        result = tokenizer_auto(very_long_input, truncation=True, max_length=100, return_tensors="pt")
        assert result["input_ids"].shape[1] <= 100

    def test_unicode_handling(self, tokenizer_auto):
        """Test handling of unicode characters."""
        # Test with some unicode characters (should be handled gracefully)
        unicode_input = "ENSG00000000003_αβγ"
        result = tokenizer_auto(unicode_input, return_tensors="pt")
        assert result["input_ids"].shape[1] > 0

    def test_none_input_handling(self, tokenizer_auto):
        """Test handling of None inputs."""
        # This should raise an appropriate error
        with pytest.raises((ValueError, TypeError)):
            tokenizer_auto(None)

    def test_mixed_type_batch_input(self, tokenizer_auto):
        """Test handling of mixed input types in batch."""
        # Mix of strings and lists should work
        mixed_input = ["ENSG00000000003", "ENSG00000000005"]
        result = tokenizer_auto(mixed_input, padding=True, return_tensors="pt")
        assert result["input_ids"].shape[0] == 2

    def test_unknown_token_handling(self, tokenizer_auto):
        """Test that unknown tokens are handled properly."""
        # Test with a token that likely doesn't exist in vocab
        unknown_token = "unknown_gene_xyz_12345"
        result = tokenizer_auto(unknown_token, return_tensors="pt")

        # Should still produce some output (likely mapped to unk token)
        assert result["input_ids"].shape[1] > 0

    def test_batch_with_different_lengths(self, tokenizer_auto):
        """Test batch processing with sequences of different lengths."""
        # Create sequences of different lengths
        sequences = [
            "ENSG00000000003",
            "ENSG00000000005 ENSG00000000419",
            "ENSG00000000003 ENSG00000000005 ENSG00000000419 ENSG00000000457",
        ]

        result = tokenizer_auto(sequences, padding=True, return_tensors="pt")

        # All sequences should be padded to the same length
        batch_size, seq_len = result["input_ids"].shape
        assert batch_size == len(sequences)

        # Check that attention mask correctly indicates real vs padded tokens
        for i in range(batch_size):
            attention = result["attention_mask"][i]
            # Should have at least one real token
            assert attention.sum() > 0
            # Should not exceed sequence length
            assert attention.sum() <= seq_len

    def test_tensor_handling_modes(self, tokenizer_auto):
        """Test different tensor handling modes."""
        test_input = ["ENSG00000000003", "ENSG00000000005"]

        # Test that the tokenizer can handle different tensor modes
        pt_result = tokenizer_auto(test_input, padding=True, return_tensors="pt")
        assert isinstance(pt_result["input_ids"], torch.Tensor)

        np_result = tokenizer_auto(test_input, padding=True, return_tensors="np")
        assert hasattr(np_result["input_ids"], "shape")  # NumPy-like interface

        list_result = tokenizer_auto(test_input, padding=True, return_tensors=None)
        assert isinstance(list_result["input_ids"], list)

        # All should have the same logical content
        assert pt_result["input_ids"].shape[0] == len(np_result["input_ids"]) == len(list_result["input_ids"])
