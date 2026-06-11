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

"""Pytest configuration for geneformer tests."""

import os
import tempfile

import pytest
import requests
import torch
from transformers.models.bert.configuration_bert import BertConfig


@pytest.fixture
def get_config():
    """Get a test configuration for BERT model."""
    return BertConfig(
        attention_probs_dropout_prob=0.02,
        classifier_dropout=None,
        hidden_act="relu",
        hidden_dropout_prob=0.02,
        hidden_size=256,
        initializer_range=0.02,
        intermediate_size=1024,
        layer_norm_eps=1e-12,
        max_position_embeddings=2048,
        micro_batch_size=4,
        model_type="bert",
        num_attention_heads=8,
        num_hidden_layers=6,
        pad_token_id=0,
        position_embedding_type="absolute",
        seq_length=2048,
        transformers_version="4.52.0.dev0",
        type_vocab_size=2,
        use_cache=True,
        use_te_layers=False,
        vocab_size=25427,
    )


@pytest.fixture
def input_data():
    """Create realistic geneformer input data using actual gene token dictionary.

    Following the actual Geneformer implementation:
    - input_ids: Gene token IDs from actual gene vocabulary (truncated to max length)
    - attention_mask: Mask for valid tokens (1 for genes, 0 for unused positions)
    - labels: Masked language modeling labels (-100 for non-masked tokens)

    Note: Geneformer truncates sequences rather than padding them, as seen in the tokenizer.
    """
    import pickle

    # Download the token dictionary from Hugging Face
    token_dict_url = "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/token_dictionary_gc104M.pkl"

    # Create a temporary file to store the downloaded dictionary
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
        tmp_file_path = tmp_file.name

    try:
        print(f"Downloading geneformer token dictionary from {token_dict_url}")
        response = requests.get(token_dict_url, stream=True)
        response.raise_for_status()

        # Write the downloaded content to the temporary file
        with open(tmp_file_path, "wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))

        # Load the token dictionary from the temporary file
        with open(tmp_file_path, "rb") as f:
            token_dictionary = pickle.load(f)

        # Get some actual gene token IDs (excluding special tokens)
        gene_tokens = [
            token_id for token_id in token_dictionary.values() if isinstance(token_id, int) and token_id > 2
        ]  # Exclude special tokens

        assert len(gene_tokens) > 0, "No valid gene tokens found in token dictionary"

        # Use actual gene tokens for realistic testing
        batch_size = 2
        seq_length = 2048  # taken from examples/pretraining_new_model/pretrain_geneformer_w_deepspeed.py
        num_genes = min(seq_length, len(gene_tokens))  # Use full sequence length

        # Create input_ids using actual gene tokens (based on Geneformer/geneformer/tokenizer.py)
        # No padding needed - Geneformer truncates sequences like in tokenizer.py line 742
        input_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
        for i in range(batch_size):
            # example["input_ids"][0 : self.model_input_size]
            gene_indices = torch.randint(0, len(gene_tokens), (num_genes,))
            input_ids[i, :num_genes] = torch.tensor([gene_tokens[idx] for idx in gene_indices])

        # Create attention mask - 1 for real tokens, 0 for unused positions
        # based on models/geneformer/Geneformer/geneformer/evaluation_utils.py line 45
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bfloat16)
        for i in range(batch_size):
            attention_mask[i, num_genes:] = 0  # Mask out unused positions

        # Create labels for masked language modeling
        # -100 for non-masked tokens, actual token ID for masked tokens
        labels = torch.full((batch_size, seq_length), -100, dtype=torch.long)

        # Mask some tokens randomly (15% masking rate, typical for MLM)
        mask_indices = torch.rand(batch_size, seq_length) < 0.15
        mask_indices = mask_indices & (attention_mask.bool())  # Only mask real tokens
        labels[mask_indices] = input_ids[mask_indices]

        print("Created realistic geneformer input data:")
        print(f"  - Using {len(gene_tokens)} actual gene tokens")
        print(f"  - Sequence length: {seq_length}")
        print(f"  - Actual genes per sequence: {num_genes}")
        print(f"  - Masked tokens: {(labels != -100).sum().item()}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@pytest.fixture
def te_config():
    """Create a TEBertConfig for testing with TE layers enabled."""
    from geneformer.modeling_bert_te import TEBertConfig

    config_dict = {
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "hidden_act": "relu",
        "max_position_embeddings": 4096,  # Model capacity
        "vocab_size": 20275,  # Geneformer vocabulary size
        "torch_dtype": torch.bfloat16,
        "use_te_layers": True,
        "fuse_qkv_params": True,  # Enable fused QKV parameters for TE optimization
    }
    return TEBertConfig(**config_dict)
