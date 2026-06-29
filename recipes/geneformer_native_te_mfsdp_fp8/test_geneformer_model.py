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

# A series of unit tests to check the GeneformerHF model vs the GeneformerTE Model


import pytest
import torch
from transformers.models.bert.configuration_bert import BertConfig

from modeling_bert_te import BertForMaskedLM, BertLayer, TEBertLayer


@pytest.fixture
def get_config():
    """Get a test configuration for BERT model."""
    return BertConfig(
        attention_probs_dropout_prob=0.02,
        classifier_dropout=None,
        hidden_act="gelu",
        hidden_dropout_prob=0.02,
        hidden_size=256,
        initializer_range=0.02,
        intermediate_size=1024,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
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


def test_geneformer_hf_model_has_vanilla_pt_layers(get_config):
    """Test that HuggingFace model has vanilla PyTorch layers."""
    model = BertForMaskedLM(get_config)
    assert model is not None
    # Make sure that the BertModel has BertLayer
    bert_layers = list(model.bert.encoder.layer)
    # Check that these layers are of type BertLayer
    for layer in bert_layers:
        assert isinstance(layer, BertLayer)


def test_geneformer_te_model_has_te_layers(get_config):
    """Test that TE model has TransformerEngine layers."""
    te_config = get_config
    te_config.use_te_layers = True
    te_model = BertForMaskedLM(te_config)
    # Check that the two models have the same parameters
    bert_layers = list(te_model.bert.encoder.layer)
    # Check that these layers are of type BertLayer
    for layer in bert_layers:
        assert isinstance(layer, TEBertLayer)


# Check models have same parameter count
def test_geneformer_te_and_hf_models_same_parameters(get_config):
    """Test that TE and HF models have same parameter count."""
    te_config = get_config
    te_config.use_te_layers = True
    te_model = BertForMaskedLM(te_config)
    hf_model = BertForMaskedLM(get_config)
    assert te_model.num_parameters() == hf_model.num_parameters()


def test_te_bert_layer_and_hf_bert_layer_same_output_shapes(get_config):
    """Test that TE and HF BERT layers have same output shapes."""
    te_config = get_config
    te_config.use_te_layers = True
    te_model = BertForMaskedLM(te_config)
    hf_model = BertForMaskedLM(get_config)
    # Extract the BertLayer from the models
    te_bert_layer = te_model.bert.encoder.layer[0]
    hf_bert_layer = hf_model.bert.encoder.layer[0]
    # Check that the two layers have the same output
    # Send in a random input of shape (12, 2048, 256) into both and confirm that they have the same output
    random_input = torch.randn(12, 2048, 256).cuda()

    te_output = te_bert_layer(hidden_states=random_input)
    hf_output = hf_bert_layer(hidden_states=random_input)

    assert te_output[0].shape == hf_output[0].shape


# def test_runtime_of_te_bert_layer_and_hf_bert_layer(get_config):
#     te_config = get_config
#     te_config.use_te_layers = True
#     te_model = BertForMaskedLM(te_config)
#     hf_model = BertForMaskedLM(get_config)
#     # Extract the BertLayer from the models
#     te_bert_layer = te_model.bert.encoder.layer[0]
#     hf_bert_layer = hf_model.bert.encoder.layer[0]
#     # Send in a random input of shape (12, 2048, 256) into both and confirm that they have the same output
#     random_input = torch.randn(12, 2048, 256).cuda()
#     import time

#     tic = time.time()#TODO: Get layer input.
#     te_output = te_bert_layer(hidden_states=random_input)
#     te_output_time = time.time() - tic
#     tic = time.time()
#     hf_output = hf_bert_layer(hidden_states=random_input)
#     hf_output_time = time.time() - tic
#     print(f"TE output time: {te_output_time}, HF output time: {hf_output_time}")
#     # assert te_output_time < hf_output_time


def test_te_bert_layer_and_hf_bert_layer_similar_output_values(get_config):
    """Test that TE and HF BERT layers have similar output values."""
    te_config = get_config
    te_config.use_te_layers = True
    te_model = BertForMaskedLM(te_config)
    hf_model = BertForMaskedLM(get_config)
    # Extract the BertLayer from the models
    te_bert_layer = te_model.bert.encoder.layer[0]
    hf_bert_layer = hf_model.bert.encoder.layer[0]
    # Load the weights from the pretrained model
    te_bert_layer.load_state_dict(hf_bert_layer.state_dict())
    # Send in a random input of shape (12, 2048, 256) into both and confirm that they have the same output
    random_input = torch.randn(12, 2048, 256).cuda()
    te_output = te_bert_layer(hidden_states=random_input)
    hf_output = hf_bert_layer(hidden_states=random_input)
    assert torch.abs(te_output[0] - hf_output[0]).mean() < 0.04
