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

import torch
from transformers.models.bert.configuration_bert import BertConfig

from geneformer.modeling_bert_te import BertForMaskedLM, BertLayer, TEBertLayer


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
    te_config.fuse_qkv_params = True
    te_model = BertForMaskedLM(te_config)
    # Check that the two models have the same parameters
    bert_layers = list(te_model.bert.encoder.layer)
    # Check that these layers are of type BertLayer
    for layer in bert_layers:
        assert isinstance(layer, TEBertLayer)


# Check models have same parameter count
def test_geneformer_te_and_hf_models_same_parameters(get_config):
    """Test that TE and HF models have same parameter count."""
    hf_model = BertForMaskedLM(get_config)

    te_config = BertConfig(**get_config.to_dict())
    te_config.use_te_layers, te_config.fuse_qkv_params = True, True
    te_model = BertForMaskedLM(te_config)

    assert te_model.num_parameters() == hf_model.num_parameters()


def test_te_bert_layer_and_hf_bert_layer_same_output_shapes(get_config):
    """Test that TE and HF BERT layers have same output shapes."""
    hf_model = BertForMaskedLM(get_config)

    te_config = BertConfig(**get_config.to_dict())
    te_config.use_te_layers, te_config.fuse_qkv_params = True, True
    te_model = BertForMaskedLM(te_config)

    # Extract the BertLayer from the models
    te_bert_layer = te_model.bert.encoder.layer[0]
    hf_bert_layer = hf_model.bert.encoder.layer[0]
    # Check that the two layers have the same output
    # Send in a random input of shape (12, 2048, 256) into both and confirm that they have the same output
    random_input = torch.randn(12, 2048, 256).cuda()
    te_model = te_model.cuda()
    hf_model = hf_model.cuda()

    te_output = te_bert_layer(hidden_states=random_input)
    hf_output = hf_bert_layer(hidden_states=random_input)

    assert te_output[0].shape == hf_output[0].shape


def test_te_bert_layer_and_hf_bert_layer_similar_output_values_random_inputs(get_config):
    """Test that TE and HF BERT layers have similar output values using proper conversion."""
    from geneformer.convert import convert_geneformer_hf_to_te

    hf_config = BertConfig(**get_config.to_dict())
    hf_model = BertForMaskedLM(hf_config)

    # Convert HF model to TE format using the conversion utility
    te_model = convert_geneformer_hf_to_te(hf_model)

    hf_model = hf_model.cuda()
    te_model = te_model.cuda()

    # Verify all weights are properly initialized (not NaN or uninitialized)
    for name, param in te_model.named_parameters():
        if param.requires_grad:
            assert not torch.isnan(param).any(), f"Parameter {name} contains NaN values after weight transfer"
            assert not torch.isinf(param).any(), f"Parameter {name} contains Inf values after weight transfer"

    te_model.eval()
    hf_model.eval()
    # Extract the BertLayer from the models
    te_bert_layer = te_model.bert.encoder.layer[0]
    hf_bert_layer = hf_model.bert.encoder.layer[0]

    # Send in a random input of shape (12, 2048, 256) into both and confirm that they have the same output
    torch.manual_seed(18)
    random_input = torch.randn(12, 2048, 256).cuda()
    with torch.no_grad():
        te_output = te_bert_layer(hidden_states=random_input)
        hf_output = hf_bert_layer(hidden_states=random_input)

    # With identical weights, outputs should be very close (allowing for numerical differences)
    torch.testing.assert_close(te_output[0], hf_output[0], atol=2e-4, rtol=5e-3)


def test_geneformer_model_loss_validity(input_data, get_config):
    """Test that the geneformer model produces valid loss values."""
    from geneformer.convert import convert_geneformer_hf_to_te

    # Create HF model first
    hf_model = BertForMaskedLM(get_config)

    # Convert HF model to TE format using the conversion utility
    te_model = convert_geneformer_hf_to_te(hf_model)

    device = torch.device("cuda")
    hf_model = hf_model.to(device)
    te_model = te_model.to(device)

    input_data = {k: v.to(device) for k, v in input_data.items()}

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            te_outputs = te_model(**input_data)
            hf_outputs = hf_model(**input_data)

    for model_name, outputs in [("TE", te_outputs), ("HF", hf_outputs)]:
        assert outputs.loss is not None, f"{model_name} model should produce a loss"
        assert not torch.isnan(outputs.loss), f"{model_name} model loss should not be NaN"
        assert not torch.isinf(outputs.loss), f"{model_name} model loss should not be infinite"

    torch.testing.assert_close(te_outputs.loss, hf_outputs.loss, atol=1e-2, rtol=1e-3)


def test_geneformer_model_logits_shape(input_data, te_config):
    """Test that the geneformer model produces logits with correct shape."""
    model = BertForMaskedLM(te_config)

    device = torch.device("cuda")
    model = model.to(device)
    input_data = {k: v.to(device) for k, v in input_data.items()}

    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**input_data)

    assert outputs.logits is not None, "Model should produce logits"
    expected_logits_shape = (input_data["input_ids"].shape[0], input_data["input_ids"].shape[1], te_config.vocab_size)
    assert outputs.logits.shape == expected_logits_shape, (
        f"Logits shape {outputs.logits.shape} should be {expected_logits_shape}"
    )


def test_geneformer_model_loss_convergence(input_data, te_config):
    """Test that the geneformer model loss decreases during training steps (CUDA required)."""
    import torch.optim as optim

    device = torch.device("cuda")
    model = BertForMaskedLM(te_config)
    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    input_data = {k: v.to(device) for k, v in input_data.items()}
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**input_data)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
    # Check that final loss is lower than initial loss
    assert losses[-1] < losses[0], f"Final loss {losses[-1]} should be lower than initial loss {losses[0]}"
