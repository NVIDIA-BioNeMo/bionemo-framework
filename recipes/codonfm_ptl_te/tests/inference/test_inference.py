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

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.data.metadata import MetadataFields
from src.inference.base import BaseInference
from src.inference.encodon import EncodonInference
from src.inference.model_outputs import (
    DownstreamPredictionOutput,
    EmbeddingOutput,
    FitnessPredictionOutput,
    MaskedLMOutput,
    MutationPredictionOutput,
)
from src.inference.task_types import TaskTypes


class DummyInference(BaseInference):
    def configure_model(self):
        self.model = MagicMock()

    def _predict_step(self, batch, batch_idx):
        return {"ok": True, "batch_idx": batch_idx}


def test_base_inference_predict_step_increments_counter():
    inf = DummyInference(model_path="/tmp/dummy.ckpt", task_type="dummy")
    before = inf.prediction_counter
    out = inf.predict_step({}, 0)
    assert out["ok"] is True
    assert inf.prediction_counter == before + 1


@pytest.fixture
def mock_ckpt(tmp_path):
    # Minimal lightning-like checkpoint content
    ckpt = {
        "hyper_parameters": {
            "vocab_size": 69,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "intermediate_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 3,
            "position_embedding_type": "rotary",
            "classifier_dropout": 0.0,
            "rotary_theta": 1e4,
            "ignore_index": -100,
            "loss_type": "regression",
            "lora": False,
            "lora_alpha": 32.0,
            "lora_r": 16,
            "lora_dropout": 0.0,
            "finetune_strategy": "full",
            "num_classes": 2,
            "use_downstream_head": False,
            "cross_attention_hidden_dim": 32,
            "cross_attention_num_heads": 4,
            "max_position_embeddings": 256,
        },
        "state_dict": {},
    }
    p = tmp_path / "model.ckpt"
    torch.save(ckpt, p)
    return str(p)


def _make_lm_batch(bs=2, seqlen=8, vocab_size=69):
    return {
        MetadataFields.INPUT_IDS: torch.randint(0, vocab_size, (bs, seqlen)),
        MetadataFields.ATTENTION_MASK: torch.ones(bs, seqlen),
        MetadataFields.LABELS: torch.randint(0, vocab_size, (bs, seqlen)),
        MetadataFields.INPUT_MASK: torch.ones(bs, seqlen, dtype=torch.bool),
    }


@patch("src.inference.encodon.EncodonPL")
@patch("src.inference.encodon.EncodonTEPL")
@pytest.mark.parametrize("use_te", [False, True])
def test_encodon_inference_configure_and_mlm(mock_te_pl, mock_pl, mock_ckpt, use_te):
    # Make forward return an object with .logits
    class Out:
        def __init__(self, logits):
            self.logits = logits

    instance = MagicMock()
    instance.side_effect = lambda batch: Out(
        torch.randn(batch[MetadataFields.INPUT_IDS].shape[0], batch[MetadataFields.INPUT_IDS].shape[1], 69)
    )
    mock_pl.return_value = instance
    mock_te_pl.return_value = instance

    if use_te and not torch.cuda.is_available():
        pytest.skip("Transformer Engine requires CUDA")
    inf = EncodonInference(
        model_path=mock_ckpt, task_type=TaskTypes.MASKED_LANGUAGE_MODELING, use_transformer_engine=use_te
    )
    inf.configure_model()
    batch = _make_lm_batch()
    out = inf._predict_step(batch, 0)
    assert isinstance(out, MaskedLMOutput)
    assert out.preds.ndim == 2  # collapsed by mask


@patch("src.inference.encodon.EncodonPL")
def test_encodon_inference_mutation(mock_pl, mock_ckpt):
    from types import SimpleNamespace

    instance = MagicMock()
    # logits for batch=2, seq=5, vocab=69
    instance.return_value = SimpleNamespace(logits=torch.randn(2, 5, 69))
    mock_pl.return_value = instance

    inf = EncodonInference(model_path=mock_ckpt, task_type=TaskTypes.MUTATION_PREDICTION)
    inf.configure_model()
    batch = {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (2, 5)),
        MetadataFields.ATTENTION_MASK: torch.ones(2, 5),
        MetadataFields.REF_CODON_TOKS: torch.tensor([1, 2]),
        MetadataFields.ALT_CODON_TOKS: torch.tensor([3, 4]),
        MetadataFields.MUTATION_TOKEN_IDX: torch.tensor([1, 2]),
    }
    out = inf._predict_step(batch, 0)
    assert isinstance(out, MutationPredictionOutput)
    assert out.ref_likelihoods.shape == (2,)


@patch("src.inference.encodon.EncodonPL")
def test_encodon_inference_embeddings(mock_pl, mock_ckpt):
    from types import SimpleNamespace

    instance = MagicMock()
    instance.return_value = SimpleNamespace(
        all_hidden_states=[torch.randn(2, 5, 16), torch.randn(2, 5, 16)],
        last_hidden_state=torch.randn(2, 5, 16),
    )
    mock_pl.return_value = instance

    inf = EncodonInference(model_path=mock_ckpt, task_type=TaskTypes.EMBEDDING_PREDICTION)
    inf.configure_model()
    batch = {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (2, 5)),
        MetadataFields.ATTENTION_MASK: torch.ones(2, 5),
    }
    out = inf._predict_step(batch, 0)
    assert isinstance(out, EmbeddingOutput)
    assert out.embeddings.shape[0] == 2


@patch("src.inference.encodon.EncodonPL")
@patch("src.inference.encodon.EncodonTEPL")
@pytest.mark.parametrize("use_te", [False, True])
def test_encodon_inference_fitness(mock_te_pl, mock_pl, mock_ckpt, use_te):
    from types import SimpleNamespace

    instance = MagicMock()
    instance.return_value = SimpleNamespace(logits=torch.randn(2, 5, 69))
    mock_pl.return_value = instance
    mock_te_pl.return_value = instance

    if use_te and not torch.cuda.is_available():
        pytest.skip("Transformer Engine requires CUDA")
    inf = EncodonInference(model_path=mock_ckpt, task_type=TaskTypes.FITNESS_PREDICTION)
    inf.configure_model()
    batch = {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (2, 5)),
        MetadataFields.ATTENTION_MASK: torch.ones(2, 5),
    }
    out = inf._predict_step(batch, 0)
    assert isinstance(out, FitnessPredictionOutput)
    assert out.fitness.shape == (2,)


@patch("src.inference.encodon.EncodonPL")
@patch("src.inference.encodon.EncodonTEPL")
@pytest.mark.parametrize("use_te", [False, True])
def test_encodon_inference_downstream(mock_te_pl, mock_pl, mock_ckpt, use_te):
    from types import SimpleNamespace

    instance = MagicMock()

    # Configure the return value for model forward pass
    instance.return_value = SimpleNamespace(last_hidden_state=torch.randn(2, 5, 64))

    # Configure model attributes for downstream prediction
    instance.model = MagicMock()
    instance.model.cross_attention_input_proj = MagicMock(return_value=torch.randn(2, 5, 32))
    instance.model.cross_attention_head = MagicMock(return_value=torch.randn(2, 1))

    # Configure hparams for loss_type check
    instance.hparams = SimpleNamespace(loss_type="regression")

    mock_pl.return_value = instance
    mock_te_pl.return_value = instance

    if use_te and not torch.cuda.is_available():
        pytest.skip("Transformer Engine requires CUDA")

    inf = EncodonInference(
        model_path=mock_ckpt, task_type=TaskTypes.DOWNSTREAM_PREDICTION, use_transformer_engine=use_te
    )
    inf.configure_model()
    batch = {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (2, 5)),
        MetadataFields.ATTENTION_MASK: torch.ones(2, 5),
    }
    out = inf._predict_step(batch, 0)
    assert isinstance(out, DownstreamPredictionOutput)
    assert out.predictions.shape == (2,)
