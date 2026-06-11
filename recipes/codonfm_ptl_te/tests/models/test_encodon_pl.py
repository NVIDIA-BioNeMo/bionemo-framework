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


from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.data.metadata import MetadataFields
from src.models.encodon_pl import EncodonPL
from src.models.encodon_te_pl import EncodonTEPL


@dataclass
class MockOptimizer:
    """Mock optimizer for testing"""

    def __init__(self, params):
        self.param_groups = [{"params": params}]


@dataclass
class MockScheduler:
    """Mock scheduler for testing"""

    def __init__(self, optimizer):
        self.optimizer = optimizer


@pytest.fixture
def base_config():
    """Base configuration for EncodonPL model"""
    return {
        "vocab_size": 69,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 3,
        "position_embedding_type": "rotary",
        "classifier_dropout": 0.1,
        "rotary_theta": 1e4,
        "ignore_index": -100,
        "loss_type": "regression",
        "lora": False,
        "lora_alpha": 32.0,
        "lora_r": 16,
        "lora_dropout": 0.1,
        "finetune_strategy": "full",
        "num_classes": 2,
        "use_downstream_head": False,
        "cross_attention_hidden_dim": 256,
        "cross_attention_num_heads": 8,
        "max_position_embeddings": 2048,
        "attn_input_format": "bshd",
        "optimizer": lambda params: MockOptimizer(params),
        "scheduler": lambda optimizer: MockScheduler(optimizer),
    }


@pytest.fixture
def sample_batch():
    """Sample batch data for language modeling testing"""
    batch_size = 2
    seq_len = 128
    return {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (batch_size, seq_len)),
        MetadataFields.ATTENTION_MASK: torch.ones(batch_size, seq_len),
        MetadataFields.LABELS: torch.randint(0, 69, (batch_size, seq_len)),
        MetadataFields.INPUT_MASK: torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


@pytest.fixture
def sample_regression_batch():
    """Sample batch data for regression testing with downstream head"""
    batch_size = 2
    seq_len = 128
    return {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (batch_size, seq_len)),
        MetadataFields.ATTENTION_MASK: torch.ones(batch_size, seq_len),
        MetadataFields.LABELS: torch.randn(batch_size),  # Continuous labels for regression
    }


@pytest.fixture
def sample_classification_batch():
    """Sample batch data for classification testing with downstream head"""
    batch_size = 2
    seq_len = 128
    return {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (batch_size, seq_len)),
        MetadataFields.ATTENTION_MASK: torch.ones(batch_size, seq_len),
        MetadataFields.LABELS: torch.randint(0, 2, (batch_size,)),  # Class labels
    }


class TestEncodonPLBasic:
    """Basic tests for EncodonPL model initialization and configuration"""

    def test_model_initialization_basic(self, base_config):
        """Test basic model initialization without LoRA or downstream head"""
        model = EncodonPL(**base_config)
        assert model.hparams.vocab_size == 69
        assert model.hparams.hidden_size == 128
        assert model.hparams.lora is False
        assert model.hparams.loss_type == "regression"
        assert model.hparams.use_downstream_head is False
        assert model.model is None  # Model not configured yet

    def test_model_initialization_with_language_modeling_loss(self, base_config):
        """Test model initialization with language modeling loss (no downstream head)"""
        model = EncodonPL(**base_config)
        assert isinstance(model.loss, nn.CrossEntropyLoss)
        assert model.loss.ignore_index == -100
        assert model.task_type == "language_modeling"

    def test_model_initialization_with_regression_loss(self, base_config):
        """Test model initialization with regression loss and downstream head"""
        base_config["loss_type"] = "regression"
        base_config["use_downstream_head"] = True
        model = EncodonPL(**base_config)
        assert isinstance(model.loss, nn.MSELoss)
        assert model.task_type == "regression"

    def test_model_initialization_with_classification_loss(self, base_config):
        """Test model initialization with classification loss and downstream head"""
        base_config["loss_type"] = "classification"
        base_config["use_downstream_head"] = True
        model = EncodonPL(**base_config)
        assert isinstance(model.loss, nn.CrossEntropyLoss)
        assert model.task_type == "classification"

    def test_invalid_loss_type_raises_error(self, base_config):
        """Test that invalid loss type raises ValueError"""
        base_config["loss_type"] = "invalid_loss"
        base_config["use_downstream_head"] = True
        with pytest.raises(ValueError, match="Unknown loss type"):
            EncodonPL(**base_config)


class TestEncodonPLLoRA:
    """Tests for LoRA functionality in EncodonPL"""

    def test_lora_initialization(self, base_config):
        """Test LoRA configuration during model initialization"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        model = EncodonPL(**base_config)
        model.configure_model()

        # Check that PEFT model was applied
        assert hasattr(model.model, "peft_config")
        assert model.model.peft_config["default"].peft_type.name == "LORA"
        assert model.model.peft_config["default"].r == 16
        assert model.model.peft_config["default"].lora_alpha == 32.0
        assert model.model.peft_config["default"].lora_dropout == 0.1

    def test_lora_target_modules(self, base_config):
        """Test that LoRA targets the correct modules"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        model = EncodonPL(**base_config)
        model.configure_model()

        expected_targets = ["query", "value", "intermediate_dense", "output_dense"]
        actual_targets = model.model.peft_config["default"].target_modules
        assert set(actual_targets) == set(expected_targets)

    def test_lora_cls_head_trainable(self, base_config):
        """Test that classification head remains trainable with LoRA"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        model = EncodonPL(**base_config)
        model.configure_model()

        # Check that cls parameters are trainable
        cls_params_trainable = any(param.requires_grad for param in model.model.cls.parameters())
        assert cls_params_trainable, "Classification head should remain trainable with LoRA"

    def test_lora_downstream_head_trainable(self, base_config):
        """Test that downstream heads remain trainable with LoRA"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        model = EncodonPL(**base_config)
        model.configure_model()

        # Check that downstream head parameters are trainable
        proj_params_trainable = any(
            param.requires_grad for param in model.model.cross_attention_input_proj.parameters()
        )
        head_params_trainable = any(param.requires_grad for param in model.model.cross_attention_head.parameters())
        assert proj_params_trainable, "Cross attention input projection should remain trainable with LoRA"
        assert head_params_trainable, "Cross attention head should remain trainable with LoRA"

    def test_lora_parameter_count_changes(self, base_config):
        """Test that LoRA changes the parameter count"""
        # Create model without LoRA (using full strategy)
        base_config["finetune_strategy"] = "full"
        base_config["lora"] = False
        model_no_lora = EncodonPL(**base_config)
        model_no_lora.configure_model()
        total_params_no_lora = sum(p.numel() for p in model_no_lora.model.parameters())

        # Create model with LoRA
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        model_with_lora = EncodonPL(**base_config)
        model_with_lora.configure_model()
        total_params_with_lora = sum(p.numel() for p in model_with_lora.model.parameters())

        # LoRA should add parameters
        assert total_params_with_lora > total_params_no_lora

    def test_lora_forward_pass(self, base_config, sample_batch):
        """Test forward pass with LoRA enabled"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        model = EncodonPL(**base_config)
        model.configure_model()

        # Test forward pass
        output = model.forward(sample_batch)
        assert output.logits is not None
        assert output.logits.shape == (2, 128, 69)  # batch_size, seq_len, vocab_size

    def test_lora_strategy_configuration(self, base_config):
        """Test LoRA with finetune_strategy='lora'"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        model = EncodonPL(**base_config)
        model.configure_model()

        # Check that LoRA is applied
        assert hasattr(model.model, "peft_config")

        # Check that cls head is trainable
        cls_params_trainable = any(param.requires_grad for param in model.model.cls.parameters())
        assert cls_params_trainable, "Classification head should be trainable"

        # Check that LoRA parameters are trainable
        lora_params_trainable = any(
            "lora" in name and param.requires_grad for name, param in model.model.named_parameters()
        )
        assert lora_params_trainable, "Some LoRA parameters should be trainable"

    def test_lora_state_dict_loading(self, base_config):
        """Test loading state dict with LoRA parameters"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        model = EncodonPL(**base_config)
        model.configure_model()

        # Create a mock state dict with LoRA parameters
        state_dict = model.state_dict()
        lora_keys = [k for k in state_dict.keys() if "lora" in k]
        assert len(lora_keys) > 0, "Should have LoRA parameters in state dict"

        # Test loading the state dict
        new_model = EncodonPL(**base_config)
        new_model.configure_model(state_dict)

        # Verify that LoRA was applied
        assert hasattr(new_model.model, "peft_config")


class TestEncodonPLDownstreamHead:
    """Tests for downstream head functionality with cross-attention"""

    def test_downstream_head_creation_regression(self, base_config):
        """Test that downstream heads are created for regression tasks"""
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        model = EncodonPL(**base_config)
        model.configure_model()

        assert hasattr(model.model, "cross_attention_input_proj")
        assert hasattr(model.model, "cross_attention_head")
        assert isinstance(model.model.cross_attention_input_proj, nn.Linear)

        # Check that regression head outputs single value
        assert model.model.cross_attention_head.output.out_features == 1

    def test_downstream_head_creation_classification(self, base_config):
        """Test that downstream heads are created for classification tasks"""
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "classification"
        base_config["num_classes"] = 3
        model = EncodonPL(**base_config)
        model.configure_model()

        assert hasattr(model.model, "cross_attention_input_proj")
        assert hasattr(model.model, "cross_attention_head")

        # Check that classification head outputs correct number of classes
        assert model.model.cross_attention_head.output.out_features == 3

    def test_downstream_head_forward_pass_regression(self, base_config, sample_regression_batch):
        """Test forward pass with downstream head for regression"""
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        model = EncodonPL(**base_config)
        model.configure_model()

        # Test model step
        loss, preds, targets = model.model_step(sample_regression_batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert preds.shape == (2,)  # batch_size predictions
        assert targets.shape == (2,)  # batch_size targets

    def test_downstream_head_forward_pass_classification(self, base_config, sample_classification_batch):
        """Test forward pass with downstream head for classification"""
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "classification"
        model = EncodonPL(**base_config)
        model.configure_model()

        # Test model step
        loss, preds, targets = model.model_step(sample_classification_batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert preds.shape == (2, 2)  # batch_size, num_classes predictions
        assert targets.shape == (2,)  # batch_size targets

    def test_downstream_head_with_head_only_random(self, base_config):
        """Test downstream head with head_only_random finetune strategy"""
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        base_config["finetune_strategy"] = "head_only_random"
        base_config["lora"] = False
        model = EncodonPL(**base_config)
        model.configure_model()

        # Check that downstream heads exist and are trainable
        assert hasattr(model.model, "cross_attention_input_proj")
        assert hasattr(model.model, "cross_attention_head")

        proj_params_trainable = any(
            param.requires_grad for param in model.model.cross_attention_input_proj.parameters()
        )
        head_params_trainable = any(param.requires_grad for param in model.model.cross_attention_head.parameters())
        assert proj_params_trainable, "Cross attention input projection should be trainable"
        assert head_params_trainable, "Cross attention head should be trainable"

        # Check that backbone is frozen
        backbone_params_frozen = all(not param.requires_grad for param in model.model.embeddings.parameters())
        assert backbone_params_frozen, "Backbone should be frozen with head_only_random"

    def test_downstream_head_with_lora(self, base_config):
        """Test downstream head with LoRA finetune strategy"""
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        model = EncodonPL(**base_config)
        model.configure_model()

        # Check that both LoRA and downstream heads are configured
        assert hasattr(model.model, "peft_config")
        assert hasattr(model.model, "cross_attention_input_proj")
        assert hasattr(model.model, "cross_attention_head")

        # Check that downstream heads are trainable
        proj_params_trainable = any(
            param.requires_grad for param in model.model.cross_attention_input_proj.parameters()
        )
        head_params_trainable = any(param.requires_grad for param in model.model.cross_attention_head.parameters())
        assert proj_params_trainable, "Cross attention input projection should be trainable"
        assert head_params_trainable, "Cross attention head should be trainable"


class TestEncodonPLLoRADownstreamIntegration:
    """Integration tests for LoRA + downstream head combination"""

    def test_lora_with_downstream_head_regression(self, base_config, sample_regression_batch):
        """Test LoRA combined with downstream head for regression"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        model = EncodonPL(**base_config)
        model.configure_model()

        # Check that both LoRA and downstream heads are configured
        assert hasattr(model.model, "peft_config")
        assert hasattr(model.model, "cross_attention_input_proj")
        assert hasattr(model.model, "cross_attention_head")

        # Test forward pass
        loss, preds, targets = model.model_step(sample_regression_batch)
        assert isinstance(loss, torch.Tensor)
        assert preds.shape == (2,)

    def test_lora_with_downstream_head_classification(self, base_config, sample_classification_batch):
        """Test LoRA combined with downstream head for classification"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "classification"
        model = EncodonPL(**base_config)
        model.configure_model()

        # Check that both LoRA and downstream heads are configured
        assert hasattr(model.model, "peft_config")
        assert hasattr(model.model, "cross_attention_input_proj")
        assert hasattr(model.model, "cross_attention_head")

        # Test forward pass
        loss, preds, targets = model.model_step(sample_classification_batch)
        assert isinstance(loss, torch.Tensor)
        assert preds.shape == (2, 2)

    def test_lora_downstream_head_state_dict_loading(self, base_config, sample_regression_batch):
        """Test loading state dict with both LoRA and downstream head"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        model = EncodonPL(**base_config)
        model.configure_model()

        # Get initial state dict
        initial_state_dict = model.state_dict()

        # Create new model and load state dict
        new_model = EncodonPL(**base_config)
        new_model.configure_model(initial_state_dict)

        # Verify both LoRA and downstream heads are present
        assert hasattr(new_model.model, "peft_config")
        assert hasattr(new_model.model, "cross_attention_input_proj")
        assert hasattr(new_model.model, "cross_attention_head")

        # Test forward pass works
        loss, preds, targets = new_model.model_step(sample_regression_batch)
        assert isinstance(loss, torch.Tensor)


class TestEncodonPLOptimizer:
    """Tests for optimizer configuration with different finetuning strategies"""

    def test_optimizer_full_finetuning(self, base_config):
        """Test optimizer configuration for full finetuning"""
        model = EncodonPL(**base_config)
        model.configure_model()

        # Mock trainer
        model.trainer = MagicMock()
        model.trainer.model = model

        optimizer_config = model.configure_optimizers()
        assert "optimizer" in optimizer_config

    def test_optimizer_head_only_finetuning(self, base_config):
        """Test optimizer configuration for head-only finetuning"""
        base_config["finetune_strategy"] = "head_only_pretrained"
        model = EncodonPL(**base_config)
        model.configure_model()

        optimizer_config = model.configure_optimizers()
        assert "optimizer" in optimizer_config

    def test_optimizer_lora_finetuning_downstream_head(self, base_config):
        """Test optimizer configuration for LoRA finetuning with downstream head"""
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        model = EncodonPL(**base_config)
        model.configure_model()

        # Mock trainer
        model.trainer = MagicMock()
        model.trainer.model = model

        optimizer_config = model.configure_optimizers()
        assert "optimizer" in optimizer_config


class TestEncodonPLTraining:
    """Tests for training functionality"""

    def test_training_step_language_modeling(self, base_config, sample_batch):
        """Test training step with language modeling loss (no downstream head)"""
        model = EncodonPL(**base_config)
        model.configure_model()

        loss = model.training_step(sample_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

    def test_training_step_regression(self, base_config, sample_regression_batch):
        """Test training step with regression loss and downstream head"""
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        model = EncodonPL(**base_config)
        model.configure_model()

        loss = model.training_step(sample_regression_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert model.hparams.loss_type == "regression"
        assert loss.ndim == 0

    def test_training_step_classification(self, base_config, sample_classification_batch):
        """Test training step with classification loss and downstream head"""
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "classification"
        model = EncodonPL(**base_config)
        model.configure_model()

        loss = model.training_step(sample_classification_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_validation_step(self, base_config, sample_batch):
        """Test validation step"""
        model = EncodonPL(**base_config)
        model.configure_model()

        # Should not raise any errors
        model.validation_step(sample_batch, 0)

    def test_gradient_clipping_configuration(self, base_config):
        """Test gradient clipping configuration"""
        model = EncodonPL(**base_config)
        model.configure_model()

        # Mock trainer
        model.trainer = MagicMock()
        model.trainer.strategy.__class__.__name__ = "DDPStrategy"
        model.trainer.gradient_clip_val = None  # Set to None to avoid conflicts
        model.trainer.gradient_clip_algorithm = None  # Set to None to avoid conflicts

        # Mock optimizer
        optimizer = MagicMock()

        # Test normal gradient clipping
        model.configure_gradient_clipping(optimizer, 1.0, "norm")

        # Test invalid gradient clip val
        with pytest.raises(ValueError, match="gradient_clip_val must be non-negative"):
            model.configure_gradient_clipping(optimizer, -1.0, "norm")

        # Test invalid algorithm
        with pytest.raises(ValueError, match="gradient_clip_algorithm must be one of"):
            model.configure_gradient_clipping(optimizer, 1.0, "invalid")


@pytest.mark.parametrize(
    "loss_type,use_downstream_head,finetune_strategy",
    [
        # Language modeling combinations (use_downstream_head=False)
        ("regression", False, "lora"),  # Note: loss_type ignored when downstream head disabled
        ("regression", False, "full"),
        ("regression", False, "head_only_pretrained"),
        ("regression", False, "head_only_random"),
        # Downstream head regression combinations (use_downstream_head=True)
        ("regression", True, "lora"),
        ("regression", True, "full"),
        ("regression", True, "head_only_pretrained"),
        ("regression", True, "head_only_random"),
        # Downstream head classification combinations (use_downstream_head=True)
        ("classification", True, "lora"),
        ("classification", True, "full"),
        ("classification", True, "head_only_pretrained"),
        ("classification", True, "head_only_random"),
    ],
)
def test_model_combinations(
    base_config,
    sample_batch,
    sample_regression_batch,
    sample_classification_batch,
    loss_type,
    use_downstream_head,
    finetune_strategy,
):
    """Test various valid combinations of loss types, downstream head usage, and finetuning strategies"""
    # Set lora based on finetune_strategy
    lora = finetune_strategy == "lora"

    base_config["lora"] = lora
    base_config["loss_type"] = loss_type
    base_config["use_downstream_head"] = use_downstream_head
    base_config["finetune_strategy"] = finetune_strategy

    model = EncodonPL(**base_config)
    model.configure_model()

    # Use appropriate batch based on task type
    if not use_downstream_head:
        test_batch = sample_batch  # Language modeling
    elif loss_type == "regression":
        test_batch = sample_regression_batch
    else:  # classification
        test_batch = sample_classification_batch

    # Test that model can be created and forward pass works
    try:
        loss, preds, targets = model.model_step(test_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

        # Verify that lora setting matches expectation
        if finetune_strategy == "lora":
            assert hasattr(model.model, "peft_config"), "LoRA should be enabled for finetune_strategy='lora'"
        else:
            # For non-LoRA strategies, peft_config should not exist
            assert not hasattr(model.model, "peft_config"), (
                f"LoRA should not be enabled for finetune_strategy='{finetune_strategy}'"
            )

        # Verify downstream head configuration
        if use_downstream_head:
            assert hasattr(model.model, "cross_attention_input_proj"), "Downstream head should be present"
            assert hasattr(model.model, "cross_attention_head"), "Downstream head should be present"
        else:
            assert not hasattr(model.model, "cross_attention_input_proj"), "Downstream head should not be present"
            assert not hasattr(model.model, "cross_attention_head"), "Downstream head should not be present"

    except Exception as e:
        pytest.fail(
            f"Model combination failed: loss_type={loss_type}, use_downstream_head={use_downstream_head}, finetune_strategy={finetune_strategy}. Error: {e}"
        )


def _make_lm_batch(batch_size=2, seq_len=16, vocab_size=69):
    return {
        MetadataFields.INPUT_IDS: torch.randint(0, vocab_size, (batch_size, seq_len)),
        MetadataFields.ATTENTION_MASK: torch.ones(batch_size, seq_len),
        MetadataFields.LABELS: torch.randint(0, vocab_size, (batch_size, seq_len)),
    }


def test_language_modeling_input_mask_fallback(base_config):
    # Ensure that when INPUT_MASK is missing, ATTENTION_MASK is used to mask labels
    model = EncodonPL(**base_config)
    model.configure_model()

    batch = _make_lm_batch()
    # Create sparse attention mask
    batch[MetadataFields.ATTENTION_MASK][0, 0:4] = 0

    loss, preds, targets = model.model_step(batch)
    # targets should include ignore_index for masked positions
    ignore_index = base_config["ignore_index"]
    flat_targets = targets.view(-1)
    assert (flat_targets == ignore_index).any()


def test_optimizer_step_skips_on_nan(base_config, monkeypatch):
    # Create a real optimizer over model params so step can be intercepted
    def _opt(params):
        return torch.optim.SGD([p for g in params for p in g["params"]], lr=0.1)

    base_config["optimizer"] = _opt
    model = EncodonPL(**base_config)
    model.configure_model()

    # Force a NaN gradient on one parameter
    p = next(model.parameters())
    p.grad = torch.full_like(p.data, float("nan"))

    # Build an SGD optimizer on the whole model so it has a .step method
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    called = {"step": 0}
    real_step = opt.step

    def patched_step(*a, **k):
        called["step"] += 1
        return real_step(*a, **k)

    monkeypatch.setattr(opt, "step", patched_step, raising=True)

    # Call optimizer_step using Lightning-style signature; gradients should be cleared before step
    model.optimizer_step(epoch=0, batch_idx=0, optimizer=opt)
    # Step should be invoked, but with zeroed/cleared grads so it has no effect
    assert called["step"] >= 1
    # Verify the NaN grad was cleared
    assert (p.grad is None) or torch.all(p.grad == 0)


class TestEncodonTEPL:
    """Tests for EncodonTEPL (Transformer Engine version)"""

    def test_te_model_initialization_basic(self, base_config):
        """Test basic EncodonTEPL model initialization"""
        if not torch.cuda.is_available():
            pytest.skip("Transformer Engine requires CUDA")
        model = EncodonTEPL(**base_config)
        assert model.hparams.vocab_size == 69
        assert model.hparams.hidden_size == 128
        assert model.hparams.attn_input_format == "bshd"
        assert model.model is None  # Model not configured yet

    def test_te_model_with_downstream_head_regression(self, base_config, sample_regression_batch):
        """Test EncodonTEPL with downstream head for regression"""
        if not torch.cuda.is_available():
            pytest.skip("Transformer Engine requires CUDA")
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        base_config["lora_dropout"] = 0.0
        model = EncodonTEPL(**base_config)
        model.configure_model()

        # Move model to CUDA
        model = model.cuda()

        # Move batch to CUDA
        sample_regression_batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in sample_regression_batch.items()
        }

        # Test forward pass
        loss, preds, targets = model.model_step(sample_regression_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert preds.shape == (2,)

    def test_te_model_with_downstream_head_classification(self, base_config, sample_classification_batch):
        """Test EncodonTEPL with downstream head for classification"""
        if not torch.cuda.is_available():
            pytest.skip("Transformer Engine requires CUDA")
        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "classification"
        base_config["lora_dropout"] = 0.0
        model = EncodonTEPL(**base_config)
        model.configure_model()

        # Move model to CUDA
        model = model.cuda()

        # Move batch to CUDA
        sample_classification_batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in sample_classification_batch.items()
        }

        # Test forward pass
        loss, preds, targets = model.model_step(sample_classification_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert preds.shape == (2, 2)

    @pytest.mark.parametrize("lora_dropout", [0.0, 0.1])
    def test_te_model_with_lora_and_dropout(self, base_config, sample_regression_batch, lora_dropout):
        """Test EncodonTEPL with LoRA finetuning strategy.

        This test validates that:
        1. LoRA works with Transformer Engine when lora_dropout=0.0
        2. LoRA fails gracefully with Transformer Engine when lora_dropout!=0.0

        Background: TransformerEngine's LayerNormLinear modules use ParamWrapper
        as a fallback in PEFT, which doesn't support non-zero dropout.
        """
        if not torch.cuda.is_available():
            pytest.skip("Transformer Engine requires CUDA")

        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        base_config["lora_r"] = 16
        base_config["lora_alpha"] = 32.0
        base_config["lora_dropout"] = lora_dropout

        if lora_dropout != 0.0:
            with pytest.raises(ValueError):
                model = EncodonTEPL(**base_config)
                model.configure_model()
        else:
            model = EncodonTEPL(**base_config)
            # Should work fine with lora_dropout=0.0
            model.configure_model()

            # Verify LoRA is properly configured
            assert hasattr(model.model, "peft_config"), "LoRA should be enabled"
            assert model.model.peft_config is not None

            # Move model to CUDA
            model = model.cuda()

            # Move batch to CUDA
            sample_regression_batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in sample_regression_batch.items()
            }

            # Test forward pass
            _loss, preds, _targets = model.model_step(sample_regression_batch)
            assert isinstance(_loss, torch.Tensor)
            assert _loss.ndim == 0
            assert preds.shape == (2,)

            # Verify that LoRA parameters are trainable
            lora_params = [p for n, p in model.named_parameters() if "lora" in n.lower()]
            assert len(lora_params) > 0, "Should have LoRA parameters"
            assert all(p.requires_grad for p in lora_params), "All LoRA parameters should be trainable"

            # Verify downstream heads are trainable
            cls_params = [p for n, p in model.named_parameters() if "cls" in n]
            assert len(cls_params) > 0, "Should have classifier parameters"
            assert all(p.requires_grad for p in cls_params), "All classifier parameters should be trainable"

            cross_attn_params = [p for n, p in model.named_parameters() if "cross_attention" in n]
            assert len(cross_attn_params) > 0, "Should have cross-attention parameters"
            assert all(p.requires_grad for p in cross_attn_params), (
                "All cross-attention parameters should be trainable"
            )

    def test_te_model_lora_single_gpu_integration(self, base_config, sample_regression_batch):
        """Integration test for LoRA + TransformerEngine on single GPU.

        This test simulates a full training step to ensure the entire pipeline works:
        - Model initialization with LoRA
        - Forward pass
        - Backward pass
        - Optimizer step
        """
        if not torch.cuda.is_available():
            pytest.skip("Transformer Engine requires CUDA")

        base_config["use_downstream_head"] = True
        base_config["loss_type"] = "regression"
        base_config["finetune_strategy"] = "lora"
        base_config["lora"] = True
        base_config["lora_r"] = 32
        base_config["lora_alpha"] = 32.0
        base_config["lora_dropout"] = 0.0  # Must be 0.0 for TE

        model = EncodonTEPL(**base_config)
        model.configure_model()
        model = model.cuda()

        # Move batch to CUDA
        sample_regression_batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in sample_regression_batch.items()
        }

        # Get trainable parameters before training
        trainable_params_before = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        # Perform a training step
        model.train()
        loss, _preds, _targets = model.model_step(sample_regression_batch)

        # Backward pass
        loss.backward()

        # Optimizer step (manual)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        optimizer.step()
        optimizer.zero_grad()

        # Verify that trainable parameters have changed
        params_changed = []
        for name, param in model.named_parameters():
            if param.requires_grad and name in trainable_params_before:
                if not torch.allclose(param, trainable_params_before[name], atol=1e-8):
                    params_changed.append(name)

        assert len(params_changed) > 0, "Some trainable parameters should have changed after optimizer step"

        # Verify LoRA parameters are among those that changed
        lora_params_changed = [n for n in params_changed if "lora" in n.lower()]
        assert len(lora_params_changed) > 0, "LoRA parameters should have been updated"
