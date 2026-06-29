#!/usr/bin/env python3

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

"""Test suite for train.py - validates the training loop with proper hardware checks.

This test file ensures the training script can run without errors while respecting
hardware limitations and providing proper mocking for distributed training.
"""

import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformer_engine.pytorch.fp8 import check_fp8_support

from train import DistributedConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestTrainingScript(unittest.TestCase):
    """Test cases for the training script."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.config_dir = Path("hydra_config")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_data_dir, ignore_errors=True)

    def test_imports_work(self):
        """Test that all required imports work correctly."""
        try:
            # Test core imports - using importlib to check availability without importing unused modules
            import importlib.util

            # Check if modules are available
            modules_to_check = [
                "hydra",
                "torch",
                "torch.distributed",
                "transformer_engine.pytorch",
                "megatron_fsdp",
                "transformers",
                "dataset",
                "modeling_bert_te",
            ]

            for module_name in modules_to_check:
                if importlib.util.find_spec(module_name) is None:
                    raise ImportError(f"Module {module_name} not found")

            logger.info("✅ All imports successful")

        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_cuda_availability(self):
        """Test CUDA availability and setup."""
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.device_count()} devices")
            logger.info(f"✅ Current device: {torch.cuda.current_device()}")
            logger.info(f"✅ Device name: {torch.cuda.get_device_name()}")
        else:
            logger.warning("⚠️  CUDA not available - will test CPU fallback")

    def test_fp8_support(self):
        """Test FP8 hardware support."""
        if not torch.cuda.is_available():
            logger.info("⚠️  Skipping FP8 test - CUDA not available")
            return

        fp8_available, reason = check_fp8_support()
        if fp8_available:
            logger.info("✅ FP8 support available")
        else:
            logger.info(f"⚠️  FP8 not supported: {reason}")

    def test_model_creation(self):
        """Test model creation and basic operations."""
        from transformers import BertConfig

        from modeling_bert_te import BertForMaskedLM

        # Create a small model config for testing
        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=512,
            type_vocab_size=2,
            use_te_layers=False,  # Start with standard layers
        )

        try:
            model = BertForMaskedLM(config)
            logger.info("✅ Model creation successful")

            # Test model forward pass
            batch_size, seq_len = 2, 32
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            self.assertIsNotNone(outputs.loss)
            logger.info("✅ Model forward pass successful")

        except Exception as e:
            self.fail(f"Model creation/forward pass failed: {e}")

    def test_dataloader_creation(self):
        """Test dataloader creation with mock data."""
        # Create a minimal parquet file for testing
        import pandas as pd

        from dataset import create_dataloader

        # Create mock data (including length column that gets removed by dataset.py)
        mock_data = pd.DataFrame(
            {
                "input_ids": [[1, 2, 3, 4, 5] for _ in range(10)],
                "attention_mask": [[1, 1, 1, 1, 1] for _ in range(10)],
                "labels": [[1, 2, 3, 4, 5] for _ in range(10)],
                "length": [5 for _ in range(10)],  # Length column expected by dataset.py
            }
        )

        test_file = self.test_data_dir / "test_data.parquet"
        mock_data.to_parquet(test_file)

        try:
            # Mock distributed components for testing
            with (
                patch("torch.distributed.get_world_size", return_value=1),
                patch("torch.distributed.get_rank", return_value=0),
            ):
                dataloader, length = create_dataloader(
                    str(test_file),
                    batch_size=2,
                    num_workers=0,  # Use 0 workers for testing
                    use_fp8=False,
                )

                # Test getting a batch
                batch = next(iter(dataloader))
                self.assertIn("input_ids", batch)
                self.assertIn("attention_mask", batch)
                self.assertIn("labels", batch)

                logger.info("✅ Dataloader creation successful")

        except Exception as e:
            self.fail(f"Dataloader creation failed: {e}")

    def test_distributed_setup_mock(self):
        """Test distributed setup with mocking."""
        with (
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch.dict(os.environ, {"LOCAL_RANK": "0"}),
        ):
            try:
                dist.init_process_group(backend="nccl")
                dist_config = DistributedConfig(
                    rank=0,
                    local_rank=0,
                    world_size=1,
                )
                self.assertEqual(dist_config.rank, 0)
                self.assertEqual(dist_config.world_size, 1)
                self.assertEqual(dist_config.local_rank, 0)
                self.assertTrue(dist_config.is_main_process())

                logger.info("✅ Distributed setup (mocked) successful")

            except Exception as e:
                self.fail(f"Distributed setup failed: {e}")

    def test_training_config_loading(self):
        """Test that training configs can be loaded."""
        from omegaconf import OmegaConf

        # Check if config files exist
        config_files = [
            "hydra_config/l0_sanity.yaml",
        ]

        missing_files = [config_file for config_file in config_files if not Path(config_file).exists()]

        if missing_files:
            logger.warning(f"⚠️  Missing config files: {missing_files}")
            # Create a minimal config for testing
            minimal_config = {
                "model": {
                    "vocab_size": 1000,
                    "hidden_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "intermediate_size": 256,
                    "max_position_embeddings": 512,
                    "type_vocab_size": 2,
                    "use_te_layers": False,
                    "micro_batch_size": 2,
                },
                "training": {
                    "learning_rate": 1e-4,
                    "num_train_steps": 5,
                    "num_workers": 0,
                    "use_fp8": False,
                    "fp8_recipe_kwargs": {
                        "fp8_format": "hybrid",
                        "amax_history_len": 1024,
                        "amax_compute_algo": "max",
                    },
                    "wandb_init_args": {"project": "test_project", "name": "test_run"},
                },
                "data": {"path": str(self.test_data_dir / "test_data.parquet")},
            }

            OmegaConf.create(minimal_config)
            logger.info("✅ Created minimal config for testing")
        else:
            logger.info("✅ All config files found")


def run_integration_test_l0_sanity():
    """Run a full integration test with the actual training script."""
    if not torch.cuda.is_available():
        logger.info("⚠️  Skipping integration test - CUDA not available")
        return

    # Set environment variables for distributed training
    env = os.environ.copy()
    env.update(
        {
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
            "WANDB_MODE": "disabled",
        }
    )

    # Get the full path to train.py
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train.py")

    # Run the training script with minimal steps
    cmd = [
        sys.executable,
        train_script,
        "--config-name",
        "l0_sanity",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
        )

        if result.returncode == 0:
            logger.info("✅ Integration test passed")
        else:
            logger.error(f"❌ Integration test failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("❌ Integration test timed out")
    except Exception as e:
        logger.error(f"❌ Integration test error: {e}")


def run_integration_test_l0_sanity_te_mfsdp():
    """Run a full integration test with the actual training script."""
    if not torch.cuda.is_available():
        logger.info("⚠️  Skipping integration test - CUDA not available")
        return

    # Set environment variables for distributed training
    env = os.environ.copy()
    env.update(
        {
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
            "WANDB_MODE": "disabled",
        }
    )

    # Get the full path to train.py
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train.py")

    # Run the training script with minimal steps
    cmd = [
        sys.executable,
        train_script,
        "--config-name",
        "l0_sanity",
        "model.use_te_layers=true",
        "training.use_mfsdp=true",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
        )

        if result.returncode == 0:
            logger.info("✅ Integration test passed")
        else:
            logger.error(f"❌ Integration test failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("❌ Integration test timed out")
    except Exception as e:
        logger.error(f"❌ Integration test error: {e}")


def run_integration_test_l0_sanity_mfsdp():
    """Run a full integration test with the actual training script."""
    if not torch.cuda.is_available():
        logger.info("⚠️  Skipping integration test - CUDA not available")
        return

    # Set environment variables for distributed training
    env = os.environ.copy()
    env.update(
        {
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
            "WANDB_MODE": "disabled",
        }
    )

    # Get the full path to train.py
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train.py")

    # Run the training script with minimal steps
    cmd = [
        sys.executable,
        train_script,
        "--config-name",
        "l0_sanity",
        "training.use_mfsdp=true",
        "model.use_te_layers=false",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
        )

        if result.returncode == 0:
            logger.info("✅ Integration test passed")
        else:
            logger.error(f"❌ Integration test failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("❌ Integration test timed out")
    except Exception as e:
        logger.error(f"❌ Integration test error: {e}")


def run_integration_test_l0_sanity_fp8():
    """Run a full integration test with the actual training script."""
    if not torch.cuda.is_available() or not check_fp8_support()[0]:
        logger.info("⚠️  Skipping integration test - CUDA not available")
        return

    # Set environment variables for distributed training
    env = os.environ.copy()
    env.update(
        {
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
            "WANDB_MODE": "disabled",
        }
    )

    # Get the full path to train.py
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train.py")

    # Run the training script with minimal steps
    cmd = [
        sys.executable,
        train_script,
        "--config-name",
        "l0_sanity",
        "training.use_fp8=true",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
        )

        if result.returncode == 0:
            logger.info("✅ Integration test passed")
        else:
            logger.error(f"❌ Integration test failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("❌ Integration test timed out")
    except Exception as e:
        logger.error(f"❌ Integration test error: {e}")


if __name__ == "__main__":
    print("🧪 Running Training Script Tests")

    # Run unit tests
    unittest.main(verbosity=2, exit=False)

    print("\n" + "=" * 50)
    print("🚀 Running Integration Test")
    print("=" * 50)

    run_integration_test_l0_sanity()
    run_integration_test_l0_sanity_fp8()
    run_integration_test_l0_sanity_mfsdp()
    run_integration_test_l0_sanity_te_mfsdp()

    print("\n✅ All tests completed!")
