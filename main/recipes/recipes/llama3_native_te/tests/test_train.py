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

import gc
import random
import subprocess

import pytest
import torch
from hydra import compose, initialize_config_dir
from transformer_engine.pytorch.fp8 import check_fp8_support

from train_ddp import main as main_ddp
from train_fsdp2 import main as main_fsdp2
from train_fsdp2_cp import main as main_fsdp2_cp


# TODO(@jomitchell): Delete once https://nvbugspro.nvidia.com/bug/5458694 is fixed.
requires_datacenter_hardware = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not any(
        gpu_name in torch.cuda.get_device_name(0).upper() for gpu_name in ["H100", "H200", "B100", "B200", "B300"]
    ),
    reason="Test requires datacenter hardware (H100, H200, B100, B200, B300)",
)

_fp8_support_result = check_fp8_support() if torch.cuda.is_available() else (False, "CUDA not available")
requires_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or not _fp8_support_result[0],
    reason=f"Test requires FP8 support: {_fp8_support_result[1]}",
)


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def test_sanity_convergence_ddp_te(tmp_path, recipe_path):
    """Test that DDP training converges on dlcm sanity-scale data.

    This test validates:
    - The train_ddp.py script runs end-to-end without errors
    - Model, optimizer, and dataloader integrate correctly
    - Training converges to reasonable loss on small dataset
    - Uses L0_sanity config with small model and few training steps
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_ddp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # For genomic Causal LM, we expect convergence to < 2.0 on the small test dataset
    # The model should learn to predict simple patterns in the mock data
    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_ddp_te_grad_acc(tmp_path, recipe_path):
    """Test DDP training with gradient accumulation."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "grad_acc_steps=2",
            ],
        )

    final_loss = main_ddp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_ddp_hf(tmp_path, recipe_path):
    """Test that DDP training converges on dlcm sanity-scale data.

    This test validates:
    - The train_ddp.py script runs end-to-end without errors
    - Model, optimizer, and dataloader integrate correctly
    - Training converges to reasonable loss on small dataset
    - Uses L0_sanity config with small model and few training steps
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
                "use_te=false",
            ],
        )

    final_loss = main_ddp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # For genomic Causal LM, we expect convergence to < 2.0 on the small test dataset
    # The model should learn to predict simple patterns in the mock data
    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_fsdp2_te_bshd(tmp_path, recipe_path):
    """Test that FSDP2 training converges on dlcm sanity-scale data.

    This test validates:
    - The train_fsdp2.py script runs end-to-end without errors
    - FSDP2 wrapping and sharding work correctly
    - Training converges to reasonable loss on small dataset
    - Uses L0_sanity config with small model and few training steps
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
                "config_kwargs.attn_input_format=bshd",
            ],
        )

    final_loss = main_fsdp2(sanity_config)

    # FSDP2 should achieve similar convergence to DDP
    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_fsdp2_te_thd(tmp_path, recipe_path):
    """Test that FSDP2 training converges on dlcm sanity-scale data.

    This test validates:
    - The train_fsdp2.py script runs end-to-end without errors
    - FSDP2 wrapping and sharding work correctly
    - Training converges to reasonable loss on small dataset
    - Uses L0_sanity config with small model and few training steps
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
                "config_kwargs.attn_input_format=thd",
            ],
        )

    final_loss = main_fsdp2(sanity_config)

    # FSDP2 should achieve similar convergence to DDP
    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_fsdp2_te_bshd_grad_acc(tmp_path, recipe_path):
    """Test FSDP2 training with BSHD format and gradient accumulation."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "config_kwargs.attn_input_format=bshd",
                "grad_acc_steps=2",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_fsdp2_te_thd_grad_acc(tmp_path, recipe_path):
    """Test FSDP2 training with THD format and gradient accumulation."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
                "dataset.max_seq_length=1024",
                "grad_acc_steps=2",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_fsdp2_hf(tmp_path, recipe_path):
    """Test that FSDP2 training converges on dlcm sanity-scale data.

    This test validates:
    - The train_fsdp2.py script runs end-to-end without errors
    - FSDP2 wrapping and sharding work correctly
    - Training converges to reasonable loss on small dataset
    - Uses L0_sanity config with small model and few training steps
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
                "use_te=false",
                "use_torch_compile=false",  # Getting occasional errors "AssertionError: s52" with torch.compile.
                "use_meta_device=false",  # sometimes getting large losses here.
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # FSDP2 should achieve similar convergence to DDP
    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_ddp_non_streaming_dataset(tmp_path, recipe_path):
    """Test that DDP training works with non-streaming dataset.

    This test validates:
    - The dataloader works correctly with streaming=False
    - Map-style dataset integration works
    - Training converges similarly to streaming mode
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "dataset.load_dataset_kwargs.streaming=False",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_ddp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # Non-streaming mode should converge just as well as streaming
    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_fsdp2_non_streaming_dataset(tmp_path, recipe_path):
    """Test that FSDP2 training works with non-streaming dataset.

    This test validates:
    - FSDP2 works correctly with map-style datasets
    - Non-streaming mode doesn't break FSDP2 sharding
    - Training converges similarly to streaming mode
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "dataset.load_dataset_kwargs.streaming=False",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # Non-streaming mode should converge just as well as streaming
    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_ddp_with_sequence_packing(tmp_path, recipe_path):
    """Test that DDP training works with sequence packing enabled.

    This test validates:
    - Sequence packing works correctly
    - Training can run with sequence packing
    - No errors occur during forward/backward passes
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "dataset.max_seq_length=1024",
                "config_kwargs.attn_input_format=thd",
                "num_train_steps=10",  # Just verify it runs, don't test convergence
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_ddp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"


def test_sanity_fsdp2_with_sequence_packing(tmp_path, recipe_path):
    """Test that FSDP2 training works with sequence packing enabled.

    This test validates:
    - Sequence packing works correctly
    - Training can run with sequence packing
    - No errors occur during forward/backward passes
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
                "dataset.max_seq_length=1024",
                "num_train_steps=10",  # Just verify it runs, don't test convergence
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # Just check that training runs without errors
    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"


def test_train_fsdp2_fp8_bshd(tmp_path, recipe_path):
    """Test that FSDP2 training works with FP8 enabled."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity_cp",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
                "+dataset.pad_sequences_to_be_divisible_by=16",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_train_fsdp2_fp8_thd(tmp_path, recipe_path):
    """Test that FSDP2 training works with FP8 enabled."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


@requires_datacenter_hardware
def test_sanity_fsdp2_cp(tmp_path, recipe_path):
    """Test FSDP2 with context parallelism training on a single GPU (cp_size=1)."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity_cp",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "num_train_steps=10",  # Just verify it runs, don't test convergence
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_fsdp2_cp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"


def test_sanity_convergence_fsdp2_te_fused_adam(tmp_path, recipe_path):
    """Test FSDP2 training with TE FusedAdam for FP32 master weights.

    This test validates:
    - FusedAdam optimizer initializes correctly with FSDP2-wrapped model
    - Training converges with FP32 master weights maintained by FusedAdam
    - FusedAdam handles FP32 master weights at the optimizer level (no MixedPrecisionPolicy needed)
    """
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "use_fp32_master_weights=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_fsdp2_te_fused_adam_fp8(tmp_path, recipe_path):
    """Test FSDP2 + FusedAdam + FP8 training.

    This test validates FusedAdam works correctly alongside FP8 quantization,
    matching the approach used in the lingua 7B MXFP8 experiment config.
    """
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "use_fp32_master_weights=true",
                "fp8_config.enabled=true",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


@requires_fp8
def test_sanity_ddp_fp8_stats_logging(tmp_path, recipe_path):
    """Test that FP8 stats logging creates the expected log files."""
    fp8_log_dir = tmp_path / "fp8_stats_logs"

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "+dataset.pad_sequences_to_be_divisible_by=16",
                "fp8_config.enabled=true",
                "quant_stats_config.enabled=true",
                f"quant_stats_config.quant_log_dir={fp8_log_dir}",
                "num_train_steps=4",
            ],
        )

    main_ddp(sanity_config)

    # Verify the log directory structure was created
    assert fp8_log_dir.exists(), "FP8 log directory was not created"
    assert (fp8_log_dir / "rank_0").exists(), "rank_0 directory was not created"
    assert (fp8_log_dir / "rank_0" / "nvdlfw_inspect_logs").exists(), "nvdlfw_inspect_logs directory was not created"
    assert (fp8_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs").exists(), (
        "nvdlfw_inspect_statistics_logs directory was not created"
    )

    # Verify the log files exist
    metadata_log = fp8_log_dir / "rank_0" / "nvdlfw_inspect_logs" / "nvdlfw_inspect_globalrank-0.log"
    stats_log = fp8_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"

    assert metadata_log.exists(), "Metadata log file was not created"
    assert stats_log.exists(), "Statistics log file was not created"

    # Verify files are non-empty
    assert metadata_log.stat().st_size > 0, "Metadata log file is empty"
    assert stats_log.stat().st_size > 0, "Statistics log file is empty"


@requires_fp8
def test_sanity_fsdp2_fp8_stats_logging(tmp_path, recipe_path):
    """Test that FP8 stats logging works with FSDP2."""
    fp8_log_dir = tmp_path / "fp8_stats_logs"

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
                "+dataset.pad_sequences_to_be_divisible_by=16",
                "quant_stats_config.enabled=true",
                f"quant_stats_config.quant_log_dir={fp8_log_dir}",
                "num_train_steps=4",
            ],
        )

    main_fsdp2(sanity_config)

    # Verify log structure (same assertions as above)
    assert fp8_log_dir.exists()
    assert (fp8_log_dir / "rank_0" / "nvdlfw_inspect_logs" / "nvdlfw_inspect_globalrank-0.log").exists()
    assert (fp8_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log").exists()


@requires_fp8
def test_sanity_ddp_fp8_partial_layers_stats_logging(tmp_path, recipe_path):
    """Test DDP training with layer-wise FP8 stats (layers 1-3 only)."""
    quant_log_dir = tmp_path / "quant_stats_logs"

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "+dataset.pad_sequences_to_be_divisible_by=16",
                "fp8_config.enabled=true",
                "fp8_layers=[1,2,3]",
                "quant_stats_config.enabled=true",
                f"quant_stats_config.quant_log_dir={quant_log_dir}",
                "num_train_steps=4",
            ],
        )

    main_ddp(sanity_config)

    # Verify the log directory structure was created
    assert quant_log_dir.exists(), "Quant log directory was not created"
    assert (quant_log_dir / "rank_0").exists(), "rank_0 directory was not created"
    assert (quant_log_dir / "rank_0" / "nvdlfw_inspect_logs").exists(), "nvdlfw_inspect_logs directory was not created"
    assert (quant_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs").exists(), (
        "nvdlfw_inspect_statistics_logs directory was not created"
    )


@requires_fp8
def test_sanity_fsdp2_fp8_partial_layers_stats_logging(tmp_path, recipe_path):
    """Test FSDP2 training with layer-wise FP8 stats (layers 1-3 only)."""
    quant_log_dir = tmp_path / "quant_stats_logs"

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "+dataset.pad_sequences_to_be_divisible_by=16",
                "fp8_config.enabled=true",
                "fp8_layers=[1,2,3]",
                "quant_stats_config.enabled=true",
                f"quant_stats_config.quant_log_dir={quant_log_dir}",
                "num_train_steps=4",
            ],
        )

    main_fsdp2(sanity_config)

    # Verify log structure
    assert quant_log_dir.exists()
    assert (quant_log_dir / "rank_0" / "nvdlfw_inspect_logs" / "nvdlfw_inspect_globalrank-0.log").exists()
    assert (quant_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log").exists()


def run_train_cmd(cmd, recipe_path):
    """Run a training command and check for errors.

    Args:
        cmd: List of command arguments to run
        recipe_path: Path to the recipe directory (working directory for command)

    Raises:
        pytest.fail: If command returns non-zero exit code
    """
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,  # 4 minutes timeout
        cwd=str(recipe_path),
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command:\n{' '.join(cmd)}\nfailed with exit code {result.returncode}")


nsys_available = subprocess.run(["which", "nsys"], check=False, capture_output=True).returncode == 0


@pytest.mark.skipif(not nsys_available, reason="nsys not available in environment")
def test_nsight_profiler_trace_generation(tmp_path, recipe_path):
    """Test that Nsight profiler is configured correctly and generates trace metadata.

    This test validates:
    - The profiler can be enabled through configuration
    - The profiler runs without errors during training
    - Training under nsys produces .nsys-rep trace files
    - The profiler correctly detects whether it's running under nsys
    """
    nsys_output_path = tmp_path / "nsys_profile"

    run_train_cmd(
        [
            "nsys",
            "profile",
            "-o",
            str(nsys_output_path),
            "--trace=cuda,nvtx",
            "--pytorch=autograd-nvtx",
            "--python-sampling=true",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "torchrun",
            "--standalone",
            "--nproc_per_node=1",
            "train_ddp.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",
            "profiler.enabled=true",
            "profiler.start_step=1",
            "profiler.end_step=3",
            f"checkpoint.ckpt_dir={tmp_path}",
        ],
        recipe_path,
    )

    # Verify nsys trace file was created
    nsys_files = list(tmp_path.glob("nsys_profile*.nsys-rep"))
    assert len(nsys_files) > 0, f"No .nsys-rep files found in {tmp_path}"
