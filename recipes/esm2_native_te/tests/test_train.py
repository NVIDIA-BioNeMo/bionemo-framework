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

import random

import pytest
import torch
from hydra import compose, initialize_config_dir
from transformer_engine.pytorch.fp8 import check_fp8_support

from train_ddp import main as main_ddp
from train_fsdp2 import main as main_fsdp2
from train_mfsdp import main as main_mfsdp


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


requires_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or not check_fp8_support()[0],
    reason="Test requires FP8 support: " + check_fp8_support()[1],
)


def test_sanity_convergence_mfsdp(tmp_path, recipe_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
            ],
        )

    final_loss = main_mfsdp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_ddp(tmp_path, recipe_path):
    """Test that the main function can be invoked wrapping the model in DDP."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
            ],
        )

    final_loss = main_ddp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_ddp_non_streaming_dataset(tmp_path, recipe_path):
    """Test that the training script works with a non-streaming dataset."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "dataset.load_dataset_kwargs.streaming=False",
            ],
        )

    final_loss = main_ddp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2(tmp_path, recipe_path):
    """Test that the main function can be invoked wrapping the model in FSDP2."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


@requires_fp8
def test_sanity_mfsdp_fp8(tmp_path, recipe_path):
    # For MFSDP, we only check that the script can run successfully with FP8, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "fp8_config.enabled=true",
                f"checkpoint.ckpt_dir={tmp_path}",
                "num_train_steps=4",
            ],
        )

    main_mfsdp(sanity_config)


@requires_fp8
def test_sanity_ddp_fp8(tmp_path, recipe_path):
    # For DDP, we only check that the script can run successfully with FP8, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
                "num_train_steps=4",
            ],
        )

    main_ddp(sanity_config)


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
def test_sanity_ddp_fp8_partial_layers_stats_logging(tmp_path, recipe_path):
    """Test that layer-wise FP8 stats logging only collects stats for the specified FP8 layers."""
    fp8_log_dir = tmp_path / "fp8_stats_logs"

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
                "fp8_layers=[1,2,3]",
                "quant_stats_config.enabled=true",
                f"quant_stats_config.quant_log_dir={fp8_log_dir}",
                "num_train_steps=4",
            ],
        )

    main_ddp(sanity_config)

    metadata_log = fp8_log_dir / "rank_0" / "nvdlfw_inspect_logs" / "nvdlfw_inspect_globalrank-0.log"
    stats_log = fp8_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"

    assert metadata_log.exists(), "Metadata log file was not created"
    assert stats_log.exists(), "Statistics log file was not created"
    assert metadata_log.stat().st_size > 0, "Metadata log file is empty"
    assert stats_log.stat().st_size > 0, "Statistics log file is empty"

    # Verify that stats are only collected for layers 1-3 (the FP8 layers), not 4-6 (BF16 layers).
    stats_content = stats_log.read_text()
    for layer in [1, 2, 3]:
        assert f"layers.{layer}." in stats_content, f"Expected stats for FP8 layer {layer}"
    for layer in [4, 5, 6]:
        assert f"layers.{layer}." not in stats_content, f"Unexpected stats for BF16 layer {layer}"


@requires_fp8
def test_sanity_fsdp2_fp8_partial_layers_stats_logging(tmp_path, recipe_path):
    """Test that layer-wise FP8 stats logging works with FSDP2 for a subset of layers."""
    fp8_log_dir = tmp_path / "fp8_stats_logs"

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
                "fp8_layers=[1,2,3]",
                "quant_stats_config.enabled=true",
                f"quant_stats_config.quant_log_dir={fp8_log_dir}",
                "num_train_steps=4",
            ],
        )

    main_fsdp2(sanity_config)

    metadata_log = fp8_log_dir / "rank_0" / "nvdlfw_inspect_logs" / "nvdlfw_inspect_globalrank-0.log"
    stats_log = fp8_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"

    assert metadata_log.exists(), "Metadata log file was not created"
    assert stats_log.exists(), "Statistics log file was not created"
    assert metadata_log.stat().st_size > 0, "Metadata log file is empty"
    assert stats_log.stat().st_size > 0, "Statistics log file is empty"

    stats_content = stats_log.read_text()
    for layer in [1, 2, 3]:
        assert f"layers.{layer}." in stats_content, f"Expected stats for FP8 layer {layer}"
    for layer in [4, 5, 6]:
        assert f"layers.{layer}." not in stats_content, f"Unexpected stats for BF16 layer {layer}"


@requires_fp8
def test_sanity_convergence_fsdp2_fp8(tmp_path, recipe_path):
    """For FSDP2, we check that the script can run successfully with FP8 and check convergence."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "fp8_config.enabled=true",
                f"checkpoint.ckpt_dir={tmp_path}",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


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
                "quant_stats_config.enabled=true",
                f"quant_stats_config.quant_log_dir={fp8_log_dir}",
                "num_train_steps=4",
            ],
        )

    main_fsdp2(sanity_config)

    # Verify log structure (same assertions as above)
    assert fp8_log_dir.exists(), "FP8 log directory was not created"
    assert (fp8_log_dir / "rank_0").exists(), "rank_0 directory was not created"
    assert (fp8_log_dir / "rank_0" / "nvdlfw_inspect_logs").exists(), "nvdlfw_inspect_logs directory was not created"
    assert (fp8_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs").exists(), (
        "nvdlfw_inspect_statistics_logs directory was not created"
    )

    metadata_log = fp8_log_dir / "rank_0" / "nvdlfw_inspect_logs" / "nvdlfw_inspect_globalrank-0.log"
    stats_log = fp8_log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"

    assert metadata_log.exists(), "Metadata log file was not created"
    assert stats_log.exists(), "Statistics log file was not created"
    assert metadata_log.stat().st_size > 0, "Metadata log file is empty"
    assert stats_log.stat().st_size > 0, "Statistics log file is empty"


@requires_fp8
@pytest.mark.xfail(reason="MFSDP doesn't seem to support quantized_model_init (BIONEMO-3012)")
def test_sanity_mfsdp_fp8_and_model_init(tmp_path, recipe_path):
    # For MFSDP, we only check that the script can run successfully with FP8, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "fp8_config.enabled=true",
                "+config_kwargs.use_quantized_model_init=true",
                f"checkpoint.ckpt_dir={tmp_path}",
                "num_train_steps=4",
            ],
        )

    main_mfsdp(sanity_config)


@requires_fp8
def test_sanity_ddp_fp8_and_model_init(tmp_path, recipe_path):
    # For DDP, we only check that the script can run successfully with FP8, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
                "+config_kwargs.use_quantized_model_init=true",
                "num_train_steps=4",
            ],
        )

    main_ddp(sanity_config)


@requires_fp8
def test_sanity_convergence_fsdp2_fp8_and_model_init(tmp_path, recipe_path):
    """For FSDP2, we check that the script can run successfully with FP8 and check convergence."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "fp8_config.enabled=true",
                "+config_kwargs.use_quantized_model_init=true",
                f"checkpoint.ckpt_dir={tmp_path}",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_thd(tmp_path, monkeypatch, recipe_path):
    """For FSDP2, we check that the script can run successfully with FP8 and check convergence."""
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


@requires_fp8
def test_sanity_convergence_fsdp2_thd_fp8(tmp_path, monkeypatch, recipe_path):
    """For FSDP2, we check that the script can run successfully with THD + FP8 and check convergence."""
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "fp8_config.enabled=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_ddp_thd(tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    # For DDP, we only check that the script can run successfully with THD, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "num_train_steps=4",
            ],
        )

    main_ddp(sanity_config)


def test_sanity_mfsdp_thd(tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    # For MFSDP, we only check that the script can run successfully with THD, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "num_train_steps=4",
            ],
        )

    main_mfsdp(sanity_config)


@requires_fp8
def test_sanity_ddp_thd_fp8(tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    # For DDP, we only check that the script can run successfully with THD, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "use_sequence_packing=true",
                "num_train_steps=4",
                "fp8_config.enabled=true",
                "checkpoint.resume_from_checkpoint=false",
            ],
        )

    main_ddp(sanity_config)


@requires_fp8
def test_sanity_mfsdp_thd_fp8(tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    # For MFSDP, we only check that the script can run successfully with THD, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "use_sequence_packing=true",
                "num_train_steps=4",
                "fp8_config.enabled=true",
            ],
        )

    main_mfsdp(sanity_config)


def test_sanity_convergence_fsdp2_no_meta_device(tmp_path, recipe_path):
    """Meta-device init is now enabled by default, so we test that the script can be invoked without it."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "use_meta_device=false",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_mfsdp_huggingface_model(tmp_path, recipe_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "config_name_or_path=facebook/esm2_t6_8M_UR50D",
                "dataset.tokenizer_name=facebook/esm2_t6_8M_UR50D",
                "use_te=false",
            ],
        )

    final_loss = main_mfsdp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_ddp_huggingface_model(tmp_path, recipe_path):
    """Test that the main function can be invoked wrapping the model in DDP."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "config_name_or_path=facebook/esm2_t6_8M_UR50D",
                "dataset.tokenizer_name=facebook/esm2_t6_8M_UR50D",
                "use_te=false",
                "checkpoint.resume_from_checkpoint=false",
            ],
        )

    final_loss = main_ddp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_huggingface_model(tmp_path, recipe_path):
    """Test that the main function can be invoked wrapping the model in FSDP2."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "config_name_or_path=facebook/esm2_t6_8M_UR50D",
                "dataset.tokenizer_name=facebook/esm2_t6_8M_UR50D",
                "use_te=false",
                "checkpoint.resume_from_checkpoint=false",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_ddp_thd_token_packing(tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    # For DDP, we only check that the script can run successfully with THD, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "num_train_steps=4",
            ],
        )

    main_ddp(sanity_config)


def test_sanity_mfsdp_thd_token_packing(tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "num_train_steps=4",
            ],
        )

    main_mfsdp(sanity_config)


def test_sanity_fsdp2_thd_token_packing(tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "num_train_steps=4",
            ],
        )

    main_fsdp2(sanity_config)


def test_sanity_ddp_thd_token_packing_huggingface_model(tmp_path, recipe_path):
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "config_name_or_path=facebook/esm2_t6_8M_UR50D",
                "dataset.tokenizer_name=facebook/esm2_t6_8M_UR50D",
                "use_te=false",
                "num_train_steps=4",
                "use_torch_compile=false",
            ],
        )

    main_ddp(sanity_config)
