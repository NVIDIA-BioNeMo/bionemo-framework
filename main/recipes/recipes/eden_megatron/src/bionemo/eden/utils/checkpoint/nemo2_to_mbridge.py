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

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from megatron.bridge.training.checkpointing import save_tokenizer_assets
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MIXED_PRECISION_RECIPES
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

from bionemo.eden.models.eden_provider import EDEN_MODEL_OPTIONS, EdenModelProvider
from bionemo.eden.recipes.eden import eden_pretrain_config


logger = logging.getLogger(__name__)


def convert_nemo2_dcp_to_megatron(
    src_path: str | Path,
    dest_path: str | Path,
):
    """Convert a torch_dist format checkpoint with nemo2 style names to one with megatron bridge style names.

    Args:
        src_path: Path to the source DCP checkpoint.
        dest_path: Path to the destination DCP checkpoint.
    """
    logger.info(f"Reading metadata from {src_path}...")
    reader = FileSystemReader(str(src_path))
    metadata = reader.read_metadata()

    # 1. Pre-allocate state_dict based on metadata
    # We need to construct the state_dict so dcp.load knows what to load.
    state_dict = {}
    total_size_bytes = 0

    for key, item_meta in metadata.state_dict_metadata.items():
        if isinstance(item_meta, BytesStorageMetadata):
            # Skip or handle non-tensor data if necessary
            continue

        # Create empty tensor on CPU with correct shape/dtype
        # DCP will load data into these tensors in-place
        state_dict[key] = torch.empty(item_meta.size, dtype=item_meta.properties.dtype, device="cpu")

        # Track size to calculate shard count later
        total_size_bytes += state_dict[key].numel() * state_dict[key].element_size()

    print(f"Loading {len(state_dict)} tensors into memory (Approx {total_size_bytes / 1e9:.2f} GB)...")

    # 2. Load directly from DCP to memory (no_dist=True for single process)
    dcp.load(state_dict=state_dict, storage_reader=reader, no_dist=True)

    # 3. Rename keys in-place (strip "module." prefix)
    for k in list(state_dict.keys()):
        if k.startswith("module."):
            state_dict[k[len("module.") :]] = state_dict.pop(k)

    logger.info(f"Keys munged. saving to {dest_path}...")

    # 4. Save to DCP with Sharding
    writer = FileSystemWriter(
        dest_path,
        single_file_per_rank=False,
        thread_count=os.cpu_count(),
    )

    dcp.save(state_dict=state_dict, storage_writer=writer, no_dist=True)
    del state_dict
    logger.info("Conversion complete.")


def _dummy_train_state() -> dict[str, torch.Tensor]:
    """Use for train_state.pt file, and latest_train_state.pt file in mbridge checkpoint."""
    return {
        "step": torch.tensor(1, dtype=torch.int32),
        "consumed_train_samples": torch.tensor(0, dtype=torch.int32),
        "skipped_train_samples": torch.tensor(0, dtype=torch.int32),
        "consumed_valid_samples": torch.tensor(0, dtype=torch.int32),
        "floating_point_operations_so_far": torch.tensor(0, dtype=torch.float64),
        "do_train": torch.tensor(True, dtype=torch.bool),
        "do_valid": torch.tensor(True, dtype=torch.bool),
        "do_test": torch.tensor(True, dtype=torch.bool),
    }


def _dummy_common_pt_dict() -> dict[str, Any]:
    """Use for common.pt file in mbridge checkpoint."""
    return {
        "checkpoint_version": 3.0,
        "iteration": 1,
        "optimizer": {"param_state_sharding_type": "dp_reshardable"},
        "opt_param_scheduler": {
            "max_lr": 0.0003,
            "lr_warmup_steps": 10,
            "num_steps": 2560,
            "lr_decay_style": "cosine",
            "lr_decay_steps": 25600,
            "min_lr": 3e-05,
            "start_wd": 0.01,
            "end_wd": 0.01,
            "wd_incr_style": "constant",
            "wd_incr_steps": 3072,
        },
        "content_metadata": {
            "singleton_local_shards": False,
            "distrib_optim_sharding_type": "dp_reshardable",
            "chained_optim_avoid_prefix": True,
        },
    }


def _dummy_format_metadata() -> dict[str, Any]:
    """Use for metadata.json file in mbridge checkpoint."""
    return {
        "sharded_backend": "torch_dist",
        "sharded_backend_version": 1,
        "common_backend": "torch",
        "common_backend_version": 1,
    }


def nemo2_to_mbridge(
    nemo2_ckpt_dir: Path,
    tokenizer_path: Path,
    mbridge_ckpt_dir: Path,
    model_provider: EdenModelProvider,
    mixed_precision_recipe: str,
) -> Path:
    """Convert a Nemo2 checkpoint to a Megatron Bridge checkpoint.

    Args:
        nemo2_ckpt_dir: Path to the Nemo2 checkpoint directory.
        tokenizer_path: Path to the tokenizer directory.
        mbridge_ckpt_dir: Path to the Megatron Bridge checkpoint directory.
        model_provider: Model provider to use for the model.
        mixed_precision_recipe: Mixed precision recipe to use for the model.

    Returns:
        Path to the Megatron Bridge checkpoint directory.

    Structure of a megatron bridge checkpoint:
    <mbridge_ckpt_dir>
    |-- latest_checkpointed_iteration.txt # the older megatron way of communicating the latest checkpointed iteration
    |-- latest_train_state.pt # a copy of train_state.pt from the latest iteration, used by megatron bridge
    ├── iter_0000001
    |   ├── __*_*.distcp  # distcp checkpoint files for each shard (sometiems rank sometimes arbitrary shards)
    |   ├── .metadata  # metadata for the distcp checkpoint files
    |   ├── common.pt  # common metadata (training configuration related)
    |   ├── metadata.json  # metadata for the checkpoint format etc
    |   ├── run_config.yaml  # training configuration
    |   ├── tokenizer  # tokenizer assets
    |   ├── train_state.pt  # training state, eg current step, etc.
    """
    assert not mbridge_ckpt_dir.exists(), f"Checkpoint directory {mbridge_ckpt_dir} already exists"
    mbridge_ckpt_dir.mkdir(parents=True, exist_ok=True)
    mbridge_ckpt_iter_dir = mbridge_ckpt_dir / "iter_0000001"
    nemo2_model_path = nemo2_ckpt_dir / "weights"
    convert_nemo2_dcp_to_megatron(nemo2_model_path, mbridge_ckpt_iter_dir)
    assert mbridge_ckpt_iter_dir.exists(), f"Checkpoint directory {mbridge_ckpt_iter_dir} does not exist"
    with open(mbridge_ckpt_dir / "latest_checkpointed_iteration.txt", "w") as f:
        f.write("1\n")
    train_state = _dummy_train_state()
    torch.save(train_state, mbridge_ckpt_iter_dir / "train_state.pt")
    torch.save(train_state, mbridge_ckpt_dir / "latest_train_state.pt")

    common_pt_dict = _dummy_common_pt_dict()
    torch.save(common_pt_dict, mbridge_ckpt_iter_dir / "common.pt")
    format_metadata = _dummy_format_metadata()
    with open(mbridge_ckpt_iter_dir / "metadata.json", "w") as f:
        json.dump(format_metadata, f)
    config_container: ConfigContainer = eden_pretrain_config(
        precision_config=mixed_precision_recipe, hf_tokenizer_model_or_path=tokenizer_path, mock=True
    )
    tokenizer = build_tokenizer(config_container.tokenizer)
    model_provider.vocab_size = tokenizer.vocab_size
    config_container.model = model_provider
    config_container.to_yaml(str(mbridge_ckpt_iter_dir / "run_config.yaml"))
    save_tokenizer_assets(tokenizer, config_container.tokenizer, str(mbridge_ckpt_iter_dir))
    return mbridge_ckpt_dir


def run_nemo2_to_mbridge(
    nemo2_ckpt_dir: Path,
    tokenizer_path: Path,
    mbridge_ckpt_dir: Path,
    model_size: str,
    seq_length: int,
    mixed_precision_recipe: str,
) -> Path:
    """Convert a Nemo2 checkpoint to a Megatron Bridge checkpoint.

    Args:
        nemo2_ckpt_dir: Path to the Nemo2 checkpoint directory.
        tokenizer_path: Path to the tokenizer directory.
        mbridge_ckpt_dir: Path to the Megatron Bridge checkpoint directory.
        model_size: Model size to use for the model.
        seq_length: Sequence length to use for the model.
        mixed_precision_recipe: Mixed precision recipe to use for the model.

    Returns:
        Path to the Megatron Bridge checkpoint directory.
    """
    model_provider = EDEN_MODEL_OPTIONS[model_size](seq_length=seq_length)
    res_dir = nemo2_to_mbridge(
        nemo2_ckpt_dir, tokenizer_path, mbridge_ckpt_dir, model_provider, mixed_precision_recipe
    )
    logger.info(f"Megatron Bridge checkpoint saved to {res_dir}")
    return res_dir


def main():
    """Main function for handling cli args and running the conversion."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--nemo2-ckpt-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--mbridge-ckpt-dir", type=Path, required=True)
    parser.add_argument("--model-size", type=str, choices=sorted(EDEN_MODEL_OPTIONS.keys()), required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument(
        "--mixed-precision-recipe",
        type=str,
        choices=list(MIXED_PRECISION_RECIPES.keys()),
        default="bf16_mixed",
        help="Mixed precision recipe to use for training.",
    )
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    run_nemo2_to_mbridge(
        args.nemo2_ckpt_dir,
        args.tokenizer_path,
        args.mbridge_ckpt_dir,
        args.model_size,
        args.seq_length,
        args.mixed_precision_recipe,
    )


if __name__ == "__main__":
    main()
