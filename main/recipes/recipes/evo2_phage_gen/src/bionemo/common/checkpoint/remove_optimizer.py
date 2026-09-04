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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: recipes/evo2_megatron/src/bionemo/common/checkpoint/remove_optimizer.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

"""Remove optimizer state from an MBridge DCP checkpoint.

Reads a Megatron Bridge checkpoint (which may contain model state, optimizer
state, LR scheduler state, and RNG state), strips non-model training state,
and writes a new checkpoint. By default the existing weights-only behavior
also omits serialized model objects. Callers that need a schema-complete native
MBridge checkpoint may opt in to preserving objects such as Transformer Engine
``_extra_state``.

This module depends only on PyTorch and the standard library -- it must NOT
import megatron, nemo, or mbridge.
"""

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, DefaultSavePlanner, FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import BytesStorageMetadata


logger = logging.getLogger(__name__)

_STRIP_KEY_PREFIXES = ("optimizer.", "opt_param_scheduler.", "rng_state")


def _resolve_iter_dir(ckpt_dir: Path) -> Path:
    """Resolve the iter_NNNNNNN directory inside a checkpoint root."""
    if re.match(r"^iter_\d+$", ckpt_dir.name):
        return ckpt_dir
    latest_file = ckpt_dir / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        iteration = latest_file.read_text().strip()
        return ckpt_dir / f"iter_{int(iteration):07d}"
    iter_dirs = sorted(ckpt_dir.glob("iter_*"))
    if not iter_dirs:
        raise FileNotFoundError(f"No iter_* directories in {ckpt_dir}")
    return iter_dirs[-1]


def _is_optimizer_key(key: str) -> bool:
    """Return True if *key* belongs to optimizer, scheduler, or RNG state."""
    return any(key.startswith(prefix) for prefix in _STRIP_KEY_PREFIXES)


def remove_optimizer(
    src_ckpt_dir: Path,
    dst_ckpt_dir: Path,
    *,
    preserve_model_object_state: bool = False,
) -> Path:
    """Strip optimizer / scheduler / RNG state from an MBridge checkpoint.

    Args:
        src_ckpt_dir: Source checkpoint root (contains ``iter_NNNNNNN/``).
        dst_ckpt_dir: Destination directory.  Must not already exist.
        preserve_model_object_state: Retain non-optimizer DCP object shards such as
            Transformer Engine ``_extra_state``. Defaults to ``False`` to preserve
            the reducer's historical weights-only behavior.

    Returns:
        Path to the destination checkpoint directory.
    """
    if dst_ckpt_dir.exists():
        raise FileExistsError(f"Destination already exists: {dst_ckpt_dir}")

    src_iter_dir = _resolve_iter_dir(src_ckpt_dir)
    logger.info(f"Source iter directory: {src_iter_dir}")

    # Determine the iteration name so the destination mirrors the structure.
    iter_name = src_iter_dir.name
    dst_iter_dir = dst_ckpt_dir / iter_name
    dst_iter_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load only model state from DCP ---
    reader = FileSystemReader(str(src_iter_dir))
    metadata = reader.read_metadata()

    state_dict: dict[str, Any] = {}
    model_object_keys: set[str] = set()
    skipped_keys: list[str] = []
    for key, item_meta in metadata.state_dict_metadata.items():
        if _is_optimizer_key(key):
            skipped_keys.append(key)
        elif isinstance(item_meta, BytesStorageMetadata):
            if preserve_model_object_state:
                # DCP replaces this metadata placeholder with the deserialized object during load.
                state_dict[key] = item_meta
                model_object_keys.add(key)
            else:
                skipped_keys.append(key)
        else:
            state_dict[key] = torch.empty(item_meta.size, dtype=item_meta.properties.dtype, device="cpu")

    retained_kind = "model-state" if preserve_model_object_state else "model-weight"
    logger.info(f"Loading {len(state_dict)} {retained_kind} keys (skipping {len(skipped_keys)} other keys)")
    if skipped_keys:
        logger.debug(f"Skipped keys (first 20): {skipped_keys[:20]}")

    # The mapping is deliberately flat and uses the source metadata FQNs as exact
    # keys. Disabling planner flattening keeps serialized objects such as TE
    # ``_extra_state`` under those keys even when an object contains tensors.
    dcp.load(
        state_dict=state_dict,
        storage_reader=reader,
        planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )

    # --- 2. Save model-only state dict to destination ---
    writer = FileSystemWriter(str(dst_iter_dir), single_file_per_rank=False, thread_count=os.cpu_count())
    dcp.save(
        state_dict=state_dict,
        storage_writer=writer,
        planner=DefaultSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )
    expected_keys = set(state_dict)
    del state_dict

    written_metadata = FileSystemReader(str(dst_iter_dir)).read_metadata().state_dict_metadata
    if set(written_metadata) != expected_keys:
        missing = sorted(expected_keys - set(written_metadata))
        unexpected = sorted(set(written_metadata) - expected_keys)
        raise RuntimeError(f"Model-only checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    changed_object_keys = [
        key for key in model_object_keys if not isinstance(written_metadata[key], BytesStorageMetadata)
    ]
    if changed_object_keys:
        raise RuntimeError(f"Model object state was not preserved as object storage: {sorted(changed_object_keys)}")

    # --- 3. Copy non-DCP artefacts (run_config, tokenizer, train_state, etc.) ---
    for name in ("run_config.yaml", "train_state.pt"):
        src_file = src_iter_dir / name
        if src_file.exists():
            shutil.copy2(src_file, dst_iter_dir / name)

    tokenizer_src = src_iter_dir / "tokenizer"
    if tokenizer_src.is_dir():
        shutil.copytree(tokenizer_src, dst_iter_dir / "tokenizer")

    # --- 4. Write metadata.json (same format descriptor) ---
    src_meta_json = src_iter_dir / "metadata.json"
    if src_meta_json.exists():
        shutil.copy2(src_meta_json, dst_iter_dir / "metadata.json")
    else:
        with open(dst_iter_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "sharded_backend": "torch_dist",
                    "sharded_backend_version": 1,
                    "common_backend": "torch",
                    "common_backend_version": 1,
                },
                f,
            )

    # --- 5. Write common.pt without optimizer metadata ---
    src_common = src_iter_dir / "common.pt"
    if src_common.exists():
        common = torch.load(src_common, map_location="cpu", weights_only=False)
        common.pop("optimizer", None)
        common.pop("opt_param_scheduler", None)
        if "content_metadata" in common:
            common["content_metadata"].pop("distrib_optim_sharding_type", None)
        torch.save(common, dst_iter_dir / "common.pt")

    # --- 6. Write top-level files ---
    src_latest = src_ckpt_dir / "latest_checkpointed_iteration.txt"
    if src_latest.exists():
        shutil.copy2(src_latest, dst_ckpt_dir / "latest_checkpointed_iteration.txt")

    src_train_state = src_ckpt_dir / "latest_train_state.pt"
    if src_train_state.exists():
        shutil.copy2(src_train_state, dst_ckpt_dir / "latest_train_state.pt")

    logger.info(f"Wrote optimizer-free checkpoint to {dst_ckpt_dir}")
    return dst_ckpt_dir


def main():
    """CLI entry point for removing optimizer state from an MBridge checkpoint."""
    parser = argparse.ArgumentParser(
        description="Remove optimizer state from a Megatron Bridge DCP checkpoint, "
        "producing a smaller weights-only checkpoint by default."
    )
    parser.add_argument(
        "--src-ckpt-dir",
        type=Path,
        required=True,
        help="Source checkpoint directory (containing iter_NNNNNNN/)",
    )
    parser.add_argument(
        "--dst-ckpt-dir",
        type=Path,
        required=True,
        help="Destination directory for the reduced checkpoint",
    )
    parser.add_argument(
        "--preserve-model-object-state",
        action="store_true",
        help="Retain non-optimizer serialized model objects such as Transformer Engine _extra_state",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    remove_optimizer(
        args.src_ckpt_dir,
        args.dst_ckpt_dir,
        preserve_model_object_state=args.preserve_model_object_state,
    )


if __name__ == "__main__":
    main()
