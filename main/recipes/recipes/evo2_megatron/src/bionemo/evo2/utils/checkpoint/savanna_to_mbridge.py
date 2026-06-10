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

"""Savanna -> MBridge checkpoint converter.

Converts ARC's Savanna .pt checkpoint format directly to Megatron Bridge
DCP format, bypassing the NeMo2 intermediate step.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import huggingface_hub.errors
import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import hf_hub_download
from megatron.bridge.training.checkpointing import save_tokenizer_assets
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MIXED_PRECISION_RECIPES
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from torch.distributed.checkpoint import FileSystemWriter

from bionemo.evo2.models.evo2_provider import HYENA_MODEL_OPTIONS, HyenaModelProvider
from bionemo.evo2.recipes.evo2 import evo2_1b_pretrain_config as pretrain_config


logger = logging.getLogger(__name__)


def _download_shards(repo_id: str, weights_filename: str, download_dir: str, revision: str | None = None) -> list[str]:
    """Download multi-part checkpoint shards from HuggingFace."""
    parts = []
    part_num = 0
    while True:
        try:
            part_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{weights_filename}.part{part_num}",
                local_dir=download_dir,
                revision=revision,
            )
            parts.append(part_path)
            part_num += 1
        except huggingface_hub.errors.EntryNotFoundError:  # noqa: PERF203
            break
    return parts


def _cleanup_parts(parts: list[str]) -> None:
    """Remove downloaded shard parts after joining."""
    for part in parts:
        try:
            os.remove(part)
        except OSError:  # noqa: PERF203
            pass


def download_savanna_checkpoint(repo_id: str, cache_dir: Path | None = None, revision: str | None = None) -> Path:
    """Download a Savanna checkpoint from HuggingFace Hub.

    Handles both single-file and multi-part (sharded) checkpoints.

    Args:
        repo_id: HuggingFace repo ID (e.g. 'arcinstitute/savanna_evo2_1b_base').
        cache_dir: Optional directory to cache downloads in.
        revision: HuggingFace revision to use for the savanna checkpoint.

    Returns:
        Path to the downloaded .pt file.
    """
    modelname = repo_id.split("/")[-1]
    weights_filename = f"{modelname}.pt"
    download_dir = str(cache_dir) if cache_dir else None

    try:
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=weights_filename,
            local_dir=download_dir,
            revision=revision,
        )
        return Path(weights_path)
    except Exception:
        logger.warning(f"Single-file download failed for {repo_id}, trying multi-part shards...")
        if download_dir is None:
            download_dir = str(Path.home() / ".cache" / "savanna_checkpoints" / repo_id)
        final_path = Path(download_dir) / weights_filename
        if final_path.exists():
            return final_path

        parts = _download_shards(repo_id, weights_filename, download_dir, revision)

        if not parts:
            raise FileNotFoundError(f"No checkpoint files found in {repo_id}")

        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, "wb") as outfile:
            for part in parts:
                with open(part, "rb") as infile:
                    while True:
                        chunk = infile.read(8192 * 1024)
                        if not chunk:
                            break
                        outfile.write(chunk)

        _cleanup_parts(parts)

        return final_path


def load_savanna_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a Savanna checkpoint and strip module/sequential prefixes.

    Uses mmap=True to avoid loading all weights into RAM at once;
    tensors are paged in from disk on demand.

    Args:
        path: Path to the .pt checkpoint file.

    Returns:
        Flat state dict with keys like 'sequential.{i}.xxx'.
    """
    raw = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    if "module" in raw:
        raw = raw["module"]

    state_dict = {}
    for k in list(raw.keys()):
        v = raw.pop(k)
        key = k.removeprefix("module.")
        state_dict[key] = v

    return state_dict


_FP32_SUFFIXES = frozenset({"h", "decay", "gamma", "R", "p"})
"""Parameter name suffixes that stay in float32; everything else is cast to ``params_dtype``.

These correspond to Hyena filter parameters that are initialised with
``dtype=torch.float32`` in the model (``ImplicitModalFilter``,
``ExplicitSingleDecayFilter``).  The old Savanna→NeMo converter preserves
the same split; matching it here ensures the DCP checkpoint dtypes align
with what the model's ``sharded_state_dict()`` declares.
"""


def savanna_to_mbridge_state_dict(
    savanna_state_dict: dict[str, torch.Tensor],
    hybrid_override_pattern: str,
    te_enabled: bool = True,
    medium_conv_len: int = 128,
    params_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Convert Savanna state dict keys to MBridge (Megatron) format.

    The Savanna checkpoint uses keys like:
      sequential.0.word_embeddings.weight           (embedding)
      sequential.{i+2}.{layer_keys}                 (layer i; index 1 is a no-param lambda)
      sequential.{N+3}.norm.weight                  (final norm; N+2 is another lambda)

    MBridge uses:
      embedding.word_embeddings.weight
      decoder.layers.{i}.{layer_keys}
      decoder.final_norm.weight

    Args:
        savanna_state_dict: State dict loaded from Savanna checkpoint.
        hybrid_override_pattern: The layer pattern string (e.g. "SDH*SDH...").
        te_enabled: Whether TransformerEngine fused layernorm is used.
        medium_conv_len: Truncation length for medium hyena filter (filter.h / filter.decay).
            Matches NeMo's ``HyenaConfig.hyena_medium_conv_len`` (default 128).
        params_dtype: Target dtype for most parameters (default bfloat16).
            Filter parameters whose suffix is in ``_FP32_SUFFIXES`` stay float32.

    Returns:
        State dict with MBridge-style keys.
    """
    num_layers = len(hybrid_override_pattern)
    mapping = {}

    mapping["sequential.0.word_embeddings.weight"] = "embedding.word_embeddings.weight"
    # Savanna sequential layout varies by model:
    #   40b: 0=embedding, 1=lambda, 2..N+1=layers, N+2=lambda, N+3=final_norm
    #   20b: 0=embedding, 1=lambda, 2..N+1=layers, N+2=final_norm  (no second lambda)
    # Auto-detect by checking which index contains the final norm.
    norm_key_plus3 = f"sequential.{num_layers + 3}.norm.weight"
    norm_key_plus2 = f"sequential.{num_layers + 2}.norm.weight"
    if norm_key_plus3 in savanna_state_dict:
        mapping[norm_key_plus3] = "decoder.final_norm.weight"
    elif norm_key_plus2 in savanna_state_dict:
        mapping[norm_key_plus2] = "decoder.final_norm.weight"
    else:
        raise KeyError(
            f"Cannot find final norm weight at sequential.{num_layers + 3} or sequential.{num_layers + 2}. "
            f"Check that hybrid_override_pattern length ({num_layers}) matches the checkpoint."
        )

    for i, symbol in enumerate(hybrid_override_pattern):
        src_idx = i + 2

        if te_enabled:
            mapping[f"sequential.{src_idx}.pre_mlp_layernorm.weight"] = (
                f"decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight"
            )
        else:
            mapping[f"sequential.{src_idx}.pre_mlp_layernorm.weight"] = f"decoder.layers.{i}.pre_mlp_layernorm.weight"
        mapping[f"sequential.{src_idx}.mlp.w3.weight"] = f"decoder.layers.{i}.mlp.linear_fc2.weight"

        if symbol != "*":
            if te_enabled:
                mapping[f"sequential.{src_idx}.input_layernorm.weight"] = (
                    f"decoder.layers.{i}.mixer.dense_projection.layer_norm_weight"
                )
            else:
                mapping[f"sequential.{src_idx}.input_layernorm.weight"] = f"decoder.layers.{i}.norm.weight"

            mapping[f"sequential.{src_idx}.mixer.dense_projection.weight"] = (
                f"decoder.layers.{i}.mixer.dense_projection.weight"
            )
            mapping[f"sequential.{src_idx}.mixer.hyena_proj_conv.short_conv_weight"] = (
                f"decoder.layers.{i}.mixer.hyena_proj_conv.short_conv_weight"
            )
            mapping[f"sequential.{src_idx}.mixer.dense.weight"] = f"decoder.layers.{i}.mixer.dense.weight"
            mapping[f"sequential.{src_idx}.mixer.dense.bias"] = f"decoder.layers.{i}.mixer.dense.bias"

            if symbol == "S":
                mapping[f"sequential.{src_idx}.mixer.mixer.short_conv.short_conv_weight"] = (
                    f"decoder.layers.{i}.mixer.mixer.short_conv.short_conv_weight"
                )
            elif symbol == "D":
                mapping[f"sequential.{src_idx}.mixer.mixer.conv_bias"] = f"decoder.layers.{i}.mixer.mixer.conv_bias"
                mapping[f"sequential.{src_idx}.mixer.mixer.filter.h"] = f"decoder.layers.{i}.mixer.mixer.filter.h"
                mapping[f"sequential.{src_idx}.mixer.mixer.filter.decay"] = (
                    f"decoder.layers.{i}.mixer.mixer.filter.decay"
                )
            elif symbol == "H":
                mapping[f"sequential.{src_idx}.mixer.mixer.conv_bias"] = f"decoder.layers.{i}.mixer.mixer.conv_bias"
                mapping[f"sequential.{src_idx}.mixer.mixer.filter.gamma"] = (
                    f"decoder.layers.{i}.mixer.mixer.filter.gamma"
                )
                mapping[f"sequential.{src_idx}.mixer.mixer.filter.R"] = f"decoder.layers.{i}.mixer.mixer.filter.R"
                mapping[f"sequential.{src_idx}.mixer.mixer.filter.p"] = f"decoder.layers.{i}.mixer.mixer.filter.p"

        elif symbol == "*":
            if te_enabled:
                mapping[f"sequential.{src_idx}.input_layernorm.weight"] = (
                    f"decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight"
                )
            else:
                mapping[f"sequential.{src_idx}.input_layernorm.weight"] = f"decoder.layers.{i}.input_layernorm.weight"

            mapping[f"sequential.{src_idx}.mixer.dense_projection.weight"] = (
                f"decoder.layers.{i}.self_attention.linear_qkv.weight"
            )
            mapping[f"sequential.{src_idx}.mixer.dense.weight"] = (
                f"decoder.layers.{i}.self_attention.linear_proj.weight"
            )
            mapping[f"sequential.{src_idx}.mixer.dense.bias"] = f"decoder.layers.{i}.self_attention.linear_proj.bias"

    mbridge_state_dict = {}

    for savanna_key, mbridge_key in mapping.items():
        if savanna_key in savanna_state_dict:
            t = savanna_state_dict.pop(savanna_key).clone()
            if "filter.h" in mbridge_key or "filter.decay" in mbridge_key:
                t = t[:, :medium_conv_len]
            mbridge_state_dict[mbridge_key] = t

    for i, symbol in enumerate(hybrid_override_pattern):
        src_idx = i + 2
        w1_key = f"sequential.{src_idx}.mlp.w1.weight"
        w2_key = f"sequential.{src_idx}.mlp.w2.weight"
        fc1_key = f"decoder.layers.{i}.mlp.linear_fc1.weight"

        if w1_key in savanna_state_dict and w2_key in savanna_state_dict:
            w1 = savanna_state_dict.pop(w1_key)
            w2 = savanna_state_dict.pop(w2_key)
            mbridge_state_dict[fc1_key] = torch.cat([w1, w2], dim=0)
            del w1, w2

    unmapped = {k for k in savanna_state_dict if "_extra_state" not in k}
    if unmapped:
        logger.warning(f"Unmapped savanna keys ({len(unmapped)}): {sorted(unmapped)[:20]}")

    for key in mbridge_state_dict:
        suffix = key.rsplit(".", 1)[-1]
        target_dtype = torch.float32 if suffix in _FP32_SUFFIXES else params_dtype
        if mbridge_state_dict[key].dtype != target_dtype:
            mbridge_state_dict[key] = mbridge_state_dict[key].to(target_dtype)

    return mbridge_state_dict


def _dummy_train_state() -> dict[str, torch.Tensor]:
    """Dummy train state for mbridge checkpoint metadata."""
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


def package_mbridge_checkpoint(
    state_dict: dict[str, torch.Tensor],
    mbridge_ckpt_dir: Path,
    model_provider: HyenaModelProvider,
    tokenizer_path: Path,
    mixed_precision_recipe: str = "bf16_mixed",
) -> Path:
    """Package a state dict into the full mbridge checkpoint directory structure.

    Args:
        state_dict: The converted state dict with mbridge-style keys.
        mbridge_ckpt_dir: Output directory for the mbridge checkpoint.
        model_provider: The model provider instance with correct hyperparameters.
        tokenizer_path: Path to the HF tokenizer.
        mixed_precision_recipe: Mixed precision recipe name.

    Returns:
        Path to the mbridge checkpoint directory.
    """
    mbridge_ckpt_dir.mkdir(parents=True, exist_ok=True)
    iter_dir = mbridge_ckpt_dir / "iter_0000001"
    iter_dir.mkdir(parents=True, exist_ok=True)

    writer = FileSystemWriter(
        str(iter_dir),
        single_file_per_rank=False,
        thread_count=os.cpu_count(),
    )
    dcp.save(state_dict=state_dict, storage_writer=writer, no_dist=True)

    with open(mbridge_ckpt_dir / "latest_checkpointed_iteration.txt", "w") as f:
        f.write("1\n")

    train_state = _dummy_train_state()
    torch.save(train_state, iter_dir / "train_state.pt")
    torch.save(train_state, mbridge_ckpt_dir / "latest_train_state.pt")

    torch.save(
        {
            "checkpoint_version": 3.0,
            "iteration": 1,
            "optimizer": {"param_state_sharding_type": "dp_reshardable"},
            "opt_param_scheduler": {},
            "content_metadata": {
                "singleton_local_shards": False,
                "distrib_optim_sharding_type": "dp_reshardable",
                "chained_optim_avoid_prefix": True,
            },
        },
        iter_dir / "common.pt",
    )

    with open(iter_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "sharded_backend": "torch_dist",
                "sharded_backend_version": 1,
                "common_backend": "torch",
                "common_backend_version": 1,
            },
            f,
        )

    config_container: ConfigContainer = pretrain_config(
        precision_config=mixed_precision_recipe,
        hf_tokenizer_model_or_path=tokenizer_path,
        mock=True,
    )
    tokenizer = build_tokenizer(config_container.tokenizer)
    model_provider.vocab_size = tokenizer.vocab_size
    config_container.model = model_provider
    config_container.to_yaml(str(iter_dir / "run_config.yaml"))
    save_tokenizer_assets(tokenizer, config_container.tokenizer, str(iter_dir))

    return mbridge_ckpt_dir


def savanna_to_mbridge(
    savanna_ckpt_path: Path | str,
    mbridge_ckpt_dir: Path,
    model_size: str,
    tokenizer_path: Path,
    seq_length: int | None = None,
    te_enabled: bool = True,
    mixed_precision_recipe: str = "bf16_mixed",
    revision: str | None = None,
) -> Path:
    """Convert a Savanna checkpoint to MBridge format end-to-end.

    Args:
        savanna_ckpt_path: Path to savanna .pt file, or HuggingFace repo ID.
        mbridge_ckpt_dir: Output directory for the mbridge checkpoint.
        model_size: Model size key (e.g. 'evo2_1b_base', 'evo2_7b').
        tokenizer_path: Path to the HF tokenizer.
        seq_length: Override sequence length (uses provider default if None).
        te_enabled: Whether TE fused layernorm keys are used.
        mixed_precision_recipe: Mixed precision recipe name.
        revision: HuggingFace revision to use for the savanna checkpoint.

    Returns:
        Path to the mbridge checkpoint directory.
    """
    savanna_path = Path(savanna_ckpt_path)
    if not savanna_path.exists():
        logger.info(f"Path {savanna_ckpt_path} not found locally, treating as HF repo ID...")
        savanna_path = download_savanna_checkpoint(str(savanna_ckpt_path), revision=revision)

    logger.info(f"Loading savanna checkpoint from {savanna_path}...")
    savanna_sd = load_savanna_state_dict(savanna_path)

    provider_cls = HYENA_MODEL_OPTIONS[model_size]
    kwargs = {}
    if seq_length is not None:
        kwargs["seq_length"] = seq_length
    model_provider = provider_cls(**kwargs)
    pattern = model_provider.hybrid_override_pattern

    medium_conv_len = getattr(model_provider, "hyena_medium_conv_len", 128)
    params_dtype = getattr(model_provider, "params_dtype", torch.bfloat16)
    logger.info(
        f"Converting with pattern={pattern}, te_enabled={te_enabled}, "
        f"medium_conv_len={medium_conv_len}, params_dtype={params_dtype}"
    )
    mbridge_sd = savanna_to_mbridge_state_dict(
        savanna_sd, pattern, te_enabled=te_enabled, medium_conv_len=medium_conv_len, params_dtype=params_dtype
    )
    del savanna_sd
    logger.info(f"Converted {len(mbridge_sd)} keys")

    result = package_mbridge_checkpoint(
        mbridge_sd, mbridge_ckpt_dir, model_provider, tokenizer_path, mixed_precision_recipe
    )
    logger.info(f"MBridge checkpoint saved to {result}")
    return result


def main():
    """CLI entry point for savanna-to-mbridge conversion."""
    parser = argparse.ArgumentParser(description="Convert Savanna checkpoint to MBridge format")
    parser.add_argument(
        "--savanna-ckpt-path",
        type=str,
        required=True,
        help="Path to savanna .pt file or HuggingFace repo ID (e.g. arcinstitute/savanna_evo2_1b_base)",
    )
    parser.add_argument("--mbridge-ckpt-dir", type=Path, required=True)
    parser.add_argument("--model-size", type=str, choices=sorted(HYENA_MODEL_OPTIONS.keys()), required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--no-te", action="store_true", help="Disable TE fused layernorm key mapping")
    parser.add_argument(
        "--mixed-precision-recipe",
        type=str,
        choices=list(MIXED_PRECISION_RECIPES.keys()),
        default="bf16_mixed",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HuggingFace revision to use for the savanna checkpoint. It is STRONGLY encouraged to use a specific "
        "revision to ensure reproducibility and security. It is possible that a checkpoint on huggingface could be "
        "compromised with malware, so providing a revision (commit SHA) is strongly recommended. If no revision is "
        "provided, the latest commit will be used.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    savanna_to_mbridge(
        savanna_ckpt_path=args.savanna_ckpt_path,
        mbridge_ckpt_dir=args.mbridge_ckpt_dir,
        model_size=args.model_size,
        tokenizer_path=args.tokenizer_path,
        seq_length=args.seq_length,
        te_enabled=not args.no_te,
        mixed_precision_recipe=args.mixed_precision_recipe,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
