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

"""Streaming MBridge DCP to native Evo2 vLLM safetensors export."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import yaml
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

from bionemo.evo2.vllm.config import Evo2Config


_ACTIVATION_TARGETS = {
    "torch._C._nn.gelu": "gelu",
    "torch.nn.functional.gelu": "gelu",
    "torch.nn.functional.silu": "silu",
}


def resolve_iteration_dir(checkpoint_path: Path | str) -> Path:
    """Resolve a checkpoint root or explicit ``iter_XXXXXXX`` directory."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_path.name.startswith("iter_") and checkpoint_path.is_dir():
        return checkpoint_path
    latest_path = checkpoint_path / "latest_checkpointed_iteration.txt"
    if latest_path.exists():
        iteration = int(latest_path.read_text().strip())
        iteration_dir = checkpoint_path / f"iter_{iteration:07d}"
        if iteration_dir.is_dir():
            return iteration_dir
    iteration_dirs = sorted(path for path in checkpoint_path.glob("iter_*") if path.is_dir())
    if not iteration_dirs:
        raise FileNotFoundError(f"no iter_* checkpoint directories found under {checkpoint_path}")
    return iteration_dirs[-1]


def infer_evo2_config(checkpoint_path: Path | str) -> tuple[Evo2Config, str]:
    """Construct an export config from the MBridge checkpoint run configuration."""
    iteration_dir = resolve_iteration_dir(checkpoint_path)
    run_config_path = iteration_dir / "run_config.yaml"
    if not run_config_path.exists():
        raise FileNotFoundError(f"checkpoint run configuration is missing: {run_config_path}")
    run_config = yaml.safe_load(run_config_path.read_text())
    model = run_config.get("model", {})
    provider = str(model.get("_target_", "unknown"))
    required = (
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "ffn_hidden_size",
        "vocab_size",
        "seq_length",
        "hybrid_override_pattern",
        "activation_func",
        "gated_linear_unit",
        "remove_activation_post_first_layer",
    )
    missing = [name for name in required if model.get(name) is None]
    if missing:
        raise ValueError(f"checkpoint run config is missing Evo2 model fields: {missing}")

    activation_config = model["activation_func"]
    activation_target = (
        str(activation_config.get("_target_", "")) if isinstance(activation_config, dict) else str(activation_config)
    )
    hidden_act = _ACTIVATION_TARGETS.get(activation_target)
    if hidden_act is None:
        raise ValueError(f"unsupported Evo2 activation_func target: {activation_target}")

    config = Evo2Config(
        hidden_size=int(model["hidden_size"]),
        num_hidden_layers=int(model["num_layers"]),
        num_attention_heads=int(model["num_attention_heads"]),
        num_key_value_heads=int(model.get("num_query_groups") or model["num_attention_heads"]),
        intermediate_size=int(model["ffn_hidden_size"]),
        vocab_size=int(model["vocab_size"]),
        max_position_embeddings=int(model["seq_length"]),
        hybrid_override_pattern=str(model["hybrid_override_pattern"]),
        num_groups_hyena=int(model.get("num_groups_hyena") or model["hidden_size"]),
        num_groups_hyena_short=int(model.get("num_groups_hyena_short") or 256),
        num_groups_hyena_medium=int(model.get("num_groups_hyena_medium") or 256),
        rms_norm_eps=float(model.get("layernorm_epsilon") or 1e-6),
        rotary_base=float(model.get("rotary_base") or 10000.0),
        seq_len_interpolation_factor=(
            None if model.get("seq_len_interpolation_factor") is None else float(model["seq_len_interpolation_factor"])
        ),
        hidden_act=hidden_act,
        gelu_approximate="none",
        gated_linear_unit=bool(model["gated_linear_unit"]),
        remove_activation_post_first_layer=bool(model["remove_activation_post_first_layer"]),
        torch_dtype="bfloat16",
    )
    return config, provider


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=parent,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
    return "unknown"


def _tensor_nbytes(metadata) -> int:
    return math.prod(metadata.size) * torch.empty((), dtype=metadata.properties.dtype).element_size()


def _plan_shards(tensor_metadata: dict[str, object], max_shard_size: int) -> list[list[str]]:
    if max_shard_size < 1:
        raise ValueError("max_shard_size must be positive")
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in sorted(tensor_metadata):
        tensor_size = _tensor_nbytes(tensor_metadata[name])
        if current and current_size + tensor_size > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += tensor_size
    if current:
        shards.append(current)
    return shards


def _load_one_tensor(reader: FileSystemReader, name: str, metadata) -> torch.Tensor:
    tensor = torch.empty(tuple(metadata.size), dtype=metadata.properties.dtype, device="cpu")
    dcp.load(state_dict={name: tensor}, storage_reader=reader, no_dist=True)
    return tensor


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def export_mbridge_to_vllm(
    checkpoint_path: Path | str,
    output_dir: Path | str,
    *,
    config: Evo2Config | None = None,
    max_shard_size: int = 2 * 1024**3,
) -> dict[str, object]:
    """Export one MBridge iteration without materializing the full model state dict."""
    iteration_dir = resolve_iteration_dir(checkpoint_path)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    inferred_config, model_provider = infer_evo2_config(iteration_dir)
    config = inferred_config if config is None else config
    config.save_pretrained(output_dir)

    reader = FileSystemReader(str(iteration_dir))
    checkpoint_metadata = reader.read_metadata()
    tensor_metadata = {
        name: metadata
        for name, metadata in checkpoint_metadata.state_dict_metadata.items()
        if not isinstance(metadata, BytesStorageMetadata)
    }
    if not tensor_metadata:
        raise ValueError(f"checkpoint contains no tensor weights: {iteration_dir}")
    shard_plan = _plan_shards(tensor_metadata, max_shard_size)
    shard_count = len(shard_plan)
    weight_map: dict[str, str] = {}
    shard_sizes = []

    for shard_index, names in enumerate(shard_plan, start=1):
        shard_name = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        tensors = {name: _load_one_tensor(reader, name, tensor_metadata[name]).contiguous() for name in names}
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})
        shard_sizes.append(sum(_tensor_nbytes(tensor_metadata[name]) for name in names))
        weight_map.update({name: shard_name for name in names})
        del tensors

    total_size = sum(_tensor_nbytes(metadata) for metadata in tensor_metadata.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path = output_dir / "model.safetensors.index.json"
    _write_json(index_path, index)

    tokenizer_source = iteration_dir / "tokenizer"
    if tokenizer_source.is_dir():
        shutil.copytree(tokenizer_source, output_dir / "tokenizer")

    config_path = output_dir / "config.json"
    run_config_path = iteration_dir / "run_config.yaml"
    largest_tensor_bytes = max(_tensor_nbytes(metadata) for metadata in tensor_metadata.values())
    peak_shard_bytes = max(shard_sizes)
    input_shard_sizes = [path.stat().st_size for path in iteration_dir.glob("*.distcp")]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_checkpoint": str(iteration_dir),
        "source_iteration": int(iteration_dir.name.removeprefix("iter_")),
        "model_provider": model_provider,
        "converter_git_commit": _git_commit(),
        "tensor_count": len(tensor_metadata),
        "total_size": total_size,
        "dtypes": sorted({str(metadata.properties.dtype) for metadata in tensor_metadata.values()}),
        "max_shard_size": max_shard_size,
        "peak_shard_bytes": peak_shard_bytes,
        "largest_tensor_bytes": largest_tensor_bytes,
        "largest_input_shard_bytes": max(input_shard_sizes, default=0),
        "estimated_peak_buffered_bytes": peak_shard_bytes + largest_tensor_bytes,
        "run_config_sha256": _sha256(run_config_path),
        "config_sha256": _sha256(config_path),
        "index_sha256": _sha256(index_path),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parse_size(value: str) -> int:
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    normalized = value.strip().lower()
    for suffix in sorted(units, key=len, reverse=True):
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)].strip()
            try:
                return int(float(number) * units[suffix])
            except ValueError as error:
                raise argparse.ArgumentTypeError(f"invalid size: {value}") from error
    try:
        return int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid size: {value}") from error


def main(argv: list[str] | None = None) -> int:
    """Run the MBridge-to-vLLM export CLI."""
    parser = argparse.ArgumentParser(description="Export an MBridge Evo2 checkpoint for vLLM")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, help="optional directory or config.json override")
    parser.add_argument("--max-shard-size", type=_parse_size, default=_parse_size("2GiB"))
    args = parser.parse_args(argv)
    config = Evo2Config.from_pretrained(args.config) if args.config is not None else None
    export_mbridge_to_vllm(args.checkpoint, args.output, config=config, max_shard_size=args.max_shard_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
