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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Prepare an SFT Megatron Bridge checkpoint for NeMo-RL.

The preparation preserves model weights, tokenizer assets, configuration, and checkpoint
metadata while removing optimizer, scheduler, RNG, and mutable training state. It also clears
serialized callbacks and timers that are owned by the SFT process so NeMo-RL can install its own
runtime state safely. The source checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)

PREPARATION_SCHEMA_VERSION = 2
MODEL_RUNTIME_FIELDS = (
    "no_sync_func",
    "grad_sync_func",
    "param_sync_func",
    "finalize_model_grads_func",
    "grad_scale_func",
    "timers",
)
OPTIMIZER_RUNTIME_FIELDS = ("timers",)
STRIPPED_DCP_KEY_PREFIXES = ("optimizer.", "opt_param_scheduler.", "rng_state")
_ITERATION_RE = re.compile(r"^iter_(\d+)$")


def remove_optimizer(
    source: Path,
    destination: Path,
    *,
    preserve_model_object_state: bool = False,
) -> Path:
    """Load the heavyweight DCP rewrite utility only when preparation starts."""
    from bionemo.common.checkpoint.remove_optimizer import remove_optimizer as rewrite_checkpoint

    return rewrite_checkpoint(
        source,
        destination,
        preserve_model_object_state=preserve_model_object_state,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_stats(root: Path) -> tuple[int, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def _resolve_iteration(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if _ITERATION_RE.fullmatch(checkpoint.name):
        iteration = checkpoint
    else:
        pointer = checkpoint / "latest_checkpointed_iteration.txt"
        if pointer.is_file():
            try:
                step = int(pointer.read_text().strip())
            except ValueError as error:
                raise ValueError(f"invalid checkpoint iteration in {pointer}") from error
            iteration = checkpoint / f"iter_{step:07d}"
        else:
            candidates = [
                path for path in checkpoint.glob("iter_*") if path.is_dir() and _ITERATION_RE.fullmatch(path.name)
            ]
            if not candidates:
                raise FileNotFoundError(f"no iter_* checkpoint directories under {checkpoint}")
            iteration = max(candidates, key=lambda path: int(path.name.removeprefix("iter_")))

    required = (iteration / ".metadata", iteration / "run_config.yaml")
    missing = [str(path) for path in required if not path.is_file()]
    if not list(iteration.glob("*.distcp")):
        missing.append(f"{iteration}/*.distcp")
    if missing:
        raise FileNotFoundError("incomplete Megatron Bridge checkpoint; missing " + ", ".join(missing))
    return iteration.resolve()


def _sanitize_run_config(path: Path) -> tuple[list[str], list[str]]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint run config must be a mapping: {path}")

    sanitized: dict[str, list[str]] = {}
    for section_name, fields in (
        ("model", MODEL_RUNTIME_FIELDS),
        ("optimizer", OPTIMIZER_RUNTIME_FIELDS),
    ):
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"checkpoint run config has no {section_name} mapping: {path}")
        present = [field for field in fields if field in section]
        for field in present:
            section[field] = None
        sanitized[section_name] = present

    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return sanitized["model"], sanitized["optimizer"]


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FileExistsError(f"existing prepared SFT checkpoint has no valid manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise FileExistsError(f"existing prepared SFT checkpoint manifest is not a mapping: {path}")
    return manifest


def _reuse_existing(source: Path, output: Path, source_facts: Mapping[str, Any]) -> Path | None:
    manifest_path = output / "preparation-manifest.json"
    manifest = _load_manifest(manifest_path)
    expected_checkpoint = (output / source.name).resolve()
    source_matches = all(
        (
            manifest.get("state") == "succeeded",
            manifest.get("source_sft_checkpoint") == str(source),
            manifest.get("source_run_config_sha256") == source_facts["run_config_sha256"],
            manifest.get("source_metadata_sha256") == source_facts["metadata_sha256"],
            manifest.get("source_payload_file_count") == source_facts["payload_file_count"],
            manifest.get("source_payload_bytes") == source_facts["payload_bytes"],
            manifest.get("prepared_sft_checkpoint") == str(expected_checkpoint),
        )
    )
    if manifest.get("schema_version") == 1 and source_matches:
        logger.warning(
            "Replacing schema-1 SFT checkpoint preparation at %s because it may omit serialized model object state",
            output,
        )
        return None

    matches_source = all(
        (
            manifest.get("schema_version") == PREPARATION_SCHEMA_VERSION,
            manifest.get("model_object_state_preserved") is True,
            source_matches,
        )
    )
    if not matches_source:
        raise FileExistsError(
            f"existing prepared SFT checkpoint {output} does not match the selected source checkpoint; "
            "inspect it and choose a different --output-dir or remove it explicitly"
        )

    try:
        _resolve_iteration(expected_checkpoint)
        file_count, payload_bytes = _tree_stats(expected_checkpoint)
        output_matches = (
            _sha256(expected_checkpoint / "run_config.yaml") == manifest.get("prepared_run_config_sha256")
            and file_count == manifest.get("payload_file_count")
            and payload_bytes == manifest.get("payload_bytes")
        )
    except (OSError, ValueError):
        output_matches = False
    if not output_matches:
        raise FileExistsError(
            f"existing prepared SFT checkpoint {output} is incomplete or changed; inspect it and remove it explicitly"
        )
    logger.info("Reusing prepared SFT checkpoint for RL at %s", expected_checkpoint)
    return expected_checkpoint


def _install_prepared_output(candidate: Path, output: Path, staging: Path, replace_existing: bool) -> None:
    """Install a completed preparation, atomically replacing a recognized legacy output."""
    if not replace_existing:
        os.replace(candidate, output)
        return

    previous = staging / "schema-1-sft-checkpoint"
    os.replace(output, previous)
    try:
        os.replace(candidate, output)
    except BaseException:
        os.replace(previous, output)
        raise
    shutil.rmtree(previous)


def prepare_sft_checkpoint_for_rl(source_checkpoint: Path, output_dir: Path) -> Path:
    """Create or reuse an RL-ready copy of an SFT checkpoint.

    The copy omits optimizer, scheduler, RNG, and mutable training state, and clears serialized
    SFT-process callbacks and timers. Model state and portable checkpoint metadata are retained,
    and ``source_checkpoint`` remains unchanged.
    """
    source = _resolve_iteration(source_checkpoint)
    output = output_dir.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("--output-dir must not be the source checkpoint or a directory inside it")

    source_file_count, source_bytes = _tree_stats(source)
    source_facts = {
        "run_config_sha256": _sha256(source / "run_config.yaml"),
        "metadata_sha256": _sha256(source / ".metadata"),
        "payload_file_count": source_file_count,
        "payload_bytes": source_bytes,
    }
    replace_existing = False
    if output.exists():
        reused = _reuse_existing(source, output, source_facts)
        if reused is not None:
            return reused
        replace_existing = True

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    candidate = staging / "sft-checkpoint"
    try:
        remove_optimizer(source, candidate, preserve_model_object_state=True)
        prepared = candidate / source.name
        for training_state in (prepared / "train_state.pt", candidate / "latest_train_state.pt"):
            training_state.unlink(missing_ok=True)
        match = _ITERATION_RE.fullmatch(source.name)
        if match:
            (candidate / "latest_checkpointed_iteration.txt").write_text(f"{int(match.group(1))}\n")

        model_fields, optimizer_fields = _sanitize_run_config(prepared / "run_config.yaml")
        if (
            _sha256(source / "run_config.yaml") != source_facts["run_config_sha256"]
            or _sha256(source / ".metadata") != source_facts["metadata_sha256"]
            or _tree_stats(source) != (source_file_count, source_bytes)
        ):
            raise RuntimeError("source SFT checkpoint changed while its RL-ready copy was being prepared")

        payload_file_count, payload_bytes = _tree_stats(prepared)
        final_checkpoint = (output / source.name).resolve()
        manifest = {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "state": "succeeded",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source_sft_checkpoint": str(source),
            "prepared_sft_checkpoint": str(final_checkpoint),
            "copy_mode": "model-only-dcp-rewrite",
            "source_sft_checkpoint_tree_unchanged": True,
            "model_object_state_preserved": True,
            "payload_file_count": payload_file_count,
            "payload_bytes": payload_bytes,
            "source_payload_file_count": source_file_count,
            "source_payload_bytes": source_bytes,
            "source_run_config_sha256": source_facts["run_config_sha256"],
            "prepared_run_config_sha256": _sha256(prepared / "run_config.yaml"),
            "source_metadata_sha256": source_facts["metadata_sha256"],
            "sanitized_model_runtime_fields": model_fields,
            "sanitized_optimizer_runtime_fields": optimizer_fields,
            "stripped_dcp_key_prefixes": list(STRIPPED_DCP_KEY_PREFIXES),
            "omitted_training_state_files": ["iter_*/train_state.pt", "latest_train_state.pt"],
        }
        (candidate / "preparation-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _install_prepared_output(candidate, output, staging, replace_existing)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info(
        "Prepared SFT checkpoint for RL at %s (%d bytes; source was %d bytes)",
        final_checkpoint,
        payload_bytes,
        source_bytes,
    )
    return final_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        required=True,
        help="Selected SFT iter_* directory or checkpoint root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for the reusable, model-only SFT checkpoint prepared for RL",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare the selected SFT checkpoint and print the direct RL checkpoint path."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(prepare_sft_checkpoint_for_rl(args.source_checkpoint, args.output_dir))


if __name__ == "__main__":
    main()
