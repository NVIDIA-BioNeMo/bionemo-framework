"""Strict migrations for legacy Megatron Bridge run-config metadata."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.resolver import BaseResolver


@dataclass(frozen=True)
class RunConfigMigrationReceipt:
    """Identity of a completed run-config migration."""

    source_path: Path
    destination_path: Path
    source_sha256: str
    destination_sha256: str
    removed_fields: tuple[str, ...]


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError(f"unhashable YAML mapping key at line {key_node.start_mark.line + 1}") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def migrate_mbridge_run_config(source: Path, destination: Path) -> RunConfigMigrationReceipt:
    """Remove the retired null tiktoken field from one immutable byte snapshot.

    No other semantic field is changed. The source remains untouched, and an
    existing destination is never overwritten.
    """
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    source_bytes = source.read_bytes()
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"run config is not UTF-8: {source}") from exc
    config = yaml.load(source_text, Loader=_UniqueKeyLoader)
    if type(config) is not dict:
        raise ValueError("run config root must be a mapping")
    tokenizer = config.get("tokenizer")
    if type(tokenizer) is not dict:
        raise ValueError("run config tokenizer must be a mapping")
    legacy_field = "tiktoken_special_tokens"
    if legacy_field not in tokenizer:
        raise ValueError(f"legacy field tokenizer.{legacy_field} is absent")
    if tokenizer[legacy_field] is not None:
        raise ValueError(f"legacy field tokenizer.{legacy_field} must be null")
    del tokenizer[legacy_field]

    destination_bytes = yaml.safe_dump(config, sort_keys=False, allow_unicode=False).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(destination_bytes)
        handle.flush()
        os.fsync(handle.fileno())

    return RunConfigMigrationReceipt(
        source_path=source,
        destination_path=destination,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        destination_sha256=hashlib.sha256(destination_bytes).hexdigest(),
        removed_fields=(f"tokenizer.{legacy_field}",),
    )
