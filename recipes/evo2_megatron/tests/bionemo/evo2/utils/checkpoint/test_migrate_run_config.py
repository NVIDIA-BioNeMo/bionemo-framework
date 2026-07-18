"""Tests for strict MBridge run-config metadata migration."""

from pathlib import Path

import pytest
import yaml

from bionemo.evo2.utils.checkpoint.migrate_run_config import migrate_mbridge_run_config


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_legacy_config(path: Path, *, legacy_value: str = "null") -> None:
    path.write_text(
        "model:\n"
        "  num_layers: 32\n"
        "tokenizer:\n"
        "  tokenizer_type: HuggingFaceTokenizer\n"
        "  tiktoken_num_special_tokens: 1000\n"
        f"  tiktoken_special_tokens: {legacy_value}\n"
        "train:\n"
        "  eval_iters: 32\n",
        encoding="utf-8",
    )


def test_migrate_mbridge_run_config_removes_only_known_null_field(tmp_path: Path) -> None:
    source = tmp_path / "legacy.yaml"
    destination = tmp_path / "current.yaml"
    _write_legacy_config(source)

    receipt = migrate_mbridge_run_config(source, destination)

    source_data = yaml.safe_load(source.read_text(encoding="utf-8"))
    expected_data = yaml.safe_load(source.read_text(encoding="utf-8"))
    del expected_data["tokenizer"]["tiktoken_special_tokens"]
    actual_data = yaml.safe_load(destination.read_text(encoding="utf-8"))
    _require(actual_data == expected_data, "migration changed fields beyond the removed legacy key")
    _require("tiktoken_special_tokens" in source_data["tokenizer"], "source config was mutated")
    _require(receipt.source_path == source.resolve(), "receipt source path mismatch")
    _require(receipt.destination_path == destination.resolve(), "receipt destination path mismatch")
    _require(receipt.source_sha256 != receipt.destination_sha256, "migration digest did not change")
    _require(
        receipt.removed_fields == ("tokenizer.tiktoken_special_tokens",),
        "removed-field receipt mismatch",
    )


def test_migrate_mbridge_run_config_rejects_non_null_legacy_value(tmp_path: Path) -> None:
    source = tmp_path / "legacy.yaml"
    destination = tmp_path / "current.yaml"
    _write_legacy_config(source, legacy_value="['<special>']")

    with pytest.raises(ValueError, match="must be null"):
        migrate_mbridge_run_config(source, destination)

    _require(not destination.exists(), "rejected migration created output")


def test_migrate_mbridge_run_config_rejects_duplicate_keys_and_existing_output(tmp_path: Path) -> None:
    duplicate_source = tmp_path / "duplicate.yaml"
    duplicate_source.write_text(
        "tokenizer:\n"
        "  tiktoken_special_tokens: null\n"
        "tokenizer:\n"
        "  tiktoken_special_tokens: null\n",
        encoding="utf-8",
    )
    destination = tmp_path / "current.yaml"

    with pytest.raises(ValueError, match="duplicate YAML key"):
        migrate_mbridge_run_config(duplicate_source, destination)

    source = tmp_path / "legacy.yaml"
    _write_legacy_config(source)
    destination.write_text("owned: true\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        migrate_mbridge_run_config(source, destination)
    _require(destination.read_text(encoding="utf-8") == "owned: true\n", "existing output was modified")
