# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Contracts for the production mixed-length Evo2 RL prompt bank."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PATH = RECIPE_ROOT / "data" / "phage_prompts_paper_useful_rl_mixed_8.jsonl"
VALIDATION_PATH = RECIPE_ROOT / "data" / "phage_prompts_paper_useful_rl_validation_mixed_8x12.jsonl"
HISTORICAL_PATH = RECIPE_ROOT / "data" / "phage_prompts_paper_useful_rl_validation_prompt10_96.jsonl"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _prompt(record: dict) -> str:
    return record["messages"][0]["content"].removeprefix("+~")


def test_mixed_training_bank_has_one_prompt_per_length_stratum() -> None:
    records = _records(TRAIN_PATH)

    _require(len(records) == 8, f"expected 8 prompts, got {len(records)}")
    _require([record["length_stratum"] for record in records] == list(range(4, 12)), "wrong strata")
    _require([len(_prompt(record)) for record in records] == list(range(4, 12)), "wrong lengths")
    _require(
        hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()
        == "7f3306efab2fbe416053f1b5c629fdcb23037f9a3c07d98c757543c86f201b1f",
        "mixed training bank hash drifted",
    )


def test_mixed_validation_bank_is_interleaved_and_dp2_balanced() -> None:
    records = _records(VALIDATION_PATH)

    _require(len(records) == 96, f"expected 96 validation rows, got {len(records)}")
    _require(
        hashlib.sha256(VALIDATION_PATH.read_bytes()).hexdigest()
        == "fa9bc74d3784333a5daf29f2c1149dbd7baa302907723ca449aec4bd5e1b8a6b",
        "mixed validation bank hash drifted",
    )
    _require([record["order_index"] for record in records] == list(range(96)), "order drifted")
    _require([record["validation_seed"] for record in records] == list(range(42, 138)), "seed drifted")

    for rollout_ordinal in range(12):
        rows = records[rollout_ordinal * 8 : (rollout_ordinal + 1) * 8]
        _require(
            [row["length_stratum"] for row in rows] == list(range(4, 12)),
            f"rollout {rollout_ordinal} is not interleaved by length",
        )
        _require(
            all(row["rollout_ordinal"] == rollout_ordinal for row in rows),
            f"rollout {rollout_ordinal} labels drifted",
        )

    for dp_rank, rows in enumerate((records[:48], records[48:])):
        counts = {length: 0 for length in range(4, 12)}
        for row in rows:
            counts[row["length_stratum"]] += 1
        _require(
            counts == {length: 6 for length in range(4, 12)},
            f"DP rank {dp_rank} does not own six rows per stratum: {counts}",
        )


def test_prompt10_control_remains_byte_frozen_and_single_length() -> None:
    records = _records(HISTORICAL_PATH)

    _require(len(records) == 96, f"expected 96 historical rows, got {len(records)}")
    _require(all(len(_prompt(record)) == 10 for record in records), "historical control is mixed")
    _require(
        hashlib.sha256(HISTORICAL_PATH.read_bytes()).hexdigest()
        == "7188a847f2cb9c1435617317ec8c06ca3f81dd923c11b2b473a6b7f0fd55f570",
        "historical prompt10 control hash drifted",
    )
