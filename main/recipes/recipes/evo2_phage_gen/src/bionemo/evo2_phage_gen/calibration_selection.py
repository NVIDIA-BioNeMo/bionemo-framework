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

"""Build uncertainty-aware, within-setting calibration evidence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from bionemo.evo2_phage_gen.calibration_scoring import CELL_RE, EXTERNAL_OBJECTIVES


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _bootstrap_mean(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    if not len(values):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_setting(path: Path, *, bootstrap_seed: int = 174, bootstrap_replicates: int = 2000) -> dict:
    """Summarize one scored calibration setting with bootstrap uncertainty."""
    scored = pd.read_csv(path)
    cell = path.name.removesuffix(".scores.csv")
    match = CELL_RE.fullmatch(cell)
    cell_seed = bootstrap_seed + int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16)
    support_columns = [f"{prefix}_measurement_available" for prefix in EXTERNAL_OBJECTIVES.values()]
    support = pd.concat([_numeric(scored, column) for column in support_columns], axis=1).min(axis=1)
    target_signal = pd.concat(
        [
            _numeric(scored, "reward_external_protein_hit_count"),
            _numeric(scored, "reward_external_tropism"),
            _numeric(scored, "reward_external_required_genes"),
        ],
        axis=1,
    ).mean(axis=1)
    metrics = {
        "aggregate_reward": _numeric(scored, "reward"),
        "target_signal": target_signal,
        "full_qc": _numeric(scored, "reward_binary_full_qc_pass"),
        "full_qc_cluster_deduplicated": _numeric(scored, "reward_binary_full_qc_cluster_deduplicated_pass"),
    }
    cluster_count = _numeric(scored, "mmseqs_cluster_num_clusters").max()
    row: dict[str, float | int | str | bool] = {
        "cell": cell,
        "prefix_length": int(match.group("prefix")) if match else -1,
        "temperature": float(match.group("temperature")) if match else float("nan"),
        "records": len(scored),
        "metric_environment_ok": bool(len(scored) and (_numeric(scored, "external_qc_tool_succeeded") == 1.0).all()),
        "all_external_measurements_available_rate": float(support.mean()),
        "within_setting_99pct_cluster_count": int(cluster_count) if pd.notna(cluster_count) else 0,
        "within_setting_clusterable_count": int(_numeric(scored, "mmseqs_cluster_valid_for_clustering").sum()),
    }
    clusterable = max(1, int(row["within_setting_clusterable_count"]))
    row["within_setting_99pct_distinct_rate"] = min(
        1.0,
        float(row["within_setting_99pct_cluster_count"]) / clusterable,
    )
    row["within_setting_99pct_singleton_rate"] = float(
        _numeric(scored, "mmseqs_cluster_is_singleton").sum() / clusterable
    )
    for name, values in metrics.items():
        array = values.to_numpy(dtype=float)
        low, high = _bootstrap_mean(array, cell_seed, bootstrap_replicates)
        row[f"{name}_mean"] = float(array.mean()) if len(array) else 0.0
        row[f"{name}_ci_low"] = low
        row[f"{name}_ci_high"] = high
    for objective in EXTERNAL_OBJECTIVES:
        row[f"{objective}_reward_mean"] = float(_numeric(scored, f"reward_external_{objective}").mean())
    return row


def build_selection_table(
    score_dir: Path,
    *,
    novelty_summary: Path | None = None,
    bootstrap_seed: int = 174,
    bootstrap_replicates: int = 2000,
    comparability_margin: float = 0.05,
) -> pd.DataFrame:
    """Build the uncertainty-aware selection table across calibration settings."""
    if not np.isfinite(comparability_margin) or comparability_margin < 0:
        raise ValueError("comparability_margin must be finite and non-negative")
    rows = [
        summarize_setting(
            path,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        )
        for path in sorted(score_dir.glob("*.scores.csv"))
    ]
    if not rows:
        raise FileNotFoundError(f"no score CSVs under {score_dir}")
    table = pd.DataFrame(rows)
    if novelty_summary is not None:
        table = table.merge(pd.read_csv(novelty_summary), on="cell", how="left", validate="one_to_one")
    table["eligible"] = table["metric_environment_ok"] & (table["records"] > 0)
    best_reward = table.loc[table["eligible"], "aggregate_reward_mean"].max()
    best_target = table.loc[table["eligible"], "target_signal_mean"].max()
    table["comparability_margin"] = comparability_margin
    table["reward_practically_comparable"] = table["aggregate_reward_ci_high"] >= best_reward - comparability_margin
    table["target_signal_practically_comparable"] = (
        table["target_signal_ci_high"] >= best_target - comparability_margin
    )
    table["temperature_1_default_candidate"] = (
        table["eligible"]
        & table["temperature"].eq(1.0)
        & table["reward_practically_comparable"]
        & table["target_signal_practically_comparable"]
    )
    return table.sort_values(["temperature", "prefix_length"]).reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--novelty-summary", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=174)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--comparability-margin", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    """Build and write the calibration selection table from CLI arguments."""
    args = _parse_args()
    table = build_selection_table(
        args.score_dir,
        novelty_summary=args.novelty_summary,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=max(100, args.bootstrap_replicates),
        comparability_margin=args.comparability_margin,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()
