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

"""Reproduce the RiboNN Random Forest TE-prediction eval against a HF CodonFM model.

Loads CDS sequences + labels from `ribonn_cds.parquet` (produced by `preprocess.py`),
extracts CLS-token embeddings from a Hugging Face CodonFM checkpoint, runs leave-one-fold-out
cross-validation with a RandomForestRegressor, and writes per-fold metrics to a CSV.

Embeddings are cached to disk and validated on load against the current ids, targets, folds,
sequences, and max_seq_length, so re-running the script only re-tunes the RF without
re-extracting embeddings — unless any of those inputs change, in which case the cache is
treated as stale and re-extracted.

Usage:
    python evaluate_rf.py \
        --model-name-or-path nvidia/NV-CodonFM-Encodon-TE-Cdwt-1B-v1 \
        --demo-size 500
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# `extract_embeddings`, `CodonFMForMaskedLM`, and `CodonTokenizer` live at the recipe root.
RECIPE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RECIPE_ROOT))

from extract_embeddings import extract_embeddings  # noqa: E402
from modeling_codonfm_te import CodonFMConfig, CodonFMForMaskedLM  # noqa: E402
from tokenizer import CodonTokenizer  # noqa: E402


SCRIPT_DIR = Path(__file__).parent


def _slugify(s: str) -> str:
    """Make a model name/path safe to embed in a filename."""
    return s.replace("/", "__").replace(":", "_")


def load_or_extract_embeddings(
    df: pl.DataFrame,
    model_name_or_path: str,
    cache_path: Path,
    batch_size: int,
    device: str,
    force_extract: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, ids, targets, folds), loading from cache when it is still valid."""
    df_pd = df.to_pandas()
    ids = df_pd["id"].to_numpy()
    targets = df_pd["mean_te"].to_numpy()
    folds = df_pd["fold"].to_numpy()
    raw_sequences = df_pd["cds_sequence"].tolist()
    seqs_hash = hashlib.sha256("\n".join(raw_sequences).encode()).hexdigest()

    max_seq_length = CodonFMConfig.from_pretrained(model_name_or_path).max_position_embeddings

    if cache_path.exists() and not force_extract:
        z = np.load(cache_path, allow_pickle=False)
        cache_valid = (
            "max_seq_length" in z.files
            and int(z["max_seq_length"]) == max_seq_length
            and "seqs_hash" in z.files
            and str(z["seqs_hash"].item()) == seqs_hash
            and np.array_equal(z["ids"], ids)
            and np.array_equal(z["targets"], targets)
            and np.array_equal(z["folds"], folds)
        )
        if cache_valid:
            print(f"Loading cached embeddings from {cache_path}")
            return z["embeddings"], z["ids"], z["targets"], z["folds"]
        print(f"⚠️  Cache at {cache_path} is stale; re-extracting.")

    # CodonTokenizer (DNA mode) does not normalise 'U' — unhandled 'U' codons would tokenize
    # to <UNK>. Match the notebook by uppercasing and replacing U->T before encoding.
    sequences = [s.upper().replace("U", "T") for s in raw_sequences]
    records = list(zip([str(i) for i in ids], sequences))

    print(f"Loading model from {model_name_or_path}")
    model = CodonFMForMaskedLM.from_pretrained(model_name_or_path).to(device).eval()
    tokenizer = CodonTokenizer()

    print(f"Extracting embeddings for {len(records)} sequences...")
    out = extract_embeddings(
        model,
        tokenizer,
        records,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        device=device,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=out.embeddings,
        ids=ids,
        targets=targets,
        folds=folds,
        max_seq_length=np.array(max_seq_length),
        seqs_hash=np.array(seqs_hash),
    )
    print(f"✅ Cached embeddings to {cache_path}")
    return out.embeddings, ids, targets, folds


def cross_validate(
    embeddings: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> list[dict]:
    """Run leave-one-fold-out CV with RandomForestRegressor; return per-fold metrics."""
    rows: list[dict] = []
    for fold in np.unique(folds):
        train_mask = folds != fold
        test_mask = ~train_mask
        x_train, x_test = embeddings[train_mask], embeddings[test_mask]
        y_train, y_test = targets[train_mask], targets[test_mask]

        rf = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=2,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        pearson_r, _ = pearsonr(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))

        print(f"Fold {fold}: R² = {r2:.4f}, r = {pearson_r:.4f}, RMSE = {rmse:.4f}")
        rows.append(
            {"fold": int(fold), "r2": float(r2), "pearson_r": float(pearson_r), "mse": float(mse), "rmse": rmse}
        )

    return rows


def main() -> None:
    """CLI entrypoint for the RiboNN RF evaluation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Hugging Face Hub tag or local directory with a CodonFM checkpoint.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=SCRIPT_DIR / "ribonn_cds.parquet",
        help="Parquet file produced by preprocess.py (default: ribonn_cds.parquet next to this script).",
    )
    parser.add_argument(
        "--demo-size",
        type=int,
        default=None,
        help="If set, stratified-sample this many rows by 'fold'. Notebook uses 500.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract embeddings even if a cached file exists.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(args.data_path)
    print(f"Loaded {len(df)} rows from {args.data_path}")

    # Subsample stratified by fold (mirrors notebook section 4).
    if args.demo_size is not None and args.demo_size < len(df):
        print(f"=== SUBSAMPLING DATA to {args.demo_size} rows ===")
        sample_fraction = args.demo_size / len(df)
        _, sampled_pd = train_test_split(
            df.to_pandas(),
            test_size=sample_fraction,
            stratify=df["fold"].to_numpy(),
            random_state=args.seed,
        )
        df = pl.from_pandas(sampled_pd)

    n_tag = f"n{len(df)}"
    cache_path = args.output_dir / f"embeddings_{_slugify(args.model_name_or_path)}_{n_tag}.npz"

    embeddings, _ids, targets, folds = load_or_extract_embeddings(
        df=df,
        model_name_or_path=args.model_name_or_path,
        cache_path=cache_path,
        batch_size=args.batch_size,
        device=args.device,
        force_extract=args.force_extract,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    print("\n=== TRAINING RANDOM FOREST ===")
    rows = cross_validate(embeddings, targets, folds, seed=args.seed)

    metrics_path = args.output_dir / "metrics.csv"
    pl.DataFrame(rows).write_csv(metrics_path)
    print(f"\n✅ Wrote per-fold metrics to {metrics_path}")

    # Summary stats — mirrors notebook section 5: Mean RMSE uses sqrt(mean(MSE)),
    # not mean(sqrt(MSE)).
    r2 = np.array([r["r2"] for r in rows])
    pr = np.array([r["pearson_r"] for r in rows])
    mse = np.array([r["mse"] for r in rows])
    summary = {
        "mean_r2": float(r2.mean()),
        "std_r2": float(r2.std()),
        "mean_pearson_r": float(pr.mean()),
        "std_pearson_r": float(pr.std()),
        "mean_rmse": float(np.sqrt(mse.mean())),
    }
    summary_path = args.output_dir / "metrics_summary.csv"
    pl.DataFrame([summary]).write_csv(summary_path)

    print("\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Mean R²: {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
    print(f"Mean Pearson r: {summary['mean_pearson_r']:.4f} ± {summary['std_pearson_r']:.4f}")
    print(f"Mean RMSE: {summary['mean_rmse']:.4f}")
    print(f"✅ Wrote summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
