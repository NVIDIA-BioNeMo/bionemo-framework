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

"""Reproduce the RiboNN Random Forest TE-prediction eval against a PTL CodonFM model.

Mirror of `codonfm_native_te/evaluation/ribonn/evaluate_rf.py`, but using the PyTorch-Lightning
`EncodonInference` wrapper from this recipe (which is what the published
`nvidia/NV-CodonFM-Encodon-TE-*` checkpoints on Hugging Face Hub were trained with).

Loads CDS sequences + labels from `ribonn_cds.parquet` (produced by `preprocess.py` in the
native_te recipe — the parquet schema is identical), extracts CLS-token embeddings, runs
leave-one-fold-out cross-validation with a RandomForestRegressor, and writes per-fold and
aggregate metrics to CSV.

Embeddings are cached to disk and validated on load against the current ids, targets, folds,
sequence hash, and use_transformer_engine flag, so re-running the script only re-tunes the
RF unless any of those inputs change.

Usage:
    python evaluate_rf.py \
        --model-name-or-path nvidia/NV-CodonFM-Encodon-TE-80M-v1 \
        --demo-size 500
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# The `src.*` PTL inference modules live at the recipe root.
RECIPE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RECIPE_ROOT))

from src.data.metadata import MetadataFields  # noqa: E402
from src.inference.encodon import EncodonInference  # noqa: E402
from src.inference.task_types import TaskTypes  # noqa: E402
from src.utils.load_checkpoint import download_checkpoint  # noqa: E402


SCRIPT_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT_CACHE = SCRIPT_DIR / "checkpoints"


def _slugify(s: str) -> str:
    """Make a model name/path safe to embed in a filename."""
    return s.replace("/", "__").replace(":", "_")


def _resolve_model_path(model_name_or_path: str) -> str:
    """Return a local checkpoint dir, downloading from HF Hub if `model_name_or_path` isn't local."""
    p = Path(model_name_or_path)
    if p.is_dir():
        return str(p)
    local_dir = DEFAULT_CHECKPOINT_CACHE / p.name
    print(f"Downloading checkpoint {model_name_or_path} -> {local_dir}")
    return download_checkpoint(repo_id=model_name_or_path, local_dir=str(local_dir))


def extract_embeddings(
    encodon_model: EncodonInference,
    sequences: list[str],
    batch_size: int,
) -> np.ndarray:
    """Return CLS embeddings for `sequences`, looping verbatim from the RiboNN notebook."""
    all_embeddings: list[np.ndarray] = []

    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i : i + batch_size]

        # Prepare batch
        batch_items = []
        for raw_seq in batch_seqs:
            seq = raw_seq.upper().replace("U", "T")
            tokens = encodon_model.tokenizer.tokenize(seq)
            input_ids = encodon_model.tokenizer.convert_tokens_to_ids(tokens)

            # Truncate if needed
            if len(input_ids) > encodon_model.model.hparams.max_position_embeddings - 2:  # Leave room for CLS/SEP
                input_ids = input_ids[: encodon_model.model.hparams.max_position_embeddings - 2]

            # Add special tokens
            input_ids = [encodon_model.tokenizer.cls_token_id, *input_ids, encodon_model.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)

            batch_items.append(
                {
                    MetadataFields.INPUT_IDS: input_ids,
                    MetadataFields.ATTENTION_MASK: attention_mask,
                }
            )

        # Pad batch
        max_len = encodon_model.model.hparams.max_position_embeddings

        padded_input_ids = []
        padded_attention_masks = []

        for item in batch_items:
            input_ids = item[MetadataFields.INPUT_IDS]
            attention_mask = item[MetadataFields.ATTENTION_MASK]

            # Pad
            pad_len = max_len - len(input_ids)
            input_ids.extend([encodon_model.tokenizer.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)

            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)

        # Create batch tensor
        batch = {
            MetadataFields.INPUT_IDS: torch.tensor(padded_input_ids, dtype=torch.long).to(encodon_model.device),
            MetadataFields.ATTENTION_MASK: torch.tensor(padded_attention_masks, dtype=torch.long).to(
                encodon_model.device
            ),
        }

        # Extract embeddings
        output = encodon_model.extract_embeddings(batch)
        all_embeddings.append(output.embeddings)

    return np.vstack(all_embeddings)


def load_or_extract_embeddings(
    df: pl.DataFrame,
    model_name_or_path: str,
    cache_path: Path,
    batch_size: int,
    device: str,
    use_transformer_engine: bool,
    force_extract: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, ids, targets, folds), loading from cache when it is still valid."""
    df_pd = df.to_pandas()
    ids = df_pd["id"].to_numpy()
    targets = df_pd["mean_te"].to_numpy()
    folds = df_pd["fold"].to_numpy()
    raw_sequences = df_pd["cds_sequence"].tolist()
    seqs_hash = hashlib.sha256("\n".join(raw_sequences).encode()).hexdigest()

    if cache_path.exists() and not force_extract:
        z = np.load(cache_path, allow_pickle=False)
        cache_valid = (
            "use_transformer_engine" in z.files
            and bool(z["use_transformer_engine"]) == use_transformer_engine
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

    checkpoint_path = _resolve_model_path(model_name_or_path)
    print(f"Loading PTL model from {checkpoint_path} (use_transformer_engine={use_transformer_engine})")
    encodon_model = EncodonInference(
        model_path=checkpoint_path,
        task_type=TaskTypes.EMBEDDING_PREDICTION,
        use_transformer_engine=use_transformer_engine,
    )
    encodon_model.configure_model()
    encodon_model.to(device)
    encodon_model.eval()

    print(f"Extracting embeddings for {len(raw_sequences)} sequences...")
    embeddings = extract_embeddings(encodon_model, raw_sequences, batch_size=batch_size)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=embeddings,
        ids=ids,
        targets=targets,
        folds=folds,
        seqs_hash=np.array(seqs_hash),
        use_transformer_engine=np.array(use_transformer_engine),
    )
    print(f"✅ Cached embeddings to {cache_path}")
    return embeddings, ids, targets, folds


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
    """CLI entrypoint for the PTL RiboNN RF evaluation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help=(
            "Local checkpoint directory or HF Hub repo id (e.g. nvidia/NV-CodonFM-Encodon-TE-80M-v1). "
            "If not a local directory, the checkpoint is downloaded to ./checkpoints/<basename> next to this script."
        ),
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
        "--no-te",
        action="store_true",
        help="Disable TransformerEngine in EncodonInference (default: TE enabled, matching the notebook).",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract embeddings even if a cached file exists.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_transformer_engine = not args.no_te

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
        use_transformer_engine=use_transformer_engine,
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
