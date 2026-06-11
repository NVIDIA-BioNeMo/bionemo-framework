# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import read_h5ad
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_benchmark(data, labels, use_pca=True):
    """Run the accuracy, precision, recall, and F1-score benchmarks.

    Args:
        data: (R, C) contains the single cell expression (or whatever feature) in each row.
        labels: (R,) contains the string label for each cell
        use_pca: whether to fit PCA to the data.

    Returns:
        results_out: (dict) contains the accuracy, precision, recall, and F1-score for each class.
        conf_matrix: (R, R) contains the confusion matrix.
    """
    np.random.seed(1337)
    # Get input and output dimensions
    n_features = data.shape[1]
    hidden_size = 128

    # Define the target dimension 'n_components' for PCA
    n_components = min(10, n_features)  # ensure we don't try to get more components than features

    # Create a pipeline that includes scaling and MLPClassifier
    if use_pca:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("projection", PCA(n_components=n_components)),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(hidden_size,),
                        max_iter=500,
                        random_state=1337,
                        early_stopping=True,  # Enable early stopping
                        validation_fraction=0.1,  # Use 10% of training data for validation
                        n_iter_no_change=50,  # Stop if validation score doesn't improve for 10 iterations
                        verbose=False,  # Print convergence messages
                    ),
                ),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(hidden_size,),
                        max_iter=500,
                        random_state=1337,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=50,
                        verbose=False,
                    ),
                ),
            ]
        )

    # Set up StratifiedKFold to ensure each fold reflects the overall distribution of labels
    cv = StratifiedKFold(n_splits=5)

    # Define the scoring functions
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="macro"),  # 'macro' averages over classes
        "recall": make_scorer(recall_score, average="macro"),
        "f1_score": make_scorer(f1_score, average="macro"),
    }

    # Track convergence warnings
    convergence_warnings = []
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("always", category=ConvergenceWarning)

        # Perform stratified cross-validation with multiple metrics using the pipeline
        results = cross_validate(pipeline, data, labels, cv=cv, scoring=scoring, return_train_score=False)

        # Collect any convergence warnings
        convergence_warnings = [warn.message for warn in w if issubclass(warn.category, ConvergenceWarning)]

    # Print the cross-validation results
    print("Cross-validation metrics:")
    results_out = {}
    for metric, scores in results.items():
        if metric.startswith("test_"):
            results_out[metric] = (scores.mean(), scores.std())
            print(f"{metric[5:]}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    predictions = cross_val_predict(pipeline, data, labels, cv=cv)

    # v Return confusion matrix and metrics.
    conf_matrix = confusion_matrix(labels, predictions)

    # Print convergence information
    if convergence_warnings:
        print("\nConvergence Warnings:")
        for warning in convergence_warnings:
            print(f"- {warning}")
    else:
        print("\nAll folds converged successfully")

    return results_out, conf_matrix


def load_data_run_benchmark(result_path, adata_path, write_results=True):
    """Load the inference embeddings, original h5ad file for metadata, and finally run the benchmark.

    Args:
        result_path: (Path) path to the directory containing the inference results
        adata_path: (AnnData) the original AnnData object- IMPORTANT, this is used for fetching labels.
        write_results: (bool) whether to write the results to a csv file.
    """
    import torch

    adata = read_h5ad(adata_path)
    # TODO: update the prediction writer to support model and data parallelism, and modify the
    #   path accordingly.
    infer_Xs = torch.load(result_path / "predictions__rank_0.pt")["embeddings"].float().cpu().numpy()
    assert len(adata) == len(infer_Xs), (len(adata), len(infer_Xs))

    infer_metadata = adata.obs
    labels = infer_metadata["cell_type"].values

    # Now we assign integer labels to each of our strings. These do not need to be transformed into one-hot vectors as Random Forest is non-parametric.
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    print(integer_labels)

    # run actual benchmark
    results, cm = run_benchmark(infer_Xs, integer_labels, use_pca=False)

    data = {
        "f1_score_mean": [
            results["test_f1_score"][0],
        ],
        "f1_score_std": [
            results["test_f1_score"][1],
        ],
        "accuracy_mean": [
            results["test_accuracy"][0],
        ],
        "accuracy_std": [
            results["test_accuracy"][1],
        ],
    }

    output_path = result_path / "results.csv"

    # get the column names in a stable order
    columns = list(data.keys())

    rows = []
    for i in range(len(data[columns[0]])):
        row = [data[col][i] for col in columns]
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    if write_results:
        with open(output_path, "w") as f:
            # figure out how many rows we have (assumes all lists are same length)
            # write header
            f.write(",".join(columns) + "\n")
            for row in rows:
                f.write(",".join(map(str, row)) + "\n")

    return df


def main():  # noqa: D103
    adata_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    load_data_run_benchmark(result_path, adata_path)


if __name__ == "__main__":
    main()
