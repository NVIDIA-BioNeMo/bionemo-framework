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

"""Launch the ESM2 SAE dashboard locally.

Usage:
    # After scp'ing dashboard data from server:
    scp -r server:/path/to/outputs/650m_5k/eval/dashboard ./dash

    python scripts/launch_dashboard.py --data-dir ./dash
"""

import argparse
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path


def _get_live_feature_ids(data_dir: Path):
    """Return set of feature_ids with activation_freq > 0."""
    import pyarrow.parquet as pq

    meta_path = data_dir / "feature_metadata.parquet"
    if not meta_path.exists():
        return None
    table = pq.read_table(meta_path)
    df = table.to_pandas()
    live = df.loc[df["activation_freq"] > 0, "feature_id"]
    return set(live.tolist())


def _filter_and_copy_parquet(src: Path, dst: Path, live_ids: set):
    """Filter a parquet file to only include live feature_ids."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(src)
    df = table.to_pandas()
    if "feature_id" not in df.columns:
        shutil.copy2(src, dst)
        return len(df), len(df)
    n_before = len(df)
    df = df[df["feature_id"].isin(live_ids)]
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), dst)
    return n_before, len(df)


def main():  # noqa: D103
    p = argparse.ArgumentParser(description="Launch ESM2 SAE dashboard")
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing features_atlas.parquet, feature_metadata.parquet, feature_examples.parquet",
    )
    p.add_argument("--port", type=int, default=5176)
    p.add_argument("--filter-dead", action="store_true", help="Filter out dead latents (activation_freq == 0)")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    dashboard_dir = Path(__file__).resolve().parent.parent / "protein_dashboard"

    if not (dashboard_dir / "package.json").exists():
        raise FileNotFoundError(f"Dashboard not found at {dashboard_dir}")

    # Determine live features (opt-in)
    filter_dead = args.filter_dead
    live_ids = None
    if filter_dead:
        live_ids = _get_live_feature_ids(data_dir)
        if live_ids is not None:
            print(f"Filtering to {len(live_ids)} live features (activation_freq > 0)")
        else:
            print("No feature_metadata.parquet found, skipping dead latent filtering")
            filter_dead = False

    # Copy parquet files into dashboard's public/ dir
    public_dir = dashboard_dir / "public"
    public_dir.mkdir(exist_ok=True)

    parquet_files = ["features_atlas.parquet", "feature_metadata.parquet", "feature_examples.parquet"]
    json_files = ["vocab_logits.json", "cluster_labels.json"]

    for fname in parquet_files:
        src = data_dir / fname
        if not src.exists():
            print(f"WARNING: {fname} not found in {data_dir}")
            continue
        if filter_dead and live_ids is not None:
            n_before, n_after = _filter_and_copy_parquet(src, public_dir / fname, live_ids)
            print(f"Copied {fname} ({n_after}/{n_before} rows, {n_before - n_after} dead filtered)")
        else:
            shutil.copy2(src, public_dir / fname)
            print(f"Copied {fname}")

    for fname in json_files:
        src = data_dir / fname
        if src.exists():
            shutil.copy2(src, public_dir / fname)
            print(f"Copied {fname}")

    # Install deps if needed
    if not (dashboard_dir / "node_modules").exists():
        print("Installing dashboard dependencies...")
        subprocess.run(["npm", "install"], cwd=dashboard_dir, check=True)

    # Launch dev server
    print(f"\nStarting dashboard on http://localhost:{args.port}")
    proc = subprocess.Popen(
        ["npx", "vite", "--port", str(args.port)],
        cwd=dashboard_dir,
    )

    time.sleep(2)
    webbrowser.open(f"http://localhost:{args.port}")

    try:
        input("Dashboard running. Press Enter to stop.\n")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
