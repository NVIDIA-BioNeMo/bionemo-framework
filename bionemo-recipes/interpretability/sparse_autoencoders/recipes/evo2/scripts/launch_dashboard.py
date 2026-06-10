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

"""Launch the evo2 SAE feature-explorer dashboard on data you provide.

Reads the precomputed atlas parquets from --data-dir (it does NOT generate them — that is a
separate offline step), stages them into the dashboard's public/ dir, and starts Vite:

    python scripts/launch_dashboard.py --data-dir /path/to/dashboard_data

The Feature-atlas tab is fully static (served from those parquets). The Sequence-inspector and
Generative-steering tabs call the live backend — start it separately:

    scripts/launch_inference.sh serve
"""

import argparse
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


REQUIRED_PARQUETS = ("features_atlas.parquet", "feature_metadata.parquet", "feature_examples.parquet")
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "feature_explorer"


def stage_dashboard_data(data_dir, public_dir) -> list[str]:
    """Validate the user-provided atlas parquets in `data_dir` and copy them into `public_dir`.

    Checks each required parquet exists and has a `feature_id` column (so a wrong directory fails
    fast rather than rendering an empty dashboard). Returns the staged filenames.
    """
    import pyarrow.parquet as pq

    data_dir, public_dir = Path(data_dir), Path(public_dir)
    missing = [f for f in REQUIRED_PARQUETS if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"--data-dir {data_dir} is missing required parquet(s): {', '.join(missing)}")
    for f in REQUIRED_PARQUETS:
        cols = pq.read_schema(data_dir / f).names
        if "feature_id" not in cols:
            raise ValueError(f"{f} has no 'feature_id' column (got {cols}) — wrong file?")
    public_dir.mkdir(parents=True, exist_ok=True)
    for f in REQUIRED_PARQUETS:
        shutil.copy2(data_dir / f, public_dir / f)
    return list(REQUIRED_PARQUETS)


def main():
    """Stage the provided dashboard data and start the Vite dev server."""
    ap = argparse.ArgumentParser(description="Launch the evo2 SAE feature-explorer dashboard")
    ap.add_argument(
        "--data-dir",
        help=f"Directory with {', '.join(REQUIRED_PARQUETS)} for the Feature-atlas tab. "
        "Omit to launch with the inspector + steering tabs only (which use the live backend).",
    )
    ap.add_argument("--port", type=int, default=5176)
    ap.add_argument("--no-open", action="store_true", help="Don't open a browser")
    args = ap.parse_args()

    if not (DASHBOARD_DIR / "package.json").exists():
        sys.exit(f"dashboard not found at {DASHBOARD_DIR}")

    if args.data_dir:
        staged = stage_dashboard_data(args.data_dir, DASHBOARD_DIR / "public")
        print(f"staged {len(staged)} parquet(s) -> {DASHBOARD_DIR / 'public'}")
    else:
        print("no --data-dir: Feature-atlas tab will be empty; inspector + steering use the backend.")

    if not (DASHBOARD_DIR / "node_modules").exists():
        print("installing dashboard dependencies (npm install)...")
        subprocess.run(["npm", "install"], cwd=DASHBOARD_DIR, check=True)

    print(f"\ndashboard: http://localhost:{args.port}")
    print("inspector + steering tabs need the backend: scripts/launch_inference.sh serve\n")
    proc = subprocess.Popen(["npx", "vite", "--port", str(args.port)], cwd=DASHBOARD_DIR)
    if not args.no_open:
        time.sleep(2)
        webbrowser.open(f"http://localhost:{args.port}")
    try:
        input("dashboard running — press Enter to stop.\n")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
