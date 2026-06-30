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

"""Fetch an Evo2 NeMo2 checkpoint by identifier and convert it to MBridge.

``bionemo_load(tag)`` (NGC/PBSS) followed by ``run_nemo2_to_mbridge(...)``, so no
checkpoint needs to be pre-staged. Idempotent: if ``<mbridge-ckpt-dir>/iter_0000001``
already exists it is reused. The resulting parent dir is what ``predict --ckpt-dir``
expects.

    python prepare_1b_checkpoint.py --mbridge-ckpt-dir /data/.../evo2_1b_mbridge
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Fetch + convert an Evo2 NeMo2 checkpoint to MBridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mbridge-ckpt-dir", type=Path, required=True, help="Output MBridge checkpoint parent dir")
    p.add_argument("--model-tag", type=str, default="evo2/1b-8k-bf16:1.0", help="bionemo_load identifier (NGC/PBSS)")
    p.add_argument("--model-size", type=str, default="evo2_1b_base", help="MODEL_OPTIONS key for the converter")
    p.add_argument("--seq-length", type=int, default=8192)
    p.add_argument("--mixed-precision-recipe", type=str, default="bf16_mixed")
    p.add_argument("--vortex-style-fp8", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    """Fetch the NeMo2 checkpoint and convert it to MBridge (idempotent)."""
    args = parse_args()
    iter_dir = args.mbridge_ckpt_dir / "iter_0000001"
    if iter_dir.exists():
        print(f"Reusing existing MBridge checkpoint: {iter_dir}")
        return

    # bionemo.common on the migrated recipes layout; bionemo.core on older builds.
    try:
        from bionemo.common.data.load import load as bionemo_load
    except ImportError:
        from bionemo.core.data.load import load as bionemo_load
    from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
    from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge

    print(f"Fetching {args.model_tag} (set BIONEMO_DATA_SOURCE=pbss if NGC is unavailable) ...")
    nemo2_ckpt_path = bionemo_load(args.model_tag)

    args.mbridge_ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    res_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_ckpt_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=args.mbridge_ckpt_dir,
        model_size=args.model_size,
        seq_length=args.seq_length,
        mixed_precision_recipe=args.mixed_precision_recipe,
        vortex_style_fp8=args.vortex_style_fp8,
    )
    print(f"MBridge checkpoint ready: {res_dir}/iter_0000001")


if __name__ == "__main__":
    main()
