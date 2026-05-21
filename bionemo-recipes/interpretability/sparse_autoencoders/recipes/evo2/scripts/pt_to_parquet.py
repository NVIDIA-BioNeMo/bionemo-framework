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

"""Convert predict_evo2 .pt outputs to SAE ActivationStore parquet shards.

predict_evo2 with --embedding-layer writes dicts of:
  hidden_embeddings: [B, S, H] (bf16)
  pad_mask:         [B, S]    (1 = valid token, 0 = padding)
  seq_idx, tokens:  metadata, ignored here

We read each file, mask out padding, flatten to [N_tokens, H], and append
to an ActivationStore so train.py's load_activations() can consume it.
"""

import argparse
import json
from pathlib import Path

import torch
from sae.activation_store import ActivationStore, ActivationStoreConfig
from tqdm import tqdm


def main():
    """Walk predict_evo2 .pt files, mask padding, and write to an ActivationStore."""
    p = argparse.ArgumentParser()
    p.add_argument("--predict-dir", type=Path, required=True, help="Dir containing predictions__*.pt")
    p.add_argument("--output", type=Path, required=True, help="ActivationStore output dir")
    p.add_argument("--model-name", type=str, required=True, help="Stamped into metadata.json")
    p.add_argument("--layer", type=int, required=True, help="Stamped into metadata.json")
    p.add_argument("--shard-size", type=int, default=100_000)
    args = p.parse_args()

    pt_files = sorted(args.predict_dir.rglob("predictions__*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No predictions__*.pt under {args.predict_dir}")

    store = ActivationStore(args.output, ActivationStoreConfig(shard_size=args.shard_size))
    n_sequences = 0
    for pt in tqdm(pt_files, desc="pt->parquet"):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        hidden = d["hidden_embeddings"]
        mask = d["pad_mask"].bool()
        flat = hidden[mask].float()
        store.append(flat)
        n_sequences += hidden.shape[0]

    store.finalize(metadata={"model_name": args.model_name, "layer": args.layer, "n_sequences": n_sequences})
    print(json.dumps(store.metadata, indent=2))


if __name__ == "__main__":
    main()
