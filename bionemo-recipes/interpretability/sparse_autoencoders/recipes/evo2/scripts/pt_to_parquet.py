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

This shim splits the input .pt files across N parallel writers. Each writer
owns its own ActivationStore (writing to a temp dir), so there's no lock
contention on the buffer/shard sequence — true parallelism on both read and
write. After all writers finish, a single-threaded merge step renames + moves
their shards into the final output dir and writes a unified metadata.json.
"""

import argparse
import json
import shutil
# ProcessPoolExecutor — not threads — because the per-shard work (torch.load,
# Arrow encoding, parquet write) is GIL-bound and saturates a single Python
# interpreter, defeating the point of multiple writers. Subprocesses give
# us a real Nx speedup on multi-core boxes.
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from sae.activation_store import ActivationStore, ActivationStoreConfig
from tqdm import tqdm


def _load_one(pt_path: Path) -> tuple[torch.Tensor, int]:
    """Load one .pt, mask padding, return (flat [N_tokens, H] fp32, n_sequences)."""
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    hidden = d["hidden_embeddings"]
    mask = d["pad_mask"].bool()
    flat = hidden[mask].float()
    return flat, hidden.shape[0]


def _writer_worker(worker_id: int, pt_subset: list[Path], temp_dir: Path, shard_size: int) -> dict:
    """Process a slice of .pt files into its own ActivationStore at temp_dir.

    Returns a dict with the worker's metadata, used by the merge step.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    store = ActivationStore(temp_dir, ActivationStoreConfig(shard_size=shard_size))
    n_sequences = 0
    for pt in pt_subset:
        flat, n_seqs = _load_one(pt)
        store.append(flat)
        n_sequences += n_seqs
    store.finalize(metadata={"n_sequences": n_sequences})
    return {"worker_id": worker_id, "metadata": store.metadata}


def _merge_temp_stores(temp_dirs: list[Path], output: Path, model_name: str, layer: int) -> dict:
    """Move shards from each temp dir into output, renumbered sequentially. Returns unified metadata."""
    output.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    total_samples = 0
    total_sequences = 0
    hidden_dim = None
    shard_size = None

    for tmp in temp_dirs:
        meta_path = tmp / "metadata.json"
        if not meta_path.exists():
            raise RuntimeError(f"Worker temp dir missing metadata: {tmp}")
        with open(meta_path) as f:
            tmp_meta = json.load(f)
        hidden_dim = tmp_meta["hidden_dim"]
        shard_size = tmp_meta["shard_size"]
        for i in range(tmp_meta["n_shards"]):
            src = tmp / f"shard_{i:05d}.parquet"
            dst = output / f"shard_{shard_idx:05d}.parquet"
            shutil.move(str(src), str(dst))
            shard_idx += 1
        total_samples += tmp_meta["n_samples"]
        total_sequences += tmp_meta.get("n_sequences", 0)
        shutil.rmtree(tmp)

    final_meta = {
        "n_samples": total_samples,
        "hidden_dim": hidden_dim,
        "n_shards": shard_idx,
        "shard_size": shard_size,
        "model_name": model_name,
        "layer": layer,
        "n_sequences": total_sequences,
    }
    with open(output / "metadata.json", "w") as f:
        json.dump(final_meta, f, indent=2)
    return final_meta


def main():
    """Walk predict_evo2 .pt files in parallel, write parquet shards via N writers, then merge."""
    p = argparse.ArgumentParser()
    p.add_argument("--predict-dir", type=Path, required=True, help="Dir containing predictions__*.pt")
    p.add_argument("--output", type=Path, required=True, help="ActivationStore output dir")
    p.add_argument("--model-name", type=str, required=True, help="Stamped into metadata.json")
    p.add_argument("--layer", type=int, required=True, help="Stamped into metadata.json")
    p.add_argument("--shard-size", type=int, default=100_000)
    p.add_argument("--writers", type=int, default=4, help="Parallel writer workers (each owns its own ActivationStore)")
    args = p.parse_args()

    pt_files = sorted(args.predict_dir.rglob("predictions__*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No predictions__*.pt under {args.predict_dir}")

    # Split files evenly across writers.
    n_writers = max(1, min(args.writers, len(pt_files)))
    chunk = (len(pt_files) + n_writers - 1) // n_writers
    splits = [pt_files[i : i + chunk] for i in range(0, len(pt_files), chunk)]
    temp_dirs = [args.output.with_name(args.output.name + f".tmp_writer_{i}") for i in range(len(splits))]

    print(f"Sharding {len(pt_files)} .pt files across {len(splits)} writers (~{chunk} files each)")

    with ProcessPoolExecutor(max_workers=len(splits)) as ex:
        futures = {
            ex.submit(_writer_worker, i, split, temp_dirs[i], args.shard_size): i for i, split in enumerate(splits)
        }
        for fut in tqdm(futures, desc="writers"):
            result = fut.result()
            print(f"  writer {result['worker_id']}: {result['metadata']['n_samples']:,} tokens")

    print(f"Merging {len(temp_dirs)} temp stores into {args.output}")
    final = _merge_temp_stores(temp_dirs, args.output, args.model_name, args.layer)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
