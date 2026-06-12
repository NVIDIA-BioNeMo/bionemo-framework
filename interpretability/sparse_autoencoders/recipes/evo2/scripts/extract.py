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

r"""Streaming Evo2 activation extractor (with merge + crash recovery).

Pass an Evo2 MBridge checkpoint + a layer + a FASTA; under torchrun this streams
that layer's residual-stream activations into an SAE ActivationStore (parquet
shards), one data-parallel rank per GPU, merged at the end. Model-size-agnostic
(1B/7B/40B). It reuses bionemo.evo2.run.predict for the Megatron inference path
and only swaps predict's per-batch .pt writer for an in-process store (no .pt).

    torchrun --nproc_per_node 8 extract.py --ckpt-dir CKPT --embedding-layer 26 \
        --fasta SEQS.fasta --activation-store-dir OUT --max-tokens 500000000 \
        --micro-batch-size 4 --dtype fp32

Recovery (no GPU, no Megatron): if a run dies after writing some shards, merge
the surviving per-rank dirs into a usable store:

    python extract.py --recover --activation-store-dir OUT --layer 26

predict/sae are imported lazily on the extract path, so --recover only needs
pyarrow. --activation-store-dir/--max-tokens/--model-name/--dtype/--recover/
--layer are consumed here; all other flags forward verbatim to predict.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch


SHARD_SIZE = 100_000
# bf16 excluded: NumPy/Arrow have no bfloat16, so ActivationStore.append (-> .numpy())
# can't write it. train.py upcasts to fp32 on load, so fp16 is lossless and halves disk.
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}

# Per-rank state. Each torchrun rank is its own process, so this is rank-local.
_state: dict = {
    "store": None,  # ActivationStore for this rank
    "n_tokens": 0,  # tokens appended on this rank
    "n_sequences": 0,  # raw sequences seen
    "budget": 0,  # per-rank token cap (0 = no cap)
    "store_root": None,  # final output dir; per-rank tmp is <root>.tmp_rank_<i>
    "cast": torch.float32,  # dtype activations are stored as
}


# --- per-rank shard merge (pyarrow only; used by rank-0 finalize and --recover) ---


def _merge_temp_stores(tmp_dirs, store_root: Path, model_name: str, layer: int) -> Path:
    """Fold per-rank tmp dirs into one store at store_root.

    Shards are MOVED (rename within the same filesystem) — never copied. Each
    shard's parquet footer is validated first; a shard truncated mid-write when
    the run died (no footer) is skipped rather than moved, so recovery is robust
    to a crash. Counts come from the shards actually merged, so they are correct
    whether or not a per-rank metadata.json was written.
    """
    store_root = Path(store_root)
    store_root.mkdir(parents=True, exist_ok=True)
    # Resume-safe: metadata.json is written last, so if a prior merge moved some
    # shards before crashing, continue past them instead of overwriting.
    existing = sorted(store_root.glob("shard_*.parquet"))
    idx = len(existing)
    total_samples = sum(pq.read_metadata(s).num_rows for s in existing)
    total_sequences = 0
    hidden_dim = pq.read_metadata(existing[0]).num_columns if existing else None
    for d in map(Path, tmp_dirs):
        mp = d / "metadata.json"  # best-effort sequence count (lost if the rank crashed pre-finalize)
        if mp.exists():
            try:
                total_sequences += int(json.load(open(mp)).get("n_sequences", 0))
            except (OSError, ValueError):
                pass
        for shard in sorted(d.glob("shard_*.parquet")):
            try:
                md = pq.read_metadata(shard)
            except Exception:
                print(f"[merge] skipping unreadable shard (truncated mid-write?): {shard}")
                continue
            hidden_dim = hidden_dim or md.num_columns
            shutil.move(str(shard), str(store_root / f"shard_{idx:05d}.parquet"))
            idx += 1
            total_samples += md.num_rows
    with open(store_root / "metadata.json", "w") as f:
        json.dump(
            {
                "n_samples": total_samples,
                "hidden_dim": hidden_dim,
                "n_shards": idx,
                "shard_size": SHARD_SIZE,
                "model_name": model_name,
                "layer": layer,
                "n_sequences": total_sequences,
            },
            f,
            indent=2,
        )
    for d in map(Path, tmp_dirs):  # drop now-empty per-rank dirs
        try:
            (d / "metadata.json").unlink(missing_ok=True)
            d.rmdir()
        except OSError:
            pass
    print(f"[merge] {idx} shards, {total_samples:,} samples (hidden_dim={hidden_dim}) -> {store_root}")
    return store_root


def _recover(store_root: Path, model_name: str, layer: int) -> None:
    """Merge a crashed run's surviving per-rank shards into a usable store."""
    store_root = Path(store_root)
    if (store_root / "metadata.json").exists():
        print(f"{store_root}/metadata.json exists — store looks complete; refusing to clobber.")
        return
    tmp_dirs = sorted(
        p
        for p in store_root.parent.glob(store_root.name + ".tmp_rank_*")
        if p.is_dir() and any(p.glob("shard_*.parquet"))
    )
    if not tmp_dirs:
        print(f"No per-rank shards under {store_root.parent} matching {store_root.name}.tmp_rank_*")
        return
    print(f"Recovering {len(tmp_dirs)} rank dir(s): {[d.name for d in tmp_dirs]}")
    _merge_temp_stores(tmp_dirs, store_root, model_name, layer)


# --- streaming writer (monkeypatched into predict) ---


def _store_writer(
    predictions,
    output_dir,
    batch_idx,
    global_rank,
    dp_rank,
    files_per_subdir=None,
    num_files_written=0,
    data_parallel_world_size=1,
):
    """Replacement for predict._write_predictions_batch — append to ActivationStore.

    Signature matches the original; returns (path, updated_count, 0).
    """
    if not predictions:
        return output_dir, num_files_written, 0
    # Once we've hit the per-rank budget, skip writes (forward passes still run;
    # cheap relative to the I/O we're skipping).
    if _state["budget"] and _state["n_tokens"] >= _state["budget"]:
        return output_dir, num_files_written, 0

    hidden = predictions["hidden_embeddings"]  # [B, S, H]
    mask = predictions["pad_mask"].bool()
    # Cast before the device->host copy: Evo2 runs bf16, which NumPy/Arrow can't store.
    flat = hidden[mask].to(_state["cast"]).cpu()  # [N_unpadded_tokens, H]

    if _state["store"] is None:
        from sae.activation_store import ActivationStore, ActivationStoreConfig  # lazy: keeps --recover sae-free

        rank_tmp = _state["store_root"].with_name(_state["store_root"].name + f".tmp_rank_{dp_rank}")
        rank_tmp.mkdir(parents=True, exist_ok=True)
        _state["store"] = ActivationStore(rank_tmp, ActivationStoreConfig(shard_size=SHARD_SIZE))

    _state["store"].append(flat)
    _state["n_tokens"] += flat.shape[0]
    _state["n_sequences"] += hidden.shape[0]
    return output_dir, num_files_written + 1, 0


def _finalize_and_maybe_merge(model_name: str, layer: int) -> None:
    """Finalize this rank's store, then rank 0 waits for all ranks and merges.

    File-based wait (poll for sibling metadata.json), not dist.barrier():
    predict.main() tears down the process group before this hook runs, so a
    barrier silently no-ops and rank 0 would race ahead of slower ranks
    (observed in the prok+euk run — 18M tokens orphaned before this fix).
    """
    if _state["store"] is not None:
        _state["store"].finalize(metadata={"n_sequences": _state["n_sequences"]})
    if int(os.environ.get("RANK", "0")) != 0:
        return

    import time

    store_root: Path = _state["store_root"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    def _ready() -> int:
        return sum(
            (store_root.with_name(store_root.name + f".tmp_rank_{r}") / "metadata.json").exists()
            for r in range(world_size)
        )

    deadline = time.time() + 600  # 10 min cap
    while time.time() < deadline and _ready() < world_size:
        time.sleep(2)
    if _ready() < world_size:
        print(
            f"[extract] WARN: only {_ready()}/{world_size} ranks finalized in 10 min; --recover can fold in the rest"
        )

    tmp_dirs = sorted(
        p
        for p in store_root.parent.glob(store_root.name + ".tmp_rank_*")
        if p.is_dir() and (p / "metadata.json").exists()
    )
    if tmp_dirs:
        _merge_temp_stores(tmp_dirs, store_root, model_name, layer)
    else:
        print(f"[extract] no rank tmp dirs under {store_root.parent} — nothing to merge")


def main() -> None:
    """Run recovery (--recover) or, under torchrun, stream-extract activations."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--activation-store-dir", type=Path, required=True, help="Output ActivationStore directory.")
    parser.add_argument("--max-tokens", type=int, default=0, help="Cap total tokens across DP ranks (0 = no cap).")
    parser.add_argument(
        "--model-name",
        type=str,
        default="arcinstitute/savanna_evo2_1b_base",
        help="Stamped into store metadata (provenance).",
    )
    parser.add_argument(
        "--dtype",
        choices=list(_DTYPES),
        default="fp32",
        help="Stored precision. fp16 halves disk; train.py upcasts on load.",
    )
    parser.add_argument("--recover", action="store_true", help="Merge a crashed run's shards and exit (no GPU).")
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer index, for --recover metadata (extract reads it from the forwarded --embedding-layer).",
    )
    args, remaining = parser.parse_known_args()

    if args.recover:
        _recover(args.activation_store_dir, args.model_name, args.layer)
        return

    _state["store_root"] = args.activation_store_dir
    # max(1, ...): a small --max-tokens (< world_size) floors to 0, which would be
    # read as "no cap"; clamp so a positive cap is always enforced.
    _state["budget"] = max(1, args.max_tokens // int(os.environ.get("WORLD_SIZE", "1"))) if args.max_tokens else 0
    _state["cast"] = _DTYPES[args.dtype]

    # Force batch write-interval so our writer runs every iteration (epoch mode
    # would buffer everything in memory). predict requires --output-dir; give it
    # a throwaway (our writer never writes there).
    if "--write-interval" not in remaining:
        remaining += ["--write-interval", "batch"]
    if "--output-dir" not in remaining:
        scratch = _state["store_root"].with_name(_state["store_root"].name + ".predict_unused")
        scratch.mkdir(parents=True, exist_ok=True)
        remaining += ["--output-dir", str(scratch)]

    layer = args.layer
    for i, a in enumerate(remaining):
        if a == "--embedding-layer":
            layer = int(remaining[i + 1])
        elif a.startswith("--embedding-layer="):
            layer = int(a.split("=", 1)[1])

    from bionemo.evo2.run import predict as predict_mod  # lazy: the heavy Megatron import

    predict_mod._write_predictions_batch = _store_writer  # module-attr swap (predict calls the bare name)
    sys.argv = [sys.argv[0]] + remaining
    try:
        predict_mod.main()
    finally:
        _finalize_and_maybe_merge(args.model_name, layer)


if __name__ == "__main__":
    main()
