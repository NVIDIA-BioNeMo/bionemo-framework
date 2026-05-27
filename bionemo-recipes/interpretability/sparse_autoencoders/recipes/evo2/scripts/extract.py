# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Streaming Evo2 activation extractor — codonfm-style.

Reuses `bionemo.evo2.run.predict` for all the heavy machinery (Megatron
model load, DP/CP/TP/PP setup, FASTA dataloader, inference loop) but
swaps the per-batch `.pt` writer for an in-process `ActivationStore`
that streams parquet shards directly during inference.

Why: predict_evo2's `.pt` intermediate doubles disk volume and forces a
slow downstream pt->parquet shim. For SAE training, the activation tensor
is all we need; writing it directly into the SAE's ActivationStore format
removes the shim entirely, mirroring how codonfm's scripts/extract.py
already works.

Invocation:

    torchrun --nproc_per_node 4 extract.py \
        --fasta path/to/seq.fasta \
        --ckpt-dir path/to/mbridge_ckpt \
        --embedding-layer 20 \
        --activation-store-dir /data/.../parquet_out \
        --max-tokens 25000000 \
        --micro-batch-size 4

All non-`--activation-store-dir`/`--max-tokens` flags are forwarded to
predict_evo2's argparse (`--fasta`, `--ckpt-dir`, `--embedding-layer`,
`--micro-batch-size`, etc.) so the inference surface is identical.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# predict.py provides the entire Megatron inference plumbing.
from bionemo.evo2.run import predict as predict_mod  # noqa: E402

# The SAE activation store — same format the existing pt_to_parquet.py emits.
from sae.activation_store import ActivationStore, ActivationStoreConfig  # noqa: E402

# Reuse the merge step we already have in pt_to_parquet.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pt_to_parquet import _merge_temp_stores  # noqa: E402


# Per-rank state. Each torchrun rank is its own Python process, so this
# module-level dict is rank-local — exactly what we want.
_state: dict = {
    "store": None,        # ActivationStore for this rank
    "n_tokens": 0,        # tokens appended on this rank
    "n_sequences": 0,     # hidden.shape[0] across batches (raw seqs)
    "budget": 0,          # per-rank token cap (0 = no cap)
    "store_root": None,   # Path — final output dir; per-rank tmp lives under <root>.tmp_rank_<i>
    "rank_tmp": None,     # Path — this rank's temp dir
}


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

    Signature matches the original; return shape `(path, updated_count, 0)`.
    """
    if not predictions:
        return output_dir, num_files_written, 0

    # Once we've hit the per-rank budget, skip remaining writes (forward
    # passes still run; cheap relative to the I/O we're skipping).
    if _state["budget"] and _state["n_tokens"] >= _state["budget"]:
        return output_dir, num_files_written, 0

    hidden = predictions["hidden_embeddings"]  # [B, S, H]
    mask = predictions["pad_mask"].bool()
    flat = hidden[mask].cpu()  # [N_unpadded_tokens, H]

    if _state["store"] is None:
        rank_tmp = _state["store_root"].with_name(
            _state["store_root"].name + f".tmp_rank_{dp_rank}"
        )
        rank_tmp.mkdir(parents=True, exist_ok=True)
        _state["rank_tmp"] = rank_tmp
        _state["store"] = ActivationStore(rank_tmp, ActivationStoreConfig(shard_size=100_000))

    _state["store"].append(flat)
    _state["n_tokens"] += flat.shape[0]
    _state["n_sequences"] += hidden.shape[0]
    return output_dir, num_files_written + 1, 0


def _finalize_and_maybe_merge(model_name: str, layer: int) -> None:
    """Finalize this rank's store, then rank 0 waits for all ranks and merges.

    We use a file-based wait (poll for sibling ranks' metadata.json) rather
    than torch.distributed.barrier(): predict.main() tears down the process
    group before this hook runs, so dist.barrier() silently no-ops and rank 0
    would race ahead of slower ranks (observed in the prok+euk run — rank 0
    merged its own dir before ranks 1-3 finalized, leaving 18M tokens orphaned).
    """
    if _state["store"] is not None:
        _state["store"].finalize(metadata={"n_sequences": _state["n_sequences"]})

    if int(os.environ.get("RANK", "0")) != 0:
        return

    # Rank 0 waits for all siblings to finalize before merging.
    import time

    store_root: Path = _state["store_root"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    deadline = time.time() + 600  # 10 min wait cap

    def _ready_count() -> int:
        return sum(
            1
            for r in range(world_size)
            if (store_root.with_name(store_root.name + f".tmp_rank_{r}") / "metadata.json").exists()
        )

    while time.time() < deadline:
        ready = _ready_count()
        if ready >= world_size:
            break
        time.sleep(2)
    else:
        print(
            f"[extract] WARN: only {_ready_count()}/{world_size} ranks finalized within 10 min — "
            "merging what's available; some activations may be orphaned"
        )

    tmp_dirs = sorted(
        p
        for p in store_root.parent.glob(store_root.name + ".tmp_rank_*")
        if p.is_dir() and (p / "metadata.json").exists()
    )
    if not tmp_dirs:
        print(f"[extract] no rank tmp dirs found under {store_root.parent} — nothing to merge")
        return
    print(f"[extract] merging {len(tmp_dirs)} rank tmp dirs into {store_root}")
    final = _merge_temp_stores(tmp_dirs, store_root, model_name, layer)
    print(f"[extract] done: {final}")


def main() -> None:
    """Parse the extractor-specific flags, monkey-patch predict's writer, run predict.main()."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--activation-store-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=0, help="Cap total tokens across DP ranks (0 = no cap).")
    parser.add_argument("--model-name", type=str, default="arcinstitute/savanna_evo2_1b_base")
    extract_args, remaining = parser.parse_known_args()

    dp_size = int(os.environ.get("WORLD_SIZE", "1"))
    _state["store_root"] = extract_args.activation_store_dir
    _state["budget"] = extract_args.max_tokens // dp_size if extract_args.max_tokens else 0

    # Force batch write-interval so our writer is called every iteration
    # (epoch mode would buffer everything in memory, defeating the point).
    if "--write-interval" not in remaining:
        remaining.extend(["--write-interval", "batch"])

    # predict.main() requires --output-dir; we point it at a throwaway path
    # (writer never actually writes there).
    if "--output-dir" not in remaining:
        scratch = _state["store_root"].with_name(_state["store_root"].name + ".predict_unused")
        scratch.mkdir(parents=True, exist_ok=True)
        remaining.extend(["--output-dir", str(scratch)])

    # Capture for the merge metadata; we need to know which layer / model
    # to stamp into the merged ActivationStore.metadata.
    layer = 0
    for i, a in enumerate(remaining):
        if a == "--embedding-layer":
            layer = int(remaining[i + 1])

    # Substitute our writer for predict's. predict.py calls the bare name
    # `_write_predictions_batch(...)` in its module scope, so module-attr
    # replacement is enough.
    predict_mod._write_predictions_batch = _store_writer

    # Hand predict's parser only the args it expects.
    sys.argv = [sys.argv[0]] + remaining

    try:
        predict_mod.main()
    finally:
        _finalize_and_maybe_merge(extract_args.model_name, layer)


if __name__ == "__main__":
    main()
