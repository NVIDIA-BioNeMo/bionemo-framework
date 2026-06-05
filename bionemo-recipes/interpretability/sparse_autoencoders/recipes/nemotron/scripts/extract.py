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

r"""Step 1: Extract Nemotron-3-Nano activations and save them to disk.

Runs Nemotron-3-Nano over FineWeb text and writes per-token residual-stream
activations from a target layer to a sharded Parquet activation store. This is
the expensive stage (it loads the 30B model), so it is decoupled from training:
extract once, then sweep many SAE configs against the same cache.

Unlike the data-parallel ESM2 extraction, the 30B Nemotron model does not fit on
a single GPU, so this runs in a single process with ``device_map="auto"`` to
shard the model across all visible GPUs (model parallelism). Do NOT use torchrun.

This is step 1 of the 3-step Nemotron SAE workflow:
    1. extract.py  -- extract activations from Nemotron-3-Nano  (this file)
    2. train.py    -- train SAE on cached activations
    3. eval.py     -- evaluate SAE (reconstruction + loss recovered)

Usage:
    # Uses activations.cache_dir from the config (required)
    python scripts/extract.py activations.cache_dir=.cache/activations/nemotron_l39

    # More data, a different layer
    python scripts/extract.py \
        activations.cache_dir=.cache/activations/nemotron_l26 \
        activations.layer=26 data.max_samples=50000
"""

from pathlib import Path

import hydra
from nemotron_sae.data import load_fineweb
from nemotron_sae.models import NemotronModel
from omegaconf import DictConfig, OmegaConf
from sae.activation_store import ActivationStore, ActivationStoreConfig
from sae.utils import get_device, set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Extract Nemotron activations to a sharded activation store."""
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    cache_dir = cfg.activations.get("cache_dir", None)
    if not cache_dir:
        raise ValueError(
            "extract.py requires activations.cache_dir to be set (where to write the "
            "activation store). Example: activations.cache_dir=.cache/activations/nemotron_l39"
        )
    # Hydra changes cwd per run; resolve relative paths against the original cwd.
    cache_path = Path(hydra.utils.get_original_cwd()) / cache_dir

    # Skip if a finalized cache already exists (extraction is idempotent).
    if (cache_path / "metadata.json").exists():
        store = ActivationStore(cache_path)
        meta = store.metadata
        print(
            f"Cache already exists at {cache_path}: "
            f"{meta['n_samples']:,} tokens, {meta['n_shards']} shards. Skipping extraction."
        )
        return

    # Load text corpus.
    texts = load_fineweb(
        split=cfg.data.get("split", "train"),
        max_samples=cfg.data.get("max_samples"),
        min_length=cfg.data.get("min_length", 50),
        subset=cfg.data.get("subset", "sample-10BT"),
    )
    print(f"Loaded {len(texts)} text samples")

    # Load Nemotron (model-parallel across all visible GPUs via device_map="auto").
    print(f"Loading {cfg.activations.model_name} (layer {cfg.activations.layer})...")
    nemotron = NemotronModel(
        model_name=cfg.activations.model_name,
        layer=cfg.activations.layer,
        device=get_device(),
        max_length=cfg.activations.max_length,
    )
    # Extract incrementally and append to disk: bounded host memory regardless of
    # corpus size. stream_activations yields per-batch [valid_tokens, hidden_dim].
    shard_size = cfg.activations.get("shard_size", 100_000)
    store = ActivationStore(cache_path, ActivationStoreConfig(shard_size=shard_size))

    for flat in nemotron.stream_activations(texts, batch_size=cfg.activations.batch_size, show_progress=True):
        store.append(flat)

    store.finalize(
        metadata={
            "model_name": cfg.activations.model_name,
            "layer": cfg.activations.layer,
            "n_texts": len(texts),
            "max_length": cfg.activations.max_length,
        }
    )

    meta = store.metadata
    print("\nExtraction complete:")
    print(f"  Output:     {cache_path}")
    print(f"  Texts:      {len(texts)}")
    print(f"  Tokens:     {meta['n_samples']:,}")
    print(f"  Hidden dim: {meta['hidden_dim']}")
    print(f"  Shards:     {meta['n_shards']}")


if __name__ == "__main__":
    main()
