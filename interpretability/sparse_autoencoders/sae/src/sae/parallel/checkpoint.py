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

"""Sharded checkpointing for the tensor-parallel TopK SAE.

Each rank saves its own latent slice (`save_sharded`); `load_and_merge` reassembles
the shards into a single dense `TopKSAE` (on CPU) so the existing dense eval / loss-
recovered path can be reused without any TP machinery.
"""

import json
import os

import torch


def save_sharded(model, out_dir: str, rank=None) -> None:
    """Save this rank's shard to `out_dir/shard_{rank}.pt` (+ meta.json on rank 0)."""
    rank = model.rank if rank is None else rank
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "W_enc_local": model.W_enc_local.detach().cpu(),
            "latent_bias_local": model.latent_bias_local.detach().cpu(),
            "W_dec_local": model.W_dec_local.detach().cpu(),
            "pre_bias": model.pre_bias.detach().cpu(),
            "rank": rank,
        },
        os.path.join(out_dir, f"shard_{rank:03d}.pt"),
    )
    if rank == 0:
        meta = {
            "world_size": model.world_size,
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "top_k": model.top_k,
            "normalize_input": model.normalize_input,
            "auxk": model.auxk,
            "auxk_coef": model.auxk_coef,
            "dead_tokens_threshold": model.dead_tokens_threshold,
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def load_and_merge(out_dir: str):
    """Reassemble sharded files into a single dense TopKSAE (CPU) for eval."""
    from ..architectures.topk import TopKSAE  # lazy import to avoid any import cycle

    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    world_size = meta["world_size"]

    shards = [torch.load(os.path.join(out_dir, f"shard_{r:03d}.pt"), map_location="cpu") for r in range(world_size)]
    w_enc = torch.cat([s["W_enc_local"] for s in shards], dim=0)  # [n, d]
    w_dec = torch.cat([s["W_dec_local"] for s in shards], dim=1)  # [d, n]
    latent_bias = torch.cat([s["latent_bias_local"] for s in shards], dim=0)  # [n]
    pre_bias = shards[0]["pre_bias"]  # replicated

    sae = TopKSAE(
        input_dim=meta["input_dim"],
        hidden_dim=meta["hidden_dim"],
        top_k=meta["top_k"],
        normalize_input=meta["normalize_input"],
        auxk=meta["auxk"],
        auxk_coef=meta["auxk_coef"],
        dead_tokens_threshold=meta["dead_tokens_threshold"],
        init_encoder_from_decoder=False,
        init_pre_bias=False,
    )
    with torch.no_grad():
        sae.encoder.weight.copy_(w_enc)
        sae.decoder.weight.copy_(w_dec)
        sae.latent_bias.copy_(latent_bias)
        sae.pre_bias.copy_(pre_bias)
    return sae
