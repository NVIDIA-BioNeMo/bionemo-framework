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

"""Minimal tensor-parallel training loop for ShardedTopKSAE.

Kept separate from the DDP `Trainer` (which is untouched): TP shards the model (not
the batch), so every rank trains on the *same* data, runs no DDP wrap, and only
all-reduces the replicated `pre_bias` gradient. Per-step correctness is covered by
the B1 parity test; this loop is the orchestration around it.
"""

import time

import torch
import torch.distributed as dist


def train_tp_loop(
    sae,
    dataloader,
    *,
    lr: float,
    max_steps: int,
    device: str,
    log_interval: int = 100,
    max_grad_norm=None,
    checkpoint_dir=None,
    group=None,
    perf_logger=None,
) -> float:
    """Train a ShardedTopKSAE for `max_steps` optimizer steps. Returns final loss.

    If `perf_logger` is provided (rank 0 only), per-step metrics are logged through it
    (same metrics/W&B path as the dense recipe); otherwise rank 0 prints periodically.
    """
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    sae = sae.to(device)
    sae.train()
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # Tensor parallelism replicates the batch: every rank MUST train on the same data,
    # else the all-reduced reconstruction combines different inputs and diverges. Rank 0
    # drives the dataloader and broadcasts each batch to the TP group.
    distributed = dist.is_initialized() and dist.get_world_size(group) > 1
    data_iter = iter(dataloader) if rank == 0 else None

    final_loss = float("nan")
    t0 = time.time()
    for step in range(max_steps):
        if rank == 0:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            x = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device).contiguous()
            meta = torch.tensor([x.shape[0], x.shape[1]], dtype=torch.long, device=device)
        else:
            meta = torch.empty(2, dtype=torch.long, device=device)
        if distributed:
            dist.broadcast(meta, src=0, group=group)
            if rank != 0:
                x = torch.empty(int(meta[0]), int(meta[1]), device=device)
            dist.broadcast(x, src=0, group=group)

        optimizer.zero_grad(set_to_none=True)
        out = sae.loss(x)
        loss = out["total"]
        loss.backward()
        sae.reduce_replicated_grads()  # all-reduce replicated (pre_bias) grad
        grad_norm = None
        if max_grad_norm is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(sae.parameters(), max_grad_norm))
        optimizer.step()
        if hasattr(sae, "post_step"):
            sae.post_step()  # e.g. decoder normalization (keeps TopK training stable)

        final_loss = float(loss.detach())
        if rank == 0 and perf_logger is not None:
            perf_logger.log_step(
                step=step,
                batch=x,
                loss_dict={k: float(v) for k, v in out.items()},
                grad_norm=grad_norm,
                lr=lr,
                extra_metrics={"dead_pct": float(out["dead_pct"])},
            )
        elif rank == 0 and (step % log_interval == 0 or step == max_steps - 1):
            rate = (step + 1) / (time.time() - t0)
            mem = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
            print(
                f"step {step:>7} | loss {final_loss:.6f} | fvu {float(out['fvu']):.4f} "
                f"| dead% {float(out['dead_pct']):.2f} | {rate:.1f} steps/s | peak {mem:.1f}GB",
                flush=True,
            )

    if rank == 0 and perf_logger is not None:
        perf_logger.finish()
    if checkpoint_dir is not None:
        from .checkpoint import save_sharded

        save_sharded(sae, checkpoint_dir, rank=rank)
        if rank == 0:
            print(f"Saved sharded checkpoint ({sae.world_size} shards) to {checkpoint_dir}", flush=True)
    return final_loss
