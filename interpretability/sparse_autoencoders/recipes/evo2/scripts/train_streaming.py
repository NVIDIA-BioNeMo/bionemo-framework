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

r"""Train an Evo2 SAE with producer-consumer activation streaming.

This is the Evo2 analogue of the ESM2 ``train_streaming.py``. It does NOT
materialize an ActivationStore on disk: a background producer thread runs Evo2
(Megatron) forward passes and pushes flattened layer activations into a bounded
queue; the SAE ``Trainer`` consumes re-batched activation tensors from that
queue in the main process.

Bridge to Megatron
------------------
Evo2's forward pass is ``bionemo.evo2.run.predict``, which owns its own batch
loop and exposes no generator API -- it only calls a module-level
``_write_predictions_batch(...)`` callback per batch. We reuse the same hook
``extract.py`` uses: monkeypatch ``predict._write_predictions_batch`` with a
writer that, instead of appending to a parquet store, pushes the flattened
``[n_tokens, hidden_dim]`` activations onto a queue. ``predict.main()`` runs in
a daemon thread; this script's producer generator drains the queue and yields
chunks to ``sae.streaming``. Converting predict's push-callback into a
pull-generator REQUIRES a thread + queue (predict cannot be paused mid-loop).

Launch
------
Run under ``torchrun --nproc_per_node 1`` (NOT bare ``python``): ``predict``
calls ``initialize_inference_distributed`` and needs ``RANK``/``WORLD_SIZE``/
``MASTER_*``. SAE training uses ``--dp-size 1``. Only ``--dp-size 1`` is
supported here (single producer/consumer on one GPU); multi-GPU streaming would
need one predict replica per rank.

    torchrun --nproc_per_node 1 train_streaming.py \
        --ckpt-dir CKPT --fasta SEQS.fasta --embedding-layer 12 --input-dim 1920 \
        --max-tokens 100000 --micro-batch-size 4 \
        --model-type topk --expansion-factor 8 --top-k 32 --normalize-input \
        --n-epochs 1 --batch-size 256 --lr 1e-4 --no-init-pre-bias --no-wandb \
        --output-dir OUT --checkpoint-dir OUT/checkpoints
"""

from __future__ import annotations

import argparse
import queue
import sys
import tempfile
import threading
from pathlib import Path

import torch
from sae.architectures import ReLUSAE, TopKSAE
from sae.perf_logger import PerfLogger
from sae.streaming import StreamingConfig, make_streaming_dataloader
from sae.training import ParallelConfig, Trainer, TrainingConfig, WandbConfig
from sae.utils import get_device, set_seed


# Stored precision for activations. bf16 is excluded: NumPy/Arrow (and the SAE's
# float math downstream) want a numpy-representable dtype, so we cast Evo2's bf16
# residual stream to fp32/fp16 on the device->host copy (mirrors extract.py).
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}

# Sentinel marking the producer thread finished (unique object; never a chunk).
_DONE = object()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Train an SAE directly from streamed Evo2 activations (no disk store)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_argument_group("Evo2 producer")
    src.add_argument("--ckpt-dir", type=str, required=True, help="Evo2 MBridge checkpoint directory")
    src.add_argument("--fasta", type=str, required=True, help="Input FASTA (chunk long sequences first)")
    src.add_argument("--embedding-layer", "--layer", dest="layer", type=int, required=True, help="Layer to extract")
    src.add_argument("--input-dim", type=int, required=True, help="Residual-stream width of --embedding-layer")
    src.add_argument("--micro-batch-size", type=int, default=4, help="Sequences per Evo2 forward pass")
    src.add_argument("--max-tokens", type=int, default=0, help="Cap activation rows produced (0 = no cap)")
    src.add_argument("--dtype", choices=list(_DTYPES), default="fp32", help="Activation cast (bf16 not storable)")

    sae_group = p.add_argument_group("SAE model")
    sae_group.add_argument("--model-type", type=str, default="topk", choices=["topk", "relu"])
    sae_group.add_argument("--expansion-factor", type=int, default=8)
    sae_group.add_argument("--top-k", type=int, default=32)
    sae_group.add_argument("--normalize-input", action=argparse.BooleanOptionalAction, default=False)
    sae_group.add_argument("--auxk", type=int, default=None)
    sae_group.add_argument("--auxk-coef", type=float, default=1 / 32)
    sae_group.add_argument("--dead-tokens-threshold", type=int, default=10_000_000)
    sae_group.add_argument("--aggregate-loss", action=argparse.BooleanOptionalAction, default=False)
    sae_group.add_argument("--dead-count-global", action=argparse.BooleanOptionalAction, default=False)
    sae_group.add_argument("--init-pre-bias", action=argparse.BooleanOptionalAction, default=False)
    sae_group.add_argument("--pre-bias-sample-size", type=int, default=32768)
    sae_group.add_argument("--l1-coeff", type=float, default=1e-2, help="L1 coefficient (relu only)")

    train_group = p.add_argument_group("SAE training")
    train_group.add_argument("--lr", type=float, default=1e-4)
    train_group.add_argument("--n-epochs", type=int, default=1)
    train_group.add_argument("--max-steps", type=int, default=None, help="Exact optimizer-step budget")
    train_group.add_argument("--batch-size", type=int, default=1024, help="Activation rows per SAE training batch")
    train_group.add_argument("--log-interval", type=int, default=50)
    train_group.add_argument("--max-grad-norm", type=float, default=None)
    train_group.add_argument("--grad-accumulation-steps", type=int, default=1)
    train_group.add_argument("--warmup-steps", type=int, default=0)
    train_group.add_argument("--lr-schedule", choices=["constant", "cosine", "linear"], default="constant")
    train_group.add_argument("--lr-min", type=float, default=0.0)
    train_group.add_argument("--lr-decay-steps", type=int, default=None)

    stream_group = p.add_argument_group("Producer-consumer streaming")
    stream_group.add_argument("--queue-size", type=int, default=4, help="Activation chunks buffered (backpressure)")
    stream_group.add_argument("--shuffle-buffer-size", type=int, default=65536)
    stream_group.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=False)

    wb_group = p.add_argument_group("Weights & Biases")
    wb_group.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, dest="wandb_enabled")
    wb_group.add_argument("--wandb-project", type=str, default="evo2-sae-v2-diverse")
    wb_group.add_argument("--wandb-run-name", type=str, default=None)
    wb_group.add_argument("--wandb-group", type=str, default=None)
    wb_group.add_argument("--wandb-job-type", type=str, default=None)

    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--checkpoint-steps", type=int, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="Device for SAE training")
    p.add_argument("--dp-size", type=int, default=1, help="Only dp-size=1 is supported by this streaming script")
    return p.parse_args()


def build_sae(args: argparse.Namespace, input_dim: int) -> torch.nn.Module:
    """Build an SAE model (mirrors evo2 train.py:build_sae)."""
    hidden_dim = input_dim * args.expansion_factor
    if args.model_type == "topk":
        return TopKSAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            top_k=args.top_k,
            normalize_input=args.normalize_input,
            auxk=args.auxk,
            auxk_coef=args.auxk_coef,
            dead_tokens_threshold=args.dead_tokens_threshold,
            aggregate_loss=args.aggregate_loss,
            dead_count_global=args.dead_count_global,
        )
    return ReLUSAE(input_dim=input_dim, hidden_dim=hidden_dim, l1_coeff=args.l1_coeff)


class Evo2ActivationProducer:
    """Producer factory bridging Evo2 ``predict`` (push) to a pull-generator.

    Calling an instance returns a fresh generator that runs ``predict.main()`` in
    a daemon thread (with the per-batch writer monkeypatched to enqueue flattened
    activations) and yields ``[n_tokens, hidden_dim]`` fp32 CPU chunks until the
    thread finishes. Exceptions from the predict thread are re-raised in the
    consumer so a dead producer fails loudly instead of hanging.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def _make_writer(self, q: "queue.Queue", state: dict):
        """Return a drop-in replacement for predict._write_predictions_batch."""
        cast = _DTYPES[self.args.dtype]
        budget = self.args.max_tokens

        def writer(
            predictions,
            output_dir,
            batch_idx,
            global_rank,
            dp_rank,
            files_per_subdir=None,
            num_files_written=0,
            data_parallel_world_size=1,
        ):
            if not predictions:
                return output_dir, num_files_written, 0
            # Past the row budget: keep running cheap forwards but stop enqueuing.
            if budget and state["n_tokens"] >= budget:
                return output_dir, num_files_written, 0
            hidden = predictions["hidden_embeddings"]  # [B, S, H]
            mask = predictions["pad_mask"].bool()
            flat = hidden[mask].to(cast).cpu()  # [N_unpadded_tokens, H]
            q.put(flat)
            state["n_tokens"] += flat.shape[0]
            return output_dir, num_files_written + 1, 0

        return writer

    def __call__(self):
        args = self.args
        q: "queue.Queue" = queue.Queue(maxsize=args.queue_size)
        state = {"n_tokens": 0}

        from bionemo.evo2.run import predict as predict_mod  # lazy: heavy Megatron import

        def run_predict() -> None:
            # predict requires --output-dir even though our writer ignores it.
            scratch = tempfile.mkdtemp(prefix="evo2_stream_predict_unused_")
            predict_mod._write_predictions_batch = self._make_writer(q, state)
            sys.argv = [
                "predict_evo2",
                "--ckpt-dir", args.ckpt_dir,
                "--fasta", args.fasta,
                "--embedding-layer", str(args.layer),
                "--micro-batch-size", str(args.micro_batch_size),
                "--write-interval", "batch",
                "--output-dir", scratch,
            ]
            try:
                predict_mod.main()
                q.put(_DONE)
            except BaseException as exc:  # surface producer failure to the consumer
                q.put(exc)

        thread = threading.Thread(target=run_predict, name="evo2-predict-producer", daemon=True)
        thread.start()
        checked = False
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                if not checked:
                    # Fail clearly if --input-dim disagrees with the layer's true residual
                    # width, instead of an opaque matmul shape error inside the SAE encoder.
                    if item.shape[1] != args.input_dim:
                        raise ValueError(
                            f"--input-dim={args.input_dim} but streamed Evo2 activations are width "
                            f"{item.shape[1]} (embedding-layer {args.layer}); set --input-dim to match."
                        )
                    checked = True
                yield item
        finally:
            thread.join(timeout=30.0)


def main() -> None:
    """Run streaming SAE training."""
    args = parse_args()
    if args.dp_size != 1:
        raise ValueError("train_streaming.py supports only --dp-size 1; use one GPU for this streaming path.")

    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")

    input_dim = args.input_dim
    sae = build_sae(args, input_dim)
    print(f"SAE: {args.model_type}, input_dim={input_dim}, hidden_dim={sae.hidden_dim}")

    producer = Evo2ActivationProducer(args)
    if args.init_pre_bias:
        # Streaming pre-bias init would need a *second* Evo2 `predict` pass to sample
        # activations before training. `predict` initializes Megatron global state
        # (num-microbatches calculator, model-parallel groups) that it never tears down,
        # so a second pass crashes with "num microbatches calculator is already
        # initialized". Fail loudly instead of dying cryptically mid-run. Tracked for a
        # single-pass fix (sample the first rows off the one training stream).
        raise NotImplementedError(
            "--init-pre-bias is not supported by the streaming path (a second Evo2 predict "
            "pass collides with Megatron global state). Re-run with --no-init-pre-bias."
        )

    dataloader = make_streaming_dataloader(
        producer,
        batch_size=args.batch_size,
        config=StreamingConfig(
            enabled=True,
            queue_size=args.queue_size,
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=args.seed,
            drop_last=args.drop_last,
        ),
    )

    training_config = TrainingConfig(
        lr=args.lr,
        n_epochs=args.n_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=device,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_steps=args.checkpoint_steps,
        grad_accumulation_steps=args.grad_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        lr_decay_steps=args.lr_decay_steps,
    )
    wandb_config = WandbConfig(
        enabled=args.wandb_enabled,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        config=vars(args),
    )
    perf_logger = PerfLogger(
        log_interval=args.log_interval,
        use_wandb=args.wandb_enabled,
        print_logs=True,
        device=device,
    )
    trainer = Trainer(
        sae,
        training_config,
        wandb_config=wandb_config,
        perf_logger=perf_logger,
        parallel_config=ParallelConfig(dp_size=args.dp_size),
    )
    trainer.fit(dataloader, resume_from=args.resume_from, data_sharded=True)


if __name__ == "__main__":
    main()
