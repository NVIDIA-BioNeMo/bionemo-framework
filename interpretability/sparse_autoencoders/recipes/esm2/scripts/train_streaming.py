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

"""Train an ESM2 SAE with producer-consumer activation streaming.

This path does not materialize an ActivationStore. A background producer thread
runs ESM2 forward passes and pushes flattened layer activations into a bounded
queue; the SAE Trainer consumes re-batched activation tensors from that queue.
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import torch
from esm2_sae.data import download_swissprot, download_uniref50
from esm2_sae.data.fasta import stream_fasta
from sae.architectures import ReLUSAE, TopKSAE
from sae.perf_logger import PerfLogger
from sae.streaming import StreamingConfig, make_streaming_dataloader
from sae.training import ParallelConfig, Trainer, TrainingConfig, WandbConfig
from sae.utils import get_device, set_seed
from transformers import AutoConfig, AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Train an SAE directly from streamed ESM2 activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--fasta", type=str, help="Path to input FASTA file")
    data.add_argument("--source", type=str, choices=["uniref50", "swissprot"], help="Download/use this source")
    p.add_argument("--data-dir", type=str, default="./data", help="Directory for downloaded sequence data")
    p.add_argument("--num-proteins", type=int, default=None, help="Maximum number of proteins per producer epoch")
    p.add_argument("--filter-length", action="store_true", default=False)

    p.add_argument("--model-name", type=str, default="nvidia/esm2_t33_650M_UR50D")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--extract-batch-size", type=int, default=8, help="Sequences per ESM2 producer forward pass")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--remove-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--producer-device", type=str, default=None, help="Device for ESM2 producer model")

    sae_group = p.add_argument_group("SAE model")
    sae_group.add_argument("--model-type", type=str, default="topk", choices=["topk", "relu"])
    sae_group.add_argument("--expansion-factor", type=int, default=8)
    sae_group.add_argument("--top-k", type=int, default=32)
    sae_group.add_argument("--normalize-input", action=argparse.BooleanOptionalAction, default=False)
    sae_group.add_argument("--auxk", type=int, default=None)
    sae_group.add_argument("--auxk-coef", type=float, default=1 / 32)
    sae_group.add_argument("--dead-tokens-threshold", type=int, default=10_000_000)
    sae_group.add_argument("--init-pre-bias", action=argparse.BooleanOptionalAction, default=False)
    sae_group.add_argument("--pre-bias-sample-size", type=int, default=32768)
    sae_group.add_argument("--l1-coeff", type=float, default=1e-2)

    train_group = p.add_argument_group("SAE training")
    train_group.add_argument("--lr", type=float, default=3e-4)
    train_group.add_argument("--n-epochs", type=int, default=1)
    train_group.add_argument("--max-steps", type=int, default=None, help="Exact optimizer-step budget")
    train_group.add_argument("--batch-size", type=int, default=4096, help="Activation rows per SAE training batch")
    train_group.add_argument("--log-interval", type=int, default=50)
    train_group.add_argument("--max-grad-norm", type=float, default=None)
    train_group.add_argument("--grad-accumulation-steps", type=int, default=1)
    train_group.add_argument("--warmup-steps", type=int, default=0)
    train_group.add_argument("--lr-schedule", choices=["constant", "cosine", "linear"], default="constant")
    train_group.add_argument("--lr-min", type=float, default=0.0)
    train_group.add_argument("--lr-decay-steps", type=int, default=None)

    stream_group = p.add_argument_group("Producer-consumer streaming")
    stream_group.add_argument("--queue-size", type=int, default=4)
    stream_group.add_argument("--shuffle-buffer-size", type=int, default=65536)
    stream_group.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=False)

    wb_group = p.add_argument_group("Weights & Biases")
    wb_group.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, dest="wandb_enabled")
    wb_group.add_argument("--wandb-project", type=str, default="sae_esm2_recipe")
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


def resolve_fasta(args: argparse.Namespace) -> Path:
    """Resolve a FASTA path, downloading sequence data when requested."""
    if args.fasta:
        return Path(args.fasta)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "uniref50":
        dl_max_length = args.max_length if args.filter_length else None
        if args.num_proteins:
            if dl_max_length:
                fasta_path = data_dir / f"uniref50_{args.num_proteins}_maxlen{dl_max_length}.fasta"
            else:
                fasta_path = data_dir / f"uniref50_first_{args.num_proteins}.fasta"
        else:
            fasta_path = data_dir / "uniref50.fasta.gz"
        if not fasta_path.exists():
            download_uniref50(data_dir, max_proteins=args.num_proteins, max_length=dl_max_length)
        return fasta_path

    fasta_path = data_dir / "uniprot_sprot.fasta.gz"
    if not fasta_path.exists():
        download_swissprot(data_dir)
    return fasta_path


def model_dtype(name: str) -> torch.dtype:
    """Map dtype CLI value to torch dtype."""
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def build_sae(args: argparse.Namespace, input_dim: int) -> torch.nn.Module:
    """Build an SAE model."""
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
        )
    return ReLUSAE(input_dim=input_dim, hidden_dim=hidden_dim, l1_coeff=args.l1_coeff)


class ESM2ActivationProducer:
    """Callable producer factory that yields flattened ESM2 activations."""

    def __init__(self, args: argparse.Namespace, fasta_path: Path):
        self.args = args
        self.fasta_path = fasta_path
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        if self.model is not None:
            return
        args = self.args
        device = args.producer_device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": model_dtype(args.dtype),
            "add_pooling_layer": False,
        }
        print(f"Loading producer model {args.model_name} on {device} ({args.dtype})")
        self.model = AutoModel.from_pretrained(args.model_name, **model_kwargs).to(device).eval()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    def __call__(self):
        self._load()
        args = self.args
        assert self.model is not None
        assert self.tokenizer is not None
        input_device = next(self.model.parameters()).device

        records = stream_fasta(self.fasta_path, max_length=args.max_length)
        if args.num_proteins is not None:
            records = islice(records, args.num_proteins)

        batch = []
        with torch.no_grad():
            for rec in records:
                batch.append(rec.sequence)
                if len(batch) >= args.extract_batch_size:
                    yield self._extract_batch(batch, input_device)
                    batch = []
            if batch:
                yield self._extract_batch(batch, input_device)

    def _extract_batch(self, sequences: list[str], input_device: torch.device) -> torch.Tensor:
        args = self.args
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer].float().cpu()
        mask = inputs["attention_mask"].cpu()
        if args.remove_special_tokens:
            keep = mask.clone()
            keep[:, 0] = 0
            lengths = mask.sum(dim=1)
            for b in range(keep.shape[0]):
                eos = int(lengths[b].item()) - 1
                if eos > 0:
                    keep[b, eos] = 0
            mask = keep
        return hidden[mask.bool()]


def init_pre_bias_from_stream(sae: torch.nn.Module, producer: ESM2ActivationProducer, sample_size: int) -> None:
    """Initialize SAE pre_bias from the first streamed activation rows."""
    rows = []
    n_rows = 0
    for chunk in producer():
        rows.append(chunk)
        n_rows += chunk.shape[0]
        if n_rows >= sample_size:
            break
    if not rows:
        raise RuntimeError("No activation rows produced for pre-bias initialization")
    sample = torch.cat(rows, dim=0)[:sample_size].float()
    sae.init_pre_bias_from_data(sample)
    print(f"pre_bias initialized from {len(sample):,} streamed activation rows")


def main() -> None:
    """Run streaming SAE training."""
    args = parse_args()
    if args.dp_size != 1:
        raise ValueError("train_streaming.py currently supports only --dp-size 1; use one Lepton GPU for this path.")

    set_seed(args.seed)
    fasta_path = resolve_fasta(args)
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    input_dim = getattr(cfg, "hidden_size", None)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    if input_dim is None or num_layers is None:
        raise ValueError(f"Model config for {args.model_name} must expose hidden_size and num_hidden_layers")
    input_dim = int(input_dim)
    num_layers = int(num_layers)
    if args.layer < 0 or args.layer > num_layers:
        raise ValueError(f"--layer {args.layer} out of range [0, {num_layers}]")

    device = args.device or get_device()
    sae = build_sae(args, input_dim)
    print(f"SAE: {args.model_type}, input_dim={input_dim}, hidden_dim={sae.hidden_dim}")

    producer = ESM2ActivationProducer(args, fasta_path)
    if args.init_pre_bias and hasattr(sae, "init_pre_bias_from_data"):
        init_pre_bias_from_stream(sae, producer, args.pre_bias_sample_size)

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
