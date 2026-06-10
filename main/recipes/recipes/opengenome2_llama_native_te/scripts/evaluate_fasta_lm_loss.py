# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""Evaluate a checkpoint on a pre-sampled FASTA file (per-sequence log probs).

This matches the evaluation methodology used by John in the Evo2 project:

1. Pre-sample sequences from the test set into a FASTA file (done once,
   shared across all models being compared).
2. For each model checkpoint, tokenize the sequences and compute per-token
   log probabilities via ``log_softmax`` + ``gather``.
3. Mask non-ACGT (degenerate) bases so they do not affect the score.
4. Report **per-sequence mean log probability** and aggregate statistics.

Because every model sees the exact same sequences in the exact same order,
the per-sequence metrics are directly comparable regardless of how the
models' *training* data was ordered.

Supports checkpoint formats:
  - FSDP2 DCP checkpoints  (from train_fsdp2.py — the default)
  - DDP  checkpoints        (checkpoint.pt files from train_ddp.py)
  - Consolidated safetensors (final_model/ directories)

Example usage::

    cd bionemo-recipes/recipes/opengenome2_llama_native_te/scripts

    # ── Model 1: HF window-shuffle ───────────────────────────────────────
    torchrun --nproc_per_node=1 evaluate_fasta_lm_loss.py \
        --checkpoint-dir /data/checkpoints/og2-7b-model-1 \
        --checkpoint-step 30000 \
        --fasta metagenomics.fasta \
        --output /data/eval_results/model1_fasta_eval.json

    # ── Model 2: Eden dataset ────────────────────────────────────────────
    torchrun --nproc_per_node=1 evaluate_fasta_lm_loss.py \
        --checkpoint-dir /data/checkpoints/og2-7b-model-2 \
        --checkpoint-step 30000 \
        --fasta metagenomics.fasta \
        --output /data/eval_results/model2_fasta_eval.json

    # ── Compare ──────────────────────────────────────────────────────────
    python -c "
    import json, statistics
    hf   = json.load(open('/path/to/model1_fasta_eval.json'))
    eden = json.load(open('/path/to/model2_fasta_eval.json'))
    print()
    print('=== Per-sequence log-prob comparison (FASTA-based) ===')
    for tag, r in [('HF Window-Shuffle', hf), ('Eden Dataset', eden)]:
        a = r['aggregate']
        print(f'  {tag:20s}:  loss={a[\"token_weighted_ce_loss\"]:.4f}  '
              f'ppl={a[\"token_weighted_perplexity\"]:.2f}  '
              f'tokens={a[\"total_valid_tokens\"]:,}')
    diff = abs(hf['aggregate']['token_weighted_ce_loss']
               - eden['aggregate']['token_weighted_ce_loss'])
    print(f'  Loss difference:      {diff:.4f}')
    "
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import transformer_engine.pytorch
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer


# Add parent directory to path so we can import recipe modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checkpoint import AppState, LenientLoadPlanner
from distributed_config import DistributedConfig
from opengenome_modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from scheduler import get_cosine_annealing_schedule_with_warmup


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Valid DNA token IDs (A, C, G, T upper + lowercase) — matches training masking
DNA_TOKENS = frozenset({65, 67, 71, 84, 97, 99, 103, 116})


# ---------------------------------------------------------------------------
# FASTA dataset
# ---------------------------------------------------------------------------


class FastaDataset(Dataset):
    """Dataset that reads a FASTA file and tokenises each sequence for LM eval.

    Each ``__getitem__`` returns a dictionary with:
      - ``input_ids`` - tokenised sequence, right-padded to *max_seq_length*
      - ``attention_mask`` - 1 for real tokens, 0 for padding
      - ``loss_mask`` - 1 for valid DNA tokens, 0 for degenerate/padding
      - ``seq_idx`` - integer index of the sequence in the file
      - ``seq_name`` - FASTA header (without the leading ``>``)
    """

    def __init__(  # noqa: D107
        self,
        fasta_path: str | Path,
        tokenizer_name_or_path: str,
        max_seq_length: int = 8192,
        mask_degenerate_bases: bool = True,
    ):
        self.fasta_path = Path(fasta_path)
        self.max_seq_length = max_seq_length
        self.mask_degenerate_bases = mask_degenerate_bases

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pre-compute a tensor of valid DNA token IDs for fast masking
        self._dna_tokens_t = torch.tensor(sorted(DNA_TOKENS), dtype=torch.long)

        self.names: list[str] = []
        self.sequences: list[str] = []
        self._parse_fasta()
        logger.info("Loaded %d sequences from %s", len(self.names), self.fasta_path)

    # ------------------------------------------------------------------

    def _parse_fasta(self) -> None:
        current_name: str | None = None
        parts: list[str] = []
        with open(self.fasta_path) as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_name is not None:
                        self.names.append(current_name)
                        self.sequences.append("".join(parts))
                    current_name = line[1:]
                    parts = []
                else:
                    parts.append(line)
            if current_name is not None:
                self.names.append(current_name)
                self.sequences.append("".join(parts))

    def __len__(self) -> int:  # noqa: D105
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:  # noqa: D105
        seq = self.sequences[idx]

        encoded = self.tokenizer(
            seq,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids: torch.Tensor = encoded["input_ids"].squeeze(0)
        attention_mask: torch.Tensor = encoded["attention_mask"].squeeze(0)

        # Loss mask: keep only valid DNA positions (and exclude padding)
        if self.mask_degenerate_bases:
            is_dna = torch.isin(input_ids, self._dna_tokens_t)
            loss_mask = (attention_mask.bool() & is_dna).long()
        else:
            loss_mask = attention_mask.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "seq_idx": idx,
            "seq_name": self.names[idx],
        }


def _fasta_collate(features: list[dict]) -> dict:
    """Stack tensor fields and keep metadata as lists."""
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "loss_mask": torch.stack([f["loss_mask"] for f in features]),
        "seq_idx": torch.tensor([f["seq_idx"] for f in features]),
        "seq_name": [f["seq_name"] for f in features],
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers (mirrors evaluate_test_loss.py)
# ---------------------------------------------------------------------------


def find_checkpoint_path(checkpoint_dir: str, step: int | None = None) -> tuple[Path, str]:
    """Locate the checkpoint inside *checkpoint_dir* and return ``(path, type)``."""
    root = Path(checkpoint_dir)

    # 1. safetensors
    for candidate in [root, root / "final_model", root / "train_fsdp2" / "final_model"]:
        if (candidate / "model.safetensors").exists():
            return candidate, "safetensors"

    # 2. FSDP2 DCP step directories
    fsdp2_dir = root / "train_fsdp2" if (root / "train_fsdp2").exists() else root
    step_dirs = sorted(
        [d for d in fsdp2_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if step_dirs:
        if step is not None:
            target = fsdp2_dir / f"step_{step}"
            if not target.exists():
                raise FileNotFoundError(f"step_{step} not found. Available: {[d.name for d in step_dirs]}")
            chosen = target
        else:
            chosen = step_dirs[-1]
        if (chosen / ".metadata").exists() or any(chosen.glob("*.distcp")):
            return chosen, "dcp"
        if (chosen / "checkpoint.pt").exists():
            return chosen, "ddp"
        return chosen, "dcp"

    # 3. root itself
    if (root / "checkpoint.pt").exists():
        return root, "ddp"
    if (root / ".metadata").exists() or any(root.glob("*.distcp")):
        return root, "dcp"

    raise FileNotFoundError(f"No recognisable checkpoint in {checkpoint_dir}")


# ---------------------------------------------------------------------------
# Model building + loading
# ---------------------------------------------------------------------------


def _build_model_config(config_name_or_path: str, num_kv_heads: int = 8) -> NVLlamaConfig:
    """Build the 7B config. Use num_kv_heads=8 for GQA, 32 for MHA (older models)."""
    return NVLlamaConfig.from_pretrained(
        config_name_or_path,
        dtype=torch.float32,
        vocab_size=256,
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=8192,
        initializer_range=0.02,
        attn_input_format="bshd",
        self_attn_mask_type="causal",
        rope_theta=500000,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 1,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
            "original_max_position_embeddings": 8192,
        },
    )


def load_model_from_checkpoint(
    ckpt_path: Path,
    ckpt_type: str,
    config: NVLlamaConfig,
    dist_config: DistributedConfig,
    device_mesh,
) -> NVLlamaForCausalLM:
    """Create, FSDP2-shard, and load weights from *ckpt_path*.

    For safetensors/DDP checkpoints (full state dicts of plain Tensors), weights
    are loaded *before* FSDP2 wrapping to avoid DTensor/Tensor mixing errors in
    newer PyTorch.  DCP checkpoints are loaded *after* wrapping because DCP
    handles DTensor resharding internally.
    """
    model = NVLlamaForCausalLM(config)
    if dist_config.rank == 0:
        logger.info("Model created (%s parameters)", f"{sum(p.numel() for p in model.parameters()):,}")

    # --- Load full state dicts BEFORE FSDP2 wrapping (plain Tensor → plain Parameter) ---
    if ckpt_type == "safetensors":
        if dist_config.rank == 0:
            logger.info("Loading safetensors from %s …", ckpt_path)
        from safetensors.torch import load_file

        weights = load_file(str(ckpt_path / "model.safetensors"))
        model.load_state_dict(weights, strict=False)
        if dist_config.rank == 0:
            logger.info("Safetensors loaded")

    elif ckpt_type == "ddp":
        if dist_config.rank == 0:
            logger.info("Loading DDP checkpoint from %s …", ckpt_path)
        ckpt = torch.load(ckpt_path / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"], strict=False)
        if dist_config.rank == 0:
            logger.info("DDP checkpoint loaded (step=%d)", ckpt.get("step", -1))

    # --- FSDP2 wrapping ---
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=False,
    )
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    # --- Load DCP checkpoints AFTER FSDP2 wrapping (DCP handles DTensor resharding) ---
    if ckpt_type == "dcp":
        if dist_config.rank == 0:
            logger.info("Loading FSDP2 DCP checkpoint from %s …", ckpt_path)
        from torch.optim import AdamW

        optimizer = AdamW(model.parameters(), lr=1e-5)
        scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, num_warmup_steps=100, num_decay_steps=1000)
        app_state = AppState(model=model, optimizer=optimizer, scheduler=scheduler)
        dcp_load(
            {"app": app_state},
            checkpoint_id=ckpt_path,
            process_group=device_mesh.get_group("dp"),
            planner=LenientLoadPlanner(),
        )
        if dist_config.rank == 0:
            logger.info("DCP checkpoint loaded (step=%d, epoch=%d)", app_state.step, app_state.epoch)

    elif ckpt_type not in ("safetensors", "ddp"):
        raise ValueError(f"Unknown checkpoint type: {ckpt_type}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core: per-sequence log-probability computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_per_sequence_log_probs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dist_config: DistributedConfig,
) -> list[dict]:
    r"""Compute per-sequence mean log-probability (John's method).

    For each sequence the procedure is:

    1. Forward pass → logits  ``(B, S, V)``
    2. Shift:  ``logits[:, :-1]``  predicts  ``tokens[:, 1:]``
    3. ``log_softmax`` over vocab → per-position log-probs
    4. ``gather`` the log-prob of the *actual* next token
    5. Multiply by ``loss_mask[:, 1:]``  (0 for degenerate bases & padding)
    6. ``mean_log_prob = sum(masked_log_probs) / num_valid_tokens``

    This is equivalent to ``-cross_entropy_loss`` per sequence.
    """
    model.eval()
    results: list[dict] = []

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        seq_indices = batch["seq_idx"]
        seq_names = batch["seq_name"]

        # Forward — no labels ⇒ model returns logits only, no loss
        with transformer_engine.pytorch.fp8_autocast(enabled=False):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits  # (B, S, V)

        # Shift: position i predicts token i+1
        shift_logits = logits[:, :-1, :].float()  # (B, S-1, V)
        shift_tokens = input_ids[:, 1:]  # (B, S-1)
        shift_mask = loss_mask[:, 1:].float()  # (B, S-1)

        # Per-token log-probabilities
        log_probs = torch.log_softmax(shift_logits, dim=-1)  # (B, S-1, V)
        token_log_probs = torch.gather(log_probs, 2, shift_tokens.unsqueeze(-1)).squeeze(-1)  # (B, S-1)

        # Apply mask
        masked_log_probs = token_log_probs * shift_mask

        # Per-sequence aggregates
        num_valid = shift_mask.sum(dim=1).clamp(min=1.0)  # (B,)
        sum_lp = masked_log_probs.sum(dim=1)  # (B,)
        mean_lp = sum_lp / num_valid  # (B,)

        results.extend(
            {
                "seq_idx": seq_indices[i].item(),
                "seq_name": seq_names[i],
                "mean_log_prob": mean_lp[i].item(),
                "sum_log_prob": sum_lp[i].item(),
                "num_valid_tokens": int(num_valid[i].item()),
                "cross_entropy_loss": -mean_lp[i].item(),
                "perplexity": math.exp(-mean_lp[i].item()),
                "seq_length": int(attention_mask[i].sum().item()),
            }
            for i in range(input_ids.size(0))
        )

        if batch_idx % 10 == 0 and dist_config.rank == 0:
            logger.info("  [batch %d]  %d sequences processed", batch_idx, len(results))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a pre-sampled FASTA file (per-sequence log probs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint-dir", type=str, required=True, help="Root checkpoint directory.")
    p.add_argument("--checkpoint-step", type=int, default=None, help="Specific step to load (latest if omitted).")
    p.add_argument("--fasta", type=str, required=True, help="Path to the FASTA file.")
    p.add_argument("--config-name-or-path", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--tokenizer", type=str, default="../tokenizers/nucleotide_fast_tokenizer")
    p.add_argument("--micro-batch-size", type=int, default=1, help="Batch size per GPU.")
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="Number of key-value heads. 8 = GQA (default), 32 = MHA (older models).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mask-degenerate-bases",
        action="store_true",
        default=True,
        help="Mask non-ACGT bases in loss (default: True — matches training).",
    )
    p.add_argument("--no-mask-degenerate-bases", action="store_false", dest="mask_degenerate_bases")
    p.add_argument("--output", type=str, default=None, help="Path to write results JSON.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    args = parse_args()

    # ── distributed setup ─────────────────────────────────────────────────
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    set_seed(args.seed)

    if dist_config.rank == 0:
        logger.info("=" * 70)
        logger.info("FASTA-based LM Loss Evaluation (per-sequence log probs)")
        logger.info("=" * 70)
        logger.info("  checkpoint_dir  : %s", args.checkpoint_dir)
        logger.info("  fasta           : %s", args.fasta)
        logger.info("  max_seq_length  : %d", args.max_seq_length)
        logger.info("  micro_batch_size: %d", args.micro_batch_size)
        logger.info("  mask_degenerate : %s", args.mask_degenerate_bases)
        logger.info("  seed            : %d", args.seed)
        logger.info("  world_size      : %d", dist_config.world_size)
        logger.info("=" * 70)

    # ── checkpoint ────────────────────────────────────────────────────────
    ckpt_path, ckpt_type = find_checkpoint_path(args.checkpoint_dir, args.checkpoint_step)
    if dist_config.rank == 0:
        logger.info("Resolved checkpoint: %s  (type=%s)", ckpt_path, ckpt_type)

    # ── model ─────────────────────────────────────────────────────────────
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("dp",))
    config = _build_model_config(args.config_name_or_path, num_kv_heads=args.num_kv_heads)
    model = load_model_from_checkpoint(ckpt_path, ckpt_type, config, dist_config, device_mesh)

    # ── dataset / dataloader ──────────────────────────────────────────────
    dataset = FastaDataset(
        fasta_path=args.fasta,
        tokenizer_name_or_path=args.tokenizer,
        max_seq_length=args.max_seq_length,
        mask_degenerate_bases=args.mask_degenerate_bases,
    )

    sampler: DistributedSampler | None = None
    if dist_config.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_config.world_size,
            rank=dist_config.rank,
            shuffle=False,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=_fasta_collate,
        num_workers=0,
        pin_memory=True,
    )

    # ── evaluation ────────────────────────────────────────────────────────
    if dist_config.rank == 0:
        logger.info("Computing per-sequence log probabilities for %d sequences …", len(dataset))

    local_results = compute_per_sequence_log_probs(model, dataloader, device, dist_config)

    # ── gather across ranks ───────────────────────────────────────────────
    if dist_config.world_size > 1:
        gathered: list[list[dict] | None] = [None] * dist_config.world_size
        dist.all_gather_object(gathered, local_results)
        if dist_config.rank == 0:
            all_results: list[dict] = []
            for rank_results in gathered:
                if rank_results is not None:
                    all_results.extend(rank_results)
            all_results.sort(key=lambda r: r["seq_idx"])
        else:
            all_results = []
    else:
        all_results = local_results

    # ── report + save ─────────────────────────────────────────────────────
    if dist_config.rank == 0 and all_results:
        total_valid = sum(r["num_valid_tokens"] for r in all_results)
        weighted_sum_lp = sum(r["sum_log_prob"] for r in all_results)
        weighted_mean_lp = weighted_sum_lp / max(total_valid, 1)
        weighted_ce = -weighted_mean_lp

        avg_ce = sum(r["cross_entropy_loss"] for r in all_results) / len(all_results)
        avg_lp = sum(r["mean_log_prob"] for r in all_results) / len(all_results)

        logger.info("=" * 70)
        logger.info("RESULTS  (%d sequences, %s valid tokens)", len(all_results), f"{total_valid:,}")
        logger.info("=" * 70)
        logger.info("  Token-weighted CE loss   : %.4f", weighted_ce)
        logger.info("  Token-weighted perplexity: %.2f", math.exp(weighted_ce))
        logger.info("  Token-weighted log-prob  : %.4f", weighted_mean_lp)
        logger.info("  Per-seq avg CE loss      : %.4f", avg_ce)
        logger.info("  Per-seq avg perplexity   : %.2f", math.exp(avg_ce))
        logger.info("  Per-seq avg log-prob     : %.4f", avg_lp)
        logger.info("=" * 70)

        logger.info("Per-sequence detail:")
        for r in all_results:
            logger.info(
                "  [%3d] %-55s  loss=%.4f  ppl=%.2f  log_prob=%.4f  tokens=%d/%d",
                r["seq_idx"],
                r["seq_name"][:55],
                r["cross_entropy_loss"],
                r["perplexity"],
                r["mean_log_prob"],
                r["num_valid_tokens"],
                r["seq_length"],
            )

        # ── save JSON ─────────────────────────────────────────────────────
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "checkpoint": str(ckpt_path),
                "checkpoint_type": ckpt_type,
                "fasta": str(args.fasta),
                "seed": args.seed,
                "max_seq_length": args.max_seq_length,
                "mask_degenerate_bases": args.mask_degenerate_bases,
                "world_size": dist_config.world_size,
                "aggregate": {
                    "token_weighted_ce_loss": weighted_ce,
                    "token_weighted_perplexity": math.exp(weighted_ce),
                    "token_weighted_log_prob": weighted_mean_lp,
                    "per_seq_avg_ce_loss": avg_ce,
                    "per_seq_avg_perplexity": math.exp(avg_ce),
                    "per_seq_avg_log_prob": avg_lp,
                    "total_valid_tokens": total_valid,
                    "total_sequences": len(all_results),
                },
                "per_sequence": all_results,
            }
            with open(out, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info("Results saved → %s", out)

    # ── cleanup ───────────────────────────────────────────────────────────
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
