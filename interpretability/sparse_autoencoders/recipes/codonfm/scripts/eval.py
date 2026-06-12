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

"""Step 3: Evaluate CodonFM SAE (loss recovered).

Loads a trained SAE checkpoint and evaluates loss recovered against
the Encodon model. F1 and dashboard generation are deferred to future work.

IMPORTANT: Run on a single GPU. Do NOT use torchrun.

    python scripts/eval.py \
        --checkpoint ./outputs/encodon_1b/checkpoints/checkpoint_final.pt \
        --model-path path/to/encodon_1b \
        --layer -2 --top-k 32 \
        --csv-path path/to/data.csv \
        --output-dir ./outputs/encodon_1b/eval
"""

import argparse
import json
import sys
from pathlib import Path

import torch


# Use codonfm_ptl_te recipe (has TransformerEngine support)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
_CODONFM_TE_DIR = _REPO_ROOT / "recipes" / "codonfm_ptl_te"
sys.path.insert(0, str(_CODONFM_TE_DIR))

from codonfm_sae.data import read_codon_csv  # noqa: E402
from codonfm_sae.eval import evaluate_codonfm_loss_recovered  # noqa: E402
from sae.architectures import TopKSAE  # noqa: E402
from sae.utils import get_device, set_seed  # noqa: E402
from src.inference.encodon import EncodonInference  # noqa: E402


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Evaluate CodonFM SAE")

    # Checkpoint
    p.add_argument("--checkpoint", type=str, required=True, help="Path to SAE checkpoint .pt file")
    p.add_argument("--top-k", type=int, default=None, help="Override top-k (default: read from checkpoint)")

    # Model
    p.add_argument("--model-path", type=str, required=True, help="Path to Encodon checkpoint")
    p.add_argument("--layer", type=int, default=-2)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)

    # Data
    p.add_argument("--csv-path", type=str, required=True, help="CSV with DNA sequences for evaluation")
    p.add_argument("--seq-column", type=str, default=None)
    p.add_argument("--num-sequences", type=int, default=100)

    # Output
    p.add_argument("--output-dir", type=str, default="./outputs/eval")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_sae_from_checkpoint(checkpoint_path: str, top_k_override: int | None = None) -> TopKSAE:
    """Load SAE from a Trainer checkpoint, handling DDP module. prefix."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    input_dim = ckpt.get("input_dim")
    hidden_dim = ckpt.get("hidden_dim")
    if input_dim is None or hidden_dim is None:
        w = state_dict["encoder.weight"]
        hidden_dim = hidden_dim or w.shape[0]
        input_dim = input_dim or w.shape[1]

    model_config = ckpt.get("model_config", {})
    normalize_input = model_config.get("normalize_input", False)

    top_k = top_k_override or model_config.get("top_k")
    if top_k is None:
        raise ValueError("top_k not found in checkpoint. Pass --top-k explicitly.")
    if top_k_override and model_config.get("top_k") and top_k_override != model_config["top_k"]:
        print(f"  WARNING: overriding checkpoint top_k={model_config['top_k']} with --top-k={top_k_override}")

    sae = TopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        top_k=top_k,
        normalize_input=normalize_input,
    )
    sae.load_state_dict(state_dict)

    print(f"Loaded SAE: {input_dim} -> {hidden_dim:,} latents (top-{top_k}, normalize_input={normalize_input})")
    return sae


def main():  # noqa: D103
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SAE
    sae = load_sae_from_checkpoint(args.checkpoint, top_k_override=args.top_k)

    # 2. Load Encodon model
    print(f"Loading Encodon from {args.model_path}...")
    inference = EncodonInference(
        model_path=args.model_path, task_type="embedding_prediction", use_transformer_engine=True
    )
    inference.configure_model()
    inference.model.to(device).eval()

    num_layers = len(inference.model.model.layers)
    target_layer = args.layer if args.layer >= 0 else num_layers + args.layer
    print(f"  Layers: {num_layers}, Target layer: {target_layer}, Hidden: {inference.model.model.config.hidden_size}")

    # 3. Load sequences
    max_codons = args.context_length - 2
    records = read_codon_csv(
        args.csv_path,
        seq_column=args.seq_column,
        max_sequences=args.num_sequences,
        max_codons=max_codons,
    )
    sequences = [r.sequence for r in records]
    print(f"Loaded {len(sequences)} sequences for evaluation")

    # 4. Loss recovered
    print("\n" + "=" * 60)
    print("LOSS RECOVERED EVALUATION")
    print("=" * 60)

    result = evaluate_codonfm_loss_recovered(
        sae=sae,
        inference=inference,
        sequences=sequences,
        layer=args.layer,
        context_length=args.context_length,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )

    print(f"  Loss recovered: {result.loss_recovered:.4f}")
    print(f"  CE original:    {result.ce_original:.4f}")
    print(f"  CE SAE:         {result.ce_sae:.4f}")
    print(f"  CE zero:        {result.ce_zero:.4f}")
    print(f"  Tokens:         {result.n_tokens:,}")

    # Save
    lr_path = output_dir / "loss_recovered.json"
    with open(lr_path, "w") as f:
        json.dump(
            {
                "loss_recovered": result.loss_recovered,
                "ce_original": result.ce_original,
                "ce_sae": result.ce_sae,
                "ce_zero": result.ce_zero,
                "n_tokens": result.n_tokens,
            },
            f,
            indent=2,
        )
    print(f"Saved to {lr_path}")

    del inference
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
