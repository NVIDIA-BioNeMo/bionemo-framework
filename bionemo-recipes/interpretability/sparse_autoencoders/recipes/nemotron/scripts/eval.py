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

r"""Step 3: Evaluate a trained SAE on Nemotron-3-Nano.

Loads an SAE checkpoint written by ``train.py``, runs it over a fresh batch of
FineWeb activations, and reports reconstruction quality, sparsity, dead latents,
and cross-entropy *loss recovered* (logits with vs. without the SAE spliced into
the residual stream).

Run on the GPUs only -- it reloads the 30B model for the loss-recovered metric.

This is step 3 of the 3-step Nemotron SAE workflow:
    1. extract.py  -- extract activations from Nemotron-3-Nano
    2. train.py    -- train SAE on cached activations
    3. eval.py     -- evaluate SAE (reconstruction + loss recovered)  (this file)

Usage:
    # Uses checkpoint.dir/checkpoint_final.pt by default
    python scripts/eval.py checkpoint.dir=outputs/k32_8x/checkpoints

    # Or point at a specific checkpoint
    python scripts/eval.py eval.checkpoint=outputs/k32_8x/checkpoints/checkpoint_final.pt
"""

from pathlib import Path

import hydra
import torch
from nemotron_sae.data import load_fineweb
from nemotron_sae.eval import evaluate_nemotron_loss_recovered
from nemotron_sae.models import NemotronModel
from omegaconf import DictConfig, OmegaConf
from sae.architectures import ReLUSAE, TopKSAE
from sae.eval import evaluate_sae
from sae.utils import get_device, set_seed


def load_sae_from_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
    """Rebuild an SAE from a Trainer checkpoint (handles DDP ``module.`` prefix)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    # Dims from checkpoint metadata, falling back to the encoder weight shape.
    input_dim = ckpt.get("input_dim")
    hidden_dim = ckpt.get("hidden_dim")
    if input_dim is None or hidden_dim is None:
        w = state_dict["encoder.weight"]
        hidden_dim = hidden_dim or w.shape[0]
        input_dim = input_dim or w.shape[1]

    mc = ckpt.get("model_config", {})
    if "top_k" in mc:
        sae = TopKSAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            top_k=mc["top_k"],
            normalize_input=mc.get("normalize_input", False),
            auxk=mc.get("auxk"),
            auxk_coef=mc.get("auxk_coef", 1 / 32),
            dead_tokens_threshold=mc.get("dead_tokens_threshold", 10_000_000),
        )
        print(f"Loaded TopKSAE: {input_dim} -> {hidden_dim:,} latents (top-{mc['top_k']})")
    else:
        sae = ReLUSAE(input_dim=input_dim, hidden_dim=hidden_dim, l1_coeff=mc.get("l1_coeff", 1e-2))
        print(f"Loaded ReLUSAE: {input_dim} -> {hidden_dim:,} latents")

    sae.load_state_dict(state_dict)
    return sae


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Evaluate a trained SAE checkpoint on Nemotron activations."""
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = cfg.device if cfg.device is not None else get_device()

    # --- Resolve checkpoint path (relative to original cwd) ---
    orig = Path(hydra.utils.get_original_cwd())
    eval_cfg = OmegaConf.select(cfg, "eval", default={}) or {}
    ckpt_override = eval_cfg.get("checkpoint", None)
    ckpt_dir = (orig / cfg.checkpoint.dir) if cfg.checkpoint.get("dir") else None

    # A tensor-parallel run writes a sharded checkpoint dir (meta.json + shard_*.pt);
    # merge it into a dense TopKSAE so the rest of eval is identical.
    if ckpt_dir is not None and (ckpt_dir / "meta.json").exists():
        from sae.parallel import load_and_merge

        print(f"Loading sharded (tensor-parallel) checkpoint from {ckpt_dir} and merging...")
        sae = load_and_merge(str(ckpt_dir)).to(device)
    else:
        if ckpt_override:
            checkpoint_path = orig / ckpt_override
        elif ckpt_dir is not None:
            checkpoint_path = ckpt_dir / "checkpoint_final.pt"
        else:
            raise ValueError(
                "No checkpoint to evaluate. Set eval.checkpoint=<path/to.pt> or "
                "checkpoint.dir=<dir with checkpoint_final.pt or sharded shard_*.pt>."
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        sae = load_sae_from_checkpoint(checkpoint_path)

    # --- Load Nemotron (needed for loss recovered) + fresh eval activations ---
    print(f"Loading {cfg.activations.model_name} (layer {cfg.activations.layer})...")
    nemotron = NemotronModel(
        model_name=cfg.activations.model_name,
        layer=cfg.activations.layer,
        device=device,
        max_length=cfg.activations.max_length,
    )

    eval_texts = load_fineweb(
        split=cfg.data.get("split", "train"),
        max_samples=eval_cfg.get("max_samples", 200),
        min_length=cfg.data.get("min_length", 50),
        subset=cfg.data.get("subset", "sample-10BT"),
    )
    eval_embeddings, eval_masks = nemotron.generate_activations(
        texts=eval_texts, batch_size=cfg.activations.batch_size
    )
    eval_activations_flat = eval_embeddings[eval_masks.bool()]
    print(f"Eval on {eval_activations_flat.shape[0]} tokens")

    # --- Loss recovered (reuses the loaded CausalLM) ---
    loss_recovered_fn = None
    if eval_cfg.get("run_loss_recovered", True):
        eval_seqs = eval_texts[: eval_cfg.get("loss_recovered_n_sequences", 100)]
        loss_recovered_fn = lambda: evaluate_nemotron_loss_recovered(  # noqa: E731
            sae=sae,
            model=nemotron.model,
            tokenizer=nemotron.tokenizer,
            texts=eval_seqs,
            layer_idx=cfg.activations.layer,
            device=device,
        )

    results = evaluate_sae(
        sae,
        eval_activations_flat,
        batch_size=cfg.training.batch_size,
        device=device,
        loss_recovered_fn=loss_recovered_fn,
    )
    results.print_summary()

    output_dir = orig / cfg.get("output_dir", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    results.save(output_dir / "eval_results.json")
    print(f"Saved eval results to {output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
