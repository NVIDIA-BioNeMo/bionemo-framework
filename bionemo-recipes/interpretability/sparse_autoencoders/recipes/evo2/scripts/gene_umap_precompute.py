# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""End-to-end gene-UMAP precompute.

Reads genes.tsv -> Evo2-1B layer-20 activations -> SAE encode -> per-gene
mean-aggregated feature vector -> UMAP + HDBSCAN -> static outputs the
frontend ships to the browser.

Pipeline:
  1. genes.tsv -> FASTA with `>gene_{i} {symbol}` headers (idx = primary key).
  2. predict_evo2 (torchrun, 4 GPUs) writes per-batch .pt files at layer 20.
  3. Iterate .pt files, group rows by seq_idx -> per-gene [L, 1920] tensors.
  4. Load TopK SAE checkpoint, encode each gene's tokens -> [L, 15360],
     mean across positions -> [15360] per gene. Stack to G [n_genes, 15360].
  5. umap.UMAP on G (cosine) -> 2D coords. hdbscan on coords -> cluster_id.
  6. Per-feature stats: n_firing (count of genes where activation > 0),
     mean_act_when_firing. Filter for n_firing >= 10 at the frontend.

Outputs in --output-dir:
  G.npz                 # {'G': float32[n_genes, 15360], 'gene_symbols': str array}
  genes_umap.parquet    # gene_symbol, species, x, y, cluster_id
  feature_stats.parquet # feature_id, n_firing, mean_act_when_firing
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _build_fasta(genes_df: pd.DataFrame, fasta_path: Path) -> None:
    """Write genes_df rows to fasta with `>gene_{i} {symbol}` headers."""
    with open(fasta_path, "w") as f:
        for i, row in genes_df.iterrows():
            f.write(f">gene_{i} {row.gene_symbol}\n")
            seq = row.sequence
            # Wrap at 80 columns so predict_evo2's parser doesn't choke on
            # multi-megabase single-line records.
            for j in range(0, len(seq), 80):
                f.write(seq[j : j + 80] + "\n")


def _run_predict(
    fasta: Path,
    ckpt_dir: Path,
    predict_dir: Path,
    layer: int,
    n_gpus: int,
    master_port: int,
    venv_python: Path,
) -> None:
    """Invoke predict_evo2 via torchrun. Idempotent: skips if .pt files exist."""
    if any(predict_dir.glob("predictions__*.pt")):
        print(f"[predict] reusing existing .pt files in {predict_dir}")
        return
    predict_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        f"--master-port={master_port}",
        "--no-python",
        "predict_evo2",
        "--fasta",
        str(fasta),
        "--ckpt-dir",
        str(ckpt_dir),
        "--output-dir",
        str(predict_dir),
        "--embedding-layer",
        str(layer),
        "--micro-batch-size",
        "4",
        "--write-interval",
        "batch",
    ]
    print(f"[predict] launching: {' '.join(cmd)}")
    # Run with the evo2_megatron venv's python on PATH.
    env = {**__import__("os").environ, "PATH": f"{venv_python.parent}:{__import__('os').environ.get('PATH', '')}"}
    subprocess.run(cmd, check=True, env=env)


def _aggregate_per_gene(predict_dir: Path, n_genes: int) -> list[torch.Tensor]:
    """Walk predict_dir/.pt files, group rows by seq_idx, return one tensor per gene."""
    per_gene: list[list[torch.Tensor]] = [[] for _ in range(n_genes)]
    pt_files = sorted(predict_dir.glob("predictions__*.pt"))
    print(f"[aggregate] reading {len(pt_files)} .pt files")
    for pt_path in pt_files:
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        hidden = d["hidden_embeddings"]  # [B, S, H]
        mask = d["pad_mask"].bool()  # [B, S]
        seq_idx = d["seq_idx"]  # [B]
        for i in range(hidden.shape[0]):
            sid = int(seq_idx[i].item())
            if sid >= n_genes:
                continue
            unpadded = hidden[i][mask[i]].float()  # [L_i, H]
            per_gene[sid].append(unpadded)
    out = []
    for sid, parts in enumerate(per_gene):
        if not parts:
            print(f"[aggregate] WARN: gene {sid} has no tokens")
            out.append(torch.zeros(1, hidden.shape[-1]))
        else:
            out.append(torch.cat(parts, dim=0))
    return out


def _load_sae(ckpt_path: Path, device: str = "cuda") -> torch.nn.Module:
    """Load TopK SAE from a training checkpoint, stripping DDP `module.` prefix."""
    from sae.architectures.topk import TopKSAE

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    sae = TopKSAE(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        top_k=cfg["top_k"],
        normalize_input=cfg.get("normalize_input", False),
        auxk=cfg.get("auxk", 0),
        auxk_coef=cfg.get("auxk_coef", 0.0),
        dead_tokens_threshold=cfg.get("dead_tokens_threshold", 500_000),
    )
    state = {k.removeprefix("module."): v for k, v in ckpt["model_state_dict"].items()}
    sae.load_state_dict(state)
    sae.to(device).eval()
    print(
        f"[sae] loaded TopK SAE: input_dim={cfg['input_dim']} hidden_dim={cfg['hidden_dim']} top_k={cfg['top_k']}"
    )
    return sae


@torch.inference_mode()
def _encode_and_mean(sae: torch.nn.Module, hidden_per_gene: list[torch.Tensor], device: str = "cuda") -> np.ndarray:
    """Encode each gene's [L, 1920] hidden states -> [15360] mean SAE features."""
    n_genes = len(hidden_per_gene)
    hidden_dim = sae.hidden_dim if hasattr(sae, "hidden_dim") else 15360
    G = np.zeros((n_genes, hidden_dim), dtype=np.float32)
    for sid, hidden in enumerate(hidden_per_gene):
        if hidden.shape[0] == 0:
            continue
        x = hidden.to(device, non_blocking=True)
        encoded = sae.encode(x)  # [L, 15360], sparse via TopK
        G[sid] = encoded.mean(dim=0).cpu().numpy()
        if (sid + 1) % 50 == 0:
            print(f"[sae] encoded {sid + 1}/{n_genes} genes")
    return G


def _run_umap_and_cluster(G: np.ndarray, seed: int = 42) -> pd.DataFrame:
    """UMAP-cosine -> 2D, then HDBSCAN on the 2D coords -> cluster_id."""
    import umap

    print(f"[umap] running UMAP on {G.shape}")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
    coords = reducer.fit_transform(G)
    print(f"[umap] coords shape: {coords.shape}")

    import hdbscan

    print("[hdbscan] clustering 2D coords")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3)
    cluster_ids = clusterer.fit_predict(coords)
    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    print(f"[hdbscan] found {n_clusters} clusters (+ {(cluster_ids == -1).sum()} noise)")
    return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster_id": cluster_ids})


def _compute_feature_stats(G: np.ndarray) -> pd.DataFrame:
    """For each feature: n_firing genes + mean activation among firing genes."""
    n_firing = (G > 0).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean_act = np.where(n_firing > 0, G.sum(axis=0) / np.maximum(n_firing, 1), 0.0)
    return pd.DataFrame(
        {
            "feature_id": np.arange(G.shape[1]),
            "n_firing": n_firing,
            "mean_act_when_firing": mean_act,
        }
    )


def main():
    """Run the full gene-UMAP precompute pipeline end-to-end."""
    p = argparse.ArgumentParser()
    p.add_argument("--genes-tsv", type=Path, default=Path("/data/interp/evo2/scratch/fake_genes.tsv"))
    p.add_argument(
        "--sae-ckpt",
        type=Path,
        default=Path("/data/interp/evo2/sae/evo2_1b_base_layer20_25M_diverse_B/checkpoints/checkpoint_final.pt"),
    )
    p.add_argument(
        "--evo2-ckpt-dir", type=Path, default=Path("/data/interp/evo2/checkpoints/evo2_1b_base_mbridge")
    )
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--n-gpus", type=int, default=4)
    p.add_argument("--master-port", type=int, default=29502)
    p.add_argument("--n-limit", type=int, default=0, help="Only process first N genes (0 = all)")
    p.add_argument("--output-dir", type=Path, default=Path("/data/interp/evo2/sae/gene_umap"))
    p.add_argument(
        "--evo2-venv-python",
        type=Path,
        default=Path("/workspace/bionemo-framework/bionemo-recipes/recipes/evo2_megatron/.venv/bin/python"),
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workdir = args.output_dir / "_workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Load genes
    genes_df = pd.read_csv(args.genes_tsv, sep="\t")
    if args.n_limit > 0:
        genes_df = genes_df.head(args.n_limit).copy()
    genes_df = genes_df.reset_index(drop=True)
    print(f"[load] {len(genes_df)} genes from {args.genes_tsv}")

    # 2. FASTA
    fasta_path = workdir / "genes.fasta"
    _build_fasta(genes_df, fasta_path)
    print(f"[fasta] wrote {fasta_path}")

    # 3. predict_evo2 -> per-batch .pt files
    predict_dir = workdir / "predict"
    _run_predict(
        fasta=fasta_path,
        ckpt_dir=args.evo2_ckpt_dir,
        predict_dir=predict_dir,
        layer=args.layer,
        n_gpus=args.n_gpus,
        master_port=args.master_port,
        venv_python=args.evo2_venv_python,
    )

    # 4. Aggregate per gene
    per_gene_hidden = _aggregate_per_gene(predict_dir, n_genes=len(genes_df))
    lengths = [t.shape[0] for t in per_gene_hidden]
    print(f"[aggregate] gene token counts: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.0f}")

    # 5. SAE encode + mean
    sae = _load_sae(args.sae_ckpt)
    G = _encode_and_mean(sae, per_gene_hidden)
    print(f"[sae] G shape: {G.shape}, nonzero rate: {(G > 0).mean():.3%}")

    # 6. Save G
    g_path = args.output_dir / "G.npz"
    np.savez_compressed(g_path, G=G, gene_symbols=genes_df.gene_symbol.values)
    print(f"[save] {g_path}")

    # 7. UMAP + cluster
    umap_df = _run_umap_and_cluster(G)
    umap_df["gene_symbol"] = genes_df.gene_symbol.values
    umap_df["species"] = genes_df.species.values
    umap_df = umap_df[["gene_symbol", "species", "x", "y", "cluster_id"]]
    umap_path = args.output_dir / "genes_umap.parquet"
    umap_df.to_parquet(umap_path, index=False)
    print(f"[save] {umap_path}")

    # 8. Feature stats
    fstats = _compute_feature_stats(G)
    fstats_path = args.output_dir / "feature_stats.parquet"
    fstats.to_parquet(fstats_path, index=False)
    n_kept = (fstats.n_firing >= 10).sum()
    print(f"[save] {fstats_path} ({n_kept} features fire on >=10 genes)")

    # 9. Manifest
    manifest = {
        "n_genes": len(genes_df),
        "layer": args.layer,
        "sae_ckpt": str(args.sae_ckpt),
        "evo2_ckpt": str(args.evo2_ckpt_dir),
        "G_shape": list(G.shape),
        "n_clusters": int(umap_df.cluster_id.max() + 1) if (umap_df.cluster_id >= 0).any() else 0,
        "n_features_firing_ge_10": int(n_kept),
    }
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[save] {args.output_dir / 'manifest.json'}")
    print("[done] precompute complete")


if __name__ == "__main__":
    main()
