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

"""Evaluate CodonFM SAE features via gene-level GSEA enrichment.

For each SAE feature, ranks genes by activation strength and runs GSEA
against GO, InterPro, and Pfam databases to identify biologically
meaningful feature labels.

IMPORTANT: Run on a single GPU. Do NOT use torchrun.

    python scripts/eval_gene_enrichment.py \
        --checkpoint ./outputs/1b_layer16/checkpoints/checkpoint_final.pt \
        --model-path checkpoints/NV-CodonFM-Encodon-TE-Cdwt-1B-v1/model.safetensors \
        --layer 16 \
        --csv-path /path/to/genes.csv \
        --output-dir ./outputs/1b_layer16/gene_enrichment
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# Use codonfm_ptl_te recipe (has TransformerEngine support)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
_CODONFM_TE_DIR = _REPO_ROOT / "recipes" / "codonfm_ptl_te"
sys.path.insert(0, str(_CODONFM_TE_DIR))

from codonfm_sae.data import read_codon_csv  # noqa: E402
from codonfm_sae.eval.gene_enrichment import (  # noqa: E402
    ANNOTATION_DATABASES,
    GeneEnrichmentReport,
    compute_feature_pli,
    detect_gene_families,
    download_obo_files,
    load_pli_scores,
    rollup_go_slim,
    run_gene_enrichment,
)
from sae.architectures import TopKSAE  # noqa: E402
from sae.utils import get_device, set_seed  # noqa: E402
from src.data.preprocess.codon_sequence import process_item  # noqa: E402
from src.inference.encodon import EncodonInference  # noqa: E402


# ── SAE loading (duplicated from eval_swissprot_f1.py — KISS > DRY) ─────


def load_sae_from_checkpoint(checkpoint_path: str, top_k_override: Optional[int] = None) -> TopKSAE:
    """Load SAE from a Trainer checkpoint."""
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

    sae = TopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        top_k=top_k,
        normalize_input=normalize_input,
    )
    sae.load_state_dict(state_dict)
    print(f"Loaded SAE: {input_dim} -> {hidden_dim:,} latents (top-{top_k})")
    return sae


# ── Activation extraction (duplicated from eval_swissprot_f1.py) ────────


def extract_activations_3d(
    inference: "EncodonInference",
    sequences: List[str],
    layer: int,
    context_length: int = 2048,
    batch_size: int = 1,
    device: str = "cuda",
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 3D activations from CodonFM for codon sequences.

    Returns:
        (activations, masks) where:
        - activations: (n_sequences, max_codon_len, hidden_dim) float32, padded
        - masks: (n_sequences, max_codon_len) bool, 1=valid codon position
    """
    all_embeddings = []
    all_masks = []

    n_batches = (len(sequences) + batch_size - 1) // batch_size
    iterator = range(0, len(sequences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="Extracting activations")

    with torch.no_grad():
        for i in iterator:
            batch_seqs = sequences[i : i + batch_size]
            items = [process_item(s, context_length=context_length, tokenizer=inference.tokenizer) for s in batch_seqs]

            batch = {
                "input_ids": torch.tensor(np.stack([it["input_ids"] for it in items])).to(device),
                "attention_mask": torch.tensor(np.stack([it["attention_mask"] for it in items])).to(device),
            }

            out = inference.model(batch, return_hidden_states=True)
            layer_acts = out.all_hidden_states[layer]  # [B, L, D]

            for j, it in enumerate(items):
                seq_len = it["attention_mask"].sum()
                # Strip CLS (pos 0) and SEP (last real pos)
                acts = layer_acts[j, 1 : seq_len - 1, :].float().cpu()  # [n_codons, D]
                n_codons = acts.shape[0]
                mask = torch.ones(n_codons, dtype=torch.bool)
                all_embeddings.append(acts)
                all_masks.append(mask)

            del out, layer_acts, batch

    # Pad to same length for 3D stacking
    max_len = max(e.shape[0] for e in all_embeddings)
    hidden_dim = all_embeddings[0].shape[1]

    padded_emb = []
    padded_masks = []
    for emb, msk in zip(all_embeddings, all_masks):
        L = emb.shape[0]
        if L < max_len:
            emb = torch.cat([emb, torch.zeros(max_len - L, hidden_dim)], dim=0)
            msk = torch.cat([msk, torch.zeros(max_len - L, dtype=torch.bool)])
        padded_emb.append(emb.unsqueeze(0))
        padded_masks.append(msk.unsqueeze(0))

    return torch.cat(padded_emb, dim=0), torch.cat(padded_masks, dim=0)


# ── Gene-level activation computation ───────────────────────────────────


def compute_gene_activations(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    masks: torch.Tensor,
    gene_names: List[str],
    device: str = "cuda",
    show_progress: bool = True,
) -> Dict[int, Dict[str, float]]:
    """Compute per-gene, per-feature max activations via streaming SAE encode.

    For each sequence, encodes through SAE and takes the max activation per
    feature across all valid codon positions. Then groups by gene name and
    takes the max across sequences within each gene.

    Args:
        sae: Trained SAE model.
        activations: (n_sequences, max_len, hidden_dim) padded activations.
        masks: (n_sequences, max_len) bool masks.
        gene_names: Gene name for each sequence (len == n_sequences).
        device: Device for SAE encoding.
        show_progress: Whether to show progress bar.

    Returns:
        feature_idx -> gene_name -> max activation score.
    """
    sae = sae.eval().to(device)
    n_sequences = activations.shape[0]
    n_features = sae.hidden_dim

    # Phase 1: Compute per-sequence max activations -> (n_sequences, n_features)
    # Stream one sequence at a time to avoid OOM
    seq_max_acts = np.zeros((n_sequences, n_features), dtype=np.float32)

    iterator = range(n_sequences)
    if show_progress:
        iterator = tqdm(iterator, desc="SAE encode (per-sequence max)")

    with torch.no_grad():
        for seq_idx in iterator:
            emb = activations[seq_idx].to(device)  # (max_len, hidden_dim)
            mask = masks[seq_idx].numpy().astype(bool)
            sae_acts = sae.encode(emb).cpu().numpy()  # (max_len, n_features)

            # Apply mask
            seq_len = min(len(mask), sae_acts.shape[0])
            valid_acts = sae_acts[:seq_len][mask[:seq_len]]

            if valid_acts.shape[0] > 0:
                seq_max_acts[seq_idx] = valid_acts.max(axis=0)

    # Phase 2: Group by gene and take max
    df = pd.DataFrame(seq_max_acts)
    df["gene"] = gene_names
    gene_max = df.groupby("gene").max()  # (n_genes, n_features)

    # Convert to dict[feature_idx, dict[gene, score]]
    result: Dict[int, Dict[str, float]] = {}
    for feat_idx in range(n_features):
        col = gene_max[feat_idx]
        # Only include genes with non-zero activation
        nonzero = col[col > 0]
        if len(nonzero) > 0:
            result[feat_idx] = nonzero.to_dict()

    return result


# ── Report serialization ────────────────────────────────────────────────


def _enrichment_result_to_dict(er):
    """Convert EnrichmentResult to JSON-serializable dict."""
    if er is None:
        return None
    return asdict(er)


def save_report_json(report: GeneEnrichmentReport, path: Path):
    """Save GeneEnrichmentReport to JSON."""
    data = {
        "databases_used": report.databases_used,
        "n_features_with_enrichment": report.n_features_with_enrichment,
        "n_features_total": report.n_features_total,
        "frac_enriched": report.frac_enriched,
        "per_database_stats": report.per_database_stats,
        "significance_threshold": report.significance_threshold,
        "per_feature": [],
    }

    for fl in report.per_feature:
        entry = {
            "feature_idx": fl.feature_idx,
            "overall_best": _enrichment_result_to_dict(fl.overall_best),
            "go_slim_term": fl.go_slim_term,
            "go_slim_name": fl.go_slim_name,
            "best_per_database": {db: _enrichment_result_to_dict(er) for db, er in fl.best_per_database.items()},
            "n_significant": len(fl.all_significant),
        }
        data["per_feature"].append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_enrichment_parquet(report: GeneEnrichmentReport, path: Path):
    """Save all significant enrichment results to a parquet file."""
    rows = []
    for fl in report.per_feature:
        for er in fl.all_significant:
            rows.append(asdict(er))

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False, engine="pyarrow")
    else:
        # Empty parquet with correct schema
        df = pd.DataFrame(
            columns=[
                "feature_idx",
                "term_id",
                "term_name",
                "database",
                "enrichment_score",
                "pvalue",
                "fdr",
                "n_genes_in_term",
            ]
        )
        df.to_parquet(path, index=False, engine="pyarrow")


def update_atlas_with_gsea(report: GeneEnrichmentReport, atlas_path: Path):
    """Append gsea_* columns to features_atlas.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not atlas_path.exists():
        print(f"  WARNING: {atlas_path} does not exist, skipping atlas update")
        return

    table = pq.read_table(atlas_path)
    n = table.num_rows

    for col_name, label_dict in report.feature_label_columns.items():
        parquet_col = f"gsea_{col_name}"
        values = [label_dict.get(i, "unlabeled") for i in range(n)]

        # Drop existing column if present
        if parquet_col in table.column_names:
            table = table.drop(parquet_col)
        table = table.append_column(parquet_col, pa.array(values))

    pq.write_table(table, atlas_path, compression="snappy")
    print(f"  Updated {atlas_path} with {len(report.feature_label_columns)} GSEA columns")


def update_atlas_with_pli(feature_pli: Dict[int, Dict[str, float]], atlas_path: Path):
    """Append pLI metric columns to features_atlas.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not atlas_path.exists():
        print(f"  WARNING: {atlas_path} does not exist, skipping pLI atlas update")
        return

    table = pq.read_table(atlas_path)
    n = table.num_rows

    for metric in ["mean_pli", "frac_constrained", "max_pli"]:
        col_name = f"pli_{metric}"
        values = [feature_pli.get(i, {}).get(metric) for i in range(n)]

        if col_name in table.column_names:
            table = table.drop(col_name)
        table = table.append_column(col_name, pa.array(values, type=pa.float32()))

    pq.write_table(table, atlas_path, compression="snappy")
    print(f"  Updated {atlas_path} with 3 pLI columns")


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate CodonFM SAE features via gene-level GSEA enrichment")

    # SAE checkpoint
    p.add_argument("--checkpoint", type=str, required=True, help="Path to SAE checkpoint .pt file")
    p.add_argument("--top-k", type=int, default=None, help="Override top-k (default: read from checkpoint)")

    # Encodon model
    p.add_argument("--model-path", type=str, required=True, help="Path to Encodon checkpoint (.safetensors)")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)

    # Data
    p.add_argument("--csv-path", type=str, required=True, help="CSV with gene sequences (must have 'gene' column)")
    p.add_argument("--num-sequences", type=int, default=None, help="Max sequences to process (default: all)")

    # GSEA parameters
    p.add_argument("--n-workers", type=int, default=4, help="Parallel workers for GSEA")
    p.add_argument("--fdr-threshold", type=float, default=0.05, help="FDR threshold for significance")
    p.add_argument(
        "--databases",
        type=str,
        nargs="+",
        default=None,
        help="Enrichr library names (default: GO + InterPro + Pfam)",
    )

    # GO Slim
    p.add_argument("--no-go-slim", action="store_true", help="Skip GO Slim rollup")
    p.add_argument("--obo-dir", type=str, default=None, help="Directory for OBO files (default: output-dir/obo)")

    # Output
    p.add_argument("--output-dir", type=str, default="./outputs/gene_enrichment")
    p.add_argument(
        "--dashboard-dir",
        type=str,
        default=None,
        help="If provided, updates features_atlas.parquet with GSEA columns",
    )

    # pLI scores
    p.add_argument(
        "--pli-path",
        type=str,
        default=None,
        help="Path to gnomAD pLI constraint file (gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz)",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    """Run gene-level GSEA enrichment evaluation."""
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    databases = args.databases or list(ANNOTATION_DATABASES)

    # 1. Load SAE
    print("\n" + "=" * 60)
    print("LOADING SAE")
    print("=" * 60)
    sae = load_sae_from_checkpoint(args.checkpoint, top_k_override=args.top_k)

    # 2. Load CSV and extract gene names
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    records = read_codon_csv(args.csv_path, max_sequences=args.num_sequences)
    print(f"Loaded {len(records)} sequences")

    # Extract gene names from records
    gene_names = []
    valid_records = []
    for rec in records:
        gene = rec.metadata.get("gene")
        if gene is not None and str(gene).strip():
            gene_names.append(str(gene).strip())
            valid_records.append(rec)

    if not valid_records:
        print("ERROR: No sequences with gene names found. CSV must have a 'gene' column.")
        return

    print(f"  {len(valid_records)} sequences with gene names ({len(set(gene_names))} unique genes)")
    sequences = [rec.sequence for rec in valid_records]

    # 3. Load Encodon model
    print(f"\nLoading Encodon from {args.model_path}...")
    inference = EncodonInference(
        model_path=args.model_path, task_type="embedding_prediction", use_transformer_engine=True
    )
    inference.configure_model()
    inference.model.to(device).eval()

    num_layers = len(inference.model.model.layers)
    target_layer = args.layer if args.layer >= 0 else num_layers + args.layer
    print(f"  Layers: {num_layers}, Target layer: {target_layer}")

    # 4. Extract 3D activations
    print("\n" + "=" * 60)
    print("EXTRACTING ACTIVATIONS")
    print("=" * 60)
    activations, masks = extract_activations_3d(
        inference,
        sequences,
        args.layer,
        context_length=args.context_length,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"  Activations shape: {activations.shape}")

    # Free Encodon model memory
    del inference
    torch.cuda.empty_cache()

    # 5. Compute per-gene activation matrix (or load from cache)
    gene_acts_cache = output_dir / "gene_activations_cache.json"
    if gene_acts_cache.exists():
        print("\n" + "=" * 60)
        print("LOADING CACHED GENE ACTIVATIONS")
        print("=" * 60)
        with open(gene_acts_cache) as f:
            raw = json.load(f)
        gene_activations = {int(k): v for k, v in raw.items()}
        print(f"  Loaded {len(gene_activations)} features from cache")

        # Free GPU resources we don't need
        del activations, masks
        torch.cuda.empty_cache()
        sae = sae.cpu()
    else:
        print("\n" + "=" * 60)
        print("COMPUTING PER-GENE ACTIVATIONS")
        print("=" * 60)
        gene_activations = compute_gene_activations(sae, activations, masks, gene_names, device=device)
        print(f"  {len(gene_activations)} features with non-zero gene activations")

        # Free activations memory
        del activations, masks
        torch.cuda.empty_cache()

        # Move SAE to CPU to free GPU
        sae = sae.cpu()

        # Cache gene activations so we can restart from here
        print(f"  Caching gene activations to {gene_acts_cache}...")
        with open(gene_acts_cache, "w") as f:
            json.dump({str(k): v for k, v in gene_activations.items()}, f)

    # 6. Run GSEA
    print("\n" + "=" * 60)
    print("RUNNING GSEA ENRICHMENT")
    print("=" * 60)
    print(f"  Databases: {databases}")
    print(f"  FDR threshold: {args.fdr_threshold}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Features to process: {len(gene_activations)}")

    t0 = time.time()
    report = run_gene_enrichment(
        gene_activations=gene_activations,
        databases=databases,
        fdr_threshold=args.fdr_threshold,
        n_workers=args.n_workers,
    )
    gsea_time = time.time() - t0
    print(f"\n  GSEA completed in {gsea_time:.1f}s")

    # 6b. Detect gene families
    print("  Detecting gene families...")
    gene_families = detect_gene_families(gene_activations)
    print(f"  {len(gene_families)} features with dominant gene family")

    # 6c. Compute pLI metrics (optional)
    feature_pli = {}
    if args.pli_path:
        print(f"  Loading pLI scores from {args.pli_path}...")
        pli_scores = load_pli_scores(args.pli_path)
        print(f"  Loaded pLI for {len(pli_scores)} genes")
        feature_pli = compute_feature_pli(gene_activations, pli_scores)
        print(f"  {len(feature_pli)} features with pLI metrics")

    # Update report label columns with gene families
    from codonfm_sae.eval.gene_enrichment import build_feature_label_columns

    report.feature_label_columns = build_feature_label_columns(
        report.per_feature, report.n_features_total, gene_families=gene_families
    )

    # 7. Save results (before GO Slim so we don't lose GSEA work on failure)
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    report_path = output_dir / "gene_enrichment_report.json"
    save_report_json(report, report_path)
    print(f"  Saved report to {report_path}")

    parquet_path = output_dir / "enrichment_results.parquet"
    save_enrichment_parquet(report, parquet_path)
    print(f"  Saved enrichment results to {parquet_path}")

    # 8. Optional GO Slim rollup
    if not args.no_go_slim:
        print("\n" + "=" * 60)
        print("GO SLIM ROLLUP")
        print("=" * 60)
        obo_dir = args.obo_dir or str(output_dir / "obo")
        go_basic_path, go_slim_path = download_obo_files(obo_dir)
        rollup_go_slim(report.per_feature, str(go_basic_path), str(go_slim_path))

        # Rebuild label columns with GO Slim info
        from codonfm_sae.eval.gene_enrichment import build_feature_label_columns

        report.feature_label_columns = build_feature_label_columns(
            report.per_feature, report.n_features_total, gene_families=gene_families
        )

        n_slim = sum(1 for fl in report.per_feature if fl.go_slim_name is not None)
        slim_names = {fl.go_slim_name for fl in report.per_feature if fl.go_slim_name is not None}
        print(f"  {n_slim} features mapped to {len(slim_names)} GO Slim categories")

        # Re-save with GO Slim data
        save_report_json(report, report_path)
        print("  Updated report with GO Slim labels")

    # 9. Update dashboard atlas if requested
    if args.dashboard_dir:
        dashboard_dir = Path(args.dashboard_dir)
        atlas_path = dashboard_dir / "features_atlas.parquet"
        update_atlas_with_gsea(report, atlas_path)
        if feature_pli:
            update_atlas_with_pli(feature_pli, atlas_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Features total:          {report.n_features_total}")
    print(f"  Features with enrichment: {report.n_features_with_enrichment}")
    print(f"  Fraction enriched:       {report.frac_enriched:.3f}")
    for db, stats in report.per_database_stats.items():
        print(f"  {db}: {stats['n_enriched']} enriched, {stats['n_unique_terms']} unique terms")

    print(f"\nAll results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
