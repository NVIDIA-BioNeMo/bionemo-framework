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

"""Gene-level GSEA enrichment metric for CodonFM SAE features.

For each SAE feature, ranks genes by activation strength and runs GSEA
(Gene Set Enrichment Analysis) against GO and InterPro databases.
This captures functional/pathway-level interpretability that residue-level
F1 misses (e.g., a feature that fires on all ribosomal protein genes).

Dependencies: gseapy>=1.0, goatools>=1.3
Install via: pip install codonfm-sae[gsea]
"""

import logging
import re
import warnings
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

ANNOTATION_DATABASES = [
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "InterPro_Domains_2019",
]

_GO_ID_PATTERN = re.compile(r"\((GO:\d+)\)")


# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class EnrichmentResult:
    """Single (feature, term) enrichment result."""

    feature_idx: int
    term_id: str
    term_name: str
    database: str
    enrichment_score: float
    pvalue: float
    fdr: float
    n_genes_in_term: int


@dataclass
class FeatureLabels:
    """Best label for a single feature, per database."""

    feature_idx: int
    best_per_database: Dict[str, Optional[EnrichmentResult]]
    overall_best: Optional[EnrichmentResult]
    go_slim_term: Optional[str] = None
    go_slim_name: Optional[str] = None
    all_significant: List[EnrichmentResult] = field(default_factory=list)


@dataclass
class GeneEnrichmentReport:
    """Full results across all features."""

    per_feature: List[FeatureLabels]
    databases_used: List[str]
    n_features_with_enrichment: int
    n_features_total: int
    frac_enriched: float
    per_database_stats: Dict[str, dict]
    feature_label_columns: Dict[str, Dict[int, str]]
    significance_threshold: float


# ── Utilities ────────────────────────────────────────────────────────────


def reduce_to_gene_level(
    per_codon_activations: Dict[int, Dict[str, List[float]]],
    method: str = "max",
) -> Dict[int, Dict[str, float]]:
    """Collapse per-codon activation lists to per-gene scalars.

    Args:
        per_codon_activations: feature_idx -> gene_name -> list of per-codon values
        method: Aggregation method ("max" or "mean").

    Returns:
        feature_idx -> gene_name -> single scalar score
    """
    agg_fn = np.max if method == "max" else np.mean
    result = {}
    for feat_idx, gene_dict in per_codon_activations.items():
        result[feat_idx] = {gene: float(agg_fn(vals)) for gene, vals in gene_dict.items()}
    return result


def validate_databases(databases: List[str]) -> List[str]:
    """Check Enrichr library names are available, warn on missing ones.

    Returns the subset of databases that are available.
    """
    import gseapy

    available = set(gseapy.get_library_name())
    valid = []
    for db in databases:
        if db in available:
            valid.append(db)
        else:
            # Try to find a close match
            candidates = [a for a in available if db.split("_")[0] in a]
            if candidates:
                logger.warning("Database '%s' not found in Enrichr. Similar: %s", db, candidates[:3])
            else:
                logger.warning("Database '%s' not found in Enrichr.", db)
    return valid


def _parse_go_id(term_string: str) -> str:
    """Extract GO ID from Enrichr term string like 'translation (GO:0006412)'."""
    match = _GO_ID_PATTERN.search(term_string)
    return match.group(1) if match else term_string


def _parse_term_name(term_string: str) -> str:
    """Extract human-readable name from Enrichr term string."""
    # Remove trailing GO ID like " (GO:0006412)"
    name = _GO_ID_PATTERN.sub("", term_string).strip()
    return name if name else term_string


# ── Per-feature GSEA ─────────────────────────────────────────────────────


def run_gsea_for_feature(
    feature_idx: int,
    gene_scores: Dict[str, float],
    databases: List[str],
    fdr_threshold: float = 0.05,
) -> FeatureLabels:
    """Run gseapy.prerank() against all databases for one feature.

    Args:
        feature_idx: Index of the SAE feature.
        gene_scores: gene_name -> activation score.
        databases: List of Enrichr library names.
        fdr_threshold: FDR cutoff for significance.

    Returns:
        FeatureLabels with best enrichment per database.
    """
    import gseapy

    # Build ranked gene list (descending by activation)
    series = pd.Series(gene_scores).sort_values(ascending=False)

    best_per_db: Dict[str, Optional[EnrichmentResult]] = {}
    all_significant: List[EnrichmentResult] = []
    overall_best: Optional[EnrichmentResult] = None

    for db in databases:
        best_per_db[db] = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = gseapy.prerank(
                    rnk=series,
                    gene_sets=db,
                    min_size=5,
                    max_size=1000,
                    no_plot=True,
                    outdir=None,
                    verbose=False,
                    seed=42,
                )

            if res.res2d is None or res.res2d.empty:
                continue

            df = res.res2d.copy()
            # gseapy returns FDR as 'FDR q-val' or 'fdr'
            fdr_col = "FDR q-val" if "FDR q-val" in df.columns else "fdr"
            es_col = "NES" if "NES" in df.columns else "nes"
            pval_col = "NOM p-val" if "NOM p-val" in df.columns else "pval"
            df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
            df = df.dropna(subset=[fdr_col])

            if df.empty:
                continue

            # Best = lowest FDR
            best_row = df.loc[df[fdr_col].idxmin()]
            term_raw = str(best_row.get("Term", best_row.name))

            # Parse GO ID if applicable
            is_go = db.startswith("GO_")
            term_id = _parse_go_id(term_raw) if is_go else term_raw
            term_name = _parse_term_name(term_raw) if is_go else term_raw

            fdr_val = float(best_row[fdr_col])
            es_val = float(best_row.get(es_col, 0.0))
            pval = float(best_row.get(pval_col, 1.0))

            # Parse n_genes from "Tag %" column (format: "6/200") or fall back to 0
            n_genes = 0
            tag_pct = str(best_row.get("Tag %", ""))
            if "/" in tag_pct:
                try:
                    n_genes = int(tag_pct.split("/")[1])
                except (ValueError, IndexError):
                    pass

            result = EnrichmentResult(
                feature_idx=feature_idx,
                term_id=term_id,
                term_name=term_name,
                database=db,
                enrichment_score=es_val,
                pvalue=pval,
                fdr=fdr_val,
                n_genes_in_term=n_genes,
            )

            if fdr_val < fdr_threshold:
                best_per_db[db] = result
                all_significant.append(result)
                if overall_best is None or fdr_val < overall_best.fdr:
                    overall_best = result

            # Also collect all significant terms from this database
            sig_rows = df[df[fdr_col] < fdr_threshold]
            for _, row in sig_rows.iterrows():
                t_raw = str(row.get("Term", row.name))
                if t_raw == term_raw:
                    continue  # Already added the best
                t_id = _parse_go_id(t_raw) if is_go else t_raw
                t_name = _parse_term_name(t_raw) if is_go else t_raw
                row_n_genes = 0
                row_tag = str(row.get("Tag %", ""))
                if "/" in row_tag:
                    try:
                        row_n_genes = int(row_tag.split("/")[1])
                    except (ValueError, IndexError):
                        pass
                all_significant.append(
                    EnrichmentResult(
                        feature_idx=feature_idx,
                        term_id=t_id,
                        term_name=t_name,
                        database=db,
                        enrichment_score=float(row.get(es_col, 0.0)),
                        pvalue=float(row.get(pval_col, 1.0)),
                        fdr=float(row[fdr_col]),
                        n_genes_in_term=row_n_genes,
                    )
                )

        except Exception as e:
            logger.debug("GSEA failed for feature %d, db %s: %s", feature_idx, db, e)
            continue

    return FeatureLabels(
        feature_idx=feature_idx,
        best_per_database=best_per_db,
        overall_best=overall_best,
        all_significant=all_significant,
    )


# ── Worker function for multiprocessing ──────────────────────────────────


def _worker_gsea(args: Tuple[int, Dict[str, float], List[str], float]) -> FeatureLabels:
    """Multiprocessing worker: run GSEA for a single feature."""
    feature_idx, gene_scores, databases, fdr_threshold = args
    return run_gsea_for_feature(feature_idx, gene_scores, databases, fdr_threshold)


# ── Parallel dispatch ────────────────────────────────────────────────────


def run_gene_enrichment(
    gene_activations: Dict[int, Dict[str, float]],
    databases: Optional[List[str]] = None,
    fdr_threshold: float = 0.05,
    n_workers: int = 4,
    show_progress: bool = True,
) -> GeneEnrichmentReport:
    """Run GSEA enrichment for all features in parallel.

    Args:
        gene_activations: feature_idx -> gene_name -> activation score.
        databases: Enrichr library names (default: ANNOTATION_DATABASES).
        fdr_threshold: FDR cutoff for significance.
        n_workers: Number of parallel workers.
        show_progress: Whether to show a progress bar.

    Returns:
        GeneEnrichmentReport with enrichment results for all features.
    """
    from tqdm import tqdm

    if databases is None:
        databases = list(ANNOTATION_DATABASES)

    # Validate databases
    valid_databases = validate_databases(databases)
    if not valid_databases:
        raise ValueError(f"None of the requested databases are available in Enrichr: {databases}")
    if len(valid_databases) < len(databases):
        logger.warning("Using %d/%d databases: %s", len(valid_databases), len(databases), valid_databases)

    # Build work items, skip dead/flat features
    work_items = []
    skipped = 0
    for feat_idx, gene_scores in gene_activations.items():
        vals = list(gene_scores.values())
        if not vals or max(vals) == 0:
            skipped += 1
            continue
        if len(set(vals)) <= 1:
            skipped += 1
            continue
        work_items.append((feat_idx, gene_scores, valid_databases, fdr_threshold))

    if skipped > 0:
        logger.info("Skipped %d dead/flat features", skipped)

    n_features_total = len(gene_activations)
    per_feature_results: List[FeatureLabels] = []

    if n_workers <= 1:
        iterator = work_items
        if show_progress:
            iterator = tqdm(iterator, desc="GSEA enrichment")
        for item in iterator:
            per_feature_results.append(_worker_gsea(item))
    else:
        with Pool(n_workers) as pool:
            iterator = pool.imap_unordered(_worker_gsea, work_items)
            if show_progress:
                iterator = tqdm(iterator, total=len(work_items), desc="GSEA enrichment")
            for result in iterator:
                per_feature_results.append(result)

    # Sort by feature index for deterministic output
    per_feature_results.sort(key=lambda x: x.feature_idx)

    # Compute stats
    n_enriched = sum(1 for fl in per_feature_results if fl.overall_best is not None)
    per_db_stats = {}
    for db in valid_databases:
        db_enriched = [fl for fl in per_feature_results if fl.best_per_database.get(db) is not None]
        unique_terms = {fl.best_per_database[db].term_id for fl in db_enriched}
        per_db_stats[db] = {"n_enriched": len(db_enriched), "n_unique_terms": len(unique_terms)}

    # Build label columns
    label_columns = build_feature_label_columns(per_feature_results, n_features_total)

    return GeneEnrichmentReport(
        per_feature=per_feature_results,
        databases_used=valid_databases,
        n_features_with_enrichment=n_enriched,
        n_features_total=n_features_total,
        frac_enriched=n_enriched / max(n_features_total, 1),
        per_database_stats=per_db_stats,
        feature_label_columns=label_columns,
        significance_threshold=fdr_threshold,
    )


# ── GO Slim rollup ──────────────────────────────────────────────────────


def download_obo_files(output_dir: str) -> Tuple[Path, Path]:
    """Download go-basic.obo and goslim_generic.obo if not present.

    Returns:
        (go_basic_path, go_slim_path)
    """
    import urllib.request

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    go_basic_path = output_dir / "go-basic.obo"
    go_slim_path = output_dir / "goslim_generic.obo"

    if not go_basic_path.exists():
        logger.info("Downloading go-basic.obo...")
        urllib.request.urlretrieve(
            "http://purl.obolibrary.org/obo/go/go-basic.obo",
            str(go_basic_path),
        )

    if not go_slim_path.exists():
        logger.info("Downloading goslim_generic.obo...")
        urllib.request.urlretrieve(
            "http://purl.obolibrary.org/obo/go/subsets/goslim_generic.obo",
            str(go_slim_path),
        )

    return go_basic_path, go_slim_path


def rollup_go_slim(
    feature_labels: List[FeatureLabels],
    go_basic_path: str,
    go_slim_path: str,
) -> List[FeatureLabels]:
    """Walk GO DAG upward to nearest GO Slim ancestor for each feature.

    Modifies FeatureLabels in-place, setting go_slim_term and go_slim_name.

    Args:
        feature_labels: List of FeatureLabels (modified in-place).
        go_basic_path: Path to go-basic.obo file.
        go_slim_path: Path to goslim_generic.obo file.

    Returns:
        The same list, with go_slim_term/go_slim_name populated.
    """
    from goatools.obo_parser import GODag

    go_dag = GODag(str(go_basic_path))
    slim_dag = GODag(str(go_slim_path))
    slim_ids = set(slim_dag.keys())

    def _find_slim_ancestor(go_id: str) -> Optional[Tuple[str, str]]:
        """Walk parents until we hit a GO Slim term."""
        if go_id not in go_dag:
            return None
        if go_id in slim_ids:
            term = go_dag[go_id]
            return go_id, term.name

        visited = set()
        queue = [go_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current not in go_dag:
                continue
            term = go_dag[current]
            for parent in term.parents:
                if parent.id in slim_ids:
                    return parent.id, parent.name
                queue.append(parent.id)
        return None

    for fl in feature_labels:
        # Find a GO term to roll up from
        go_term_id = None
        for db_key in ["GO_Biological_Process_2023", "GO_Molecular_Function_2023", "GO_Cellular_Component_2023"]:
            best = fl.best_per_database.get(db_key)
            if best is not None and best.term_id.startswith("GO:"):
                go_term_id = best.term_id
                break
        if fl.overall_best is not None and fl.overall_best.term_id.startswith("GO:"):
            go_term_id = fl.overall_best.term_id

        if go_term_id is not None:
            slim = _find_slim_ancestor(go_term_id)
            if slim is not None:
                fl.go_slim_term, fl.go_slim_name = slim

    return feature_labels


# ── Gene family detection ────────────────────────────────────────────────


def _gene_prefix(gene_name: str) -> str:
    """Extract the alphabetic prefix of a gene name (letters before first digit)."""
    prefix = ""
    for c in gene_name:
        if c.isdigit():
            break
        prefix += c
    return prefix


def detect_gene_families(
    gene_activations: Dict[int, Dict[str, float]],
    top_k: int = 10,
    min_fraction: float = 0.5,
) -> Dict[int, str]:
    """Detect dominant gene family for each feature based on top-K gene name prefixes.

    Args:
        gene_activations: feature_idx -> gene_name -> activation score.
        top_k: Number of top genes to examine per feature.
        min_fraction: Minimum fraction of top-K genes sharing a prefix to call it a family.

    Returns:
        feature_idx -> gene family label (e.g., "OR family (8/10)") or absent if no family.
    """
    from collections import Counter

    result = {}
    for feat_idx, gene_scores in gene_activations.items():
        top_genes = sorted(gene_scores.keys(), key=lambda g: gene_scores[g], reverse=True)[:top_k]
        if len(top_genes) < 3:
            continue
        prefixes = [_gene_prefix(g) for g in top_genes]
        counts = Counter(p for p in prefixes if len(p) >= 2)
        if not counts:
            continue
        top_prefix, top_count = counts.most_common(1)[0]
        if top_count / len(top_genes) >= min_fraction:
            result[feat_idx] = f"{top_prefix} family ({top_count}/{len(top_genes)})"
    return result


# ── pLI scores ──────────────────────────────────────────────────────────


def load_pli_scores(pli_path: str) -> Dict[str, float]:
    """Load gnomAD pLI scores from the constraint metrics TSV.

    Supports both plain TSV and bgzipped (.bgz/.gz) files.
    The file should have 'gene' and 'pLI' columns (gnomAD v2.1.1 format).

    Returns:
        gene_name -> pLI score
    """
    if str(pli_path).endswith(".bgz") or str(pli_path).endswith(".gz"):
        df = pd.read_csv(pli_path, sep="\t", compression="gzip", usecols=["gene", "pLI"])
    else:
        df = pd.read_csv(pli_path, sep="\t", usecols=["gene", "pLI"])

    df = df.dropna(subset=["pLI"])
    # Keep first occurrence per gene (canonical transcript)
    df = df.drop_duplicates(subset=["gene"], keep="first")
    return dict(zip(df["gene"], df["pLI"].astype(float)))


def compute_feature_pli(
    gene_activations: Dict[int, Dict[str, float]],
    pli_scores: Dict[str, float],
    top_k: int = 20,
) -> Dict[int, Dict[str, float]]:
    """Compute pLI-based metrics for each feature.

    For each feature, looks at the top-K genes by activation and computes:
    - mean_pli: mean pLI score of top-K genes (with known pLI)
    - frac_constrained: fraction of top-K genes with pLI > 0.9
    - max_pli: max pLI among top-K genes

    Args:
        gene_activations: feature_idx -> gene_name -> activation score.
        pli_scores: gene_name -> pLI score.
        top_k: Number of top genes to examine per feature.

    Returns:
        feature_idx -> {"mean_pli": float, "frac_constrained": float, "max_pli": float}
    """
    result = {}
    for feat_idx, gene_scores in gene_activations.items():
        top_genes = sorted(gene_scores.keys(), key=lambda g: gene_scores[g], reverse=True)[:top_k]
        pli_vals = [pli_scores[g] for g in top_genes if g in pli_scores]
        if not pli_vals:
            continue
        result[feat_idx] = {
            "mean_pli": float(np.mean(pli_vals)),
            "frac_constrained": float(np.mean([1.0 if v > 0.9 else 0.0 for v in pli_vals])),
            "max_pli": float(np.max(pli_vals)),
        }
    return result


# ── Label columns for UMAP ──────────────────────────────────────────────


def build_feature_label_columns(
    per_feature: List[FeatureLabels],
    n_features: int,
    gene_families: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[int, str]]:
    """Build dict[column_name, dict[feature_idx, label]] for UMAP dropdown.

    Keys: overall_best, GO_Biological_Process, GO_Molecular_Function,
          GO_Cellular_Component, InterPro_Domains, GO_Slim
    """
    db_to_column = {
        "GO_Biological_Process_2023": "GO_Biological_Process",
        "GO_Molecular_Function_2023": "GO_Molecular_Function",
        "GO_Cellular_Component_2023": "GO_Cellular_Component",
        "InterPro_Domains_2019": "InterPro_Domains",
    }

    columns: Dict[str, Dict[int, str]] = {
        "overall_best": {},
        "GO_Biological_Process": {},
        "GO_Molecular_Function": {},
        "GO_Cellular_Component": {},
        "InterPro_Domains": {},
        "GO_Slim": {},
        "gene_family": {},
    }

    for fl in per_feature:
        idx = fl.feature_idx

        if fl.overall_best is not None:
            columns["overall_best"][idx] = fl.overall_best.term_name
        else:
            columns["overall_best"][idx] = "unlabeled"

        for db, col_name in db_to_column.items():
            best = fl.best_per_database.get(db)
            if best is not None:
                columns[col_name][idx] = best.term_name
            else:
                columns[col_name][idx] = "unlabeled"

        if fl.go_slim_name is not None:
            columns["GO_Slim"][idx] = fl.go_slim_name
        else:
            columns["GO_Slim"][idx] = "unlabeled"

        if gene_families and idx in gene_families:
            columns["gene_family"][idx] = gene_families[idx]
        else:
            columns["gene_family"][idx] = "unlabeled"

    # Fill missing feature indices with "unlabeled"
    for col in columns:
        for i in range(n_features):
            if i not in columns[col]:
                columns[col][i] = "unlabeled"

    return columns
