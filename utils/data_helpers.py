"""
Data loading helpers used by baselines, irAEGIS, and evaluation scripts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import issparse

from utils.config import (
    cohort_h5ad, PRIOR_NPZ,
    QC_MIN_GENES_PER_CELL, QC_MIN_CELLS_PER_PAT_CT, QC_MIN_PATIENT_FRACTION,
)

# ---------------------------------------------------------------------------
# Metadata update
# ---------------------------------------------------------------------------

# GSE285888: disease_group and tissue_type were "unknown".
# Correct values come from pre-ICI_irAE_sample_info_details_02.18.2026_v5_brief.xlsx.
_GSE285888_DISEASE = "NSCLC"
_GSE285888_TISSUE  = "PBMC"

GRADE_RELABEL_COHORTS = {
    "GSE249898_integrated_pre_ici": 3,
    "GSE189125_pre_ici": 3,
}


def relabel_by_grade(adata, cohort_id: str) -> None:
    # Update irAE_status using irAE_grade >= threshold.
    threshold = GRADE_RELABEL_COHORTS.get(cohort_id)
    if threshold is None:
        return
    if "irAE_grade" not in adata.obs.columns:
        return

    grade = pd.to_numeric(adata.obs["irAE_grade"], errors="coerce")
    new_status = np.where(grade >= threshold, "Yes", "No")

    col = adata.obs["irAE_status"]
    if hasattr(col, "cat"):
        for v in ["Yes", "No"]:
            if v not in col.cat.categories:
                adata.obs["irAE_status"] = col.cat.add_categories(v)
    adata.obs["irAE_status"] = new_status

    n_yes = int((new_status == "Yes").sum())
    n_no = int((new_status == "No").sum())
    n_pat = adata.obs["patient_id"].nunique()
    print(f"[relabel_by_grade] {cohort_id}: grade>={threshold} → "
          f"{n_yes:,} Yes cells, {n_no:,} No cells ({n_pat} patients)")


def patch_metadata(adata) -> None:
    # Fix known metadata gaps in the input h5ad.
    if "dataset_id" not in adata.obs.columns:
        return
    mask = adata.obs["dataset_id"].astype(str) == "GSE285888_pre_ici"
    if not mask.any():
        return

    if "disease_group" in adata.obs.columns:
        col = adata.obs["disease_group"]
        if hasattr(col, "cat") and _GSE285888_DISEASE not in col.cat.categories:
            adata.obs["disease_group"] = col.cat.add_categories(_GSE285888_DISEASE)
        adata.obs.loc[mask, "disease_group"] = _GSE285888_DISEASE
    if "tissue_type" in adata.obs.columns:
        col = adata.obs["tissue_type"]
        if hasattr(col, "cat") and _GSE285888_TISSUE not in col.cat.categories:
            adata.obs["tissue_type"] = col.cat.add_categories(_GSE285888_TISSUE)
        adata.obs.loc[mask, "tissue_type"] = _GSE285888_TISSUE

    n = int(mask.sum())
    print(f"[patch_metadata] GSE285888: set disease_group='{_GSE285888_DISEASE}', "
          f"tissue_type='{_GSE285888_TISSUE}' for {n:,} cells.")


# ---------------------------------------------------------------------------
# Data QC filtering
# ---------------------------------------------------------------------------

def filter_low_quality_cells(adata):
    # Remove cells expressing fewer than QC_MIN_GENES_PER_CELL genes.
    X = adata.X
    if issparse(X):
        genes_per_cell = np.array((X > 0).sum(axis=1)).ravel()
    else:
        genes_per_cell = (np.asarray(X) > 0).sum(axis=1)

    keep = genes_per_cell >= QC_MIN_GENES_PER_CELL
    n_drop = int((~keep).sum())
    if n_drop > 0:
        print(f"[cell QC] Dropping {n_drop:,}/{adata.n_obs:,} cells "
              f"with <{QC_MIN_GENES_PER_CELL} genes")
        adata = adata[keep].copy()
    return adata


def filter_sparse_celltypes(adata, patients, labels):
    # Drop CTs where <QC_MIN_PATIENT_FRACTION of patients have >=QC_MIN_CELLS_PER_PAT_CT cells.
    # Returns filtered adata (cells from dropped CTs removed) and list of dropped CT names.
    
    if "final_celltype" not in adata.obs.columns:
        return adata, []

    pat_ids = adata.obs["patient_id"].values.astype(str)
    ct_ids = adata.obs["final_celltype"].values.astype(str)
    all_cts = sorted(set(ct_ids) - {"Unknown", "__EXCLUDE__"})
    n_patients = len(patients)

    drop_cts = []
    for ct in all_cts:
        ct_mask = ct_ids == ct
        n_adequate = 0
        for p in patients:
            n_cells = int((ct_mask & (pat_ids == p)).sum())
            if n_cells >= QC_MIN_CELLS_PER_PAT_CT:
                n_adequate += 1
        frac = n_adequate / max(n_patients, 1)
        if frac < QC_MIN_PATIENT_FRACTION:
            drop_cts.append(ct)

    if drop_cts:
        print(f"[CT QC] Dropping {len(drop_cts)} cell types "
              f"(<{QC_MIN_PATIENT_FRACTION*100:.0f}% patients with "
              f">={QC_MIN_CELLS_PER_PAT_CT} cells): {drop_cts}")
        keep = ~np.isin(ct_ids, drop_cts)
        adata = adata[keep].copy()

    return adata, drop_cts


# Load dataset
def load_h5ad(h5ad_path: str | Path, cohort_id: str | None = None):
    # Load h5ad file 
    import scanpy as sc
    if h5ad_path is None:
        if cohort_id is None:
            raise ValueError("load_h5ad: pass either h5ad_path or cohort_id")
        h5ad_path = cohort_h5ad(cohort_id)
    return sc.read_h5ad(Path(h5ad_path))


def prepare_cohort(adata, cohort_id: str, apply_qc: bool = True):
    # Apply metadata patches, grade re-labeling, and QC filtering.
    # h5ads are already per-cohort, so no dataset_id slicing is needed.
    patch_metadata(adata)
    relabel_by_grade(adata, cohort_id)

    if apply_qc:
        adata = filter_low_quality_cells(adata)
        patients, _, _ = extract_patients(adata)
        adata, _ = filter_sparse_celltypes(adata, patients, None)

    return adata


def extract_patients(adata) -> tuple[list[str], np.ndarray, np.ndarray]:
    # Extract patient IDs, labels, and per-cell patient assignments.

    pat_ids = adata.obs["patient_id"].values.astype(str)
    pat_labs = adata.obs.groupby("patient_id")["irAE_status"].first()

    patients = sorted(pat_labs.index.tolist())
    labels = np.array([1.0 if str(pat_labs[p]).strip() in ("Yes", "Severe") else 0.0
                       for p in patients])
    return patients, labels, pat_ids


def get_expression_matrix(adata) -> np.ndarray:
    # Get expression matrix as dense float32, log-normalised if raw counts
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    if X.max() > 50:
        X = X / (X.sum(1, keepdims=True) + 1e-9) * 1e4
        X = np.log1p(X).astype(np.float32)
    return X


def select_top_variance_genes(X: np.ndarray, gene_names: np.ndarray,
                               n_genes: int) -> tuple[np.ndarray, np.ndarray]:
    # Keep top n_genes by variance
    if n_genes is not None and n_genes < X.shape[1]:
        var = X.var(axis=0)
        idx = np.argsort(var)[::-1][:n_genes]
        return X[:, idx], gene_names[idx]
    return X, gene_names


def select_prior_genes(X: np.ndarray, gene_names: np.ndarray,
                       prior_path: str | Path | None = None,
                       hvg_k: int = 2000,
                       ) -> tuple[np.ndarray, np.ndarray]:
    # Subset to pathway-active genes + top hvg_k high-variance genes.

    if prior_path is None:
        prior_path = PRIOR_NPZ
    prior_path = Path(prior_path)
    data = np.load(prior_path, allow_pickle=True)
    prior_genes = list(data["gene_names"].astype(str))
    mask = data["mask"]
    if mask.shape[0] != len(prior_genes):
        mask = mask.T
    active = set(g for g, s in zip(prior_genes, mask.sum(axis=1)) if s > 0)

    gn_list = list(gene_names)
    pw_idx = np.array([i for i, g in enumerate(gn_list) if g in active], dtype=int)
    non_pw = np.array([i for i, g in enumerate(gn_list) if g not in active], dtype=int)

    if hvg_k > 0 and len(non_pw) > hvg_k:
        var_non_pw = np.asarray(X[:, non_pw].var(axis=0)).ravel()
        hvg = non_pw[np.argsort(var_non_pw)[-hvg_k:]]
        idx = np.sort(np.concatenate([pw_idx, hvg]))
    elif hvg_k > 0:
        # Non-pathway pool smaller than hvg_k (e.g. MIX/SIM with 5k-gene h5ad).
        # Match irAEGIS's all_prior fallback and keep all shared genes so baselines see the same input as irAEGIS.
        idx = np.arange(len(gn_list))
    else:
        idx = pw_idx
    return X[:, idx], gene_names[idx]


def load_cells_cohort(cohort: str, n_genes: int | None = None,
                      h5ad_path=None, prior_genes_path=None):
    """
    Cached cell-level cohort loader.
    Returns (X, pat_ids, celltypes, gene_names, patients, labels).

    patients : sorted list of patient IDs
    labels   : (n_patients,) 1.0=Yes, 0.0=No aligned to patients
    """
    import hashlib
    cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "cells"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_parts = [cohort, str(n_genes), str(h5ad_path or ""), str(prior_genes_path or "")]
    key = hashlib.md5("|".join(key_parts).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"{cohort}_{key}.npz"

    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        print(f"Cohort {cohort}: cached cells {d['X'].shape} [{cache_path.name}]")
        return (d["X"], d["pat_ids"].astype(str),
                d["celltypes"].astype(str), d["gene_names"].astype(str),
                list(d["patients"].astype(str)), d["labels"])

    adata = load_h5ad(h5ad_path)
    adata = prepare_cohort(adata, cohort)
    patients, labels, pat_ids = extract_patients(adata)
    ct = adata.obs["final_celltype"].astype(str).values if "final_celltype" in adata.obs else np.array([""] * adata.n_obs)
    gene_names = np.array(adata.var_names.tolist())
    X = get_expression_matrix(adata)
    if prior_genes_path is not None:
        X, gene_names = select_prior_genes(X, gene_names, prior_genes_path)
    else:
        X, gene_names = select_top_variance_genes(X, gene_names, n_genes)

    np.savez(cache_path, X=X, pat_ids=pat_ids, celltypes=ct,
             gene_names=gene_names, patients=np.array(patients), labels=labels)
    print(f"Cohort {cohort}: cells {X.shape} cached → {cache_path.name}")
    return X, pat_ids, ct, gene_names, patients, labels

