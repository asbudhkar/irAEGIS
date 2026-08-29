"""
irAEGIS-specific data loading: cell-level cohort loading with CT grouping
and pathway prior alignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import issparse

from utils.config import cohort_h5ad, PRIOR_NPZ, HVG_BACKFILL
from utils.data_helpers import (
    relabel_by_grade, filter_low_quality_cells, filter_sparse_celltypes,
    extract_patients,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
def _rel(p):
    try: return Path(p).resolve().relative_to(_REPO_ROOT)
    except ValueError: return Path(p)


def load_cohort_data(cohort_id: str, h5ad_path=None, prior_path=None,
                     prior_genes_only: bool = False, verbose: bool = True,
                     split_ct_groups: list | None = None,
                     gene_list_path: str | Path | None = None,
                     defer_selection: bool = False):
    """Load a cohort into (X, obs, gene_names, ct_groups, ct_ids, pat_ids,
    pat_labels, prior).

    defer_selection: when True, skip the two data-dependent selection steps —
        the HVG backfill and cell-type grouping — and return the full shared
        gene space with raw `final_celltype` labels. The caller is then
        responsible for performing both selections using training-fold cells
        only (see analysis/fold_selection.py). Required for leakage-free
        per-fold cross-validation, where both the gene set and the surviving
        cell types must be derived without the held-out patient.
        In this mode ct_groups is [] and ct_ids is all -1.
    """
    # Load data
    import scanpy as sc
    from utils.celltype_groups import infer_celltype_groups

    src = Path(h5ad_path) if h5ad_path else cohort_h5ad(cohort_id)
    prior_src = Path(prior_path) if prior_path else PRIOR_NPZ

    if verbose:
        print(f"Loading {_rel(src)} ...")
    adata = sc.read_h5ad(src)

    if "dataset_id" in adata.obs.columns:
        ids = adata.obs["dataset_id"].astype(str)
        if (ids == cohort_id).any():
            adata = adata[ids == cohort_id].copy()
    if verbose:
        print(f"  Cohort {cohort_id}: {adata.n_obs:,} cells")

    # Optional: restrict to a precomputed shared gene vocabulary so attributions
    # are directly comparable across cohorts trained on the same gene set.
    if gene_list_path:
        gl_path = Path(gene_list_path)
        if gl_path.exists():
            with gl_path.open() as f:
                keep_genes = set(g.strip() for g in f if g.strip())
            in_list = [g for g in adata.var_names if g in keep_genes]
            n_before = adata.n_vars
            adata = adata[:, in_list].copy()
            if verbose:
                print(f"  --gene-list: kept {adata.n_vars:,}/{n_before:,} genes "
                      f"(from {gl_path})")
        elif verbose:
            print(f"  --gene-list: file not found at {gl_path}; "
                  f"using cohort's native gene vocabulary")

    relabel_by_grade(adata, cohort_id)

    # Data QC filtering
    adata = filter_low_quality_cells(adata)
    patients_qc, labels_qc, _ = extract_patients(adata)
    adata, dropped_cts = filter_sparse_celltypes(adata, patients_qc, labels_qc)

    if verbose and dropped_cts:
        print(f"  Dropped CTs after QC: {dropped_cts}")

    # Process data
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32, order="C")

    if X.max() > 50:
        row_sums = X.sum(axis=1, keepdims=True) + 1e-9
        X /= row_sums
        X *= 1e4
        np.log1p(X, out=X)

    # Required obs columns: patient_id, final_celltype, irAE_status.
    keep_cols = ["patient_id", "final_celltype", "irAE_status"]
    if "dataset_id" in adata.obs.columns:
        keep_cols.append("dataset_id")
    obs = adata.obs[keep_cols].copy()
    if "dataset_id" not in obs.columns:
        obs["dataset_id"] = cohort_id
    obs.insert(0, "cell_barcode", adata.obs_names)
    gene_names = list(adata.var_names)

    # CT grouping — skipped under defer_selection so the caller can derive it
    # per fold from training patients only.
    if defer_selection:
        ct_groups = []
        ct_ids = np.full(len(obs), -1, dtype=np.int64)
        if verbose:
            print("  defer_selection: skipping CT grouping "
                  "(caller derives it per fold)")
    else:
        if verbose:
            print("  Inferring cell-type groups ...")
        ct_groups_dict, ct_label_map = infer_celltype_groups(
            obs, min_patients=3, min_cells_per_patient=10, verbose=verbose,
            split_groups=split_ct_groups)
        ct_groups = list(ct_groups_dict.keys())

        # Build CT integer IDs; exclude Unknown / Other
        ct_to_id = {}
        for gid, (gname, labels) in enumerate(ct_groups_dict.items()):
            for lab in labels:
                ct_to_id[lab] = gid

        ct_ids_raw = np.array([ct_to_id.get(ct, -1)
                                for ct in obs["final_celltype"]], dtype=np.int64)
        keep_mask = ct_ids_raw >= 0
        X        = X[keep_mask]
        obs      = obs[keep_mask].reset_index(drop=True)
        ct_ids   = ct_ids_raw[keep_mask]

        if verbose:
            n_excl = (~keep_mask).sum()
            if n_excl:
                print(f"  Excluded {n_excl:,} Unknown/Other cells")
            print(f"  CT groups ({len(ct_groups)}): {ct_groups}")

    pat_ids = obs["patient_id"].values
    pat_label_raw = obs.groupby("patient_id")["irAE_status"].first()
    pat_labels = {p: (1 if str(v).strip() in ("Yes", "Severe") else 0)
                  for p, v in pat_label_raw.items()}

    if verbose:
        n_yes = sum(v for v in pat_labels.values())
        n_no  = len(pat_labels) - n_yes
        print(f"  Patients: {len(pat_labels)} (Yes={n_yes}, No={n_no})")

    # Pathway prior - align genes to h5ad gene space
    prior_raw  = np.load(prior_src, allow_pickle=True)
    prior_genes = list(prior_raw["gene_names"])
    pw_names    = list(prior_raw["pathway_names"])
    mask_raw    = prior_raw["mask"]

    # mask_raw may be (G_prior, P) or (P, G_prior) - normalise to (G_matched, P)
    if mask_raw.shape[0] == len(prior_genes):
        mask_gp = mask_raw
    else:
        mask_gp = mask_raw.T

    # Intersect genes
    prior_set = set(prior_genes)
    shared    = [g for g in gene_names if g in prior_set]
    if verbose:
        print(f"  Genes in h5ad: {len(gene_names)}, in prior: {len(prior_genes)}, "
              f"shared: {len(shared)}")

    # Build aligned mask and subset X
    prior_idx = {g: i for i, g in enumerate(prior_genes)}
    h5ad_idx  = {g: i for i, g in enumerate(gene_names)}
    shared_h5ad = [h5ad_idx[g] for g in shared]
    shared_pri  = [prior_idx[g] for g in shared]

    X_shared    = X[:, shared_h5ad]
    mask_shared = mask_gp[shared_pri, :]

    # HVG backfill — skipped under defer_selection so the caller can rank
    # variance using training-fold cells only.
    if prior_genes_only and not defer_selection:
        active = mask_shared.sum(axis=1) > 0
        n_pw_total = int(active.sum())
        n_before = len(shared)

        keep_pw = np.where(active)[0]
        zero_idx = np.where(~active)[0]
        if len(zero_idx) > HVG_BACKFILL:
            var_per_gene = np.asarray(X_shared[:, zero_idx].var(axis=0)).ravel()
            top_k = zero_idx[np.argsort(var_per_gene)[-HVG_BACKFILL:]]
            keep = np.sort(np.concatenate([keep_pw, top_k]))
        else:
            keep = np.arange(len(shared))
        X_shared    = X_shared[:, keep]
        mask_shared = mask_shared[keep, :]
        shared      = [shared[i] for i in keep]
    elif defer_selection and verbose:
        active = mask_shared.sum(axis=1) > 0
        print(f"  defer_selection: keeping all {len(shared):,} shared genes "
              f"({int(active.sum()):,} pathway-active, "
              f"{int((~active).sum()):,} backfill candidates)")

    prior = {
        "mask":          mask_shared,
        "gene_names":    shared,
        "pathway_names": pw_names,
    }

    # Free space
    del adata, X, mask_gp, mask_raw, prior_raw
    import gc; gc.collect()

    return X_shared, obs, shared, ct_groups, ct_ids, pat_ids, pat_labels, prior
