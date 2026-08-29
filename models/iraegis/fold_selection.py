#!/usr/bin/env python3
"""Leakage-free per-fold selection of cell-type groups and HVG backfill genes.

Both selections in the production loader are data-dependent and computed over
all cells, including the held-out patient's:

  1. `infer_celltype_groups` keeps a cell type only if >= min_patients patients
     have >= min_cells_per_patient cells of it. Verified to vary under
     leave-one-patient-out on GSE189125 (5 of 16 folds; Neutrophils and
     Platelets move in and out).
  2. The HVG backfill ranks non-pathway genes by variance across all cells.

Under a strict per-fold protocol both must be derived from training-fold cells
only, which means the surviving cell types AND the gene set differ per fold.
Each fold therefore trains its own model with its own n_genes and n_ct.

Memory note: deferring gene selection means holding the full shared gene space
(~17k genes, ~12 GB for the largest cohort) rather than the ~5.7k selected
genes (~4 GB). `plan_folds` avoids that by computing each fold's gene set in a
chunked streaming pass, then taking the union across folds so X is subset once.
Per-fold sets are stored as indices into that union.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.celltype_groups import infer_celltype_groups, EXCLUDE_LABEL
from utils.config import HVG_BACKFILL

CT_MIN_PATIENTS = 3
CT_MIN_CELLS_PER_PATIENT = 10
_VAR_CHUNK = 2000          # genes per chunk when streaming the variance pass


def select_ct_groups(obs: pd.DataFrame, train_cell_mask: np.ndarray,
                     split_ct_groups: list | None):
    """Cell-type groups derived from training-fold cells only.

    Returns (ct_groups, ct_ids, cell_keep) where ct_ids/cell_keep cover ALL
    cells: held-out cells are assigned using the fold's mapping, and any cell
    whose type did not survive in this fold is dropped via cell_keep — the same
    treatment the production loader gives Unknown/Other cells.
    """
    groups_dict, _ = infer_celltype_groups(
        obs.loc[train_cell_mask], min_patients=CT_MIN_PATIENTS,
        min_cells_per_patient=CT_MIN_CELLS_PER_PATIENT,
        verbose=False, split_groups=split_ct_groups)
    ct_groups = list(groups_dict.keys())

    ct_to_id = {}
    for gid, (_gname, labels) in enumerate(groups_dict.items()):
        for lab in labels:
            ct_to_id[lab] = gid

    ct_ids = np.array([ct_to_id.get(c, -1)
                       for c in obs["final_celltype"]], dtype=np.int64)
    return ct_groups, ct_ids, ct_ids >= 0


def select_hvg_genes(X: np.ndarray, mask: np.ndarray,
                     train_cell_mask: np.ndarray,
                     n_backfill: int = HVG_BACKFILL) -> np.ndarray:
    """Pathway-active genes + top-variance non-pathway genes, ranked using
    training-fold cells only. Returns sorted gene indices into X's columns.

    Variance is accumulated in gene chunks so no (n_train x n_backfill_pool)
    array is ever materialised.
    """
    active = np.asarray(mask.sum(axis=1)).ravel() > 0
    keep_pw = np.where(active)[0]
    pool = np.where(~active)[0]
    if len(pool) <= n_backfill:
        return np.arange(X.shape[1])

    rows = np.where(train_cell_mask)[0]
    n = len(rows)
    var = np.empty(len(pool), dtype=np.float64)
    for s in range(0, len(pool), _VAR_CHUNK):
        cols = pool[s:s + _VAR_CHUNK]
        sub = X[np.ix_(rows, cols)].astype(np.float64, copy=False)
        m = sub.sum(axis=0) / n
        var[s:s + len(cols)] = (sub * sub).sum(axis=0) / n - m * m
        del sub

    top = pool[np.argsort(var)[-n_backfill:]]
    return np.sort(np.concatenate([keep_pw, top]))


def plan_folds(X: np.ndarray, obs: pd.DataFrame, mask: np.ndarray,
               pat_ids: np.ndarray, held_out_patients: list,
               split_ct_groups: list | None,
               n_backfill: int = HVG_BACKFILL, verbose: bool = True) -> dict:
    """Precompute every fold's CT groups and gene set in one streaming pass.

    Returns a dict with:
      folds       — {patient: {ct_groups, ct_ids, cell_keep, gene_idx}}
                    where gene_idx indexes the ORIGINAL gene axis
      union_genes — sorted union of all folds' gene sets
    Caller subsets X/mask to union_genes once, then maps each fold's gene_idx
    into that union via np.searchsorted.
    """
    folds = {}
    union = set()
    for i, p in enumerate(held_out_patients):
        train_cells = pat_ids != p
        ct_groups, ct_ids, cell_keep = select_ct_groups(
            obs, train_cells, split_ct_groups)
        # rank variance on training cells that also survive this fold's CT set
        gene_idx = select_hvg_genes(X, mask, train_cells & cell_keep, n_backfill)
        folds[p] = {"ct_groups": ct_groups, "ct_ids": ct_ids,
                    "cell_keep": cell_keep, "gene_idx": gene_idx}
        union.update(gene_idx.tolist())
        if verbose:
            print(f"    [plan {i+1}/{len(held_out_patients)}] {p}: "
                  f"{len(ct_groups)} CTs, {len(gene_idx):,} genes, "
                  f"{int(cell_keep.sum()):,} cells", flush=True)

    union_genes = np.array(sorted(union), dtype=np.int64)
    if verbose:
        sizes = {len(f["gene_idx"]) for f in folds.values()}
        ncts = {len(f["ct_groups"]) for f in folds.values()}
        print(f"  gene-set sizes across folds: {sorted(sizes)}")
        print(f"  CT-group counts across folds: {sorted(ncts)}")
        print(f"  union gene space: {len(union_genes):,} "
              f"(vs {X.shape[1]:,} shared, {sorted(sizes)[0]:,} per fold)")
    return {"folds": folds, "union_genes": union_genes}
