#!/usr/bin/env python3
"""ScRAT, singleDeep and hierarchical MIL under the leakage-free protocol (R3.6).

These three are separate published codebases, but they share a single leak, and
it is the same one irAEGIS had. All of them reach the data through
utils.data_helpers.load_cells_cohort, which selects the top-N most variable
genes over the WHOLE cohort before any fold is formed:

    X, gene_names = select_top_variance_genes(X, gene_names, n_genes)

Everything after that is already fold-correct - each wrapper loops over
leave-one-patient-out folds and calls the method's own train_fold() with the
fold's patients. So the fix is narrow: defer gene selection, then rank genes by
variance over TRAINING cells only inside each fold and hand the method its own
gene subset.

    for each held-out patient:
        select genes from TRAINING cells only, by default using irAEGIS's exact
        rule (pathway-active genes + top 2000 non-pathway HVGs), so feature
        selection is identical across all methods as R3.6 requires
        rebuild that method's inputs on the fold's gene set
        call the method's own train_fold() unchanged
        score the held-out patient

The methods themselves are untouched - no hyperparameters, architectures or
training procedures are modified - so any change against the published numbers
is attributable to gene selection alone.

Per-method input handling differs only in shape:
    scrat        takes the full matrix and subsets patients internally
    singledeep   takes X_tr / X_te separately
    hiermil      takes per-(patient, cell type) bags, rebuilt per fold

Outputs to results/iraegis/<cohort>/external_baselines_leakage_free/<method>/

Usage:
    python scripts/run_external_baselines_leakage_free.py --cohort GSE189125_pre_ici --method scrat
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

SCPIP = Path("/Users/aishu/scPIP")
sys.path.insert(0, str(SCPIP))
REPO = Path(__file__).resolve().parents[1]

from utils.config import RANDOM_STATE, get_cv_folds
from utils import profiler
from utils.data_helpers import load_cells_cohort
from utils.celltype_groups import infer_celltype_groups, EXCLUDE_LABEL

N_GENES = 2000
N_BOOT = 1000
_CHUNK = 2000
SPLIT = ["T_cells", "Monocytes", "Dendritic"]


def hvg_train_only(X, train_mask, k=N_GENES):
    """Top-k genes by variance over TRAINING cells only, chunked."""
    rows = np.where(train_mask)[0]; n = len(rows)
    var = np.empty(X.shape[1], dtype=np.float64)
    for s in range(0, X.shape[1], _CHUNK):
        cols = np.arange(s, min(s + _CHUNK, X.shape[1]))
        sub = X[np.ix_(rows, cols)].astype(np.float64)
        m = sub.sum(0) / n
        var[s:s + len(cols)] = (sub * sub).sum(0) / n - m * m
    return np.sort(np.argsort(var)[-k:])


def _boot(y, p, metric, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        v.append(metric(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def pathway_mask_for(gene_names):
    """Binary (genes x pathways) Hallmark membership aligned to `gene_names`.

    Needed for the `matched` gene space, which reproduces irAEGIS's rule of
    keeping every pathway-active gene plus the top non-pathway HVGs.
    """
    from utils.config import PRIOR_NPZ
    raw = np.load(PRIOR_NPZ, allow_pickle=True)
    pg = list(raw["gene_names"]); m = raw["mask"]
    m = m if m.shape[0] == len(pg) else m.T
    idx = {g: i for i, g in enumerate(pg)}
    out = np.zeros((len(gene_names), m.shape[1]), dtype=m.dtype)
    for i, g in enumerate(gene_names):
        j = idx.get(g)
        if j is not None:
            out[i] = m[j]
    return out


def load_deferred(cohort):
    """Load with gene selection DEFERRED (n_genes=None), then apply the same
    cell-type grouping and filtering the published wrappers use."""
    X, pat_ids, ct_raw, genes, patients, labels = load_cells_cohort(
        cohort, None, None, None)
    _, ct_map = infer_celltype_groups(
        pd.DataFrame({"final_celltype": ct_raw, "patient_harmony": pat_ids}),
        min_patients=3, min_cells_per_patient=10, split_groups=SPLIT)
    ct = np.array([ct_map.get(c, EXCLUDE_LABEL) for c in ct_raw])
    keep = (ct != EXCLUDE_LABEL) & (ct != "Other")
    return X[keep], pat_ids[keep], ct[keep], genes, patients, labels


def run(cohort, method, gene_space="matched"):
    print(f"\n{'='*70}\n  Leakage-free {method}: {cohort}\n{'='*70}", flush=True)
    X, pat_ids, ct, gene_names, patients, labels = load_deferred(cohort)
    pmask = pathway_mask_for(gene_names) if gene_space == "matched" else None
    sorted_pats, sorted_labs, folds = get_cv_folds(patients, labels)
    y = np.array(sorted_labs, dtype=np.int8)
    pat_to_lab = dict(zip(sorted_pats, sorted_labs))
    print(f"  {X.shape[0]:,} cells x {X.shape[1]:,} genes (gene selection deferred), "
          f"{len(sorted_pats)} patients; HVG={N_GENES} chosen per fold", flush=True)

    oof = np.full(len(sorted_pats), np.nan, dtype=np.float32)
    # same profiler the published baseline runs used, so wall time and peak
    # memory land in results/runtime_summary.csv alongside the existing rows
    profiler.start(f"{method}_leakage_free_{gene_space}", cohort)
    t0 = time.time()

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        pt = [sorted_pats[i] for i in tr_idx]
        pe = [sorted_pats[i] for i in va_idx]
        tr_cells = np.isin(pat_ids, pt)
        if gene_space == "matched":
            from models.iraegis.fold_selection import select_hvg_genes
            genes = select_hvg_genes(X, pmask, tr_cells)   # irAEGIS's exact rule
        else:
            genes = hvg_train_only(X, tr_cells)            # TRAINING cells only
        Xf = X[:, genes]
        try:
            if method == "scrat":
                import models.baselines.scrat as scrat
                probs = scrat.train_fold(
                    X=Xf, pat_ids=pat_ids, labels=labels,
                    patients_tr=pt, patients_te=pe,
                    patients_all=sorted_pats, labels_all=y,
                    n_genes=len(genes), celltypes=ct, seed=fold_i)
            elif method == "singledeep":
                import models.baselines.singledeep as sd
                te = np.isin(pat_ids, pe)
                cell_lab = np.array([pat_to_lab[p] for p in pat_ids], dtype=np.int8)
                probs = sd.train_fold(
                    X_tr=Xf[tr_cells], y_tr_cell=cell_lab[tr_cells],
                    ct_tr=ct[tr_cells], pat_tr=pat_ids[tr_cells],
                    X_te=Xf[te], ct_te=ct[te], pat_te=pat_ids[te],
                    patients_te=pe, ct_names=sorted(set(ct.tolist())),
                    pat_labels_tr=np.array([pat_to_lab[p] for p in pt], dtype=np.int8))
            elif method == "hiermil":
                import models.baselines.hierarchical_mil as hmil
                all_ct = sorted(set(ct.tolist()))
                def bags_for(ps):                     # rebuilt on this fold's genes
                    b = {}
                    for p in ps:
                        idx = np.where(pat_ids == p)[0]
                        b[p] = {c: Xf[idx[ct[idx] == c]] for c in all_ct
                                if (ct[idx] == c).any()}
                    return b
                probs = hmil.train_fold(
                    bags_for(pt), np.array([pat_to_lab[p] for p in pt], dtype=np.int8),
                    bags_for(pe), pe, n_genes=len(genes), all_ct=all_ct)
            else:
                raise ValueError(method)
            for vi, pi in enumerate(va_idx):
                oof[pi] = float(probs[vi])
        except Exception as e:
            print(f"    fold {fold_i} FAILED: {type(e).__name__}: {e}", flush=True)
        if (fold_i + 1) % 4 == 0 or fold_i == len(folds) - 1:
            print(f"    fold {fold_i+1}/{len(folds)}  ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
        del Xf

    profiler.stop()
    m = ~np.isnan(oof)
    out = REPO / "results" / "iraegis" / cohort / f"external_baselines_leakage_free_{gene_space}" / method
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient": sorted_pats, "label": y, "oof_prob": oof}).to_csv(
        out / "per_fold.csv", index=False)
    if m.sum() >= 3 and len(set(y[m])) == 2:
        auc = float(roc_auc_score(y[m], oof[m]))
        lo, hi = _boot(y[m], oof[m], roc_auc_score)
        ap = float(average_precision_score(y[m], oof[m]))
        (out / "summary.json").write_text(json.dumps(
            {"cohort": cohort, "method": method, "n_scored": int(m.sum()),
             "gene_space": gene_space, "hvg_per_fold": N_GENES, "auc": auc, "auc_ci95": [lo, hi], "auprc": ap,
             "protocol": "gene selection from training cells only; method itself "
                         "unmodified", "total_seconds": time.time() - t0}, indent=2))
        print(f"\n  AUC = {auc:.4f}  95% CI [{lo:.3f}, {hi:.3f}]   AUPRC = {ap:.4f} "
              f"  ({int(m.sum())}/{len(sorted_pats)} scored)", flush=True)
    else:
        print(f"\n  too few folds scored ({int(m.sum())})", flush=True)
    print(f"  -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--method", required=True, choices=["scrat", "singledeep", "hiermil"])
    ap.add_argument("--gene-space", choices=["matched", "hvg2000"], default="matched",
                    help="matched (default) = the published rule these methods "
                         "already used via select_prior_genes - pathway-active "
                         "genes + top-2000 non-pathway HVG - ranked on training "
                         "cells only. hvg2000 = no pathway prior, a stricter variant.")
    a = ap.parse_args()
    run(a.cohort, a.method, a.gene_space)


if __name__ == "__main__":
    main()
