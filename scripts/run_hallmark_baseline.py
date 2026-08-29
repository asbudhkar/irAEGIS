#!/usr/bin/env python3
"""Fixed Hallmark pathway-score baseline — irAEGIS with no learned representation.

Answers Reviewer 1's question directly: is the pathway-masked autoencoder doing
anything, or would simply computing Hallmark pathway activity and running a
classifier work just as well?

    expression -> fixed Hallmark scores -> patient x cell-type aggregation
                                        -> gate + patient-level LR -> prediction

The pathway score for cell i and pathway p is the mean log-normalised
expression of the genes annotated to p:

    s[i, p] = (X @ M)[i, p] / |{g : M[g, p] = 1}|

Everything downstream of that is identical to irAEGIS — the same top-25%-by-norm
patient x cell-type aggregation, the same inner-LOOCV cell-type gate, the same
stacked patient-level classifier. The single difference is that the 50-dimensional
pathway representation is *fixed by the gene sets* rather than learned. Any gap
is therefore attributable to representation learning, not to the classifier.

Leakage: pathway scores are a per-cell function of fixed external gene sets, so
they involve no fitting and cannot leak. Cell-type grouping is data-dependent
and is derived per fold from training patients only (--fold-selection). The
classifiers, gate and stacker are already fold-restricted. No HVG selection is
needed since scoring uses the full annotated gene set.

Outputs to results/iraegis/<cohort>/hallmark_baseline/:
    per_fold.csv   per-patient predictions and fold geometry
    summary.json   AUC / AUPRC with bootstrap 95% CIs

Usage:
    python scripts/run_hallmark_baseline.py --cohort GSE189125_pre_ici
    python scripts/run_hallmark_baseline.py --all --fold-selection
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.config import RESULTS_IRAEGIS, COHORTS_REAL
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.train_utils import train_h_concat_gated_concat_en
from models.iraegis.fold_selection import select_ct_groups

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"
N_BOOTSTRAP = 1000
_CHUNK = 20000          # cells per chunk when scoring, to bound peak memory


def hallmark_scores(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-cell pathway scores: mean log-normalised expression of member genes.

    X    : (cells, genes) log-normalised expression
    mask : (genes, pathways) binary Hallmark membership
    -> (cells, pathways) float32

    Computed in cell chunks so the full expression matrix is never duplicated;
    the result is only cells x 50.
    """
    n_per_pw = np.asarray(mask.sum(axis=0)).ravel().clip(min=1)
    out = np.empty((X.shape[0], mask.shape[1]), dtype=np.float32)
    for i in range(0, X.shape[0], _CHUNK):
        out[i:i + _CHUNK] = (X[i:i + _CHUNK] @ mask) / n_per_pw[None, :]
    return out


def _bootstrap_ci(y, p, metric, n=N_BOOTSTRAP, seed=0):
    """Stratified patient-level percentile bootstrap."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        vals.append(metric(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_cohort(cohort: str, fold_selection: bool = True) -> dict:
    print(f"\n{'=' * 70}\n  Hallmark pathway-score baseline: {cohort}\n{'=' * 70}")

    X, obs, _gn, ct_groups, ct_ids, pat_ids, pat_labels, prior = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=fold_selection,
        verbose=False)

    patients = sorted(pat_labels.keys())
    y_all = np.array([pat_labels[p] for p in patients], dtype=np.int64)

    # Fixed pathway scores replace the learned h. No fitting, no leakage.
    S = hallmark_scores(X, prior["mask"])
    del X
    pw_names = list(prior["pathway_names"])
    print(f"  {S.shape[0]:,} cells scored over {S.shape[1]} Hallmark pathways "
          f"({len(patients)} patients, {int(y_all.sum())} positive)")

    out_dir = RESULTS_IRAEGIS / cohort / "hallmark_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    oof = np.full(len(patients), np.nan)
    rows, t0 = [], time.time()

    for i, held in enumerate(patients):
        if fold_selection:
            # cell-type grouping from training-fold patients only
            groups_f, ct_f_all, keep = select_ct_groups(
                obs, pat_ids != held, SPLIT_CT_GROUPS)
            S_f, ct_f, pat_f = S[keep], ct_f_all[keep], pat_ids[keep]
        else:
            groups_f, S_f, ct_f, pat_f = ct_groups, S, ct_ids, pat_ids

        hi = patients.index(held)
        res = train_h_concat_gated_concat_en(
            S_f, pat_f, ct_f, pat_labels, groups_f, verbose=False,
            only_patient_idx=hi)
        oof[hi] = res["oof_probs"][hi]
        rows.append({"patient": held, "label": int(y_all[hi]),
                     "oof_prob": float(oof[hi]), "n_ct": len(groups_f),
                     "n_cells": int(S_f.shape[0])})
        pd.DataFrame(rows).to_csv(out_dir / "per_fold.csv", index=False)
        print(f"  [{i + 1}/{len(patients)}] {held} (label {y_all[hi]}): "
              f"{oof[hi]:.4f}   {len(groups_f)} CTs")

    auc = float(roc_auc_score(y_all, oof))
    auprc = float(average_precision_score(y_all, oof))
    a_lo, a_hi = _bootstrap_ci(y_all, oof, roc_auc_score)
    p_lo, p_hi = _bootstrap_ci(y_all, oof, average_precision_score)

    summary = {
        "cohort": cohort,
        "model": "fixed Hallmark pathway scores (mean log-normalised expression "
                 "of member genes); no autoencoder, no latent representation",
        "downstream": "identical to irAEGIS: top-25%-by-norm patient x CT "
                      "aggregation, inner-LOOCV cell-type gate, stacked "
                      "patient-level logistic regression",
        "fold_wise_ct_grouping": fold_selection,
        "n_patients": len(patients), "n_positive": int(y_all.sum()),
        "n_pathways": int(S.shape[1]),
        "auc": auc, "auc_ci95": [a_lo, a_hi],
        "auprc": auprc, "auprc_ci95": [p_lo, p_hi],
        "n_bootstrap": N_BOOTSTRAP,
        "total_seconds": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  AUC   = {auc:.4f}  95% CI [{a_lo:.3f}, {a_hi:.3f}]")
    print(f"  AUPRC = {auprc:.4f}  95% CI [{p_lo:.3f}, {p_hi:.3f}]")
    print(f"  -> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-fold-selection", action="store_true",
                    help="Use cohort-wide cell-type grouping instead of "
                         "deriving it per fold from training patients.")
    args = ap.parse_args()

    cohorts = [args.cohort] if args.cohort else (
        list(COHORTS_REAL) if args.all else ap.error("pass --cohort or --all"))
    res = [run_cohort(c, not args.no_fold_selection) for c in cohorts]

    if len(res) > 1:
        print(f"\n{'=' * 70}\n  SUMMARY — Hallmark pathway-score baseline\n{'=' * 70}")
        print(f"  {'cohort':<34} {'AUC':>7} {'95% CI':>16} {'AUPRC':>7}")
        for s in res:
            ci = f"[{s['auc_ci95'][0]:.3f}, {s['auc_ci95'][1]:.3f}]"
            print(f"  {s['cohort']:<34} {s['auc']:>7.4f} {ci:>16} {s['auprc']:>7.4f}")


if __name__ == "__main__":
    main()
