#!/usr/bin/env python3
"""per-CT-PCA + Elastic Net patient-level classifier, leakage-free per fold.

An earlier internal analysis (docs/permutation_test_results.txt, 2026-04-29)
compared eleven patient-level combiners across the four cohorts and concluded:

    "No single method wins all cohorts. The best approach depends on signal
     structure: concentrated (stacking wins) vs distributed (gated/PCA+EN wins)."
    "Stacked meta-learner ... brilliant when individual CTs have strong signal
     (GSE189125: 0.937), but collapses when per-CT signal is weak/noisy
     (GSE249898: 0.463, GSE216329: 0.237)."
    "per-CT-PCA + EN is the most robust choice - never collapses, consistently
     top-2 across all 4 cohorts. Recommended as primary patient-level method."

That recommendation predates the leakage-free protocol, so this is a
pre-specified alternative rather than one chosen to rescue a failing cohort.
This reimplements it under the strict protocol: each fold reloads the
autoencoder trained without its held-out patient, and every fitted object -
the per-cell-type scalers, the PCAs, the outer scaler and the elastic net -
is fit on training patients only.

Method: each cell type's 50 pathway means -> PCA -> concatenate across cell
types -> elastic net. n_components, l1_ratio and C are chosen per fold by
nested inner leave-one-out CV on the training patients, so no hyperparameter
is set by hand or by looking at held-out performance.

Aggregation is a plain per-cell-type MEAN, as in the original implementation.

Outputs to results/iraegis/<cohort>/per_ct_pca_en/

Usage:
    python scripts/per_ct_pca_en.py --cohort GSE249898_integrated_pre_ici
"""
from __future__ import annotations

import argparse, gc, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.config import RESULTS_IRAEGIS, DEVICE, RANDOM_STATE
from models.iraegis.model_utils import PathwayAE
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.train_utils import (
    precompute_embeddings, AE_LATENT_DIM, AE_DROPOUT,
)
from models.iraegis.fold_selection import plan_folds

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"
PCA_GRID, L1_GRID, C_GRID = [2, 3], [0.5, 0.8, 1.0], [0.1, 1.0]
N_BOOT = 1000


def _boot(y, p, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _patient_mean(h, pat_f, ct_f, patients, n_ct):
    """patient x CT x pathway, plain mean (the original implementation)."""
    out = np.zeros((len(patients), n_ct, h.shape[1]), dtype=np.float32)
    for j in range(n_ct):
        for i, p in enumerate(patients):
            m = (pat_f == p) & (ct_f == j)
            if m.any():
                out[i, j] = h[m].mean(0)
    return out


def _en(C, l1):
    return LogisticRegression(penalty="elasticnet", solver="saga", C=C,
                              l1_ratio=l1, max_iter=10000,
                              class_weight="balanced", random_state=RANDOM_STATE)


def _fit_predict(mat, y, tr, held, n_pca, l1, C):
    """PCA per CT (fit on tr only) -> concat -> EN (fit on tr only) -> score held."""
    tr_parts, he_parts = [], []
    for j in range(mat.shape[1]):
        sc = StandardScaler().fit(mat[tr, j])
        a, b = sc.transform(mat[tr, j]), sc.transform(mat[[held], j])
        k = max(1, min(n_pca, len(tr) - 1, a.shape[1]))
        pca = PCA(n_components=k, random_state=RANDOM_STATE).fit(a)
        tr_parts.append(pca.transform(a)); he_parts.append(pca.transform(b))
    Xtr = np.concatenate(tr_parts, 1); Xhe = np.concatenate(he_parts, 1)
    sc2 = StandardScaler().fit(Xtr)
    Xtr_s, Xhe_s = sc2.transform(Xtr), sc2.transform(Xhe)
    try:
        m = _en(C, l1).fit(Xtr_s, y[tr])
        return float(m.predict_proba(Xhe_s)[0, 1]), Xtr_s
    except Exception:
        return 0.5, Xtr_s


def run(cohort, src_dir):
    print(f"\n{'='*70}\n  per-CT-PCA + Elastic Net (leakage-free): {cohort}\n{'='*70}")
    ck_dir = RESULTS_IRAEGIS / cohort / src_dir / "checkpoints"
    X, obs, _g, _cg, _ci, pat_ids, pat_labels, prior = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)
    patients = sorted(pat_labels.keys())
    y = np.array([pat_labels[p] for p in patients], dtype=np.int64)
    plan = plan_folds(X, obs, prior["mask"], pat_ids, patients, SPLIT_CT_GROUPS,
                      verbose=False)
    n_pw = prior["mask"].shape[1]
    print(f"  {len(patients)} patients ({y.sum()} pos); inner CV over "
          f"{len(PCA_GRID)*len(L1_GRID)*len(C_GRID)} combos per fold")

    out_dir = RESULTS_IRAEGIS / cohort / "per_ct_pca_en"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(patients), np.nan)
    recs, t0 = [], time.time()

    for i, held in enumerate(patients):
        ck = ck_dir / f"fold_{held}.pt"
        if not ck.exists():
            continue
        fs = plan["folds"][held]
        cells, genes = fs["cell_keep"], fs["gene_idx"]
        X_f = X[np.ix_(cells, genes)]
        ct_f, pat_f = fs["ct_ids"][cells], pat_ids[cells]
        obs_f = obs.loc[cells].reset_index(drop=True)
        groups_f = fs["ct_groups"]
        mask_f = torch.tensor(prior["mask"][genes, :], dtype=torch.float32)
        ae = PathwayAE(X_f.shape[1], n_pw, mask_f, AE_LATENT_DIM, AE_DROPOUT,
                       norm="ctbn", act="gelu", n_ct=len(groups_f))
        ae.attach_ct_head(len(groups_f))
        ae.load_state_dict(torch.load(ck, map_location="cpu"))
        ae.to(DEVICE); ae.eval()
        with tempfile.TemporaryDirectory(prefix="pcaen_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp), ct_ids=ct_f,
                                         verbose=False, suffix="_fold")
        mat = _patient_mean(h, pat_f, ct_f, patients, len(groups_f))
        hi = patients.index(held)
        tr = np.array([k for k in range(len(patients)) if k != hi])

        # nested inner LOOCV on the training patients to pick hyperparameters
        best, best_auc = None, -1.0
        for n_pca in PCA_GRID:
            for l1 in L1_GRID:
                for C in C_GRID:
                    inner = np.zeros(len(tr))
                    for a, b in LeaveOneOut().split(tr):
                        p_, _ = _fit_predict(mat, y, tr[a], tr[b[0]], n_pca, l1, C)
                        inner[b[0]] = p_
                    try:
                        au = roc_auc_score(y[tr], inner)
                    except Exception:
                        au = 0.5
                    if au > best_auc:
                        best_auc, best = au, (n_pca, l1, C)
        n_pca, l1, C = best
        oof[hi], _ = _fit_predict(mat, y, tr, hi, n_pca, l1, C)
        recs.append({"patient": held, "label": int(y[hi]), "oof_prob": float(oof[hi]),
                     "n_pca": n_pca, "l1_ratio": l1, "C": C, "inner_auc": float(best_auc)})
        pd.DataFrame(recs).to_csv(out_dir / "per_fold.csv", index=False)
        print(f"  [{i+1}/{len(patients)}] {held} (label {y[hi]}): {oof[hi]:.4f}   "
              f"pca={n_pca} l1={l1} C={C}   ({(time.time()-t0)/60:.1f} min)")
        del ae, h, X_f, mat; gc.collect()

    m = ~np.isnan(oof)
    auc = float(roc_auc_score(y[m], oof[m]))
    lo, hi_ = _boot(y[m], oof[m])
    summary = {"cohort": cohort, "method": "per-CT-PCA + elastic net, nested inner LOOCV",
               "aggregation": "plain per-cell-type mean",
               "n_patients": int(m.sum()), "n_positive": int(y[m].sum()),
               "auc": auc, "auc_ci95": [lo, hi_],
               "auprc": float(average_precision_score(y[m], oof[m])),
               "total_seconds": time.time() - t0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  AUC = {auc:.4f}  95% CI [{lo:.3f}, {hi_:.3f}]   (n={int(m.sum())})")
    print(f"  -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--src-dir", default="ae_per_fold_deterministic_foldsel")
    a = ap.parse_args()
    run(a.cohort, a.src_dir)


if __name__ == "__main__":
    main()
