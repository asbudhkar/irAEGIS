#!/usr/bin/env python3
# irAEGIS patient-level inference: gated cell-type stacking + top-25 aggregation.

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
IRAEGIS_DIR = REPO / "results" / "iraegis"
IRAEGIS_OOF = REPO / "results" / "iraegis_oof"
GATE = 0.50

REGROUP = {
    "Naive B cells": "B_cells", "Plasma B cells": "B_cells",
    "Pre-B cells": "B_cells", "Pro-B cells": "B_cells",
    "Natural killer  cells": "NK_cells", "Natural killer cells": "NK_cells",
    "Progenitor cells": "HSC_Prog", "HSC/MPP cells": "HSC_Prog",
    "Erythroid-like and erythroid precursor cells": "Erythroid",
}


def aggregate_top25_mean(h, pat_ids, ct_ids, pats, n_ct):
    """Per (patient, CT) mean of top-25%-by-norm cells."""
    P = h.shape[1]
    out = np.zeros((len(pats), n_ct, P), dtype=np.float32)
    for j in range(n_ct):
        for i, p in enumerate(pats):
            mask = (pat_ids == p) & (ct_ids == j)
            if mask.sum() == 0:
                continue
            cells = h[mask]
            if cells.shape[0] >= 4:
                norms = np.linalg.norm(cells, axis=1)
                cutoff = np.percentile(norms, 75)
                top = cells[norms >= cutoff]
                out[i, j] = top.mean(axis=0) if len(top) > 0 else cells.mean(axis=0)
            else:
                out[i, j] = cells.mean(axis=0)
    return out


def per_ct_inner_loocv_aucs(pat_h, pat_labels, train_idx, C=0.1):
    n_pat, n_ct, _ = pat_h.shape
    aucs = np.zeros(n_ct)
    tr_labels = pat_labels[train_idx]
    if (tr_labels == 1).sum() < 2 or (tr_labels == 0).sum() < 2:
        return aucs
    for j in range(n_ct):
        Xj = pat_h[train_idx, j]
        loo = LeaveOneOut()
        preds = np.zeros(len(train_idx))
        for tr2, va2 in loo.split(Xj):
            sc = StandardScaler().fit(Xj[tr2])
            try:
                lr = LogisticRegression(solver="liblinear", max_iter=2000,
                                          C=C, class_weight="balanced",
                                          random_state=0)
                lr.fit(sc.transform(Xj[tr2]), tr_labels[tr2])
                preds[va2[0]] = lr.predict_proba(sc.transform(Xj[va2]))[0, 1]
            except Exception:
                preds[va2[0]] = 0.5
        try:
            aucs[j] = roc_auc_score(tr_labels, preds)
        except Exception:
            aucs[j] = 0.5
    return aucs


def gated_ct_stacking_predict(pat_h, pat_labels, P_idx, gate=GATE, use_logit=True):
    # patient-level prediction.

    n_pat, n_ct, _ = pat_h.shape
    tr_idx = np.array([i for i in range(n_pat) if i != P_idx])
    per_ct = per_ct_inner_loocv_aucs(pat_h, pat_labels, tr_idx)
    selected = [j for j, a in enumerate(per_ct) if a >= gate]
    if not selected:
        selected = [int(np.argmax(per_ct))]
    feats = np.full((n_pat, len(selected)), 0.5)
    for col, j in enumerate(selected):
        X_tr_full = pat_h[tr_idx, j]
        y_tr_full = pat_labels[tr_idx]
        if (y_tr_full == 1).sum() < 2 or (y_tr_full == 0).sum() < 2:
            continue
        sc = StandardScaler().fit(X_tr_full)
        try:
            lr = LogisticRegression(max_iter=2000, solver="lbfgs",
                                      class_weight="balanced", random_state=0, C=0.1)
            lr.fit(sc.transform(X_tr_full), y_tr_full)
            feats[P_idx, col] = lr.predict_proba(sc.transform(pat_h[[P_idx], j]))[0, 1]
        except Exception:
            feats[P_idx, col] = 0.5
        for i in tr_idx:
            tr_minus_i = np.array([m for m in tr_idx if m != i])
            X_tr2 = pat_h[tr_minus_i, j]
            y_tr2 = pat_labels[tr_minus_i]
            if (y_tr2 == 1).sum() < 2 or (y_tr2 == 0).sum() < 2:
                continue
            try:
                sc2 = StandardScaler().fit(X_tr2)
                lr2 = LogisticRegression(max_iter=2000, solver="lbfgs",
                                          class_weight="balanced", random_state=0, C=0.1)
                lr2.fit(sc2.transform(X_tr2), y_tr2)
                feats[i, col] = lr2.predict_proba(sc2.transform(pat_h[[i], j]))[0, 1]
            except Exception:
                feats[i, col] = 0.5
    f = feats.copy()
    if use_logit:
        f = np.log(np.clip(f, 1e-6, 1 - 1e-6) / np.clip(1 - f, 1e-6, 1 - 1e-6))
    outer = LogisticRegression(C=1.0, max_iter=2000,
                                class_weight="balanced", random_state=0)
    outer.fit(f[tr_idx], pat_labels[tr_idx])
    return outer.predict_proba(f[[P_idx]])[0, 1]

def run_cohort(cohort: str):
    src = IRAEGIS_DIR / cohort
    if not src.exists():
        print(f"  [{cohort}] SKIP — no iraegis/ dir")
        return None
    h = np.load(src / "h_cells.npy")
    meta = pd.read_csv(src / "cell_meta.csv")
    meta["label"] = meta["irAE_status"].astype(str).str.strip().isin(
        ["Yes", "Severe"]).astype(int)
    ct_groups = json.load(open(src / "ct_groups.json"))["ct_groups"]
    fine = meta["final_celltype"].astype(str).values
    coarse = np.array([REGROUP.get(c, c) for c in fine])
    name_to_idx = {n: i for i, n in enumerate(ct_groups)}
    ct_ids = np.array([name_to_idx.get(n, -1) for n in coarse])
    valid = ct_ids != -1
    h = h[valid]; meta = meta.loc[valid].reset_index(drop=True); ct_ids = ct_ids[valid]
    pat_ids = meta["patient_id"].astype(str).values
    pats = sorted(set(pat_ids))
    pat_labels = np.array([
        int((meta.loc[meta["patient_id"].astype(str) == p, "label"]).iloc[0])
        for p in pats])

    pat_h = aggregate_top25_mean(h, pat_ids, ct_ids, pats, len(ct_groups))
    n_pat = len(pats)
    probs = np.zeros(n_pat)
    for P_idx in range(n_pat):
        probs[P_idx] = gated_ct_stacking_predict(pat_h, pat_labels, P_idx)
    auc = float(roc_auc_score(pat_labels, probs))
    print(f"  [{cohort}]  n={n_pat}, gated CT stacking + top-25 AUC={auc:.4f}")

    out_d = IRAEGIS_OOF / cohort
    out_d.mkdir(parents=True, exist_ok=True)
    np.save(out_d / "iraegis_oof_probs.npy", probs.astype(np.float32))
    np.save(out_d / "iraegis_oof_labels.npy", pat_labels.astype(np.int8))
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=None)
    args = ap.parse_args()
    cohorts = args.cohorts or [
        "GSE189125_pre_ici", "GSE216329_integrated_pre_ici",
        "GSE249898_integrated_pre_ici", "GSE285888_pre_ici",
        # simulated cohorts: RS = Restricted signal, DS = Distributed signal
        "RS_cohort1", "RS_cohort2", "RS_cohort3",
        "DS_cohort1", "DS_cohort2", "DS_cohort3",
    ]
    for c in cohorts:
        try:
            run_cohort(c)
        except Exception as e:
            print(f"  [{c}] ERROR — {e}")


if __name__ == "__main__":
    main()
