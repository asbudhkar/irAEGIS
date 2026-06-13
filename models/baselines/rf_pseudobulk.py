#!/usr/bin/env python3

# Random Forest on per-CT pseudobulk 
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.config import (RANDOM_STATE, get_cv_folds,
                          effective_cv_mode, cohort_h5ad, SIM_PRIOR_NPZ, PRIOR_NPZ)
from utils.profiler import start as prof_start, stop as prof_stop
from utils.data_helpers import load_cells_cohort

N_PCA = 2
N_ESTIMATORS = 500
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 3
CLASS_WEIGHT = "balanced"
MIN_CELLS_PER_CT = 50


def _patient_pseudobulk(X, pat_ids, celltypes, patients, ct_name):
    n_feat = X.shape[1]
    out = np.zeros((len(patients), n_feat), dtype=np.float32)
    mask = celltypes == ct_name
    X_ct, pat_ct = X[mask], pat_ids[mask]
    for pi, p in enumerate(patients):
        cells = X_ct[pat_ct == p]
        if len(cells) > 0:
            out[pi] = cells.mean(axis=0)
    return out


def _fit_rf(X_tr, y_tr, X_va):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        n_jobs=-1,
        random_state=RANDOM_STATE)
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_va_s)[:, 1]


def _summarize(pat_arr, oof_probs, valid):
    prevalence = float(pat_arr.sum() / len(pat_arr))
    try:
        auc = float(roc_auc_score(pat_arr[valid], oof_probs[valid]))
    except Exception:
        auc = float("nan")
    try:
        ap = float(average_precision_score(pat_arr[valid], oof_probs[valid]))
    except Exception:
        ap = float("nan")
    preds = (oof_probs[valid] >= 0.5).astype(int)
    try:
        ba = float(balanced_accuracy_score(pat_arr[valid], preds))
    except Exception:
        ba = float("nan")
    try:
        mcc = float(matthews_corrcoef(pat_arr[valid], preds))
    except Exception:
        mcc = float("nan")
    return auc, ap, ba, mcc, prevalence


def run(args):
    h5ad = Path(args.h5ad) if args.h5ad else None
    if args.sim and h5ad is None:
        h5ad = cohort_h5ad(args.cohort)
    prior = Path(args.prior) if args.prior else None
    if prior is None and args.prior_genes:
        prior = SIM_PRIOR_NPZ if args.sim else PRIOR_NPZ

    X, pat_ids, celltypes, gene_names, patients, labels = load_cells_cohort(
        args.cohort, n_genes=5000, h5ad_path=h5ad,
        prior_genes_path=prior if args.prior_genes else None)

    from utils.celltype_groups import infer_celltype_groups, EXCLUDE_LABEL
    _, ct_map = infer_celltype_groups(
        pd.DataFrame({"final_celltype": celltypes, "patient_id": pat_ids}),
        min_patients=3, min_cells_per_patient=10,
        split_groups=["T_cells", "Monocytes", "Dendritic"])
    celltypes = np.array([ct_map.get(c, EXCLUDE_LABEL) for c in celltypes])
    keep = (celltypes != EXCLUDE_LABEL) & (celltypes != "Other")
    X, pat_ids, celltypes = X[keep], pat_ids[keep], celltypes[keep]

    sorted_patients, sorted_labels, folds = get_cv_folds(patients, labels)
    patients = sorted_patients
    pat_arr = np.array([dict(zip(patients, sorted_labels))[p] for p in patients])
    pat_labels = dict(zip(patients, sorted_labels))

    ct_unique = []
    for ct in sorted(set(celltypes)):
        ct_mask = celltypes == ct
        if int(ct_mask.sum()) < MIN_CELLS_PER_CT:
            continue
        ct_pats = set(pat_ids[ct_mask])
        n_yes = sum(1 for p in ct_pats if pat_labels.get(p, 0) == 1)
        n_no = sum(1 for p in ct_pats if pat_labels.get(p, 0) == 0)
        if n_yes < 2 or n_no < 2:
            continue
        ct_unique.append(ct)

    out_dir = Path(args.results_dir) / args.cohort / "rf_pseudobulk"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pats = len(patients)
    print(f"\n[Patient-level] per-CT PCA({N_PCA}) concat + RF "
          f"(n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
          f"min_samples_leaf={MIN_SAMPLES_LEAF}) — {len(ct_unique)} CTs kept")

    pat_h_per_ct = {ct: _patient_pseudobulk(X, pat_ids, celltypes, patients, ct)
                    for ct in ct_unique}

    oof = np.full(n_pats, np.nan, dtype=np.float32)
    cnt = np.zeros(n_pats, dtype=np.int32)

    for tr_idx, va_idx in folds:
        tr_parts, va_parts = [], []
        for ct in ct_unique:
            ct_h = pat_h_per_ct[ct]
            sc = StandardScaler()
            ct_tr = sc.fit_transform(ct_h[tr_idx])
            ct_va = sc.transform(ct_h[va_idx])
            n_comp = min(N_PCA, ct_tr.shape[0] - 1, ct_tr.shape[1])
            n_comp = max(n_comp, 1)
            pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
            tr_parts.append(pca.fit_transform(ct_tr))
            va_parts.append(pca.transform(ct_va))

        X_tr = np.concatenate(tr_parts, axis=1)
        X_va = np.concatenate(va_parts, axis=1)
        va_probs = _fit_rf(X_tr, pat_arr[tr_idx], X_va)
        for vi, pi in enumerate(va_idx):
            if np.isnan(oof[pi]):
                oof[pi] = 0.0
            oof[pi] += va_probs[vi]
            cnt[pi] += 1

    valid = cnt > 0
    oof[valid] /= cnt[valid]
    auc, ap, ba, mcc, prev = _summarize(pat_arr, oof, valid)
    print(f"  Patient AUC={auc:.4f}  BalAcc={ba:.4f}  MCC={mcc:.4f}  AUPRC={ap:.4f}")

    summaries = {
        "__patient_level__": {
            "fold_aucs": [auc], "mean_fold_auc": auc, "std_fold_auc": 0.0,
            "fold_aps": [ap], "mean_fold_ap": ap,
            "balanced_accuracy": ba, "mcc": mcc, "prevalence": prev,
            "cv_mode": effective_cv_mode(len(patients)),
            "n_patients_yes": int(pat_arr.sum()),
            "n_patients_no": int(len(pat_arr) - pat_arr.sum()),
            "n_celltypes_kept": len(ct_unique),
        }
    }

    np.save(out_dir / "patient_oof_probs.npy", oof.astype(np.float32))
    np.save(out_dir / "patient_oof_labels.npy", pat_arr.astype(np.int8))

    with open(out_dir / "irae_per_ct_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nDone -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--h5ad", default=None)
    ap.add_argument("--prior", default=None)
    ap.add_argument("--prior-genes", action="store_true")
    import os
    ap.add_argument("--results-dir", default=os.environ.get(
        "IRAEGIS_BASELINES_DIR", str(REPO_ROOT / "results" / "baselines")))
    args = ap.parse_args()
    prof_start("rf_pseudobulk", args.cohort)
    run(args)
    prof_stop()


if __name__ == "__main__":
    main()
