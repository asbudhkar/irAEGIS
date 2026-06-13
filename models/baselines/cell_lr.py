#!/usr/bin/env python3

# Trains an L2 logistic regression on per patient and cell-type pseudobulk
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.config import (RANDOM_STATE, get_cv_folds,
                          effective_cv_mode, CV_AUTO_FALLBACK_SPLITS,
                          cohort_h5ad, SIM_PRIOR_NPZ, PRIOR_NPZ)
from utils.profiler import start as prof_start, stop as prof_stop
from utils.data_helpers import load_cells_cohort

MIN_CELLS_PER_CT = 50

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

    pat_labels = dict(zip(patients, labels))
    ct_unique = sorted(set(celltypes))
    ct_to_id = {c: i for i, c in enumerate(ct_unique)}
    ct_ids = np.array([ct_to_id[c] for c in celltypes])

    out_dir = Path(args.results_dir) / args.cohort / "cell_lr"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-CT irAE classification LR
    sorted_patients, sorted_labels, folds = get_cv_folds(patients, labels)
    patients = sorted_patients
    labels = sorted_labels
    pat_labels = dict(zip(patients, labels))
    pat_arr = np.array([pat_labels[p] for p in patients])

    print(f"\n[Per-CT] irAE classification (patient-level pseudobulk LR)")

    per_ct_summaries = {}

    for ct_idx, ct_name in enumerate(ct_unique):
        ct_mask = ct_ids == ct_idx
        ct_cell_idx = np.where(ct_mask)[0]

        if len(ct_cell_idx) < MIN_CELLS_PER_CT:
            print(f"\n  [{ct_name}] SKIP — only {len(ct_cell_idx)} cells")
            continue

        ct_pat_set = set(pat_ids[ct_cell_idx])
        ct_pats_yes = set(p for p in ct_pat_set if pat_labels[p] == 1)
        ct_pats_no = set(p for p in ct_pat_set if pat_labels[p] == 0)
        if len(ct_pats_yes) < 2 or len(ct_pats_no) < 2:
            print(f"\n  [{ct_name}] SKIP — too few patients "
                  f"(Yes={len(ct_pats_yes)}, No={len(ct_pats_no)})")
            continue

        n_feat = X.shape[1]
        X_ct = X[ct_cell_idx]
        pat_ct = pat_ids[ct_cell_idx]

        print(f"\n  [{ct_name}] {len(ct_cell_idx)} cells, "
              f"{len(ct_pats_yes)} Yes / {len(ct_pats_no)} No patients")

        eff_mode = effective_cv_mode(len(patients))
        is_loocv = eff_mode == "loocv"
        fold_aucs, fold_aps = [], []
        n_pats = len(patients)
        oof_pat_probs = np.full(n_pats, np.nan, dtype=np.float32)
        oof_pat_counts = np.zeros(n_pats, dtype=np.int32)

        for fold_i, (tr_idx, va_idx) in enumerate(folds):
            patients_tr = [patients[i] for i in tr_idx]
            patients_va = [patients[i] for i in va_idx]

            tr_mask = np.isin(pat_ct, patients_tr)
            X_tr_cells, pat_tr_cells = X_ct[tr_mask], pat_ct[tr_mask]
            y_tr = pat_arr[tr_idx]

            pat_X_tr = np.zeros((len(patients_tr), n_feat), dtype=np.float32)
            for pi, p in enumerate(patients_tr):
                p_cells = X_tr_cells[pat_tr_cells == p]
                if len(p_cells) > 0:
                    pat_X_tr[pi] = p_cells.mean(axis=0)

            pat_X_va = np.zeros((len(patients_va), n_feat), dtype=np.float32)
            for pi, p in enumerate(patients_va):
                p_cells = X_ct[pat_ct == p]
                if len(p_cells) > 0:
                    pat_X_va[pi] = p_cells.mean(axis=0)

            sc = StandardScaler()
            X_tr_scaled = sc.fit_transform(pat_X_tr)
            X_va_scaled = sc.transform(pat_X_va)

            clf = LogisticRegression(
                max_iter=2000, solver="liblinear", class_weight="balanced",
                random_state=RANDOM_STATE)
            clf.fit(X_tr_scaled, y_tr)

            try:
                va_probs = clf.predict_proba(X_va_scaled)[:, 1]
            except Exception:
                va_probs = np.full(len(va_idx), 0.5)

            for vi, pi in enumerate(va_idx):
                if np.isnan(oof_pat_probs[pi]):
                    oof_pat_probs[pi] = 0.0
                oof_pat_probs[pi] += va_probs[vi]
                oof_pat_counts[pi] += 1


            if not is_loocv:
                va_labels = pat_arr[va_idx]
                if len(set(va_labels.astype(int).tolist())) < 2:
                    fold_aucs.append(float("nan"))
                    fold_aps.append(float("nan"))
                else:
                    try:
                        fold_auc = roc_auc_score(va_labels, va_probs)
                        fold_ap = average_precision_score(va_labels, va_probs)
                    except Exception:
                        fold_auc = float("nan")
                        fold_ap = float("nan")
                    fold_aucs.append(fold_auc)
                    fold_aps.append(fold_ap)

        valid_oof = oof_pat_counts > 0
        oof_pat_probs[valid_oof] /= oof_pat_counts[valid_oof]
        prevalence = float(pat_arr.sum() / len(pat_arr))

        if is_loocv:
            try:
                mean_fold_auc = float(roc_auc_score(pat_arr[valid_oof], oof_pat_probs[valid_oof]))
            except Exception:
                mean_fold_auc = float("nan")
            std_fold_auc = 0.0
            try:
                mean_fold_ap = float(average_precision_score(pat_arr[valid_oof], oof_pat_probs[valid_oof]))
            except Exception:
                mean_fold_ap = float("nan")
        else:
            valid_aucs = [a for a in fold_aucs if not np.isnan(a)]
            valid_aps = [a for a in fold_aps if not np.isnan(a)]
            mean_fold_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
            std_fold_auc = float(np.std(valid_aucs)) if valid_aucs else float("nan")
            mean_fold_ap = float(np.mean(valid_aps)) if valid_aps else float("nan")

        oof_preds = (oof_pat_probs[valid_oof] >= 0.5).astype(int)
        try:
            bal_acc = float(balanced_accuracy_score(pat_arr[valid_oof], oof_preds))
        except Exception:
            bal_acc = float("nan")
        try:
            mcc = float(matthews_corrcoef(pat_arr[valid_oof], oof_preds))
        except Exception:
            mcc = float("nan")

        cv_label = {"loocv": "LOOCV", "kfold": f"{CV_AUTO_FALLBACK_SPLITS}-fold CV"}.get(eff_mode, "3×3 CV")
        print(f"    {cv_label} AUC: {mean_fold_auc:.4f}  BalAcc: {bal_acc:.4f}  "
              f"MCC: {mcc:.4f}  AUPRC: {mean_fold_ap:.4f} (base={prevalence:.3f})")

        per_ct_summaries[ct_name] = {
            "fold_aucs": fold_aucs if not is_loocv else [mean_fold_auc],
            "mean_fold_auc": mean_fold_auc,
            "std_fold_auc": std_fold_auc,
            "fold_aps": fold_aps if not is_loocv else [mean_fold_ap],
            "mean_fold_ap": mean_fold_ap,
            "balanced_accuracy": bal_acc,
            "mcc": mcc,
            "prevalence": prevalence,
            "cv_mode": eff_mode,
            "n_folds": len(folds),
            "n_cells": len(ct_cell_idx),
            "n_patients_yes": len(ct_pats_yes),
            "n_patients_no": len(ct_pats_no),
        }

    # Patient-level prediction
    print(f"\nPatient irAE prediction")
    eff_mode = effective_cv_mode(len(patients))
    is_loocv = eff_mode == "loocv"
    n_feat = X.shape[1]
    pat_fold_aucs, pat_fold_aps = [], []
    pat_oof_probs = np.full(len(pat_arr), np.nan, dtype=np.float32)
    pat_oof_counts = np.zeros(len(pat_arr), dtype=np.int32)

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        patients_tr = [patients[i] for i in tr_idx]
        patients_va = [patients[i] for i in va_idx]
        y_tr = pat_arr[tr_idx]

        pat_X_tr = np.zeros((len(patients_tr), n_feat), dtype=np.float32)
        for pi, p in enumerate(patients_tr):
            p_cells = X[pat_ids == p]
            if len(p_cells) > 0:
                pat_X_tr[pi] = p_cells.mean(axis=0)

        pat_X_va = np.zeros((len(patients_va), n_feat), dtype=np.float32)
        for pi, p in enumerate(patients_va):
            p_cells = X[pat_ids == p]
            if len(p_cells) > 0:
                pat_X_va[pi] = p_cells.mean(axis=0)

        sc = StandardScaler()
        X_tr_scaled = sc.fit_transform(pat_X_tr)
        X_va_scaled = sc.transform(pat_X_va)

        clf = LogisticRegression(
            max_iter=2000, solver="liblinear", class_weight="balanced",
            random_state=RANDOM_STATE)
        clf.fit(X_tr_scaled, y_tr)

        try:
            va_probs = clf.predict_proba(X_va_scaled)[:, 1]
        except Exception:
            va_probs = np.full(len(va_idx), 0.5)

        for vi, pi in enumerate(va_idx):
            if np.isnan(pat_oof_probs[pi]):
                pat_oof_probs[pi] = 0.0
            pat_oof_probs[pi] += va_probs[vi]
            pat_oof_counts[pi] += 1

        if not is_loocv:
            va_labels = pat_arr[va_idx]
            if len(set(va_labels.astype(int).tolist())) < 2:
                pat_fold_aucs.append(float("nan"))
                pat_fold_aps.append(float("nan"))
            else:
                try:
                    pat_fold_aucs.append(roc_auc_score(va_labels, va_probs))
                    pat_fold_aps.append(average_precision_score(va_labels, va_probs))
                except Exception:
                    pat_fold_aucs.append(float("nan"))
                    pat_fold_aps.append(float("nan"))

    valid_oof = pat_oof_counts > 0
    pat_oof_probs[valid_oof] /= pat_oof_counts[valid_oof]
    prevalence = float(pat_arr.sum() / len(pat_arr))

    if is_loocv:
        try:
            pat_mean = float(roc_auc_score(pat_arr[valid_oof], pat_oof_probs[valid_oof]))
        except Exception:
            pat_mean = float("nan")
        pat_std = 0.0
        try:
            pat_mean_ap = float(average_precision_score(pat_arr[valid_oof], pat_oof_probs[valid_oof]))
        except Exception:
            pat_mean_ap = float("nan")
    else:
        valid_pat = [a for a in pat_fold_aucs if not np.isnan(a)]
        valid_pat_aps = [a for a in pat_fold_aps if not np.isnan(a)]
        pat_mean = float(np.mean(valid_pat)) if valid_pat else float("nan")
        pat_std = float(np.std(valid_pat)) if valid_pat else float("nan")
        pat_mean_ap = float(np.mean(valid_pat_aps)) if valid_pat_aps else float("nan")

    pat_oof_preds = (pat_oof_probs[valid_oof] >= 0.5).astype(int)
    try:
        pat_bal_acc = float(balanced_accuracy_score(pat_arr[valid_oof], pat_oof_preds))
    except Exception:
        pat_bal_acc = float("nan")
    try:
        pat_mcc = float(matthews_corrcoef(pat_arr[valid_oof], pat_oof_preds))
    except Exception:
        pat_mcc = float("nan")

    cv_label = {"loocv": "LOOCV", "kfold": f"{CV_AUTO_FALLBACK_SPLITS}-fold CV"}.get(eff_mode, "3×3 CV")
    print(f"  {cv_label} patient AUC: {pat_mean:.4f}  BalAcc: {pat_bal_acc:.4f}  "
          f"MCC: {pat_mcc:.4f}  AUPRC: {pat_mean_ap:.4f} (base={prevalence:.3f})")

    per_ct_summaries["__patient_level__"] = {
        "fold_aucs": pat_fold_aucs if not is_loocv else [pat_mean],
        "mean_fold_auc": pat_mean,
        "std_fold_auc": pat_std,
        "fold_aps": pat_fold_aps if not is_loocv else [pat_mean_ap],
        "mean_fold_ap": pat_mean_ap,
        "balanced_accuracy": pat_bal_acc,
        "mcc": pat_mcc,
        "prevalence": prevalence,
        "cv_mode": eff_mode,
        "n_patients_yes": int(pat_arr.sum()),
        "n_patients_no": int(len(pat_arr) - pat_arr.sum()),
    }

    # Save patient-level OOF probs + labels for downstream bootstrap CI
    np.save(out_dir / "patient_oof_probs.npy", pat_oof_probs.astype(np.float32))
    np.save(out_dir / "patient_oof_labels.npy", pat_arr.astype(np.int8))

    with open(out_dir / "irae_per_ct_summary.json", "w") as f:
        json.dump(per_ct_summaries, f, indent=2)

    print(f"\n  Per-CT irAE AUC summary:")
    for ct_name, s in per_ct_summaries.items():
        if ct_name == "__patient_level__":
            continue
        print(f"    {ct_name}: AUC={s['mean_fold_auc']:.4f} ± {s['std_fold_auc']:.4f}")
    print(f"  Patient-level AUC: {pat_mean:.4f} ± {pat_std:.4f}")

    print(f"\nDone → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--h5ad", default=None)
    ap.add_argument("--prior", default=None)
    ap.add_argument("--prior-genes", action="store_true")
    import os as _os
    ap.add_argument("--results-dir", default=_os.environ.get(
        "IRAEGIS_BASELINES_DIR", str(REPO_ROOT / "results" / "baselines")))
    args = ap.parse_args()
    prof_start("cell_lr", args.cohort)
    run(args)
    prof_stop()


if __name__ == "__main__":
    main()
