#!/usr/bin/env python3
"""Strict leakage-free leave-one-patient-out evaluation of irAEGIS.

Every data-dependent step is performed inside the fold, using training-fold
patients only. The held-out patient's cells are used exactly once: encoded by
the frozen fold-specific autoencoder to produce that patient's prediction.

  step                          | derived from
  ------------------------------|---------------------------------------------
  cell QC (<200 genes)          | each cell independently        (not fold-dep.)
  normalise + log1p             | each cell independently        (not fold-dep.)
  Hallmark pathway prior        | external gene sets             (not fold-dep.)
  cell-type grouping            | TRAINING-FOLD patients only
  HVG backfill gene selection   | TRAINING-FOLD cells only
  pathway autoencoder           | TRAINING-FOLD cells only
  per-cell-type classifiers     | TRAINING-FOLD patients only
  cell-type gate (inner LOOCV)  | TRAINING-FOLD patients only
  patient-level stacker         | TRAINING-FOLD patients only

Because cell-type grouping and gene selection both depend on the data, the
surviving cell types and the gene set differ between folds. Each fold therefore
trains its own model with its own n_genes and n_ct. This is intentional: fixing
either across folds would require looking at the held-out patient.

Hyperparameters are fixed a priori and identical for every cohort and fold
(see models/iraegis/train_utils.py). No hyperparameter is selected using
held-out performance.

Outputs, under results/iraegis/<cohort>/loocv_leakage_free/:
    per_fold.csv   one row per fold: patient, label, prediction, fold geometry
    summary.json   AUC / AUPRC with bootstrap 95% CIs

Usage:
    python scripts/run_loocv_leakage_free.py --cohort GSE189125_pre_ici
    python scripts/run_loocv_leakage_free.py --all
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.config import DEVICE, RANDOM_STATE, RESULTS_IRAEGIS, COHORTS_REAL
from models.iraegis.model_utils import PathwayAE
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.train_utils import (
    train_ae, precompute_embeddings, train_h_concat_gated_concat_en,
    AE_LATENT_DIM, AE_DROPOUT, AE_N_EPOCHS,
)
from models.iraegis.fold_selection import plan_folds

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"
N_BOOTSTRAP = 1000


def _pin_seeds(seed: int) -> None:
    """Fix every RNG so folds differ only by their training data."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError):
            pass


def _free_mem() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


def _bootstrap_ci(y: np.ndarray, p: np.ndarray, metric,
                  n: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    """Stratified patient-level percentile bootstrap.

    Positives and negatives are resampled separately so a resample can never
    collapse to a single class, which would leave the metric undefined.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        vals.append(metric(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_cohort(cohort: str, ae_epochs: int = AE_N_EPOCHS,
               seed: int = RANDOM_STATE) -> dict:
    print(f"\n{'=' * 70}\n  Leakage-free LOOCV: {cohort}\n{'=' * 70}")

    # defer_selection=True returns the full shared gene space with raw cell-type
    # labels, so both selections can be redone per fold below.
    X, obs, _gene_names, _ct_groups, _ct_ids, pat_ids, pat_labels, prior = \
        load_cohort_data(
            cohort, prior_genes_only=True,
            gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
            split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)

    patients = sorted(pat_labels.keys())
    y_all = np.array([pat_labels[p] for p in patients], dtype=np.int64)
    print(f"  {len(X):,} cells x {X.shape[1]:,} shared genes, "
          f"{len(patients)} patients ({int(y_all.sum())} positive)")

    # Plan every fold's cell-type groups and gene set in one streaming pass,
    # then subset X once to their union so the full gene space is not held.
    print("  Planning per-fold selections ...")
    plan = plan_folds(X, obs, prior["mask"], pat_ids, patients,
                      SPLIT_CT_GROUPS, verbose=False)
    union = plan["union_genes"]
    X = np.ascontiguousarray(X[:, union])
    mask_union = prior["mask"][union, :]
    gene_names_union = [prior["gene_names"][i] for i in union]
    for f in plan["folds"].values():
        f["gene_idx"] = np.searchsorted(union, f["gene_idx"])

    n_pw = mask_union.shape[1]
    sizes = sorted({len(f["gene_idx"]) for f in plan["folds"].values()})
    ncts = sorted({len(f["ct_groups"]) for f in plan["folds"].values()})
    print(f"  genes per fold {sizes}, union {len(union):,} | "
          f"cell-type groups per fold {ncts}")

    out_dir = RESULTS_IRAEGIS / cohort / "loocv_leakage_free"
    out_dir.mkdir(parents=True, exist_ok=True)

    oof = np.full(len(patients), np.nan)
    rows: list[dict] = []
    t_start = time.time()

    for i, held_out in enumerate(patients):
        t0 = time.time()
        fold = plan["folds"][held_out]
        cells, genes = fold["cell_keep"], fold["gene_idx"]

        X_f = X[np.ix_(cells, genes)]
        ct_f = fold["ct_ids"][cells]
        pat_f = pat_ids[cells]
        obs_f = obs.loc[cells].reset_index(drop=True)
        ct_groups_f = fold["ct_groups"]
        mask_f = torch.tensor(mask_union[genes, :], dtype=torch.float32)

        is_train = pat_f != held_out

        # ---- autoencoder: training-fold cells only --------------------------
        _pin_seeds(seed)
        ae = PathwayAE(X_f.shape[1], n_pw, mask_f, AE_LATENT_DIM, AE_DROPOUT,
                       norm="ctbn", act="gelu", n_ct=len(ct_groups_f))
        train_ae(ae, X_f[is_train], n_epochs=ae_epochs,
                 ct_ids=ct_f[is_train], verbose=False)

        # ---- encode; the held-out patient enters only here ------------------
        with tempfile.TemporaryDirectory(prefix="loocv_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp),
                                         ct_ids=ct_f, verbose=False,
                                         suffix="_fold")

        # ---- classifiers, gate and stacker: training-fold patients only -----
        held_idx = patients.index(held_out)
        result = train_h_concat_gated_concat_en(
            h, pat_f, ct_f, pat_labels, ct_groups_f, verbose=False,
            only_patient_idx=held_idx)
        oof[held_idx] = result["oof_probs"][held_idx]

        dt = time.time() - t0
        rows.append({
            "patient": held_out, "label": int(y_all[held_idx]),
            "oof_prob": float(oof[held_idx]),
            "n_genes": int(len(genes)), "n_ct": len(ct_groups_f),
            "n_cells": int(cells.sum()), "n_train_cells": int(is_train.sum()),
            "ct_groups": json.dumps(list(ct_groups_f)), "seconds": dt,
        })
        pd.DataFrame(rows).to_csv(out_dir / "per_fold.csv", index=False)
        print(f"  [{i + 1}/{len(patients)}] {held_out} (label {y_all[held_idx]}): "
              f"{oof[held_idx]:.4f}   {len(genes):,} genes, "
              f"{len(ct_groups_f)} CTs   ({dt:.0f}s)")

        del ae, h, X_f
        _free_mem()

    auc = float(roc_auc_score(y_all, oof))
    auprc = float(average_precision_score(y_all, oof))
    auc_lo, auc_hi = _bootstrap_ci(y_all, oof, roc_auc_score)
    ap_lo, ap_hi = _bootstrap_ci(y_all, oof, average_precision_score)

    summary = {
        "cohort": cohort,
        "protocol": "leakage-free LOOCV (per-fold CT grouping, HVG selection, "
                    "autoencoder, classifiers, gate and stacker)",
        "n_patients": len(patients), "n_positive": int(y_all.sum()),
        "ae_epochs": ae_epochs, "seed": seed,
        "auc": auc, "auc_ci95": [auc_lo, auc_hi],
        "auprc": auprc, "auprc_ci95": [ap_lo, ap_hi],
        "n_bootstrap": N_BOOTSTRAP,
        "genes_per_fold": sizes, "ct_groups_per_fold": ncts,
        "total_seconds": time.time() - t_start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  AUC   = {auc:.4f}  95% CI [{auc_lo:.3f}, {auc_hi:.3f}]")
    print(f"  AUPRC = {auprc:.4f}  95% CI [{ap_lo:.3f}, {ap_hi:.3f}]")
    print(f"  -> {out_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", help="cohort id, e.g. GSE189125_pre_ici")
    ap.add_argument("--all", action="store_true", help="run all real cohorts")
    ap.add_argument("--ae-epochs", type=int, default=AE_N_EPOCHS)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = ap.parse_args()

    if args.cohort:
        cohorts = [args.cohort]
    elif args.all:
        cohorts = list(COHORTS_REAL)
    else:
        ap.error("pass --cohort <id> or --all")

    results = [run_cohort(c, args.ae_epochs, args.seed) for c in cohorts]

    if len(results) > 1:
        print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
        print(f"  {'cohort':<34} {'AUC':>7} {'95% CI':>16} {'AUPRC':>7}")
        for s in results:
            ci = f"[{s['auc_ci95'][0]:.3f}, {s['auc_ci95'][1]:.3f}]"
            print(f"  {s['cohort']:<34} {s['auc']:>7.4f} {ci:>16} {s['auprc']:>7.4f}")


if __name__ == "__main__":
    main()
