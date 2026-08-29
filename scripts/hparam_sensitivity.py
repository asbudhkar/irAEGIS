#!/usr/bin/env python3
"""Regularization and gate-threshold sensitivity for the patient-level classifier.

Reviewer 3 notes that L2 regularization is claimed but never varied, and asks
for a sensitivity analysis demonstrating that the framework is stable rather
than tuned to a lucky operating point. This sweeps the three hyperparameters
that govern the stacked patient-level classifier:

    C_inner    L2 strength of the per-cell-type classifiers   (default 0.1)
    C_outer    L2 strength of the patient-level stacker       (default 1.0)
    auc_gate   inner-LOOCV AUC a cell type must reach to be   (default 0.50)
               admitted to the stacker

The representation is held fixed: each fold reloads the autoencoder trained
without that fold's held-out patient, encodes once, and every hyperparameter
configuration is then scored from that same h. No autoencoder is retrained, so
differences are attributable to the downstream hyperparameters alone, and the
sweep costs a few minutes rather than days.

Note this sweep is diagnostic, not a tuning procedure. The published model uses
the a-priori defaults; selecting a configuration from this grid on the basis of
its LOOCV AUC would be exactly the selection-on-test-performance that Reviewer 3
is worried about. What the grid is meant to show is the SPREAD - if AUC is flat
across two orders of magnitude of C, the reported result does not depend on a
fortunate hyperparameter choice.

Outputs to results/iraegis/<cohort>/hparam_sensitivity/:
    grid.csv       one row per configuration with AUC / AUPRC
    per_config/    per-patient predictions for each configuration
    summary.json   spread statistics and the default's rank within the grid

Usage:
    python scripts/hparam_sensitivity.py --cohort GSE189125_pre_ici
"""
from __future__ import annotations

import argparse, gc, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.config import RESULTS_IRAEGIS, DEVICE
from models.iraegis.model_utils import PathwayAE
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.train_utils import (
    precompute_embeddings, train_h_concat_gated_concat_en,
    AE_LATENT_DIM, AE_DROPOUT,
)
from models.iraegis.fold_selection import plan_folds

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"

DEFAULT = {"C_inner": 0.1, "C_outer": 1.0, "auc_gate": 0.50}
C_GRID = [0.01, 0.1, 1.0, 10.0]
GATE_GRID = [0.45, 0.50, 0.55, 0.60]
N_BOOT = 1000


def _configs():
    """4x4 regularization grid at the default gate, plus a gate sweep."""
    seen, out = set(), []
    for ci in C_GRID:
        for co in C_GRID:
            k = (ci, co, DEFAULT["auc_gate"])
            if k not in seen:
                seen.add(k); out.append({"C_inner": ci, "C_outer": co,
                                         "auc_gate": DEFAULT["auc_gate"]})
    for g in GATE_GRID:
        k = (DEFAULT["C_inner"], DEFAULT["C_outer"], g)
        if k not in seen:
            seen.add(k); out.append({"C_inner": DEFAULT["C_inner"],
                                     "C_outer": DEFAULT["C_outer"], "auc_gate": g})
    return out


def _boot(y, p, metric, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        v.append(metric(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def run(cohort: str, src_dir: str) -> dict:
    print(f"\n{'='*74}\n  Hyperparameter sensitivity: {cohort}\n{'='*74}")
    ck_dir = RESULTS_IRAEGIS / cohort / src_dir / "checkpoints"
    if not ck_dir.exists():
        raise SystemExit(f"no checkpoints at {ck_dir}")

    X, obs, _gn, _cg, _ci, pat_ids, pat_labels, prior = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)
    patients = sorted(pat_labels.keys())
    y = np.array([pat_labels[p] for p in patients], dtype=np.int64)
    plan = plan_folds(X, obs, prior["mask"], pat_ids, patients,
                      SPLIT_CT_GROUPS, verbose=False)
    n_pw = prior["mask"].shape[1]

    cfgs = _configs()
    oof = {i: np.full(len(patients), np.nan) for i in range(len(cfgs))}
    print(f"  {len(patients)} patients ({y.sum()} pos), {len(cfgs)} configurations, "
          f"{len(list(ck_dir.glob('fold_*.pt')))} checkpoints")

    out_dir = RESULTS_IRAEGIS / cohort / "hparam_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

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
        with tempfile.TemporaryDirectory(prefix="hps_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp),
                                         ct_ids=ct_f, verbose=False,
                                         suffix="_fold")
        hi = patients.index(held)
        for ci, cfg in enumerate(cfgs):
            r = train_h_concat_gated_concat_en(
                h, pat_f, ct_f, pat_labels, groups_f, verbose=False,
                only_patient_idx=hi, **cfg)
            oof[ci][hi] = r["oof_probs"][hi]
        el = time.time() - t0
        print(f"  [{i+1}/{len(patients)}] {held}: {len(cfgs)} configs "
              f"({el:.0f}s elapsed, eta {el/(i+1)*(len(patients)-i-1)/60:.1f} min)")
        del ae, h, X_f; gc.collect()

    rows = []
    for ci, cfg in enumerate(cfgs):
        m = ~np.isnan(oof[ci])
        if m.sum() < 3 or len(set(y[m])) < 2:
            continue
        auc = float(roc_auc_score(y[m], oof[ci][m]))
        lo, hi_ = _boot(y[m], oof[ci][m], roc_auc_score)
        rows.append({**cfg, "auc": auc, "auc_lo": lo, "auc_hi": hi_,
                     "auprc": float(average_precision_score(y[m], oof[ci][m])),
                     "n_scored": int(m.sum()),
                     "is_default": cfg == DEFAULT})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "grid.csv", index=False)

    d = df[df.is_default].iloc[0] if df.is_default.any() else None
    reg = df[df.auc_gate == DEFAULT["auc_gate"]]
    gate = df[(df.C_inner == DEFAULT["C_inner"]) & (df.C_outer == DEFAULT["C_outer"])]

    print(f"\n  regularization grid (gate = {DEFAULT['auc_gate']}):")
    print(f"    {'C_inner':>9} {'C_outer':>9} {'AUC':>8}")
    for _, r in reg.sort_values(["C_inner", "C_outer"]).iterrows():
        star = "  <- default" if r.is_default else ""
        print(f"    {r.C_inner:>9g} {r.C_outer:>9g} {r.auc:>8.4f}{star}")
    print(f"\n  gate sweep (C_inner={DEFAULT['C_inner']}, C_outer={DEFAULT['C_outer']}):")
    for _, r in gate.sort_values("auc_gate").iterrows():
        star = "  <- default" if r.is_default else ""
        print(f"    gate={r.auc_gate:<5g} AUC={r.auc:.4f}{star}")

    summary = {
        "cohort": cohort, "n_configs": len(df),
        "default": DEFAULT,
        "default_auc": float(d.auc) if d is not None else None,
        "auc_min": float(df.auc.min()), "auc_max": float(df.auc.max()),
        "auc_median": float(df.auc.median()), "auc_iqr":
            [float(df.auc.quantile(.25)), float(df.auc.quantile(.75))],
        "spread": float(df.auc.max() - df.auc.min()),
        "default_rank": int((df.auc > (d.auc if d is not None else -1)).sum()) + 1,
        "reg_grid_spread": float(reg.auc.max() - reg.auc.min()),
        "gate_spread": float(gate.auc.max() - gate.auc.min()),
        "total_seconds": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  AUC across all {len(df)} configs: "
          f"min {summary['auc_min']:.4f}, median {summary['auc_median']:.4f}, "
          f"max {summary['auc_max']:.4f}  (spread {summary['spread']:.4f})")
    if d is not None:
        print(f"  default ({DEFAULT}) = {d.auc:.4f}, "
              f"rank {summary['default_rank']}/{len(df)}")
    print(f"  -> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--src-dir", default="ae_per_fold_deterministic_foldsel")
    a = ap.parse_args()
    run(a.cohort, a.src_dir)


if __name__ == "__main__":
    main()
