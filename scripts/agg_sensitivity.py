#!/usr/bin/env python3
"""Sensitivity of irAEGIS to the patient x cell-type aggregation fraction.

The published model summarises each (patient, cell type) by the mean of the
top 25% of its cells, ranked by the L2 norm of their pathway vector. That 25%
is a bare literal in the code: it has never been varied, and Reviewer 3 asks
for sensitivity analysis of "the regularization or other key hyperparameters".
The rule matters - switching it to a plain mean changes AUC by -0.06 to +0.23
depending on cohort - so its value deserves the same scrutiny as C_inner.

This sweeps the retained fraction over a pre-registered grid while holding
everything else fixed. Each fold reloads its own autoencoder checkpoint - the
one trained without that fold's held-out patient - so the representation, the
folds, the gate and the stacker are identical across settings and only the
aggregation differs.

    top_frac = 0.10   keep the most extreme tenth
    top_frac = 0.25   the published rule
    top_frac = 0.50
    top_frac = 0.75
    top_frac = 1.00   keep every cell, i.e. a plain mean

This is diagnostic, not a tuning procedure. Reporting the full curve for every
cohort is the point; picking the fraction that maximises AUC would be selection
on the test metric.

Outputs to results/iraegis/<cohort>/agg_sensitivity/:
    per_fold.csv   per-patient prediction at each fraction
    summary.json   AUC and bootstrap CI per fraction

Usage:
    python scripts/agg_sensitivity.py --cohort GSE249898_integrated_pre_ici
"""
from __future__ import annotations

import argparse, gc, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

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
FRACS = [0.10, 0.25, 0.50, 0.75, 1.00]      # pre-registered; 0.25 is published
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


def run(cohort, src_dir):
    print(f"\n{'='*70}\n  Aggregation-fraction sensitivity: {cohort}\n{'='*70}")
    ck_dir = RESULTS_IRAEGIS / cohort / src_dir / "checkpoints"
    if not ck_dir.exists():
        raise SystemExit(f"no checkpoints at {ck_dir}")

    X, obs, _g, _cg, _ci, pat_ids, pat_labels, prior = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)
    patients = sorted(pat_labels.keys())
    y = np.array([pat_labels[p] for p in patients], dtype=np.int64)
    plan = plan_folds(X, obs, prior["mask"], pat_ids, patients, SPLIT_CT_GROUPS,
                      verbose=False)
    n_pw = prior["mask"].shape[1]
    print(f"  {len(patients)} patients ({y.sum()} pos), fractions {FRACS}")

    out_dir = RESULTS_IRAEGIS / cohort / "agg_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = {f: np.full(len(patients), np.nan) for f in FRACS}
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
        with tempfile.TemporaryDirectory(prefix="agg_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp), ct_ids=ct_f,
                                         verbose=False, suffix="_fold")
        hi = patients.index(held)
        rec = {"patient": held, "label": int(y[hi])}
        for f in FRACS:
            r = train_h_concat_gated_concat_en(
                h, pat_f, ct_f, pat_labels, groups_f, verbose=False,
                only_patient_idx=hi, top_frac=f)
            oof[f][hi] = r["oof_probs"][hi]
            rec[f"frac_{f:.2f}"] = float(oof[f][hi])
        recs.append(rec)
        pd.DataFrame(recs).to_csv(out_dir / "per_fold.csv", index=False)
        el = time.time() - t0
        print(f"  [{i+1}/{len(patients)}] {held}: " +
              " ".join(f"{f:.2f}={oof[f][hi]:.3f}" for f in FRACS) +
              f"   ({el/60:.1f} min)")
        del ae, h, X_f; gc.collect()

    summary = {"cohort": cohort, "fracs": FRACS, "published_frac": 0.25,
               "results": {}}
    print(f"\n  {'top_frac':>9} {'AUC':>8} {'95% CI':>18}")
    print("  " + "-" * 38)
    for f in FRACS:
        m = ~np.isnan(oof[f])
        if m.sum() < 3 or len(set(y[m])) < 2:
            continue
        a = float(roc_auc_score(y[m], oof[f][m]))
        lo, hi_ = _boot(y[m], oof[f][m])
        summary["results"][f"{f:.2f}"] = {"auc": a, "auc_ci95": [lo, hi_],
                                          "n": int(m.sum())}
        star = "  <- published" if f == 0.25 else ("  (= plain mean)" if f == 1.0 else "")
        print(f"  {f:>9.2f} {a:>8.4f} {f'[{lo:.3f}, {hi_:.3f}]':>18}{star}")
    summary["total_seconds"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
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
