#!/usr/bin/env python3
"""How variable is AUC itself at these cohort sizes?

Reviewer 2 argued that competing-method AUCs of 0.2-0.4 indicate a broken
setup. This tests the prior question: what range of AUC values arises purely by
chance when a cohort is this small?

The predictions are held FIXED and the patient outcome labels are shuffled many
times, recomputing AUC each time. That gives the distribution of AUC expected
when prediction and outcome have no relationship whatsoever - the null against
which any observed value should be read.

Unlike a refit permutation test (which asks what the whole pipeline produces on
noise), this isolates the sampling variability of the ranking statistic itself.
It depends only on the number of positive and negative patients, so it applies
equally to every method evaluated on that cohort.

Outputs results/auc_null_small_n.json

Usage:
    python scripts/auc_null_at_small_n.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from utils.config import RESULTS_IRAEGIS

N_PERM = 50000
COHORTS = [("GSE189125_pre_ici", "GSE189125"),
           ("GSE216329_integrated_pre_ici", "GSE216329"),
           ("GSE249898_integrated_pre_ici", "GSE249898"),
           ("GSE285888_pre_ici", "GSE285888")]
EXT = pd.read_csv("/Users/aishu/scPIP/results/bootstrap_ci_baselines.csv")

out = {}
for coh, short in COHORTS:
    f = RESULTS_IRAEGIS / coh / "hallmark_baseline" / "per_fold.csv"
    d = pd.read_csv(f); y = d.label.values.astype(int); p = d.oof_prob.values
    n1, n0 = int(y.sum()), int((y == 0).sum())
    rng = np.random.default_rng(0)
    null = np.array([roc_auc_score(rng.permutation(y), p) for _ in range(N_PERM)])
    # analytic Mann-Whitney null for comparison
    sd_theory = np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0))
    q = np.percentile(null, [2.5, 5, 50, 95, 97.5])
    print(f"\n  ══ {short}:  {n1} irAE / {n0} non-irAE  ══")
    print(f"    null mean {null.mean():.3f}   sd {null.std():.3f} "
          f"(theory {sd_theory:.3f})")
    print(f"    95% of chance AUCs fall in [{q[0]:.3f}, {q[4]:.3f}]")
    print(f"    P(AUC <= 0.30) = {(null<=0.30).mean():6.2%}    "
          f"P(AUC >= 0.70) = {(null>=0.70).mean():6.2%}")
    print(f"    P(AUC <= 0.20) = {(null<=0.20).mean():6.2%}    "
          f"P(AUC >= 0.80) = {(null>=0.80).mean():6.2%}")
    # where do the competing methods actually sit?
    g = EXT[EXT.cohort == coh].sort_values("auc_point")
    if len(g):
        inside = sum(1 for a in g.auc_point if q[0] <= a <= q[4])
        print(f"    competing methods inside the chance range: {inside}/{len(g)}"
              f"   (range {g.auc_point.min():.3f}-{g.auc_point.max():.3f})")
    out[short] = {"n_pos": n1, "n_neg": n0, "null_mean": float(null.mean()),
                  "null_sd": float(null.std()), "sd_theory": float(sd_theory),
                  "p2p5": float(q[0]), "p97p5": float(q[4]),
                  "P_le_0.30": float((null<=0.30).mean()),
                  "P_ge_0.70": float((null>=0.70).mean()),
                  "P_le_0.20": float((null<=0.20).mean()),
                  "P_ge_0.80": float((null>=0.80).mean()),
                  "n_competing_inside_chance_range": int(inside) if len(g) else None,
                  "n_competing": int(len(g))}
Path("results/auc_null_small_n.json").write_text(json.dumps(out, indent=2))
print(f"\n  -> results/auc_null_small_n.json   ({N_PERM:,} permutations per cohort)")
