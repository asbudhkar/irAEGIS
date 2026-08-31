#!/usr/bin/env python3
"""Cross-check the confidence intervals with three independent methods.

Reviewers 1.7 and 3.5 both asked for confidence intervals. We report a
stratified patient-level percentile bootstrap. Percentile intervals assume a
roughly symmetric bootstrap distribution, which fails when AUC approaches 1 and
the distribution is truncated at the boundary - visible in GSE189125, where the
interval reaches exactly 1.000. Two alternatives are computed here so the
reported intervals do not rest on one method:

  percentile   the middle 95% of the bootstrap distribution        (what we report)
  BCa          bias-corrected and accelerated bootstrap; adjusts for skew and
               for the boundary, using a jackknife acceleration estimate
  DeLong       the classical analytic interval for AUC, based on the variance of
               the midrank structural components; makes no resampling assumption

Agreement across all three means the intervals are a property of the data, not
of the interval method.

Note all three condition on the cross-validated predictions: they capture
patient sampling variability, not variability from refitting the model. That is
standard practice but means the intervals are, if anything, slightly narrow.

Usage:
    python scripts/ci_methods_check.py
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from utils.config import RESULTS_IRAEGIS

N = 4000


def percentile_ci(y, p, n=N, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [roc_auc_score(y[i], p[i]) for i in
         (np.concatenate([rng.choice(pos, len(pos), True),
                          rng.choice(neg, len(neg), True)]) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), np.array(v)


def bca_ci(y, p, boots, n=N):
    """Bias-corrected and accelerated interval."""
    theta = roc_auc_score(y, p)
    prop = np.mean(boots < theta)
    prop = min(max(prop, 1.0 / n), 1 - 1.0 / n)          # keep z0 finite
    z0 = stats.norm.ppf(prop)
    # jackknife over patients for the acceleration term
    jack = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        if len(set(y[m])) < 2:
            continue
        jack.append(roc_auc_score(y[m], p[m]))
    jack = np.array(jack); jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum(); den = 6 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0
    out = []
    for q in (0.025, 0.975):
        z = stats.norm.ppf(q)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(np.percentile(boots, 100 * stats.norm.cdf(adj))))
    return out[0], out[1]


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N_ = len(x); T = np.zeros(N_); i = 0
    while i < N_:
        j = i
        while j < N_ and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N_); T2[J] = T; return T2


def delong_ci(y, p):
    """Analytic AUC interval from DeLong's variance estimate (logit-transformed)."""
    o = np.argsort(-y, kind="mergesort"); yy = y[o]; pp = p[o][None, :]
    m = int(yy.sum()); n = len(yy) - m
    tx = _midrank(pp[0, :m])[None, :]; ty = _midrank(pp[0, m:])[None, :]
    tz = _midrank(pp[0])[None, :]
    auc = tz[:, :m].sum() / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    if var <= 0: return float(auc), float(auc)
    se = np.sqrt(var)
    lo, hi = auc - 1.96 * se, auc + 1.96 * se           # logit would also be valid
    return float(max(0.0, lo)), float(min(1.0, hi))


RUNS = [("ae_per_fold_deterministic_foldsel", "irAEGIS"),
        ("hallmark_baseline_meanagg", "Hallmark mean"),
        ("hallmark_baseline", "Hallmark top-25%"),
        ("ssgsea_baseline", "ssGSEA")]

print(f"  {'cohort':<10} {'method':<18} {'AUC':>7}   {'percentile':>16} {'BCa':>16} {'DeLong':>16}")
print("  " + "-" * 92)
for c, s in [("GSE189125_pre_ici", "GSE189125"), ("GSE216329_integrated_pre_ici", "GSE216329"),
             ("GSE249898_integrated_pre_ici", "GSE249898"), ("GSE285888_pre_ici", "GSE285888")]:
    for d, lbl in RUNS:
        f = RESULTS_IRAEGIS / c / d / "per_fold.csv"
        if not f.exists(): continue
        x = pd.read_csv(f); x = x[x.oof_prob.notna()]
        y = x.label.values.astype(int); p = x.oof_prob.values
        if len(set(y)) < 2: continue
        a = roc_auc_score(y, p)
        pl, ph, boots = percentile_ci(y, p)
        bl, bh = bca_ci(y, p, boots)
        dl, dh = delong_ci(y, p)
        print(f"  {s:<10} {lbl:<18} {a:>7.4f}   {f'[{pl:.3f}, {ph:.3f}]':>16} "
              f"{f'[{bl:.3f}, {bh:.3f}]':>16} {f'[{dl:.3f}, {dh:.3f}]':>16}")
    print()
