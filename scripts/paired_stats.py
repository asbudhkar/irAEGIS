#!/usr/bin/env python3
"""Paired statistical comparison and stability analysis for the ablations.

Reviewer 1 asks for paired statistical comparisons and stability analyses to
accompany the point estimates; Reviewer 3 raises the same concern from the
overfitting side. Every ablation is evaluated on the SAME patients under the
SAME folds as the full model, so the two AUCs are correlated and an unpaired
comparison would be wrong. Two paired tests are reported:

  DeLong        the standard analytic test for two correlated ROC curves. Fast
                and conventional, but it is an asymptotic normal approximation
                and these cohorts have 16-24 patients, so its p-values should be
                read as indicative rather than exact.

  paired
  bootstrap     resamples PATIENTS (stratified by class, so a resample can never
                lose a class), recomputes both AUCs on the same resample, and
                takes the distribution of the difference. This respects the
                pairing, makes no normality assumption, and is the estimate to
                quote. The reported p is the two-sided proportion of resamples
                in which the difference reverses sign, with the standard +1
                correction: p = 2 * (min(#<=0, #>=0) + 1) / (B + 1).

Because n is 16-24, differences smaller than roughly 0.15 AUC are not resolvable
at conventional significance. A non-significant result here means the study is
underpowered for that comparison, NOT that the component does nothing - the
effect size and its interval are the informative output, not the p-value.

Usage:
    python scripts/paired_stats.py --cohort GSE189125_pre_ici
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.config import RESULTS_IRAEGIS

N_BOOT = 10000

VARIANTS = {
    "": "full model",
    "a0p0": "- CT auxiliary loss",
    "b0p0": "- pathway decorrelation",
    "mf0p0": "- denoising",
    "nolatent": "- latent bottleneck + reconstruction",
    "shufmask": "- pathway prior (shuffled)",
}


# ---------------------------------------------------------------- DeLong ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2


def delong(y, p1, p2):
    """Two-sided DeLong test for two correlated ROC curves."""
    order = np.argsort(-y, kind="mergesort")     # positives first
    yy = y[order]
    preds = np.vstack([p1[order], p2[order]])
    m = int(yy.sum()); n = len(yy) - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = 2
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0] - aucs[1]), float("nan"), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0] - aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


# ------------------------------------------------------ paired bootstrap ----
def paired_boot(y, p1, p2, B=N_BOOT, seed=0):
    """Stratified patient-level paired bootstrap of the AUC difference."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        yy = y[idx]
        if yy.min() == yy.max():
            d[b] = np.nan; continue
        d[b] = roc_auc_score(yy, p1[idx]) - roc_auc_score(yy, p2[idx])
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    n_le = int((d <= 0).sum()); n_ge = int((d >= 0).sum())
    p = 2.0 * (min(n_le, n_ge) + 1) / (len(d) + 1)
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def load(cohort, tag):
    d = RESULTS_IRAEGIS / cohort / (
        "ae_per_fold_deterministic" + (f"_{tag}" if tag else "") + "_foldsel")
    f = d / "per_fold.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[df.oof_prob.notna()][["patient", "label", "oof_prob"]]
    return df.set_index("patient")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    a = ap.parse_args()

    runs = {t: load(a.cohort, t) for t in VARIANTS}
    runs = {t: v for t, v in runs.items() if v is not None}
    if "" not in runs:
        raise SystemExit("full model run not found")

    common = sorted(set.intersection(*[set(v.index) for v in runs.values()]))
    full = runs[""].loc[common]
    y = full.label.values.astype(int)
    print(f"\n{'='*78}\n  Paired ablation statistics: {a.cohort}\n{'='*78}")
    print(f"  {len(common)} patients common to all {len(runs)} runs "
          f"({y.sum()} positive); {a.bootstrap:,} bootstrap resamples\n")
    print(f"  full model AUC = {roc_auc_score(y, full.oof_prob.values):.4f}\n")
    print(f"  {'ablation':<36} {'AUC':>7} {'dAUC':>7} {'95% CI':>16} "
          f"{'p_boot':>8} {'p_DeLong':>9}")
    print("  " + "-" * 88)

    out = []
    for tag, name in VARIANTS.items():
        if tag == "" or tag not in runs:
            continue
        sub = runs[tag].loc[common]
        p1, p2 = full.oof_prob.values, sub.oof_prob.values
        auc = roc_auc_score(y, p2)
        dd, z, pdl = delong(y, p1, p2)
        db, lo, hi, pb = paired_boot(y, p1, p2, a.bootstrap)
        print(f"  {name:<36} {auc:>7.4f} {dd:>+7.4f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>16} {pb:>8.4f} "
              f"{(f'{pdl:.4f}' if np.isfinite(pdl) else 'n/a'):>9}")
        out.append({"ablation": name, "tag": tag, "auc": float(auc),
                    "delta_auc": dd, "boot_mean_delta": db,
                    "boot_ci95": [lo, hi], "p_paired_bootstrap": pb,
                    "p_delong": pdl if np.isfinite(pdl) else None})

    print(f"\n  dAUC > 0 means the full model is better than the ablation.")
    print(f"  At n={len(common)}, only differences of roughly 0.15+ AUC are")
    print(f"  resolvable; treat intervals as the result, not the p-values.")

    res = {"cohort": a.cohort, "n_patients": len(common),
           "n_positive": int(y.sum()),
           "full_auc": float(roc_auc_score(y, full.oof_prob.values)),
           "n_bootstrap": a.bootstrap, "comparisons": out}
    p = RESULTS_IRAEGIS / a.cohort / "paired_ablation_stats.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
