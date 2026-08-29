#!/usr/bin/env python3
"""Leave-pair-out cross-validation of the fixed Hallmark pathway-score baseline.

Pooled LOOCV AUC is a biased estimator of the concordance probability in small
cohorts: each fold's training set has a different class composition from the
full cohort, and the n predictions it pools were produced by n *different*
models, so ranking them against one another is not the quantity AUC is meant to
estimate. Leave-pair-out removes that mismatch by construction. For every
(positive, negative) pair the pair is held out together, a single model is
trained on the remaining n-2 patients, and both members are scored by that one
model. The AUC is then exactly the fraction of pairs the model orders correctly

    AUC_LPO = mean over pairs of [ 1(s+ > s-) + 0.5 * 1(s+ == s-) ]

which is the definition of the concordance probability, estimated without
pooling scores across models and without any class-composition shift.

Unlike RLOOCV this involves no arbitrary choice of which patient to discard, so
it is deterministic. It costs n_pos * n_neg fits rather than n, which is why it
is applied to the pathway-score baselines (no autoencoder to retrain per fold)
rather than to the full model.

Cell-type grouping is recomputed per pair from the remaining patients only, so
the held-out pair influences nothing.

Outputs to results/iraegis/<cohort>/hallmark_baseline_lpo/:
    per_pair.csv   one row per (positive, negative) pair with both scores
    summary.json   LPO AUC with a pair-level bootstrap CI

Usage:
    python scripts/run_lpo_baseline.py --cohort GSE216329_integrated_pre_ici
    python scripts/run_lpo_baseline.py --all
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.config import RESULTS_IRAEGIS, COHORTS_REAL
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.train_utils import train_h_concat_gated_concat_en
from models.iraegis.fold_selection import select_ct_groups
from scripts.run_hallmark_baseline import hallmark_scores, SPLIT_CT_GROUPS, SHARED_GENES

N_BOOTSTRAP = 1000


def _pair_bootstrap_ci(conc: np.ndarray, pos_of: np.ndarray, neg_of: np.ndarray,
                       n_pos: int, n_neg: int, n: int = N_BOOTSTRAP, seed: int = 0):
    """Bootstrap the LPO AUC by resampling PATIENTS (not pairs).

    Pairs sharing a patient are dependent, so resampling pairs directly would
    understate the variance. Resampling patients and then averaging the
    concordances of the induced pairs respects that dependence.
    """
    rng = np.random.default_rng(seed)
    lut = {}
    for c, p, q in zip(conc, pos_of, neg_of):
        lut[(int(p), int(q))] = float(c)
    vals = []
    for _ in range(n):
        ps = rng.choice(n_pos, n_pos, replace=True)
        ns = rng.choice(n_neg, n_neg, replace=True)
        vals.append(np.mean([lut[(int(p), int(q))] for p in ps for q in ns]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_cohort(cohort: str) -> dict:
    print(f"\n{'='*70}\n  Leave-pair-out — Hallmark baseline: {cohort}\n{'='*70}")

    X, obs, _gn, _cg, _ci, pat_ids, pat_labels, prior = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)

    patients = sorted(pat_labels.keys())
    y = np.array([pat_labels[p] for p in patients], dtype=np.int64)
    S = hallmark_scores(X, prior["mask"]); del X

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pairs = [(int(a), int(b)) for a in pos_idx for b in neg_idx]
    print(f"  {len(patients)} patients ({len(pos_idx)} pos, {len(neg_idx)} neg) "
          f"-> {len(pairs)} pairs, {S.shape[1]} pathways")

    out = RESULTS_IRAEGIS / cohort / "hallmark_baseline_lpo"
    out.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.time()
    for k, (ip, ineg) in enumerate(pairs):
        held = {patients[ip], patients[ineg]}
        train_cells = ~np.isin(pat_ids, list(held))

        groups_f, ct_all, keep = select_ct_groups(obs, train_cells, SPLIT_CT_GROUPS)
        res = train_h_concat_gated_concat_en(
            S[keep], pat_ids[keep], ct_all[keep], pat_labels, groups_f,
            verbose=False, holdout_idx=[ip, ineg])

        sp, sn = res["oof_probs"][ip], res["oof_probs"][ineg]
        conc = 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
        rows.append({"pos_patient": patients[ip], "neg_patient": patients[ineg],
                     "pos_score": float(sp), "neg_score": float(sn),
                     "concordant": conc, "n_ct": len(groups_f),
                     "pos_i": ip, "neg_i": ineg})
        if (k + 1) % 10 == 0 or k == len(pairs) - 1:
            done = np.mean([r["concordant"] for r in rows])
            el = time.time() - t0
            print(f"  [{k+1}/{len(pairs)}] running LPO AUC = {done:.4f}   "
                  f"({el:.0f}s, {el/(k+1):.2f}s/pair, "
                  f"eta {el/(k+1)*(len(pairs)-k-1)/60:.1f} min)")
            pd.DataFrame(rows).to_csv(out / "per_pair.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_pair.csv", index=False)
    auc = float(df["concordant"].mean())

    pos_rank = {p: i for i, p in enumerate(pos_idx)}
    neg_rank = {p: i for i, p in enumerate(neg_idx)}
    lo, hi = _pair_bootstrap_ci(
        df["concordant"].values,
        np.array([pos_rank[i] for i in df["pos_i"]]),
        np.array([neg_rank[i] for i in df["neg_i"]]),
        len(pos_idx), len(neg_idx))

    summary = {
        "cohort": cohort,
        "validation": "leave-pair-out CV (each (pos, neg) pair held out together "
                      "and scored by one model trained on the remaining n-2)",
        "model": "fixed Hallmark pathway scores; no autoencoder",
        "n_patients": len(patients), "n_positive": int(len(pos_idx)),
        "n_pairs": len(pairs),
        "auc_lpo": auc, "auc_ci95": [lo, hi],
        "n_bootstrap": N_BOOTSTRAP,
        "total_seconds": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  LPO AUC = {auc:.4f}  95% CI [{lo:.3f}, {hi:.3f}]   -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort"); ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    cohorts = [a.cohort] if a.cohort else (list(COHORTS_REAL) if a.all
               else ap.error("pass --cohort or --all"))
    res = [run_cohort(c) for c in cohorts]
    if len(res) > 1:
        print(f"\n{'='*70}\n  SUMMARY — leave-pair-out\n{'='*70}")
        for s in res:
            ci = f"[{s['auc_ci95'][0]:.3f}, {s['auc_ci95'][1]:.3f}]"
            print(f"  {s['cohort']:<34} {s['auc_lpo']:>7.4f} {ci:>16}")


if __name__ == "__main__":
    main()
