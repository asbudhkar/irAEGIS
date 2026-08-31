#!/usr/bin/env python3
"""Is there an aggregation rule that works on ALL cohorts?

The published model summarises each (patient, cell type) by the mean of the top
25% of cells ranked by pathway-vector L2 norm. That rule costs 0.234 AUC on
GSE249898 while helping the other cohorts, so it is worth asking whether a
different summary is uniformly better. Five rules are compared, holding the
frozen per-fold autoencoders, the folds, the gate and the stacker fixed:

    top_norm   mean of the top 25% by L2 norm            (published)
    random     mean of a RANDOM 25% of the same size     (control)
    mean       mean of every cell
    median     per-pathway median of every cell
    trimmed    mean after dropping the top and bottom decile by norm

`random` is the key control. top_norm does two things at once - it sub-samples,
and it sub-samples by magnitude. Comparing against a random subset of identical
size separates them:

    random ~= mean      -> the NORM-RANKING is what matters
    random ~= top_norm  -> the SUB-SAMPLING is what matters, regardless of rule

Decision rule fixed in advance: a rule replaces top_norm only if it matches or
beats it on ALL FOUR cohorts. Winning on average, or rescuing the cohort that
currently fails, is not sufficient.

Outputs to results/iraegis/<cohort>/agg_alternatives/:
    per_fold.csv   per-patient prediction under each rule
    summary.json   AUC and bootstrap CI per rule

Usage:
    python scripts/agg_alternatives.py --cohort GSE249898_integrated_pre_ici
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
MODES = ["top_norm", "top_z", "top_centroid", "random", "mean"]
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
    print(f"\n{'='*70}\n  Aggregation alternatives: {cohort}\n{'='*70}")
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
    oof = {m: np.full(len(patients), np.nan) for m in MODES}
    print(f"  {len(patients)} patients ({y.sum()} pos), rules: {MODES}")

    out_dir = RESULTS_IRAEGIS / cohort / "agg_alternatives"
    out_dir.mkdir(parents=True, exist_ok=True)
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
        with tempfile.TemporaryDirectory(prefix="aggalt_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp), ct_ids=ct_f,
                                         verbose=False, suffix="_fold")
        hi = patients.index(held)
        rec = {"patient": held, "label": int(y[hi])}
        for m in MODES:
            r = train_h_concat_gated_concat_en(
                h, pat_f, ct_f, pat_labels, groups_f, verbose=False,
                only_patient_idx=hi, agg_mode=m)
            oof[m][hi] = r["oof_probs"][hi]
            rec[m] = float(oof[m][hi])
        recs.append(rec)
        pd.DataFrame(recs).to_csv(out_dir / "per_fold.csv", index=False)
        print(f"  [{i+1}/{len(patients)}] {held}: " +
              "  ".join(f"{m}={oof[m][hi]:.3f}" for m in MODES) +
              f"   ({(time.time()-t0)/60:.1f} min)")
        del ae, h, X_f; gc.collect()

    summary = {"cohort": cohort, "modes": MODES, "results": {}}
    print(f"\n  {'rule':<10} {'AUC':>8} {'95% CI':>18}")
    print("  " + "-" * 38)
    for m in MODES:
        k = ~np.isnan(oof[m])
        if k.sum() < 3 or len(set(y[k])) < 2:
            continue
        a = float(roc_auc_score(y[k], oof[m][k]))
        lo, hi_ = _boot(y[k], oof[m][k])
        summary["results"][m] = {"auc": a, "auc_ci95": [lo, hi_], "n": int(k.sum())}
        tag = "  <- published" if m == "top_norm" else ""
        print(f"  {m:<10} {a:>8.4f} {f'[{lo:.3f}, {hi_:.3f}]':>18}{tag}")
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
