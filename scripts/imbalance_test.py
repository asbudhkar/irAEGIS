#!/usr/bin/env python3
"""Does class imbalance explain irAEGIS's failure on the minority-positive cohorts?

Across the four cohorts, irAEGIS AUC correlates with the fraction of irAE-positive
patients (r=+0.93). But positive fraction is itself correlated with cohort size
(r=-0.89) at n=4, so the observational comparison cannot separate the two.

This breaks the confound experimentally. The autoencoder is label-blind, so the
frozen per-fold checkpoints of a cohort where irAEGIS works (GSE216329, AUC
0.8296, 62% positive) can be reused unchanged while ONLY the patient set given to
the downstream classifier varies:

    imbalanced   drop positives at random until the positive fraction matches the
                 failing cohorts (~0.34), keeping all negatives
    control      drop patients at random down to the SAME total n while keeping
                 the original positive fraction

Comparing the two arms isolates imbalance from sample size: both have identical
n, identical representation, identical folds. If AUC collapses in the imbalanced
arm but holds in the control, imbalance is the cause. If both fall equally, the
loss is a sample-size effect and imbalance is not implicated.

Note this measures the DOWNSTREAM's sensitivity to class balance. The
autoencoder is held fixed by design - it is unsupervised, so its training is
unaffected by which labels are present.

Usage:
    python scripts/imbalance_test.py --cohort GSE216329_integrated_pre_ici --draws 20
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
TARGET_POS_FRAC = 0.34          # the failing cohorts' positive fraction


def run(cohort, src_dir, draws, seed=0):
    print(f"\n{'='*72}\n  Class-imbalance test: {cohort}\n{'='*72}")
    ck_dir = RESULTS_IRAEGIS / cohort / src_dir / "checkpoints"
    X, obs, _g, _cg, _ci, pat_ids, pat_labels, prior = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)
    patients = sorted(pat_labels.keys())
    y = np.array([pat_labels[p] for p in patients])
    plan = plan_folds(X, obs, prior["mask"], pat_ids, patients, SPLIT_CT_GROUPS,
                      verbose=False)
    n_pw = prior["mask"].shape[1]

    pos = [p for p in patients if pat_labels[p] == 1]
    neg = [p for p in patients if pat_labels[p] == 0]
    # imbalanced arm: keep all negatives, keep k positives so k/(k+|neg|) ~= target
    k_pos = max(3, int(round(TARGET_POS_FRAC * len(neg) / (1 - TARGET_POS_FRAC))))
    n_keep = k_pos + len(neg)
    # control arm: same total n, original positive fraction
    c_pos = max(3, min(len(pos), int(round(n_keep * len(pos) / len(patients)))))
    c_neg = n_keep - c_pos
    print(f"  full cohort : {len(patients)} patients, {len(pos)} pos / {len(neg)} neg "
          f"(pos frac {len(pos)/len(patients):.2f})")
    print(f"  imbalanced  : {n_keep} patients, {k_pos} pos / {len(neg)} neg "
          f"(pos frac {k_pos/n_keep:.2f})")
    print(f"  control     : {n_keep} patients, {c_pos} pos / {c_neg} neg "
          f"(pos frac {c_pos/n_keep:.2f})\n")

    # encode every fold ONCE and cache; h does not depend on the patient subset
    print("  caching per-fold embeddings ...")
    cache, t0 = {}, time.time()
    for i, held in enumerate(patients):
        ck = ck_dir / f"fold_{held}.pt"
        if not ck.exists():
            continue
        fs = plan["folds"][held]
        cells, genes = fs["cell_keep"], fs["gene_idx"]
        X_f = X[np.ix_(cells, genes)]
        ct_f, pat_f = fs["ct_ids"][cells], pat_ids[cells]
        obs_f = obs.loc[cells].reset_index(drop=True)
        mask_f = torch.tensor(prior["mask"][genes, :], dtype=torch.float32)
        ae = PathwayAE(X_f.shape[1], n_pw, mask_f, AE_LATENT_DIM, AE_DROPOUT,
                       norm="ctbn", act="gelu", n_ct=len(fs["ct_groups"]))
        ae.attach_ct_head(len(fs["ct_groups"]))
        ae.load_state_dict(torch.load(ck, map_location="cpu")); ae.to(DEVICE); ae.eval()
        with tempfile.TemporaryDirectory(prefix="imb_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp), ct_ids=ct_f,
                                         verbose=False, suffix="_fold")
        cache[held] = (h.astype(np.float32), pat_f, ct_f, list(fs["ct_groups"]))
        del ae, X_f; gc.collect()
        if (i+1) % 6 == 0:
            print(f"    {i+1}/{len(patients)} ({time.time()-t0:.0f}s)")

    def score(keep):
        """LOOCV AUC over `keep` only, reusing each fold's frozen AE."""
        keep = list(keep)
        lab = {p: pat_labels[p] for p in keep}
        oof = {}
        for held in keep:
            if held not in cache: continue
            h, pat_f, ct_f, groups = cache[held]
            m = np.isin(pat_f, keep)
            r = train_h_concat_gated_concat_en(
                h[m], pat_f[m], ct_f[m], lab, groups, verbose=False,
                only_patient_idx=sorted(keep).index(held))
            oof[held] = r["oof_probs"][sorted(keep).index(held)]
        ks = [p for p in sorted(keep) if p in oof and not np.isnan(oof[p])]
        yy = np.array([lab[p] for p in ks])
        if len(set(yy)) < 2: return np.nan
        return roc_auc_score(yy, [oof[p] for p in ks])

    full = score(patients)
    print(f"\n  full cohort AUC (sanity, expect the published value): {full:.4f}\n")

    rng = np.random.default_rng(seed)
    res = {"imbalanced": [], "control": []}
    for d in range(draws):
        imb = list(rng.choice(pos, k_pos, replace=False)) + neg
        ctl = list(rng.choice(pos, c_pos, replace=False)) + \
              list(rng.choice(neg, c_neg, replace=False))
        res["imbalanced"].append(score(imb))
        res["control"].append(score(ctl))
        print(f"  draw {d+1:2d}/{draws}  imbalanced={res['imbalanced'][-1]:.4f}   "
              f"control={res['control'][-1]:.4f}")

    out = {"cohort": cohort, "full_auc": float(full), "draws": draws,
           "n_keep": n_keep, "imbalanced_pos": k_pos, "control_pos": c_pos,
           "imbalanced": [float(v) for v in res["imbalanced"]],
           "control": [float(v) for v in res["control"]]}
    a = np.array(res["imbalanced"]); b = np.array(res["control"])
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    print(f"\n  {'arm':<12} {'mean AUC':>9} {'sd':>7} {'min':>7} {'max':>7}")
    print("  "+"-"*46)
    print(f"  {'imbalanced':<12} {a.mean():>9.4f} {a.std():>7.4f} {a.min():>7.4f} {a.max():>7.4f}")
    print(f"  {'control':<12} {b.mean():>9.4f} {b.std():>7.4f} {b.min():>7.4f} {b.max():>7.4f}")
    from scipy import stats
    t = stats.mannwhitneyu(a, b)
    print(f"\n  imbalanced - control = {a.mean()-b.mean():+.4f}   Mann-Whitney p={t.pvalue:.4f}")
    out["imbalanced_mean"]=float(a.mean()); out["control_mean"]=float(b.mean())
    out["p_mannwhitney"]=float(t.pvalue)
    p = RESULTS_IRAEGIS / cohort / "imbalance_test.json"
    p.write_text(json.dumps(out, indent=2)); print(f"  -> {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--src-dir", default="ae_per_fold_deterministic_foldsel")
    ap.add_argument("--draws", type=int, default=20)
    a = ap.parse_args()
    run(a.cohort, a.src_dir, a.draws)


if __name__ == "__main__":
    main()
