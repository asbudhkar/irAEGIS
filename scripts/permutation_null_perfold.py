#!/usr/bin/env python3
"""Leakage-free per-fold permutation null for the full irAEGIS pipeline.

An earlier permutation test used fixed pathway scores. This one runs the whole
leakage-free per-fold pipeline under permuted labels, so the null reflects the
actual model rather than a simplified stand-in.

Reusing the frozen per-fold autoencoders is valid here, and worth stating why:
the autoencoder is UNSUPERVISED, and the per-fold HVG selection and cell-type
grouping depend only on which patients are in the training fold - never on their
labels. Permuting labels therefore cannot change any of them. Everything that
DOES depend on labels - the cell-type gate, the per-cell-type classifiers and
the patient-level stacker - is refit from scratch under each permutation, inside
each fold, with the held-out patient excluded exactly as in the real run.

So each permutation is a complete leakage-free LOOCV of the real architecture,
and the resulting null answers: what AUC does this pipeline produce on this
cohort when the labels carry no information?

Outputs to results/iraegis/<cohort>/permutation_null_perfold.json

Usage:
    python scripts/permutation_null_perfold.py --cohort GSE249898_integrated_pre_ici --perms 200
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


def run(cohort, src_dir, perms, seed=0):
    print(f"\n{'='*70}\n  Leakage-free permutation null: {cohort}\n{'='*70}")
    ck_dir = RESULTS_IRAEGIS / cohort / src_dir / "checkpoints"
    X, obs, _g, _cg, _ci, pat_ids, pat_labels, prior = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)
    patients = sorted(pat_labels.keys())
    y = np.array([pat_labels[p] for p in patients], dtype=np.int64)
    plan = plan_folds(X, obs, prior["mask"], pat_ids, patients, SPLIT_CT_GROUPS,
                      verbose=False)
    n_pw = prior["mask"].shape[1]

    print(f"  caching per-fold embeddings (label-blind, so valid under permutation) ...")
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
        ae.load_state_dict(torch.load(ck, map_location="cpu"))
        ae.to(DEVICE); ae.eval()
        with tempfile.TemporaryDirectory(prefix="perm_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp), ct_ids=ct_f,
                                         verbose=False, suffix="_fold")
        cache[held] = (h.astype(np.float32), pat_f, ct_f, list(fs["ct_groups"]))
        del ae, X_f; gc.collect()
        if (i + 1) % 8 == 0:
            print(f"    {i+1}/{len(patients)}  ({time.time()-t0:.0f}s)")
    print(f"  cached {len(cache)} folds in {(time.time()-t0)/60:.1f} min")

    def loocv(labels):
        """One complete leakage-free LOOCV under the given patient labels."""
        lab = {p: int(v) for p, v in zip(patients, labels)}
        oof = np.full(len(patients), np.nan)
        for hi, held in enumerate(patients):
            if held not in cache:
                continue
            h, pat_f, ct_f, groups = cache[held]
            r = train_h_concat_gated_concat_en(
                h, pat_f, ct_f, lab, groups, verbose=False, only_patient_idx=hi)
            oof[hi] = r["oof_probs"][hi]
        m = ~np.isnan(oof)
        if m.sum() < 3 or len(set(np.asarray(labels)[m])) < 2:
            return np.nan
        return roc_auc_score(np.asarray(labels)[m], oof[m])

    obs_auc = loocv(y)
    print(f"\n  observed AUC (real labels): {obs_auc:.4f}")
    print(f"  running {perms} permutations ...")
    rng = np.random.default_rng(seed)
    null, t1 = [], time.time()
    for k in range(perms):
        null.append(loocv(rng.permutation(y)))
        if (k + 1) % 10 == 0:
            v = np.array([x for x in null if np.isfinite(x)])
            el = time.time() - t1
            print(f"    {k+1}/{perms}  null mean={v.mean():.4f}  "
                  f"({el/60:.1f} min, eta {el/(k+1)*(perms-k-1)/60:.0f} min)")
    v = np.array([x for x in null if np.isfinite(x)])
    n_le = int((v <= obs_auc).sum())
    res = {"cohort": cohort, "observed_auc": float(obs_auc), "n_perms": int(len(v)),
           "null_mean": float(v.mean()), "null_median": float(np.median(v)),
           "null_sd": float(v.std()),
           "null_p2p5": float(np.percentile(v, 2.5)),
           "null_p97p5": float(np.percentile(v, 97.5)),
           "n_le_observed": n_le,
           "p_one_sided_lower": float((n_le + 1) / (len(v) + 1)),
           "p_one_sided_upper": float((int((v >= obs_auc).sum()) + 1) / (len(v) + 1)),
           "frac_null_below_half": float((v < 0.5).mean()),
           "null_aucs": [float(x) for x in v]}
    p = RESULTS_IRAEGIS / cohort / "permutation_null_perfold.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n  observed            : {obs_auc:.4f}")
    print(f"  null mean / median  : {res['null_mean']:.4f} / {res['null_median']:.4f}")
    print(f"  null sd             : {res['null_sd']:.4f}")
    print(f"  null 2.5 / 97.5 pct : {res['null_p2p5']:.4f} / {res['null_p97p5']:.4f}")
    print(f"  frac of null < 0.5  : {res['frac_null_below_half']:.0%}")
    print(f"  p (one-sided lower) : {res['p_one_sided_lower']:.4f}")
    print(f"  p (one-sided upper) : {res['p_one_sided_upper']:.4f}")
    print(f"  -> {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--src-dir", default="ae_per_fold_deterministic_foldsel")
    ap.add_argument("--perms", type=int, default=200)
    a = ap.parse_args()
    run(a.cohort, a.src_dir, a.perms)


if __name__ == "__main__":
    main()
