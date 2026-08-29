#!/usr/bin/env python3
"""Robustness of the learned representation to test-time technical variation.

Reviewer 1 notes that robustness is presented as the primary motivation for the
denoising autoencoder but is never directly demonstrated, and asks for
experiments under varying dropout rates, sequencing depths, cell numbers, and
expression noise - compared against simpler pathway-based alternatives.

Design. The model is frozen and the perturbation is applied ONLY to the
held-out patient's cells, at prediction time. That is the clinically meaningful
question: a new patient arrives whose data is shallower, sparser, or noisier
than the training cohort - does the representation still place them correctly?
Perturbing the training data instead would confound representation robustness
with retraining effects.

Both representations are compared on byte-identical perturbed cells with the
same downstream classifier (gate + stacker), so any difference is attributable
to the representation:

    learned     per-fold pathway-masked denoising autoencoder (irAEGIS)
    hallmark    fixed Hallmark pathway scores, no learning

Perturbations are applied in approximate count space (expm1 of the
log-normalised matrix), then renormalised to 1e4 and re-log1p'd, mirroring the
real preprocessing so that a degraded cell is processed exactly as a genuinely
degraded cell would be.

    dropout     Bernoulli zeroing of captured transcripts (capture failure)
    depth       Poisson thinning of counts (shallower sequencing)
    cells       random subsampling of the patient's cells
    noise       multiplicative log-normal expression noise

Outputs to results/iraegis/<cohort>/robustness/:
    per_fold.csv   per-patient prediction under every perturbation and level
    curves.csv     AUC per (representation, perturbation, level)
    summary.json   degradation slopes and the learned-vs-fixed gap

Usage:
    python scripts/robustness_sweep.py --cohort GSE189125_pre_ici
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
from scripts.run_hallmark_baseline import hallmark_scores

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"
N_BOOT = 1000

PERTURBATIONS = {
    "dropout": [0.0, 0.1, 0.25, 0.50, 0.75],
    "depth":   [1.0, 0.5, 0.25, 0.10, 0.05],
    "cells":   [1.0, 0.5, 0.25, 0.10, 0.05],
    "noise":   [0.0, 0.25, 0.50, 1.0, 1.5],
}


def _renorm(counts):
    """Renormalise to 1e4 per cell and log1p, as the real pipeline does."""
    s = counts.sum(axis=1, keepdims=True) + 1e-9
    out = counts / s * 1e4
    return np.log1p(out, out=out)


def perturb(X_held, kind, level, rng):
    """Apply one perturbation to a patient's cells. Returns (X', keep_rows)."""
    if kind == "cells":
        if level >= 1.0:
            return X_held, np.arange(len(X_held))
        k = max(4, int(round(len(X_held) * level)))
        idx = rng.choice(len(X_held), min(k, len(X_held)), replace=False)
        return X_held[idx], idx

    if (kind == "dropout" and level <= 0) or (kind == "depth" and level >= 1.0) \
       or (kind == "noise" and level <= 0):
        return X_held, np.arange(len(X_held))

    c = np.expm1(X_held.astype(np.float64))     # back to approximate counts
    if kind == "dropout":
        c *= (rng.random(c.shape) >= level)     # captured transcripts lost
    elif kind == "depth":
        c = rng.poisson(c * level).astype(np.float64)   # shallower sequencing
    elif kind == "noise":
        c *= rng.lognormal(0.0, level, size=c.shape)
    return _renorm(c).astype(np.float32), np.arange(len(X_held))


def _boot(y, p, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def run(cohort: str, src_dir: str, reps: int = 1) -> dict:
    print(f"\n{'='*76}\n  Robustness sweep: {cohort}\n{'='*76}")
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

    jobs = [(k, lv) for k, lvs in PERTURBATIONS.items() for lv in lvs]
    print(f"  {len(patients)} patients ({y.sum()} pos), {len(jobs)} perturbation "
          f"settings x 2 representations x {reps} rep(s)")

    out_dir = RESULTS_IRAEGIS / cohort / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()

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
        mask_np = prior["mask"][genes, :]

        ae = PathwayAE(X_f.shape[1], n_pw, mask_f, AE_LATENT_DIM, AE_DROPOUT,
                       norm="ctbn", act="gelu", n_ct=len(groups_f))
        ae.attach_ct_head(len(groups_f))
        ae.load_state_dict(torch.load(ck, map_location="cpu"))
        ae.to(DEVICE); ae.eval()

        is_held = pat_f == held
        hi = patients.index(held)

        for kind, level in jobs:
            for rep in range(reps):
                rng = np.random.default_rng(hash((held, kind, rep)) % (2**32))
                Xh, _ = perturb(X_f[is_held], kind, level, rng)
                Xp = np.vstack([X_f[~is_held], Xh])
                ctp = np.concatenate([ct_f[~is_held], ct_f[is_held][:len(Xh)]])
                patp = np.concatenate([pat_f[~is_held], pat_f[is_held][:len(Xh)]])
                obsp = pd.concat([obs_f[~is_held],
                                  obs_f[is_held].iloc[:len(Xh)]]).reset_index(drop=True)

                with tempfile.TemporaryDirectory(prefix="rob_") as tmp:
                    h, _ = precompute_embeddings(ae, Xp, obsp, Path(tmp),
                                                 ct_ids=ctp, verbose=False,
                                                 suffix="_fold")
                r = train_h_concat_gated_concat_en(
                    h, patp, ctp, pat_labels, groups_f, verbose=False,
                    only_patient_idx=hi)
                rows.append({"patient": held, "label": int(y[hi]), "rep": rep,
                             "perturbation": kind, "level": level,
                             "representation": "learned",
                             "oof_prob": float(r["oof_probs"][hi])})

                S = hallmark_scores(Xp, mask_np)
                r2 = train_h_concat_gated_concat_en(
                    S, patp, ctp, pat_labels, groups_f, verbose=False,
                    only_patient_idx=hi)
                rows.append({"patient": held, "label": int(y[hi]), "rep": rep,
                             "perturbation": kind, "level": level,
                             "representation": "hallmark",
                             "oof_prob": float(r2["oof_probs"][hi])})
                del h, S

        pd.DataFrame(rows).to_csv(out_dir / "per_fold.csv", index=False)
        el = time.time() - t0
        print(f"  [{i+1}/{len(patients)}] {held}: {len(jobs)*2*reps} predictions "
              f"({el/60:.1f} min elapsed, eta {el/(i+1)*(len(patients)-i-1)/60:.0f} min)")
        del ae, X_f; gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_fold.csv", index=False)

    curves = []
    for (rep_, kind, level), g in df.groupby(["representation", "perturbation", "level"]):
        m = g.groupby("patient").agg(label=("label", "first"),
                                     p=("oof_prob", "mean"))
        if m.label.nunique() < 2:
            continue
        auc = float(roc_auc_score(m.label, m.p))
        lo, hi_ = _boot(m.label.values, m.p.values)
        curves.append({"representation": rep_, "perturbation": kind,
                       "level": level, "auc": auc, "lo": lo, "hi": hi_,
                       "n": len(m)})
    cdf = pd.DataFrame(curves)
    cdf.to_csv(out_dir / "curves.csv", index=False)

    print(f"\n  {'perturbation':<12} {'level':>7} {'learned':>9} {'hallmark':>10} {'gap':>8}")
    print("  " + "-" * 52)
    for kind in PERTURBATIONS:
        for lv in PERTURBATIONS[kind]:
            a = cdf[(cdf.perturbation == kind) & (cdf.level == lv)]
            L = a[a.representation == "learned"].auc
            H = a[a.representation == "hallmark"].auc
            if L.empty or H.empty:
                continue
            print(f"  {kind:<12} {lv:>7g} {L.iloc[0]:>9.4f} {H.iloc[0]:>10.4f} "
                  f"{L.iloc[0]-H.iloc[0]:>+8.4f}")

    summary = {"cohort": cohort, "n_patients": int(df.patient.nunique()),
               "reps": reps, "perturbations": PERTURBATIONS,
               "curves": cdf.to_dict("records"),
               "total_seconds": time.time() - t0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  -> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--src-dir", default="ae_per_fold_deterministic_foldsel")
    ap.add_argument("--reps", type=int, default=1,
                    help="independent perturbation draws per patient per level")
    a = ap.parse_args()
    run(a.cohort, a.src_dir, a.reps)


if __name__ == "__main__":
    main()
