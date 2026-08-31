#!/usr/bin/env python3
"""Can a properly regularized concat combiner replace the gated stacker?

The gated stacker compresses each surviving cell type's 50 pathway values into a
single probability, so the patient-level model sees roughly three numbers. That
is efficient when a few cell types each carry signal, and destructive when the
signal is spread thinly. The alternative - concatenating every cell type's
pathway profile into one wide logistic regression - keeps all of it, but has
only ever been run at C=1.0 with ~550 features and ~23 training patients, which
is severely under-regularized. Its wins and losses so far may say more about
that than about the combiner itself.

This sweeps the concat combiner's L2 strength on a pre-registered grid and
compares it against the gated stacker on identical folds, identical frozen
autoencoders and identical top-25% aggregation, so the combiner is the only
thing that varies.

Decision rule, fixed before running: a regularized concat replaces the stacker
only if a SINGLE value of C matches or beats it on ALL FOUR cohorts. Winning on
average, or winning on the cohort that currently fails, is not sufficient -
that would be selecting the combiner on the test metric.

Outputs to results/iraegis/<cohort>/concat_regularization/:
    per_fold.csv   per-patient prediction for the stacker and each C
    summary.json   AUC and bootstrap CI per setting

Usage:
    python scripts/concat_regularization.py --cohort GSE189125_pre_ici
"""
from __future__ import annotations

import argparse, gc, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.config import RESULTS_IRAEGIS, RANDOM_STATE, DEVICE
from models.iraegis.model_utils import PathwayAE
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.train_utils import (
    precompute_embeddings, train_h_concat_gated_concat_en,
    AE_LATENT_DIM, AE_DROPOUT,
)
from models.iraegis.fold_selection import plan_folds

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"
C_GRID = [0.001, 0.01, 0.1, 1.0]           # pre-registered
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


def _patient_ct_matrix(h, pat_ids, ct_ids, patients, n_ct):
    """(patients, CT, pathways), top-25%-by-norm - the published aggregation."""
    P = h.shape[1]
    out = np.zeros((len(patients), n_ct, P), dtype=np.float32)
    for j in range(n_ct):
        for i, p in enumerate(patients):
            m = (pat_ids == p) & (ct_ids == j)
            if not m.any():
                continue
            c = h[m]
            if c.shape[0] >= 4:
                nm = np.linalg.norm(c, axis=1)
                top = c[nm >= np.percentile(nm, 75)]
                out[i, j] = top.mean(0) if len(top) else c.mean(0)
            else:
                out[i, j] = c.mean(0)
    return out


def run(cohort, src_dir):
    print(f"\n{'='*70}\n  Concat regularization sweep: {cohort}\n{'='*70}")
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
    settings = ["gate_stack"] + [f"concat_C{c:g}" for c in C_GRID]
    oof = {s: np.full(len(patients), np.nan) for s in settings}
    print(f"  {len(patients)} patients ({y.sum()} pos), C grid {C_GRID}")

    out_dir = RESULTS_IRAEGIS / cohort / "concat_regularization"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs, t0, n_feat = [], time.time(), None

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
        with tempfile.TemporaryDirectory(prefix="creg_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp), ct_ids=ct_f,
                                         verbose=False, suffix="_fold")

        hi = patients.index(held)
        tr = np.array([k for k in range(len(patients)) if k != hi])
        rec = {"patient": held, "label": int(y[hi])}

        r = train_h_concat_gated_concat_en(
            h, pat_f, ct_f, pat_labels, groups_f, verbose=False,
            only_patient_idx=hi)
        oof["gate_stack"][hi] = r["oof_probs"][hi]
        rec["gate_stack"] = float(oof["gate_stack"][hi])

        feat = _patient_ct_matrix(h, pat_f, ct_f, patients,
                                  len(groups_f)).reshape(len(patients), -1)
        n_feat = feat.shape[1]        # captured here; `feat` is freed below
        sc = StandardScaler().fit(feat[tr])
        Ztr, Zh = sc.transform(feat[tr]), sc.transform(feat[[hi]])
        for c in C_GRID:
            lr = LogisticRegression(max_iter=5000, class_weight="balanced",
                                    random_state=RANDOM_STATE, C=c)
            lr.fit(Ztr, y[tr])
            k = f"concat_C{c:g}"
            oof[k][hi] = float(lr.predict_proba(Zh)[0, 1])
            rec[k] = oof[k][hi]
        recs.append(rec)
        pd.DataFrame(recs).to_csv(out_dir / "per_fold.csv", index=False)
        print(f"  [{i+1}/{len(patients)}] {held}: " +
              "  ".join(f"{s.replace('concat_','')}={oof[s][hi]:.3f}" for s in settings))
        del ae, h, X_f, feat; gc.collect()

    summary = {"cohort": cohort, "C_grid": C_GRID, "n_features": int(n_feat or 0),
               "results": {}}
    print(f"\n  {'setting':<14} {'AUC':>8} {'95% CI':>18}")
    print("  " + "-" * 42)
    for s in settings:
        m = ~np.isnan(oof[s])
        if m.sum() < 3 or len(set(y[m])) < 2:
            continue
        a = float(roc_auc_score(y[m], oof[s][m]))
        lo, hi_ = _boot(y[m], oof[s][m])
        summary["results"][s] = {"auc": a, "auc_ci95": [lo, hi_], "n": int(m.sum())}
        tag = "  <- current model" if s == "gate_stack" else ""
        print(f"  {s:<14} {a:>8.4f} {f'[{lo:.3f}, {hi_:.3f}]':>18}{tag}")
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
