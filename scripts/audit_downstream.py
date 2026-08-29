#!/usr/bin/env python3
"""Is the gated/stacked patient-level classifier earning its place?

The Hallmark comparator showed that gating + stacking on a signal-poor
representation does not degrade gracefully to chance — it produces systematic
anti-ranking. irAEGIS uses the same downstream, so the question is whether that
component is fragile in general or only when the representation carries no
signal.

This holds the representation FIXED and varies only the downstream. For each
leave-one-patient-out fold it reloads that fold's autoencoder checkpoint — the
one trained without the held-out patient — encodes every cell with it, and then
scores the held-out patient three different ways from the identical h:

    gate_stack   per-CT classifiers -> inner-LOOCV gate -> stacked LR  (Phase 2c)
    concat_lr    all cell types concatenated -> one logistic regression
    mean_lr      pathway profile averaged over cell types -> one logistic regression

Because the autoencoders are reused rather than retrained, every variant sees
exactly the same learned representation and exactly the same fold geometry, so
any difference in AUC is attributable to the downstream alone.

Requires a completed --fold-selection run (its checkpoints/ directory).

Outputs to results/iraegis/<cohort>/downstream_audit/:
    per_fold.csv   per-patient prediction under each variant
    summary.json   AUC per variant with bootstrap 95% CIs

Usage:
    python scripts/audit_downstream.py --cohort GSE189125_pre_ici
"""
from __future__ import annotations

import argparse, gc, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.config import RESULTS_IRAEGIS, RANDOM_STATE, DEVICE
from models.iraegis.model_utils import PathwayAE
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.train_utils import (
    precompute_embeddings, train_h_concat_gated_concat_en,
    AE_LATENT_DIM, AE_DROPOUT,
)
from models.iraegis.fold_selection import plan_folds, select_ct_groups
from scripts.run_hallmark_baseline import hallmark_scores

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"
N_BOOTSTRAP = 1000


def _boot(y, p, metric, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        v.append(metric(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _patient_ct_matrix(h, pat_ids, ct_ids, patients, n_ct):
    """(patients, CT, pathways) using the model's top-25%-by-norm aggregation."""
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


def _plain_lr(feat, y, tr, held):
    """Fit one logistic regression on training patients, score the held-out one."""
    sc = StandardScaler().fit(feat[tr])
    lr = LogisticRegression(max_iter=5000, class_weight="balanced",
                            random_state=RANDOM_STATE, C=1.0)
    lr.fit(sc.transform(feat[tr]), y[tr])
    return float(lr.predict_proba(sc.transform(feat[[held]]))[0, 1])


def _score_fold(h, ct_f, pat_f, groups_f, held, i, patients, y, pat_labels,
                oof, rows, variants, out_dir):
    """Score one held-out patient three ways from one representation."""
    hi = patients.index(held)
    tr = np.array([k for k in range(len(patients)) if k != hi])

    res = train_h_concat_gated_concat_en(
        h, pat_f, ct_f, pat_labels, groups_f, verbose=False, only_patient_idx=hi)
    oof["gate_stack"][hi] = res["oof_probs"][hi]

    pat_h = _patient_ct_matrix(h, pat_f, ct_f, patients, len(groups_f))
    oof["concat_lr"][hi] = _plain_lr(pat_h.reshape(len(patients), -1), y, tr, hi)
    oof["mean_lr"][hi] = _plain_lr(pat_h.mean(axis=1), y, tr, hi)

    rows.append({"patient": held, "label": int(y[hi]), "n_ct": len(groups_f),
                 **{v: float(oof[v][hi]) for v in variants}})
    pd.DataFrame(rows).to_csv(out_dir / "per_fold.csv", index=False)
    print(f"  [{i+1}/{len(patients)}] {held} (label {y[hi]}): " +
          "  ".join(f"{v}={oof[v][hi]:.3f}" for v in variants))


def run_cohort(cohort: str, src_dir: str,
               representation: str = "learned") -> dict:
    print(f"\n{'='*70}\n  Downstream audit [{representation}]: {cohort}\n{'='*70}")
    ck_dir = RESULTS_IRAEGIS / cohort / src_dir / "checkpoints"
    if representation == "learned" and not ck_dir.exists():
        raise SystemExit(f"no checkpoints at {ck_dir}")

    X, obs, _gn, _cg, _ci, pat_ids, pat_labels, prior = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)

    patients = sorted(pat_labels.keys())
    y = np.array([pat_labels[p] for p in patients], dtype=np.int64)
    n_pw = prior["mask"].shape[1]
    if representation == "learned":
        # fold-wise HVG selection + CT grouping, then that fold's frozen AE
        plan = plan_folds(X, obs, prior["mask"], pat_ids, patients,
                          SPLIT_CT_GROUPS, verbose=False)
        print(f"  {len(patients)} patients ({y.sum()} pos), "
              f"{len(list(ck_dir.glob('fold_*.pt')))} checkpoints")
    else:
        # fixed Hallmark scores over the full annotated gene set: no fitting,
        # so no HVG step and no checkpoint. Fold-wise CT grouping still applies.
        plan = None
        S_fixed = hallmark_scores(X, prior["mask"])
        print(f"  {len(patients)} patients ({y.sum()} pos), fixed Hallmark "
              f"scores over {X.shape[1]:,} genes -> {S_fixed.shape[1]} pathways")

    out_dir = (RESULTS_IRAEGIS / cohort /
               ("downstream_audit" if representation == "learned"
                else "downstream_audit_hallmark"))
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["gate_stack", "concat_lr", "mean_lr"]
    oof = {v: np.full(len(patients), np.nan) for v in variants}
    rows, t0 = [], time.time()

    for i, held in enumerate(patients):
        if representation == "hallmark":
            groups_f, ct_all, keep = select_ct_groups(
                obs, pat_ids != held, SPLIT_CT_GROUPS)
            h, ct_f, pat_f = S_fixed[keep], ct_all[keep], pat_ids[keep]
            _score_fold(h, ct_f, pat_f, groups_f, held, i, patients, y,
                        pat_labels, oof, rows, variants, out_dir)
            continue

        ck = ck_dir / f"fold_{held}.pt"
        if not ck.exists():
            print(f"  [{i+1}/{len(patients)}] {held}: no checkpoint, skipped")
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
        ae.attach_ct_head(len(groups_f))      # checkpoint carries the aux head
        ae.load_state_dict(torch.load(ck, map_location="cpu"))
        ae.to(DEVICE)
        ae.eval()

        with tempfile.TemporaryDirectory(prefix="audit_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp),
                                         ct_ids=ct_f, verbose=False,
                                         suffix="_fold")

        _score_fold(h, ct_f, pat_f, groups_f, held, i, patients, y,
                    pat_labels, oof, rows, variants, out_dir)
        del ae, h, X_f
        gc.collect()

    summary = {"cohort": cohort, "representation": representation,
               "source_checkpoints": src_dir if representation == "learned" else None,
               "n_patients": len(patients), "n_positive": int(y.sum()),
               "note": "identical per-fold autoencoder and fold geometry across "
                       "variants; only the patient-level classifier differs",
               "variants": {}}
    print(f"\n  {'variant':<14} {'AUC':>7} {'95% CI':>16} {'AUPRC':>8}")
    for v in variants:
        m = ~np.isnan(oof[v])
        if m.sum() < 3:
            continue
        a = float(roc_auc_score(y[m], oof[v][m]))
        p = float(average_precision_score(y[m], oof[v][m]))
        lo, hi_ = _boot(y[m], oof[v][m], roc_auc_score)
        summary["variants"][v] = {"auc": a, "auc_ci95": [lo, hi_], "auprc": p,
                                  "n_scored": int(m.sum())}
        print(f"  {v:<14} {a:>7.4f} {f'[{lo:.3f}, {hi_:.3f}]':>16} {p:>8.4f}")
    summary["total_seconds"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  -> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--src-dir", default="ae_per_fold_deterministic_foldsel",
                    help="results subdir holding the per-fold checkpoints")
    ap.add_argument("--representation", choices=["learned", "hallmark"],
                    default="learned",
                    help="learned = per-fold autoencoder h; hallmark = fixed "
                         "Hallmark pathway scores (no learning, no checkpoint)")
    a = ap.parse_args()
    run_cohort(a.cohort, a.src_dir, a.representation)


if __name__ == "__main__":
    main()
