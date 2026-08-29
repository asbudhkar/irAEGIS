#!/usr/bin/env python3
"""Can the top-ranked cell types alone preserve predictive performance?

Reviewer 1 (comment 2) notes that irAEGIS estimates per-cell-type predictive
importance but then retains every eligible cell type at inference, and asks
whether restricting to the highest-ranked cell types preserves accuracy - which
would both validate the importance scores and cut inference cost.

The ranking is the inner-LOOCV AUC each cell-type classifier achieves on the
TRAINING patients of that fold - the same quantity the gate already uses. It
therefore never sees the held-out patient, and the set of retained cell types is
allowed to differ between folds, exactly as the gate's does.

k is pre-registered as {1, 2, 3, 5} plus the unrestricted model, and every value
is reported. This is deliberately a small fixed grid: searching k until one
value beat the full model would be selection on the test metric, which is the
practice Reviewer 3 is already probing elsewhere in the manuscript.

Cost is reported as the fraction of cell-type classifiers that must be fitted
at inference relative to the unrestricted model, since reduced computational
cost is half of what the reviewer asks about.

Outputs to results/iraegis/<cohort>/topk_celltypes/:
    per_fold.csv   per-patient prediction at each k, with the cell types kept
    summary.json   AUC and CI per k, plus the retained-cell-type frequencies

Usage:
    python scripts/topk_celltypes.py --cohort GSE189125_pre_ici
"""
from __future__ import annotations

import argparse, gc, json, sys, tempfile, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

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
K_GRID = [1, 2, 3, 5]          # pre-registered
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


def run(cohort: str, src_dir: str) -> dict:
    print(f"\n{'='*72}\n  Top-k cell-type restriction: {cohort}\n{'='*72}")
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

    settings = [("full", None)] + [(f"top{k}", k) for k in K_GRID]
    oof = {n: np.full(len(patients), np.nan) for n, _ in settings}
    kept = {n: Counter() for n, _ in settings}
    n_sel = {n: [] for n, _ in settings}

    out_dir = RESULTS_IRAEGIS / cohort / "topk_celltypes"
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

        ae = PathwayAE(X_f.shape[1], n_pw, mask_f, AE_LATENT_DIM, AE_DROPOUT,
                       norm="ctbn", act="gelu", n_ct=len(groups_f))
        ae.attach_ct_head(len(groups_f))
        ae.load_state_dict(torch.load(ck, map_location="cpu"))
        ae.to(DEVICE); ae.eval()
        with tempfile.TemporaryDirectory(prefix="topk_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp),
                                         ct_ids=ct_f, verbose=False,
                                         suffix="_fold")
        hi = patients.index(held)
        row = {"patient": held, "label": int(y[hi]), "n_ct_available": len(groups_f)}
        for name, k in settings:
            r = train_h_concat_gated_concat_en(
                h, pat_f, ct_f, pat_labels, groups_f, verbose=False,
                only_patient_idx=hi, top_k=k)
            oof[name][hi] = r["oof_probs"][hi]
            sel = r.get("fold_selected_cts") or []
            sel = sel[0] if sel and isinstance(sel[0], list) else list(sel)
            kept[name].update(sel); n_sel[name].append(len(sel))
            row[name] = float(r["oof_probs"][hi])
            row[f"{name}_cts"] = json.dumps(list(sel))
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "per_fold.csv", index=False)
        print(f"  [{i+1}/{len(patients)}] {held}: " +
              "  ".join(f"{n}={oof[n][hi]:.3f}" for n, _ in settings))
        del ae, h, X_f; gc.collect()

    full_cost = np.mean(n_sel["full"]) if n_sel["full"] else np.nan
    print(f"\n  {'setting':<8} {'AUC':>7} {'95% CI':>16} {'AUPRC':>7} "
          f"{'CTs used':>9} {'cost':>7}")
    print("  " + "-" * 60)
    summary = {"cohort": cohort, "k_grid": K_GRID, "settings": {}}
    for name, k in settings:
        m = ~np.isnan(oof[name])
        if m.sum() < 3:
            continue
        auc = float(roc_auc_score(y[m], oof[name][m]))
        lo, hi_ = _boot(y[m], oof[name][m])
        mean_ct = float(np.mean(n_sel[name]))
        cost = mean_ct / full_cost if full_cost else float("nan")
        print(f"  {name:<8} {auc:>7.4f} {f'[{lo:.3f}, {hi_:.3f}]':>16} "
              f"{average_precision_score(y[m], oof[name][m]):>7.4f} "
              f"{mean_ct:>9.1f} {cost:>6.0%}")
        summary["settings"][name] = {
            "k": k, "auc": auc, "auc_ci95": [lo, hi_],
            "auprc": float(average_precision_score(y[m], oof[name][m])),
            "mean_cts_used": mean_ct, "relative_cost": cost,
            "most_kept": kept[name].most_common(5)}
    summary["total_seconds"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  most frequently retained at top3: "
          f"{[c for c, _ in kept['top3'].most_common(3)]}")
    print(f"  -> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--src-dir", default="ae_per_fold_deterministic_foldsel")
    a = ap.parse_args()
    run(a.cohort, a.src_dir)


if __name__ == "__main__":
    main()
