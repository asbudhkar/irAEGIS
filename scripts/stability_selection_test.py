#!/usr/bin/env python3
"""Does stability selection fix the cell-type gate's fold-to-fold instability?

The gate keeps a cell type when its inner-LOOCV AUC on the training fold clears
0.50. That AUC is a noisy statistic at n = 16-33, so a hard threshold flips
membership between folds whenever several cell types score similarly - on
GSE249898 the two strongest cell types are dropped in 12 and 16 of 32 folds.

Stability selection replaces "clears the bar once" with "clears it reproducibly":
subsample the TRAINING patients repeatedly, re-run the gate on each subsample,
and keep only cell types selected in at least `threshold` of them. The held-out
patient appears in no subsample, so this stays leakage-free.

    S_c = #{inner resamples selecting cell type c} / #{inner resamples}
    keep c  iff  S_c >= threshold

Both arms reuse the same frozen per-fold autoencoders and identical folds, so
the only difference is how cell types are chosen.

Outputs to results/iraegis/<cohort>/stability_selection/

Usage:
    python scripts/stability_selection_test.py --cohort GSE249898_integrated_pre_ici
"""
from __future__ import annotations

import argparse, gc, json, sys, tempfile, time
from collections import Counter
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
from models.iraegis.train_utils import (precompute_embeddings,
    train_h_concat_gated_concat_en, AE_LATENT_DIM, AE_DROPOUT)
from models.iraegis.fold_selection import plan_folds

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"
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


def run(cohort, src_dir, threshold, n_resamples):
    print(f"\n{'='*70}\n  Stability selection: {cohort}\n{'='*70}")
    ck = RESULTS_IRAEGIS / cohort / src_dir / "checkpoints"
    X, obs, _g, _cg, _ci, pat, lab, prior = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)
    pats = sorted(lab.keys()); y = np.array([lab[p] for p in pats], dtype=np.int64)
    plan = plan_folds(X, obs, prior["mask"], pat, pats, SPLIT_CT_GROUPS, verbose=False)
    n_pw = prior["mask"].shape[1]
    print(f"  {len(pats)} patients, threshold={threshold}, {n_resamples} resamples")

    arms = ["hard_gate", "stability"]
    oof = {a: np.full(len(pats), np.nan) for a in arms}
    sel = {a: Counter() for a in arms}
    out = RESULTS_IRAEGIS / cohort / "stability_selection"; out.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()

    for i, held in enumerate(pats):
        f = ck / f"fold_{held}.pt"
        if not f.exists():
            continue
        fs = plan["folds"][held]
        cells, genes = fs["cell_keep"], fs["gene_idx"]
        Xf = X[np.ix_(cells, genes)]; ctf = fs["ct_ids"][cells]; patf = pat[cells]
        obsf = obs.loc[cells].reset_index(drop=True); groups = fs["ct_groups"]
        mask = torch.tensor(prior["mask"][genes, :], dtype=torch.float32)
        ae = PathwayAE(Xf.shape[1], n_pw, mask, AE_LATENT_DIM, AE_DROPOUT,
                       norm="ctbn", act="gelu", n_ct=len(groups))
        ae.attach_ct_head(len(groups))
        ae.load_state_dict(torch.load(f, map_location="cpu")); ae.to(DEVICE); ae.eval()
        with tempfile.TemporaryDirectory() as tmp:
            h, _ = precompute_embeddings(ae, Xf, obsf, Path(tmp), ct_ids=ctf,
                                         verbose=False, suffix="_f")
        hi = pats.index(held); rec = {"patient": held, "label": int(y[hi])}
        for a in arms:
            kw = dict(stability_selection=True, n_resamples=n_resamples,
                      stability_threshold=threshold) if a == "stability" else {}
            r = train_h_concat_gated_concat_en(h, patf, ctf, lab, groups,
                                               verbose=False, only_patient_idx=hi, **kw)
            oof[a][hi] = r["oof_probs"][hi]; rec[a] = float(oof[a][hi])
            s = r.get("fold_selected_cts") or []
            s = s[0] if s and isinstance(s[0], list) else list(s)
            sel[a].update(s); rec[f"{a}_n"] = len(s)
        recs.append(rec); pd.DataFrame(recs).to_csv(out / "per_fold.csv", index=False)
        print(f"  [{i+1}/{len(pats)}] {held}: hard={oof['hard_gate'][hi]:.3f} "
              f"stab={oof['stability'][hi]:.3f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        del ae, h, Xf; gc.collect()

    summary = {"cohort": cohort, "threshold": threshold, "n_resamples": n_resamples,
               "arms": {}}
    print(f"\n  {'arm':<12} {'AUC':>8} {'95% CI':>18} {'CTs/fold':>10}")
    print("  " + "-"*52)
    df = pd.DataFrame(recs)
    for a in arms:
        m = ~np.isnan(oof[a])
        if m.sum() < 3: continue
        auc = float(roc_auc_score(y[m], oof[a][m])); lo, hi_ = _boot(y[m], oof[a][m])
        nsel = float(df[f"{a}_n"].mean())
        summary["arms"][a] = {"auc": auc, "auc_ci95": [lo, hi_], "mean_cts": nsel,
                              "selection_freq": dict(sel[a])}
        print(f"  {a:<12} {auc:>8.4f} {f'[{lo:.3f}, {hi_:.3f}]':>18} {nsel:>10.1f}")
    print(f"\n  selection frequency (cell type: hard / stability, out of {int(m.sum())} folds)")
    for ctn in sorted(set(sel['hard_gate']) | set(sel['stability']),
                      key=lambda c: -sel['stability'][c]):
        print(f"    {ctn:<32} {sel['hard_gate'][ctn]:>3} / {sel['stability'][ctn]:>3}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--src-dir", default="ae_per_fold_deterministic_foldsel")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--n-resamples", type=int, default=20)
    a = ap.parse_args()
    run(a.cohort, a.src_dir, a.threshold, a.n_resamples)


if __name__ == "__main__":
    main()
