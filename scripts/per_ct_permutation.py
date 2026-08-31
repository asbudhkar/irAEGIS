#!/usr/bin/env python3
"""Which cell types carry signal, under the PER-FOLD protocol?

An earlier version of this test used the production model - one autoencoder
trained on all cells. This runs it under the leakage-free per-fold protocol
instead, so it answers a sharper question: when each fold's representation is
learned without the held-out patient, does any individual cell type still carry
patient-level signal, and does that differ from what the production model saw?

For each cell type, its per-cell-type classifier is evaluated by leave-one-
patient-out AUC using that fold's own frozen autoencoder, then compared against
a null built by permuting the patient labels through the identical procedure.
Reusing the checkpoints is valid because the autoencoder is unsupervised and the
fold geometry is label-blind - permuting labels cannot change either. Everything
label-dependent is refit under each permutation.

Reports per cell type: observed AUC, null mean, and the two-sided permutation
p-value with the standard +1 correction.

Outputs to results/iraegis/<cohort>/per_ct_permutation.json

Usage:
    python scripts/per_ct_permutation.py --cohort GSE249898_integrated_pre_ici --perms 200
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

from utils.config import RESULTS_IRAEGIS, DEVICE, RANDOM_STATE
from models.iraegis.model_utils import PathwayAE
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.train_utils import (
    precompute_embeddings, AE_LATENT_DIM, AE_DROPOUT,
)
from models.iraegis.fold_selection import plan_folds

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"


def _agg(h, pat_f, ct_f, patients, n_ct):
    """patient x CT x pathway, top-25%-by-norm (the published rule)."""
    P = h.shape[1]
    out = np.zeros((len(patients), n_ct, P), dtype=np.float32)
    for j in range(n_ct):
        for i, p in enumerate(patients):
            m = (pat_f == p) & (ct_f == j)
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


def run(cohort, src_dir, perms, seed=0):
    print(f"\n{'='*72}\n  Per-cell-type permutation test (per-fold): {cohort}\n{'='*72}")
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

    # For each fold, aggregate that fold's own representation to patient x CT.
    print("  building per-fold patient x cell-type matrices ...")
    fold_mat, ct_names, t0 = {}, None, time.time()
    for i, held in enumerate(patients):
        ck = ck_dir / f"fold_{held}.pt"
        if not ck.exists():
            continue
        fs = plan["folds"][held]
        cells, genes = fs["cell_keep"], fs["gene_idx"]
        X_f = X[np.ix_(cells, genes)]
        ct_f, pat_f = fs["ct_ids"][cells], pat_ids[cells]
        obs_f = obs.loc[cells].reset_index(drop=True)
        groups = list(fs["ct_groups"])
        mask_f = torch.tensor(prior["mask"][genes, :], dtype=torch.float32)
        ae = PathwayAE(X_f.shape[1], n_pw, mask_f, AE_LATENT_DIM, AE_DROPOUT,
                       norm="ctbn", act="gelu", n_ct=len(groups))
        ae.attach_ct_head(len(groups))
        ae.load_state_dict(torch.load(ck, map_location="cpu"))
        ae.to(DEVICE); ae.eval()
        with tempfile.TemporaryDirectory(prefix="pct_") as tmp:
            h, _ = precompute_embeddings(ae, X_f, obs_f, Path(tmp), ct_ids=ct_f,
                                         verbose=False, suffix="_fold")
        fold_mat[held] = (_agg(h, pat_f, ct_f, patients, len(groups)), groups)
        if ct_names is None or len(groups) > len(ct_names):
            ct_names = groups
        del ae, h, X_f; gc.collect()
        if (i + 1) % 8 == 0:
            print(f"    {i+1}/{len(patients)} ({time.time()-t0:.0f}s)")
    print(f"  {len(fold_mat)} folds cached, {len(ct_names)} cell types\n")

    def ct_auc(ct, labels):
        """LOOCV AUC for one cell type: each patient scored by its own fold."""
        oof, keep = [], []
        for hi, held in enumerate(patients):
            if held not in fold_mat:
                continue
            mat, groups = fold_mat[held]
            if ct not in groups:
                continue
            j = groups.index(ct)
            tr = [k for k, p in enumerate(patients) if p != held and p in fold_mat]
            ytr = labels[tr]
            if (ytr == 1).sum() < 2 or (ytr == 0).sum() < 2:
                continue
            sc = StandardScaler().fit(mat[tr, j])
            try:
                m = LogisticRegression(solver="liblinear", max_iter=2000, C=0.1,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)
                m.fit(sc.transform(mat[tr, j]), ytr)
                oof.append(m.predict_proba(sc.transform(mat[[hi], j]))[0, 1])
            except Exception:
                oof.append(0.5)
            keep.append(hi)
        if len(keep) < 5 or len(set(labels[keep])) < 2:
            return np.nan
        return roc_auc_score(labels[keep], oof)

    rng = np.random.default_rng(seed)
    perm_labels = [rng.permutation(y) for _ in range(perms)]
    rows = []
    for ct in ct_names:
        obs_a = ct_auc(ct, y)
        if not np.isfinite(obs_a):
            continue
        null = np.array([ct_auc(ct, pl) for pl in perm_labels])
        null = null[np.isfinite(null)]
        n_ge = int((null >= obs_a).sum()); n_le = int((null <= obs_a).sum())
        p2 = min(1.0, 2 * (min(n_ge, n_le) + 1) / (len(null) + 1))
        rows.append({"cell_type": ct, "observed_auc": float(obs_a),
                     "null_mean": float(null.mean()), "null_sd": float(null.std()),
                     "p_two_sided": float(p2), "n_perms": int(len(null))})
        print(f"  {ct:<32} obs={obs_a:.3f}  null={null.mean():.3f}"
              f"±{null.std():.3f}  p={p2:.4f}{'  *' if p2 < 0.05 else ''}")

    df = pd.DataFrame(rows).sort_values("p_two_sided")
    # Holm correction across cell types
    m = len(df)
    df["p_holm"] = np.minimum(1.0, df.p_two_sided.values *
                              (m - np.arange(m)))
    df["p_holm"] = np.maximum.accumulate(df.p_holm.values)
    print(f"\n  after Holm correction across {m} cell types: "
          f"{int((df.p_holm < 0.05).sum())} significant")
    for _, r in df[df.p_holm < 0.05].iterrows():
        print(f"    {r.cell_type:<32} obs={r.observed_auc:.3f}  p_holm={r.p_holm:.4f}")
    out = {"cohort": cohort, "perms": perms, "per_ct": df.to_dict("records")}
    p = RESULTS_IRAEGIS / cohort / "per_ct_permutation.json"
    p.write_text(json.dumps(out, indent=2))
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
