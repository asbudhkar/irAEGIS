#!/usr/bin/env python3
"""Does stronger AE regularization reduce fold-to-fold representation divergence?

On GSE249898, performance tracks how stable the representation is across folds:
fixed Hallmark prior 0.7186 > production AE 0.7056 > one fixed AE 0.6667 >
per-fold AE 0.4113. If representation variance is what hurts, then constraining
the autoencoder should reduce that variance - and the point of this experiment
is to measure divergence directly, NOT to tune until one cohort looks better.

For each configuration we retrain the autoencoder on a few folds and measure how
much the learned representation moves between them. Divergence is measured in
PATHWAY space: each fold's AE encodes its own cells to h (always 50 Hallmark
dimensions, with fixed semantic meaning), we build the patient x cell-type mean
profile, and compare those profiles across folds by cosine similarity. This is
comparable across folds even though the underlying gene sets differ slightly,
which raw encoder weights are not.

    stability = mean over patient x CT of  cos( h_profile_foldA , h_profile_foldB )
                averaged over all fold pairs

Higher stability with equal-or-better AUC would be a principled result. Higher
stability with worse AUC means the representation was collapsing, not settling.

Outputs to results/iraegis/<cohort>/ae_reg_stability/<config>.json

Usage:
    python scripts/ae_regularization_stability.py --cohort GSE249898_integrated_pre_ici \
        --config wd1e-2 --folds 3
"""
from __future__ import annotations

import argparse, gc, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.config import RESULTS_IRAEGIS, DEVICE, RANDOM_STATE
from models.iraegis.model_utils import PathwayAE
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.train_utils import (train_ae, precompute_embeddings,
    AE_LATENT_DIM, AE_DROPOUT, AE_N_EPOCHS, AE_LR, AE_WEIGHT_DECAY)
from models.iraegis.fold_selection import plan_folds

SPLIT_CT_GROUPS = ["T_cells", "Monocytes", "Dendritic"]
SHARED_GENES = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"

# each config changes exactly ONE knob from the published setting
CONFIGS = {
    "baseline": dict(),
    "wd1e-2":   dict(wd=1e-2),
    "lr3e-4":   dict(lr=3e-4),
    "ep30":     dict(n_epochs=30),
    "latent8":  dict(latent_dim=8),
}


def _pin(seed):
    import random as r
    r.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try: torch.mps.manual_seed(seed)
        except Exception: pass


def profile(h, pat_f, ct_f, patients, n_ct):
    """patient x CT mean of h — comparable across folds (pathway space)."""
    out = np.full((len(patients), n_ct, h.shape[1]), np.nan, dtype=np.float32)
    for j in range(n_ct):
        for i, p in enumerate(patients):
            m = (pat_f == p) & (ct_f == j)
            if m.any():
                out[i, j] = h[m].mean(0)
    return out


def run(cohort, cfg_name, n_folds, src_dir):
    cfg = CONFIGS[cfg_name]
    print(f"\n{'='*70}\n  AE stability — {cohort} — config '{cfg_name}' {cfg}\n{'='*70}",
          flush=True)
    X, obs, _g, _cg, _ci, pat, lab, prior = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(SHARED_GENES) if SHARED_GENES.exists() else None,
        split_ct_groups=SPLIT_CT_GROUPS, defer_selection=True, verbose=False)
    patients = sorted(lab.keys())
    plan = plan_folds(X, obs, prior["mask"], pat, patients, SPLIT_CT_GROUPS, verbose=False)
    n_pw = prior["mask"].shape[1]
    use = patients[:n_folds]
    latent = cfg.get("latent_dim", AE_LATENT_DIM)
    ck_dir = RESULTS_IRAEGIS / cohort / src_dir / "checkpoints"

    profs, t0 = [], time.time()
    for held in use:
        fs = plan["folds"][held]
        cells, genes = fs["cell_keep"], fs["gene_idx"]
        Xf = X[np.ix_(cells, genes)]; ctf = fs["ct_ids"][cells]; patf = pat[cells]
        obsf = obs.loc[cells].reset_index(drop=True); groups = fs["ct_groups"]
        mask = torch.tensor(prior["mask"][genes, :], dtype=torch.float32)
        is_tr = patf != held

        _pin(RANDOM_STATE)
        ae = PathwayAE(Xf.shape[1], n_pw, mask, latent, AE_DROPOUT,
                       norm="ctbn", act="gelu", n_ct=len(groups))
        if cfg_name == "baseline" and (ck_dir / f"fold_{held}.pt").exists():
            ae.attach_ct_head(len(groups))
            ae.load_state_dict(torch.load(ck_dir / f"fold_{held}.pt", map_location="cpu"))
            note = "loaded existing checkpoint"
        else:
            train_ae(ae, Xf[is_tr], ct_ids=ctf[is_tr], verbose=False,
                     n_epochs=cfg.get("n_epochs", AE_N_EPOCHS),
                     lr=cfg.get("lr", AE_LR), wd=cfg.get("wd", AE_WEIGHT_DECAY))
            note = "trained"
        ae.to(DEVICE); ae.eval()
        with tempfile.TemporaryDirectory() as tmp:
            h, _ = precompute_embeddings(ae, Xf, obsf, Path(tmp), ct_ids=ctf,
                                         verbose=False, suffix="_f")
        profs.append(profile(h, patf, ctf, patients, len(groups)))
        print(f"    fold {held}: {note}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        del ae, h, Xf; gc.collect()

    # pairwise cosine similarity of patient x CT profiles across folds
    sims = []
    for a in range(len(profs)):
        for b in range(a + 1, len(profs)):
            A, B = profs[a], profs[b]
            ok = ~(np.isnan(A).any(-1) | np.isnan(B).any(-1))
            u, v = A[ok], B[ok]
            num = (u * v).sum(-1)
            den = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12
            sims.append(float(np.mean(num / den)))
    res = {"cohort": cohort, "config": cfg_name, "params": cfg, "n_folds": len(profs),
           "pairwise_cosine": sims, "mean_stability": float(np.mean(sims)),
           "min_stability": float(np.min(sims)), "latent_dim": latent,
           "minutes": (time.time() - t0) / 60}
    out = RESULTS_IRAEGIS / cohort / "ae_reg_stability"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg_name}.json").write_text(json.dumps(res, indent=2))
    print(f"\n  representation stability (mean pairwise cosine): "
          f"{res['mean_stability']:.5f}   min {res['min_stability']:.5f}", flush=True)
    print(f"  -> {out / (cfg_name + '.json')}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--src-dir", default="ae_per_fold_deterministic_foldsel")
    a = ap.parse_args()
    run(a.cohort, a.config, a.folds, a.src_dir)


if __name__ == "__main__":
    main()
