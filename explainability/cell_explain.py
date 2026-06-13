# Dataset-level cell explainability for irAEGIS.

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.iraegis.model_utils import PathwayAE
from utils.config import DEVICE


def _align_ct_groups(cell_meta, cv_ct_groups, verbose=False):
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_iraegis_inference import REGROUP

    fine = cell_meta["final_celltype"].astype(str).values
    coarse = np.array([REGROUP.get(c, c) for c in fine])
    name_to_idx = {n: i for i, n in enumerate(cv_ct_groups)}
    ct_ids = np.array([name_to_idx.get(n, -1) for n in coarse], dtype=np.int64)
    keep_mask = ct_ids >= 0
    if verbose:
        print(f"  CT align: {keep_mask.sum()} / {len(keep_mask)} cells matched.")
    return ct_ids, keep_mask


def _derive_model_gene_names(prior_path, h5ad_path, model_mask, n_genes_model):
    import h5py

    prior_raw = np.load(str(prior_path), allow_pickle=True)
    prior_genes = list(prior_raw["gene_names"].astype(str))
    mask_gp = prior_raw["mask"]
    if mask_gp.shape[0] != len(prior_genes):
        mask_gp = mask_gp.T

    # Read h5ad gene names without loading X
    if h5ad_path is None:
        raise ValueError("cell_explain: h5ad_path is required (no default).")
    h5ad_src = h5ad_path
    with h5py.File(str(h5ad_src), "r") as f:
        idx_key = f["var"].attrs.get("_index", "gene")
        h5ad_genes = list(f["var"][idx_key][:].astype(str))

    # Intersection in h5ad order
    prior_set = set(prior_genes)
    shared = [g for g in h5ad_genes if g in prior_set]
    prior_idx_map = {g: i for i, g in enumerate(prior_genes)}
    shared_pri = [prior_idx_map[g] for g in shared]
    mask_shared = mask_gp[shared_pri, :]

    # Identify the 3693 pathway genes in h5ad order
    active_mask = mask_shared.sum(axis=1) > 0
    pw_genes_ordered = [shared[i] for i in range(len(shared)) if active_mask[i]]

    # Map model positions: non-zero mask rows → pathway genes in h5ad order
    model_active = model_mask.sum(axis=1) > 0
    n_pw_model = int(model_active.sum())

    gene_names = [f"_HVG_{i}" for i in range(n_genes_model)]
    if n_pw_model == len(pw_genes_ordered):
        pw_positions = np.where(model_active)[0]
        for pos, name in zip(pw_positions, pw_genes_ordered):
            gene_names[pos] = name
    else:
        print(f"  [WARN] pathway gene count mismatch: model={n_pw_model}, "
              f"prior={len(pw_genes_ordered)} — using prior order as fallback")
        gene_names = prior_genes[:n_genes_model]

    return gene_names



def compute_cell_gene_attribution(
        cell_pw_attr: pd.DataFrame,
        W_mask_np: np.ndarray,
        gene_names: list,
        pw_names: list,
        top_n: int = 50,
) -> pd.DataFrame:
    # Project classifier-aligned differential pathway activity to gene space.
    pw_to_idx = {p: i for i, p in enumerate(pw_names)}
    rows = []

    cell_pw_attr = cell_pw_attr.assign(
        _M_sign=np.sign(cell_pw_attr["w_CT"].fillna(0))
                * cell_pw_attr["h_diff"].fillna(0))

    for ct, grp in cell_pw_attr.groupby("celltype"):
        attr_vec = np.zeros(len(pw_names))
        for _, r in grp.iterrows():
            pi = pw_to_idx.get(r["pathway"])
            if pi is not None:
                attr_vec[pi] = float(r["_M_sign"])

        gene_magnitude = np.abs(W_mask_np @ attr_vec)

        gene_pw_contrib = W_mask_np * attr_vec[np.newaxis, :]
        dominant_pw = np.argmax(np.abs(gene_pw_contrib), axis=1)
        dominant_sign = np.sign(gene_pw_contrib[np.arange(len(gene_names)), dominant_pw])

        gene_scores = gene_magnitude * dominant_sign
        top_idx = np.argsort(-gene_magnitude)[:top_n]

        for rank, gi in enumerate(top_idx):
            rows.append({
                "celltype": ct,
                "rank": rank + 1,
                "gene": gene_names[gi] if gi < len(gene_names) else f"gene_{gi}",
                "score": float(gene_scores[gi]),
                "pathway": pw_names[dominant_pw[gi]],
            })

    return pd.DataFrame(rows)

def run_cell_explain(cohort_id: str, results_base: Path,
                     h5ad_path: Path = None,
                     prior_path: Path = None) -> dict:

    base_dir = results_base / cohort_id
    out_dir = base_dir / "cell_explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _rel(p):
        try: return p.relative_to(REPO_ROOT)
        except ValueError: return p

    print(f"\n{'='*70}")
    print(f"  Cell-Level Explainability — {cohort_id}")
    print(f"  Output → {_rel(out_dir)}")

    for fp in ("h_cells.npy", "cell_meta.csv"):
        if not (base_dir / fp).exists():
            raise FileNotFoundError(f"Missing: {base_dir / fp}")

    h_ft_path = base_dir / "h_cells_ft.npy"
    if h_ft_path.exists():
        h = np.load(str(h_ft_path))
        print(f"  Using fine-tuned embeddings: {h_ft_path}")
    else:
        h = np.load(str(base_dir / "h_cells.npy"))
    cell_meta = pd.read_csv(base_dir / "cell_meta.csv")
    n_pathways = h.shape[1]

    irae_summary_path = base_dir / "irae_per_ct_summary.json"
    ct_summary_path = base_dir / "ct_classifier_summary.json"
    if irae_summary_path.exists():
        with open(irae_summary_path) as f:
            irae_cv = json.load(f)
        ct_groups = list(irae_cv.keys())
        cv = {"ct_groups": ct_groups}
    elif ct_summary_path.exists():
        with open(ct_summary_path) as f:
            cv = json.load(f)
        ct_groups = cv["ct_groups"]
    else:
        raise FileNotFoundError(
            f"Need irae_per_ct_summary.json or ct_classifier_summary.json in {base_dir}")

    _prior_src = prior_path if prior_path else REPO_ROOT / "datasets/resources/pathway_prior.npz"
    prior_raw = np.load(str(_prior_src), allow_pickle=True)
    pw_names = list(prior_raw["pathway_names"])

    ae_ft_path = base_dir / "ae_encoder_ft.pt"
    ae_base_path = base_dir / "ae_encoder.pt"
    _ae_ckpt = ae_ft_path if ae_ft_path.exists() else ae_base_path
    print(f"  AE checkpoint: {_ae_ckpt.name}")
    ae_state = torch.load(str(_ae_ckpt),
                          map_location="cpu", weights_only=True)
    n_genes_model = ae_state["pw_weight"].shape[0]

    mask_t = ae_state["mask"].float()

    saved_gn = base_dir / "gene_names.npy"
    if saved_gn.exists():
        gene_names = list(np.load(str(saved_gn), allow_pickle=True))
    else:
        gene_names = _derive_model_gene_names(
            _prior_src, h5ad_path, mask_t.numpy(), n_genes_model)

    is_ctbn = any(k.startswith("pw_norm.bns.") for k in ae_state)
    n_ct_ckpt = 0
    if is_ctbn:
        n_ct_ckpt = max(int(k.split(".")[2]) for k in ae_state if k.startswith("pw_norm.bns.")) + 1
    ae = PathwayAE(n_genes_model, n_pathways, mask_t,
                   cv.get("ae_latent_dim", 32),
                   norm="ctbn" if is_ctbn else "bn",
                   n_ct=n_ct_ckpt)
    ae_state_clean = {k: v for k, v in ae_state.items() if not k.startswith("ct_head.")}
    ae.load_state_dict(ae_state_clean, strict=False)
    ae.to(DEVICE).eval()
    W_mask_np = ae.masked_weight().detach().cpu().numpy()

    ct_ids, keep_mask = _align_ct_groups(cell_meta, ct_groups, verbose=True)

    pat_col = "patient_id"
    lab_col = "irAE_status"
    pat_ids = cell_meta[pat_col].values.astype(str)
    pat_labs_raw = cell_meta.groupby(pat_col)[lab_col].first()
    pat_labels = {str(p): 1.0 if str(v).strip() in ("Yes", "Severe") else 0.0
                  for p, v in pat_labs_raw.items()}

    h_keep = h[keep_mask]
    pat_ids_keep = pat_ids[keep_mask]
    ct_ids_keep = ct_ids[keep_mask]

    pw_attr_all = pd.DataFrame(
        [{"celltype": ct, "pathway": pw}
         for ct in ct_groups for pw in pw_names[:n_pathways]])

    print(f"\n  [Attribution] Computing h_diff and w_CT per (CT, pathway) ...")
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.preprocessing import StandardScaler as _SS

    pats_sorted = sorted(set(pat_ids_keep))
    y_full = np.array([int(pat_labels.get(p, 0.0)) for p in pats_sorted], dtype=np.int64)
    P = h_keep.shape[1]

    hdiff_rows = []
    for j, ct_name in enumerate(ct_groups):
        h_pat = []
        for p in pats_sorted:
            sel = (ct_ids_keep == j) & (pat_ids_keep == p)
            n_sel = int(sel.sum())
            if n_sel == 0:
                h_pat.append(None); continue
            cells = h_keep[sel]
            if n_sel >= 4:
                norms = np.linalg.norm(cells, axis=1)
                cutoff = np.percentile(norms, 75)
                top = cells[norms >= cutoff]
                h_pat.append(top.mean(axis=0) if len(top) > 0 else cells.mean(axis=0))
            else:
                h_pat.append(cells.mean(axis=0))
        keep = [i for i, x in enumerate(h_pat) if x is not None]
        if len(keep) < 4:
            continue
        X = np.stack([h_pat[i] for i in keep])
        y_sub = y_full[keep]
        if (y_sub == 1).sum() < 2 or (y_sub == 0).sum() < 2:
            continue
        h_diff = X[y_sub == 1].mean(axis=0) - X[y_sub == 0].mean(axis=0)
        try:
            sc = _SS().fit(X)
            lr = _LR(C=0.1, max_iter=2000, solver="liblinear",
                     class_weight="balanced", random_state=0)
            lr.fit(sc.transform(X), y_sub)
            w_CT = lr.coef_.squeeze() / (sc.scale_ + 1e-9)
        except Exception:
            w_CT = np.zeros(P, dtype=np.float64)
        for p_idx, pw in enumerate(pw_names[:n_pathways]):
            hdiff_rows.append({
                "celltype": ct_name, "pathway": pw,
                "h_diff": float(h_diff[p_idx]),
                "w_CT":   float(w_CT[p_idx]),
            })
    if hdiff_rows:
        hdiff_df = pd.DataFrame(hdiff_rows)
        pw_attr_all = pw_attr_all.merge(
            hdiff_df, on=["celltype", "pathway"], how="left")

    pw_attr_all.to_csv(out_dir / "cell_pathway_attribution.csv", index=False)

    print(f"\n  [Genes] Projecting pathway attributions to genes ...")
    gene_df = compute_cell_gene_attribution(
        pw_attr_all, W_mask_np, gene_names, pw_names[:n_pathways], top_n=50)
    gene_df.to_csv(out_dir / "cell_gene_attribution.csv", index=False)
    print(f"    {len(gene_df)} rows")

    summary = {
        "cohort": cohort_id,
        "n_cells": int(keep_mask.sum()),
        "n_pathways": n_pathways,
        "n_ct": len(ct_groups),
        "ct_groups": ct_groups,
    }
    with open(out_dir / "cell_explain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Done → {_rel(out_dir)}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=str, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--h5ad", type=Path, default=None)
    ap.add_argument("--prior", type=Path, default=None)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--sim-cohorts", action="store_true")
    ap.add_argument("--all-cohorts", action="store_true")
    ap.add_argument("--cohorts", nargs="+", default=None,
                    help="Explicit list of cohort IDs")
    args = ap.parse_args()

    if args.results_dir is None:
        args.results_dir = REPO_ROOT / "results" / "iraegis"

    if args.sim and args.prior is None:
        args.prior = REPO_ROOT / "datasets" / "simulation_rs" / "sim_pathway_prior.npz"

    COHORTS_SIM = ["RS_cohort1", "RS_cohort2", "RS_cohort3",
                   "DS_cohort1", "DS_cohort2", "DS_cohort3"]
    COHORTS_REAL = [
        "GSE216329_integrated_pre_ici",
        "GSE249898_integrated_pre_ici",
        "GSE285888_pre_ici",
        "GSE189125_pre_ici",
    ]

    if args.cohorts:
        cohort_list = args.cohorts
    elif args.sim_cohorts:
        cohort_list = COHORTS_SIM
        if args.prior is None:
            args.prior = REPO_ROOT / "datasets" / "simulation_rs" / "sim_pathway_prior.npz"
    elif args.all_cohorts:
        cohort_list = COHORTS_REAL
    else:
        if args.cohort is None:
            ap.error("--cohort, --cohorts, --sim-cohorts, or --all-cohorts is required")
        cohort_list = [args.cohort]

    for cid in cohort_list:
        try:
            run_cell_explain(cid, args.results_dir, args.h5ad, args.prior)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {cid}: {e}")
            traceback.print_exc()
