#!/usr/bin/env python3
# Frozen-AE occlusion fidelity test for irAEGIS.  Pathway-level fidelity and Cell-type-level fidelity

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.config import (DEVICE, cohort_h5ad, PRIOR_NPZ,
                          SIM_PRIOR_NPZ, DATASETS)
from models.iraegis.data_utils import load_cohort_data
from models.iraegis.model_utils import PathwayAE, CTCondBatchNorm1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler



def _h5ad_prior_for_cohort(cohort_id: str):
    """Per-cohort h5ad path + appropriate Hallmark prior.
    RS_* (Restricted signal) and DS_* (Distributed signal) simulated cohorts use the simulated 
    Hallmark prior; Real cohorts use the real-cohort prior.
    """
    h5ad = cohort_h5ad(cohort_id)
    if cohort_id.startswith("RS_"):
        return h5ad, SIM_PRIOR_NPZ
    if cohort_id.startswith("DS_"):
        return h5ad, DATASETS / "simulation_ds" / "sim_pathway_prior.npz"
    return h5ad, PRIOR_NPZ


# Load trained AE
def _load_frozen_ae(cohort_dir: Path, n_genes: int, n_pathways: int,
                    mask: np.ndarray, n_ct: int) -> PathwayAE:
    # Reload the trained autoencoder and freeze it
    ft_path = cohort_dir / "ae_encoder_ft.pt"
    base_path = cohort_dir / "ae_encoder.pt"
    ckpt = ft_path if ft_path.exists() else base_path
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)

    if state["pw_weight"].shape[0] != n_genes:
        raise RuntimeError(f"AE expects {state['pw_weight'].shape[0]} genes "
                           f"but loaded data has {n_genes}.")

    is_ctbn = any(k.startswith("pw_norm.bns.") for k in state)
    n_ct_ckpt = 0
    if is_ctbn:
        n_ct_ckpt = max(int(k.split(".")[2]) for k in state
                        if k.startswith("pw_norm.bns.")) + 1

    mask_t = torch.tensor(mask, dtype=torch.float32)
    ae = PathwayAE(n_genes, n_pathways, mask_t,
                   norm="ctbn" if is_ctbn else "bn",
                   n_ct=max(n_ct_ckpt, n_ct))
    ae_state_clean = {k: v for k, v in state.items() if not k.startswith("ct_head.")}
    ae.load_state_dict(ae_state_clean, strict=False)
    ae.to(DEVICE).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def _forward_to_h(ae: PathwayAE, X: np.ndarray, ct_ids: np.ndarray,
                  batch: int = 4096) -> np.ndarray:
    # Run X through the frozen AE encoder in batches
    is_ctbn = isinstance(ae.pw_norm, CTCondBatchNorm1d)
    chunks = []
    n = X.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch):
            x_t = torch.from_numpy(X[i:i+batch]).float().to(DEVICE)
            ct_t = None
            if is_ctbn:
                ct_t = torch.from_numpy(
                    ct_ids[i:i+batch].astype(np.int64)).to(DEVICE)
            h = ae.encode_to_h(x_t, ct_ids=ct_t).detach().cpu().numpy()
            chunks.append(h)
    return np.concatenate(chunks, axis=0)


# Attribution loader
def _genes_in_top_pathways_per_ct(top_pathways_per_ct: dict[str, list[str]],
                                  pw_names: list[str], gene_names: list[str],
                                  W_mask: np.ndarray) -> dict[str, list[str]]:
    
    pw_to_idx = {pw: i for i, pw in enumerate(pw_names)}
    out = {}
    for ct, top_pws in top_pathways_per_ct.items():
        pw_idx = [pw_to_idx[p] for p in top_pws if p in pw_to_idx]
        if not pw_idx:
            out[ct] = []
            continue
        sub = W_mask[:, pw_idx]                     # (G, k)
        gene_in = (sub > 0).any(axis=1)             # union of supports
        out[ct] = [gene_names[g] for g in np.where(gene_in)[0]
                   if g < len(gene_names)]
    return out


def _top_pathways_per_ct(cohort_dir: Path, top_k: int) -> dict[str, list[str]]:
    p = cohort_dir / "cell_explainability" / "cell_pathway_attribution.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if df.empty:
        return {}
    rank_col = "h_diff" if "h_diff" in df.columns else "abs_diff"
    out = {}
    for ct, sub in df.groupby("celltype"):
        if rank_col == "h_diff":
            sub = sub.assign(_rank=sub["h_diff"].abs())
            top = sub.sort_values("_rank", ascending=False).head(top_k)
        else:
            top = sub.sort_values("abs_diff", ascending=False).head(top_k)
        out[str(ct)] = top["pathway"].astype(str).tolist()
    return out


# Masking
def _mask_columns_per_ct(M: np.ndarray, ct_ids: np.ndarray,
                         ct_groups: list[str], targets_per_ct: dict[str, list[str]],
                         name_to_idx: dict[str, int],
                         fill_value: np.ndarray) -> np.ndarray:
    out = M.copy()
    for ct_idx, ct_name in enumerate(ct_groups):
        cells = np.where(ct_ids == ct_idx)[0]
        if cells.size == 0:
            continue
        target_idx = [name_to_idx[n] for n in targets_per_ct.get(ct_name, [])
                      if n in name_to_idx]
        if not target_idx:
            continue
        target_idx = np.array(target_idx, dtype=np.int64)
        out[np.ix_(cells, target_idx)] = fill_value[target_idx]
    return out



# ── Gated CT stacking + top-25 inference  ────────
#   1. Aggregate h to per-(patient, CT) means using top quartile cells
#   2. Per-CT inner LOOCV LR to estimate per-CT AUC (gating)
#   3. Outer LOOCV: train logistic regression stack on selected CTs
#   4. Return AUC and AUPRC of OOF predictions

def _aggregate_top25_mean(h, pat_ids, ct_ids, pats, n_ct):
    # Per (patient, CT) mean of top quartile by norm cells.
    P = h.shape[1]
    out = np.zeros((len(pats), n_ct, P), dtype=np.float32)
    for j in range(n_ct):
        for i, p in enumerate(pats):
            mask = (pat_ids == p) & (ct_ids == j)
            if mask.sum() == 0:
                continue
            cells = h[mask]
            if cells.shape[0] >= 4:
                norms = np.linalg.norm(cells, axis=1)
                cutoff = np.percentile(norms, 75)
                top = cells[norms >= cutoff]
                out[i, j] = top.mean(axis=0) if len(top) > 0 else cells.mean(axis=0)
            else:
                out[i, j] = cells.mean(axis=0)
    return out

def _per_ct_inner_loocv_aucs(pat_h, pat_labels, train_idx, C=0.1):
    _, n_ct, _ = pat_h.shape
    aucs = np.zeros(n_ct)
    tr_labels = pat_labels[train_idx]
    if (tr_labels == 1).sum() < 2 or (tr_labels == 0).sum() < 2:
        return aucs
    from sklearn.model_selection import LeaveOneOut
    for j in range(n_ct):
        Xj = pat_h[train_idx, j]
        loo = LeaveOneOut()
        preds = np.zeros(len(train_idx))
        for tr2, va2 in loo.split(Xj):
            sc = StandardScaler().fit(Xj[tr2])
            try:
                lr = LogisticRegression(solver="liblinear", max_iter=2000,
                                        C=C, class_weight="balanced",
                                        random_state=0)
                lr.fit(sc.transform(Xj[tr2]), tr_labels[tr2])
                preds[va2[0]] = lr.predict_proba(sc.transform(Xj[va2]))[0, 1]
            except Exception:
                preds[va2[0]] = 0.5
        try:
            aucs[j] = roc_auc_score(tr_labels, preds)
        except Exception:
            aucs[j] = 0.5
    return aucs


def _gated_ct_stacking_predict(pat_h, pat_labels, P_idx, gate=0.50,
                        active_ct_indices: list[int] | None = None,
                        use_logit=True):
    n_pat, n_ct, _ = pat_h.shape
    tr_idx = np.array([i for i in range(n_pat) if i != P_idx])
    per_ct = _per_ct_inner_loocv_aucs(pat_h, pat_labels, tr_idx)
    if active_ct_indices is not None:
        # Mask CTs not in active set
        for j in range(n_ct):
            if j not in active_ct_indices:
                per_ct[j] = 0.0
    selected = [j for j, a in enumerate(per_ct) if a >= gate]
    if not selected:
        if active_ct_indices is not None and len(active_ct_indices) > 0:
            selected = [int(active_ct_indices[
                int(np.argmax([per_ct[j] for j in active_ct_indices]))])]
        else:
            selected = [int(np.argmax(per_ct))]
    feats = np.full((n_pat, len(selected)), 0.5)
    for col, j in enumerate(selected):
        X_tr_full = pat_h[tr_idx, j]
        y_tr_full = pat_labels[tr_idx]
        if (y_tr_full == 1).sum() < 2 or (y_tr_full == 0).sum() < 2:
            continue
        sc = StandardScaler().fit(X_tr_full)
        try:
            lr = LogisticRegression(max_iter=2000, solver="lbfgs",
                                    class_weight="balanced", random_state=0, C=0.1)
            lr.fit(sc.transform(X_tr_full), y_tr_full)
            feats[P_idx, col] = lr.predict_proba(sc.transform(pat_h[[P_idx], j]))[0, 1]
        except Exception:
            feats[P_idx, col] = 0.5
        for i in tr_idx:
            tr_minus_i = np.array([m for m in tr_idx if m != i])
            X_tr2 = pat_h[tr_minus_i, j]
            y_tr2 = pat_labels[tr_minus_i]
            if (y_tr2 == 1).sum() < 2 or (y_tr2 == 0).sum() < 2:
                continue
            try:
                sc2 = StandardScaler().fit(X_tr2)
                lr2 = LogisticRegression(max_iter=2000, solver="lbfgs",
                                          class_weight="balanced", random_state=0, C=0.1)
                lr2.fit(sc2.transform(X_tr2), y_tr2)
                feats[i, col] = lr2.predict_proba(sc2.transform(pat_h[[i], j]))[0, 1]
            except Exception:
                feats[i, col] = 0.5
    f = feats.copy()
    if use_logit:
        f = np.log(np.clip(f, 1e-6, 1 - 1e-6) / np.clip(1 - f, 1e-6, 1 - 1e-6))
    outer = LogisticRegression(C=1.0, max_iter=2000,
                                class_weight="balanced", random_state=0)
    try:
        outer.fit(f[tr_idx], pat_labels[tr_idx])
        return float(outer.predict_proba(f[[P_idx]])[0, 1])
    except Exception:
        return 0.5

def _iraegis_inference_metrics(h: np.ndarray, pat_ids: np.ndarray, ct_ids: np.ndarray,
                        pat_labels: dict, ct_groups: list[str],
                        active_cts: list[str] | None = None) -> tuple[float, float]:
    unique_pats = sorted(pat_labels.keys())
    pat_arr = np.array([pat_labels[p] for p in unique_pats])
    n_pats = len(unique_pats)

    pat_idx_map = {p: i for i, p in enumerate(unique_pats)}
    pat_int = np.array([pat_idx_map[p] for p in pat_ids])
    pat_h = _aggregate_top25_mean(h, pat_int, ct_ids,
                                   list(range(n_pats)), len(ct_groups))

    if active_cts is not None:
        ct_to_idx = {c: i for i, c in enumerate(ct_groups)}
        active_idx = [ct_to_idx[c] for c in active_cts if c in ct_to_idx]
    else:
        active_idx = None

    probs = np.zeros(n_pats)
    for P_idx in range(n_pats):
        probs[P_idx] = _gated_ct_stacking_predict(pat_h, pat_arr, P_idx,
                                            active_ct_indices=active_idx)
    try:
        auc   = float(roc_auc_score(pat_arr, probs))
        auprc = float(average_precision_score(pat_arr, probs))
    except Exception:
        auc, auprc = 0.5, float(pat_arr.mean())
    return auc, auprc

def _normalize_ct(name: str) -> str:
    return name.lower().replace("_cells", "").replace(" ", "_").strip("_")

def _top_cts_by_auc(cohort_dir: Path, ct_groups: list[str],
                    auc_threshold: float) -> list[str]:
    p = cohort_dir / "irae_per_ct_summary.json"
    if not p.exists():
        return []
    import json
    with open(p) as f:
        per_ct = json.load(f)

    norm_to_group = {_normalize_ct(g): g for g in ct_groups}
    ranked = sorted(per_ct.items(),
                    key=lambda kv: kv[1].get("mean_fold_auc", 0.0),
                    reverse=True)
    out = []
    for name, info in ranked:
        if info.get("mean_fold_auc", 0.0) <= auc_threshold:
            continue
        mapped = norm_to_group.get(_normalize_ct(name))
        if mapped and mapped not in out:
            out.append(mapped)
    return out

def run_cohort(cohort_id: str, results_dir: Path,
               top_k_pw: int,
               auc_threshold: float,
               mask_fill: str = "zero") -> dict | None:
    cohort_dir = results_dir / cohort_id
    if not cohort_dir.is_dir():
        return None
    if not (cohort_dir / "ae_encoder.pt").exists() and \
       not (cohort_dir / "ae_encoder_ft.pt").exists():
        return None
    if not (cohort_dir / "cell_explainability").is_dir():
        return None

    print(f"\n[{cohort_id}]")
    h5ad_path, prior_path = _h5ad_prior_for_cohort(cohort_id)

    (X, _obs, gene_names, ct_groups, ct_ids, pat_ids, pat_labels,
     prior) = load_cohort_data(
        cohort_id, h5ad_path=h5ad_path, prior_path=prior_path,
        prior_genes_only=True, verbose=False,
        split_ct_groups=["T_cells", "Monocytes", "Dendritic"])
    n_genes = X.shape[1]
    pw_names = list(prior["pathway_names"])
    n_pw = len(pw_names)

    saved_gn_path = cohort_dir / "gene_names.npy"
    if saved_gn_path.exists():
        saved_gn = list(np.load(str(saved_gn_path), allow_pickle=True))
        if saved_gn != list(gene_names):
            print("  WARNING: saved gene_names.npy differs from re-loaded gene order; "
                  "skipping cohort to avoid mismatched AE input.")
            return None

    ctg_path = cohort_dir / "ct_groups.json"
    if not ctg_path.exists():
        raise FileNotFoundError(
            f"{ctg_path} missing. Cohorts trained before the ct_groups.json "
            "change must be retrained — re-run run_iraegis_*.sh after deleting "
            f"results/iraegis/{cohort_id}/.")
    import json
    saved_ctg = json.load(open(ctg_path))["ct_groups"]
    if list(saved_ctg) != list(ct_groups):
        raise RuntimeError(
            f"ct_groups mismatch for {cohort_id}:\n"
            f"  saved at training time: {saved_ctg}\n"
            f"  load_cohort_data today: {list(ct_groups)}\n"
            "The grouping logic has changed since this AE was trained. "
            "Retrain the cohort to fix.")

    ae = _load_frozen_ae(cohort_dir, n_genes=n_genes, n_pathways=n_pw,
                         mask=prior["mask"], n_ct=len(ct_groups))
    W_mask = ae.masked_weight().detach().cpu().numpy()         # (G, P)

    h0 = _forward_to_h(ae, X, ct_ids)
    auc_full, auprc_full = _iraegis_inference_metrics(h0, pat_ids, ct_ids, pat_labels, ct_groups)
    print(f"  full                  AUC={auc_full:.4f}  AUPRC={auprc_full:.4f}")

    top_p = _top_pathways_per_ct(cohort_dir, top_k_pw)
    if not top_p:
        print("  no pathway attribution rankings — skipping")
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    if mask_fill == "zero":
        X_fill = np.zeros(X.shape[1], dtype=np.float32)
    elif mask_fill == "mean":
        X_fill = X.mean(axis=0)
    else:
        raise ValueError(f"mask_fill must be 'zero' or 'mean', got {mask_fill!r}")

    # Mask genes in top-K pathways
    pw_genes = _genes_in_top_pathways_per_ct(top_p, pw_names,
                                             list(gene_names), W_mask)
    if not pw_genes:
        print("  no top-pathway gene sets — skipping")
        return None
    avg_pwg = float(np.mean([len(v) for v in pw_genes.values()]))
    X_neg_pg = _mask_columns_per_ct(X, ct_ids, ct_groups, pw_genes,
                                    gene_to_idx, X_fill)
    h_neg_pg = _forward_to_h(ae, X_neg_pg, ct_ids)
    auc_neg_pg, auprc_neg_pg = _iraegis_inference_metrics(
        h_neg_pg, pat_ids, ct_ids, pat_labels, ct_groups)
    print(f"  -genes in top-{top_k_pw} PWs (necc, all CTs) "
          f"AUC={auc_neg_pg:.4f}  Δ={auc_neg_pg - auc_full:+.4f}  "
          f"AUPRC={auprc_neg_pg:.4f}  Δ={auprc_neg_pg - auprc_full:+.4f}")

    # Mask cell types with classifier AUC > threshold
    valid_top_cts = _top_cts_by_auc(cohort_dir, list(ct_groups), auc_threshold)
    if valid_top_cts:
        valid_idx = {ct_groups.index(ct) for ct in valid_top_cts
                     if ct in ct_groups}
        ct_mask = np.isin(ct_ids, list(valid_idx))
        X_compound = X.copy()
        X_compound[ct_mask] = X_fill
        n_cells_masked = int(ct_mask.sum())
        h_compound = _forward_to_h(ae, X_compound, ct_ids)
        auc_compound, auprc_compound = _iraegis_inference_metrics(
            h_compound, pat_ids, ct_ids, pat_labels, ct_groups, active_cts=None)
        print(f"  -CTs with AUC>{auc_threshold} ({len(valid_top_cts)} CTs, "
              f"{n_cells_masked} cells, necc) "
              f"AUC={auc_compound:.4f}  Δ={auc_compound - auc_full:+.4f}  "
              f"AUPRC={auprc_compound:.4f}  Δ={auprc_compound - auprc_full:+.4f}")
        avg_compound = float(X.shape[1])
    else:
        auc_compound = auprc_compound = float("nan")
        avg_compound = 0.0
        print("  CT ranking unavailable (no irae_per_ct_summary.json) — skipping CT test")

    row = {
        "cohort":                  cohort_id,
        "n_patients":              len(pat_labels),
        "n_cts":                   len(ct_groups),
        "top_k_pw":                top_k_pw,
        "auc_threshold":           auc_threshold,
        "n_cts_masked":            len(valid_top_cts),
        "mask_fill":               mask_fill,
        "top_cts":                 "|".join(valid_top_cts),
        "avg_pw_genes_per_ct":     round(avg_pwg, 1),
        "avg_compound_genes_per_ct": round(avg_compound, 1),
        "auc_full":                round(auc_full,    4),
        "auprc_full":              round(auprc_full,  4),
        "auc_neg_top_pw_genes":    round(auc_neg_pg,  4),
        "auprc_neg_top_pw_genes":  round(auprc_neg_pg, 4),
        "drop_pw_genes_necc":      round(auc_full   - auc_neg_pg,  4),
        "drop_pw_genes_necc_auprc": round(auprc_full - auprc_neg_pg, 4),
        "auc_neg_cts_compound":    round(auc_compound,   4) if not np.isnan(auc_compound)   else "",
        "auprc_neg_cts_compound":  round(auprc_compound, 4) if not np.isnan(auprc_compound) else "",
        "drop_cts_compound_necc":  round(auc_full   - auc_compound,   4) if not np.isnan(auc_compound)   else "",
        "drop_cts_compound_necc_auprc": round(auprc_full - auprc_compound, 4) if not np.isnan(auprc_compound) else "",
    }
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/iraegis")
    ap.add_argument("--cohorts", nargs="*", default=None,
                    help="If omitted, runs all cohorts under --results-dir.")
    ap.add_argument("--top-k-pw",   type=int, default=20,
                    help="Top-K pathways per CT to mask (default 20, matches "
                         "the manuscript methods text).")
    ap.add_argument("--auc-threshold", type=float, default=0.5,
                    help="Mask all CTs with per-CT classifier mean_fold_auc > "
                         "threshold (variable count per cohort). Default 0.5 "
                         "masks all above-chance discriminative CTs.")
    ap.add_argument("--mask-fill", choices=["zero", "mean"], default="zero",
                    help="Fill value for masked columns (gene/pathway). "
                         "'zero' = hard zero-out, 'mean' = cohort-mean replacement.")
    ap.add_argument("--out-csv",    default="results/fidelity_occlusion.csv")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if args.cohorts:
        cohorts = args.cohorts
    else:
        cohorts = sorted(d.name for d in results_dir.iterdir()
                         if d.is_dir() and (d / "cell_explainability").is_dir())

    rows = []
    for cohort in cohorts:
        try:
            r = run_cohort(cohort, results_dir,
                           args.top_k_pw, args.auc_threshold,
                           mask_fill=args.mask_fill)
        except Exception as e:
            print(f"  [{cohort}] FAILED: {type(e).__name__}: {e}")
            r = None
        if r is not None:
            rows.append(r)

    if not rows:
        print("\nNo cohorts produced fidelity results.")
        return

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    try: shown = out_csv.resolve().relative_to(Path.cwd())
    except ValueError: shown = out_csv
    print(f"\nWrote {len(df)} rows → {shown}")


if __name__ == "__main__":
    main()
