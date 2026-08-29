#!/usr/bin/env python3
"""AE-per-fold ablation.

For each cohort:
  For each patient p (LOOCV outer fold):
    1. Train AE from scratch on cells from all patients EXCEPT p
    2. Save AE checkpoint (results/iraegis/<cohort>/ae_per_fold/fold_<p>.pt)
    3. Encode ALL cells (including p's) with this AE → h_fold
    4. Run production Phase 2c LOOCV on h_fold → extract OOF prob for p
       (the OOF prob for p uses a Phase 2c LOOCV where p is held out,
        and the AE used to encode was trained without p — proper CV)
  Aggregate the N OOF probs → cohort AUC + AUPRC

Same hyperparameters as production (no mixup, no domain-invariance changes) —
this is the baseline "vanilla per-fold AE" reference.

Output: results/iraegis/<cohort>/ae_per_fold_vanilla/summary.json + per-fold checkpoints
Does NOT modify any shipped file. All outputs live under gitignored dirs.

Run:  python analysis/ablation_ae_per_fold.py --cohort GSE189125_pre_ici
      python analysis/ablation_ae_per_fold.py --all
"""
from __future__ import annotations
import argparse, gc, json, sys, time, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.config import DEVICE, RANDOM_STATE, RESULTS_IRAEGIS
from models.iraegis.model_utils import PathwayAE
from models.iraegis.train_utils import (
    train_ae, precompute_embeddings, train_h_concat_gated_concat_en,
    AE_LATENT_DIM, AE_DROPOUT, AE_N_EPOCHS,
    AE_CT_AUX_WEIGHT, AE_DECORR_WEIGHT, AE_MASK_FRAC,
)
from models.iraegis.data_utils import load_cohort_data


def _pin_seeds(seed: int):
    """Reproducible per-fold seeding — each fold offsets from RANDOM_STATE."""
    import random as _stdlib_random
    _stdlib_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try: torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError): pass


def _enable_determinism():
    """Force deterministic algorithms so per-fold AE variance is a function
    of only the missing-patient training data, not GPU-kernel or Adam noise."""
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For MPS, deterministic ops warn but don't fully guarantee determinism
    except Exception as e:
        print(f"  WARNING: could not enable deterministic algorithms: {e}")


def _free_gpu_mem():
    """Release AE-fold memory before starting the next fold."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try: torch.mps.empty_cache()
        except AttributeError: pass


def _hspace_ood_stats(h_full: np.ndarray, pat_ids: np.ndarray,
                       held_out: str, k: int = 10) -> dict:
    """h-space OOD stat, computed on all cells + per-CT breakdown."""
    from scipy.spatial import cKDTree
    train_mask = pat_ids != held_out
    h_tr = h_full[train_mask]
    h_ho = h_full[~train_mask]
    if len(h_ho) == 0 or len(h_tr) == 0:
        return {"held_mean_nn_dist": None, "train_mean_nn_dist": None,
                "ood_ratio": None}
    tree = cKDTree(h_tr)
    d_ho, _ = tree.query(h_ho, k=k)
    d_tr, _ = tree.query(h_tr, k=k + 1)  # k+1 so we can drop the self-match at k=0
    held_mean  = float(d_ho.mean())
    train_mean = float(d_tr[:, 1:].mean())
    return {
        "held_mean_nn_dist":  held_mean,
        "train_mean_nn_dist": train_mean,
        "ood_ratio":          held_mean / train_mean if train_mean > 0 else None,
    }


def _per_ct_hspace_ood(h_full: np.ndarray, pat_ids: np.ndarray,
                       ct_ids: np.ndarray, ct_groups: list,
                       held_out: str, k: int = 10) -> dict:
    """Per-CT h_OOD ratio — is the held-out patient OOD only in some CTs?"""
    from scipy.spatial import cKDTree
    train_mask = pat_ids != held_out
    ood = {}
    for j, ct_name in enumerate(ct_groups):
        ct_mask = ct_ids == j
        h_tr = h_full[train_mask & ct_mask]
        h_ho = h_full[(~train_mask) & ct_mask]
        if len(h_ho) < k or len(h_tr) < k + 1:
            ood[ct_name] = None
            continue
        tree = cKDTree(h_tr)
        d_ho, _ = tree.query(h_ho, k=k)
        d_tr, _ = tree.query(h_tr, k=k + 1)
        held_mean  = float(d_ho.mean())
        train_mean = float(d_tr[:, 1:].mean())
        ood[ct_name] = held_mean / train_mean if train_mean > 0 else None
    return ood


def _held_recon_mse(ae, X: np.ndarray, ct_ids: np.ndarray,
                    pat_ids: np.ndarray, held_out: str,
                    batch_size: int = 4096) -> dict:
    """Reconstruction MSE for held-out vs training cells using the SAME AE.

    If held_mse >> train_mse, the AE fails at input-space reconstruction of
    the held-out patient — meaning the AE hasn't learned to generalize to
    unseen batches. If they're comparable, the AE handles held-out expression
    fine and any downstream AUC drop is not caused by input-space failure.
    """
    ae.eval()
    train_mask = pat_ids != held_out
    ct_gpu = torch.tensor(ct_ids, dtype=torch.long, device=DEVICE)

    def _batched_mse(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return None
        total_sqerr = 0.0
        total_n = 0
        with torch.no_grad():
            for i in range(0, len(idx), batch_size):
                b = idx[i:i + batch_size]
                xb = torch.tensor(X[b], dtype=torch.float32, device=DEVICE)
                ctb = ct_gpu[b]
                _, _, xr = ae(xb, ct_ids=ctb)  # no denoising mask — clean input
                sq = ((xr - xb) ** 2).sum().item()
                total_sqerr += sq
                total_n += xb.numel()
        return total_sqerr / max(total_n, 1)

    train_mse = _batched_mse(train_mask)
    held_mse  = _batched_mse(~train_mask)
    if train_mse is None or held_mse is None:
        return {"train_recon_mse_clean": None,
                "held_recon_mse_clean": None,
                "recon_ood_ratio": None}
    return {
        "train_recon_mse_clean": float(train_mse),
        "held_recon_mse_clean":  float(held_mse),
        "recon_ood_ratio":       float(held_mse / max(train_mse, 1e-9)),
    }


def _per_ct_diagnostics(h_full: np.ndarray, pat_ids: np.ndarray,
                        ct_ids: np.ndarray, pat_labels: dict,
                        ct_groups: list, held_out: str,
                        C: float = 0.1) -> dict:
    """Mirror Phase 2c's per-CT AUC + held-out patient's φ_p feature vector.

    Uses the SAME top-25 aggregation + StandardScaler + LR(C=0.1, liblinear,
    class_weight="balanced") as train_h_concat_gated_concat_en.

    Returns:
        per_ct_aucs         — list of per-CT LOOCV AUCs on training patients
        per_ct_names        — matching CT names
        held_phi_p_probs    — per-CT prediction probs for held-out patient
        held_phi_p_logits   — logit(probs) — the actual φ_p features fed to
                              the outer stacker
        n_ct_gated_in       — # CTs with train AUC ≥ 0.5 (would be selected)
        gated_cts           — list of CT names that pass the gate
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut

    unique_pats = sorted(pat_labels.keys())
    pat_arr = np.array([pat_labels[p] for p in unique_pats], dtype=np.int64)
    held_idx = unique_pats.index(held_out)
    tr_idx = np.array([i for i in range(len(unique_pats)) if i != held_idx])
    K = len(ct_groups)
    P_dim = h_full.shape[1]

    # Top-25 aggregation (same as Phase 2c lines 485-491)
    pat_h = np.zeros((len(unique_pats), K, P_dim), dtype=np.float32)
    for j in range(K):
        for i, p in enumerate(unique_pats):
            mask = (pat_ids == p) & (ct_ids == j)
            if not mask.any():
                continue
            cells = h_full[mask]
            if cells.shape[0] >= 4:
                norms = np.linalg.norm(cells, axis=1)
                cutoff = np.percentile(norms, 75)
                top = cells[norms >= cutoff]
                pat_h[i, j] = top.mean(axis=0) if len(top) else cells.mean(axis=0)
            else:
                pat_h[i, j] = cells.mean(axis=0)

    per_ct_aucs = np.full(K, np.nan)
    per_ct_probs = np.full(K, 0.5)  # φ_p component for held-out (as prob)
    tr_labels = pat_arr[tr_idx]

    for j in range(K):
        Xj = pat_h[tr_idx, j]
        if (tr_labels == 1).sum() < 2 or (tr_labels == 0).sum() < 2:
            continue
        # Inner LOOCV over training patients → per-CT LOOCV AUC (gate signal)
        preds = np.zeros(len(tr_idx))
        for itr, iva in LeaveOneOut().split(Xj):
            sc = StandardScaler().fit(Xj[itr])
            try:
                lr = LogisticRegression(solver="liblinear", C=C,
                                        class_weight="balanced", max_iter=2000)
                lr.fit(sc.transform(Xj[itr]), tr_labels[itr])
                preds[iva[0]] = lr.predict_proba(sc.transform(Xj[iva]))[0, 1]
            except Exception:
                preds[iva[0]] = 0.5
        try:
            per_ct_aucs[j] = roc_auc_score(tr_labels, preds)
        except Exception:
            per_ct_aucs[j] = 0.5

        # φ_p component: fit LR on all training patients, predict held-out
        try:
            sc = StandardScaler().fit(Xj)
            lr = LogisticRegression(solver="liblinear", C=C,
                                    class_weight="balanced", max_iter=2000)
            lr.fit(sc.transform(Xj), tr_labels)
            per_ct_probs[j] = lr.predict_proba(sc.transform(pat_h[[held_idx], j]))[0, 1]
        except Exception:
            per_ct_probs[j] = 0.5

    per_ct_logits = np.log(np.clip(per_ct_probs, 1e-6, 1 - 1e-6) /
                            np.clip(1 - per_ct_probs, 1e-6, 1 - 1e-6))
    gate_mask = per_ct_aucs >= 0.5
    gated_cts = [ct_groups[j] for j in range(K) if gate_mask[j]]

    return {
        "per_ct_names":       list(ct_groups),
        "per_ct_aucs":        [float(x) if not np.isnan(x) else None
                               for x in per_ct_aucs],
        "held_phi_p_probs":   [float(x) for x in per_ct_probs],
        "held_phi_p_logits":  [float(x) for x in per_ct_logits],
        "n_ct_gated_in":      int(gate_mask.sum()),
        "gated_cts":          gated_cts,
    }


def _load_production_baseline(cohort: str) -> dict:
    """Read the production AUC / AUPRC for side-by-side comparison."""
    prod = RESULTS_IRAEGIS / cohort / "h_concat_gated_concat_en_summary.json"
    if not prod.exists():
        return {"prod_auc": None, "prod_auprc": None}
    with open(prod) as f:
        d = json.load(f)
    return {"prod_auc": d.get("mean_auc"), "prod_auprc": d.get("mean_fold_ap")}


def run_ae_per_fold(cohort: str, ae_epochs: int = AE_N_EPOCHS,
                    resume: bool = True, deterministic: bool = False,
                    seed: int | None = None,
                    ct_aux_weight: float | None = None,
                    decorr_weight: float | None = None,
                    diagnostics: bool = True,
                    fold_selection: bool = False,
                    mask_frac: float | None = None,
                    shuffle_mask: bool = False,
                    no_latent: bool = False):
    """Full per-fold AE ablation for one cohort.

    Args:
        deterministic: if True, apply Fix J+Q — same seed across folds AND
            deterministic torch algorithms. Isolates the missing-patient
            effect from init/kernel-noise variance. Writes results to a
            separate output dir so vanilla results are preserved.
        ct_aux_weight: override AE_CT_AUX_WEIGHT (α). None = production 0.3.
        decorr_weight: override AE_DECORR_WEIGHT (β). None = production 0.1.
            Higher β penalises pathway-activity correlation, which reduces
            rotational slack in the representation and should therefore make
            h more reproducible across folds.
    """
    print(f"\n{'='*70}\n  AE-per-fold ablation: {cohort}"
          f"{' (deterministic)' if deterministic else ''}\n{'='*70}")
    if deterministic:
        _enable_determinism()

    # NB: use the SAME data loading args as production (train.py defaults):
    #   --prior-genes-only            → prior_genes_only=True
    #   --gene-list <shared_genes.txt> → gene_list_path=<shared_genes.txt>
    # Without gene_list_path the cohort would load its native ~17K genes rather
    # than the ~5,693-gene cross-cohort vocabulary the production AE trains on.
    shared_genes = REPO / "datasets" / "processed_h5ad" / "shared_genes.txt"
    SPLIT_CTS = ["T_cells", "Monocytes", "Dendritic"]
    (X, obs, gene_names, ct_groups, ct_ids, pat_ids, pat_labels,
     prior) = load_cohort_data(
        cohort, prior_genes_only=True,
        gene_list_path=str(shared_genes) if shared_genes.exists() else None,
        split_ct_groups=SPLIT_CTS,
        defer_selection=fold_selection)

    # Leakage-free mode: derive the gene set and CT groups per fold from
    # training-fold cells only. Both vary by fold, so each fold trains its own
    # model with its own n_genes / n_ct. plan_folds computes every fold's
    # selection in one streaming pass and returns the union gene space so X is
    # subset once rather than held at full width (~17k genes).
    # Shuffled-prior control (R1.1: does the *biological* prior matter, or would
    # any sparse projection of the same shape do?). Permutes the gene->pathway
    # assignment among pathway-active genes only, so per-pathway gene counts and
    # the active/inactive gene partition are preserved -- gene selection is
    # therefore unchanged and only the biology of the mask is randomised.
    if shuffle_mask:
        _m = prior["mask"]
        _active = np.where(np.asarray(_m.sum(axis=1)).ravel() > 0)[0]
        _rng = np.random.default_rng(RANDOM_STATE)
        _perm = _rng.permutation(_active)
        _m2 = _m.copy()
        _m2[_active, :] = _m[_perm, :]
        prior["mask"] = _m2
        print(f"  SHUFFLED PRIOR: permuted gene->pathway assignment among "
              f"{len(_active):,} pathway-active genes (counts preserved)")

    fold_plan = None
    if fold_selection:
        from models.iraegis.fold_selection import plan_folds
        print("  fold-wise selection: planning CT groups + HVG per fold ...")
        fold_plan = plan_folds(X, obs, prior["mask"], pat_ids,
                               sorted(pat_labels.keys()), SPLIT_CTS,
                               verbose=True)
        union = fold_plan["union_genes"]
        X = np.ascontiguousarray(X[:, union])
        prior["mask"] = prior["mask"][union, :]
        prior["gene_names"] = [prior["gene_names"][i] for i in union]
        gene_names = list(prior["gene_names"])
        # remap each fold's gene indices into the union space
        for f in fold_plan["folds"].values():
            f["gene_idx"] = np.searchsorted(union, f["gene_idx"])

    mask_t = torch.tensor(prior["mask"], dtype=torch.float32)
    n_pw = prior["mask"].shape[1]
    n_ct = len(ct_groups)
    n_genes = X.shape[1]

    unique_pats = sorted(pat_labels.keys())
    n_pats = len(unique_pats)
    y_full = np.array([pat_labels[p] for p in unique_pats])
    baseline = _load_production_baseline(cohort)

    print(f"  n_cells={len(X):,}  n_genes={n_genes}  n_pathways={n_pw}  "
          f"n_ct={n_ct}  n_patients={n_pats}")
    print(f"  AE epochs per fold: {ae_epochs}")
    if baseline["prod_auc"] is not None:
        print(f"  Production baseline (AE-fixed): "
              f"AUC={baseline['prod_auc']:.4f}  AUPRC={baseline['prod_auprc']:.4f}")

    # Deterministic runs go to a separate dir so vanilla results are preserved.
    # Alternate seeds (--seed N) go to their own subdir to preserve default results.
    active_seed = RANDOM_STATE if seed is None else int(seed)
    dir_name = "ae_per_fold_deterministic" if deterministic else "ae_per_fold_vanilla"
    if seed is not None:
        dir_name += f"_seed{active_seed}"
    # α / β overrides get their own subdir so the production-hyperparameter
    # results stay intact (e.g. ae_per_fold_deterministic_b1p0).
    def _tag(v):
        return str(v).replace(".", "p")
    if ct_aux_weight is not None:
        dir_name += f"_a{_tag(ct_aux_weight)}"
    if decorr_weight is not None:
        dir_name += f"_b{_tag(decorr_weight)}"
    if mask_frac is not None:
        dir_name += f"_mf{_tag(mask_frac)}"
    if shuffle_mask:
        dir_name += "_shufmask"
    if no_latent:
        dir_name += "_nolatent"
    if fold_selection:
        dir_name += "_foldsel"
    print(f"  AE loss weights: α={ct_aux_weight if ct_aux_weight is not None else AE_CT_AUX_WEIGHT}"
          f"  β={decorr_weight if decorr_weight is not None else AE_DECORR_WEIGHT}")
    out_dir = RESULTS_IRAEGIS / cohort / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ae_ckpt_dir = out_dir / "checkpoints"
    ae_ckpt_dir.mkdir(exist_ok=True)

    # Resume: load per_fold.csv if it exists
    per_fold_csv = out_dir / "per_fold.csv"
    done_patients: dict[str, dict] = {}
    if resume and per_fold_csv.exists():
        prior_df = pd.read_csv(per_fold_csv)
        for _, r in prior_df.iterrows():
            if pd.notna(r.get("oof_prob")) and \
               (ae_ckpt_dir / f"fold_{r['patient']}.pt").exists():
                done_patients[str(r["patient"])] = r.to_dict()
        if done_patients:
            print(f"  RESUME: skipping {len(done_patients)} completed folds")

    oof_probs = np.full(n_pats, np.nan, dtype=np.float64)
    per_fold_data: list[dict] = []
    for pat, row in done_patients.items():
        idx = unique_pats.index(pat)
        oof_probs[idx] = row["oof_prob"]
        per_fold_data.append(row)

    total_start = time.time()

    for fold_i, held_out in enumerate(unique_pats):
        if held_out in done_patients:
            print(f"\n  --- Fold {fold_i+1}/{n_pats}: {held_out} (SKIP — done) ---")
            continue

        t_fold = time.time()
        print(f"\n  --- Fold {fold_i+1}/{n_pats}: hold out {held_out} "
              f"(label {pat_labels[held_out]}) ---")

        # Per-fold view of the data. Under --fold-selection the gene set, the
        # surviving cell types, and therefore the cell subset are all derived
        # from training-fold cells only, so they differ between folds and this
        # fold gets its own n_genes / n_ct. Otherwise these are the cohort-wide
        # values computed once by load_cohort_data.
        if fold_plan is not None:
            fs = fold_plan["folds"][held_out]
            cell_keep = fs["cell_keep"]
            gene_idx  = fs["gene_idx"]
            X_f       = X[np.ix_(cell_keep, gene_idx)]
            ct_f      = fs["ct_ids"][cell_keep]
            pat_f     = pat_ids[cell_keep]
            obs_f     = obs.loc[cell_keep].reset_index(drop=True)
            ct_groups_f = fs["ct_groups"]
            mask_f    = torch.tensor(prior["mask"][gene_idx, :], dtype=torch.float32)
            n_genes_f, n_ct_f = X_f.shape[1], len(ct_groups_f)
            print(f"    fold selection: {n_genes_f:,} genes, {n_ct_f} CTs, "
                  f"{X_f.shape[0]:,} cells")
        else:
            X_f, ct_f, pat_f, obs_f = X, ct_ids, pat_ids, obs
            ct_groups_f, mask_f = ct_groups, mask_t
            n_genes_f, n_ct_f = n_genes, n_ct

        # Cells belonging to held_out patient — EXCLUDED from AE training
        train_mask = pat_f != held_out
        X_train = X_f[train_mask]
        ct_train = ct_f[train_mask]
        n_train = int(train_mask.sum())
        n_held = int(len(X_f) - n_train)
        print(f"    AE train cells: {n_train:,}  held-out cells: {n_held:,}")

        # Seed: deterministic mode reuses `active_seed` so per-fold AE variance
        # comes ONLY from the missing-patient effect (no init/kernel noise).
        # Vanilla mode uses `active_seed + fold_i` so folds have independent inits.
        _pin_seeds(active_seed if deterministic else active_seed + fold_i)
        ae = PathwayAE(n_genes_f, n_pw, mask_f, AE_LATENT_DIM, AE_DROPOUT,
                       norm="ctbn", act="gelu", n_ct=n_ct_f,
                       no_latent=no_latent)
        try:
            ae_hist = train_ae(ae, X_train, n_epochs=ae_epochs,
                               ct_ids=ct_train, ct_aux_weight=ct_aux_weight,
                               decorr_weight=decorr_weight, mask_frac=mask_frac,
                               verbose=False)
        except Exception as e:
            print(f"    AE training FAILED: {e}")
            per_fold_data.append({
                "patient": held_out, "label": pat_labels[held_out],
                "oof_prob": None, "error": f"ae_train: {e}",
                "seconds": time.time() - t_fold,
            })
            del ae; _free_gpu_mem()
            continue

        # Diagnostic: last-epoch train + val reconstruction MSE
        last = ae_hist[-1] if ae_hist else {}
        train_mse = last.get("train_recon_mse", last.get("train_loss"))
        val_mse   = last.get("val_recon_mse",   last.get("val_loss"))

        # Save AE checkpoint
        ckpt_path = ae_ckpt_dir / f"fold_{held_out}.pt"
        torch.save(ae.state_dict(), ckpt_path)

        # Diagnostic: input-space reconstruction MSE, held-out vs training
        # (uses the trained AE directly, before encoding all cells)
        if diagnostics:
            recon = _held_recon_mse(ae, X_f, ct_f, pat_f, held_out)
        else:
            recon = {"train_recon_mse_clean": None, "held_recon_mse_clean": None,
                     "recon_ood_ratio": None}

        # Encode ALL cells with this AE. Suffix skips the cell_meta.csv write.
        with tempfile.TemporaryDirectory(prefix="ae_fold_") as tmp:
            h_full, _z_full = precompute_embeddings(
                ae, X_f, obs_f, Path(tmp), ct_ids=ct_f,
                verbose=False, suffix="_fold")

        # Diagnostic: h-space OOD stat overall + per-CT breakdown.
        # NB: cKDTree on ~176K points in 50-D degenerates to near-linear scan,
        # costing ~15 min/fold — 40% of total runtime. Skipped unless requested.
        if diagnostics:
            ood     = _hspace_ood_stats(h_full, pat_f, held_out)
            per_ct_ood = _per_ct_hspace_ood(h_full, pat_f, ct_f, ct_groups_f, held_out)
            diag = _per_ct_diagnostics(h_full, pat_f, ct_f, pat_labels,
                                        ct_groups_f, held_out)
        else:
            ood = {"held_mean_nn_dist": None, "train_mean_nn_dist": None,
                   "ood_ratio": None}
            per_ct_ood = {}
            diag = {"per_ct_names": list(ct_groups_f), "per_ct_aucs": [],
                    "held_phi_p_probs": [], "held_phi_p_logits": [],
                    "n_ct_gated_in": -1, "gated_cts": []}

        # Run Phase 2c LOOCV on h_full — held-out patient's OOF prob comes from
        # per-CT classifiers refit on training patients (which includes held_out's
        # cells in AE-encoded form, but with an AE that never saw held_out).
        try:
            held_idx = unique_pats.index(held_out)
            # only_patient_idx: the outer LOOCV loop is independent per patient,
            # so computing just this fold's held-out patient is identical to
            # computing all n and discarding n-1 (verified fold-for-fold).
            result = train_h_concat_gated_concat_en(
                h_full, pat_f, ct_f, pat_labels, ct_groups_f, verbose=False,
                only_patient_idx=held_idx)
            oof_p = result["oof_probs"][held_idx]
            oof_probs[held_idx] = oof_p
            avg_selected = result["cv_summary"].get("avg_selected_cts")
            elapsed = time.time() - t_fold
            ratio = ood.get("ood_ratio")
            ratio_s = f"{ratio:.2f}×" if ratio is not None else "—"
            recon_r = recon.get("recon_ood_ratio")
            recon_s = f"{recon_r:.2f}×" if recon_r is not None else "—"
            print(f"    → OOF prob = {oof_p:.4f}  h_OOD = {ratio_s}  "
                  f"recon_OOD = {recon_s}  gate={diag['n_ct_gated_in']}/{len(ct_groups_f)}  "
                  f"({elapsed:.1f}s)")
            per_fold_data.append({
                "patient": held_out, "label": pat_labels[held_out],
                "oof_prob": float(oof_p),
                # AE training diagnostics
                "train_recon_mse":       train_mse,
                "val_recon_mse":         val_mse,
                # AE input-space generalization diagnostics
                "train_recon_mse_clean": recon["train_recon_mse_clean"],
                "held_recon_mse_clean":  recon["held_recon_mse_clean"],
                "recon_ood_ratio":       recon["recon_ood_ratio"],
                # h-space OOD diagnostics
                "h_held_nn_dist":        ood["held_mean_nn_dist"],
                "h_train_nn_dist":       ood["train_mean_nn_dist"],
                "h_ood_ratio":           ratio,
                "per_ct_h_ood":          json.dumps(per_ct_ood),
                # Phase 2c per-CT diagnostics
                "per_ct_aucs":           json.dumps(diag["per_ct_aucs"]),
                "per_ct_names":          json.dumps(diag["per_ct_names"]),
                "gated_cts":             json.dumps(diag["gated_cts"]),
                "n_ct_gated_in":         diag["n_ct_gated_in"],
                "held_phi_p_probs":      json.dumps(diag["held_phi_p_probs"]),
                "held_phi_p_logits":     json.dumps(diag["held_phi_p_logits"]),
                "phase2c_avg_selected_cts": avg_selected,
                "seconds": elapsed,
            })
        except Exception as e:
            print(f"    Phase 2c FAILED: {e}")
            per_fold_data.append({
                "patient": held_out, "label": pat_labels[held_out],
                "oof_prob": None, "error": f"phase2c: {e}",
                "seconds": time.time() - t_fold,
            })

        # Incremental save so partial results survive a crash
        pd.DataFrame(per_fold_data).to_csv(per_fold_csv, index=False)
        del ae, h_full; _free_gpu_mem()

    # Aggregate
    valid = ~np.isnan(oof_probs)
    if valid.sum() < 3 or (y_full[valid] == 1).sum() < 2 or (y_full[valid] == 0).sum() < 2:
        auc = float("nan"); auprc = float("nan")
        print(f"\n  Not enough valid folds ({valid.sum()}) to compute AUC.")
    else:
        auc = float(roc_auc_score(y_full[valid], oof_probs[valid]))
        auprc = float(average_precision_score(y_full[valid], oof_probs[valid]))

    total_elapsed = time.time() - total_start
    summary = {
        "cohort": cohort,
        "mode": "ae_per_fold_deterministic" if deterministic else "ae_per_fold_vanilla",
        "ae_epochs": ae_epochs,
        "n_patients_total": n_pats,
        "n_patients_valid": int(valid.sum()),
        "mean_auc": auc,
        "mean_auprc": auprc,
        "prod_auc": baseline["prod_auc"],
        "prod_auprc": baseline["prod_auprc"],
        "delta_auc_vs_prod": (auc - baseline["prod_auc"])
                             if baseline["prod_auc"] is not None and not np.isnan(auc)
                             else None,
        "prevalence": float(y_full.mean()),
        "total_seconds": total_elapsed,
        "seconds_per_fold_mean": total_elapsed / max(n_pats, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(out_dir / "oof_probs.npy", oof_probs)
    pd.DataFrame(per_fold_data).to_csv(per_fold_csv, index=False)

    print(f"\n  === {cohort} SUMMARY ===")
    print(f"    per-fold AE  : AUC = {auc:.4f}   AUPRC = {auprc:.4f}")
    if baseline["prod_auc"] is not None:
        print(f"    production  : AUC = {baseline['prod_auc']:.4f}   "
              f"AUPRC = {baseline['prod_auprc']:.4f}")
        if not np.isnan(auc):
            print(f"    ΔAUC        : {auc - baseline['prod_auc']:+.4f}")
    print(f"    total time  : {total_elapsed/60:.1f} min "
          f"({total_elapsed/max(n_pats,1):.1f} s/fold)")
    print(f"    → {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=str, default=None,
                    help="Single cohort to run. Overrides --all.")
    ap.add_argument("--all", action="store_true",
                    help="Run all 4 real cohorts sequentially")
    ap.add_argument("--ae-epochs", type=int, default=AE_N_EPOCHS,
                    help=f"AE epochs per fold (default {AE_N_EPOCHS}, matches production)")
    ap.add_argument("--deterministic", action="store_true",
                    help="Fix J+Q: same seed across folds + deterministic torch. "
                         "Isolates missing-patient effect from init/kernel noise. "
                         "Writes to ae_per_fold_deterministic/ subdir.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override RANDOM_STATE for deterministic runs. "
                         "Writes to ae_per_fold_deterministic_seed<S>/ subdir "
                         "when set. Use to test whether the default seed was "
                         "unlucky vs the failure being structural.")
    ap.add_argument("--ae-ct-aux-weight", type=float, default=None,
                    help=f"Override AE auxiliary CT-classification weight α "
                         f"(production {AE_CT_AUX_WEIGHT}).")
    ap.add_argument("--ae-decorr-weight", type=float, default=None,
                    help=f"Override AE pathway-decorrelation weight β "
                         f"(production {AE_DECORR_WEIGHT}). Higher β reduces "
                         f"rotational slack in h and should improve cross-fold "
                         f"reproducibility of the representation.")
    ap.add_argument("--ae-mask-frac", type=float, default=None,
                    help=f"Override the denoising corruption rate "
                         f"(production {AE_MASK_FRAC}). Pass 0 for the "
                         f"no-denoising ablation.")
    ap.add_argument("--shuffle-mask", action="store_true",
                    help="Randomly permute the gene-to-pathway assignment "
                         "among pathway-active genes, preserving per-pathway "
                         "gene counts. Tests whether the biological prior "
                         "contributes beyond a sparse projection of the same "
                         "shape.")
    ap.add_argument("--no-latent", action="store_true",
                    help="Omit the intermediate latent representation: the "
                         "decoder reconstructs directly from the pathway "
                         "activations h. Tests whether the bottleneck adds "
                         "anything beyond the pathway layer, which is what "
                         "all downstream classifiers and attributions use.")
    ap.add_argument("--fold-selection", action="store_true",
                    help="Derive HVG gene selection AND cell-type grouping "
                         "from training-fold cells only. Both are data-"
                         "dependent and currently computed over all cells, "
                         "including the held-out patient. Writes to a "
                         "*_foldsel/ subdir.")
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="Skip h-space OOD / recon-MSE / per-CT diagnostics. "
                         "These cost ~15 min/fold (cKDTree in 50-D) and do "
                         "not affect the OOF predictions or AUC.")
    args = ap.parse_args()

    if args.cohort:
        cohorts = [args.cohort]
    elif args.all:
        cohorts = ["GSE189125_pre_ici",
                   "GSE216329_integrated_pre_ici",
                   "GSE249898_integrated_pre_ici",
                   "GSE285888_pre_ici"]
    else:
        ap.error("Specify --cohort <ID> or --all")

    summaries = []
    for c in cohorts:
        try:
            s = run_ae_per_fold(c, ae_epochs=args.ae_epochs,
                                deterministic=args.deterministic,
                                seed=args.seed,
                                ct_aux_weight=args.ae_ct_aux_weight,
                                decorr_weight=args.ae_decorr_weight,
                                diagnostics=not args.no_diagnostics,
                                fold_selection=args.fold_selection,
                                mask_frac=args.ae_mask_frac,
                                shuffle_mask=args.shuffle_mask,
                                no_latent=args.no_latent)
            summaries.append(s)
        except Exception as e:
            print(f"\n  {c} FAILED: {e}")
            import traceback; traceback.print_exc()
            summaries.append({"cohort": c, "error": str(e)})

    # Final side-by-side
    print(f"\n\n{'='*70}\n  ALL COHORTS — AE-PER-FOLD (VANILLA)\n{'='*70}")
    print(f"  {'Cohort':<35}  {'AUC':>7}  {'AUPRC':>7}  {'Valid':>7}  {'min':>6}")
    for s in summaries:
        if "error" in s:
            print(f"  {s['cohort']:<35}  ERROR: {s['error'][:60]}")
        else:
            print(f"  {s['cohort']:<35}  "
                  f"{s['mean_auc']:>7.4f}  {s['mean_auprc']:>7.4f}  "
                  f"{s['n_patients_valid']:>3}/{s['n_patients_total']:<3}  "
                  f"{s['total_seconds']/60:>6.1f}")


if __name__ == "__main__":
    main()
